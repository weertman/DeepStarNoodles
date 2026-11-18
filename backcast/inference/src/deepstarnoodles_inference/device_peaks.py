# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Device-side candidates for the exact host peak and seed-embedding rules.

``decode.find_peaks`` and ``decode_embed._sample_reference_embedding`` define
the decoder contract. Here only their data-parallel parts run on the field's
device: the -inf-padded maximum filter (an exact maximum), the float32
threshold tests and the row-major candidate list, plus 5x5 embedding/support
windows around seeds. Score ordering, greedy suppression and window means still
run in NumPy on identical float32 values, so ties and rounding are unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SEED_RADIUS = 2


# Device pruning is deliberately looser than the host rule, which then decides.
_PRUNE_MARGIN = 1e-4


def peak_candidates(heat: torch.Tensor, threshold: float, nms_radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return host ``(ys, xs, scores)`` exactly as ``find_peaks`` selects them.

    The device computes the -inf-padded maximum filter (an exact maximum) and
    keeps a superset of candidates in row-major order. NumPy then applies the
    original float32 tests to their heat and pooled values, so candidate order
    and membership match ``np.nonzero`` on the full host arrays.
    """
    if heat.ndim != 2 or heat.dtype != torch.float32:
        raise ValueError("heat must be a float32 (H,W) tensor")
    radius = int(nms_radius)
    # max_pool2d's implicit padding never wins a maximum: scipy cval=-inf.
    pooled = F.max_pool2d(heat[None, None], kernel_size=2 * radius + 1, stride=1, padding=radius)[0, 0]
    loose = (heat >= pooled - _PRUNE_MARGIN) & (heat >= threshold - _PRUNE_MARGIN)
    coordinates = torch.nonzero(loose)
    values = torch.stack((heat[coordinates[:, 0], coordinates[:, 1]],
                          pooled[coordinates[:, 0], coordinates[:, 1]]))
    coordinates, values = coordinates.cpu().numpy(), values.cpu().numpy()
    heat_values, pooled_values = values[0], values[1]
    exact = (heat_values >= pooled_values - 1e-6) & (heat_values >= threshold)
    return coordinates[exact, 0], coordinates[exact, 1], heat_values[exact]


def suppress_candidates(ys: np.ndarray, xs: np.ndarray, scores: np.ndarray, nms_radius: int) -> np.ndarray:
    """Exact host tail of ``decode.find_peaks`` after candidate extraction."""
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    order = np.argsort(-scores)
    keep: list[int] = []
    taken = np.zeros(len(xs), dtype=bool)
    for i in order:
        if taken[i]:
            continue
        keep.append(i)
        d2 = (xs - xs[i]) ** 2 + (ys - ys[i]) ** 2
        taken |= d2 <= nms_radius * nms_radius
    keep_a = np.asarray(keep, dtype=int)
    return np.stack([xs[keep_a], ys[keep_a], scores[keep_a]], axis=1).astype(np.float32)


def find_peaks_device(heat: torch.Tensor, threshold: float, nms_radius: int) -> np.ndarray:
    """``decode.find_peaks`` for a float32 device heatmap; returns (N,3) x, y, score."""
    ys, xs, scores = peak_candidates(heat, threshold, nms_radius)
    return suppress_candidates(ys, xs, scores, nms_radius)


def seed_windows(embedding: torch.Tensor, support: torch.Tensor, tips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gather (N,D,5,5) embedding and (N,5,5) support windows centred on tips.

    Out-of-image cells are filled but never read: the host rule clips windows to
    the image exactly as ``_sample_reference_embedding`` does.
    """
    if embedding.ndim != 3 or support.ndim != 2 or support.shape != embedding.shape[-2:]:
        raise ValueError("embedding must be (D,H,W) and support (H,W)")
    height, width = support.shape
    centers = np.rint(np.asarray(tips, dtype=np.float64)[:, :2]).astype(np.int64)
    # Python's round() and np.rint both round half to even.
    offsets = np.arange(-SEED_RADIUS, SEED_RADIUS + 1, dtype=np.int64)
    rows = np.clip(centers[:, 1, None] + offsets, 0, height - 1)
    cols = np.clip(centers[:, 0, None] + offsets, 0, width - 1)
    device = embedding.device
    rows_t = torch.as_tensor(rows, device=device)[:, :, None]
    cols_t = torch.as_tensor(cols, device=device)[:, None, :]
    emb = embedding[:, rows_t, cols_t].permute(1, 0, 2, 3)
    sup = support[rows_t, cols_t]
    return emb.cpu().numpy(), sup.cpu().numpy()


def seed_embedding_from_window(emb_window: np.ndarray, support_window: np.ndarray, x: float, y: float,
                               height: int, width: int) -> np.ndarray:
    """``_sample_reference_embedding(emb, x, y, 2, mask)`` from a centred window.

    Valid for in-image centres, which peak seeds always are.
    """
    radius = SEED_RADIUS
    iy, ix = int(round(y)), int(round(x))
    if not (0 <= iy < height and 0 <= ix < width):
        raise ValueError("seed window centres must lie inside the image")
    y0, y1 = max(0, iy - radius), min(height, iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(width, ix + radius + 1)
    if y0 < y1 and x0 < x1:
        local_y0, local_x0 = y0 - (iy - radius), x0 - (ix - radius)
        local_y1, local_x1 = local_y0 + (y1 - y0), local_x0 + (x1 - x0)
        sub_mask = support_window[local_y0:local_y1, local_x0:local_x1]
        if bool(sub_mask.any()):
            patch = emb_window[:, local_y0:local_y1, local_x0:local_x1][:, sub_mask]
            return patch.mean(axis=1).astype(np.float32)
    cy = min(max(iy, 0), height - 1)
    cx = min(max(ix, 0), width - 1)
    return emb_window[:, cy - (iy - radius), cx - (ix - radius)].astype(np.float32)


def device_seed_inputs(raw: torch.Tensor, params, image_index: int):
    """Tips, seed embeddings and base peaks for one image of raw device fields.

    Mirrors the host fields used by the fixed-slot decoder: raw support,
    sigmoid support/base/tip, and the raw identity embedding channels.
    """
    from .noodle_tracer import select_trace_seeds

    fields = raw[image_index]
    height, width = int(fields.shape[-2]), int(fields.shape[-1])
    raw_support = fields[0]
    activated_support = torch.sigmoid(fields[0])
    activated_base = torch.sigmoid(fields[8])
    activated_tip = torch.sigmoid(fields[9])
    gated = bool(params.support_gated_peaks)
    peak_support = raw_support > 0 if gated else None
    tip_heat = torch.where(peak_support, activated_tip, 0.0) if gated else activated_tip
    tips = select_trace_seeds(
        find_peaks_device(tip_heat, params.tip_threshold, params.tip_nms_radius),
        max_tips=params.max_tips,
        min_separation=params.tip_seed_separation,
    )
    seeds = None
    if len(tips):
        embedding = fields[10:10 + params.embedding_dim]
        support_values = raw_support if gated else activated_support
        emb_windows, support_windows = seed_windows(embedding, support_values, tips)
        support_mask = support_windows > 0 if gated else support_windows > 0.2
        seeds = np.stack([
            seed_embedding_from_window(emb_windows[k], support_mask[k], float(tip[0]), float(tip[1]),
                                       height, width)
            for k, tip in enumerate(tips)
        ])
    base_heat = torch.where(peak_support, activated_base, 0.0) if gated else activated_base
    base_peaks = find_peaks_device(base_heat, params.base_threshold, 3)
    return tips, seeds, base_peaks
