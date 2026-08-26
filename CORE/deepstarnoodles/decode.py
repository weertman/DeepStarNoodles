# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Decode the 10-channel hydra representation back into polylines and points.

The decoder inverts the dense representation produced by
:func:`deepstarnoodles.dataset.rasterize_targets` (and approximated by the model):

- point landmarks (mouth / madreporite / butthole) from their heatmaps,
- per-arm polylines, ordered base -> tip, by tracing the centerline ridge.

Arm tracing is a shortest-path problem over the support mask. Edge costs are
built from three complementary signals so that crossings and adjacent arms
stay separable:

- ridge: pixels with high centerline probability are cheap,
- tangent: the (dx, dy) field points base -> tip, so walking tip -> base means
  moving against it; steps aligned WITH the tangent are penalized,
- progress: walking tip -> base must descend the progress field; ascending
  steps are penalized.

The same decoder runs on rasterized targets (perfect inputs, used to validate
the algorithm) and on activated model outputs (used in inference).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class DecodeParams:
    support_threshold: float = 0.5
    # Peak extraction.
    tip_threshold: float = 0.3
    base_threshold: float = 0.3
    point_threshold: float = 0.3
    tip_nms_radius: int = 4
    base_nms_radius: int = 3
    point_nms_radius: int = 8
    base_hit_radius: float = 3.0  # truncate a traced path at the first base peak it touches
    peak_snap_radius: float = 8.0  # max distance to snap a peak onto the support mask
    # Edge costs.
    ridge_weight: float = 2.0
    align_weight: float = 1.0
    progress_weight: float = 2.0
    # Progress must descend smoothly tip->base (~1/arm_length per px). A steep
    # drop means the path jumped onto another arm's inner half at a crossing.
    progress_drop_tolerance: float = 0.05  # free descent rate, per px
    progress_drop_weight: float = 20.0
    turn_weight: float = 1.0  # momentum: penalize direction changes (resolves crossings)
    cost_floor: float = 0.01
    # Path acceptance.
    max_mean_step_cost: float = 1.2  # reject paths whose mean per-px cost is worse
    min_path_px: float = 10.0
    # Output polyline shape.
    smooth_window: int = 7
    resample_spacing: float = 4.0


def activate_logits(pred: torch.Tensor) -> torch.Tensor:
    """Map raw model output (10, H, W) to the activated target representation."""
    out = torch.empty_like(pred, dtype=torch.float32)
    out[0] = torch.sigmoid(pred[0])
    out[1] = torch.sigmoid(pred[1])
    out[2] = torch.tanh(pred[2])
    out[3] = torch.tanh(pred[3])
    out[4] = torch.sigmoid(pred[4])
    out[5:] = torch.sigmoid(pred[5:])
    return out


def find_peaks(heat: np.ndarray, threshold: float, nms_radius: int) -> np.ndarray:
    """Local maxima above threshold with max-pool NMS. Returns (N, 3): x, y, score."""
    k = 2 * int(nms_radius) + 1
    pooled = maximum_filter(
        np.asarray(heat, dtype=np.float32),
        size=k,
        mode="constant",
        cval=-np.inf,
    )
    ys, xs = np.nonzero((heat >= pooled - 1e-6) & (heat >= threshold))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    scores = heat[ys, xs]
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


_OFFSETS = tuple((oy, ox) for oy in (-1, 0, 1) for ox in (-1, 0, 1) if (oy, ox) != (0, 0))


def _build_state_graph(
    node_id: np.ndarray,
    centerline: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    progress: np.ndarray,
    params: DecodeParams,
    source_pixels: list[int],
) -> tuple[csr_matrix, int]:
    """Directed graph over (pixel, incoming-direction) states for tip -> base walks.

    The direction state gives paths momentum: changing direction pays a turn
    penalty, which is what disambiguates arm crossings where the averaged
    tangent/progress fields are uninformative. Node layout: pixel p with
    incoming direction d is state p * 8 + d. One virtual source node per entry
    of `source_pixels` is appended (ids n_pix * 8 ..), connected to all 8
    direction states of its pixel at ~zero cost so the first step pays no turn
    penalty.

    Returns (graph, n_state_nodes).
    """
    h, w = node_id.shape
    n_pix = int(node_id.max()) + 1
    n_states = n_pix * 8

    unit = np.array([(ox / np.hypot(ox, oy), oy / np.hypot(ox, oy)) for oy, ox in _OFFSETS])
    turn_cost = params.turn_weight * 0.5 * (1.0 - unit @ unit.T)  # [d_in, d_out]

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for d_out, (oy, ox) in enumerate(_OFFSETS):
        step = float(np.hypot(ox, oy))
        sy0, sy1 = max(0, -oy), min(h, h - oy)
        sx0, sx1 = max(0, -ox), min(w, w - ox)
        src = node_id[sy0:sy1, sx0:sx1]
        dst = node_id[sy0 + oy:sy1 + oy, sx0 + ox:sx1 + ox]
        valid = (src >= 0) & (dst >= 0)
        if not valid.any():
            continue
        sy, sx = np.nonzero(valid)
        sy += sy0
        sx += sx0
        ty, tx = sy + oy, sx + ox

        ridge = 1.0 - 0.5 * (centerline[sy, sx] + centerline[ty, tx])
        # Tangent points base->tip; a tip->base walk should move against it.
        ux, uy = ox / step, oy / step
        align = ux * dx[sy, sx] + uy * dy[sy, sx]  # +1 = toward tip (bad)
        align_cost = 0.5 * (1.0 + align)
        d_prog = (progress[ty, tx] - progress[sy, sx]) / step
        prog_rise = np.maximum(d_prog, 0.0)
        prog_drop = np.maximum(-d_prog - params.progress_drop_tolerance, 0.0)
        base_cost = step * (
            params.cost_floor
            + params.ridge_weight * ridge
            + params.align_weight * align_cost
            + params.progress_weight * prog_rise
            + params.progress_drop_weight * prog_drop
        )
        src_pix = node_id[sy, sx]
        dst_state = node_id[ty, tx] * 8 + d_out
        for d_in in range(8):
            rows.append(src_pix * 8 + d_in)
            cols.append(dst_state)
            weights.append((base_cost + turn_cost[d_in, d_out]).astype(np.float32))

    src_rows = []
    src_cols = []
    for i, pix in enumerate(source_pixels):
        for d in range(8):
            src_rows.append(n_states + i)
            src_cols.append(pix * 8 + d)
    rows.append(np.asarray(src_rows, dtype=np.int64))
    cols.append(np.asarray(src_cols, dtype=np.int64))
    weights.append(np.full(len(src_rows), 1e-6, dtype=np.float32))

    n_total = n_states + len(source_pixels)
    graph = csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_total, n_total),
    )
    return graph, n_states


def _snap_to_mask(peaks: np.ndarray, node_id: np.ndarray, node_yx: np.ndarray, radius: float) -> list[int]:
    """Snap (x, y) peaks to nearest mask node within radius; -1 when too far."""
    out: list[int] = []
    for x, y, _ in peaks:
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < node_id.shape[0] and 0 <= ix < node_id.shape[1] and node_id[iy, ix] >= 0:
            out.append(int(node_id[iy, ix]))
            continue
        d2 = (node_yx[:, 0] - y) ** 2 + (node_yx[:, 1] - x) ** 2
        j = int(np.argmin(d2))
        out.append(j if d2[j] <= radius * radius else -1)
    return out


def _extract_path(predecessors: np.ndarray, target: int) -> list[int]:
    path = [target]
    while predecessors[path[-1]] >= 0:
        path.append(int(predecessors[path[-1]]))
    return path  # target ... source(tip)


def _smooth_resample(points: np.ndarray, window: int, spacing: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) >= window >= 3:
        kernel = np.ones(window, dtype=np.float32) / window
        pad = window // 2
        padded = np.concatenate([pts[:1].repeat(pad, axis=0), pts, pts[-1:].repeat(pad, axis=0)])
        pts = np.stack([np.convolve(padded[:, i], kernel, mode='valid') for i in range(2)], axis=1)
        pts[0], pts[-1] = points[0], points[-1]  # keep exact endpoints
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= spacing:
        return np.stack([pts[0], pts[-1]])
    distances = np.arange(0.0, total, spacing, dtype=np.float32)
    if float(distances[-1]) < total:
        distances = np.concatenate([distances, [total]])
    out = np.empty((len(distances), 2), dtype=np.float32)
    seg_i = 0
    for k, d in enumerate(distances):
        while seg_i < len(seg) - 1 and d > cum[seg_i + 1]:
            seg_i += 1
        t = (d - cum[seg_i]) / max(float(seg[seg_i]), 1e-6)
        out[k] = (1.0 - t) * pts[seg_i] + t * pts[seg_i + 1]
    return out


def decode_arms(act: np.ndarray, params: DecodeParams = DecodeParams()) -> list[dict]:
    """Trace per-arm polylines from activated channels (10, H, W).

    Returns a list of dicts with keys: polyline (N, 2 xy, base->tip), tip_score,
    base_score, mean_step_cost, length_px.
    """
    support, centerline = act[0], act[1]
    dx, dy, progress = act[2], act[3], act[4]
    mask = support > params.support_threshold
    if not mask.any():
        return []

    tips = find_peaks(act[9], params.tip_threshold, params.tip_nms_radius)
    bases = find_peaks(act[8], params.base_threshold, params.base_nms_radius)
    if len(tips) == 0:
        return []

    h, w = mask.shape
    node_id = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.nonzero(mask)
    node_id[ys, xs] = np.arange(len(ys))
    node_yx = np.stack([ys, xs], axis=1)

    tip_nodes = _snap_to_mask(tips, node_id, node_yx, params.peak_snap_radius)
    base_nodes = _snap_to_mask(bases, node_id, node_yx, params.peak_snap_radius) if len(bases) else []
    live_tips = [(i, n) for i, n in enumerate(tip_nodes) if n >= 0]
    if not live_tips:
        return []

    graph, n_states = _build_state_graph(
        node_id, centerline, dx, dy, progress, params,
        source_pixels=[n for _, n in live_tips],
    )
    sources = [n_states + r for r in range(len(live_tips))]
    dist, pred = dijkstra(graph, directed=True, indices=sources, return_predecessors=True)

    # Collapse direction states to per-pixel best cost/state.
    pix_dist = dist[:, :n_states].reshape(len(live_tips), -1, 8)
    pix_best_state = np.argmin(pix_dist, axis=2)
    pix_min_dist = np.take_along_axis(pix_dist, pix_best_state[:, :, None], axis=2)[:, :, 0]

    live_bases = [(j, n) for j, n in enumerate(base_nodes) if n >= 0]
    # Global 1:1 tip-base assignment (Hungarian). Independent per-tip choices
    # let one arm steal another's base via a cheap shortcut across the disk;
    # the global optimum is forced to spread arms over distinct bases.
    assign: dict[int, int] = {}
    if live_bases:
        big = 1e9
        cost_m = np.full((len(live_tips), len(live_bases)), big, dtype=np.float64)
        for r in range(len(live_tips)):
            for k, (j, bn) in enumerate(live_bases):
                c = float(pix_min_dist[r, bn])
                if np.isfinite(c):
                    cost_m[r, k] = c
        rr, kk = linear_sum_assignment(cost_m)
        for r, k in zip(rr, kk):
            if cost_m[r, k] < big:
                assign[r] = live_bases[k][0]

    arms: list[dict] = []
    for r, (ti, tn) in enumerate(live_tips):
        if r in assign:
            best_j = assign[r]
            target_pix = base_nodes[best_j]
            cost = float(pix_min_dist[r, target_pix])
            base_score = float(bases[best_j, 2])
        else:
            # Fallback: descend to the lowest-progress reachable pixel.
            reach = np.isfinite(pix_min_dist[r])
            if not reach.any():
                continue
            ridx = np.nonzero(reach)[0]
            prog_vals = progress[node_yx[ridx, 0], node_yx[ridx, 1]]
            target_pix = int(ridx[int(np.argmin(prog_vals))])
            cost = float(pix_min_dist[r, target_pix])
            base_score = 0.0
        target_state = target_pix * 8 + int(pix_best_state[r, target_pix])
        state_path = _extract_path(pred[r], target_state)  # base ... tip(, virtual)
        pix_path = [s // 8 for s in state_path if s < n_states]
        yx = node_yx[pix_path]
        # Truncate at the first touch of the ASSIGNED base (walking from the
        # tip): the tail beyond it is disk wandering, not arm.
        if r in assign and params.base_hit_radius > 0:
            bx, by = bases[assign[r], 0], bases[assign[r], 1]
            d2 = (yx[:, 1] - bx) ** 2 + (yx[:, 0] - by) ** 2
            hits = np.nonzero(d2 <= params.base_hit_radius ** 2)[0]
            if len(hits):
                yx = yx[int(hits[-1]):]  # path is base...tip; last hit = first from tip
        xy = yx[:, ::-1].astype(np.float32)  # base->tip in (x, y)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        length = float(seg.sum())
        if length < params.min_path_px:
            continue
        mean_cost = cost / max(length, 1e-6)
        if mean_cost > params.max_mean_step_cost:
            continue
        polyline = _smooth_resample(xy, params.smooth_window, params.resample_spacing)
        arms.append({
            'polyline': polyline,
            'tip_score': float(tips[ti, 2]),
            'base_score': base_score,
            'mean_step_cost': mean_cost,
            'length_px': length,
        })
    return arms


def decode_image(act: np.ndarray, params: DecodeParams = DecodeParams()) -> dict:
    """Full decode: arms + point landmarks from activated channels (10, H, W)."""
    return {
        'arms': decode_arms(act, params),
        'mouth': find_peaks(act[5], params.point_threshold, params.point_nms_radius),
        'madreporite': find_peaks(act[6], params.point_threshold, params.point_nms_radius),
        'butthole': find_peaks(act[7], params.point_threshold, params.point_nms_radius),
    }


def decode_params_with(**overrides) -> DecodeParams:
    return replace(DecodeParams(), **overrides)
