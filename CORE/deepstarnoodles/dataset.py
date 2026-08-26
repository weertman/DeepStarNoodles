# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""DeepStarNoodles dense dataset, trimmed to the canonical 10-channel hydra contract.

Channel contract (fixed):
0 arm_support     broad arm support mask (capsule, support_arm_width)
1 arm_centerline  compact polynomial centerline peak, clipped to support
2 arm_dx          unit tangent x, disk-to-tip
3 arm_dy          unit tangent y, disk-to-tip
4 arm_progress    normalized order along each arm, disk=0, tip=1
5 mouth           point heatmap
6 madreporite     point heatmap
7 butthole        point heatmap
8 arm_base        per-arm base endpoint heatmap
9 arm_tip         per-arm tip endpoint heatmap

Rasterization is always smooth-capsule (Chaikin-smoothed, resampled polylines
drawn as overlapping disks) and the centerline is always the compact polynomial
kernel max(0, 1 - (d/r)^2)^2 clipped to the support mask.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

from deepstarnoodles.paths import resolve_repo_path, to_repo_relative

CHANNEL_NAMES = (
    'arm_support',
    'arm_centerline',
    'arm_dx',
    'arm_dy',
    'arm_progress',
    'mouth',
    'madreporite',
    'butthole',
    'arm_base',
    'arm_tip',
)
CHANNEL_NAMES_WITHOUT_CENTERLINE = tuple(
    name for name in CHANNEL_NAMES if name != 'arm_centerline'
)
POINT_LABEL_TO_CHANNEL = {'mouth': 5, 'madreporite': 6, 'butthole': 7}
DEFAULT_MAX_BRIDGE_PAIRS = 0


def _shape_points(shape: dict[str, Any]) -> np.ndarray:
    raw = shape.get('points') or []
    return np.array([(float(raw[i]), float(raw[i + 1])) for i in range(0, len(raw) - 1, 2)], dtype=np.float32)


def _draw_gaussian(out: np.ndarray, x: float, y: float, sigma: float) -> None:
    h, w = out.shape
    radius = max(1, int(math.ceil(3 * sigma)))
    x0 = max(0, int(math.floor(x)) - radius)
    x1 = min(w, int(math.floor(x)) + radius + 1)
    y0 = max(0, int(math.floor(y)) - radius)
    y1 = min(h, int(math.floor(y)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], patch)


def _chaikin_smooth(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3 or iterations <= 0:
        return pts
    for _ in range(iterations):
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            out.append(0.75 * a + 0.25 * b)
            out.append(0.25 * a + 0.75 * b)
        out.append(pts[-1])
        pts = np.asarray(out, dtype=np.float32)
    return pts


def _resample_polyline(points: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return pts
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cum[-1])
    if total <= 1e-6:
        return pts[:1]
    distances = np.arange(0.0, total, max(float(spacing), 0.25), dtype=np.float32)
    if len(distances) == 0 or float(distances[-1]) < total:
        distances = np.concatenate([distances, np.asarray([total], dtype=np.float32)])
    out = []
    seg_i = 0
    for d in distances:
        while seg_i < len(seg_lengths) - 1 and d > cum[seg_i + 1]:
            seg_i += 1
        denom = max(float(seg_lengths[seg_i]), 1e-6)
        t = float((d - cum[seg_i]) / denom)
        out.append((1.0 - t) * pts[seg_i] + t * pts[seg_i + 1])
    return np.asarray(out, dtype=np.float32)


def _prepared_arm_polyline(shape: dict[str, Any]) -> np.ndarray:
    points = _shape_points(shape)
    if shape.get('_hydra_prepared_arm_polyline'):
        return points
    return _resample_polyline(_chaikin_smooth(points, iterations=2), spacing=1.0)


def trim_overlapping_arm_bases(
    shapes: Sequence[dict[str, Any]],
    *,
    arm_base_sigma: float,
    kernel_radius_sigmas: float = 1.5,
    max_trim_fraction: float = 0.5,
    min_remaining_length: float = 8.0,
) -> list[dict[str, Any]]:
    """Shorten smooth polylines to one projected arm-base ring.

    The original bases define a global elliptical ring metric. If any kernels
    overlap, every arm advances to the smallest common expanded ring that
    resolves them within the configured trim limits. Source shapes are copied.
    """
    copied = [dict(shape) for shape in shapes]
    arm_shape_indices: list[int] = []
    arms: list[np.ndarray] = []
    cumulative: list[np.ndarray] = []
    max_indices: list[int] = []
    for shape_index, shape in enumerate(copied):
        if (
            shape.get('outside')
            or shape.get('label') != 'arm'
            or shape.get('type') != 'polyline'
            or len(_shape_points(shape)) < 2
        ):
            continue
        points = _prepared_arm_polyline(shape)
        if len(points) < 2:
            continue
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(lengths))).astype(np.float32)
        total = float(arc[-1])
        trim_limit = min(
            max(0.0, total * max(0.0, float(max_trim_fraction))),
            max(0.0, total - max(0.0, float(min_remaining_length))),
        )
        arm_shape_indices.append(shape_index)
        arms.append(points)
        cumulative.append(arc)
        max_indices.append(int(np.searchsorted(arc, trim_limit, side='right') - 1))

    if len(arms) < 2:
        return copied

    kernel_radius = (
        max(0.0, float(kernel_radius_sigmas))
        * max(0.0, float(arm_base_sigma))
    )
    separation = 2.0 * kernel_radius
    base_points = np.stack([arm[0] for arm in arms])
    initial_distances = np.linalg.norm(
        base_points[:, None, :] - base_points[None, :, :], axis=2
    )
    upper = np.triu_indices(len(arms), k=1)
    if not bool((initial_distances[upper] <= separation).any()):
        trim_indices = [0] * len(arms)
    else:
        # Erode the complete base ring as one wavefront: every arm loses the
        # same proximal arc length. This preserves the annotated cyclic boundary
        # instead of creating independent, staggered pairwise solutions.
        maximum_common_trim = min(
            float(arc[max_index])
            for arc, max_index in zip(cumulative, max_indices)
        )
        candidates = np.arange(
            0.0, maximum_common_trim + 0.5, 0.5, dtype=np.float32
        )
        best_indices = [0] * len(arms)
        best_score: tuple[float, ...] | None = None
        for erosion_distance in candidates:
            proposed = [
                min(
                    max_index,
                    int(np.searchsorted(arc, erosion_distance, side='left')),
                )
                for arc, max_index in zip(cumulative, max_indices)
            ]
            selected = np.stack([arm[index] for arm, index in zip(arms, proposed)])
            pair_distances = np.linalg.norm(
                selected[:, None, :] - selected[None, :, :], axis=2
            )[upper]
            overlap_depth = np.maximum(0.0, separation + 1e-3 - pair_distances)
            overlap_count = float((pair_distances <= separation).sum())
            score = (
                overlap_count,
                float(overlap_depth.sum()),
                float(erosion_distance),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_indices = proposed
            if overlap_count == 0.0:
                break
        trim_indices = best_indices

    for arm_index, shape_index in enumerate(arm_shape_indices):
        points = arms[arm_index][trim_indices[arm_index]:]
        copied[shape_index]['points'] = points.reshape(-1).astype(float).tolist()
        copied[shape_index]['_hydra_prepared_arm_polyline'] = True
    return copied


def _segment_intersection(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> tuple[float, float] | None:
    """Return segment parameters at a proper geometric intersection."""
    r, s = b - a, d - c
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) < 1e-7:
        return None
    q = c - a
    t = float((q[0] * s[1] - q[1] * s[0]) / denom)
    u = float((q[0] * r[1] - q[1] * r[0]) / denom)
    return (t, u) if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else None


def derive_crossing_bridge_pairs(
    shapes: Sequence[dict[str, Any]],
    *,
    max_pairs: int = DEFAULT_MAX_BRIDGE_PAIRS,
    bridge_offset: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-size same-arm endpoint pairs around polyline crossings.

    Pairs are derived from vector annotations before support overlap is
    rasterized. Each geometric crossing contributes one bridge for each
    participating arm; the crossing itself is never assigned an owner.
    """
    if max_pairs <= 0:
        return (
            torch.zeros((0, 2, 2), dtype=torch.float32),
            torch.zeros(0, dtype=torch.bool),
            torch.tensor(0, dtype=torch.long),
        )
    arms = [
        _prepared_arm_polyline(shape)
        for shape in shapes
        if not shape.get('outside')
        and shape.get('label') == 'arm'
        and shape.get('type') == 'polyline'
        and len(_shape_points(shape)) >= 2
    ]
    pairs: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    offset = max(1, int(round(bridge_offset)))
    for arm_i, first in enumerate(arms):
        for arm_j in range(arm_i + 1, len(arms)):
            second = arms[arm_j]
            for i, (a, b) in enumerate(zip(first[:-1], first[1:])):
                for j, (c, d) in enumerate(zip(second[:-1], second[1:])):
                    hit = _segment_intersection(a, b, c, d)
                    if hit is None:
                        continue
                    # Dense resampling can produce adjacent duplicate segment
                    # hits at a shared pixel; quantize the crossing location.
                    point = a + hit[0] * (b - a)
                    key = (arm_i * len(arms) + arm_j, int(round(point[0])), int(round(point[1])))
                    if key in seen:
                        continue
                    seen.add(key)
                    for polyline, index in ((first, i), (second, j)):
                        lo, hi = max(0, index - offset), min(len(polyline) - 1, index + 1 + offset)
                        if lo < hi:
                            pairs.append(np.stack((polyline[lo], polyline[hi])).astype(np.float32))
    padded = torch.zeros((max_pairs, 2, 2), dtype=torch.float32)
    valid = torch.zeros(max_pairs, dtype=torch.bool)
    count = min(len(pairs), max_pairs)
    if count:
        padded[:count] = torch.from_numpy(np.stack(pairs[:count]))
        valid[:count] = True
    return padded, valid, torch.tensor(count, dtype=torch.long)


def count_valid_arms(shapes: Sequence[dict[str, Any]]) -> int:
    return sum(
        not shape.get('outside')
        and shape.get('label') == 'arm'
        and shape.get('type') == 'polyline'
        and len(_shape_points(shape)) >= 2
        for shape in shapes
    )


def _capsule_mask(size: tuple[int, int], a: np.ndarray, b: np.ndarray, width: float) -> np.ndarray:
    img = Image.new('L', size, 0)
    draw = ImageDraw.Draw(img)
    radius = max(0.5, float(width) / 2.0)
    seg_len = float(np.linalg.norm(b - a))
    steps = max(1, int(math.ceil(seg_len / max(0.5, radius * 0.5))))
    for j in range(steps + 1):
        t = j / steps
        p = (1.0 - t) * a + t * b
        draw.ellipse((float(p[0] - radius), float(p[1] - radius), float(p[0] + radius), float(p[1] + radius)), fill=255)
    return np.array(img, dtype=np.uint8) > 0


def _capsule_local(
    a: np.ndarray, b: np.ndarray, width: float, img_w: int, img_h: int
) -> tuple[np.ndarray, int, int] | None:
    """Rasterize one capsule segment into its local bounding box.

    Returns ``(mask_local, x0, y0)`` where ``mask_local`` is the boolean capsule
    over the in-bounds bbox and ``(x0, y0)`` is its top-left offset in the full
    image; ``None`` if the segment lies entirely outside the image. Drawing on
    the offset-cropped canvas yields pixels identical to ``_capsule_mask`` (which
    allocates the whole frame), but costs only the segment's neighborhood.
    """
    radius = max(0.5, float(width) / 2.0)
    minx = min(float(a[0]), float(b[0])) - radius
    maxx = max(float(a[0]), float(b[0])) + radius
    miny = min(float(a[1]), float(b[1])) - radius
    maxy = max(float(a[1]), float(b[1])) + radius
    x0 = max(0, int(math.floor(minx)))
    y0 = max(0, int(math.floor(miny)))
    x1 = min(img_w, int(math.ceil(maxx)) + 1)
    y1 = min(img_h, int(math.ceil(maxy)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    img = Image.new('L', (x1 - x0, y1 - y0), 0)
    draw = ImageDraw.Draw(img)
    seg_len = float(np.linalg.norm(b - a))
    steps = max(1, int(math.ceil(seg_len / max(0.5, radius * 0.5))))
    for j in range(steps + 1):
        t = j / steps
        p = (1.0 - t) * a + t * b
        draw.ellipse(
            (float(p[0] - radius - x0), float(p[1] - radius - y0),
             float(p[0] + radius - x0), float(p[1] + radius - y0)),
            fill=255,
        )
    return np.array(img, dtype=np.uint8) > 0, x0, y0


def rasterize_targets(
    size: tuple[int, int],
    shapes: Sequence[dict[str, Any]],
    *,
    support_arm_width: float = 8.0,
    centerline_radius: float = 1.5,
    arm_base_sigma: float = 1.5,
    arm_tip_sigma: float = 2.0,
    point_sigma: float = 4.0,
    return_arm_instance: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Rasterize CVAT shapes into the fixed 10-channel dense target tensor.

    When ``return_arm_instance`` is True, also return a per-pixel arm-instance
    label map (int16, shape (H, W)) for the identity-embedding loss:
    ``0`` = background, ``1..n`` = the n-th annotated arm (in shape order, an
    arbitrary per-image index with no canonical meaning), ``-1`` = pixels
    covered by more than one arm (crossings) which are ignored by the loss.
    """
    width, height = size
    target = np.zeros((len(CHANNEL_NAMES), height, width), dtype=np.float32)
    dx_sum = np.zeros((height, width), dtype=np.float32)
    dy_sum = np.zeros((height, width), dtype=np.float32)
    n_vec = np.zeros((height, width), dtype=np.float32)
    progress_sum = np.zeros((height, width), dtype=np.float32)
    n_prog = np.zeros((height, width), dtype=np.float32)
    centerline_skel = np.zeros((height, width), dtype=bool)
    arm_instance = np.zeros((height, width), dtype=np.int16)
    coverage = np.zeros((height, width), dtype=np.uint8)
    arm_idx = 0

    for shape in shapes:
        if shape.get('outside'):
            continue
        label = str(shape.get('label', ''))
        shape_type = str(shape.get('type', ''))
        pts = _shape_points(shape)
        if label == 'arm' and shape_type == 'polyline' and len(pts) >= 2:
            pts = _prepared_arm_polyline(shape)
            seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total = max(float(cum[-1]), 1e-6)
            arm_idx += 1
            arm_mask = np.zeros((height, width), dtype=bool)
            for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                seg_len = max(float(seg_lengths[i]), 1e-6)
                tangent = (b - a) / seg_len
                if i == 0:
                    _draw_gaussian(target[8], float(a[0]), float(a[1]), float(arm_base_sigma))
                if i == len(seg_lengths) - 1:
                    _draw_gaussian(target[9], float(b[0]), float(b[1]), float(arm_tip_sigma))
                cap = _capsule_local(a, b, support_arm_width, width, height)
                if cap is not None:
                    m, x0, y0 = cap
                    ys_l, xs_l = np.nonzero(m)
                    yy = ys_l + y0
                    xx = xs_l + x0
                    arm_mask[yy, xx] = True
                    target[0, yy, xx] = 1.0
                    dx_sum[yy, xx] += float(tangent[0])
                    dy_sum[yy, xx] += float(tangent[1])
                    n_vec[yy, xx] += 1.0
                    rel_x = xx.astype(np.float32) - float(a[0])
                    rel_y = yy.astype(np.float32) - float(a[1])
                    along = np.clip(rel_x * float(tangent[0]) + rel_y * float(tangent[1]), 0.0, seg_len)
                    progress_sum[yy, xx] += (float(cum[i]) + along) / total
                    n_prog[yy, xx] += 1.0
                center_cap = _capsule_local(a, b, 1.0, width, height)
                if center_cap is not None:
                    cm, cx0, cy0 = center_cap
                    cys_l, cxs_l = np.nonzero(cm)
                    centerline_skel[cys_l + cy0, cxs_l + cx0] = True
            # Assign this arm's instance id; pixels shared with an earlier arm
            # (crossings) are marked -1 so the embedding loss ignores them.
            overlap = arm_mask & (coverage > 0)
            arm_instance[arm_mask & (coverage == 0)] = arm_idx
            arm_instance[overlap] = -1
            coverage[arm_mask] += 1
        elif label in POINT_LABEL_TO_CHANNEL and shape_type == 'points':
            for x, y in pts:
                _draw_gaussian(target[POINT_LABEL_TO_CHANNEL[label]], float(x), float(y), point_sigma)

    vec_mask = n_vec > 0
    target[2, vec_mask] = dx_sum[vec_mask] / n_vec[vec_mask]
    target[3, vec_mask] = dy_sum[vec_mask] / n_vec[vec_mask]
    mag = np.sqrt(target[2] * target[2] + target[3] * target[3])
    nonzero = mag > 1e-6
    target[2, nonzero] /= mag[nonzero]
    target[3, nonzero] /= mag[nonzero]
    prog_mask = n_prog > 0
    target[4, prog_mask] = progress_sum[prog_mask] / n_prog[prog_mask]

    if centerline_skel.any():
        dist = distance_transform_edt(~centerline_skel).astype(np.float32)
        base = np.clip(1.0 - (dist / max(float(centerline_radius), 1e-6)) ** 2, 0.0, 1.0)
        target[1] = (base * base).astype(np.float32)
        target[1, target[0] <= 0.0] = 0.0

    target_tensor = torch.from_numpy(target)
    if return_arm_instance:
        return target_tensor, torch.from_numpy(arm_instance.astype(np.int64))
    return target_tensor


@dataclass(frozen=True)
class HydraDatasetConfig:
    image_size: tuple[int, int] | None = None  # (width, height)
    support_arm_width: float = 8.0
    centerline_radius: float = 1.5
    arm_base_sigma: float = 1.5
    arm_tip_sigma: float = 2.0
    point_sigma: float = 4.0
    augment: bool = False
    hflip_p: float = 0.0
    vflip_p: float = 0.0
    rotate_degrees: float = 0.0
    brightness_jitter: float = 0.0
    contrast_jitter: float = 0.0
    saturation_jitter: float = 0.0
    hue_jitter: float = 0.0
    grayscale_p: float = 0.0
    blur_p: float = 0.0
    blur_radius: float = 0.0
    noise_std: float = 0.0
    gamma_jitter: float = 0.0
    # USB-degradation augment: simulate low-quality lab camera crops on
    # natively high-res (field) images only. Images whose native min side is
    # below degrade_min_native are already at that quality and are never
    # degraded further.
    degrade_p: float = 0.0
    degrade_min_native: int = 384
    # Directory of per-image 'star_ignore' polygon JSONs (unannotated neighbor
    # stars found by StarSeg behind the containment gate); pixels under these
    # polygons are excluded from the loss.
    ignore_shapes_dir: Path | None = None
    target_cache_dir: Path | None = None
    # On-disk cache of source images already resized to `image_size`. Avoids
    # re-decoding the full-resolution originals every epoch. Safe for augmented
    # runs because resize-to-target happens before augmentation either way.
    image_cache_dir: Path | None = None
    max_bridge_pairs: int = DEFAULT_MAX_BRIDGE_PAIRS
    bridge_offset: float = 6.0
    trim_overlapping_arm_bases: bool = False
    base_trim_kernel_radius_sigmas: float = 1.5
    base_trim_max_fraction: float = 0.5
    base_trim_min_remaining_length: float = 8.0
    include_centerline: bool = True


def moderate_augment_kwargs() -> dict[str, float | bool]:
    """The 'moderate_no_shift' profile that worked well in the head-family sweeps."""
    return {
        'augment': True,
        'hflip_p': 0.5,
        'vflip_p': 0.15,
        'rotate_degrees': 180.0,
        'brightness_jitter': 0.18,
        'contrast_jitter': 0.18,
        'saturation_jitter': 0.20,
        'hue_jitter': 0.03,
        'grayscale_p': 0.08,
        'blur_p': 0.15,
        'blur_radius': 1.0,
        'noise_std': 0.015,
        'gamma_jitter': 0.15,
    }


def row_artifact_name(row: dict[str, str], idx: int) -> str:
    """Stable per-image filename shared by canonical and split indexes."""
    shapes_path = row.get('shapes_path', '')
    if shapes_path:
        stem = Path(shapes_path).stem
        if stem:
            return f'{stem}.json'
    return f'image_{idx:06d}.json'


class HydraDenseDataset(Dataset):
    """Image + fixed 10-channel dense target dataset over a CVAT-derived index CSV."""

    def __init__(
        self,
        index_csv: str | Path,
        *,
        image_size: tuple[int, int] | None = None,
        support_arm_width: float = 8.0,
        centerline_radius: float = 1.5,
        arm_base_sigma: float = 1.5,
        arm_tip_sigma: float = 2.0,
        point_sigma: float = 4.0,
        augment: bool = False,
        hflip_p: float = 0.0,
        vflip_p: float = 0.0,
        rotate_degrees: float = 0.0,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        saturation_jitter: float = 0.0,
        hue_jitter: float = 0.0,
        grayscale_p: float = 0.0,
        blur_p: float = 0.0,
        blur_radius: float = 0.0,
        noise_std: float = 0.0,
        gamma_jitter: float = 0.0,
        degrade_p: float = 0.0,
        degrade_min_native: int = 384,
        ignore_shapes_dir: str | Path | None = None,
        target_cache_dir: str | Path | None = None,
        image_cache_dir: str | Path | None = None,
        max_bridge_pairs: int = DEFAULT_MAX_BRIDGE_PAIRS,
        bridge_offset: float = 6.0,
        trim_overlapping_arm_bases: bool = False,
        base_trim_kernel_radius_sigmas: float = 1.5,
        base_trim_max_fraction: float = 0.5,
        base_trim_min_remaining_length: float = 8.0,
        include_centerline: bool = True,
    ) -> None:
        self.index_csv = Path(index_csv)
        self.config = HydraDatasetConfig(
            image_size=image_size,
            support_arm_width=support_arm_width,
            centerline_radius=centerline_radius,
            arm_base_sigma=arm_base_sigma,
            arm_tip_sigma=arm_tip_sigma,
            point_sigma=point_sigma,
            augment=augment,
            hflip_p=hflip_p,
            vflip_p=vflip_p,
            rotate_degrees=rotate_degrees,
            brightness_jitter=brightness_jitter,
            contrast_jitter=contrast_jitter,
            saturation_jitter=saturation_jitter,
            hue_jitter=hue_jitter,
            grayscale_p=grayscale_p,
            blur_p=blur_p,
            blur_radius=blur_radius,
            noise_std=noise_std,
            gamma_jitter=gamma_jitter,
            degrade_p=degrade_p,
            degrade_min_native=degrade_min_native,
            ignore_shapes_dir=Path(ignore_shapes_dir) if ignore_shapes_dir is not None else None,
            target_cache_dir=Path(target_cache_dir) if target_cache_dir is not None else None,
            image_cache_dir=Path(image_cache_dir) if image_cache_dir is not None else None,
            max_bridge_pairs=max_bridge_pairs,
            bridge_offset=bridge_offset,
            trim_overlapping_arm_bases=trim_overlapping_arm_bases,
            base_trim_kernel_radius_sigmas=base_trim_kernel_radius_sigmas,
            base_trim_max_fraction=base_trim_max_fraction,
            base_trim_min_remaining_length=base_trim_min_remaining_length,
            include_centerline=include_centerline,
        )
        with self.index_csv.open(newline='', encoding='utf-8') as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f'No rows in dataset index: {self.index_csv}')

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = resolve_repo_path(row['image_path'])
        shapes = json.loads(resolve_repo_path(row['shapes_path']).read_text(encoding='utf-8'))
        if self.config.ignore_shapes_dir is not None:
            ignore_path = self.config.ignore_shapes_dir / row_artifact_name(row, idx)
            if ignore_path.exists():
                # Ignore polygons join the shape list so geometric augmentation
                # transforms them alongside the annotations; they are split back
                # out before target rasterization (and never enter the cache key).
                shapes = shapes + json.loads(ignore_path.read_text(encoding='utf-8'))

        target_size = self.config.image_size
        cached = self._load_cached_resized_image(idx, image_path, target_size)
        if cached is not None:
            # Cached image is already at target_size; scale shapes from original.
            image, original_size = cached
            if target_size is not None and tuple(original_size) != tuple(target_size):
                shapes = _scale_shapes(shapes, target_size[0] / original_size[0], target_size[1] / original_size[1])
        else:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            if target_size is not None and image.size != target_size:
                sx = target_size[0] / image.width
                sy = target_size[1] / image.height
                image = image.resize(target_size, Image.Resampling.BILINEAR)
                shapes = _scale_shapes(shapes, sx, sy)
            self._save_cached_resized_image(idx, image_path, target_size, image, original_size)

        if self.config.trim_overlapping_arm_bases:
            shapes = trim_overlapping_arm_bases(
                shapes,
                arm_base_sigma=self.config.arm_base_sigma,
                kernel_radius_sigmas=self.config.base_trim_kernel_radius_sigmas,
                max_trim_fraction=self.config.base_trim_max_fraction,
                min_remaining_length=self.config.base_trim_min_remaining_length,
            )

        if self.config.augment:
            image, shapes = _augment_image_and_shapes(image, shapes, self.config)
            if (
                self.config.degrade_p > 0
                and min(original_size) >= self.config.degrade_min_native
                and random.random() < self.config.degrade_p
            ):
                image = _degrade_to_usb_quality(image)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        ignore_polys = [s for s in shapes if s.get('label') == 'star_ignore']
        shapes = [s for s in shapes if s.get('label') != 'star_ignore']
        bridge_pairs, bridge_valid, bridge_count = derive_crossing_bridge_pairs(
            shapes,
            max_pairs=self.config.max_bridge_pairs,
            bridge_offset=self.config.bridge_offset,
        )
        arm_count = torch.tensor(count_valid_arms(shapes), dtype=torch.long)
        target, arm_instance = self._rasterize_or_load_cached_target(image.size, shapes, idx)
        channel_names = CHANNEL_NAMES
        if not self.config.include_centerline:
            target = torch.cat((target[0:1], target[2:]), dim=0)
            channel_names = CHANNEL_NAMES_WITHOUT_CENTERLINE
        ignore = _rasterize_ignore(image.size, ignore_polys)
        # Never delete supervision on the annotated star, even if a neighbor
        # polygon overlaps it.
        ignore[target[0] > 0] = 0.0
        # Drop instance supervision under ignore polygons (unannotated neighbor
        # stars) so the embedding loss never learns from them.
        arm_instance = arm_instance.clone()
        arm_instance[(ignore > 0) & (arm_instance == 0)] = -1
        return {
            'image': image_tensor,
            'target': target,
            'ignore': ignore.unsqueeze(0),
            'arm_instance': arm_instance,
            'arm_count': arm_count,
            'bridge_pairs': bridge_pairs,
            'bridge_valid': bridge_valid,
            'bridge_count': bridge_count,
            'meta': {
                'image_path': to_repo_relative(image_path),
                'original_size': original_size,
                'image_size': image.size,
                'row': row,
                'channel_names': channel_names,
            },
        }

    def _resized_image_cache_path(self, idx: int, image_path: Path, size: tuple[int, int] | None) -> Path | None:
        cache_dir = self.config.image_cache_dir
        if cache_dir is None or size is None:
            return None
        try:
            st = image_path.stat()
        except OSError:
            return None
        payload = {
            'version': 'hydra_core_img_v1',
            'idx': idx,
            'path': to_repo_relative(image_path),
            'mtime': int(st.st_mtime),
            'bytes': int(st.st_size),
            'size': list(size),
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
        return cache_dir / f'{digest}.pt'

    def _load_cached_resized_image(self, idx: int, image_path: Path, size: tuple[int, int] | None) -> tuple[Image.Image, tuple[int, int]] | None:
        cache_path = self._resized_image_cache_path(idx, image_path, size)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = torch.load(cache_path, map_location='cpu', weights_only=False)
            image = Image.fromarray(payload['image']).convert('RGB')
            return image, tuple(payload['original_size'])
        except Exception:
            return None

    def _save_cached_resized_image(self, idx: int, image_path: Path, size: tuple[int, int] | None, image: Image.Image, original_size: tuple[int, int]) -> None:
        cache_path = self._resized_image_cache_path(idx, image_path, size)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(image.convert('RGB'), dtype=np.uint8)
        tmp_path = cache_path.with_name(f'.{cache_path.name}.{os.getpid()}.tmp')
        torch.save({'image': arr, 'original_size': list(original_size)}, tmp_path)
        tmp_path.replace(cache_path)

    def _target_cache_path(self, size: tuple[int, int], shapes: Sequence[dict[str, Any]], idx: int) -> Path | None:
        cache_dir = self.config.target_cache_dir
        if cache_dir is None or self.config.augment:
            return None
        payload = {
            'version': 'hydra_core_v2',
            'idx': idx,
            'size': size,
            'shapes': shapes,
            'support_arm_width': self.config.support_arm_width,
            'centerline_radius': self.config.centerline_radius,
            'arm_base_sigma': self.config.arm_base_sigma,
            'arm_tip_sigma': self.config.arm_tip_sigma,
            'point_sigma': self.config.point_sigma,
            'trim_overlapping_arm_bases': self.config.trim_overlapping_arm_bases,
            'base_trim_kernel_radius_sigmas': self.config.base_trim_kernel_radius_sigmas,
            'base_trim_max_fraction': self.config.base_trim_max_fraction,
            'base_trim_min_remaining_length': self.config.base_trim_min_remaining_length,
            'include_centerline': self.config.include_centerline,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()
        return cache_dir / f'{digest}.pt'

    def _rasterize_or_load_cached_target(self, size: tuple[int, int], shapes: Sequence[dict[str, Any]], idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._target_cache_path(size, shapes, idx)
        if cache_path is not None and cache_path.exists():
            payload = torch.load(cache_path, map_location='cpu', weights_only=False)
            return payload['target'], payload['arm_instance']
        target, arm_instance = rasterize_targets(
            size,
            shapes,
            support_arm_width=self.config.support_arm_width,
            centerline_radius=self.config.centerline_radius,
            arm_base_sigma=self.config.arm_base_sigma,
            arm_tip_sigma=self.config.arm_tip_sigma,
            point_sigma=self.config.point_sigma,
            return_arm_instance=True,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(f'.{cache_path.name}.{os.getpid()}.tmp')
            torch.save({'target': target, 'arm_instance': arm_instance}, tmp_path)
            tmp_path.replace(cache_path)
        return target, arm_instance


def _scale_shapes(shapes: Sequence[dict[str, Any]], sx: float, sy: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for shape in shapes:
        s = dict(shape)
        raw = list(s.get('points') or [])
        s['points'] = [float(v) * (sx if i % 2 == 0 else sy) for i, v in enumerate(raw)]
        out.append(s)
    return out


def _augment_image_and_shapes(
    image: Image.Image,
    shapes: Sequence[dict[str, Any]],
    config: HydraDatasetConfig,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    shapes_out = [dict(s) for s in shapes]
    width, height = image.size
    if random.random() < config.hflip_p:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        shapes_out = _flip_shapes(shapes_out, width=width, height=height, horizontal=True)
    if random.random() < config.vflip_p:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        shapes_out = _flip_shapes(shapes_out, width=width, height=height, horizontal=False)
    if config.rotate_degrees > 0:
        angle = random.uniform(-config.rotate_degrees, config.rotate_degrees)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        shapes_out = _rotate_shapes(shapes_out, width=width, height=height, angle_degrees=angle)
    image = _augment_photometric(image, config)
    return image, shapes_out


def _augment_photometric(image: Image.Image, config: HydraDatasetConfig) -> Image.Image:
    """Image-only augmentations; annotation coordinates are unchanged."""
    image = image.convert('RGB')
    if config.brightness_jitter > 0:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(1.0 - config.brightness_jitter, 1.0 + config.brightness_jitter))
    if config.contrast_jitter > 0:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(1.0 - config.contrast_jitter, 1.0 + config.contrast_jitter))
    if config.saturation_jitter > 0:
        image = ImageEnhance.Color(image).enhance(random.uniform(1.0 - config.saturation_jitter, 1.0 + config.saturation_jitter))
    if config.hue_jitter > 0:
        hsv = np.asarray(image.convert('HSV'), dtype=np.uint8).copy()
        shift = int(round(random.uniform(-config.hue_jitter, config.hue_jitter) * 255))
        hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 256).astype(np.uint8)
        image = Image.fromarray(hsv, 'HSV').convert('RGB')
    if config.grayscale_p > 0 and random.random() < config.grayscale_p:
        gray = np.asarray(image.convert('L'), dtype=np.uint8)
        image = Image.fromarray(np.stack([gray, gray, gray], axis=-1), 'RGB')
    if config.gamma_jitter > 0:
        gamma = random.uniform(max(0.1, 1.0 - config.gamma_jitter), 1.0 + config.gamma_jitter)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        image = Image.fromarray(np.uint8(np.clip(np.power(np.clip(arr, 0, 1), gamma), 0, 1) * 255), 'RGB')
    if config.blur_p > 0 and config.blur_radius > 0 and random.random() < config.blur_p:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, config.blur_radius)))
    if config.noise_std > 0:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = arr + np.random.normal(0.0, config.noise_std, arr.shape).astype(np.float32)
        image = Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255), 'RGB')
    return image


def _rasterize_ignore(size: tuple[int, int], polys: Sequence[dict[str, Any]]) -> torch.Tensor:
    """Fill 'star_ignore' polygons into a float mask (H, W) of {0, 1}."""
    width, height = size
    if not polys:
        return torch.zeros((height, width), dtype=torch.float32)
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    for s in polys:
        pts = _shape_points(s)
        if len(pts) >= 3:
            draw.polygon([(float(x), float(y)) for x, y in pts], fill=1)
    return torch.from_numpy(np.asarray(img, dtype=np.float32))


def _degrade_to_usb_quality(image: Image.Image) -> Image.Image:
    """Simulate the low-quality behavioral-experiment USB camera pipeline on a
    clean field image: defocus, loss of native resolution, sensor noise,
    compression, contrast/brightness crush, frequent grayscale. Geometry is
    unchanged, so annotations remain valid (the sharp original's labels become
    supervision for the degraded view)."""
    w, h = image.size
    if random.random() < 0.7:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))
    # Crush to a simulated native resolution comparable to the lab crops.
    sim_native = random.uniform(96.0, 256.0)
    scale = min(1.0, sim_native / max(w, h))
    small = image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    arr = arr + np.random.normal(0.0, random.uniform(0.005, 0.03), arr.shape).astype(np.float32)
    small = Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255), 'RGB')
    if random.random() < 0.5:
        buf = io.BytesIO()
        small.save(buf, format='JPEG', quality=random.randint(40, 80))
        buf.seek(0)
        small = Image.open(buf).convert('RGB')
    image = small.resize((w, h), Image.Resampling.BILINEAR)
    if random.random() < 0.6:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.45, 0.85))
    if random.random() < 0.6:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.55, 0.95))
    if random.random() < 0.5:
        gray = np.asarray(image.convert('L'), dtype=np.uint8)
        image = Image.fromarray(np.stack([gray, gray, gray], axis=-1), 'RGB')
    return image


def _rotate_shapes(shapes: Sequence[dict[str, Any]], *, width: int, height: int, angle_degrees: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    # PIL Image.rotate uses image coordinates with y increasing downward; use the
    # opposite mathematical sign so annotation coordinates follow the pixels.
    theta = math.radians(-angle_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    for shape in shapes:
        s = dict(shape)
        raw = list(s.get('points') or [])
        rotated = []
        for i in range(0, len(raw) - 1, 2):
            dx = float(raw[i]) - cx
            dy = float(raw[i + 1]) - cy
            rotated.append(cx + cos_t * dx - sin_t * dy)
            rotated.append(cy + sin_t * dx + cos_t * dy)
        s['points'] = rotated
        out.append(s)
    return out


def _flip_shapes(shapes: Sequence[dict[str, Any]], *, width: int, height: int, horizontal: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for shape in shapes:
        s = dict(shape)
        raw = list(s.get('points') or [])
        flipped = []
        for i, v in enumerate(raw):
            if i % 2 == 0:
                flipped.append((width - 1) - float(v) if horizontal else float(v))
            else:
                flipped.append(float(v) if horizontal else (height - 1) - float(v))
        s['points'] = flipped
        out.append(s)
    return out
