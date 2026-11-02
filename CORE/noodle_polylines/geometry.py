# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""T0 primitives and T1 per-arm geometry over CSR polylines.

Every function takes the CSR arrays (``points`` (P,2) float32, ``arm_point_offsets`` (A+1,),
optionally ``frame_arm_offsets`` (F+1,) and per-frame context) and returns arrays aligned to arms
(A, ...) or points (P, ...). Units: crop-512 model px unless the name says source px.
"""
from __future__ import annotations

import numpy as np

from . import _reference as R

MODEL_SIZE = 512
K = 12


def arm_frame(frame_arm_offsets: np.ndarray) -> np.ndarray:
    """(A,) frame position of every arm."""
    return np.repeat(np.arange(len(frame_arm_offsets) - 1, dtype=np.int64), np.diff(frame_arm_offsets))


def point_arm(arm_point_offsets: np.ndarray) -> np.ndarray:
    """(P,) arm of every point."""
    return np.repeat(np.arange(len(arm_point_offsets) - 1, dtype=np.int64), np.diff(arm_point_offsets))


def arm_tips(points: np.ndarray, arm_point_offsets: np.ndarray) -> np.ndarray:
    """(A,2) last point (tip) of every arm, in the dtype of ``points``."""
    return points[arm_point_offsets[1:] - 1]


def arm_bases(points: np.ndarray, arm_point_offsets: np.ndarray) -> np.ndarray:
    """(A,2) first point (base) of every arm."""
    return points[arm_point_offsets[:-1]]


# ------------------------------------------------------------------------------------ T0.2 resampling
def resample_base(points: np.ndarray, arm_point_offsets: np.ndarray, k: int = K) -> np.ndarray:
    """T0.2 base-first resample: (A,k,2) float64, ``k`` points uniform in arc length (``np.interp``).

    Degenerate arms (arc length <= 0) repeat the base point. Used by arm deformation (T1.6/T3.1).
    Source: prepare_figure_05_measures.py:75-82.
    """
    A = len(arm_point_offsets) - 1
    out = np.empty((A, k, 2), np.float64)
    for a in range(A):
        out[a] = R.resample_arm(points[arm_point_offsets[a]:arm_point_offsets[a + 1]], k)
    return out


def resample_tip(points: np.ndarray, arm_point_offsets: np.ndarray, k: int = K,
                 arm_origin: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """T0.2 tip-first resample (the tracker's input): ((A,k,2) float32, (A,) float64 arc length).

    With ``arm_origin`` (A,2) the arms are made body-relative first (T0.5). The origin's dtype selects
    the arithmetic: float64 (figure scripts: ``poly - origin`` in float64) or float32 (runtime /
    Fig 6 stats: ``poly - origin.astype(float32)``). Degenerate arms (one point or length <= 1e-6)
    repeat the tip with length 0. Source: fused_tracker.py:62-81.
    """
    A = len(arm_point_offsets) - 1
    out = np.empty((A, k, 2), np.float32)
    lengths = np.empty(A, np.float64)
    for a in range(A):
        p = points[arm_point_offsets[a]:arm_point_offsets[a + 1]]
        if arm_origin is not None:
            o = arm_origin[a]
            p = (np.asarray(p, np.float64) - o) if o.dtype == np.float64 else (p - o)
        out[a], lengths[a] = R.tip_anchored_resample(p, k)
    return out, lengths


# ------------------------------------------------------------------------------------ T0.3 / T1.1
def arc_length_curvature(points: np.ndarray, arm_point_offsets: np.ndarray, two_point: str = 'zero'
                         ) -> tuple[np.ndarray, np.ndarray]:
    """T0.3 arc length and T1.1 curvature, (A,) float64 each, in model px and rad / model px.

    Arithmetic is float32 on the stored float32 points, exactly as both figure-script functions.
    ``two_point='zero'``: a 2-point arm has curvature 0.0 (``retrace_released.poly_metrics``, the export
    columns used by Fig 4); ``'nan'``: NaN (``prepare_figure_05_measures.arm_metrics``, Fig 5). A 1-point
    arm has length 0 and curvature NaN in both.
    """
    if two_point not in ('zero', 'nan'):
        raise ValueError("two_point must be 'zero' or 'nan'")
    fn = R.poly_metrics if two_point == 'zero' else R.arm_metrics
    A = len(arm_point_offsets) - 1
    length = np.empty(A, np.float64)
    curvature = np.empty(A, np.float64)
    for a in range(A):
        res = fn(points[arm_point_offsets[a]:arm_point_offsets[a + 1]])
        length[a], curvature[a] = res[0], res[1]
    return length, curvature


# ------------------------------------------------------------------------------------ T0.4 / T0.5
def to_source(points: np.ndarray, arm_point_offsets: np.ndarray, frame_arm_offsets: np.ndarray,
              crop: np.ndarray) -> np.ndarray:
    """T0.4 (P,2) float64 source-video px: ``x_src = left + x * side / 512`` (retrace_released.py:579-583)."""
    per_arm = arm_frame(frame_arm_offsets)
    per_point = per_arm[point_arm(arm_point_offsets)]
    c = crop[per_point]
    out = np.asarray(points, np.float64).copy()
    out[:, 0] = c[:, 0] + out[:, 0] * c[:, 2] / MODEL_SIZE
    out[:, 1] = c[:, 1] + out[:, 1] * c[:, 2] / MODEL_SIZE
    return out


def tips_source(points, arm_point_offsets, frame_arm_offsets, crop) -> np.ndarray:
    """(A,2) float64 tips in source px (the export's ``tip_source_x/y``)."""
    tips = np.asarray(arm_tips(points, arm_point_offsets), np.float64)
    c = crop[arm_frame(frame_arm_offsets)]
    out = tips.copy()
    out[:, 0] = c[:, 0] + tips[:, 0] * c[:, 2] / MODEL_SIZE
    out[:, 1] = c[:, 1] + tips[:, 1] * c[:, 2] / MODEL_SIZE
    return out


def origin_source(origin_model: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """(F,2) body origin in source px (trace_sources.flip_origin_source)."""
    out = np.empty_like(origin_model, dtype=np.float64)
    out[:, 0] = crop[:, 0] + origin_model[:, 0] * crop[:, 2] / 512.0
    out[:, 1] = crop[:, 1] + origin_model[:, 1] * crop[:, 2] / 512.0
    return out


def body_relative(points: np.ndarray, arm_point_offsets: np.ndarray, frame_arm_offsets: np.ndarray,
                  origin: np.ndarray) -> np.ndarray:
    """T0.5 (P,2) float64 ``P - body_origin_model[frame]`` (crop px)."""
    per_point = arm_frame(frame_arm_offsets)[point_arm(arm_point_offsets)]
    return np.asarray(points, np.float64) - origin[per_point]


# ------------------------------------------------------------------------------------ T1.2 - T1.6
def norm_length(arc_length: np.ndarray, frame_arm_offsets: np.ndarray, radius_model: np.ndarray) -> np.ndarray:
    """T1.2 (A,) arc length / max(per-frame model radius, 1e-6), NaN where the radius is NaN (body radii)."""
    r = radius_model[arm_frame(frame_arm_offsets)]
    out = np.full(len(arc_length), np.nan)
    ok = np.isfinite(r)
    out[ok] = arc_length[ok] / np.maximum(r[ok], 1e-6)
    return out


def tip_polar(points: np.ndarray, arm_point_offsets: np.ndarray, frame_arm_offsets: np.ndarray,
              origin: np.ndarray, origin_dtype: str = 'float64', angle_dtype: str = 'float64'
              ) -> tuple[np.ndarray, np.ndarray]:
    """T1.3 (A,) tip angle (deg, counter-clockwise with image y down) and radius (crop px) about the origin.

    ``angle = degrees(atan2(-y, x)) mod 360`` of the raw tip minus the origin. ``origin_dtype`` picks the
    subtraction: 'float64' (Sleepy, Fig 5/6 example) or 'float32' (flipping ``stats.flipping_sequences``).
    ``angle_dtype='float32'`` reproduces the float32 angles of the Fig 6B/C copies (prepare_figure_06_flip.py:117).
    Source: prepare_figure_06_stats.py:225-230.
    """
    tips32 = arm_tips(points, arm_point_offsets)
    o = origin[arm_frame(frame_arm_offsets)]
    if origin_dtype == 'float32':
        rel = tips32 - o.astype(np.float32)
    elif origin_dtype == 'float64':
        rel = tips32 - np.asarray(o, np.float64)
    else:
        raise ValueError("origin_dtype must be 'float32' or 'float64'")
    if angle_dtype == 'float32':
        if rel.dtype != np.float32:
            raise ValueError("angle_dtype='float32' needs origin_dtype='float32'")
        angle = np.degrees(np.arctan2(-rel[:, 1], rel[:, 0])) % 360
    else:
        rel = np.asarray(rel, float)
        angle = np.degrees(np.arctan2(-rel[:, 1], rel[:, 0])) % 360
    radius = np.linalg.norm(np.asarray(rel, float), axis=1)
    return angle, radius


def body_path(points: np.ndarray, arm_point_offsets: np.ndarray, frame_arm_offsets: np.ndarray,
              origin: np.ndarray, radius_model: np.ndarray, decimals: int | None = None) -> np.ndarray:
    """T1.4 (P,2) arm paths in body radii about the body origin, y flipped up (Fig 4B left).

    ``((x - cx) / r, (cy - y) / r)`` with the per-frame model radius r; ``decimals=3`` reproduces the JSON.
    Source: prepare_figure_04.py:116-127.
    """
    per_point = arm_frame(frame_arm_offsets)[point_arm(arm_point_offsets)]
    p = np.asarray(points, np.float64)
    o = origin[per_point]
    r = radius_model[per_point]
    xy = np.column_stack(((p[:, 0] - o[:, 0]) / r, (o[:, 1] - p[:, 1]) / r))
    return xy if decimals is None else np.round(xy, decimals)


def shape12(base: np.ndarray, crop_side: np.ndarray, r0, order: str = 'fig5') -> np.ndarray:
    """T1.6 (A,k,2) base-first shapes in source px / R0.

    ``order='fig5'``: ``base * (side / 512) / r0`` (prepare_figure_05_measures.py:181);
    ``order='fig4'``: ``base * side / 512 / r0`` (prepare_figure_04_kinematics.py:149-150). The two differ
    in the last bits only. ``crop_side`` and ``r0`` are per arm (A,) or scalars.
    """
    side = np.asarray(crop_side, np.float64).reshape(-1, 1, 1) if np.ndim(crop_side) else float(crop_side)
    r0 = np.asarray(r0, np.float64).reshape(-1, 1, 1) if np.ndim(r0) else float(r0)
    if order == 'fig5':
        return base * (side / MODEL_SIZE) / r0
    if order == 'fig4':
        return base * side / MODEL_SIZE / r0
    raise ValueError("order must be 'fig4' or 'fig5'")


def anchored_ghosts(points: np.ndarray, arm_point_offsets: np.ndarray, frame_arm_offsets: np.ndarray,
                    crop: np.ndarray, origin: np.ndarray, t_rel: np.ndarray, window: tuple[float, float],
                    labels: np.ndarray | None = None, min_path: int = 10) -> dict:
    """T1.5 Fig 5A ghosts: every arm of a time window anchored to the window's reference frame.

    The reference frame is the window frame nearest the window centre (first on ties). Each arm is moved to
    source px, translated so its frame's body origin lands on the reference origin, and mapped back to the
    reference crop px. Arms without a track label (``labels < 0``, e.g. frames the tracker skipped) are
    dropped, as in the script. Returns ``points`` (Q,2), ``arm_point_offsets``, ``labels``, ``frame``
    positions, ``t_ref`` and ``tip_paths`` (label -> (n,2) anchored tips, n >= ``min_path``).
    Source: prepare_figure_05_example.py:137-145. Raises ValueError for an empty window.
    """
    lo, hi = window
    win = np.flatnonzero((t_rel >= lo) & (t_rel <= hi))
    if not len(win):
        raise ValueError('min() arg is an empty sequence')
    centre = (lo + hi) / 2
    ref = int(win[np.argmin(np.abs(t_rel[win] - centre))])
    scale_ref = crop[ref, 2] / MODEL_SIZE
    centre_ref_src = origin[ref] * scale_ref + crop[ref, :2]
    out_pts, out_n, out_lab, out_frame = [], [], [], []
    paths: dict[int, list] = {}
    for f in win:
        scale = crop[f, 2] / MODEL_SIZE
        centre_src = origin[f] * scale + crop[f, :2]
        for a in range(frame_arm_offsets[f], frame_arm_offsets[f + 1]):
            lab = -1 if labels is None else int(labels[a])
            if labels is not None and lab < 0:
                continue
            poly = points[arm_point_offsets[a]:arm_point_offsets[a + 1]]
            poly_src = poly * scale + crop[f, :2]
            anchored = centre_ref_src + (poly_src - centre_src)
            g = (anchored - crop[ref, :2]) / scale_ref
            out_pts.append(g); out_n.append(len(g)); out_lab.append(lab); out_frame.append(f)
            paths.setdefault(lab, []).append(g[-1])
    from .container import _offsets
    return {'points': np.concatenate(out_pts) if out_pts else np.zeros((0, 2)),
            'arm_point_offsets': _offsets(np.asarray(out_n, np.int64)),
            'labels': np.asarray(out_lab, np.int64), 'frame': np.asarray(out_frame, np.int64),
            't_ref': float(t_rel[ref]), 'ref_frame': ref,
            'tip_paths': {k: np.asarray(v) for k, v in paths.items() if len(v) >= min_path}}
