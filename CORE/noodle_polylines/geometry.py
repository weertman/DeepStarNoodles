# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""T0 primitives and T1 per-arm geometry over CSR polylines.

Every function takes the CSR arrays (``points`` (P,2) float32, ``arm_point_offsets`` (A+1,),
optionally ``frame_arm_offsets`` (F+1,) and per-frame context) and returns arrays aligned to arms
(A, ...) or points (P, ...). Units: crop-512 model px unless the name says source px.
"""
from __future__ import annotations

import numpy as np

from . import _csr
from .container import _ranges

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


# ------------------------------------------------------------------------------------ batching
# Kernels run over blocks of arms so their temporaries stay small (tens of MB).
CHUNK_ARMS = 1 << 13     # 8,192 arms: temporaries stay in cache-sized blocks (faster than larger blocks)


def _chunks(arm_point_offsets: np.ndarray, chunk: int = CHUNK_ARMS):
    """Yield ``(a0, a1, p0, p1, rebased offsets)`` for consecutive blocks of at most ``chunk`` arms."""
    A = len(arm_point_offsets) - 1
    for a0 in range(0, A, chunk):
        a1 = min(a0 + chunk, A)
        ao = np.asarray(arm_point_offsets[a0:a1 + 1], np.int64)
        p0, p1 = int(ao[0]), int(ao[-1])
        yield a0, a1, p0, p1, ao - p0


def _points_per_frame(arm_point_offsets: np.ndarray, frame_arm_offsets: np.ndarray) -> np.ndarray:
    """(F,) number of points in each frame (to broadcast per-frame values to points with ``np.repeat``)."""
    return np.diff(np.asarray(arm_point_offsets)[np.asarray(frame_arm_offsets)])


def _seg_norm(points: np.ndarray) -> np.ndarray:
    """(P-1,) ``np.linalg.norm(np.diff(points, axis=0), axis=1)`` in the dtype of ``points``, bit-identical
    (``sqrt(dx*dx + dy*dy)`` is what the 2-element reduction computes) without the slow short-axis reduce."""
    if len(points) < 2:
        return np.zeros(0, points.dtype)
    d = np.diff(points, axis=0)
    dx, dy = d[:, 0], d[:, 1]
    out = dx * dx
    out += dy * dy
    return np.sqrt(out, out=out)


def _targets(total: np.ndarray, k: int) -> np.ndarray:
    """(n,k) ``np.linspace(0, total, k)`` for every ``total``, with numpy's arithmetic (step, denormal
    branch, exact endpoint)."""
    y = np.broadcast_to(np.arange(k, dtype=np.float64), (len(total), k)).copy()
    div = k - 1
    if div <= 0:
        return y * total[:, None]
    step = total / div
    zero = step == 0
    y *= step[:, None]
    if zero.any():
        y[zero] = (np.arange(k, dtype=np.float64) / div)[None, :] * total[zero, None]
    y[:, -1] = total
    return y


# ------------------------------------------------------------------------------------ T0.2 resampling
def resample_base(points: np.ndarray, arm_point_offsets: np.ndarray, k: int = K) -> np.ndarray:
    """T0.2 base-first resample: (A,k,2) float64, ``k`` points uniform in arc length (``np.interp``).

    Degenerate arms (arc length <= 0) repeat the base point. Used by arm deformation (T1.6/T3.1).
    Source: prepare_figure_05_measures.py:75-82. Batched over arms (no per-arm loop); bit-identical to
    the per-arm ``np.interp`` (same cumulative sums, search and interpolation formula).
    """
    A = len(arm_point_offsets) - 1
    out = np.empty((A, k, 2), np.float64)
    for a0, a1, p0, p1, ao in _chunks(arm_point_offsets):
        out[a0:a1] = _resample_base_block(np.asarray(points[p0:p1], np.float64), ao, k)
    return out


def _resample_base_block(poly: np.ndarray, ao: np.ndarray, k: int) -> np.ndarray:
    s, e = ao[:-1], ao[1:]
    n = e - s
    out = np.empty((len(s), k, 2), np.float64)
    # arc = r_[0, cumsum(norm(diff(poly)))] per arm, in the global point layout
    arc = np.zeros(len(poly), np.float64)
    arc[1:] = _seg_norm(poly)
    arc[s] = 0.0
    _csr.seg_cumsum(arc, s, n, out=arc)
    total = arc[e - 1]
    deg = total <= 0
    if deg.any():
        out[deg] = poly[s[deg]][:, None, :]
    idx = np.flatnonzero(~deg)
    if not len(idx):
        return out
    x = _targets(total[idx], k)
    hi = e[idx][:, None]
    j = _csr.seg_search(arc, x, s[idx][:, None], hi, strict=False)   # largest j with arc[j] <= x
    last = j == hi - 1
    j1 = np.where(last, j, j + 1)
    xj, xj1 = arc[j], arc[j1]
    on_node = last | (xj == x)
    with np.errstate(divide='ignore', invalid='ignore'):
        for d in (0, 1):
            fj, fj1 = poly[j, d], poly[j1, d]
            slope = (fj1 - fj) / (xj1 - xj)
            res = slope * (x - xj) + fj
            bad = np.isnan(res) & ~on_node
            if bad.any():           # np.interp: "if we get nan in one direction, try the other"
                alt = slope * (x - xj1) + fj1
                alt = np.where(np.isnan(alt) & (fj == fj1), fj, alt)
                res = np.where(bad, alt, res)
            out[idx, :, d] = np.where(on_node, fj, res)
    return out


def resample_tip(points: np.ndarray, arm_point_offsets: np.ndarray, k: int = K,
                 arm_origin: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """T0.2 tip-first resample (the tracker's input): ((A,k,2) float32, (A,) float64 arc length).

    With ``arm_origin`` (A,2) the arms are made body-relative first (T0.5). The origin's dtype selects
    the arithmetic: float64 (figure scripts: ``poly - origin`` in float64) or float32 (runtime /
    Fig 6 stats: ``poly - origin.astype(float32)``). Degenerate arms (one point or length <= 1e-6)
    repeat the tip with length 0. Source: fused_tracker.py:62-81. Batched over arms; bit-identical to
    the per-arm walk (float32 cumulative length, float64 interpolation, float32 output).
    """
    A = len(arm_point_offsets) - 1
    out = np.empty((A, k, 2), np.float32)
    lengths = np.zeros(A, np.float64)
    if arm_origin is not None:
        arm_origin = np.asarray(arm_origin)
    for a0, a1, p0, p1, ao in _chunks(arm_point_offsets):
        p = points[p0:p1]
        if arm_origin is not None:
            o = np.repeat(arm_origin[a0:a1], np.diff(ao), axis=0)
            p = (np.asarray(p, np.float64) - o) if arm_origin.dtype == np.float64 else (p - o)
        out[a0:a1], lengths[a0:a1] = _resample_tip_block(np.asarray(p, np.float32), ao, k)
    return out, lengths


def _resample_tip_block(p32: np.ndarray, ao: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    s, e = ao[:-1], ao[1:]
    n = e - s
    A = len(s)
    out = np.empty((A, k, 2), np.float32)
    L = np.zeros(A, np.float64)
    # tip first: reverse every arm in place of the global layout
    rev = np.repeat(s + e - 1, n) - np.arange(len(p32), dtype=np.int64)
    pts = p32[rev]
    seg = _seg_norm(pts)
    cum32 = np.zeros(len(pts), np.float32)
    cum32[1:] = seg
    cum32[s] = 0
    _csr.seg_cumsum(cum32, s, n, out=cum32)
    cum = cum32.astype(np.float64)
    total = cum[e - 1]
    deg = (n == 1) | (total <= 1e-6)
    if deg.any():
        out[deg] = pts[s[deg]][:, None, :]
    idx = np.flatnonzero(~deg)
    if not len(idx):
        return out, L
    T = total[idx]
    L[idx] = T
    d = _targets(T, k)
    j = _csr.seg_search(cum, d, s[idx][:, None], e[idx][:, None] - 1, strict=True)   # walk: first j with d <= cum[j+1]
    t = (d - cum[j]) / np.maximum(seg[j].astype(np.float64), 1e-6)
    P = pts.astype(np.float64)
    res = (1.0 - t)[..., None] * P[j] + t[..., None] * P[j + 1]
    out[idx] = res.astype(np.float32)
    return out, L


# ------------------------------------------------------------------------------------ T0.3 / T1.1
def arc_length_curvature(points: np.ndarray, arm_point_offsets: np.ndarray, two_point: str = 'zero'
                         ) -> tuple[np.ndarray, np.ndarray]:
    """T0.3 arc length and T1.1 curvature, (A,) float64 each, in model px and rad / model px.

    Arithmetic is float32 on the stored float32 points, exactly as both figure-script functions.
    ``two_point='zero'``: a 2-point arm has curvature 0.0 (``retrace_released.poly_metrics``, the export
    columns used by Fig 4); ``'nan'``: NaN (``prepare_figure_05_measures.arm_metrics``, Fig 5). A 1-point
    arm has length 0 and curvature NaN in both. Batched over arms: segment sums use numpy's pairwise
    order and the angle unwrap its sequential correction sum, so values are bit-identical.
    """
    if two_point not in ('zero', 'nan'):
        raise ValueError("two_point must be 'zero' or 'nan'")
    A = len(arm_point_offsets) - 1
    length = np.empty(A, np.float64)
    curvature = np.empty(A, np.float64)
    for a0, a1, p0, p1, ao in _chunks(arm_point_offsets):
        length[a0:a1], curvature[a0:a1] = _arc_curv_block(np.asarray(points[p0:p1]), ao, two_point)
    return length, curvature


def _arc_curv_block(pts: np.ndarray, ao: np.ndarray, two_point: str) -> tuple[np.ndarray, np.ndarray]:
    s, e = ao[:-1], ao[1:]
    n = e - s
    A = len(s)
    seg = np.diff(pts, axis=0) if len(pts) > 1 else np.zeros((0, 2), pts.dtype)   # segment g: points g, g+1
    norms = _seg_norm(pts)
    nseg = np.maximum(n - 1, 0)
    lsum = _csr.seg_sum(norms, s, nseg)                     # norm(segments).sum() per arm
    length = lsum.astype(np.float64)
    curvature = np.full(A, np.nan)
    if two_point == 'zero':
        curvature[n == 2] = 0.0
    big = np.flatnonzero(n >= 3)
    if len(big):
        ang = np.arctan2(seg[:, 1], seg[:, 0])
        # np.unwrap on every arm's angles at once: elementwise steps on the global diff ...
        dd = np.diff(ang)                                   # dd[g] = ang[g+1] - ang[g], valid for g in [s, e-3]
        period = 2 * np.pi
        interval_high = period / 2
        interval_low = -interval_high
        discont = period / 2
        with np.errstate(invalid='ignore'):
            ddmod = np.mod(dd - interval_low, period) + interval_low
            np.copyto(ddmod, interval_high, where=(ddmod == interval_low) & (dd > 0))
            ph = ddmod - dd
            np.copyto(ph, 0, where=abs(dd) < discont)
        # ... and the sequential correction sum only on arms whose corrections are not all zero
        sb, cb = s[big], n[big] - 2
        nz = np.zeros(len(ph) + 1, np.int64)
        np.cumsum(ph != 0, out=nz[1:])
        wrap = (nz[sb + cb] - nz[sb]) > 0
        up = ang
        if wrap.any():
            sw, cw = sb[wrap], cb[wrap]
            csum = _csr.seg_cumsum(ph, sw, cw)
            g = _ranges(sw, cw)
            up = ang.copy()
            up[g + 1] = ang[g + 1] + csum[g]
        absd = np.abs(np.diff(up))
        ksum = _csr.seg_sum(absd, sb, cb)
        den = np.maximum(length[big], 1e-6).astype(pts.dtype)
        curvature[big] = (ksum / den).astype(np.float64)
    return length, curvature


# ------------------------------------------------------------------------------------ per-container cache
_GEOM_DEPS = ('points', 'arm_point_offsets')


def base_shapes(c, k: int = K) -> np.ndarray:
    """:func:`resample_base` of a container (T1.6 shapes, T3.1 deformation).

    Shared and read-only inside a :meth:`PolylineContainer.caching` scope, else a fresh array."""
    return c.derived(('resample_base', k), _GEOM_DEPS, lambda: resample_base(c.points, c.arm_point_offsets, k))


def arm_metrics(c, two_point: str = 'zero') -> tuple[np.ndarray, np.ndarray]:
    """:func:`arc_length_curvature` of a container (shared and read-only inside a
    :meth:`PolylineContainer.caching` scope; used by the T2 aggregates)."""
    if two_point not in ('zero', 'nan'):
        raise ValueError("two_point must be 'zero' or 'nan'")
    return c.derived(('arc_length_curvature', two_point), _GEOM_DEPS,
                     lambda: arc_length_curvature(c.points, c.arm_point_offsets, two_point))


# ------------------------------------------------------------------------------------ T0.4 / T0.5
def to_source(points: np.ndarray, arm_point_offsets: np.ndarray, frame_arm_offsets: np.ndarray,
              crop: np.ndarray) -> np.ndarray:
    """T0.4 (P,2) float64 source-video px: ``x_src = left + x * side / 512`` (retrace_released.py:579-583)."""
    ppf = _points_per_frame(arm_point_offsets, frame_arm_offsets)
    side = np.repeat(crop[:, 2], ppf)
    out = np.asarray(points, np.float64).copy()
    for d in (0, 1):
        col = out[:, d]
        col *= side
        col /= MODEL_SIZE
        col += np.repeat(crop[:, d], ppf)        # left + (...) == (...) + left
    return out


def tips_source(points, arm_point_offsets, frame_arm_offsets, crop) -> np.ndarray:
    """(A,2) float64 tips in source px (the export's ``tip_source_x/y``)."""
    tips = np.asarray(arm_tips(points, arm_point_offsets), np.float64)
    c = np.repeat(crop, np.diff(frame_arm_offsets), axis=0)
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
    o = np.repeat(np.asarray(origin, np.float64), _points_per_frame(arm_point_offsets, frame_arm_offsets), axis=0)
    out = np.asarray(points, np.float64).copy()
    out -= o
    return out


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
    ``angle_dtype='float32'`` reproduces the float32 angles of the Fig 6C kymograph (prepare_figure_06_flip.py:117);
    Fig 6B (06_example) casts its polylines to float64, so its angles are float64.
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
    ppf = _points_per_frame(arm_point_offsets, frame_arm_offsets)
    p = np.asarray(points, np.float64)
    o = np.repeat(origin, ppf, axis=0)
    r = np.repeat(radius_model, ppf)
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
