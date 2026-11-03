# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""T3: tracked kinematics. Every function needs a per-arm ``track_id`` (A,) (stored or from
:func:`noodle_polylines.tracking.relink`); arms with ``track_id < 0`` are ignored.

Deformation lag variants (T3.1):

* ``'last_arm_frame'`` (default, Fig 5, prepare_figure_05_measures.py:199-202): compare with the last frame
  that had arms and a finite body origin; ``dt`` spans any gap.
* ``'prev_frame_reset'`` (Fig 4D/4E, prepare_figure_04_kinematics.py:141-156): compare with the previous
  recorded frame only; a frame without a star or arms resets the reference, so the next frame is NaN.
"""
from __future__ import annotations

import numpy as np

from . import _csr
from . import _reference as R
from . import _segments as SG
from . import geometry as G
from .container import PolylineContainer
from .frames import _r0_per_frame

MODEL_SIZE = 512
LAGS = ('last_arm_frame', 'prev_frame_reset')


def rigid_residual(a: np.ndarray, b: np.ndarray) -> float:
    """RMS distance between two (k,2) shapes after the best rotation + translation (Kabsch, SVD)."""
    return R.rigid_residual(a, b)


_PAIR_BLOCK = 1 << 15


def rigid_residuals(a: np.ndarray, b: np.ndarray, block: int = _PAIR_BLOCK) -> np.ndarray:
    """(N,) :func:`rigid_residual` of every pair of (N, k, 2) float64 shapes, bit-identical.

    The script's expression is evaluated on stacked pairs: the row-by-row ``mean(0)``, ``ca.T @ cb``,
    the 2x2 SVD, ``sign(det(u @ vt))``, ``u @ diag(1, d) @ vt`` and the squared residuals all run as
    stacked ``matmul`` / LAPACK gufuncs, which apply the same kernel to each slice as the 2-D call does;
    the final ``.mean()`` over k uses numpy's pairwise order (``_csr.seg_sum``).
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    N, k = a.shape[:2]
    out = np.empty(N)
    for s in range(0, N, block):
        pa, pb = a[s:s + block], b[s:s + block]
        n = len(pa)
        sa = np.zeros((n, 2))
        sb = np.zeros((n, 2))
        for j in range(k):                       # a.mean(0): accumulated row by row from zero
            sa += pa[:, j]
            sb += pb[:, j]
        ca = pa - (sa / k)[:, None, :]
        cb = pb - (sb / k)[:, None, :]
        u, _s, vt = np.linalg.svd(np.matmul(np.swapaxes(ca, 1, 2), cb))
        d = np.sign(np.linalg.det(np.matmul(u, vt)))
        diag = np.zeros((n, 2, 2))
        diag[:, 0, 0] = 1.0
        diag[:, 1, 1] = d
        rot = np.matmul(np.matmul(u, diag), vt)
        e = np.matmul(ca, rot)
        e -= cb
        e *= e
        sq = e[..., 0] + e[..., 1]               # .sum(1) of two values
        out[s:s + n] = np.sqrt(_csr.seg_sum(sq.ravel(), np.arange(n, dtype=np.int64) * k, np.full(n, k)) / k)
    return out


def _shapes_of(c: PolylineContainer, base: np.ndarray, arms: np.ndarray, r0: np.ndarray, order: str,
               af: np.ndarray) -> np.ndarray:
    """T1.6 shapes of the given arms only (elementwise identical to shape12 of all arms)."""
    f = af[arms]
    return G.shape12(base[arms], c.crop[f, 2], r0[f], order=order)


def _track(c: PolylineContainer, track_id) -> np.ndarray:
    if track_id is None:
        if c.track_id is None:
            raise ValueError('no track_id: pass one (tracking.relink) or use a container with stored IDs')
        return c.track_id
    t = np.asarray(track_id, np.int64)
    if t.shape != (c.n_arms,):
        raise ValueError('track_id must be (A,)')
    return t


def _link(ref_key: np.ndarray, ref_tid: np.ndarray, q_key: np.ndarray, q_tid: np.ndarray) -> np.ndarray:
    """For each query (key, tid), the position in the ref arrays of the LAST ref entry with the same
    (key, tid), or -1. This is the scripts' ``{tid: value for arm in frame}`` dict (later arms overwrite
    earlier ones) followed by ``tid in prev`` lookups, for all frames at once."""
    if not len(ref_key) or not len(q_key):
        return np.full(len(q_key), -1, np.int64)
    u, inv = np.unique(np.concatenate([ref_tid, q_tid]), return_inverse=True)
    m = np.int64(len(u))
    rk = ref_key.astype(np.int64) * m + inv[:len(ref_tid)]
    qk = q_key.astype(np.int64) * m + inv[len(ref_tid):]
    order = np.argsort(rk, kind='stable')
    rks = rk[order]
    last = np.ones(len(rks), bool)
    last[:-1] = rks[1:] != rks[:-1]
    ukeys, upos = rks[last], order[last]
    i = np.minimum(np.searchsorted(ukeys, qk), len(ukeys) - 1)
    return np.where(ukeys[i] == qk, upos[i], -1)


def _frame_median(values: np.ndarray, frame: np.ndarray, F: int) -> np.ndarray:
    """(F,) ``np.median`` of the values of each frame (``frame`` non-decreasing); NaN for frames without."""
    out = np.full(F, np.nan)
    if not len(values):
        return out
    counts = np.bincount(frame, minlength=F)
    f = np.flatnonzero(counts)
    st = SG.seg_starts(counts)[f]
    out[f] = SG.seg_median(SG.seg_sort(values, st, counts[f]), st, counts[f])
    return out


def _pairs_last_arm_frame(af: np.ndarray, tid: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ref arm, cur arm) pairs of the 'last_arm_frame' lag: every arm with a track in an ``ok`` frame,
    linked to the last arm of its track in the previous ``ok`` frame. Ordered by cur arm."""
    okf = np.flatnonzero(ok)
    ordinal = np.full(len(ok), -1, np.int64)
    ordinal[okf] = np.arange(len(okf))
    arms = np.flatnonzero((ordinal[af] >= 0) & (tid >= 0))
    k = ordinal[af[arms]]
    q = k >= 1
    hit = _link(k, tid[arms], k[q] - 1, tid[arms[q]])
    cur = arms[q][hit >= 0]
    ref = arms[hit[hit >= 0]]
    return ref, cur


def _pairs_prev_frame(af: np.ndarray, tid: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ref arm, cur arm) pairs of the 'prev_frame_reset' lag: per masked frame f and track, the last arm
    of the track in f linked to its last arm in frame f - 1 (also masked). Ordered by (frame, track)."""
    arms = np.flatnonzero(mask[af] & (tid >= 0))
    f = af[arms]
    # the cur side is a dict too: one entry (the last arm) per (frame, track)
    u, inv = np.unique(tid[arms], return_inverse=True)
    key = f * np.int64(max(len(u), 1)) + inv
    order = np.argsort(key, kind='stable')
    ks = key[order]
    last = np.ones(len(ks), bool)
    last[:-1] = ks[1:] != ks[:-1]
    cur_all = arms[order[last]]                  # ordered by (frame, track)
    fc = af[cur_all]
    q = fc >= 1
    hit = _link(fc, tid[cur_all], fc[q] - 1, tid[cur_all[q]])
    cur = cur_all[q][hit >= 0]
    ref = cur_all[hit[hit >= 0]]
    return ref, cur


def arm_deformation(c: PolylineContainer, r0, track_id: np.ndarray | None = None, lag: str = 'last_arm_frame',
                    shape_order: str | None = None, shape_mask: np.ndarray | None = None,
                    base: np.ndarray | None = None) -> np.ndarray:
    """T3.1 (F,) median over linked arms of the rigid-fit RMS shape residual / R0 / dt (R0 / s).

    Shapes are T1.6 (12-point base-first resample x side/512 / R0). ``shape_order`` defaults to 'fig4' for
    ``prev_frame_reset`` and 'fig5' for ``last_arm_frame`` (last-bit differences only). ``shape_mask`` (F,)
    restricts the frames that contribute shapes (Fig 4D uses star frames within [start - 1, stop]); the
    default is star frames for ``prev_frame_reset``. Raw values; Fig 4D plots ``log10(max(x, 1e-3))``.

    The track joins are sorted-key lookups over all frames at once and the residuals are batched
    (:func:`rigid_residuals`); the values are bit-identical to the scripts' per-frame dict loops.
    """
    if lag not in LAGS:
        raise ValueError(f'lag must be one of {LAGS}')
    c.require('crop')
    tid = _track(c, track_id)
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    order = shape_order or ('fig4' if lag == 'prev_frame_reset' else 'fig5')
    t = c.times()
    fo = c.frame_arm_offsets
    af = c.arm_frame()
    if lag == 'prev_frame_reset':
        mask = c.star_mask() if shape_mask is None else (np.asarray(shape_mask, bool) & c.star_mask())
        ref, cur = _pairs_prev_frame(af, tid, mask)
    else:
        c.require('body_origin_model')
        ok = (np.diff(fo) > 0) & np.all(np.isfinite(c.body_origin_model), axis=1)
        if shape_mask is not None:
            ok &= np.asarray(shape_mask, bool)
        ref, cur = _pairs_last_arm_frame(af, tid, ok)
    if not len(cur):
        return np.full(F, np.nan)
    if base is None:
        base = G.base_shapes(c)
    res = np.empty(len(cur))
    for s in range(0, len(cur), _PAIR_BLOCK):
        rb, cb = ref[s:s + _PAIR_BLOCK], cur[s:s + _PAIR_BLOCK]
        res[s:s + _PAIR_BLOCK] = rigid_residuals(_shapes_of(c, base, rb, r0, order, af),
                                                 _shapes_of(c, base, cb, r0, order, af))
    fc = af[cur]
    with np.errstate(divide='ignore', invalid='ignore'):
        bends = res / (t[fc] - t[af[ref]])
    return _frame_median(bends, fc, F)


def tip_motion(c: PolylineContainer, r0, track_id: np.ndarray | None = None) -> np.ndarray:
    """T3.2 (F,) median over linked arms of the body-relative tip displacement / R0 / dt (R0 / s).

    ``rel_tip = (tip - body_origin) * (side/512) / R0``; an arm counts when its track had a tip in the last
    frame with arms and a finite origin (prepare_figure_05_measures.py:192-198,203-205). Batched like
    :func:`arm_deformation`; bit-identical to the per-frame loop.
    """
    c.require('crop', 'body_origin_model')
    tid = _track(c, track_id)
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    t = c.times()
    fo = c.frame_arm_offsets
    af = c.arm_frame()
    ok = (np.diff(fo) > 0) & np.all(np.isfinite(c.body_origin_model), axis=1)
    ref, cur = _pairs_last_arm_frame(af, tid, ok)
    if not len(cur):
        return np.full(F, np.nan)
    tips = c.points[c.arm_point_offsets[1:] - 1]

    def rel(arms):
        f = af[arms]
        with np.errstate(divide='ignore', invalid='ignore'):
            return (tips[arms] - c.body_origin_model[f]) * (c.crop[f, 2] / MODEL_SIZE)[:, None] / r0[f][:, None]

    fc = af[cur]
    with np.errstate(divide='ignore', invalid='ignore'):
        moves = SG.row_norm(rel(cur) - rel(ref)) / (t[fc] - t[af[ref]])
    return _frame_median(moves, fc, F)


def tip_angle_track(c: PolylineContainer, track_id: np.ndarray | None = None, origin_dtype: str = 'auto',
                    angle_dtype: str = 'auto') -> dict[str, np.ndarray]:
    """T3.3 (A,) tip angle (deg) and radius (crop px) with each arm's frame time and track ID (Fig 6B/C).

    See :func:`noodle_polylines.geometry.tip_polar` for the dtype options. ``'auto'`` follows the panel of
    the container's ``meta['source_format']``: flipping -> the Fig 6C kymograph (float32 origin, float32
    angles, prepare_figure_06_flip.py:117-118); anything else -> Fig 6B (float64 origin and angles)."""
    c.require('body_origin_model')
    tid = _track(c, track_id)
    flipping = c.meta.get('source_format') == 'flipping'
    if origin_dtype == 'auto':
        origin_dtype = 'float32' if flipping else 'float64'
    if angle_dtype == 'auto':
        angle_dtype = 'float32' if flipping and origin_dtype == 'float32' else 'float64'
    angle, radius = G.tip_polar(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.body_origin_model,
                                origin_dtype=origin_dtype, angle_dtype=angle_dtype)
    af = c.arm_frame()
    return {'frame': af, 'frame_index': c.frame_index[af], 'time_s': c.times()[af], 'track_id': tid,
            'tip_angle_deg': angle, 'tip_radius_px': radius}
