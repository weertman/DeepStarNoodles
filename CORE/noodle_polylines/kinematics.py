# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""T3: tracked kinematics. Every function needs a per-arm ``track_id`` (A,) (stored or from
:func:`noodle_polylines.tracking.relink`); arms with ``track_id < 0`` are ignored.

Deformation lag variants (T3.1):

* ``'last_arm_frame'`` (default, Fig 5, prepare_figure_05_measures.py:199-202): compare with the last frame
  that had arms and a finite body origin; ``dt`` spans any gap.
* ``'prev_frame_reset'`` (Fig 4D/4E, prepare_figure_04_kinematics.py:141-156): compare with the previous
  recorded frame only; a frame without a star or arms resets the reference, so the next frame is NaN.

Rigid fit (T3.1): ``kabsch='closed'`` (default) solves the 2-D Kabsch rotation in closed form,
``theta = atan2(sum cross, sum dot)`` over the centred shapes, which is the best proper rotation, as the
scripts' SVD with its reflection guard is. It agrees with the scripts' SVD to about 1e-15 absolute
(5.7e-12 relative at worst over the 1.9M linked pairs of the released corpora); ``kabsch='svd'``
evaluates the scripts' SVD expression on stacked pairs and is bit-identical to them.
"""
from __future__ import annotations

import numpy as np

from . import _csr
from . import _reference as R
from . import _segments as SG
from . import geometry as G
from .container import PolylineContainer, _offsets, _ranges
from .frames import _r0_per_frame

MODEL_SIZE = 512
LAGS = ('last_arm_frame', 'prev_frame_reset')
KABSCH = ('closed', 'svd')


def rigid_residual(a: np.ndarray, b: np.ndarray) -> float:
    """RMS distance between two (k,2) shapes after the best rotation + translation (Kabsch, SVD)."""
    return R.rigid_residual(a, b)


_PAIR_BLOCK = 1 << 15
_CLOSED_BLOCK = 1 << 13          # pairs per closed-form block: its (n, k) planes stay in cache
_SPAN_FRACTION = 0.8            # resample a whole arm span when at least this fraction of it is linked


def _centre_planes(P: np.ndarray) -> None:
    """Centre (2, n, k) shape planes in place by each shape's mean, summed row by row from zero as
    ``ndarray.mean(0)`` does for a (k, 2) shape (so the centred values equal the scripts' ``a - a.mean(0)``)."""
    k = P.shape[2]
    if not P.shape[1]:
        return
    s = np.zeros(P.shape[:2])
    for j in range(k):
        s += P[:, :, j]
    s /= k
    P -= s[:, :, None]


def _closed_rows(ax, ay, bx, by) -> np.ndarray:
    """RMS residual of centred (n, k) x/y planes after the best proper rotation (closed form).

    Rotating row vectors by ``[[c, s], [-s, c]]`` maps ``(x, y)`` to ``(x c - y s, x s + y c)``; the fit
    maximises ``c * sum(ax bx + ay by) + s * sum(ax by - ay bx)``, so ``(c, s) = (p, q) / hypot(p, q)``
    (identity when ``p = q = 0``, as the SVD of a zero matrix gives). Only proper rotations are
    parameterised: this is the SVD fit with its ``sign(det)`` reflection guard."""
    k = ax.shape[1]
    p = np.einsum('ij,ij->i', ax, bx)
    p += np.einsum('ij,ij->i', ay, by)
    q = np.einsum('ij,ij->i', ax, by)
    q -= np.einsum('ij,ij->i', ay, bx)
    r = np.hypot(p, q)
    zero = r == 0
    r[zero] = 1.0
    c = p / r
    s = q / r
    c[zero] = 1.0
    s[zero] = 0.0
    c = c[:, None]
    s = s[:, None]
    ex = ax * c
    ex -= ay * s
    ex -= bx
    ey = ax * s
    ey += ay * c
    ey -= by
    ssq = np.einsum('ij,ij->i', ex, ex)
    ssq += np.einsum('ij,ij->i', ey, ey)
    ssq /= k
    return np.sqrt(ssq, out=ssq)


def _svd_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The scripts' SVD Kabsch on stacked (n, k, 2) float64 pairs, bit-identical to ``R.rigid_residual``.

    The row-by-row ``mean(0)``, ``ca.T @ cb``, the 2x2 SVD, ``sign(det(u @ vt))``, ``u @ diag(1, d) @ vt``
    and the squared residuals run as stacked ``matmul`` / LAPACK gufuncs, which apply the same kernel to
    each slice as the 2-D call does; the final ``.mean()`` over k uses numpy's pairwise order."""
    n, k = a.shape[:2]
    sa = np.zeros((n, 2))
    sb = np.zeros((n, 2))
    for j in range(k):                       # a.mean(0): accumulated row by row from zero
        sa += a[:, j]
        sb += b[:, j]
    ca = a - (sa / k)[:, None, :]
    cb = b - (sb / k)[:, None, :]
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
    return np.sqrt(_csr.seg_sum(sq.ravel(), np.arange(n, dtype=np.int64) * k, np.full(n, k)) / k)


def rigid_residuals(a: np.ndarray, b: np.ndarray, block: int = _PAIR_BLOCK, kabsch: str = 'closed') -> np.ndarray:
    """(N,) :func:`rigid_residual` of every pair of (N, k, 2) float64 shapes.

    ``kabsch='closed'``: closed-form 2-D rotation (module docstring), within about 1e-15 of the SVD;
    a block with a non-finite shape runs the SVD instead, which raises ``LinAlgError`` as the scripts do.
    ``kabsch='svd'``: the scripts' SVD expression on stacked pairs, bit-identical."""
    if kabsch not in KABSCH:
        raise ValueError(f'kabsch must be one of {KABSCH}')
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    N = a.shape[0]
    out = np.empty(N)
    if kabsch == 'svd':
        for s in range(0, N, block):
            out[s:s + block] = _svd_rows(a[s:s + block], b[s:s + block])
        return out
    step = min(block, _CLOSED_BLOCK)
    for s in range(0, N, step):
        pa = np.ascontiguousarray(np.moveaxis(a[s:s + step], 2, 0))
        pb = np.ascontiguousarray(np.moveaxis(b[s:s + step], 2, 0))
        _centre_planes(pa)
        _centre_planes(pb)
        res = _closed_rows(pa[0], pa[1], pb[0], pb[1])
        if not np.isfinite(res).all():
            res = _svd_rows(a[s:s + step], b[s:s + step])
        out[s:s + step] = res
    return out


def _base_rows(c: PolylineContainer, arms: np.ndarray, base: np.ndarray | None) -> np.ndarray:
    """(len(arms), K, 2) T0.2 base-first resamples of the given (increasing) arms only.

    From ``base`` when given, or the container's cache inside a caching scope; otherwise only these arms
    are resampled (each arm's resample depends on its own points only, so the rows are bit-identical to
    the rows of a whole-container :func:`geometry.resample_base`)."""
    if base is not None:
        return base[arms]
    if c.__dict__.get('_cache_depth'):
        return G.base_shapes(c)[arms]
    ao = c.arm_point_offsets
    if not len(arms):
        return np.empty((0, G.K, 2))
    a0, a1 = int(arms[0]), int(arms[-1]) + 1
    if len(arms) >= _SPAN_FRACTION * (a1 - a0):
        # most of the span is linked: resample the contiguous span (a slice, no point gather)
        p0 = int(ao[a0])
        b = G.resample_base(c.points[p0:int(ao[a1])], ao[a0:a1 + 1] - p0)
        return b if len(arms) == a1 - a0 else b[arms - a0]
    counts = (ao[1:] - ao[:-1])[arms]
    return G.resample_base(c.points[_ranges(ao[arms], counts)], _offsets(counts))


def _shape_planes(c: PolylineContainer, arms: np.ndarray, base: np.ndarray | None, r0: np.ndarray, order: str,
                  af: np.ndarray) -> np.ndarray:
    """(2, n, K) x/y planes of the T1.6 shapes of the given (increasing) arms: the same elementwise
    operations as :func:`geometry.shape12` (so the same values), written straight into plane layout."""
    b = _base_rows(c, arms, base)
    f = af[arms]
    side = c.crop[f, 2][None, :, None]
    r = r0[f][None, :, None]
    P = np.empty((2, len(arms), b.shape[1]))
    bt = np.moveaxis(b, 2, 0)
    if order == 'fig5':
        np.multiply(bt, side / MODEL_SIZE, out=P)
    elif order == 'fig4':
        np.multiply(bt, side, out=P)
        P /= MODEL_SIZE
    else:
        raise ValueError("order must be 'fig4' or 'fig5'")
    P /= r
    return P


def _pair_residuals(c: PolylineContainer, ref: np.ndarray, cur: np.ndarray, base: np.ndarray | None,
                    r0: np.ndarray, order: str, af: np.ndarray, kabsch: str) -> np.ndarray:
    """Rigid-fit residual of every (ref arm, cur arm) pair. Each arm's shape is built (and, for the closed
    form, centred) once, however many pairs it is in; only the arms in a pair are resampled."""
    need = np.zeros(c.n_arms, bool)
    need[ref] = True
    need[cur] = True
    arms = np.flatnonzero(need)
    pos = np.cumsum(need) - 1
    iref, icur = pos[ref], pos[cur]
    res = np.empty(len(cur))
    # shapes in blocks of arms (bounded memory); a pair's two arms lie within one frame gap of each other,
    # so blocks of pairs ordered by cur arm touch a bounded window of arms
    blk = _PAIR_BLOCK
    for s in range(0, len(cur), blk):
        ir, ic = iref[s:s + blk], icur[s:s + blk]
        lo = int(min(ir.min(), ic.min()))
        hi = int(max(ir.max(), ic.max())) + 1
        P = _shape_planes(c, arms[lo:hi], base, r0, order, af)
        ir = ir - lo
        ic = ic - lo
        if kabsch == 'svd':
            Pa = np.moveaxis(P, 0, 2)
            res[s:s + blk] = _svd_rows(Pa[ir], Pa[ic])
            continue
        _centre_planes(P)
        px, py = P[0], P[1]
        for t in range(0, len(ir), _CLOSED_BLOCK):
            jr, jc = ir[t:t + _CLOSED_BLOCK], ic[t:t + _CLOSED_BLOCK]
            out = _closed_rows(px[jr], py[jr], px[jc], py[jc])
            if not np.isfinite(out).all():          # non-finite shapes: the SVD (raises like the scripts)
                Pa = np.moveaxis(_shape_planes(c, arms[lo:hi], base, r0, order, af), 0, 2)
                out = _svd_rows(Pa[jr], Pa[jc])
            res[s + t:s + t + len(jr)] = out
    return res


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


def _want(frames, F: int) -> np.ndarray | None:
    """(F,) bool of requested output frames (bool mask or frame positions), or None for all."""
    if frames is None:
        return None
    w = np.asarray(frames)
    if w.dtype == bool:
        if w.shape != (F,):
            raise ValueError('frames mask must be (F,)')
        return w
    m = np.zeros(F, bool)
    m[w.astype(np.int64)] = True
    return m


def _keep(ref: np.ndarray, cur: np.ndarray, want: np.ndarray | None, af: np.ndarray):
    if want is None:
        return ref, cur
    k = want[af[cur]]
    return ref[k], cur[k]


def arm_deformation(c: PolylineContainer, r0, track_id: np.ndarray | None = None, lag: str = 'last_arm_frame',
                    shape_order: str | None = None, shape_mask: np.ndarray | None = None,
                    base: np.ndarray | None = None, frames=None, kabsch: str = 'closed',
                    _pairs: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """T3.1 (F,) median over linked arms of the rigid-fit RMS shape residual / R0 / dt (R0 / s).

    Shapes are T1.6 (12-point base-first resample x side/512 / R0). ``shape_order`` defaults to 'fig4' for
    ``prev_frame_reset`` and 'fig5' for ``last_arm_frame`` (last-bit differences only). ``shape_mask`` (F,)
    restricts the frames that contribute shapes (Fig 4D uses star frames within [start - 1, stop]); the
    default is star frames for ``prev_frame_reset``. Raw values; Fig 4D plots ``log10(max(x, 1e-3))``.

    ``frames`` (bool mask (F,) or frame positions) computes only those output frames, the rest NaN; the
    lag reference may lie outside them (a window's values equal the whole-sequence run's). ``kabsch``:
    'closed' (default) or 'svd' (bit-identical to the scripts), see the module docstring. ``base``
    (A, K, 2) optional precomputed :func:`geometry.resample_base`; otherwise only linked arms are
    resampled.

    The track joins are sorted-key lookups over all frames at once (index arithmetic for both lag
    variants), and every arm's shape is built once and fitted against its partner in batches.
    """
    if lag not in LAGS:
        raise ValueError(f'lag must be one of {LAGS}')
    if kabsch not in KABSCH:
        raise ValueError(f'kabsch must be one of {KABSCH}')
    c.require('crop')
    tid = _track(c, track_id)
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    order = shape_order or ('fig4' if lag == 'prev_frame_reset' else 'fig5')
    t = c.times()
    af = c.arm_frame()
    if _pairs is not None:
        ref, cur = _pairs
    elif lag == 'prev_frame_reset':
        mask = c.star_mask() if shape_mask is None else (np.asarray(shape_mask, bool) & c.star_mask())
        ref, cur = _pairs_prev_frame(af, tid, mask)
    else:
        ref, cur = _pairs_last_arm_frame(af, tid, _ok_frames(c, shape_mask))
    ref, cur = _keep(ref, cur, _want(frames, F), af)
    if not len(cur):
        return np.full(F, np.nan)
    res = _pair_residuals(c, ref, cur, base, r0, order, af, kabsch)
    fc = af[cur]
    with np.errstate(divide='ignore', invalid='ignore'):
        bends = res / (t[fc] - t[af[ref]])
    return _frame_median(bends, fc, F)


def _ok_frames(c: PolylineContainer, shape_mask=None) -> np.ndarray:
    """'last_arm_frame' frames: with arms and a finite body origin (and in ``shape_mask``)."""
    c.require('body_origin_model')
    ok = (np.diff(c.frame_arm_offsets) > 0) & np.all(np.isfinite(c.body_origin_model), axis=1)
    if shape_mask is not None:
        ok &= np.asarray(shape_mask, bool)
    return ok


def last_arm_frame_pairs(c: PolylineContainer, track_id: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(ref arm, cur arm) links of the 'last_arm_frame' lag (shared by T3.1 lag B and T3.2), for reuse
    across :func:`arm_deformation` and :func:`tip_motion` through their ``_pairs`` argument."""
    return _pairs_last_arm_frame(c.arm_frame(), _track(c, track_id), _ok_frames(c))


def tip_motion(c: PolylineContainer, r0, track_id: np.ndarray | None = None, frames=None,
               _pairs: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """T3.2 (F,) median over linked arms of the body-relative tip displacement / R0 / dt (R0 / s).

    ``rel_tip = (tip - body_origin) * (side/512) / R0``; an arm counts when its track had a tip in the last
    frame with arms and a finite origin (prepare_figure_05_measures.py:192-198,203-205). ``frames`` as in
    :func:`arm_deformation`. Batched like :func:`arm_deformation`; bit-identical to the per-frame loop.
    """
    c.require('crop', 'body_origin_model')
    tid = _track(c, track_id)
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    t = c.times()
    af = c.arm_frame()
    ref, cur = _pairs if _pairs is not None else _pairs_last_arm_frame(af, tid, _ok_frames(c))
    ref, cur = _keep(ref, cur, _want(frames, F), af)
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
