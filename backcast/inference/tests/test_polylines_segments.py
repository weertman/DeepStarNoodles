# Vendored from DeepStarNoodles tests/test_noodle_polylines_segments.py at f749a49; only imports and fixture paths
# changed (see UPSTREAM.json "noodle_polylines").
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Batched per-frame reductions (T2, T0.7 masks) and tracked kinematics (T3.1, T3.2) against the
per-frame loops they replaced, bit for bit, on edge cases the released data rarely hits: empty frames,
1-3 arms, 150-arm frames, tied values, NaN and inf, missing origins, frames without a star, per-frame
R0 with NaN and zero, duplicate track IDs in a frame, untracked arms and gaps."""
from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pytest


from deepstarnoodles_inference._vendor.noodle_polylines import _reference as R, _segments as SG  # noqa: E402
from deepstarnoodles_inference._vendor.noodle_polylines import frames as FR, geometry as G, kinematics as KI, scale as SC  # noqa: E402
from deepstarnoodles_inference._vendor.noodle_polylines.container import from_polylines  # noqa: E402
from deepstarnoodles_inference._vendor.noodle_polylines.presets import FIG4D, _window_deformation, compute_all  # noqa: E402

MODEL = 512


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.array_equal(a, b, equal_nan=True), np.nanmax(np.abs(a.astype(float) - b.astype(float)))


def close(a, b, rtol=1e-9):
    """The closed-form Kabsch default: NaN/inf positions exact, finite values within the gate's rtol 1e-9."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    fa, fb = np.isfinite(a), np.isfinite(b)
    assert np.array_equal(fa, fb) and np.array_equal(a[~fa], b[~fb], equal_nan=True)
    assert np.all(np.abs(a[fa] - b[fb]) <= rtol * np.abs(b[fb])), np.max(np.abs(a[fa] - b[fb]) / np.abs(b[fb]))


# ------------------------------------------------------------------------------------------ inputs
def _container(seed: int = 3, F: int = 400):
    rng = np.random.default_rng(seed)
    frames, tids = [], []
    for f in range(F):
        n = int(rng.choice([0, 0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 17, 150])) if f % 4 else int(rng.integers(0, 20))
        arms = []
        for i in range(n):
            k = int(rng.choice([1, 2, 3, 6, 12, 30]))
            p = 256 + np.cumsum(rng.normal(0, 4, (k, 2)), axis=0)
            if i % 7 == 3 and arms:          # an exact duplicate arm: tied lengths, tips and curvatures
                p = arms[-1]
            arms.append(np.asarray(p, np.float32))
        frames.append(arms)
        ids = rng.integers(-1, 6, n)          # small ID range: duplicates within a frame and -1
        tids.append(ids)
    side = rng.uniform(150, 400, F)
    crop = np.column_stack([rng.uniform(0, 900, F), rng.uniform(0, 500, F), side])
    origin = 256 + rng.normal(0, 10, (F, 2))
    origin[rng.random(F) < 0.08] = np.nan
    cen = 256 + rng.normal(0, 10, (F, 2))
    cen[rng.random(F) < 0.08, 0] = np.nan
    t = np.cumsum(rng.choice([0.5, 0.5, 1.0, 7.0], F))
    return from_polylines(frames, time_s=t, crop=crop, body_origin_model=origin, support_centroid_model=cen,
                          has_star=rng.random(F) > 0.1, support_area_model_px=rng.integers(0, 9000, F),
                          track_id=np.concatenate(tids).astype(np.int64), meta={'fps': 2.0})


@pytest.fixture(scope='module')
def c():
    return _container()


def _r0s(F: int, finite: bool = False):
    rng = np.random.default_rng(9)
    r = rng.uniform(20, 60, F)
    if not finite:
        r[::17] = np.nan
        r[::23] = 0.0
    return [37.5, r]


# ------------------------------------------------------------------------------------------ oracles
# The per-frame loops of the previous implementation (and of the figure scripts), kept as oracles.
def _slow_fold(c, r0, variant):
    F = c.n_frames
    r0 = FR._r0_per_frame(r0, F)
    ok = FR.valid_frames(c, variant)
    out = np.full(F, np.nan)
    fo, ao, pts = c.frame_arm_offsets, c.arm_point_offsets, c.points
    tips = G.tips_source(pts, ao, fo, c.crop)
    origin = G.origin_source(c.body_origin_model, c.crop)
    for f in np.flatnonzero(ok):
        a0, a1 = fo[f], fo[f + 1]
        if variant == 'fig4e':
            out[f] = float(np.linalg.norm(tips[a0:a1].mean(0) - origin[f]) / r0[f])
        elif variant == 'fig4d':
            scale = c.crop[f, 2] / MODEL
            offset = np.array([c.crop[f, 0], c.crop[f, 1]])
            t = np.array([np.asarray(pts[ao[a + 1] - 1], float) * scale + offset for a in range(a0, a1)])
            out[f] = float(np.linalg.norm(t.mean(0) - origin[f]) / r0[f])
        else:
            scale = c.crop[f, 2] / MODEL
            t = np.asarray([pts[ao[a + 1] - 1] for a in range(a0, a1)])
            out[f] = float(np.linalg.norm(t.mean(0) - np.asarray(c.body_origin_model[f], float)) * scale / r0[f])
    return out


def _slow_spread(c, r0, variant, L, min_arms):
    F = c.n_frames
    r0 = FR._r0_per_frame(r0, F)
    out = np.full(F, np.nan)
    fo = c.frame_arm_offsets
    for f in np.flatnonzero(FR.valid_frames(c, variant)):
        side = float(c.crop[f, 2])
        if variant == 'fig5':
            lengths = [float(x) * (c.crop[f, 2] / MODEL) / r0[f] for x in L[fo[f]:fo[f + 1]]]
        else:
            lengths = np.array([float(x) * side / MODEL for x in L[fo[f]:fo[f + 1]]]) / r0[f]
        if len(lengths) >= min_arms:
            q1, q3 = np.percentile(lengths, [25, 75])
            out[f] = float(q3 - q1)
    return out


def _slow_curv(c, r0, variant, kappa, reducer):
    F = c.n_frames
    r0 = FR._r0_per_frame(r0, F)
    out = np.full(F, np.nan)
    fo = c.frame_arm_offsets
    red = np.nanmedian if reducer == 'nanmedian' else np.median
    for f in np.flatnonzero(FR.valid_frames(c, variant)):
        if variant == 'fig5':
            scale = c.crop[f, 2] / MODEL
            out[f] = float(red([float(k) / scale * r0[f] for k in kappa[fo[f]:fo[f + 1]]]))
        else:
            side = float(c.crop[f, 2])
            out[f] = float(red([float(k) * MODEL / side for k in kappa[fo[f]:fo[f + 1]]]) * r0[f])
    return out


def _slow_isotropy(c, centre, min_tips):
    cen = getattr(c, centre)
    out = np.full(c.n_frames, np.nan)
    fo, ao, pts = c.frame_arm_offsets, c.arm_point_offsets, c.points
    for f in range(c.n_frames):
        if not np.all(np.isfinite(cen[f])) or fo[f + 1] - fo[f] < min_tips:
            continue
        out[f] = R.isotropy_frame(np.array([pts[ao[a + 1] - 1] for a in range(fo[f], fo[f + 1])]) - cen[f])
    return out


def _slow_deformation(c, r0, tid, lag, order, shape_mask=None):
    F = c.n_frames
    r0 = FR._r0_per_frame(r0, F)
    af = c.arm_frame()
    shapes = G.shape12(G.resample_base(c.points, c.arm_point_offsets), c.crop[af, 2], r0[af], order=order)
    t, fo = c.times(), c.frame_arm_offsets
    out = np.full(F, np.nan)
    if lag == 'prev_frame_reset':
        mask = c.star_mask() if shape_mask is None else (shape_mask & c.star_mask())
        prev_t, prev = None, {}
        for f in range(F):
            cur = {int(tid[a]): shapes[a] for a in range(fo[f], fo[f + 1]) if tid[a] >= 0} if mask[f] else {}
            if prev_t is not None and cur and prev:
                bends = [R.rigid_residual(prev[k], v) / (t[f] - prev_t) for k, v in cur.items() if k in prev]
                if bends:
                    out[f] = float(np.median(bends))
            prev_t, prev = t[f], cur
        return out
    ok = (np.diff(fo) > 0) & np.all(np.isfinite(c.body_origin_model), axis=1)
    if shape_mask is not None:
        ok &= shape_mask
    prev_t, prev = None, {}
    for f in np.flatnonzero(ok):
        arms = [a for a in range(fo[f], fo[f + 1]) if tid[a] >= 0]
        if prev_t is not None:
            bends = [R.rigid_residual(prev[int(tid[a])], shapes[a]) / (t[f] - prev_t) for a in arms
                     if int(tid[a]) in prev]
            if bends:
                out[f] = float(np.median(bends))
        prev, prev_t = {int(tid[a]): shapes[a] for a in arms}, t[f]
    return out


def _slow_tip_motion(c, r0, tid):
    F = c.n_frames
    r0 = FR._r0_per_frame(r0, F)
    t, fo, ao, pts = c.times(), c.frame_arm_offsets, c.arm_point_offsets, c.points
    ok = (np.diff(fo) > 0) & np.all(np.isfinite(c.body_origin_model), axis=1)
    out = np.full(F, np.nan)
    prev_t, prev = None, {}
    for f in np.flatnonzero(ok):
        scale = c.crop[f, 2] / MODEL
        arms = list(range(fo[f], fo[f + 1]))
        rel = (np.asarray([pts[ao[a + 1] - 1] for a in arms]) - np.asarray(c.body_origin_model[f], float)) \
            * scale / r0[f]
        if prev_t is not None:
            moves = [np.linalg.norm(rel[i] - prev[int(tid[a])]) / (t[f] - prev_t) for i, a in enumerate(arms)
                     if tid[a] >= 0 and int(tid[a]) in prev]
            if moves:
                out[f] = float(np.median(moves))
        prev, prev_t = {int(tid[a]): rel[i] for i, a in enumerate(arms) if tid[a] >= 0}, t[f]
    return out


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        yield


# ------------------------------------------------------------------------------------------ _segments
def test_segment_order_statistics_match_numpy():
    rng = np.random.default_rng(2)
    counts = rng.integers(0, 12, 4000)
    counts[::97] = rng.integers(12, 300, len(counts[::97]))
    st = SG.seg_starts(counts)
    vals = np.round(rng.normal(0, 5, int(counts.sum())), 1)          # many ties
    for k, v in ((0, np.nan), (1, np.inf), (2, -np.inf), (3, -0.0)):
        vals[rng.random(len(vals)) < 0.02] = v
    s = SG.seg_sort(vals, st, counts)
    seg = [vals[a:a + n] for a, n in zip(st, counts)]
    same(SG.seg_median(s, st, counts), [np.median(x) if len(x) else np.nan for x in seg])
    same(SG.seg_median(s, st, counts, skipna=True), [np.nanmedian(x) if len(x) else np.nan for x in seg])
    for q in (0, 25, 50, 75, 100, 12.5):
        same(SG.seg_percentile(s, st, counts, q), [np.percentile(x, q) if len(x) else np.nan for x in seg])


def test_segment_rowsum_and_norm_match_numpy():
    rng = np.random.default_rng(4)
    counts = rng.integers(0, 40, 3000)
    st = SG.seg_starts(counts)
    for dt in (np.float32, np.float64):
        x = (rng.random((int(counts.sum()), 2)) * 512 - 256).astype(dt)
        got = SG.seg_rowsum(x, st, counts)
        same(got, [x[a:a + n].sum(0) if n else np.zeros(2, dt) for a, n in zip(st, counts)])
    v = rng.normal(0, 1, (20000, 2)) * 10 ** rng.uniform(-6, 6, (20000, 1))
    same(SG.row_norm(v), [np.linalg.norm(r) for r in v])


# ------------------------------------------------------------------------------------------ T2
@pytest.mark.parametrize('variant', FR.VARIANTS)
def test_fold_spread_curvature_match_per_frame_loops(c, variant):
    two_point = 'nan' if variant == 'fig5' else 'zero'
    L, kappa = G.arc_length_curvature(c.points, c.arm_point_offsets, two_point)
    for r0 in _r0s(c.n_frames):
        same(FR.fold_asymmetry(c, r0, variant), _slow_fold(c, r0, variant))
        for m in (1, 4, 7):
            same(FR.arm_length_spread(c, r0, variant, L, min_arms=m), _slow_spread(c, r0, variant, L, m))
        for red in ('median', 'nanmedian'):
            same(FR.arm_curvature(c, r0, variant, kappa, reducer=red), _slow_curv(c, r0, variant, kappa, red))


def test_frame_lists_mean_matches_per_frame_loop(c):
    fl = FR.frame_lists_4b(c)
    nl, fo = fl['arm_norm_length'], c.frame_arm_offsets
    same(fl['mean_norm_length'], [float(np.mean([float(x) for x in nl[fo[f]:fo[f + 1]]])) if fo[f + 1] > fo[f]
                                  else np.nan for f in range(c.n_frames)])


@pytest.mark.parametrize('min_tips', [1, 4, 6])
def test_tip_isotropy_matches_per_frame_cov_eigvalsh(c, min_tips):
    for centre in ('support_centroid_model', 'body_origin_model'):
        same(FR.tip_isotropy(c, centre, min_tips), _slow_isotropy(c, centre, min_tips))


def test_tip_isotropy_degenerate_frames():
    # collinear, identical and nearly identical tips, plus a well-spread frame
    line = [np.array([[0, 0], [10 + i, 2 * i]], np.float32) for i in range(6)]
    same_tip = [np.array([[0, 0], [100, 100]], np.float32) for _ in range(5)]
    near = [np.array([[0, 0], [100 + 1e-4 * i, 100]], np.float32) for i in range(5)]
    ring = [np.array([[256, 256], [256 + 80 * np.cos(a), 256 + 80 * np.sin(a)]], np.float32)
            for a in np.linspace(0, 6, 9)]
    c = from_polylines([line, same_tip, near, ring], support_centroid_model=np.full((4, 2), 3.0))
    same(FR.tip_isotropy(c), _slow_isotropy(c, 'support_centroid_model', 4))


# ------------------------------------------------------------------------------------------ T0.7 masks
def test_in_any_window_matches_broadcast():
    rng = np.random.default_rng(5)
    t = np.round(rng.uniform(-5, 105, 5000), 1)
    t[::50] = np.nan
    lo = np.round(rng.uniform(0, 100, 40), 1)
    w = np.column_stack([lo, lo + np.round(rng.uniform(-1, 6, 40), 1)])   # some reversed (empty)
    w = np.vstack([w, [[10, 10], [10, 12], [12, 15], [np.nan, 3], [50, np.nan], [-np.inf, -4]]])
    ref = np.any((t[:, None] >= w[None, :, 0]) & (t[:, None] <= w[None, :, 1]), axis=1)
    same(SC.in_any_window(t, w), ref)
    same(SC.in_any_window(t, []), np.zeros(len(t), bool))
    same(SC.in_any_window(t, [(3.0, 1.0)]), np.zeros(len(t), bool))


# ------------------------------------------------------------------------------------------ T3
def test_rigid_residuals_match_reference():
    rng = np.random.default_rng(6)
    a = rng.normal(0, 1, (3000, 12, 2)) * 10 ** rng.uniform(-3, 1, (3000, 1, 1))
    b = a[:, ::-1] + rng.normal(0, 1e-3, a.shape)
    b[::3] = rng.normal(0, 1, b[::3].shape)
    a[::5] = a[::5, :1]                      # all points equal
    ref = [R.rigid_residual(x, y) for x, y in zip(a, b)]
    same(KI.rigid_residuals(a, b, block=1000, kabsch='svd'), ref)
    got = KI.rigid_residuals(a, b, block=1000)              # closed form: spec bound 1e-10 abs vs the SVD
    assert np.max(np.abs(got - ref)) <= 1e-10
    close(got[np.asarray(ref) > 1e-6], np.asarray(ref)[np.asarray(ref) > 1e-6])


def test_closed_kabsch_reflection_and_degenerate_cases():
    """Closed form vs the scripts' SVD (with its sign(det) guard) where the guard bites: mirrored shapes,
    collinear (rank-1) shapes, a single repeated point on either side, exact rotations, and huge offsets."""
    rng = np.random.default_rng(11)
    n = 4000
    a = rng.normal(0, 1, (n, 12, 2))
    th = rng.uniform(-np.pi, np.pi, n)
    rot = np.stack([np.stack([np.cos(th), np.sin(th)], -1), np.stack([-np.sin(th), np.cos(th)], -1)], 1)
    b = np.matmul(a, rot) + rng.normal(0, 1e-2, a.shape)
    b[0::7, :, 0] *= -1                                     # mirror images: det(H) < 0, the guard flips
    line = np.linspace(-1, 1, 12)[None, :, None] * rng.normal(0, 1, (n, 1, 2))
    a[1::7] = line[1::7]                                    # collinear arms (rank-1 cross-covariance)
    b[2::7] = line[2::7] * -1.0
    a[3::7] = a[3::7, :1]                                   # all points equal (zero cross-covariance)
    b[4::7] = b[4::7, :1]
    b[5::7] = np.matmul(a[5::7], rot[5::7])                 # exact rotation (residual ~0)
    a[6::7] += 1e6                                          # large offsets (centring)
    ref = np.array([R.rigid_residual(x, y) for x, y in zip(a, b)])
    got = KI.rigid_residuals(a, b, block=999)
    assert np.max(np.abs(got - ref)) <= 1e-10
    big = ref > 1e-3
    close(got[big], ref[big], rtol=1e-9)
    for bad in (np.nan, np.inf):                            # non-finite shapes raise like the scripts' SVD
        z = a[:10].copy()
        z[3, 4, 1] = bad
        with pytest.raises(np.linalg.LinAlgError):
            R.rigid_residual(z[3], b[3])
        with pytest.raises(np.linalg.LinAlgError):
            KI.rigid_residuals(z, b[:10])
    with pytest.raises(ValueError):
        KI.rigid_residuals(a, b, kabsch='qr')


@pytest.mark.parametrize('lag', KI.LAGS)
def test_deformation_matches_per_frame_dict_loop(c, lag):
    tid = c.track_id
    t = c.times()
    for r0 in _r0s(c.n_frames, finite=True):
        for order in ('fig4', 'fig5'):
            slow = _slow_deformation(c, r0, tid, lag, order)
            same(KI.arm_deformation(c, r0, tid, lag=lag, shape_order=order, kabsch='svd'), slow)
            close(KI.arm_deformation(c, r0, tid, lag=lag, shape_order=order), slow)
        mask = (t > t[40]) & (t < t[300])
        slow = _slow_deformation(c, r0, tid, lag, 'fig4', mask)
        same(KI.arm_deformation(c, r0, tid, lag=lag, shape_order='fig4', shape_mask=mask, kabsch='svd'), slow)
        close(KI.arm_deformation(c, r0, tid, lag=lag, shape_order='fig4', shape_mask=mask), slow)
    # a NaN R0 on a linked frame makes the scripts' SVD fail; the batched path fails the same way
    bad = _r0s(c.n_frames)[1]
    for fn in (lambda: KI.arm_deformation(c, bad, tid, lag=lag), lambda: _slow_deformation(c, bad, tid, lag, 'fig5'),
               lambda: KI.arm_deformation(c, bad, tid, lag=lag, kabsch='svd')):
        with pytest.raises(np.linalg.LinAlgError):
            fn()


def test_tip_motion_matches_per_frame_dict_loop(c):
    for r0 in _r0s(c.n_frames):
        same(KI.tip_motion(c, r0, c.track_id), _slow_tip_motion(c, r0, c.track_id))
    untracked = np.full(c.n_arms, -1, np.int64)
    same(KI.tip_motion(c, 40.0, untracked), np.full(c.n_frames, np.nan))
    same(KI.arm_deformation(c, 40.0, untracked), np.full(c.n_frames, np.nan))


def test_window_deformation_matches_whole_sequence_runs(c):
    t = c.times()
    windows = [(t[50], t[80]), (t[70], t[120]), (t[390], t[399] + 50), (t[200], t[199])]
    base = G.resample_base(c.points, c.arm_point_offsets)
    context, lookback = FIG4D.deformation_window
    first, per = _window_deformation(c, 41.0, c.track_id, FIG4D, windows, base, kabsch='svd')
    first_c, per_c = _window_deformation(c, 41.0, c.track_id, FIG4D, windows, None)   # closed, lazy resample
    taken = np.zeros(c.n_frames, bool)
    ref_first = np.full(c.n_frames, np.nan)
    for k, (start, stop) in enumerate(windows):
        lo, hi = start - context, stop + context
        d = _slow_deformation(c, 41.0, c.track_id, FIG4D.deformation_lag, FIG4D.shape_order,
                              (t >= lo - lookback) & (t <= hi))
        inside = (t >= lo) & (t <= hi)
        same(per[k][inside], d[inside])
        close(per_c[k][inside], d[inside])
        assert np.all(np.isnan(per[k][~inside]))
        ref_first[inside & ~taken] = d[inside & ~taken]
        taken |= inside
    same(first, ref_first)
    c2 = c.with_fields(track_id=np.maximum(c.track_id, 0))        # stored IDs: compute_all uses them
    for kabsch in ('svd', 'closed'):
        out = compute_all(c2, 'fig4d', windows=windows, kabsch=kabsch)
        first2, per2 = _window_deformation(c2, out['r0'], c2.track_id, FIG4D, windows, base, kabsch=kabsch)
        same(out['arm_deformation_window'], per2)
        same(out['arm_deformation'], first2)
        same(out['log_deformation'], np.log10(np.maximum(first2, FIG4D.log10_floor)))


# ------------------------------------------------------------------------------------------ T3 round 4
@pytest.mark.parametrize('kabsch', KI.KABSCH)
@pytest.mark.parametrize('lag', KI.LAGS)
def test_requested_frames_equal_whole_sequence(c, lag, kabsch):
    """``frames=`` computes only the requested output frames; their values equal the whole-sequence run's
    (the lag reference may lie outside the request), the rest are NaN."""
    tid = c.track_id
    whole = KI.arm_deformation(c, 41.0, tid, lag=lag, kabsch=kabsch)
    motion = KI.tip_motion(c, 41.0, tid)
    rng = np.random.default_rng(2)
    for want in ((np.arange(c.n_frames) >= 120) & (np.arange(c.n_frames) < 160), rng.random(c.n_frames) < 0.1,
                 np.array([0, 5, 6, 7, 399]), np.zeros(c.n_frames, bool)):
        m = np.zeros(c.n_frames, bool)
        m[want] = True
        exp = np.where(m, whole, np.nan)
        same(KI.arm_deformation(c, 41.0, tid, lag=lag, kabsch=kabsch, frames=want), exp)
        if lag == 'last_arm_frame':
            same(KI.tip_motion(c, 41.0, tid, frames=want), np.where(m, motion, np.nan))
    with pytest.raises(ValueError):
        KI.arm_deformation(c, 41.0, tid, frames=np.ones(3, bool))
    with pytest.raises(ValueError):
        KI.arm_deformation(c, 41.0, tid, kabsch='qr')


def test_subset_resample_and_base_sources_agree(c):
    """Resampling only the linked arms gives the rows of a whole-container resample, bit for bit; a given
    ``base``, a caching scope and the lazy path give the same deformation."""
    base = G.resample_base(c.points, c.arm_point_offsets)
    arms = np.flatnonzero(np.random.default_rng(4).random(c.n_arms) < 0.3)
    same(KI._base_rows(c, arms, None), base[arms])
    same(KI._base_rows(c, np.arange(c.n_arms), None), base)
    same(KI._base_rows(c, arms[:0], None), base[:0])
    for kabsch in KI.KABSCH:
        lazy = KI.arm_deformation(c, 41.0, c.track_id, kabsch=kabsch)
        same(KI.arm_deformation(c, 41.0, c.track_id, kabsch=kabsch, base=base), lazy)
        with c.caching():
            same(KI.arm_deformation(c, 41.0, c.track_id, kabsch=kabsch), lazy)


def test_pair_blocks_and_shared_pairs(c, monkeypatch):
    """Small pair and closed-form blocks give the same values; compute_all shares the lag-B links with tip
    motion and gives the separate calls' values."""
    tid = c.track_id
    ref = {k: KI.arm_deformation(c, 41.0, tid, kabsch=k) for k in KI.KABSCH}
    ref_a = KI.arm_deformation(c, 41.0, tid, lag='prev_frame_reset')
    monkeypatch.setattr(KI, '_PAIR_BLOCK', 37)
    monkeypatch.setattr(KI, '_CLOSED_BLOCK', 5)
    for k in KI.KABSCH:
        same(KI.arm_deformation(c, 41.0, tid, kabsch=k), ref[k])
    same(KI.arm_deformation(c, 41.0, tid, lag='prev_frame_reset'), ref_a)
    monkeypatch.undo()
    pairs = KI.last_arm_frame_pairs(c, tid)
    same(KI.arm_deformation(c, 41.0, tid, _pairs=pairs), ref['closed'])
    c2 = c.with_fields(track_id=np.maximum(tid, 0))               # stored IDs: compute_all uses them
    out = compute_all(c2, 'fig5', r0=41.0)
    same(out['arm_deformation'], KI.arm_deformation(c2, 41.0))
    same(out['tip_motion'], KI.tip_motion(c2, 41.0))
    assert 'resample_base' not in out
