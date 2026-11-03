# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Batched association (T4.1-T4.5) against the per-pair and per-frame code it replaced, bit for bit:
the common-span distance over whole cost blocks, the array-state tracker against the reference
FusedTracker (every LSA matrix and solution), step costs, the vectorised Fig 6D counters, and the
near-tie audit. Inputs include single-point and zero-length arms, equal lengths, frames of 0-300 arms,
NaN origins, gaps, states with None, and every tracker variant in both directions."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CORE'))

from noodle_polylines import _assoc as A, _reference as R, tracking as TR  # noqa: E402
from noodle_polylines.container import from_polylines  # noqa: E402


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert np.array_equal(a, b, equal_nan=True)


# ------------------------------------------------------------------------------------------ inputs
def _shapes(rng, n, k=12):
    """Tip-first float32 resamples with lengths: random walks, plus equal, zero and tiny lengths."""
    S = (rng.normal(0, 30, (n, 1, 2)) + np.cumsum(rng.normal(0, 3, (n, k, 2)), axis=1)).astype(np.float32)
    L = np.abs(rng.normal(60, 25, n)).astype(np.float32)
    L[::7] = L[0]                                          # exactly equal lengths
    L[3::11] = 0.0                                         # single-point / zero-length arms
    L[5::13] = np.float32(5e-7)                            # below the 1e-6 threshold
    L[6::17] = np.float32(1.5e-6)                          # just above it
    return S, L


def _sequence(seed=0, F=260, n_true=11, big_frame=None, nan_origin=True):
    """A drifting star: n_true arms that persist with dropouts, spurious arms, single-point arms."""
    rng = np.random.default_rng(seed)
    base = [256 + np.cumsum(rng.normal(0, 6, (int(rng.integers(2, 14)), 2)), axis=0) for _ in range(n_true)]
    emb0 = rng.normal(0, 1, (n_true, 8))
    frames, ident = [], []
    for f in range(F):
        arms, e = [], []
        n_here = int(rng.integers(0, n_true + 1)) if f % 9 else 0          # some empty frames
        for a in rng.permutation(n_true)[:n_here]:
            base[a] = base[a] + rng.normal(0, 1.5, base[a].shape)
            arms.append(base[a].copy())
            e.append(emb0[a] + rng.normal(0, 0.15, 8))
        for _ in range(int(rng.integers(0, 3))):                          # spurious arms
            arms.append(256 + np.cumsum(rng.normal(0, 8, (int(rng.integers(1, 6)), 2)), axis=0))
            e.append(rng.normal(0, 1, 8))
        if f % 23 == 5:                                                     # a zero-length arm
            arms.append(np.repeat(arms[0][:1], 3, axis=0) if arms else np.zeros((3, 2)) + 250)
            e.append(rng.normal(0, 1, 8))
        if big_frame is not None and f == big_frame:
            for _ in range(300):
                arms.append(256 + np.cumsum(rng.normal(0, 5, (6, 2)), axis=0))
                e.append(rng.normal(0, 1, 8))
        frames.append(arms)
        ident.extend(e)
    origin = 256 + rng.normal(0, 3, (F, 2))
    if nan_origin:
        origin[rng.random(F) < 0.05] = np.nan
    t = np.cumsum(rng.choice([1.0, 1.0, 1.0, 2.0, 5.0], F))
    return from_polylines(frames, time_s=t, crop=np.tile([0, 0, 512.0], (F, 1)), body_origin_model=origin,
                          identity=np.asarray(ident, np.float32).reshape(-1, 8), meta={'fps': 1.0})


@pytest.fixture(scope='module')
def seq():
    return _sequence()


def _reference_relink(c, config, frames=None, reverse=False, require_finite_origin=True, inputs=None, hook=None):
    """The pre-round-3 relink loop (reference FusedTracker)."""
    sampled, lengths, embeds = inputs if inputs is not None else TR.tracker_inputs(c, 'float64', config.k)
    all_pos = np.arange(c.n_frames) if frames is None else np.asarray(frames, np.int64)
    pos = TR._frames(c, all_pos, require_finite_origin)
    tracker = TR.FusedTracker(config, assign_hook=hook)
    out = np.full(c.n_arms, -1, np.int64)
    fo = c.frame_arm_offsets
    idx = c.frame_index
    last = int(idx[all_pos[-1]]) if len(all_pos) else 0
    for f in (pos[::-1] if reverse else pos):
        a0, a1 = fo[f], fo[f + 1]
        out[a0:a1] = tracker.step(last - int(idx[f]) if reverse else int(idx[f]), sampled[a0:a1], lengths[a0:a1],
                                  embeds[a0:a1])
    return out


# ------------------------------------------------------------------------------------------ T4.1 / T4.2
@pytest.mark.parametrize('k', [12, 5, 2])
def test_span_dist_block_equals_pairwise(k):
    rng = np.random.default_rng(k)
    S, L = _shapes(rng, 60, k)
    rows = A.arm_rows(S, L, k)
    n = 23

    def deg(i, j):
        return R.common_span_dist(S[i], float(L[i]), S[n + j], float(L[n + j]))

    got = A.span_dist_rows(rows[:n], rows[n:], k, deg)
    want = np.array([[R.common_span_dist(S[i], float(L[i]), S[n + j], float(L[n + j])) for j in range(60 - n)]
                     for i in range(n)])
    bits(got, want)
    assert A.span_dist_rows(rows[:0], rows, k).shape == (0, 60)


def test_embed_dist_equals_norm():
    rng = np.random.default_rng(1)
    for dt in (np.float32, np.float64):
        ea = rng.normal(0, 1, (17, 8)).astype(dt)
        eb = (ea[rng.integers(0, 17, 13)] + rng.normal(0, 0.05, (13, 8))).astype(dt)
        want = np.array([[float(np.linalg.norm(ea[i] - eb[j])) for j in range(13)] for i in range(17)])
        bits(A.embed_dist(ea, eb), want)


def test_cost_matrix_equals_pairwise_loop():
    rng = np.random.default_rng(2)
    S, L = _shapes(rng, 40)
    E = rng.normal(0, 1, (40, 8)).astype(np.float32)
    got = TR.cost_matrix(S[:15], L[:15], E[:15], list(S[15:]), [float(x) for x in L[15:]], list(E[15:]))
    shape = np.array([[R.common_span_dist(S[i], float(L[i]), S[15 + j], float(L[15 + j])) for j in range(25)]
                      for i in range(15)])
    emb = np.array([[float(np.linalg.norm(E[i] - E[15 + j])) for j in range(25)] for i in range(15)])
    bits(got['shape_distance'], shape)
    bits(got['embedding_distance'], emb)
    bits(got['cost'], np.where(shape <= 120.0, shape + 2.0 * emb, np.nan))
    empty = TR.cost_matrix(S[:3], L[:3], E[:3], np.zeros((0, 12, 2), np.float32), [], np.zeros((0, 8), np.float32))
    assert empty['cost'].shape == (3, 0)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_cost_matrix_nonfinite_lengths_and_points():
    """NaN / +-inf lengths and NaN points on either side: the reference's values (NaN where it is
    NaN), not an out-of-bounds gather; a NaN length is the shorter arm as in ``min(la, lb)``."""
    rng = np.random.default_rng(4)
    vals = [np.nan, np.inf, -np.inf, 0.0, 5e-7, 30.0, 60.0]
    S, _ = _shapes(rng, 2 * len(vals))
    L = np.array(vals + vals, np.float32)
    S[2] = np.nan
    S[len(vals) + 5, 3] = np.nan
    E = rng.normal(0, 1, (len(S), 8)).astype(np.float32)
    n = len(vals)
    got = TR.cost_matrix(S[:n], L[:n], E[:n], S[n:], L[n:], E[n:])['shape_distance']
    want = np.array([[R.common_span_dist(S[i], float(L[i]), S[n + j], float(L[n + j])) for j in range(n)]
                     for i in range(n)])
    bits(got, want)
    assert np.isnan(want).any() and np.isfinite(want).any()


def test_augmented_matrix():
    rng = np.random.default_rng(3)
    for n, m in ((1, 1), (3, 7), (9, 2)):
        cost = rng.uniform(0, 100, (n, m))
        cost[rng.random((n, m)) < 0.3] = A.BIG
        full = np.full((n + m, m + n), A.BIG)
        full[:n, :m] = cost
        full[:n, m:][np.diag_indices(n)] = 45.0
        full[n:, :m][np.diag_indices(m)] = 45.0
        full[n:, m:] = 0.0
        bits(A.augmented(cost, 90.0), full)
        assert A.augmented(cost, None) is cost


# ------------------------------------------------------------------------------------------ T4.4 tracker
@pytest.mark.parametrize('variant', list(TR.VARIANTS))
@pytest.mark.parametrize('reverse', [False, True])
def test_relink_equals_reference_tracker(seq, variant, reverse):
    cfg = TR.VARIANTS[variant]
    got_m, ref_m = [], []
    got = TR.relink(seq, cfg, reverse=reverse, assign_hook=lambda m, r, c: got_m.append((m.copy(), r, c)))
    ref = _reference_relink(seq, cfg, reverse=reverse, hook=lambda m, r, c: ref_m.append((m.copy(), r, c)))
    bits(got, ref)
    assert len(got_m) == len(ref_m) > 50
    for (a, r1, c1), (b, r2, c2) in zip(got_m, ref_m):
        bits(a, b)
        bits(r1, r2)
        bits(c1, c2)


def test_relink_large_frame_and_strided_frames():
    c = _sequence(seed=5, F=80, big_frame=40, nan_origin=False)
    for frames in (None, np.arange(0, 80, 3), np.arange(79, -1, -2)[::-1]):
        bits(TR.relink(c, frames=frames), _reference_relink(c, TR.RELEASED, frames=frames))


def test_relink_small_chunks_equal_one_chunk(seq):
    sampled, lengths, embeds = TR.tracker_inputs(seq)
    pos = TR._frames(seq, np.arange(seq.n_frames), True)
    outs = []
    for chunk in (1, 37, 1 << 15):
        out = np.full(seq.n_arms, -1, np.int64)
        A.run(TR.RELEASED, sampled, lengths, embeds, seq.frame_arm_offsets, pos, out, chunk_arms=chunk)
        outs.append(out)
    bits(outs[0], outs[2])
    bits(outs[1], outs[2])


def test_fallback_paths_equal_reference(seq):
    # velocity on: the reference tracker runs (prediction moves)
    moving = TR.TrackerConfig(vel_alpha=0.5)
    sampled, lengths, embeds = TR.tracker_inputs(seq)
    assert not A.fast_path_ok(moving, sampled, lengths, embeds, np.arange(seq.n_arms))
    bits(TR.relink(seq, moving), _reference_relink(seq, moving))
    # NaN-origin frames fed (diagnostics path): non-finite inputs -> the reference tracker, whose NaN
    # costs linear_sum_assignment rejects, as the figure scripts would
    bad = np.flatnonzero(~np.isfinite(seq.body_origin_model).all(1) & (seq.arms_per_frame() > 0))
    assert len(bad)
    assert not A.fast_path_ok(TR.RELEASED, sampled, lengths, embeds, TR._arms_of(seq.frame_arm_offsets, bad))
    for fn in (lambda: TR.relink(seq, require_finite_origin=False),
               lambda: _reference_relink(seq, TR.RELEASED, require_finite_origin=False)):
        with pytest.raises(ValueError, match='invalid numeric entries'), np.errstate(invalid='ignore'):
            fn()
    # a float64 identity is not the fast path's input; the reference handles it
    inputs64 = (sampled, lengths, embeds.astype(np.float64))
    assert not A.fast_path_ok(TR.RELEASED, *inputs64, np.arange(seq.n_arms))
    bits(TR.relink(seq, inputs=inputs64), _reference_relink(seq, TR.RELEASED, inputs=inputs64))
    with pytest.raises(ValueError):
        A.ArrayTracker(TR.TrackerConfig(max_match_cost=0.0), 12)


def test_step_costs_equal_reference(seq):
    pos = TR._frames(seq, None, True)
    for frame in pos[[5, 40, 120, -1]]:
        got = TR.step_costs(seq, int(frame))
        sampled, lengths, embeds = TR.tracker_inputs(seq)
        tracker = TR.FusedTracker()
        fo = seq.frame_arm_offsets
        for f in pos[pos < frame]:
            tracker.step(int(seq.frame_index[f]), sampled[fo[f]:fo[f + 1]], lengths[fo[f]:fo[f + 1]],
                         embeds[fo[f]:fo[f + 1]])
        a0, a1 = fo[frame], fo[frame + 1]
        want = R.step_cost_fig6_step(tracker, sampled[a0:a1], lengths[a0:a1], embeds[a0:a1],
                                     int(seq.frame_index[frame]), R.RELEASED)
        bits(got['track_ids'], np.array(want['track_ids'], np.int64))
        for key in ('shape_distance', 'embedding_distance', 'cost'):
            bits(got[key], want[key])
        bits(got['assignment'], np.asarray(want['assignment'], np.int64).reshape(-1, 2))
        labels = tracker.step(int(seq.frame_index[frame]), sampled[a0:a1], lengths[a0:a1], embeds[a0:a1])
        bits(got['labels'], np.asarray(labels, np.int64))
        flip = TR.step_costs(seq, int(frame), track_order='insertion')
        bits(flip['cost'], got['cost'])


# ------------------------------------------------------------------------------------------ T4.5
def _old_diagnostics(c, pos, states, fps, idx_all, variants, strides, origin_dtype):
    """The per-frame diagnostics of round 2 (frame_measures and links over label lists)."""
    from noodle_polylines import geometry as G
    fo = c.frame_arm_offsets
    angle, radius = G.tip_polar(c.points, c.arm_point_offsets, fo, c.body_origin_model, origin_dtype=origin_dtype)
    polar = [(angle[fo[f]:fo[f + 1]], radius[fo[f]:fo[f + 1]]) for f in pos]
    seq_frames = [{'index': int(idx_all[f]), 'arms': [None] * int(fo[f + 1] - fo[f]),
                   'state': None if states is None else states[f]} for f in pos]
    inputs = TR.tracker_inputs(c, origin_dtype)
    idx = np.array([fr['index'] for fr in seq_frames])

    def lab(ids, frames_):
        return [[int(x) for x in ids[fo[f]:fo[f + 1]] if x >= 0] for f in frames_]

    res = {}
    for variant, cfg in variants.items():
        fwd_ids = _reference_relink(c, cfg, frames=pos, inputs=inputs, require_finite_origin=False)
        rev_ids = _reference_relink(c, cfg, frames=pos, inputs=inputs, reverse=True, require_finite_origin=False)
        fwd, rev = lab(fwd_ids, pos), lab(rev_ids, pos)
        span = {}
        for k, labs in enumerate(fwd):
            for la in labs:
                span.setdefault(la, [idx[k], idx[k]])[1] = idx[k]
        life = {la: (b - a + 1) / fps for la, (a, b) in span.items()}
        by_state = {k: {kk: int(vv) for kk, vv in v.items()}
                    for k, v in R.frame_measures(seq_frames, fwd, rev, polar).items()}
        res[variant] = {'by_state': by_state, 'n_ids': len(span),
                        'arm_lifetime_s': np.array([life[la] for labs in fwd for la in labs], np.float32)}
        if variant == 'fused' and strides:
            st = {}
            for g in strides:
                sub_pos = pos[::g]
                sub = lab(_reference_relink(c, cfg, frames=sub_pos, inputs=inputs, require_finite_origin=False),
                          sub_pos)
                sub_idx = idx[::g]
                tp = n_sub = n_full = 0
                for m in range(len(sub) - 1):
                    if sub_idx[m + 1] - sub_idx[m] != g:
                        continue
                    ls, lf = R.links(sub[m], sub[m + 1]), R.links(fwd[m * g], fwd[(m + 1) * g])
                    tp += len(ls & lf); n_sub += len(ls); n_full += len(lf)
                st[g] = {'agree': tp, 'sub_links': n_sub, 'full_links': n_full}
            res['strides'] = st
    return res


@pytest.mark.parametrize('seed', [0, 7])
def test_diagnostics_equal_per_frame_counters(seed):
    c = _sequence(seed=seed, F=220)
    rng = np.random.default_rng(seed)
    ok = np.isfinite(c.body_origin_model).all(1)
    pos = np.flatnonzero(ok & (rng.random(c.n_frames) > 0.1))
    states = np.array(['a', 'b', 'c', None, 'b'], dtype=object)[rng.integers(0, 5, c.n_frames)]
    idx_all = TR.sequence_index(c, by='time')
    got = TR.diagnostics(c, frames=pos, states=states, fps=1.0, index=idx_all, origin_dtype='float64')
    want = _old_diagnostics(c, pos, states, 1.0, idx_all, TR.VARIANTS, TR.STRIDES, 'float64')
    for v in TR.VARIANTS:
        assert list(got[v]['by_state']) == list(want[v]['by_state'])        # key order too
        assert got[v]['by_state'] == want[v]['by_state']
        assert got[v]['n_ids'] == want[v]['n_ids']
        bits(got[v]['arm_lifetime_s'], want[v]['arm_lifetime_s'])
    assert got['strides'] == want['strides']
    assert got['fused']['by_state']['all']['links'] > 50 and got['fused']['by_state']['all']['jumps'] >= 0


def test_diagnostics_without_states_and_with_frame_index(seq):
    pos = np.flatnonzero(np.isfinite(seq.body_origin_model).all(1))
    got = TR.diagnostics(seq, frames=pos, fps=2.0, by='frame_index', variants={'fused': TR.RELEASED})
    want = _old_diagnostics(seq, pos, None, 2.0, seq.frame_index, {'fused': TR.RELEASED}, TR.STRIDES, 'float64')
    assert got['fused']['by_state'] == want['fused']['by_state'] and list(got['fused']['by_state']) == ['all']
    assert got['strides'] == want['strides']
    bits(got['fused']['arm_lifetime_s'], want['fused']['arm_lifetime_s'])


# ------------------------------------------------------------------------------------------ near-tie audit
def test_assignment_margin():
    # two arms, two tracks; swapping costs 0.5 more
    cost = np.array([[1.0, 2.0], [2.0, 1.5]])
    assert TR.assignment_margin(A.augmented(cost, 90.0)) == pytest.approx(1.5)   # swap: 4.0 - 2.5
    tie = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert TR.assignment_margin(A.augmented(tie, 90.0)) == 0.0
    # leaving an arm unmatched (45 + 45) against matching it at 89.9
    near = np.array([[89.9999995]])
    assert TR.assignment_margin(A.augmented(near, 90.0)) == pytest.approx(5e-7, abs=1e-9)
    gated = np.array([[A.BIG]])
    assert TR.assignment_margin(A.augmented(gated, 90.0)) > 1e8
    assert TR.assignment_margin(np.array([[1.0, 3.0]]), capped=False) == pytest.approx(2.0)


def test_relink_margins_on_sequence(seq):
    mats = []
    TR.relink(seq, assign_hook=lambda m, r, c: mats.append(m.copy()))
    margins = np.array([TR.assignment_margin(m) for m in mats[:60]])
    assert np.all(margins > 0)


def test_stored_ids_skip_the_tracker(seq, monkeypatch):
    from noodle_polylines.presets import compute_all
    ids = TR.relink(seq)
    stored = seq.with_fields(track_id=np.where(ids >= 0, ids, 0))

    def boom(*a, **k):
        raise AssertionError('tracker touched')

    monkeypatch.setattr(TR, 'tracker_inputs', boom)
    monkeypatch.setattr(TR, '_run', boom)
    out = compute_all(stored, 'fig5', r0=40.0)
    assert str(out['track_id_source']) == 'stored'
    bits(out['track_id'], stored.track_id)
