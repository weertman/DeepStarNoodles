# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Batched CSR geometry kernels against the per-arm reference ports, bit for bit, on edge cases the
released data rarely hits (1- and 2-point arms, repeated points, angle wraps, arms over 128 points)."""
from __future__ import annotations

import pickle
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CORE'))

from noodle_polylines import _csr, _reference as R, geometry as G, tracking as TR  # noqa: E402
from noodle_polylines.container import _offsets, from_polylines  # noqa: E402


def _arms(seed: int = 7, n_arms: int = 600) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    arms = []
    for i in range(n_arms):
        n = int(rng.choice([1, 2, 3, 4, 8, 9, 16, 17, 40, 129, 130, 200, 300])) if i % 3 else int(rng.integers(1, 60))
        kind = i % 5
        if kind == 0:            # random walk
            p = 256 + np.cumsum(rng.normal(0, 3, (n, 2)), axis=0)
        elif kind == 1:          # pointing left: segment angles straddle +-pi (unwrap corrections)
            p = np.column_stack([300 - np.arange(n) * 2.0, 200 + rng.normal(0, 0.6, n)])
        elif kind == 2:          # repeated points and zero-length segments
            p = np.repeat(256 + rng.normal(0, 5, (max(n // 2, 1), 2)), 2, axis=0)[:n]
        elif kind == 3:          # all points identical (degenerate length)
            p = np.tile(rng.uniform(0, 512, 2), (n, 1))
        else:                    # tiny arms (length <= 1e-6 after float32 rounding)
            p = 100 + np.cumsum(rng.normal(0, 1e-8, (n, 2)), axis=0)
        arms.append(np.asarray(p, np.float32))
    return arms


@pytest.fixture(scope='module')
def csr():
    arms = _arms()
    pts = np.concatenate(arms)
    ao = _offsets(np.array([len(a) for a in arms], np.int64))
    return arms, pts, ao


def test_seg_sum_matches_numpy():
    rng = np.random.default_rng(1)
    counts = np.array(list(range(0, 300)) + [1000, 5000], np.int64)
    vals = rng.random(int(counts.sum())).astype(np.float32) * 100
    starts = _offsets(counts)[:-1]
    got = _csr.seg_sum(vals, starts, counts)
    ref = np.array([np.sum(vals[s:s + c]) for s, c in zip(starts, counts)], np.float32)
    assert np.array_equal(got, ref)


def test_seg_cumsum_matches_numpy():
    rng = np.random.default_rng(2)
    counts = rng.integers(0, 150, 400)
    vals = rng.normal(0, 10, int(counts.sum())).astype(np.float32)
    starts = _offsets(counts)[:-1]
    got = _csr.seg_cumsum(vals, starts, counts)
    ref = np.concatenate([np.cumsum(vals[s:s + c]) for s, c in zip(starts, counts)])
    assert np.array_equal(got, ref)


def test_seg_search_matches_searchsorted():
    rng = np.random.default_rng(3)
    arr = np.sort(rng.integers(0, 20, 50)).astype(float)
    x = rng.uniform(-1, 21, 200)
    for strict in (False, True):
        j = _csr.seg_search(arr, x, np.zeros_like(x, np.int64), np.full(len(x), len(arr)), strict)
        ref = np.searchsorted(arr, x, side='left' if strict else 'right') - 1
        assert np.array_equal(j, np.maximum(ref, 0))


def test_resample_base_bit_identical(csr):
    arms, pts, ao = csr
    ref = np.array([R.resample_arm(a) for a in arms])
    assert np.array_equal(G.resample_base(pts, ao), ref)


@pytest.mark.parametrize('origin', [None, 'f64', 'f32'])
def test_resample_tip_bit_identical(csr, origin):
    arms, pts, ao = csr
    rng = np.random.default_rng(4)
    o = None if origin is None else rng.uniform(0, 512, (len(arms), 2))
    if origin == 'f32':
        o = o.astype(np.float32)
    got, L = G.resample_tip(pts, ao, arm_origin=o)
    ref, ref_L = [], []
    for a, p in enumerate(arms):
        if o is not None:
            p = (np.asarray(p, np.float64) - o[a]) if o.dtype == np.float64 else (p - o[a])
        r, l_ = R.tip_anchored_resample(p, 12)
        ref.append(r)
        ref_L.append(l_)
    assert np.array_equal(got, np.array(ref))
    assert np.array_equal(L, np.array(ref_L))


def test_arc_length_curvature_bit_identical(csr):
    arms, pts, ao = csr
    for two_point, fn in (('zero', R.poly_metrics), ('nan', R.arm_metrics)):
        L, k = G.arc_length_curvature(pts, ao, two_point)
        ref = np.array([fn(a)[:2] for a in arms], np.float64)
        assert np.array_equal(L, ref[:, 0], equal_nan=True)
        assert np.array_equal(k, ref[:, 1], equal_nan=True)


def test_kernels_chunked(csr, monkeypatch):
    arms, pts, ao = csr
    full = (G.resample_base(pts, ao), G.resample_tip(pts, ao), G.arc_length_curvature(pts, ao, 'nan'))
    monkeypatch.setattr(G, 'CHUNK_ARMS', 7)
    monkeypatch.setattr(G._chunks, '__defaults__', (7,))
    chunked = (G.resample_base(pts, ao), G.resample_tip(pts, ao), G.arc_length_curvature(pts, ao, 'nan'))
    assert np.array_equal(full[0], chunked[0])
    for a, b in zip(full[1] + full[2], chunked[1] + chunked[2]):
        assert np.array_equal(a, b, equal_nan=True)


def test_empty_inputs():
    pts = np.zeros((0, 2), np.float32)
    ao = np.zeros(1, np.int64)
    assert G.resample_base(pts, ao).shape == (0, 12, 2)
    tip, L = G.resample_tip(pts, ao)
    assert tip.shape == (0, 12, 2) and L.shape == (0,)
    L, k = G.arc_length_curvature(pts, ao)
    assert L.shape == k.shape == (0,)


def test_to_source_and_body_relative_match_per_point(csr):
    arms, pts, ao = csr
    rng = np.random.default_rng(5)
    counts = np.diff(np.r_[0, np.sort(rng.choice(np.arange(1, len(arms)), 40, replace=False)), len(arms)])
    fo = _offsets(counts)
    F = len(counts)
    crop = np.column_stack([rng.uniform(0, 900, F), rng.uniform(0, 500, F), rng.uniform(100, 700, F)])
    origin = rng.uniform(100, 400, (F, 2))
    per_point = G.arm_frame(fo)[G.point_arm(ao)]
    ref = np.stack([R.model_to_source(pts[i], *crop[per_point[i]]) for i in range(0, len(pts), 97)])
    assert np.array_equal(G.to_source(pts, ao, fo, crop)[::97], ref)
    assert np.array_equal(G.body_relative(pts, ao, fo, origin), np.asarray(pts, np.float64) - origin[per_point])


def test_container_cache_shared_and_invalidated():
    rng = np.random.default_rng(6)
    frames = [[256 + np.cumsum(rng.normal(0, 3, (int(rng.integers(2, 30)), 2)), axis=0).astype(np.float32)
               for _ in range(5)] for _ in range(20)]
    c = from_polylines(frames, body_origin_model=np.full((20, 2), 256.0),
                       identity=rng.normal(0, 1, (100, 8)).astype(np.float32))
    b1 = G.base_shapes(c)
    assert b1 is G.base_shapes(c) and not b1.flags.writeable
    assert np.array_equal(b1, G.resample_base(c.points, c.arm_point_offsets))
    s1, l1, e1 = TR.tracker_inputs(c)
    s2, l2, e2 = TR.tracker_inputs(c)
    assert s1 is s2 and l1 is l2 and e1 is c.identity and c.identity.flags.writeable
    assert TR.tracker_inputs(c, origin_dtype='float32')[0] is not s1        # keyed by origin dtype
    moved = c.with_fields(body_origin_model=c.body_origin_model + 1.0)
    assert TR.tracker_inputs(moved)[0] is not s1
    assert G.base_shapes(moved) is not b1                                   # new container, new cache
    c.clear_cache()
    assert G.base_shapes(c) is not b1
    clone = pickle.loads(pickle.dumps(c))
    assert '_derived_cache' not in clone.__dict__
    assert np.array_equal(G.base_shapes(clone), b1)
