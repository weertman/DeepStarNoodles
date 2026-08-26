# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Tests for deepstarnoodles.decode_embed: identity-embedding-aware polyline tracing.

The decisive scenario is two arms that cross in an X. The averaged tangent and
progress fields are ambiguous at the crossing, but a per-pixel identity
embedding gives each arm a distinct signature, so the tracer must not swap the
two arms' bases/tips at the crossing.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from deepstarnoodles.dataset import rasterize_targets
from deepstarnoodles.decode_embed import (
    EmbedDecodeParams,
    _suppress_duplicates_and_fragments,
    activated_and_embedding,
    decode_arms_embed,
    decode_image_embed,
    embed_decode_params_with,
)


def _arm(poly, identity, tip_score=0.9, base_score=0.5):
    poly = np.asarray(poly, dtype=np.float32)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()
    return {
        'polyline': poly,
        'identity': np.asarray(identity, dtype=np.float32),
        'tip_score': float(tip_score),
        'base_score': float(base_score),
        'length_px': float(seg),
        'mean_step_cost': 0.5,
        'mean_identity_dist': 0.2,
        'island_frac': 1.0,
    }

# Two arms crossing at (130, 130). Bases on the left, tips on the right.
A_BASE, A_TIP = (60.0, 100.0), (200.0, 160.0)
B_BASE, B_TIP = (60.0, 160.0), (200.0, 100.0)
CROSS = (130.0, 130.0)


def crossing_shapes() -> list[dict]:
    return [
        {'label': 'arm', 'type': 'polyline',
         'points': [A_BASE[0], A_BASE[1], CROSS[0], CROSS[1], A_TIP[0], A_TIP[1]]},
        {'label': 'arm', 'type': 'polyline',
         'points': [B_BASE[0], B_BASE[1], CROSS[0], CROSS[1], B_TIP[0], B_TIP[1]]},
    ]


def synth_embedding(arm_instance: np.ndarray, dim: int = 4) -> np.ndarray:
    """Per-pixel embedding from the instance map: a distinct unit vector per arm.

    Overlap pixels (-1) get the mean of the two arm vectors (ambiguous, as a
    real model would produce at a crossing); background stays zero.
    """
    h, w = arm_instance.shape
    emb = np.zeros((dim, h, w), dtype=np.float32)
    u1 = np.zeros(dim, dtype=np.float32); u1[0] = 1.0
    u2 = np.zeros(dim, dtype=np.float32); u2[1] = 1.0
    emb[:, arm_instance == 1] = u1[:, None]
    emb[:, arm_instance == 2] = u2[:, None]
    emb[:, arm_instance == -1] = (0.5 * (u1 + u2))[:, None]
    return emb


def _nearest(pt, candidates):
    d = [math.hypot(pt[0] - c[0], pt[1] - c[1]) for c in candidates]
    k = int(np.argmin(d))
    return k, d[k]


def test_decode_embed_separates_crossing_arms():
    shapes = crossing_shapes()
    act, arm_instance = rasterize_targets((256, 256), shapes, return_arm_instance=True)
    act = act.numpy()
    arm_instance = arm_instance.numpy()
    emb = synth_embedding(arm_instance, dim=4)

    arms = decode_arms_embed(act, emb, EmbedDecodeParams())
    assert len(arms) == 2, f'expected 2 arms, got {len(arms)}'

    true_bases = [A_BASE, B_BASE]
    true_tips = [A_TIP, B_TIP]
    for a in arms:
        poly = a['polyline']
        base_pt, tip_pt = poly[0], poly[-1]
        ti, td = _nearest(tip_pt, true_tips)
        bi, bd = _nearest(base_pt, true_bases)
        assert td < 20.0, f'tip end {tip_pt} far from any true tip ({td:.1f}px)'
        assert bd < 20.0, f'base end {base_pt} far from any true base ({bd:.1f}px)'
        # No swap at the crossing: arm A's tip must pair with arm A's base.
        assert ti == bi, f'identity swapped: tip arm {ti} but base arm {bi}'
        # The walk stayed on its own identity through the crossing.
        assert a['mean_identity_dist'] < 0.5, f'identity drift {a["mean_identity_dist"]:.3f}'

    # Both true arms are recovered exactly once.
    paired = {_nearest(a['polyline'][-1], true_tips)[0] for a in arms}
    assert paired == {0, 1}


def test_decode_embed_no_support_returns_empty():
    act = np.zeros((10, 64, 64), dtype=np.float32)
    emb = np.zeros((4, 64, 64), dtype=np.float32)
    assert decode_arms_embed(act, emb, EmbedDecodeParams()) == []


def test_activated_and_embedding_split():
    pred = torch.randn(10 + 6, 16, 16) * 3.0
    act, emb = activated_and_embedding(pred, embedding_dim=6)
    assert act.shape == (10, 16, 16)
    assert emb.shape == (6, 16, 16)
    assert act[0].min() >= 0.0 and act[0].max() <= 1.0           # sigmoid support
    assert act[2].min() >= -1.0 and act[2].max() <= 1.0          # tanh dx
    # Embedding is the raw (linear) tail, so it can exceed [-1, 1].
    assert np.allclose(emb, pred[10:16].numpy())


def _straight_line(x0, x1, y, n=40):
    xs = np.linspace(x0, x1, n)
    return np.stack([xs, np.full(n, float(y))], axis=1)


def test_suppress_removes_perfect_duplicate():
    line = _straight_line(20, 120, 64)
    arms = [_arm(line, [1, 0, 0, 0]), _arm(line.copy(), [1, 0, 0, 0])]
    kept = _suppress_duplicates_and_fragments(arms, EmbedDecodeParams())
    assert len(kept) == 1


def test_suppress_keeps_overlapping_crossing_of_different_identity():
    # Two arms that physically cross (overlap spatially) but differ in identity.
    a = _straight_line(20, 120, 64)
    b = np.stack([np.full(40, 70.0), np.linspace(20, 120, 40)], axis=1)  # vertical through the same region
    arms = [_arm(a, [1, 0, 0, 0]), _arm(b, [0, 1, 0, 0])]
    kept = _suppress_duplicates_and_fragments(arms, EmbedDecodeParams())
    assert len(kept) == 2


def test_suppress_removes_split_tip_fragment():
    long_arm = _straight_line(20, 120, 64)            # length 100
    fragment = _straight_line(110, 130, 66, n=10)      # short, same identity, touching the tip
    arms = [_arm(long_arm, [1, 0, 0, 0]), _arm(fragment, [1, 0, 0, 0])]
    kept = _suppress_duplicates_and_fragments(arms, EmbedDecodeParams())
    assert len(kept) == 1
    assert kept[0]['length_px'] > 50  # the long arm survived


def test_suppress_keeps_far_apart_same_identity_arms():
    a = _straight_line(20, 120, 30)
    b = _straight_line(20, 120, 200)  # same identity but nowhere near each other
    arms = [_arm(a, [1, 0, 0, 0]), _arm(b, [1, 0, 0, 0])]
    kept = _suppress_duplicates_and_fragments(arms, EmbedDecodeParams())
    assert len(kept) == 2


def _put_gauss(hm, x, y, s=2.0):
    yy, xx = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
    np.maximum(hm, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)).astype(np.float32), out=hm)


def _h_arm(act, emb, x0, x1, y, idvec, half_w=2):
    for xx in range(x0, x1 + 1):
        for yy in range(y - half_w, y + half_w + 1):
            act[0, yy, xx] = 1.0
            act[2, yy, xx] = 1.0  # dx = +1 (base->tip points +x)
            act[4, yy, xx] = (xx - x0) / float(x1 - x0)
            emb[:, yy, xx] = idvec
        act[1, y, xx] = 1.0  # centerline on the middle row
    _put_gauss(act[9], x1, y)  # tip
    _put_gauss(act[8], x0, y)  # base


def test_decode_drops_tiny_detached_island():
    H, W, D = 120, 200, 4
    act = np.zeros((10, H, W), dtype=np.float32)
    emb = np.zeros((D, H, W), dtype=np.float32)
    _h_arm(act, emb, 10, 180, 60, [1, 0, 0, 0], half_w=4)   # large main arm
    _h_arm(act, emb, 20, 34, 15, [0, 1, 0, 0], half_w=0)    # tiny detached island (one row)

    with_island = decode_arms_embed(act, emb, embed_decode_params_with(min_island_frac=0.0))
    without_island = decode_arms_embed(act, emb, EmbedDecodeParams())
    assert len(with_island) == 2, f'expected island traced when filter off, got {len(with_island)}'
    assert len(without_island) == 1, f'island should be dropped by default, got {len(without_island)}'
    assert without_island[0]['length_px'] > 100  # the main arm is the survivor


def test_decode_image_embed_includes_landmarks():
    shapes = crossing_shapes() + [{'label': 'mouth', 'type': 'points', 'points': [95.0, 130.0]}]
    act, arm_instance = rasterize_targets((256, 256), shapes, return_arm_instance=True)
    act = act.numpy()
    emb = synth_embedding(arm_instance.numpy(), dim=4)
    result = decode_image_embed(act, emb, embed_decode_params_with(embed_weight=4.0))
    assert 'arms' in result and len(result['arms']) == 2
    assert len(result['mouth']) == 1
    assert math.hypot(result['mouth'][0, 0] - 95.0, result['mouth'][0, 1] - 130.0) < 3.0
