# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Tests for deepstarnoodles.decode: peak extraction and polyline reconstruction."""
from __future__ import annotations

import math

import numpy as np
import torch

from deepstarnoodles.dataset import rasterize_targets
from deepstarnoodles.decode import (
    DecodeParams,
    activate_logits,
    decode_arms,
    decode_image,
    find_peaks,
)


def star_shapes(cx: float = 128.0, cy: float = 128.0, n_arms: int = 6, length: float = 80.0) -> list[dict]:
    shapes = []
    for k in range(n_arms):
        a = 2 * math.pi * k / n_arms
        bx, by = cx + 14 * math.cos(a), cy + 14 * math.sin(a)
        tx, ty = cx + length * math.cos(a), cy + length * math.sin(a)
        mx, my = cx + 0.5 * (14 + length) * math.cos(a + 0.06), cy + 0.5 * (14 + length) * math.sin(a + 0.06)
        shapes.append({'label': 'arm', 'type': 'polyline', 'points': [bx, by, mx, my, tx, ty]})
    shapes.append({'label': 'mouth', 'type': 'points', 'points': [cx, cy]})
    shapes.append({'label': 'butthole', 'type': 'points', 'points': [cx + 6, cy - 4]})
    return shapes


def test_find_peaks_recovers_gaussians():
    heat = np.zeros((64, 64), dtype=np.float32)
    for (x, y) in [(10.0, 12.0), (40.0, 45.0), (55.0, 20.0)]:
        yy, xx = np.mgrid[0:64, 0:64]
        heat = np.maximum(heat, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2)).astype(np.float32))
    peaks = find_peaks(heat, threshold=0.3, nms_radius=4)
    assert len(peaks) == 3
    found = {(round(float(px)), round(float(py))) for px, py, _ in peaks}
    assert found == {(10, 12), (40, 45), (55, 20)}


def test_find_peaks_empty():
    peaks = find_peaks(np.zeros((32, 32), dtype=np.float32), threshold=0.3, nms_radius=4)
    assert peaks.shape == (0, 3)


def test_decode_star_from_targets():
    shapes = star_shapes()
    act = rasterize_targets((256, 256), shapes).numpy()
    result = decode_image(act, DecodeParams())
    arms = result['arms']
    assert len(arms) == 6
    # Every decoded arm should start near a base ring (r=14) and end near a tip (r=80).
    for a in arms:
        base, tip = a['polyline'][0], a['polyline'][-1]
        r_base = math.hypot(base[0] - 128, base[1] - 128)
        r_tip = math.hypot(tip[0] - 128, tip[1] - 128)
        assert r_base < 25, f'base too far from disk: {r_base}'
        assert r_tip > 70, f'tip too short: {r_tip}'
    assert len(result['mouth']) == 1
    assert math.hypot(result['mouth'][0, 0] - 128, result['mouth'][0, 1] - 128) < 2
    assert len(result['butthole']) == 1


def test_decode_arms_no_support():
    act = np.zeros((10, 64, 64), dtype=np.float32)
    assert decode_arms(act, DecodeParams()) == []


def test_decode_arms_orientation_base_to_tip():
    shapes = star_shapes(n_arms=4)
    act = rasterize_targets((256, 256), shapes).numpy()
    arms = decode_arms(act, DecodeParams())
    for a in arms:
        # progress at the base end must be lower than at the tip end
        b, t = a['polyline'][0], a['polyline'][-1]
        pb = act[4, int(b[1]), int(b[0])]
        pt = act[4, int(t[1]), int(t[0])]
        assert pb < pt


def test_activate_logits_ranges():
    pred = torch.randn(10, 16, 16) * 5
    act = activate_logits(pred)
    assert float(act[0].min()) >= 0 and float(act[0].max()) <= 1
    assert float(act[2].min()) >= -1 and float(act[2].max()) <= 1
    assert float(act[3].min()) >= -1 and float(act[3].max()) <= 1
    assert float(act[5:].min()) >= 0 and float(act[5:].max()) <= 1
