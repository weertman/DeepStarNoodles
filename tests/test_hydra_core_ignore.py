# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Tests for StarSeg ignore-region masking (dataset rasterization + loss)."""
from __future__ import annotations

import torch

from deepstarnoodles.dataset import _rasterize_ignore, row_artifact_name
from deepstarnoodles.train import hydra_loss


def test_row_artifact_name_is_stable_across_split_positions():
    row = {'shapes_path': '/data/shapes/image_004321.json'}
    assert row_artifact_name(row, 0) == 'image_004321.json'
    assert row_artifact_name(row, 99) == 'image_004321.json'
    assert row_artifact_name({}, 7) == 'image_000007.json'


def test_rasterize_ignore_polygon():
    polys = [{'label': 'star_ignore', 'type': 'polygon',
              'points': [10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]}]
    mask = _rasterize_ignore((64, 64), polys)
    assert mask.shape == (64, 64)
    assert mask[20, 20] == 1.0 and mask[5, 5] == 0.0
    assert _rasterize_ignore((64, 64), []).sum() == 0


def test_ignore_mask_removes_false_negative_pressure():
    torch.manual_seed(0)
    b, h, w = 1, 64, 64
    target = torch.zeros(b, 10, h, w)
    target[:, 0, 8:24, 8:24] = 1.0  # annotated star support
    # Model confidently predicts support + an endpoint on an unannotated
    # neighbor star at the right side of the frame.
    pred = torch.full((b, 10, h, w), -6.0)
    pred[:, 0, 8:24, 8:24] = 6.0
    pred[:, 0, 40:56, 40:56] = 6.0
    pred[:, 9, 46:50, 46:50] = 6.0

    ignore = torch.zeros(b, 1, h, w)
    ignore[:, 0, 38:58, 38:58] = 1.0

    loss_unmasked, _ = hydra_loss(pred, target)
    loss_masked, _ = hydra_loss(pred, target, ignore_mask=ignore)
    assert loss_masked < loss_unmasked, (float(loss_masked), float(loss_unmasked))

    # With the neighbor fully ignored, loss should be close to the loss of a
    # prediction that never fired on the neighbor at all.
    pred_clean = pred.clone()
    pred_clean[:, 0, 40:56, 40:56] = -6.0
    pred_clean[:, 9, 46:50, 46:50] = -6.0
    loss_clean, _ = hydra_loss(pred_clean, target)
    assert abs(float(loss_masked) - float(loss_clean)) < 0.15 * float(loss_clean)


def test_ignore_mask_none_is_noop_equivalent():
    torch.manual_seed(1)
    pred = torch.randn(2, 10, 32, 32)
    target = torch.zeros(2, 10, 32, 32)
    target[:, 0, 8:20, 8:20] = 1.0
    l_none, _ = hydra_loss(pred, target)
    l_zero, _ = hydra_loss(pred, target, ignore_mask=torch.zeros(2, 1, 32, 32))
    assert torch.allclose(l_none, l_zero, rtol=1e-5)
