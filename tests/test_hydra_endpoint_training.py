# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from deepstarnoodles.dataset import rasterize_targets
from deepstarnoodles.train import (
    freeze_all_but_endpoint_head,
    hydra_loss,
    modified_focal_heatmap_loss,
    optimizer_param_groups,
    verify_optimizer_trainable_parameters,
    weighted_heatmap_mse,
)


class TinyEndpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fpn = torch.nn.Conv2d(3, 3, kernel_size=1)
        self.support_centerline_head = torch.nn.Conv2d(3, 2, kernel_size=1)
        self.geometry_head = torch.nn.Conv2d(3, 3, kernel_size=1)
        self.landmark_head = torch.nn.Conv2d(3, 3, kernel_size=1)
        self.endpoint_head = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.fpn(image)
        return torch.cat(
            (
                self.support_centerline_head(features),
                self.geometry_head(features),
                self.landmark_head(features),
                self.endpoint_head(features),
            ),
            dim=1,
        )


def _endpoint_target() -> torch.Tensor:
    target = torch.zeros(1, 2, 9, 9)
    target[0, 0, 4, 4] = 1.0
    target[0, 0, 4, 3] = target[0, 0, 4, 5] = 0.6
    target[0, 1, 2, 6] = 1.0
    target[0, 1, 2, 5] = target[0, 1, 2, 7] = 0.6
    return target


def test_modified_focal_heatmap_loss_is_finite_at_extreme_logits() -> None:
    target = _endpoint_target()
    logits = torch.tensor([-1000.0, 1000.0]).view(1, 2, 1, 1).expand_as(target).clone()
    logits.requires_grad_(True)

    loss = modified_focal_heatmap_loss(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_modified_focal_prefers_good_endpoint_predictions() -> None:
    target = _endpoint_target()
    good = torch.full_like(target, -8.0)
    good[target == 1.0] = 8.0
    bad = -good

    assert modified_focal_heatmap_loss(good, target) < modified_focal_heatmap_loss(bad, target)


def test_modified_focal_ignore_mask_removes_ignored_false_positive() -> None:
    target = _endpoint_target()
    clean = torch.full_like(target, -8.0)
    clean[target == 1.0] = 8.0
    false_positive = clean.clone()
    false_positive[:, :, 7, 7] = 8.0
    keep = torch.ones(1, 1, 9, 9)
    keep[:, :, 7, 7] = 0.0

    ignored = modified_focal_heatmap_loss(false_positive, target, keep=keep)
    baseline = modified_focal_heatmap_loss(clean, target, keep=keep)

    assert ignored == pytest.approx(float(baseline))
    assert modified_focal_heatmap_loss(false_positive, target) > ignored


def _float_endpoint_arms() -> list[dict]:
    # Sub-pixel bases and tips, as in the real annotations.
    return [
        {'label': 'arm', 'type': 'polyline', 'points': [30.3, 31.6, 50.7, 12.2]},
        {'label': 'arm', 'type': 'polyline', 'points': [33.4, 34.5, 54.6, 52.4]},
        {'label': 'arm', 'type': 'polyline', 'points': [28.8, 33.1, 9.5, 47.7]},
    ]


def test_float_keypoint_targets_have_no_exact_peaks_by_default() -> None:
    target = rasterize_targets((64, 64), _float_endpoint_arms(), arm_tip_sigma=4.0)

    assert int(target[8:10].eq(1.0).sum()) == 0


def test_snapped_endpoint_targets_have_one_positive_per_keypoint() -> None:
    arms = _float_endpoint_arms()
    target = rasterize_targets((64, 64), arms, arm_tip_sigma=4.0, snap_endpoint_peaks=True)

    for channel, point_slice in ((8, slice(0, 2)), (9, slice(2, 4))):
        positives = target[channel].eq(1.0)
        assert int(positives.sum()) == len(arms)
        for arm in arms:
            x, y = arm['points'][point_slice]
            assert bool(positives[math.floor(y + 0.5), math.floor(x + 0.5)])


def test_modified_focal_on_snapped_targets_has_positive_term() -> None:
    target = rasterize_targets(
        (64, 64), _float_endpoint_arms(), arm_tip_sigma=4.0, snap_endpoint_peaks=True
    )[8:10].unsqueeze(0)
    # Full-confidence negatives contribute ~0, so the loss is the positive term.
    logits = torch.where(target.eq(1.0), torch.tensor(-4.0), torch.tensor(-30.0))

    loss = modified_focal_heatmap_loss(logits, target)
    positive_only = -F.logsigmoid(torch.tensor(-4.0)) * (1.0 - torch.sigmoid(torch.tensor(-4.0))) ** 2

    assert float(loss) > 1.0
    assert float(loss) == pytest.approx(float(positive_only), rel=1e-4)


def test_hydra_loss_default_endpoint_objective_remains_weighted_mse() -> None:
    pred = torch.randn(1, 10, 12, 12)
    target = torch.rand(1, 10, 12, 12)

    _, parts = hydra_loss(pred, target)
    expected = weighted_heatmap_mse(pred[:, 8:10], target[:, 8:10], peak_weight=20.0)

    assert parts['endpoint_loss'] == pytest.approx(float(expected))


def test_endpoint_only_freezes_all_other_parameters_and_optimizer_groups() -> None:
    model = TinyEndpointModel()
    freeze_all_but_endpoint_head(model)
    groups = optimizer_param_groups(model, lr=1e-4, backbone_lr=1e-5)

    verify_optimizer_trainable_parameters(model, groups)
    endpoint_ids = {id(param) for param in model.endpoint_head.parameters()}
    optimizer_ids = {id(param) for group in groups for param in group['params']}
    assert optimizer_ids == endpoint_ids
    assert all(
        param.requires_grad == name.startswith('endpoint_head.')
        for name, param in model.named_parameters()
    )


def test_endpoint_only_backward_isolates_gradients_to_endpoint_head() -> None:
    model = TinyEndpointModel()
    freeze_all_but_endpoint_head(model)
    target = torch.zeros(1, 10, 12, 12)
    target[:, 0, 3:9, 3:9] = 1.0
    target[:, 8, 4, 4] = 1.0
    target[:, 9, 8, 8] = 1.0

    loss, _ = hydra_loss(
        model(torch.randn(1, 3, 12, 12)),
        target,
        endpoint_heatmap_loss='modified_focal',
    )
    loss.backward()

    assert all(param.grad is not None for param in model.endpoint_head.parameters())
    assert all(
        param.grad is None
        for name, param in model.named_parameters()
        if not name.startswith('endpoint_head.')
    )
