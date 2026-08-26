# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
# pyright: reportPrivateImportUsage=false, reportMissingImports=false
import numpy as np
import pytest
import torch

from deepstarnoodles.dataset import CHANNEL_NAMES, rasterize_targets
from deepstarnoodles.metrics import centerline_path_metrics
from deepstarnoodles.train import (
    SUPPORT_MASKED_CHANNELS,
    activated_pred,
    apply_lr_schedule,
    hydra_loss,
    hydra_metrics,
    initialize_optimizer_base_lrs,
    lr_schedule_scale,
    pred_support_masked_pred,
    should_tighten_support,
    support_boundary_ring,
    weighted_heatmap_mse,
)


class TinyJointModel(torch.nn.Module):
    """Cheap stand-in producing 10-channel logits without loading DINOv3."""

    def __init__(self):
        super().__init__()
        self.support_centerline_head = torch.nn.Conv2d(3, 2, kernel_size=1)
        self.geometry_head = torch.nn.Conv2d(3, 3, kernel_size=1)
        self.landmark_head = torch.nn.Conv2d(3, 3, kernel_size=1)
        self.endpoint_head = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x):
        return torch.cat(
            [self.support_centerline_head(x), self.geometry_head(x), self.landmark_head(x), self.endpoint_head(x)],
            dim=1,
        )


def _arm_shapes():
    return [
        {'label': 'arm', 'type': 'polyline', 'points': [8.0, 32.0, 56.0, 32.0]},
        {'label': 'mouth', 'type': 'points', 'points': [32.0, 16.0]},
    ]


def test_rasterize_targets_produces_canonical_10_channels():
    target = rasterize_targets((64, 64), _arm_shapes())
    assert target.shape == (10, 64, 64)
    assert len(CHANNEL_NAMES) == 10
    assert float(target[0].max()) == 1.0  # support
    assert float(target[1].max()) == pytest.approx(1.0, abs=1e-5)  # centerline peak
    assert float(target[5].max()) > 0.9  # mouth heatmap
    assert float(target[8].max()) > 0.9  # base endpoint
    assert float(target[9].max()) > 0.9  # tip endpoint
    assert float(target[2][target[0] > 0].mean()) > 0.9  # tangent x along horizontal arm


def test_compact_poly_centerline_is_clipped_to_support():
    target = rasterize_targets((64, 64), _arm_shapes())
    outside_support = target[0] <= 0.0
    assert float(target[1][outside_support].max()) == 0.0


def test_hydra_loss_covers_all_heads_with_gradients():
    model = TinyJointModel()
    x = torch.rand(2, 3, 32, 32)
    target = torch.zeros(2, 10, 32, 32)
    target[:, 0, 10:20, 10:20] = 1.0
    target[:, 1, 14:16, 10:20] = 1.0
    target[:, 2, 10:20, 10:20] = 1.0
    target[:, 4, 10:20, 10:20] = 0.5
    target[:, 5, 4:8, 4:8] = 1.0
    target[:, 8, 24:28, 24:28] = 1.0

    loss, parts = hydra_loss(model(x), target)
    assert torch.isfinite(loss)
    for key in ('support_loss', 'centerline_loss', 'geometry_loss', 'landmark_loss', 'endpoint_loss'):
        assert key in parts
    loss.backward()
    for head in (model.support_centerline_head, model.geometry_head, model.landmark_head, model.endpoint_head):
        grads = [p.grad for p in head.parameters()]
        assert all(g is not None for g in grads)
        assert any(float(g.abs().sum()) > 0 for g in grads if g is not None)


def test_hydra_metrics_reports_each_family():
    pred = torch.randn(1, 10, 32, 32)
    target = torch.zeros(1, 10, 32, 32)
    target[:, 0, 10:20, 10:20] = 1.0
    target[:, 1, 14:16, 10:20] = 1.0
    out = hydra_metrics(pred, target)
    for key in ('support_iou', 'centerline_path_f1_tol4', 'geometry_vector_l1', 'landmarks_mse', 'endpoints_mse'):
        assert key in out


def test_activated_pred_ranges():
    pred = torch.randn(1, 10, 8, 8) * 10
    act = activated_pred(pred)
    assert float(act[:, 0:2].min()) >= 0.0 and float(act[:, 0:2].max()) <= 1.0
    assert float(act[:, 2:4].min()) >= -1.0 and float(act[:, 2:4].max()) <= 1.0
    assert float(act[:, 4:10].min()) >= 0.0 and float(act[:, 4:10].max()) <= 1.0


def test_lr_warmup_scales_from_start_factor_to_base():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initialize_optimizer_base_lrs(optimizer)
    apply_lr_schedule(optimizer, next_step=1, warmup_steps=100, start_factor=0.1)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-3 * (0.1 + 0.9 * 0.01))
    apply_lr_schedule(optimizer, next_step=100, warmup_steps=100, start_factor=0.1)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-3)
    apply_lr_schedule(optimizer, next_step=1000, warmup_steps=100, start_factor=0.1)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-3)


def test_lr_schedule_cosine_decays_to_final_factor():
    # End of warmup: full base LR.
    assert lr_schedule_scale(100, warmup_steps=100, start_factor=0.1, total_steps=1000, final_factor=0.05) == pytest.approx(1.0)
    # Midpoint of decay: halfway between 1.0 and final factor.
    mid = lr_schedule_scale(550, warmup_steps=100, start_factor=0.1, total_steps=1000, final_factor=0.05)
    assert mid == pytest.approx(0.05 + 0.95 * 0.5)
    # Last step and beyond: final factor.
    assert lr_schedule_scale(1000, warmup_steps=100, start_factor=0.1, total_steps=1000, final_factor=0.05) == pytest.approx(0.05)
    assert lr_schedule_scale(2000, warmup_steps=100, start_factor=0.1, total_steps=1000, final_factor=0.05) == pytest.approx(0.05)
    # Monotonic decrease during decay.
    scales = [lr_schedule_scale(s, warmup_steps=100, start_factor=0.1, total_steps=1000, final_factor=0.05) for s in range(100, 1001, 100)]
    assert all(a >= b for a, b in zip(scales, scales[1:]))


def test_weighted_heatmap_mse_upweights_peak_errors():
    true = torch.zeros(1, 1, 16, 16)
    true[0, 0, 8, 8] = 1.0
    miss_peak = torch.full((1, 1, 16, 16), -10.0)  # predicts background everywhere
    flat_logits = torch.zeros(1, 1, 16, 16)  # sigmoid 0.5 everywhere

    plain_peak_miss = float(torch.nn.functional.mse_loss(torch.sigmoid(miss_peak), true))
    weighted_peak_miss = float(weighted_heatmap_mse(miss_peak, true, peak_weight=20.0))
    # Missing the peak should hurt much more under peak weighting.
    assert weighted_peak_miss > 10.0 * plain_peak_miss
    # A flat 0.5 prediction should not be rewarded relative to a confident background.
    assert float(weighted_heatmap_mse(flat_logits, true, peak_weight=20.0)) > weighted_peak_miss


def test_tiny_joint_overfit_smoke_reduces_loss():
    torch.manual_seed(0)
    model = TinyJointModel()
    x = torch.rand(2, 3, 32, 32)
    target = torch.zeros(2, 10, 32, 32)
    target[:, 0, 8:24, 8:24] = 1.0
    target[:, 1, 15:17, 8:24] = 1.0
    target[:, 2, 8:24, 8:24] = 1.0
    target[:, 4, 8:24, 8:24] = 0.5
    target[:, 6, 2:6, 2:6] = 1.0
    target[:, 9, 26:30, 26:30] = 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2)
    first_loss = None
    last_loss = None
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = hydra_loss(model(x), target)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        if first_loss is None:
            first_loss = last_loss
    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss * 0.5


def test_pred_support_masked_pred_uses_models_own_support():
    pred_a = torch.ones(1, 10, 8, 8)
    pred_a[:, 0] = 0.1  # predicted support below threshold everywhere...
    pred_a[:, 0, 2:6, 2:6] = 0.9  # ...except a confident region

    masked = pred_support_masked_pred(pred_a)

    assert SUPPORT_MASKED_CHANNELS == {1, 2, 3, 4}
    for ch in range(1, 10):
        inside = float(masked[0, ch, 2:6, 2:6].mean())
        outside = float(masked[0, ch, 0, 0])
        assert inside == pytest.approx(1.0)
        assert outside == (0.0 if ch in SUPPORT_MASKED_CHANNELS else 1.0)
    # Input is not mutated.
    assert float(pred_a[0, 1, 0, 0]) == 1.0


def test_support_boundary_ring_marks_only_near_tube_background():
    support = torch.zeros(1, 1, 16, 16)
    support[:, :, 6:10, 6:10] = 1.0
    ring = support_boundary_ring(support, radius_px=2)
    assert float(ring[0, 0, 8, 8]) == 0.0  # inside tube: not ring
    assert float(ring[0, 0, 5, 8]) == 1.0  # adjacent background: ring
    assert float(ring[0, 0, 0, 0]) == 0.0  # far background: not ring


def test_support_pos_weight_cap_and_boundary_weight_penalize_fat_prediction():
    target = torch.zeros(1, 10, 32, 32)
    target[:, 0, 14:18, 4:28] = 1.0  # thin tube
    fat_pred = torch.full((1, 10, 32, 32), -8.0)
    fat_pred[:, 0, 10:22, 4:28] = 8.0  # confident, 3x too wide

    loose, _ = hydra_loss(fat_pred, target, support_pos_weight_cap=100.0, support_boundary_weight=0.0)
    tight, _ = hydra_loss(fat_pred, target, support_pos_weight_cap=8.0, support_boundary_weight=3.0, support_boundary_radius=4)
    # Same prediction must cost relatively more under the tight config than a
    # perfect prediction does.
    perfect = torch.full((1, 10, 32, 32), -8.0)
    perfect[:, 0, 14:18, 4:28] = 8.0
    loose_perfect, _ = hydra_loss(perfect, target, support_pos_weight_cap=100.0, support_boundary_weight=0.0)
    tight_perfect, _ = hydra_loss(perfect, target, support_pos_weight_cap=8.0, support_boundary_weight=3.0, support_boundary_radius=4)
    loose_gap = float(loose - loose_perfect)
    tight_gap = float(tight - tight_perfect)
    assert tight_gap > loose_gap


def test_should_tighten_support_requires_min_epochs_and_patience():
    assert not should_tighten_support(epochs_seen=5, bad_epochs=10, min_epochs=10, patience=4)
    assert not should_tighten_support(epochs_seen=20, bad_epochs=3, min_epochs=10, patience=4)
    assert should_tighten_support(epochs_seen=10, bad_epochs=4, min_epochs=10, patience=4)


def test_path_metrics_perfect_match_is_one():
    arr = np.zeros((32, 32), dtype=np.float32)
    arr[16, 4:28] = 1.0
    out = centerline_path_metrics(arr, arr, tolerances=(2, 4))
    assert out['centerline_path_f1_tol2'] == pytest.approx(1.0)
    assert out['centerline_skel_f1'] == pytest.approx(1.0)
