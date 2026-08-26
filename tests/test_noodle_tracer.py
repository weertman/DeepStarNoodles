# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from deepstarnoodles.noodle_tracer import (
    _INFERENCE_ATTENTION_BRANCH_ATTRIBUTE,
    _features_for_indices,
    _multiscale_attention_weights,
    _sample_feature_patches,
    RGBPyramid,
    NoodleTraceParams,
    NoodleTracer,
    TraceState,
    load_noodle_tracer_checkpoint,
    save_noodle_tracer_checkpoint,
    select_trace_seeds,
    trace_arms_noodle,
)
from deepstarnoodles.noodle_tracer_data import (
    NoodleTeacherTrajectories,
    build_noodle_teacher_trajectories,
    projection_lookahead_targets,
)
from train_noodle_tracer import (
    capture_rng_state,
    restore_rng_state,
    train_or_score_batch,
)


def straight_raw(size: int = 33) -> torch.Tensor:
    raw = torch.full((18, size, size), -10.0)
    y = size // 2
    raw[0, y - 2:y + 3, 4:size - 4] = 10.0
    raw[1, y, 4:size - 4] = 10.0
    raw[2] = 1.0
    raw[3] = 0.0
    raw[8, y, 5] = 10.0
    raw[9, y, size - 6] = 10.0
    raw[10:] = 0.1
    return raw


def small_model(**kwargs) -> NoodleTracer:
    return NoodleTracer(
        dense_channels=18,
        embedding_dim=8,
        hidden_size=24,
        pretrained_rgb=False,
        freeze_rgb=True,
        dense_mean=[0.0] * 18,
        dense_std=[1.0] * 18,
        **kwargs,
    )


def zero_trainable_parameters(model: NoodleTracer) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()


def test_projection_lookahead_recovers_from_displaced_seed():
    path = torch.tensor(
        [[[20.0, 10.0], [10.0, 10.0], [0.0, 10.0]]]
    )
    teachers = NoodleTeacherTrajectories(
        image_indices=torch.tensor([0]),
        seed_positions=torch.tensor([[20.0, 13.0]]),
        seed_embeddings=torch.zeros((1, 8)),
        paths=path,
        path_valid=torch.ones((1, 3), dtype=torch.bool),
        valid_seed=torch.tensor([True]),
        matched_arms=1,
        false_tips=0,
        ignored_tips=0,
    )
    targets = projection_lookahead_targets(
        torch.tensor([[20.0, 13.0]]),
        teachers,
        torch.tensor([0]),
        lookahead_px=8.0,
        max_step_px=6.0,
    )
    assert targets.state.item() == int(TraceState.CONTINUE)
    assert torch.allclose(targets.projection[0], torch.tensor([20.0, 10.0]))
    assert targets.displacement[0, 0] < 0
    assert targets.displacement[0, 1] < 0
    assert torch.linalg.vector_norm(targets.displacement[0]).item() <= 6.0 + 1e-5


def test_noodle_teacher_keeps_detected_seed_separate_from_true_path():
    raw = straight_raw()
    y = raw.shape[-1] // 2
    arm = np.asarray([[5.0, y], [15.0, y], [27.0, y]], dtype=np.float32)
    teachers = build_noodle_teacher_trajectories(
        raw[None],
        [[arm]],
        match_radius=5.0,
    )
    assert teachers.matched_arms == 1
    assert torch.allclose(teachers.paths[0, 0], torch.tensor([27.0, float(y)]))
    assert torch.allclose(
        teachers.seed_positions[0],
        torch.tensor([27.0, float(y)]),
    )


def test_flexible_action_head_predicts_non_unit_step_lengths():
    model = small_model().eval()
    zero_trainable_parameters(model)
    dense = torch.zeros((1, 18, 33, 33))
    fine = torch.zeros((1, 16, 9, 9))
    coarse = torch.zeros((1, 24, 9, 9))
    previous = torch.zeros((1, 2))
    seed = torch.zeros((1, 8))
    hidden = torch.zeros((1, model.hidden_size))
    with torch.no_grad():
        model.direction_head.bias.copy_(torch.tensor([1.0, 0.0]))
        model.step_head.bias.fill_(-10.0)
        short, _, _ = model.forward_step(
            dense, fine, coarse, previous, seed, hidden
        )
        model.step_head.bias.fill_(10.0)
        long, _, _ = model.forward_step(
            dense, fine, coarse, previous, seed, hidden
        )
    assert 0.49 < torch.linalg.vector_norm(short).item() < 0.6
    assert 5.9 < torch.linalg.vector_norm(long).item() <= 6.0


def test_invalid_seed_decision_prevents_trace_from_starting():
    raw = straight_raw()
    image = torch.zeros((3, raw.shape[-2], raw.shape[-1]))
    model = small_model().eval()
    zero_trainable_parameters(model)
    with torch.no_grad():
        model.state_head.bias[int(TraceState.INVALID_SEED)] = 10.0
    arms = trace_arms_noodle(
        raw,
        image,
        model,
        NoodleTraceParams(suppress_duplicates=False),
    )
    assert arms == []


def test_support_gated_peaks_zero_sparse_background(monkeypatch):
    raw = torch.zeros((18, 33, 33))
    raw[0, 16, 16] = 2.0
    raw[9, 16, 16] = 4.0
    captured = []

    def record_heat(heat, threshold, nms_radius):
        captured.append(np.asarray(heat).copy())
        return np.zeros((0, 3), dtype=np.float32)

    monkeypatch.setattr(
        "deepstarnoodles.noodle_tracer.find_peaks",
        record_heat,
    )
    arms = trace_arms_noodle(
        raw,
        torch.zeros((3, 33, 33)),
        small_model().eval(),
        NoodleTraceParams(
            support_gated_peaks=True,
            suppress_duplicates=False,
        ),
    )
    assert arms == []
    assert len(captured) == 1
    assert captured[0][16, 16] > 0.2
    assert np.count_nonzero(captured[0]) == 1


def test_spatially_distinct_dense_evidence_is_not_globally_pooled_away():
    torch.manual_seed(4)
    model = small_model().eval()
    left = torch.zeros((1, 18, 33, 33))
    right = torch.zeros((1, 18, 33, 33))
    left[:, 1, 10:23, 5:8] = 5.0
    right[:, 1, 10:23, 25:28] = 5.0
    fine = torch.zeros((1, 16, 9, 9))
    coarse = torch.zeros((1, 24, 9, 9))
    previous = torch.zeros((1, 2))
    seed = torch.zeros((1, 8))
    hidden = model.initial_state(1)
    with torch.no_grad():
        _, _, hidden_left = model.forward_step(
            left, fine, coarse, previous, seed, hidden
        )
        _, _, hidden_right = model.forward_step(
            right, fine, coarse, previous, seed, hidden
        )
    assert not torch.allclose(hidden_left, hidden_right)


@pytest.mark.parametrize(("patch_size", "spatial_size"), [(33, 9), (65, 17)])
def test_single_scale_patch_geometry_is_derived(patch_size: int, spatial_size: int):
    model = small_model(dense_patch_size=patch_size).eval()
    assert model.spatial_size == spatial_size
    assert model.rgb_feature_patch_size == spatial_size
    dense = torch.zeros((1, 18, patch_size, patch_size))
    fine = torch.zeros((1, 16, spatial_size, spatial_size))
    coarse = torch.zeros((1, 24, spatial_size, spatial_size))
    with torch.no_grad():
        displacement, states, hidden = model.forward_step(
            dense,
            fine,
            coarse,
            torch.zeros((1, 2)),
            torch.zeros((1, 8)),
        )
    assert displacement.shape == (1, 2)
    assert states.shape == (1, len(TraceState))
    assert hidden.shape == (1, model.hidden_size)


@pytest.mark.parametrize("patch_size", [31, 34, 63])
def test_invalid_dense_patch_geometry_is_rejected(patch_size: int):
    with pytest.raises(ValueError, match="P % 4 == 1"):
        small_model(dense_patch_size=patch_size)


def test_multiscale_sampling_uses_dilated_image_coordinates_and_edge_padding():
    model = small_model(
        context_mode="multiscale_dilated",
        context_dilation=2,
    ).eval()
    x_ramp = torch.arange(81, dtype=torch.float32)[None, None, None].expand(
        1, 18, 81, 81
    )
    centre = torch.tensor([[40.0, 40.0]])
    dense = model.sample_dense(x_ramp, centre, None)
    assert dense.shape == (1, 2, 18, 33, 33)
    assert dense[0, 0, 0, 16, 17] == 41
    assert dense[0, 1, 0, 16, 17] == 42

    edge = model.sample_dense(x_ramp, torch.tensor([[0.0, 0.0]]), None)
    assert edge[0, 0, 0, 16, 16] == 0
    assert edge[0, 1, 0, 0, 0] == 0
    assert torch.allclose(edge[0, 1, 0, 16, 17], torch.tensor(2.0))

    fine = torch.arange(21, dtype=torch.float32)[None, None, None].expand(
        1, 16, 21, 21
    )
    coarse = torch.arange(11, dtype=torch.float32)[None, None, None].expand(
        1, 24, 11, 11
    )
    pyramid = RGBPyramid(fine=fine, coarse=coarse, image_size=(81, 81))
    rgb_fine, rgb_coarse = model.sample_rgb(
        pyramid,
        centre,
        torch.zeros(1, dtype=torch.long),
    )
    assert rgb_fine.shape == (1, 2, 16, 9, 9)
    assert rgb_coarse.shape == (1, 2, 24, 9, 9)
    # One context token advances eight image pixels on both feature maps.
    assert torch.allclose(
        rgb_fine[0, 1, 0, 4, 5] - rgb_fine[0, 1, 0, 4, 4],
        torch.tensor(2.0),
    )
    assert torch.allclose(
        rgb_coarse[0, 1, 0, 4, 5] - rgb_coarse[0, 1, 0, 4, 4],
        torch.tensor(1.0),
    )
    edge_fine, edge_coarse = model.sample_rgb(
        pyramid,
        torch.tensor([[0.0, 0.0]]),
        torch.zeros(1, dtype=torch.long),
    )
    assert edge_fine[0, 1, 0, 0, 0] == 0
    assert edge_coarse[0, 1, 0, 0, 0] == 0


def test_multiscale_scale_embeddings_participate_in_attention():
    torch.manual_seed(9)
    model = small_model(context_mode="multiscale_dilated").eval()
    dense = torch.zeros((1, 2, 18, 33, 33))
    dense[:, 1, 1] = 3.0
    fine = torch.zeros((1, 2, 16, 9, 9))
    coarse = torch.zeros((1, 2, 24, 9, 9))
    _, _, hidden = model.forward_step(
        dense,
        fine,
        coarse,
        torch.zeros((1, 2)),
        torch.zeros((1, 8)),
    )
    hidden.sum().backward()
    assert model.scale_embeddings.grad is not None
    assert torch.count_nonzero(model.scale_embeddings.grad).item() > 0


def test_intact_multiscale_attention_is_numerically_legacy_exact():
    torch.manual_seed(12)
    scores = torch.randn((3, 2, 9, 9))
    expected = torch.softmax(scores.flatten(1) / np.sqrt(64), dim=1).reshape(
        3,
        2,
        1,
        9,
        9,
    )
    observed = _multiscale_attention_weights(scores, key_channels=64)
    assert torch.equal(observed, expected)


@pytest.mark.parametrize(("branch", "omitted"), [(0, 1), (1, 0)])
def test_multiscale_attention_mask_assigns_omitted_branch_zero_probability(
    branch: int,
    omitted: int,
):
    scores = torch.randn((2, 2, 9, 9))
    attention = _multiscale_attention_weights(
        scores,
        key_channels=64,
        inference_branch=branch,
    )
    assert torch.count_nonzero(attention[:, omitted]).item() == 0
    assert torch.allclose(
        attention[:, branch].sum(dim=(1, 2, 3)),
        torch.ones(2),
    )


@pytest.mark.parametrize(("branch", "changed_branch"), [(0, 1), (1, 0)])
def test_omitted_multiscale_inputs_have_no_forward_effect(
    branch: int,
    changed_branch: int,
):
    torch.manual_seed(21)
    model = small_model(context_mode="multiscale_dilated").eval()
    dense = torch.randn((1, 2, 18, 33, 33))
    fine = torch.randn((1, 2, 16, 9, 9))
    coarse = torch.randn((1, 2, 24, 9, 9))
    changed_dense = dense.clone()
    changed_fine = fine.clone()
    changed_coarse = coarse.clone()
    changed_dense[:, changed_branch].add_(100.0)
    changed_fine[:, changed_branch].mul_(-20.0)
    changed_coarse[:, changed_branch].add_(50.0)
    previous = torch.randn((1, 2))
    seed = torch.randn((1, 8))
    hidden = torch.randn((1, model.hidden_size))
    setattr(model, _INFERENCE_ATTENTION_BRANCH_ATTRIBUTE, branch)
    try:
        with torch.no_grad():
            expected = model.forward_step(
                dense,
                fine,
                coarse,
                previous,
                seed,
                hidden,
            )
            observed = model.forward_step(
                changed_dense,
                changed_fine,
                changed_coarse,
                previous,
                seed,
                hidden,
            )
    finally:
        delattr(model, _INFERENCE_ATTENTION_BRANCH_ATTRIBUTE)
    for expected_tensor, observed_tensor in zip(expected, observed):
        assert torch.equal(expected_tensor, observed_tensor)


@pytest.mark.parametrize(
    ("context_mode", "patch_size"),
    [("single", 65), ("multiscale_dilated", 33)],
)
def test_noodle_checkpoint_roundtrip(
    tmp_path: Path,
    context_mode: str,
    patch_size: int,
):
    model = small_model(
        context_mode=context_mode,
        dense_patch_size=patch_size,
    )
    params = NoodleTraceParams(
        context_mode=context_mode,
        dense_patch_size=patch_size,
    )
    path = tmp_path / f"noodle-{context_mode}.pt"
    save_noodle_tracer_checkpoint(
        str(path),
        model,
        trace_params=params,
        metadata={"epoch": 1},
        training_state={"token": 7},
    )
    loaded, loaded_params, metadata, training_state = (
        load_noodle_tracer_checkpoint(str(path))
    )
    assert loaded.checkpoint_config() == model.checkpoint_config()
    assert loaded_params == params
    assert metadata["epoch"] == 1
    assert training_state == {"token": 7}


def test_legacy_single_scale_checkpoint_load_is_numerically_exact(tmp_path: Path):
    torch.manual_seed(14)
    model = small_model().eval()
    dense = torch.randn((2, 18, 33, 33))
    fine = torch.randn((2, 16, 9, 9))
    coarse = torch.randn((2, 24, 9, 9))
    previous = torch.randn((2, 2))
    seed = torch.randn((2, 8))
    hidden = torch.randn((2, model.hidden_size))
    with torch.no_grad():
        expected = model.forward_step(
            dense, fine, coarse, previous, seed, hidden
        )

    model_config = model.checkpoint_config()
    model_config.pop("context_mode")
    model_config.pop("context_dilation")
    trace_config = {
        "dense_patch_size": 33,
        "rgb_feature_patch_size": 9,
    }
    path = tmp_path / "legacy-single.pt"
    torch.save(
        {
            "schema": 2,
            "model": model.state_dict(),
            "model_config": model_config,
            "trace_params": trace_config,
            "metadata": {},
        },
        path,
    )
    loaded, params, _, _ = load_noodle_tracer_checkpoint(str(path))
    with torch.no_grad():
        observed = loaded.forward_step(
            dense, fine, coarse, previous, seed, hidden
        )
    assert loaded.context_mode == "single"
    assert params.context_mode == "single"
    for expected_tensor, observed_tensor in zip(expected, observed):
        assert torch.equal(expected_tensor, observed_tensor)


def test_noodle_rng_state_restore_is_exact():
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    generator = torch.Generator().manual_seed(12)
    state = capture_rng_state(generator, None)
    expected = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )
    restore_rng_state(state, generator, None)
    observed = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )
    assert observed == expected


def test_noodle_training_batch_updates_action_and_state_heads():
    raw = straight_raw()
    y = raw.shape[-1] // 2
    arm = np.asarray([[5.0, y], [15.0, y], [27.0, y]], dtype=np.float32)
    batch = {
        "raw": raw[None],
        "image": torch.rand((1, 3, raw.shape[-2], raw.shape[-1])),
        "arms": [[arm]],
        "records": [{}],
    }
    model = small_model()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    args = Namespace(
        tip_match_radius=5.0,
        ambiguous_negative_radius=20.0,
        position_noise=0.0,
        lost_example_p=0.0,
        max_policy_steps=12,
        rollout_chunk_steps=4,
        lookahead_px=8.0,
        max_step_px=6.0,
        terminate_radius_px=4.0,
        lost_radius_px=16.0,
        action_weight=1.0,
        state_weight=1.0,
        support_weight=0.1,
        endpoint_weight=0.1,
        amp=False,
    )
    before_action = model.direction_head.weight.detach().clone()
    before_state = model.state_head.weight.detach().clone()
    result = train_or_score_batch(
        model,
        batch,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scaler=None,
        amp=False,
        args=args,
        rollout_p=0.3,
    )
    assert np.isfinite(result["loss"])
    assert result["matched_arms"] == 1
    assert not torch.equal(before_action, model.direction_head.weight)
    assert not torch.equal(before_state, model.state_head.weight)


def test_feature_index_expand_matches_clone_without_growing_storage():
    features = torch.arange(16 * 8 * 8, dtype=torch.float32).reshape(1, 16, 8, 8)
    indices = torch.zeros(64, dtype=torch.long)
    selected = _features_for_indices(features, indices)
    assert selected.shape == (64, 16, 8, 8)
    assert torch.equal(selected, features.expand(64, -1, -1, -1))
    assert selected.data_ptr() == features.data_ptr()
    positions = torch.tensor([[3.0, 4.0], [5.0, 2.0]])
    viewed = _sample_feature_patches(
        features,
        positions,
        torch.zeros(2, dtype=torch.long),
        image_size=(32, 32),
        patch_size=5,
    )
    cloned = _sample_feature_patches(
        features.expand(2, -1, -1, -1).contiguous(),
        positions,
        torch.arange(2, dtype=torch.long),
        image_size=(32, 32),
        patch_size=5,
    )
    assert torch.allclose(viewed, cloned)


def test_select_trace_seeds_keeps_separated_high_scores():
    peaks = np.array(
        [
            [10.0, 10.0, 0.9],
            [11.0, 10.0, 0.8],
            [40.0, 12.0, 0.7],
            [41.0, 13.0, 0.6],
            [70.0, 8.0, 0.5],
        ],
        dtype=np.float32,
    )
    selected = select_trace_seeds(peaks, max_tips=32, min_separation=16.0)
    assert selected.shape[0] == 3
    assert np.allclose(selected[:, 2], [0.9, 0.7, 0.5])
    assert select_trace_seeds(peaks[:2], max_tips=32, min_separation=16.0).shape[0] == 1
    unchanged = select_trace_seeds(peaks[:1], max_tips=32, min_separation=16.0)
    assert np.array_equal(unchanged, peaks[:1])


def test_max_tips_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="max_tips"):
        NoodleTraceParams(max_tips=0)
