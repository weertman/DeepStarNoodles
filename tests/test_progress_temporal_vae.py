# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from deepstarnoodles.representations.progress import (
    ProgressCanonicalizationInput,
    ProgressRasterConfig,
    ProgressWindowConfig,
    ProgressWindowDataset,
    canonicalize_progress,
    estimate_characteristic_scale,
    progress_from_hydra,
)
from deepstarnoodles.representations.temporal_training import (
    ProgressTemporalTrainingConfig,
    fine_tune_progress_temporal_vae,
    masked_progress_vae_loss,
    sample_temporal_reconstruction_mask,
    train_progress_temporal_vae,
)
from deepstarnoodles.representations.temporal_vae import (
    ARCHITECTURE_ID,
    ProgressSequenceEncoder,
    ProgressTemporalVAE,
    ProgressTemporalVAEConfig,
    assert_strict_bottleneck,
    export_inference_checkpoint,
    load_progress_temporal_vae_checkpoint,
    save_progress_temporal_vae_checkpoint,
)


def _tiny_config() -> ProgressTemporalVAEConfig:
    return ProgressTemporalVAEConfig(
        context_frames=4,
        resolution=16,
        latent_dim=3,
        frame_token_dim=6,
        temporal_dim=8,
        base_channels=2,
        max_channels=4,
        transformer_layers=1,
        transformer_heads=2,
        dropout=0.0,
    )


def _training_config(*, steps: int = 2) -> ProgressTemporalTrainingConfig:
    return ProgressTemporalTrainingConfig(
        steps=steps,
        warmup_steps=0,
        validation_interval=1,
        validation_batches=1,
        checkpoint_interval=1,
        log_interval=100,
        mask_span_frames=2,
        mask_ratio=0.5,
        kl_target_nats=None,
        kl_warmup_steps=0,
        amp_dtype="float32",
        seed=7,
    )


def _batch(config: ProgressTemporalVAEConfig) -> dict[str, torch.Tensor]:
    progress = torch.rand(
        2,
        config.context_frames,
        1,
        config.resolution,
        config.resolution,
    )
    support = torch.zeros_like(progress)
    support[..., 3:13, 3:13] = 1.0
    progress *= support
    return {
        "progress": progress,
        "support": support,
        "frame_valid": torch.ones(
            2, config.context_frames, dtype=torch.bool
        ),
    }


def test_configs_have_exact_defaults_and_stable_recipe_digests() -> None:
    model = ProgressTemporalVAEConfig()
    assert (
        model.context_frames,
        model.resolution,
        model.latent_dim,
        model.frame_token_dim,
        model.temporal_dim,
        model.base_channels,
        model.max_channels,
        model.transformer_layers,
        model.transformer_heads,
        model.dropout,
    ) == (60, 64, 64, 96, 128, 24, 192, 4, 4, 0.1)
    assert ProgressRasterConfig().recipe_digest() == (
        ProgressRasterConfig().recipe_digest()
    )
    assert ProgressWindowConfig().recipe_digest() == (
        ProgressWindowConfig().recipe_digest()
    )
    assert model.recipe_digest() == ProgressTemporalVAEConfig().recipe_digest()
    with pytest.raises(ValueError):
        ProgressWindowConfig(cadence_seconds=0)
    with pytest.raises(ValueError):
        ProgressTemporalVAEConfig(temporal_dim=7, transformer_heads=2)


def test_progress_from_hydra_activation_contract() -> None:
    raw = torch.zeros(5, 3, 3)
    raw[0] = -2.0
    raw[0, 1, 1] = 2.0
    raw[4] = 0.0
    result = progress_from_hydra(raw)
    expected_support = torch.sigmoid(torch.tensor(2.0))
    assert result.support.shape == (1, 3, 3)
    assert result.hard_support.sum() == 1
    assert result.support[0, 1, 1] == pytest.approx(
        float(expected_support)
    )
    assert result.progress[0, 1, 1] == pytest.approx(
        0.5 * float(expected_support)
    )
    assert result.support[0, 0, 0] == 0
    assert bool(result.valid)
    with pytest.raises(ValueError):
        progress_from_hydra(torch.zeros(4, 3, 3))


def test_canonicalization_is_translation_and_scale_invariant() -> None:
    support = torch.zeros(2, 1, 32, 32)
    support[:, :, 8:17, 11:20] = 1.0
    progress = support * torch.linspace(0, 1, 32)[None, None, None, :]
    config = ProgressRasterConfig(
        resolution=16, half_extent_characteristic_scales=1.0
    )
    geometry = ProgressCanonicalizationInput.source_axes(
        [[0.0, 0.0, 32.0], [100.0, 250.0, 64.0]],
        model_size=32,
        characteristic_scale=[8.0, 16.0],
    )
    canonical = canonicalize_progress(
        progress, support, geometry, config=config
    )
    assert canonical.progress.shape == (2, 1, 16, 16)
    assert canonical.support.shape == (2, 1, 16, 16)
    assert canonical.valid.tolist() == [True, True]
    torch.testing.assert_close(
        canonical.progress[0], canonical.progress[1], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        canonical.support[0], canonical.support[1], atol=1e-5, rtol=1e-5
    )
    assert canonical.world_axis_name == "source_xy"
    assert not torch.equal(
        canonical.centroid_source[0], canonical.centroid_source[1]
    )


def test_characteristic_scale_estimator_tracks_crop_and_affine_scale() -> None:
    support = torch.zeros(1, 1, 32, 32)
    support[:, :, 8:24, 10:22] = 1.0
    base = estimate_characteristic_scale(
        support,
        crop_left_top_side=[0.0, 0.0, 32.0],
        model_size=32,
    )
    scaled = estimate_characteristic_scale(
        support,
        crop_left_top_side=[100.0, 200.0, 64.0],
        model_size=32,
        source_to_world_affine=[
            [3.0, 0.0, 4.0],
            [0.0, 3.0, -8.0],
            [0.0, 0.0, 1.0],
        ],
    )
    assert scaled == pytest.approx(6.0 * base)


def test_configurable_model_forward_and_strict_decoder() -> None:
    config = _tiny_config()
    model = ProgressTemporalVAE(config)
    batch = _batch(config)
    mask = torch.zeros(2, config.context_frames, dtype=torch.bool)
    mask[0, 1] = True
    mask[1, 2:] = True
    output = model(
        batch["progress"], batch["frame_valid"], mask, sample=False
    )
    assert output["mean"].shape == (2, config.latent_dim)
    assert output["progress_logits"].shape == (
        int(mask.sum()),
        1,
        config.resolution,
        config.resolution,
    )
    assert_strict_bottleneck(model)


def test_new_legacy_and_exported_checkpoints_round_trip(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    model = ProgressTemporalVAE(config)
    preprocess = ProgressRasterConfig(resolution=config.resolution)
    window = ProgressWindowConfig(minimum_valid_fraction=1.0)
    new_path = tmp_path / "new.pt"
    save_progress_temporal_vae_checkpoint(
        new_path,
        model,
        metadata={"purpose": "test"},
        preprocess_config=preprocess,
        window_config=window,
    )
    loaded = load_progress_temporal_vae_checkpoint(new_path, "cpu")
    assert loaded.config == config
    assert loaded.checkpoint_metadata == {"purpose": "test"}
    for expected, actual in zip(
        model.state_dict().values(), loaded.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected)

    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "architecture_id": ARCHITECTURE_ID,
            "model_config": asdict(config),
            "model": model.state_dict(),
            "step": 9,
        },
        legacy_path,
    )
    legacy = load_progress_temporal_vae_checkpoint(
        legacy_path, "cpu", return_info=True
    )
    assert legacy.model.config == config
    assert legacy.training_state is not None
    assert legacy.training_state["step"] == 9

    exported_path = tmp_path / "inference.pt"
    export_inference_checkpoint(
        legacy_path,
        exported_path,
        metadata={"exported": True},
        preprocess_config=preprocess,
        window_config=window,
    )
    exported = load_progress_temporal_vae_checkpoint(
        exported_path, "cpu", return_info=True
    )
    assert exported.checkpoint_type == "inference"
    assert exported.training_state is None
    assert exported.metadata["exported"] is True


def test_checkpoint_rejects_model_config_state_mismatch(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    model = ProgressTemporalVAE(config)
    incompatible = asdict(config)
    incompatible["latent_dim"] = config.latent_dim + 1
    path = tmp_path / "incompatible.pt"
    torch.save(
        {
            "architecture_id": ARCHITECTURE_ID,
            "model_config": incompatible,
            "model": model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_progress_temporal_vae_checkpoint(path)


def test_sequence_encoder_reuses_tokens_and_marks_edges_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config()
    model = ProgressTemporalVAE(config)
    observed_batch_sizes: list[int] = []
    original = model.encoder.frame_encoder.forward

    def counting_forward(values: torch.Tensor) -> torch.Tensor:
        observed_batch_sizes.append(len(values))
        return original(values)

    monkeypatch.setattr(model.encoder.frame_encoder, "forward", counting_forward)
    encoder = ProgressSequenceEncoder(
        model,
        window_config=ProgressWindowConfig(
            minimum_valid_fraction=1.0,
            nearest_tolerance=0.1,
        ),
        frame_batch_size=3,
        window_batch_size=2,
        amp=False,
    )
    progress = torch.rand(8, 1, config.resolution, config.resolution)
    encoded = encoder(
        progress,
        torch.arange(8, dtype=torch.float64),
        torch.ones(8, dtype=torch.bool),
    )
    assert encoded.offsets.tolist() == [-2.0, -1.0, 0.0, 1.0]
    assert sum(observed_batch_sizes) == 8
    assert encoded.valid.tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert torch.isnan(encoded.mean[~encoded.valid]).all()
    assert torch.isfinite(encoded.mean[encoded.valid]).all()
    assert len(encoded.recipe_digest) == 64

    trailing = ProgressSequenceEncoder(
        model,
        window_config=ProgressWindowConfig(
            centered=False,
            minimum_valid_fraction=1.0,
            nearest_tolerance=0.1,
        ),
        amp=False,
    )
    assert trailing.window_config.offsets(4).tolist() == [
        -3.0,
        -2.0,
        -1.0,
        0.0,
    ]


def test_window_dataset_and_mask_sampler() -> None:
    progress = torch.rand(7, 1, 16, 16)
    support = torch.ones_like(progress)
    frame_valid = torch.tensor(
        [True, True, False, True, True, True, True]
    )
    dataset = ProgressWindowDataset(
        progress,
        support,
        frame_valid,
        context_frames=4,
        config=ProgressWindowConfig(
            minimum_valid_fraction=0.75, output_stride=2
        ),
    )
    assert len(dataset) == 2
    assert [dataset[index]["start_position"].item() for index in range(2)] == [
        0,
        2,
    ]
    batch_valid = torch.stack(
        [dataset[index]["frame_valid"] for index in range(2)]
    )
    mask = sample_temporal_reconstruction_mask(
        batch_valid,
        ratio=0.5,
        span_frames=2,
        generator=torch.Generator().manual_seed(2),
    )
    assert not torch.any(mask & ~batch_valid)
    assert mask.any(dim=1).all()
    assert (batch_valid & ~mask).any(dim=1).all()


def test_masked_loss_is_finite_and_backpropagates() -> None:
    config = _tiny_config()
    model = ProgressTemporalVAE(config)
    batch = _batch(config)
    mask = sample_temporal_reconstruction_mask(
        batch["frame_valid"],
        ratio=0.5,
        span_frames=2,
        generator=torch.Generator().manual_seed(3),
    )
    predicted = model(
        batch["progress"], batch["frame_valid"], mask, sample=True
    )
    losses = masked_progress_vae_loss(
        predicted,
        batch["progress"],
        batch["support"],
        beta=1e-3,
        free_bits=0.02,
        background_weight=0.05,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert any(
        parameter.grad is not None
        and bool(torch.any(parameter.grad != 0))
        for parameter in model.parameters()
    )


def test_two_step_cpu_train_resume_and_fine_tune(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    batch = _batch(config)
    loader = [batch]
    train_directory = tmp_path / "train"
    result = train_progress_temporal_vae(
        ProgressTemporalVAE(config),
        loader,
        loader,
        train_directory,
        config=_training_config(),
        device="cpu",
    )
    assert result["steps"] == 2
    assert (train_directory / "best.pt").is_file()
    assert (train_directory / "latest.pt").is_file()

    resumed = train_progress_temporal_vae(
        ProgressTemporalVAE(config),
        loader,
        loader,
        train_directory,
        config=_training_config(),
        device="cpu",
        resume=train_directory / "latest.pt",
    )
    assert resumed["start_step"] == 2

    fine_tune_directory = tmp_path / "fine_tune"
    fine_tuned = fine_tune_progress_temporal_vae(
        train_directory / "best.pt",
        loader,
        loader,
        fine_tune_directory,
        config=_training_config(),
        device="cpu",
    )
    assert fine_tuned["steps"] == 2
    assert (fine_tune_directory / "latest.pt").is_file()
