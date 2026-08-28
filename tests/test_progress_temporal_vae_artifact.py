# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from deepstarnoodles.paths import (
    DEFAULT_TEMPORAL_VAE_CHECKPOINT,
    REPO_ROOT,
)
from deepstarnoodles.representations import (
    load_progress_temporal_vae_checkpoint,
)


def _sha256() -> str:
    digest = hashlib.sha256()
    with DEFAULT_TEMPORAL_VAE_CHECKPOINT.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_optional_released_temporal_vae_checksum_and_smoke() -> None:
    if not DEFAULT_TEMPORAL_VAE_CHECKPOINT.is_file():
        pytest.skip("optional temporal VAE checkpoint is not installed")
    manifest = json.loads(
        (REPO_ROOT / "config" / "artifacts.json").read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["artifacts"]
        if item["id"] == "progress-temporal-vae-sleepy"
    )
    assert _sha256() == entry["sha256"]

    model = load_progress_temporal_vae_checkpoint()
    assert model.config.context_frames == 60
    assert model.config.resolution == 64
    assert model.config.latent_dim == 64
    assert (
        model.checkpoint_metadata["source_hydra_checkpoint_sha256"]
        == entry["source_hydra_checkpoint_sha256"]
    )
    with torch.no_grad():
        encoded = model.encode(
            torch.zeros(1, 60, 1, 64, 64),
            torch.ones(1, 60, dtype=torch.bool),
        )
    assert encoded["mean"].shape == (1, 64)
    assert torch.isfinite(encoded["mean"]).all()
