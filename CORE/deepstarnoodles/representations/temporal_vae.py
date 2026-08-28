# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Strict-bottleneck temporal VAE and checkpointed sequence inference."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, overload
import uuid

import torch
from torch import nn
from torch.nn import functional as F

from deepstarnoodles.paths import DEFAULT_TEMPORAL_VAE_CHECKPOINT

from .progress import (
    ProgressCanonicalizationInput,
    ProgressRasterConfig,
    ProgressWindowConfig,
    canonical_recipe_digest,
    canonicalize_hydra_progress,
)


ARCHITECTURE_ID = "dnt_support_masked_progress_temporal_vae_v1"
LEGACY_ARCHITECTURE_ID = ARCHITECTURE_ID
TEMPORAL_LATENT_REPRESENTATION = "dnt_progress_temporal_vae_window_latent@1"
MODEL_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "deepstarnoodles.progress_temporal_vae"
CHECKPOINT_SCHEMA_VERSION = 1
LOGVAR_MIN = -20.0
LOGVAR_MAX = 10.0


def _strict_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _strict_finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class ProgressTemporalVAEConfig:
    """Checkpoint-bound dimensions for the progress temporal VAE."""

    context_frames: int = 60
    resolution: int = 64
    latent_dim: int = 64
    frame_token_dim: int = 96
    temporal_dim: int = 128
    base_channels: int = 24
    max_channels: int = 192
    transformer_layers: int = 4
    transformer_heads: int = 4
    dropout: float = 0.1
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "context_frames",
            "latent_dim",
            "frame_token_dim",
            "temporal_dim",
            "base_channels",
            "max_channels",
            "transformer_layers",
            "transformer_heads",
        ):
            _strict_positive_integer(name, getattr(self, name))
        _strict_positive_integer("resolution", self.resolution)
        if self.resolution < 16 or self.resolution > 256:
            raise ValueError("resolution must lie between 16 and 256")
        if self.resolution & (self.resolution - 1):
            raise ValueError("resolution must be a power of two")
        if self.max_channels < self.base_channels:
            raise ValueError("max_channels must be at least base_channels")
        if self.temporal_dim % self.transformer_heads:
            raise ValueError("temporal_dim must be divisible by transformer_heads")
        dropout = _strict_finite_number("dropout", self.dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != MODEL_SCHEMA_VERSION
        ):
            raise ValueError("unsupported progress temporal VAE schema version")

    @property
    def spatial_stages(self) -> int:
        """Number of factor-two stages needed to reach a 4x4 feature map."""

        return int(math.log2(self.resolution)) - 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_frames": int(self.context_frames),
            "resolution": int(self.resolution),
            "latent_dim": int(self.latent_dim),
            "frame_token_dim": int(self.frame_token_dim),
            "temporal_dim": int(self.temporal_dim),
            "base_channels": int(self.base_channels),
            "max_channels": int(self.max_channels),
            "transformer_layers": int(self.transformer_layers),
            "transformer_heads": int(self.transformer_heads),
            "dropout": float(self.dropout),
            "schema_version": int(self.schema_version),
        }

    def recipe_digest(self) -> str:
        return canonical_recipe_digest(
            {"recipe": ARCHITECTURE_ID, "config": self.as_dict()}
        )

    def digest(self) -> str:
        return self.recipe_digest()


DNTProgressTemporalVAEConfig = ProgressTemporalVAEConfig


def _group_count(channels: int) -> int:
    groups = min(16, channels)
    while channels % groups:
        groups -= 1
    return groups


class SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        return values + self.conv2(F.silu(self.norm2(hidden)))


class SpatialDownStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.residual = SpatialResidualBlock(output_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.residual(self.downsample(values))


class SpatialUpStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.residual = SpatialResidualBlock(output_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(values, scale_factor=2.0, mode="nearest")
        return self.residual(self.projection(values))


class ProgressFrameEncoder(nn.Module):
    """Encode one visible support-masked progress image into a frame token."""

    def __init__(self, config: ProgressTemporalVAEConfig) -> None:
        super().__init__()
        widths = [
            min(config.base_channels * (2**index), config.max_channels)
            for index in range(config.spatial_stages)
        ]
        stages: list[nn.Module] = []
        previous = 1
        for width in widths:
            stages.append(SpatialDownStage(previous, width))
            previous = width
        self.stages = nn.Sequential(*stages)
        self.output_norm = nn.GroupNorm(_group_count(widths[-1]), widths[-1])
        self.output = nn.Linear(widths[-1], config.frame_token_dim)

    def forward(self, progress: torch.Tensor) -> torch.Tensor:
        hidden = self.stages(progress)
        pooled = F.silu(self.output_norm(hidden)).mean(dim=(-2, -1))
        return self.output(pooled)


class MaskedAttentionPool(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        logits = self.score(tokens).squeeze(-1).masked_fill(
            ~valid, float("-inf")
        )
        weights = logits.softmax(dim=1)
        return torch.sum(tokens * weights[..., None], dim=1)


class ProgressWindowEncoder(nn.Module):
    """Encode visible frames and summarize exactly one context window."""

    def __init__(self, config: ProgressTemporalVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_encoder = ProgressFrameEncoder(config)
        self.frame_projection = nn.Sequential(
            nn.Linear(config.frame_token_dim, config.temporal_dim),
            nn.GELU(),
            nn.LayerNorm(config.temporal_dim),
        )
        self.mask_token = nn.Parameter(
            torch.empty(1, 1, config.temporal_dim)
        )
        self.temporal_position = nn.Parameter(
            torch.empty(1, config.context_frames, config.temporal_dim)
        )
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.temporal_position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.temporal_dim,
            nhead=config.transformer_heads,
            dim_feedforward=4 * config.temporal_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.temporal_dim),
            enable_nested_tensor=False,
        )
        self.clip_pool = MaskedAttentionPool(config.temporal_dim)
        self.mean_head = nn.Linear(config.temporal_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.temporal_dim, config.latent_dim)

    def forward(
        self,
        progress: torch.Tensor,
        frame_valid: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, frames = frame_valid.shape
        frame_visible = frame_valid & ~reconstruction_mask
        visible_flat = torch.nonzero(
            frame_visible.reshape(-1), as_tuple=False
        ).squeeze(1)
        if visible_flat.numel():
            visible_progress = progress.reshape(
                batch * frames,
                1,
                self.config.resolution,
                self.config.resolution,
            ).index_select(0, visible_flat)
            visible_token = self.frame_encoder(visible_progress)
            spatial_flat = visible_token.new_zeros(
                batch * frames, self.config.frame_token_dim
            ).index_copy(0, visible_flat, visible_token)
        else:
            spatial_flat = progress.new_zeros(
                batch * frames, self.config.frame_token_dim
            )
        return self.forward_tokens(
            spatial_flat.reshape(batch, frames, -1),
            frame_valid,
            reconstruction_mask,
        )

    def forward_tokens(
        self,
        spatial_token: torch.Tensor,
        frame_valid: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the exact temporal posterior from precomputed frame tokens."""

        batch, frames = frame_valid.shape
        expected = (
            batch,
            self.config.context_frames,
            self.config.frame_token_dim,
        )
        if tuple(spatial_token.shape) != expected:
            raise ValueError(f"spatial_token must have shape {expected}")
        expected_mask = (batch, self.config.context_frames)
        if tuple(frame_valid.shape) != expected_mask:
            raise ValueError(f"frame_valid must have shape {expected_mask}")
        if tuple(reconstruction_mask.shape) != expected_mask:
            raise ValueError("reconstruction_mask must match frame_valid")
        if frame_valid.dtype != torch.bool or reconstruction_mask.dtype != torch.bool:
            raise TypeError("frame_valid and reconstruction_mask must be boolean")
        if (
            spatial_token.device != frame_valid.device
            or spatial_token.device != reconstruction_mask.device
        ):
            raise ValueError("spatial tokens and masks must share a device")

        frame_visible = frame_valid & ~reconstruction_mask
        projected = self.frame_projection(spatial_token)
        masked_valid = frame_valid & reconstruction_mask
        content = torch.where(
            masked_valid[..., None],
            self.mask_token.expand(batch, frames, -1),
            projected,
        )
        frame_token = (content + self.temporal_position).masked_fill(
            ~frame_valid[..., None], 0.0
        )
        frame_latent = self.temporal_encoder(
            frame_token,
            src_key_padding_mask=~frame_valid,
        ).masked_fill(~frame_valid[..., None], 0.0)
        clip_latent = self.clip_pool(frame_latent, frame_valid)
        mean = self.mean_head(clip_latent)
        logvar = self.logvar_head(clip_latent).clamp(
            LOGVAR_MIN, LOGVAR_MAX
        )
        return {
            "spatial_token": spatial_token,
            "frame_token": frame_token,
            "frame_latent": frame_latent,
            "frame_valid": frame_valid,
            "frame_visible": frame_visible,
            "clip_latent": clip_latent,
            "mean": mean,
            "logvar": logvar,
        }


class StrictProgressDecoder(nn.Module):
    """Decode selected positions using only z and learned temporal queries."""

    def __init__(self, config: ProgressTemporalVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal_query_position = nn.Parameter(
            torch.empty(config.context_frames, config.temporal_dim)
        )
        nn.init.normal_(self.temporal_query_position, std=0.02)
        self.latent_projection = nn.Linear(
            config.latent_dim, config.temporal_dim
        )
        self.query_network = nn.Sequential(
            nn.LayerNorm(config.temporal_dim),
            nn.Linear(config.temporal_dim, 4 * config.temporal_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.temporal_dim, config.temporal_dim),
        )
        self.query_norm = nn.LayerNorm(config.temporal_dim)

        seed_channels = min(
            config.max_channels,
            config.base_channels * (2 ** (config.spatial_stages - 1)),
        )
        self.seed_channels = seed_channels
        self.seed = nn.Linear(
            config.temporal_dim, seed_channels * 4 * 4
        )
        stages: list[nn.Module] = []
        width = seed_channels
        for _ in range(config.spatial_stages):
            output_width = max(config.base_channels, width // 2)
            stages.append(SpatialUpStage(width, output_width))
            width = output_width
        self.spatial_decoder = nn.Sequential(*stages)
        self.output_norm = nn.GroupNorm(_group_count(width), width)
        self.progress_head = nn.Conv2d(width, 1, 3, padding=1)

    def forward(
        self,
        latent: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"latent must have shape [batch, {self.config.latent_dim}]"
            )
        expected_mask = (latent.shape[0], self.config.context_frames)
        if tuple(reconstruction_mask.shape) != expected_mask:
            raise ValueError(
                f"reconstruction_mask must have shape {expected_mask}"
            )
        if reconstruction_mask.dtype != torch.bool:
            raise TypeError("reconstruction_mask must be boolean")
        if reconstruction_mask.device != latent.device:
            raise ValueError("latent and reconstruction_mask must share a device")
        if not reconstruction_mask.any():
            raise ValueError("reconstruction_mask must select at least one frame")

        batch_index, frame_index = torch.where(reconstruction_mask)
        query = self.latent_projection(latent.index_select(0, batch_index))
        query = query + self.temporal_query_position.index_select(
            0, frame_index
        )
        query = self.query_norm(query + self.query_network(query))
        hidden = self.seed(query).reshape(
            query.shape[0], self.seed_channels, 4, 4
        )
        hidden = self.spatial_decoder(hidden)
        return self.progress_head(F.silu(self.output_norm(hidden)))


class ProgressTemporalVAE(nn.Module):
    """Masked temporal VAE with one Gaussian bottleneck for the whole window."""

    architecture_id = ARCHITECTURE_ID

    def __init__(
        self,
        config: ProgressTemporalVAEConfig = ProgressTemporalVAEConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = ProgressWindowEncoder(config)
        self.decoder = StrictProgressDecoder(config)
        self.checkpoint_metadata: dict[str, Any] = {}
        self.checkpoint_sha256: str | None = None
        self.checkpoint_preprocess_config: ProgressRasterConfig | None = None
        self.checkpoint_window_config: ProgressWindowConfig | None = None

    def _validate_window(
        self,
        progress: torch.Tensor,
        frame_valid: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        *,
        require_reconstruction: bool,
    ) -> None:
        batch = progress.shape[0] if progress.ndim else -1
        expected_progress = (
            batch,
            self.config.context_frames,
            1,
            self.config.resolution,
            self.config.resolution,
        )
        expected_mask = (batch, self.config.context_frames)
        if tuple(progress.shape) != expected_progress:
            raise ValueError(f"progress must have shape {expected_progress}")
        if not progress.is_floating_point():
            raise TypeError("progress must be a floating-point tensor")
        if tuple(frame_valid.shape) != expected_mask:
            raise ValueError(f"frame_valid must have shape {expected_mask}")
        if tuple(reconstruction_mask.shape) != expected_mask:
            raise ValueError(
                f"reconstruction_mask must have shape {expected_mask}"
            )
        if frame_valid.dtype != torch.bool:
            raise TypeError("frame_valid must be boolean")
        if reconstruction_mask.dtype != torch.bool:
            raise TypeError("reconstruction_mask must be boolean")
        if (
            frame_valid.device != progress.device
            or reconstruction_mask.device != progress.device
        ):
            raise ValueError("progress and masks must share a device")
        if batch < 1:
            raise ValueError("progress batch must not be empty")
        if not frame_valid.any(dim=1).all():
            raise ValueError("every window must contain at least one valid frame")
        if torch.any(reconstruction_mask & ~frame_valid):
            raise ValueError("cannot reconstruct invalid frames")
        if require_reconstruction and not reconstruction_mask.any():
            raise ValueError("reconstruction_mask must select at least one frame")

    def encode(
        self,
        progress: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode a fully visible window for downstream embeddings."""

        reconstruction_mask = torch.zeros_like(frame_valid, dtype=torch.bool)
        self._validate_window(
            progress,
            frame_valid,
            reconstruction_mask,
            require_reconstruction=False,
        )
        return self.encoder(progress, frame_valid, reconstruction_mask)

    def encode_tokens(
        self,
        spatial_token: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode windows from cached frame tokens without changing semantics."""

        expected = (
            spatial_token.shape[0],
            self.config.context_frames,
            self.config.frame_token_dim,
        )
        if tuple(spatial_token.shape) != expected:
            raise ValueError(f"spatial_token must have shape {expected}")
        if tuple(frame_valid.shape) != expected[:2]:
            raise ValueError("spatial tokens and frame validity differ")
        if frame_valid.dtype != torch.bool:
            raise TypeError("frame_valid must be boolean")
        if spatial_token.device != frame_valid.device:
            raise ValueError("spatial tokens and frame validity must share a device")
        if not frame_valid.any(dim=1).all():
            raise ValueError("every window must contain at least one valid frame")
        return self.encoder.forward_tokens(
            spatial_token,
            frame_valid,
            torch.zeros_like(frame_valid),
        )

    @staticmethod
    def reparameterize(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        *,
        sample: bool,
    ) -> torch.Tensor:
        if mean.shape != logvar.shape:
            raise ValueError("mean and logvar shapes differ")
        if not sample:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def decode(
        self,
        latent: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the latent-only decoder at selected temporal positions."""

        return self.decoder(latent, reconstruction_mask)

    def forward(
        self,
        progress: torch.Tensor,
        frame_valid: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        *,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        self._validate_window(
            progress,
            frame_valid,
            reconstruction_mask,
            require_reconstruction=True,
        )
        encoding = self.encoder(progress, frame_valid, reconstruction_mask)
        latent = self.reparameterize(
            encoding["mean"], encoding["logvar"], sample=sample
        )
        masked_batch_index, masked_frame_index = torch.where(
            reconstruction_mask
        )
        return {
            **encoding,
            "latent": latent,
            "reconstruction_mask": reconstruction_mask,
            "masked_batch_index": masked_batch_index,
            "masked_frame_index": masked_frame_index,
            "progress_logits": self.decode(latent, reconstruction_mask),
        }


DNTProgressTemporalVAE = ProgressTemporalVAE


def assert_strict_bottleneck(model: ProgressTemporalVAE) -> None:
    """Exercise reconstruction without providing any encoder activation."""

    parameter = next(model.parameters())
    latent = torch.zeros(
        2,
        model.config.latent_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    mask = torch.zeros(
        2,
        model.config.context_frames,
        dtype=torch.bool,
        device=parameter.device,
    )
    mask[0, 0] = True
    mask[1, -1] = True
    decoded = model.decode(latent, mask)
    expected = (
        2,
        1,
        model.config.resolution,
        model.config.resolution,
    )
    if tuple(decoded.shape) != expected:
        raise AssertionError(
            f"latent-only decoder returned {tuple(decoded.shape)}"
        )


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def architecture_metadata(
    model: ProgressTemporalVAE,
) -> Mapping[str, Any]:
    return {
        "architecture_id": model.architecture_id,
        "schema_version": MODEL_SCHEMA_VERSION,
        "config": model.config.as_dict(),
        "config_recipe_digest": model.config.recipe_digest(),
        "parameters": parameter_count(model),
        "input_representation": "support_masked_progress_raster",
        "input_channels": ["progress"],
        "context_frames": model.config.context_frames,
        "bottleneck": {
            "family": "diagonal_gaussian",
            "dimensions": model.config.latent_dim,
            "scope": "strict_window",
            "cache_value": "posterior_mean",
            "decoder_inputs": [
                "latent",
                "learned_temporal_query_position",
            ],
            "encoder_state_bypass": False,
            "skip_connections": False,
            "logvar_clamp": [LOGVAR_MIN, LOGVAR_MAX],
        },
        "decoder": {
            "selected_positions_only": True,
            "output": "progress_logits",
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _json_mapping(
    name: str, value: Mapping[str, Any] | None
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        return json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-safe values") from exc


def _raster_config(
    value: ProgressRasterConfig | Mapping[str, Any] | None,
) -> ProgressRasterConfig | None:
    if value is None or isinstance(value, ProgressRasterConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("preprocess_config must be a ProgressRasterConfig")
    return ProgressRasterConfig(**dict(value))


def _window_config(
    value: ProgressWindowConfig | Mapping[str, Any] | None,
) -> ProgressWindowConfig | None:
    if value is None or isinstance(value, ProgressWindowConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("window_config must be a ProgressWindowConfig")
    return ProgressWindowConfig(**dict(value))


@dataclass(frozen=True)
class LoadedProgressTemporalVAECheckpoint:
    model: ProgressTemporalVAE
    checkpoint_type: str
    schema_version: int
    metadata: Mapping[str, Any]
    preprocess_config: ProgressRasterConfig | None
    window_config: ProgressWindowConfig | None
    training_config: Mapping[str, Any] | None
    training_state: Mapping[str, Any] | None
    checkpoint_sha256: str
    path: Path


def save_progress_temporal_vae_checkpoint(
    path: str | Path,
    model: ProgressTemporalVAE,
    *,
    checkpoint_type: Literal["inference", "training"] | None = None,
    metadata: Mapping[str, Any] | None = None,
    preprocess_config: ProgressRasterConfig | Mapping[str, Any] | None = None,
    window_config: ProgressWindowConfig | Mapping[str, Any] | None = None,
    training_config: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save a schema-versioned inference or training checkpoint."""

    if not isinstance(model, ProgressTemporalVAE):
        raise TypeError("model must be a ProgressTemporalVAE")
    kind = checkpoint_type or (
        "training" if training_state is not None else "inference"
    )
    if kind not in {"inference", "training"}:
        raise ValueError("checkpoint_type must be 'inference' or 'training'")
    if kind == "training" and training_state is None:
        raise ValueError("training checkpoints require training_state")
    if kind == "inference" and training_state is not None:
        raise ValueError("inference checkpoints cannot contain training_state")
    preprocess = _raster_config(preprocess_config)
    window = _window_config(window_config)
    if preprocess is not None and preprocess.resolution != model.config.resolution:
        raise ValueError("preprocess and model resolutions differ")
    clean_metadata = _json_mapping("metadata", metadata)
    clean_training_config = (
        None
        if training_config is None
        else _json_mapping("training_config", training_config)
    )
    payload: dict[str, Any] = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": kind,
        "architecture_id": model.architecture_id,
        "architecture": dict(architecture_metadata(model)),
        "model_config": model.config.as_dict(),
        "model_config_recipe_digest": model.config.recipe_digest(),
        "model": model.state_dict(),
        "metadata": clean_metadata,
        "preprocess_config": (
            None if preprocess is None else preprocess.as_dict()
        ),
        "window_config": None if window is None else window.as_dict(),
        "training_config": clean_training_config,
    }
    if training_state is not None:
        payload["training_state"] = dict(training_state)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _legacy_training_state(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    keys = (
        "optimizer",
        "scheduler",
        "scaler",
        "kl_controller",
        "mask_generator_state",
        "step",
        "best_validation_loss",
        "torch_rng_state",
        "numpy_rng_state",
        "python_rng_state",
    )
    state = {name: checkpoint[name] for name in keys if name in checkpoint}
    return state or None


@overload
def load_progress_temporal_vae_checkpoint(
    path: str | Path = DEFAULT_TEMPORAL_VAE_CHECKPOINT,
    device: str | torch.device = "cpu",
    *,
    return_info: Literal[False] = False,
) -> ProgressTemporalVAE: ...


@overload
def load_progress_temporal_vae_checkpoint(
    path: str | Path = DEFAULT_TEMPORAL_VAE_CHECKPOINT,
    device: str | torch.device = "cpu",
    *,
    return_info: Literal[True],
) -> LoadedProgressTemporalVAECheckpoint: ...


def load_progress_temporal_vae_checkpoint(
    path: str | Path = DEFAULT_TEMPORAL_VAE_CHECKPOINT,
    device: str | torch.device = "cpu",
    *,
    return_info: bool = False,
) -> ProgressTemporalVAE | LoadedProgressTemporalVAECheckpoint:
    """Load new-schema or legacy research payloads without external access."""

    checkpoint_path = Path(path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("progress temporal VAE checkpoint must be a mapping")
    schema = checkpoint.get("checkpoint_schema")
    if schema is not None:
        if schema != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint belongs to another schema")
        if (
            checkpoint.get("checkpoint_schema_version")
            != CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported checkpoint schema version")
    if checkpoint.get("architecture_id") not in {
        ARCHITECTURE_ID,
        LEGACY_ARCHITECTURE_ID,
    }:
        raise ValueError("checkpoint architecture mismatch")
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint lacks a valid model_config")
    try:
        config = ProgressTemporalVAEConfig(**dict(raw_config))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint model configuration is invalid") from exc
    expected_digest = checkpoint.get("model_config_recipe_digest")
    if (
        expected_digest is not None
        and expected_digest != config.recipe_digest()
    ):
        raise ValueError("checkpoint model configuration digest mismatch")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint lacks a model state dictionary")
    model = ProgressTemporalVAE(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "checkpoint model state is incompatible with model_config"
        ) from exc

    preprocess = _raster_config(checkpoint.get("preprocess_config"))
    window = _window_config(checkpoint.get("window_config"))
    if preprocess is not None and preprocess.resolution != config.resolution:
        raise ValueError("checkpoint preprocessing resolution mismatch")
    metadata_value = checkpoint.get("metadata", {})
    if not isinstance(metadata_value, Mapping):
        raise ValueError("checkpoint metadata must be a mapping")
    metadata = dict(metadata_value)
    training_config_value = checkpoint.get("training_config")
    if training_config_value is not None and not isinstance(
        training_config_value, Mapping
    ):
        raise ValueError("checkpoint training_config must be a mapping")
    training_state_value = checkpoint.get("training_state")
    if training_state_value is not None and not isinstance(
        training_state_value, Mapping
    ):
        raise ValueError("checkpoint training_state must be a mapping")
    training_state = (
        dict(training_state_value)
        if training_state_value is not None
        else _legacy_training_state(checkpoint)
    )
    kind = str(
        checkpoint.get(
            "checkpoint_type",
            "training" if training_state is not None else "inference",
        )
    )
    if kind not in {"inference", "training"}:
        raise ValueError("checkpoint has an invalid checkpoint_type")
    if schema is not None:
        if kind == "inference" and training_state is not None:
            raise ValueError("inference checkpoint contains training state")
        if kind == "training" and training_state is None:
            raise ValueError("training checkpoint lacks training state")
    digest = _sha256_file(checkpoint_path)
    model.checkpoint_metadata = metadata
    model.checkpoint_sha256 = digest
    model.checkpoint_preprocess_config = preprocess
    model.checkpoint_window_config = window
    model.to(torch.device(device)).eval()
    info = LoadedProgressTemporalVAECheckpoint(
        model=model,
        checkpoint_type=kind,
        schema_version=(
            int(checkpoint.get("checkpoint_schema_version", 0))
            if schema is not None
            else 0
        ),
        metadata=metadata,
        preprocess_config=preprocess,
        window_config=window,
        training_config=(
            None
            if training_config_value is None
            else dict(training_config_value)
        ),
        training_state=training_state,
        checkpoint_sha256=digest,
        path=checkpoint_path.resolve(),
    )
    return info if return_info else model


def export_inference_checkpoint(
    source: str | Path | ProgressTemporalVAE,
    destination: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    preprocess_config: ProgressRasterConfig | Mapping[str, Any] | None = None,
    window_config: ProgressWindowConfig | Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Strip training state and export a portable inference checkpoint."""

    if isinstance(source, ProgressTemporalVAE):
        model = source
        inherited_metadata = dict(model.checkpoint_metadata)
        inherited_preprocess = model.checkpoint_preprocess_config
        inherited_window = model.checkpoint_window_config
        source_digest = model.checkpoint_sha256 or _state_dict_digest(model)
    else:
        loaded = load_progress_temporal_vae_checkpoint(
            source, "cpu", return_info=True
        )
        model = loaded.model
        inherited_metadata = dict(loaded.metadata)
        inherited_preprocess = loaded.preprocess_config
        inherited_window = loaded.window_config
        source_digest = loaded.checkpoint_sha256
    inherited_metadata.update(_json_mapping("metadata", metadata))
    inherited_metadata.setdefault("source_checkpoint_sha256", source_digest)
    selected_preprocess = (
        inherited_preprocess
        if preprocess_config is None
        else _raster_config(preprocess_config)
    )
    selected_window = (
        inherited_window
        if window_config is None
        else _window_config(window_config)
    )
    save_progress_temporal_vae_checkpoint(
        destination,
        model,
        checkpoint_type="inference",
        metadata=inherited_metadata,
        preprocess_config=selected_preprocess,
        window_config=selected_window,
    )
    output = Path(destination).resolve()
    return {
        "path": str(output),
        "checkpoint_sha256": _sha256_file(output),
        "source_checkpoint_sha256": source_digest,
        "architecture_id": model.architecture_id,
    }


@dataclass(frozen=True)
class ProgressSequenceEncoding:
    mean: torch.Tensor
    logvar: torch.Tensor
    valid: torch.Tensor
    window_valid_fraction: torch.Tensor
    timestamps: torch.Tensor
    offsets: torch.Tensor
    recipe_digest: str
    checkpoint_sha256: str
    checkpoint_metadata: Mapping[str, Any]
    representation: str = TEMPORAL_LATENT_REPRESENTATION


class ProgressSequenceEncoder:
    """Cache frame CNN tokens once, then infer timestamp-aligned windows."""

    def __init__(
        self,
        model_or_checkpoint: ProgressTemporalVAE | str | Path = (
            DEFAULT_TEMPORAL_VAE_CHECKPOINT
        ),
        *,
        device: str | torch.device = "cpu",
        window_config: ProgressWindowConfig | None = None,
        preprocess_config: ProgressRasterConfig | None = None,
        frame_batch_size: int = 256,
        window_batch_size: int = 256,
        amp: bool = True,
        amp_dtype: Literal["float16", "bfloat16"] = "bfloat16",
    ) -> None:
        _strict_positive_integer("frame_batch_size", frame_batch_size)
        _strict_positive_integer("window_batch_size", window_batch_size)
        if not isinstance(amp, bool):
            raise ValueError("amp must be boolean")
        if amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float16 or bfloat16")
        self.device = torch.device(device)
        if isinstance(model_or_checkpoint, ProgressTemporalVAE):
            self.model = model_or_checkpoint.to(self.device).eval()
            loaded_window = self.model.checkpoint_window_config
            loaded_preprocess = self.model.checkpoint_preprocess_config
            checkpoint_sha256 = (
                self.model.checkpoint_sha256
                or _state_dict_digest(self.model)
            )
        else:
            loaded = load_progress_temporal_vae_checkpoint(
                model_or_checkpoint, self.device, return_info=True
            )
            self.model = loaded.model
            loaded_window = loaded.window_config
            loaded_preprocess = loaded.preprocess_config
            checkpoint_sha256 = loaded.checkpoint_sha256
        self.window_config = (
            window_config or loaded_window or ProgressWindowConfig()
        )
        self.preprocess_config = (
            preprocess_config
            or loaded_preprocess
            or ProgressRasterConfig(resolution=self.model.config.resolution)
        )
        if self.preprocess_config.resolution != self.model.config.resolution:
            raise ValueError("preprocessing and checkpoint resolutions differ")
        self.frame_batch_size = frame_batch_size
        self.window_batch_size = window_batch_size
        self.amp = amp
        self.amp_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[amp_dtype]
        self.checkpoint_sha256 = checkpoint_sha256
        self.recipe_digest = canonical_recipe_digest(
            {
                "recipe": "progress_sequence_encoder_v1",
                "architecture": self.model.config.as_dict(),
                "architecture_recipe_digest": self.model.config.recipe_digest(),
                "checkpoint_sha256": checkpoint_sha256,
                "preprocess": self.preprocess_config.as_dict(),
                "window": self.window_config.as_dict(),
            }
        )

    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp and self.device.type == "cuda",
        )

    @torch.no_grad()
    def encode(
        self,
        progress: torch.Tensor,
        timestamps: torch.Tensor,
        valid: torch.Tensor,
    ) -> ProgressSequenceEncoding:
        if not isinstance(progress, torch.Tensor) or progress.ndim != 4:
            raise ValueError("progress must have shape [F,1,H,W]")
        expected = (
            len(progress),
            1,
            self.model.config.resolution,
            self.model.config.resolution,
        )
        if tuple(progress.shape) != expected:
            raise ValueError(f"progress must have shape {expected}")
        timestamp = torch.as_tensor(
            timestamps, dtype=torch.float64, device="cpu"
        )
        center_valid = torch.as_tensor(
            valid, dtype=torch.bool, device="cpu"
        )
        frames = len(progress)
        if timestamp.shape != (frames,) or center_valid.shape != (frames,):
            raise ValueError("timestamps and valid must have shape [F]")
        if frames < 1:
            raise ValueError("progress sequence cannot be empty")
        if not torch.isfinite(timestamp).all():
            raise ValueError("timestamps must be finite")
        if frames > 1 and not torch.all(timestamp[1:] > timestamp[:-1]):
            raise ValueError("timestamps must be strictly increasing")
        if progress.is_floating_point():
            if not torch.isfinite(progress[center_valid.to(progress.device)]).all():
                raise ValueError("valid progress frames must be finite")
        elif progress.dtype != torch.uint8:
            raise TypeError("progress must be floating point or uint8")

        token = torch.full(
            (frames, self.model.config.frame_token_dim),
            float("nan"),
            dtype=torch.float32,
        )
        visible_positions = torch.nonzero(
            center_valid, as_tuple=False
        ).flatten()
        for start in range(0, len(visible_positions), self.frame_batch_size):
            selected = visible_positions[start : start + self.frame_batch_size]
            values = progress.index_select(
                0, selected.to(progress.device)
            ).to(self.device, non_blocking=True)
            if values.dtype == torch.uint8:
                values = values.to(torch.float32).mul_(1.0 / 255.0)
            else:
                values = values.to(torch.float32)
            with self._autocast():
                encoded = self.model.encoder.frame_encoder(values)
            token.index_copy_(0, selected, encoded.float().cpu())

        context = self.model.config.context_frames
        offsets = self.window_config.offsets(context)
        minimum = math.ceil(
            self.window_config.minimum_valid_fraction * context
        )
        mean = torch.full(
            (frames, self.model.config.latent_dim),
            float("nan"),
            dtype=torch.float32,
        )
        logvar = torch.full_like(mean, float("nan"))
        output_valid = torch.zeros(frames, dtype=torch.bool)
        valid_fraction = torch.zeros(frames, dtype=torch.float32)

        for start in range(0, frames, self.window_batch_size):
            centers = torch.arange(
                start, min(start + self.window_batch_size, frames)
            )
            target = timestamp[centers, None] + offsets[None]
            flat = target.reshape(-1)
            right = torch.searchsorted(timestamp, flat)
            left = (right - 1).clamp(0, frames - 1)
            right = right.clamp(0, frames - 1)
            choose_right = (
                (timestamp[right] - flat).abs()
                < (timestamp[left] - flat).abs()
            )
            nearest = torch.where(choose_right, right, left).reshape(
                target.shape
            )
            present = (
                (timestamp[nearest] - target).abs()
                <= self.window_config.nearest_tolerance
            ) & center_valid[nearest]
            fraction = present.to(torch.float32).mean(dim=1)
            valid_fraction[centers] = fraction
            stride_selected = centers.remainder(
                self.window_config.output_stride
            ) == 0
            accepted = (
                center_valid[centers]
                & (present.sum(dim=1) >= minimum)
                & stride_selected
            )
            if not accepted.any():
                continue
            selected_centers = centers[accepted]
            selected_nearest = nearest[accepted]
            selected_present = present[accepted]
            windows = token[selected_nearest].masked_fill(
                ~selected_present[..., None], 0.0
            )
            window_token = windows.to(self.device, non_blocking=True)
            frame_valid = selected_present.to(
                self.device, non_blocking=True
            )
            with self._autocast():
                encoded = self.model.encode_tokens(
                    window_token, frame_valid
                )
            mean[selected_centers] = encoded["mean"].float().cpu()
            logvar[selected_centers] = encoded["logvar"].float().cpu()
            output_valid[selected_centers] = True

        checkpoint_metadata = {
            **self.model.checkpoint_metadata,
            "architecture": dict(architecture_metadata(self.model)),
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocess_config": self.preprocess_config.as_dict(),
            "window_config": self.window_config.as_dict(),
        }
        return ProgressSequenceEncoding(
            mean=mean,
            logvar=logvar,
            valid=output_valid,
            window_valid_fraction=valid_fraction,
            timestamps=timestamp.clone(),
            offsets=offsets,
            recipe_digest=self.recipe_digest,
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_metadata=checkpoint_metadata,
        )

    def __call__(
        self,
        progress: torch.Tensor,
        timestamps: torch.Tensor,
        valid: torch.Tensor,
    ) -> ProgressSequenceEncoding:
        return self.encode(progress, timestamps, valid)

    @torch.no_grad()
    def encode_hydra(
        self,
        raw: torch.Tensor,
        timestamps: torch.Tensor,
        canonicalization: ProgressCanonicalizationInput,
        *,
        valid: torch.Tensor | None = None,
    ) -> ProgressSequenceEncoding:
        """Encode native Hydra sequences without an intermediate archive.

        ``raw`` must contain finite logits with shape ``[frame, channel, H, W]``.
        Missing detections should use finite placeholder logits and be marked
        false in ``valid``. Canonicalization is performed with the preprocessing
        recipe associated with this encoder before frame tokens are cached.
        """

        if not isinstance(raw, torch.Tensor) or raw.ndim != 4:
            raise ValueError("raw must have shape [frame, channel, H, W]")
        if len(raw) < 1:
            raise ValueError("raw Hydra sequence cannot be empty")
        canonical = canonicalize_hydra_progress(
            raw,
            canonicalization,
            config=self.preprocess_config,
        )
        combined_valid = canonical.valid.detach().to(device="cpu", dtype=torch.bool)
        if valid is not None:
            requested = torch.as_tensor(valid, dtype=torch.bool, device="cpu")
            if requested.shape != (len(raw),):
                raise ValueError("valid must have shape [frame]")
            combined_valid &= requested
        return self.encode(
            canonical.progress,
            timestamps,
            combined_valid,
        )

    encode_hydra_sequence = encode_hydra


__all__ = [
    "ARCHITECTURE_ID",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_VERSION",
    "DNTProgressTemporalVAE",
    "DNTProgressTemporalVAEConfig",
    "LEGACY_ARCHITECTURE_ID",
    "LOGVAR_MAX",
    "LOGVAR_MIN",
    "LoadedProgressTemporalVAECheckpoint",
    "MODEL_SCHEMA_VERSION",
    "MaskedAttentionPool",
    "ProgressFrameEncoder",
    "ProgressSequenceEncoder",
    "ProgressSequenceEncoding",
    "ProgressTemporalVAE",
    "ProgressTemporalVAEConfig",
    "ProgressWindowEncoder",
    "SpatialDownStage",
    "SpatialResidualBlock",
    "SpatialUpStage",
    "StrictProgressDecoder",
    "TEMPORAL_LATENT_REPRESENTATION",
    "architecture_metadata",
    "assert_strict_bottleneck",
    "export_inference_checkpoint",
    "load_progress_temporal_vae_checkpoint",
    "parameter_count",
    "save_progress_temporal_vae_checkpoint",
]
