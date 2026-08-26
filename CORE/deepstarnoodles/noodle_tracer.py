# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Spatial multimodal recurrent policy for flexible arm-polyline tracing."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from deepstarnoodles.decode import _smooth_resample, find_peaks
from deepstarnoodles.decode_embed import (
    EmbedDecodeParams,
    _sample_reference_embedding,
    _suppress_duplicates_and_fragments,
)


def extract_centered_patches(
    fields: torch.Tensor,
    positions: torch.Tensor,
    image_indices: torch.Tensor | None,
    patch_size: int,
) -> torch.Tensor:
    """Bilinearly sample patches centred at xy positions.

    ``fields`` is (B,C,H,W), ``positions`` is (N,2), and ``image_indices`` maps
    each position to its source image. Coordinates use align_corners=True pixel
    centres, so an integer position is exactly the middle patch pixel.
    """
    if patch_size % 2 != 1:
        raise ValueError("patch_size must be odd")
    if fields.ndim != 4 or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("invalid field or position shape")
    if image_indices is None:
        if fields.shape[0] != 1:
            raise ValueError("image_indices is required for multi-image fields")
        selected = fields.expand(positions.shape[0], -1, -1, -1)
    else:
        selected = fields[image_indices.long()]
    _, _, height, width = selected.shape
    radius = patch_size // 2
    offsets = torch.arange(-radius, radius + 1, device=positions.device, dtype=positions.dtype)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    x = positions[:, 0, None, None] + xx
    y = positions[:, 1, None, None] + yy
    gx = 2.0 * x / max(width - 1, 1) - 1.0
    gy = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        selected,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )



class TraceState(IntEnum):
    INVALID_SEED = 0
    CONTINUE = 1
    TERMINATE_BASE = 2
    TERMINATE_LOST = 3


TRACE_STATE_NAMES = {
    TraceState.INVALID_SEED: "invalid_seed",
    TraceState.CONTINUE: "continue",
    TraceState.TERMINATE_BASE: "base",
    TraceState.TERMINATE_LOST: "lost",
}

_INFERENCE_ATTENTION_BRANCH_ATTRIBUTE = "_inference_attention_branch"


def _multiscale_attention_weights(
    scores: torch.Tensor,
    *,
    key_channels: int,
    inference_branch: int | None = None,
) -> torch.Tensor:
    """Normalize scale-token scores, optionally excluding one scale entirely."""
    if scores.ndim != 4 or scores.shape[1] != 2:
        raise ValueError("multiscale attention scores must be (N,2,H,W)")
    flattened = scores.flatten(1) / math.sqrt(key_channels)
    if inference_branch is not None:
        if inference_branch not in (0, 1):
            raise ValueError("inference attention branch must be 0 or 1")
        allowed = torch.zeros(
            (1, 2, 1, 1),
            dtype=torch.bool,
            device=scores.device,
        )
        allowed[:, inference_branch] = True
        flattened = flattened.masked_fill(
            ~allowed.expand_as(scores).flatten(1),
            -torch.inf,
        )
    return torch.softmax(flattened, dim=1).reshape(
        scores.shape[0],
        scores.shape[1],
        1,
        scores.shape[2],
        scores.shape[3],
    )


@dataclass(frozen=True)
class NoodleTraceParams:
    dense_patch_size: int = 33
    rgb_feature_patch_size: int | None = None
    context_mode: str = "single"
    context_dilation: int = 2
    tip_threshold: float = 0.2
    tip_nms_radius: int = 4
    min_step_px: float = 0.5
    max_step_px: float = 6.0
    max_steps: int = 128
    max_path_px: float = 768.0
    min_path_px: float = 8.0
    loop_radius: float = 2.0
    loop_history: int = 8
    smooth_window: int = 5
    resample_spacing: float = 4.0
    base_threshold: float = 0.25
    base_hit_radius: float = 6.0
    embedding_dim: int = 8
    identity_scale: float = 1.0
    suppress_duplicates: bool = True
    emit_lost: bool = False
    invalid_seed_threshold: float | None = None
    base_state_threshold: float | None = None
    lost_state_threshold: float | None = None
    max_tips: int | None = None
    tip_seed_separation: float | None = None
    support_gated_peaks: bool = False

    def __post_init__(self) -> None:
        if self.dense_patch_size < 9 or self.dense_patch_size % 4 != 1:
            raise ValueError("dense_patch_size must be >= 9 and satisfy P % 4 == 1")
        spatial_size = (self.dense_patch_size + 3) // 4
        if self.rgb_feature_patch_size is None:
            object.__setattr__(self, "rgb_feature_patch_size", spatial_size)
        elif self.rgb_feature_patch_size != spatial_size:
            raise ValueError("rgb_feature_patch_size must match the dense stem output")
        if self.context_mode not in ("single", "multiscale_dilated"):
            raise ValueError("unsupported context_mode")
        if self.context_dilation < 1:
            raise ValueError("context_dilation must be positive")
        if self.context_mode == "multiscale_dilated" and self.dense_patch_size != 33:
            raise ValueError("multiscale_dilated uses fixed 33x33 branches")
        if not 0 < self.min_step_px <= self.max_step_px:
            raise ValueError("invalid step bounds")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.max_tips is not None and self.max_tips < 1:
            raise ValueError("max_tips must be positive")
        if self.tip_seed_separation is not None and self.tip_seed_separation < 0:
            raise ValueError("tip_seed_separation must be non-negative")


class RGBPyramid(NamedTuple):
    fine: torch.Tensor
    coarse: torch.Tensor
    image_size: tuple[int, int]


def select_trace_seeds(
    peaks: np.ndarray,
    *,
    max_tips: int | None = None,
    min_separation: float | None = None,
) -> np.ndarray:
    """Keep the highest-scoring, spatially separated tip peaks.

    ``find_peaks`` already returns score-descending candidates.  A second,
    wider suppression pass is for inference only: Sleepy tip maps can light
    up along whole arms and otherwise spawn thousands of GRU walkers.
    """
    if peaks.size == 0:
        return peaks
    selected = np.asarray(peaks, dtype=np.float32)
    if min_separation is not None and min_separation > 0 and len(selected) > 1:
        keep: list[int] = []
        xy = selected[:, :2]
        limit = min_separation * min_separation
        for index in range(len(selected)):
            if keep and np.any(np.sum((xy[keep] - xy[index]) ** 2, axis=1) <= limit):
                continue
            keep.append(index)
        selected = selected[keep]
    if max_tips is not None:
        selected = selected[: int(max_tips)]
    return selected


def _features_for_indices(features: torch.Tensor, image_indices: torch.Tensor) -> torch.Tensor:
    """Index feature maps without cloning a full map per seed."""
    indices = image_indices.long()
    if indices.numel() == 0:
        return features[:0]
    if features.shape[0] == 1:
        return features.expand(indices.shape[0], -1, -1, -1)
    first = indices[0]
    if torch.equal(indices, first.expand_as(indices)):
        return features[first : first + 1].expand(indices.shape[0], -1, -1, -1)
    return features.index_select(0, indices)


def _sample_feature_patches(
    features: torch.Tensor,
    positions: torch.Tensor,
    image_indices: torch.Tensor,
    *,
    image_size: tuple[int, int],
    patch_size: int,
    image_spacing: float | None = None,
) -> torch.Tensor:
    """Sample a spatial feature patch at image-coordinate xy positions."""
    if patch_size % 2 != 1:
        raise ValueError("patch_size must be odd")
    selected = _features_for_indices(features, image_indices)
    _, _, feature_h, feature_w = selected.shape
    image_h, image_w = image_size
    x_center = positions[:, 0] * max(feature_w - 1, 1) / max(image_w - 1, 1)
    y_center = positions[:, 1] * max(feature_h - 1, 1) / max(image_h - 1, 1)
    radius = patch_size // 2
    offsets = torch.arange(
        -radius,
        radius + 1,
        device=positions.device,
        dtype=positions.dtype,
    )
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    if image_spacing is None:
        x = x_center[:, None, None] + xx
        y = y_center[:, None, None] + yy
    else:
        x = x_center[:, None, None] + (
            xx * float(image_spacing) * max(feature_w - 1, 1) / max(image_w - 1, 1)
        )
        y = y_center[:, None, None] + (
            yy * float(image_spacing) * max(feature_h - 1, 1) / max(image_h - 1, 1)
        )
    gx = 2.0 * x / max(feature_w - 1, 1) - 1.0
    gy = 2.0 * y / max(feature_h - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        selected,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class SpatialRGBEncoder(nn.Module):
    """Early MobileNetV3-Small blocks retained as fine and coarse feature maps."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights).features
        self.fine = nn.Sequential(*list(backbone.children())[:2])
        self.coarse = nn.Sequential(*list(backbone.children())[2:4])
        self.pretrained = bool(pretrained)
        self.frozen = bool(frozen)
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
        )
        if self.frozen:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> RGBPyramid:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("RGB input must be (B,3,H,W)")
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        image = image.float()
        normalized = (image - self.rgb_mean) / self.rgb_std
        if self.frozen:
            self.fine.eval()
            self.coarse.eval()
            with torch.no_grad():
                fine = self.fine(normalized)
                coarse = self.coarse(fine)
        else:
            fine_trainable = any(
                parameter.requires_grad for parameter in self.fine.parameters()
            )
            coarse_trainable = any(
                parameter.requires_grad for parameter in self.coarse.parameters()
            )
            if fine_trainable:
                fine = self.fine(normalized)
            else:
                self.fine.eval()
                with torch.no_grad():
                    fine = self.fine(normalized)
            if coarse_trainable:
                coarse = self.coarse(fine)
            else:
                self.coarse.eval()
                with torch.no_grad():
                    coarse = self.coarse(fine)
        return RGBPyramid(
            fine=fine,
            coarse=coarse,
            image_size=(int(image.shape[-2]), int(image.shape[-1])),
        )


class NoodleTracer(nn.Module):
    """Spatial CNN-attention-GRU policy with flexible actions and trace states."""

    def __init__(
        self,
        *,
        dense_channels: int = 18,
        embedding_dim: int = 8,
        dense_patch_size: int = 33,
        rgb_feature_patch_size: int | None = None,
        hidden_size: int = 128,
        spatial_size: int | None = None,
        context_mode: str = "single",
        context_dilation: int = 2,
        pretrained_rgb: bool = True,
        freeze_rgb: bool = True,
        dense_mean: Sequence[float] | None = None,
        dense_std: Sequence[float] | None = None,
        min_step_px: float = 0.5,
        max_step_px: float = 6.0,
    ) -> None:
        super().__init__()
        if dense_channels < 10 + embedding_dim:
            raise ValueError("dense channels do not contain the requested embedding")
        if dense_patch_size < 9 or dense_patch_size % 4 != 1:
            raise ValueError("dense_patch_size must be >= 9 and satisfy P % 4 == 1")
        derived_spatial_size = (dense_patch_size + 3) // 4
        spatial_size = (
            derived_spatial_size if spatial_size is None else int(spatial_size)
        )
        rgb_feature_patch_size = (
            derived_spatial_size
            if rgb_feature_patch_size is None
            else int(rgb_feature_patch_size)
        )
        if spatial_size != derived_spatial_size:
            raise ValueError("spatial_size must match the dense stem output")
        if rgb_feature_patch_size != spatial_size:
            raise ValueError("RGB feature patch and dense spatial sizes must match")
        if context_mode not in ("single", "multiscale_dilated"):
            raise ValueError("unsupported context_mode")
        if context_dilation < 1:
            raise ValueError("context_dilation must be positive")
        if context_mode == "multiscale_dilated" and dense_patch_size != 33:
            raise ValueError("multiscale_dilated uses fixed 33x33 branches")
        self.dense_channels = int(dense_channels)
        self.embedding_dim = int(embedding_dim)
        self.dense_patch_size = int(dense_patch_size)
        self.rgb_feature_patch_size = int(rgb_feature_patch_size)
        self.hidden_size = int(hidden_size)
        self.spatial_size = int(spatial_size)
        self.context_mode = str(context_mode)
        self.context_dilation = int(context_dilation)
        self.min_step_px = float(min_step_px)
        self.max_step_px = float(max_step_px)
        self.rgb_encoder = SpatialRGBEncoder(
            pretrained=pretrained_rgb,
            frozen=freeze_rgb,
        )
        mean = (
            torch.zeros(self.dense_channels)
            if dense_mean is None
            else torch.as_tensor(dense_mean)
        )
        std = (
            torch.ones(self.dense_channels)
            if dense_std is None
            else torch.as_tensor(dense_std)
        )
        if mean.numel() != self.dense_channels or std.numel() != self.dense_channels:
            raise ValueError("dense normalization must have one value per channel")
        self.register_buffer("dense_mean", mean.float().reshape(1, -1, 1, 1))
        self.register_buffer(
            "dense_std",
            std.float().clamp_min(1e-4).reshape(1, -1, 1, 1),
        )
        # Raw dense channels + explicit seed similarity + relative x/y.
        self.dense_stem = nn.Sequential(
            nn.Conv2d(self.dense_channels + 3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.rgb_fine_projection = nn.Conv2d(16, 16, 1)
        self.rgb_coarse_projection = nn.Conv2d(24, 16, 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        conditioning_size = self.hidden_size + 2 + self.embedding_dim
        self.attention_query = nn.Linear(conditioning_size, 64)
        self.attention_key = nn.Conv2d(64, 64, 1)
        if self.context_mode == "multiscale_dilated":
            self.scale_embeddings = nn.Parameter(torch.zeros(2, 64))
        recurrent_input_size = 64 + 2 + self.embedding_dim
        self.recurrent = nn.GRUCell(recurrent_input_size, self.hidden_size)
        self.direction_head = nn.Linear(self.hidden_size, 2)
        self.step_head = nn.Linear(self.hidden_size, 1)
        self.state_head = nn.Linear(self.hidden_size, len(TraceState))

    def initial_state(
        self,
        count: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return torch.zeros(
            count,
            self.hidden_size,
            device=device or self.dense_mean.device,
        )

    def encode_rgb(self, image: torch.Tensor) -> RGBPyramid:
        return self.rgb_encoder(image)

    def sample_dense(
        self,
        fields: torch.Tensor,
        positions: torch.Tensor,
        image_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        local = extract_centered_patches(
            fields,
            positions,
            image_indices,
            self.dense_patch_size,
        )
        if self.context_mode == "single":
            return local
        selected = (
            fields.expand(positions.shape[0], -1, -1, -1)
            if image_indices is None
            else fields[image_indices.long()]
        )
        height, width = selected.shape[-2:]
        radius = self.dense_patch_size // 2
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=positions.device,
            dtype=positions.dtype,
        ) * self.context_dilation
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        x = positions[:, 0, None, None] + xx
        y = positions[:, 1, None, None] + yy
        grid = torch.stack(
            (
                2.0 * x / max(width - 1, 1) - 1.0,
                2.0 * y / max(height - 1, 1) - 1.0,
            ),
            dim=-1,
        )
        context = F.grid_sample(
            selected,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return torch.stack((local, context), dim=1)

    def sample_rgb(
        self,
        pyramid: RGBPyramid,
        positions: torch.Tensor,
        image_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fine = _sample_feature_patches(
            pyramid.fine,
            positions,
            image_indices,
            image_size=pyramid.image_size,
            patch_size=self.rgb_feature_patch_size,
        )
        coarse = _sample_feature_patches(
            pyramid.coarse,
            positions,
            image_indices,
            image_size=pyramid.image_size,
            patch_size=self.rgb_feature_patch_size,
        )
        if self.context_mode == "single":
            return fine, coarse
        # The dense stem has total stride four. Sampling RGB in image
        # coordinates aligns both feature maps to the dilated dense tokens.
        context_spacing = 4 * self.context_dilation
        context_fine = _sample_feature_patches(
            pyramid.fine,
            positions,
            image_indices,
            image_size=pyramid.image_size,
            patch_size=self.rgb_feature_patch_size,
            image_spacing=context_spacing,
        )
        context_coarse = _sample_feature_patches(
            pyramid.coarse,
            positions,
            image_indices,
            image_size=pyramid.image_size,
            patch_size=self.rgb_feature_patch_size,
            image_spacing=context_spacing,
        )
        fine = torch.stack((fine, context_fine), dim=1)
        coarse = torch.stack((coarse, context_coarse), dim=1)
        return fine, coarse

    def forward_step(
        self,
        dense_patches: torch.Tensor,
        rgb_fine_patches: torch.Tensor,
        rgb_coarse_patches: torch.Tensor,
        previous_displacement: torch.Tensor,
        seed_embedding: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        identity_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = dense_patches.shape[0]
        expected_dense_shape = (
            self.dense_channels,
            self.dense_patch_size,
            self.dense_patch_size,
        )
        scale_count = 1 if self.context_mode == "single" else 2
        if self.context_mode == "single" and dense_patches.shape[1:] != expected_dense_shape:
            raise ValueError("unexpected dense patch shape")
        if (
            self.context_mode == "multiscale_dilated"
            and dense_patches.shape[1:] != (scale_count, *expected_dense_shape)
        ):
            raise ValueError("unexpected multiscale dense patch shape")
        if previous_displacement.shape != (count, 2):
            raise ValueError("previous_displacement must be (N,2)")
        if seed_embedding.shape != (count, self.embedding_dim):
            raise ValueError("seed_embedding has wrong shape")
        if hidden is None:
            hidden = self.initial_state(count, device=dense_patches.device)
        if self.context_mode == "multiscale_dilated":
            dense_input = dense_patches.flatten(0, 1)
            fine_input = rgb_fine_patches.flatten(0, 1)
            coarse_input = rgb_coarse_patches.flatten(0, 1)
            expanded_seed = seed_embedding[:, None].expand(-1, scale_count, -1)
            seed_input = expanded_seed.reshape(count * scale_count, -1)
        else:
            dense_input = dense_patches
            fine_input = rgb_fine_patches
            coarse_input = rgb_coarse_patches
            seed_input = seed_embedding
        normalized = (dense_input.float() - self.dense_mean) / self.dense_std
        local_embedding = dense_input[
            :,
            10:10 + self.embedding_dim,
        ].float()
        identity_distance = torch.linalg.vector_norm(
            local_embedding - seed_input[:, :, None, None],
            dim=1,
            keepdim=True,
        )
        identity_similarity = torch.exp(
            -0.5 * (identity_distance / max(float(identity_scale), 1e-4)).square()
        )
        coordinate = torch.linspace(
            -1,
            1,
            self.dense_patch_size,
            device=dense_input.device,
            dtype=normalized.dtype,
        )
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=0)[None].expand(
            dense_input.shape[0], -1, -1, -1
        )
        dense_spatial = self.dense_stem(
            torch.cat((normalized, identity_similarity, coordinates), dim=1)
        )
        rgb_fine = self.rgb_fine_projection(fine_input.float())
        rgb_coarse = self.rgb_coarse_projection(coarse_input.float())
        spatial = self.fusion(torch.cat((dense_spatial, rgb_fine, rgb_coarse), dim=1))
        condition = torch.cat(
            (hidden, previous_displacement.float(), seed_embedding.float()),
            dim=1,
        )
        query = self.attention_query(condition)[:, :, None, None]
        if self.context_mode == "single":
            keys = self.attention_key(spatial)
            attention = torch.softmax(
                (keys * query).sum(dim=1).flatten(1) / math.sqrt(keys.shape[1]),
                dim=1,
            ).reshape(count, 1, self.spatial_size, self.spatial_size)
            context = (spatial * attention).sum(dim=(-2, -1))
        else:
            spatial = spatial.reshape(
                count, scale_count, 64, self.spatial_size, self.spatial_size
            )
            key_input = spatial + self.scale_embeddings[None, :, :, None, None]
            keys = self.attention_key(key_input.flatten(0, 1)).reshape_as(spatial)
            scores = (keys * query[:, None]).sum(dim=2)
            inference_branch = getattr(
                self,
                _INFERENCE_ATTENTION_BRANCH_ATTRIBUTE,
                None,
            )
            if inference_branch is not None and self.training:
                raise RuntimeError("attention branch masking is inference-only")
            attention = _multiscale_attention_weights(
                scores,
                key_channels=keys.shape[2],
                inference_branch=inference_branch,
            )
            context = (spatial * attention).sum(dim=(1, 3, 4))
        hidden = self.recurrent(
            torch.cat(
                (context, previous_displacement.float(), seed_embedding.float()),
                dim=1,
            ),
            hidden,
        )
        direction = F.normalize(self.direction_head(hidden), dim=1, eps=1e-6)
        step_size = self.min_step_px + (
            self.max_step_px - self.min_step_px
        ) * torch.sigmoid(self.step_head(hidden))
        displacement = direction * step_size
        state_logits = self.state_head(hidden)
        return displacement, state_logits, hidden

    def checkpoint_config(self) -> dict:
        return {
            "dense_channels": self.dense_channels,
            "embedding_dim": self.embedding_dim,
            "dense_patch_size": self.dense_patch_size,
            "rgb_feature_patch_size": self.rgb_feature_patch_size,
            "hidden_size": self.hidden_size,
            "spatial_size": self.spatial_size,
            "context_mode": self.context_mode,
            "context_dilation": self.context_dilation,
            # The state dict contains the actual RGB weights; avoid downloads on load.
            "pretrained_rgb": False,
            "freeze_rgb": self.rgb_encoder.frozen,
            "dense_mean": self.dense_mean.flatten().detach().cpu().tolist(),
            "dense_std": self.dense_std.flatten().detach().cpu().tolist(),
            "min_step_px": self.min_step_px,
            "max_step_px": self.max_step_px,
        }


def _seed_embedding(
    embedding: np.ndarray,
    tip_xy: np.ndarray,
    params: NoodleTraceParams,
    support_mask: np.ndarray,
) -> np.ndarray:
    return _sample_reference_embedding(
        embedding[:params.embedding_dim],
        float(tip_xy[0]),
        float(tip_xy[1]),
        2,
        support_mask,
    )


def _sample_live_patches(
    model: NoodleTracer,
    fields: torch.Tensor,
    pyramid: RGBPyramid,
    positions: torch.Tensor,
    image_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample walkers without materializing one full feature map per walker.

    Walkers are created in image order and active-index compaction preserves
    that order.  Grouping grid sampling by image avoids ``index_select`` copies
    of 512px dense fields and MobileNet maps for a mixed-image batch.
    """
    batch_size = int(fields.shape[0])
    if batch_size == 1:
        dense = model.sample_dense(fields, positions, None)
        fine, coarse = model.sample_rgb(
            pyramid,
            positions,
            image_indices,
        )
        return dense, fine, coarse

    dense_parts: list[torch.Tensor] = []
    fine_parts: list[torch.Tensor] = []
    coarse_parts: list[torch.Tensor] = []
    for image_index in range(batch_size):
        local = torch.nonzero(
            image_indices == image_index,
            as_tuple=False,
        ).flatten()
        if local.numel() == 0:
            continue
        local_positions = positions[local]
        dense_parts.append(
            model.sample_dense(
                fields[image_index : image_index + 1],
                local_positions,
                None,
            )
        )
        local_pyramid = RGBPyramid(
            fine=pyramid.fine[image_index : image_index + 1],
            coarse=pyramid.coarse[image_index : image_index + 1],
            image_size=pyramid.image_size,
        )
        fine, coarse = model.sample_rgb(
            local_pyramid,
            local_positions,
            torch.zeros(
                len(local_positions),
                dtype=torch.long,
                device=positions.device,
            ),
        )
        fine_parts.append(fine)
        coarse_parts.append(coarse)
    return (
        torch.cat(dense_parts, dim=0),
        torch.cat(fine_parts, dim=0),
        torch.cat(coarse_parts, dim=0),
    )


@torch.no_grad()
def trace_arms_noodle_batch(
    raw: np.ndarray | torch.Tensor,
    images: np.ndarray | torch.Tensor,
    model: NoodleTracer,
    params: NoodleTraceParams | None = None,
) -> list[list[dict]]:
    """Trace a batch of images with one GPU-resident recurrent rollout.

    Tip/base peak extraction intentionally stays on CPU to preserve the locked
    decoder contract.  All walkers from all images are flattened into one
    policy batch; path history and termination bookkeeping remain on the model
    device until the rollout finishes.
    """
    params = params or NoodleTraceParams(
        dense_patch_size=model.dense_patch_size,
        rgb_feature_patch_size=model.rgb_feature_patch_size,
        context_mode=model.context_mode,
        context_dilation=model.context_dilation,
        min_step_px=model.min_step_px,
        max_step_px=model.max_step_px,
    )
    if (
        params.dense_patch_size != model.dense_patch_size
        or params.rgb_feature_patch_size != model.rgb_feature_patch_size
        or params.context_mode != model.context_mode
        or params.context_dilation != model.context_dilation
    ):
        raise ValueError("trace patch geometry must match the model")
    device = next(model.parameters()).device
    raw_t = torch.as_tensor(raw, dtype=torch.float32, device=device)
    image_t = torch.as_tensor(images, device=device)
    if raw_t.ndim == 3:
        raw_t = raw_t[None]
    if image_t.ndim == 3:
        image_t = image_t[None]
    if image_t.ndim == 4 and image_t.shape[-1] == 3 and image_t.shape[1] != 3:
        image_t = image_t.permute(0, 3, 1, 2)
    if image_t.ndim != 4 or image_t.shape[1] != 3:
        raise ValueError("images must be aligned RGB as (B,3,H,W) or (B,H,W,3)")
    if raw_t.ndim != 4 or raw_t.shape[1] < model.dense_channels:
        raise ValueError("raw must be (B,C,H,W) with enough dense channels")
    if raw_t.shape[0] != image_t.shape[0]:
        raise ValueError("raw and images must have the same batch size")
    if raw_t.shape[-2:] != image_t.shape[-2:]:
        raise ValueError("raw and images must be spatially aligned")
    raw_t = raw_t[:, :model.dense_channels]
    batch_size = int(raw_t.shape[0])
    outputs: list[list[dict]] = [[] for _ in range(batch_size)]

    # Preserve exact SciPy peak extraction while moving only fields used by
    # host-side seed/base postprocessing: raw support, activated support/base/
    # tip, and the identity embedding.  The full 18-channel tensor stays on GPU.
    host_fields = torch.cat(
        (
            raw_t[:, 0:1],
            torch.sigmoid(raw_t[:, 0:1]),
            torch.sigmoid(raw_t[:, 8:10]),
            raw_t[:, 10 : 10 + params.embedding_dim],
        ),
        dim=1,
    ).detach().cpu().numpy()
    tips_by_image: list[np.ndarray] = []
    peak_support_by_image: list[np.ndarray | None] = []
    seed_arrays: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    for image_index in range(batch_size):
        fields = host_fields[image_index]
        raw_support = fields[0]
        activated_support = fields[1]
        activated_tip = fields[3]
        embedding = fields[4:]
        peak_support = (
            raw_support > 0 if params.support_gated_peaks else None
        )
        tip_heat = (
            np.where(peak_support, activated_tip, 0.0)
            if peak_support is not None
            else activated_tip
        )
        tips = select_trace_seeds(
            find_peaks(tip_heat, params.tip_threshold, params.tip_nms_radius),
            max_tips=params.max_tips,
            min_separation=params.tip_seed_separation,
        )
        tips_by_image.append(tips)
        peak_support_by_image.append(peak_support)
        if not len(tips):
            continue
        seed_support = (
            peak_support
            if peak_support is not None
            else activated_support > 0.2
        )
        seed_arrays.append(
            np.stack(
                [
                    _seed_embedding(embedding, tip[:2], params, seed_support)
                    for tip in tips
                ]
            )
        )
        source_indices.append(
            np.full(len(tips), image_index, dtype=np.int64)
        )
    count = sum(len(tips) for tips in tips_by_image)
    if count == 0:
        return outputs

    tips = np.concatenate(
        [tips for tips in tips_by_image if len(tips)],
        axis=0,
    )
    positions = torch.as_tensor(tips[:, :2], dtype=torch.float32, device=device)
    image_indices = torch.as_tensor(
        np.concatenate(source_indices),
        dtype=torch.long,
        device=device,
    )
    seeds = torch.as_tensor(
        np.concatenate(seed_arrays, axis=0),
        dtype=torch.float32,
        device=device,
    )
    hidden = model.initial_state(count, device=device)
    previous = torch.zeros((count, 2), device=device)
    active = torch.ones(count, dtype=torch.bool, device=device)
    accepted = torch.zeros(count, dtype=torch.bool, device=device)
    path_history = torch.zeros(
        (count, params.max_steps + 1, 2),
        dtype=torch.float32,
        device=device,
    )
    path_history[:, 0] = positions
    path_counts = torch.ones(count, dtype=torch.long, device=device)
    path_lengths = torch.zeros(count, dtype=torch.float32, device=device)
    state_confidence = torch.zeros(count, dtype=torch.float32, device=device)

    stop_invalid = int(TraceState.INVALID_SEED)
    stop_base = int(TraceState.TERMINATE_BASE)
    stop_lost = int(TraceState.TERMINATE_LOST)
    stop_bounds = len(TraceState)
    stop_max_length = stop_bounds + 1
    stop_loop = stop_bounds + 2
    stop_max_steps = stop_bounds + 3
    stop_codes = torch.full(
        (count,),
        stop_max_steps,
        dtype=torch.int8,
        device=device,
    )
    pyramid = model.encode_rgb(image_t)
    height, width = raw_t.shape[-2:]
    for step_index in range(params.max_steps):
        live = torch.nonzero(active, as_tuple=False).flatten()
        if live.numel() == 0:
            break
        dense_patches, rgb_fine, rgb_coarse = _sample_live_patches(
            model,
            raw_t,
            pyramid,
            positions[live],
            image_indices[live],
        )
        displacement, state_logits, next_hidden = model.forward_step(
            dense_patches,
            rgb_fine,
            rgb_coarse,
            previous[live],
            seeds[live],
            hidden[live],
            identity_scale=params.identity_scale,
        )
        probabilities = torch.softmax(state_logits, dim=1)
        decisions = probabilities.argmax(dim=1)
        if params.lost_state_threshold is not None:
            decisions = torch.where(
                probabilities[:, int(TraceState.TERMINATE_LOST)]
                >= params.lost_state_threshold,
                decisions.new_full((), int(TraceState.TERMINATE_LOST)),
                decisions,
            )
        if params.base_state_threshold is not None:
            decisions = torch.where(
                probabilities[:, int(TraceState.TERMINATE_BASE)]
                >= params.base_state_threshold,
                decisions.new_full((), int(TraceState.TERMINATE_BASE)),
                decisions,
            )
        if step_index == 0 and params.invalid_seed_threshold is not None:
            reject = (
                probabilities[:, int(TraceState.INVALID_SEED)]
                >= params.invalid_seed_threshold
            )
            alternatives = probabilities.clone()
            alternatives[:, int(TraceState.INVALID_SEED)] = -1
            non_invalid_decision = alternatives.argmax(dim=1)
            decisions = torch.where(
                reject,
                decisions.new_full((), int(TraceState.INVALID_SEED)),
                torch.where(
                    decisions == int(TraceState.INVALID_SEED),
                    non_invalid_decision,
                    decisions,
                ),
            )
        hidden[live] = next_hidden
        state_confidence[live] = probabilities.gather(
            1, decisions[:, None]
        ).squeeze(1)

        terminal_local = decisions != int(TraceState.CONTINUE)
        terminal_global = live[terminal_local]
        if terminal_global.numel():
            terminal_decisions = decisions[terminal_local].to(torch.int8)
            terminal_decisions = torch.where(
                (terminal_decisions == stop_invalid) & accepted[terminal_global],
                terminal_decisions.new_full((), stop_lost),
                terminal_decisions,
            )
            stop_codes[terminal_global] = terminal_decisions
            active[terminal_global] = False

        continue_local = torch.nonzero(~terminal_local, as_tuple=False).flatten()
        if continue_local.numel() == 0:
            continue
        continue_global = live[continue_local]
        accepted[continue_global] = True
        candidates = (
            positions[continue_global] + displacement[continue_local]
        )
        in_bounds = (
            (candidates[:, 0] >= 0)
            & (candidates[:, 0] <= width - 1)
            & (candidates[:, 1] >= 0)
            & (candidates[:, 1] <= height - 1)
        )
        out_global = continue_global[~in_bounds]
        if out_global.numel():
            active[out_global] = False
            stop_codes[out_global] = stop_bounds

        valid_global = continue_global[in_bounds]
        if valid_global.numel() == 0:
            continue
        valid_candidates = candidates[in_bounds]
        valid_displacements = displacement[continue_local][in_bounds]
        positions[valid_global] = valid_candidates
        previous[valid_global] = valid_displacements
        write_slots = path_counts[valid_global]
        path_history[valid_global, write_slots] = valid_candidates
        path_counts[valid_global] = write_slots + 1
        path_lengths[valid_global] += torch.linalg.vector_norm(
            valid_displacements,
            dim=1,
        )

        max_length = path_lengths[valid_global] >= params.max_path_px
        max_length_global = valid_global[max_length]
        if max_length_global.numel():
            active[max_length_global] = False
            stop_codes[max_length_global] = stop_max_length

        loop_global = valid_global[~max_length]
        # Every still-live walker has appended exactly once per previous
        # recurrent step, so the eligible history span is shared.  Slice only
        # that populated prefix instead of comparing against all max_steps
        # preallocated slots.
        history_stop = step_index + 2 - params.loop_history
        if loop_global.numel() and history_stop > 0:
            distances = torch.linalg.vector_norm(
                path_history[loop_global, :history_stop]
                - positions[loop_global, None],
                dim=2,
            )
            looped = distances.amin(dim=1) <= params.loop_radius
            looped_global = loop_global[looped]
            if looped_global.numel():
                active[looped_global] = False
                stop_codes[looped_global] = stop_loop

    path_history_np = path_history.detach().cpu().numpy()
    path_counts_np = path_counts.detach().cpu().numpy()
    accepted_np = accepted.detach().cpu().numpy()
    state_confidence_np = state_confidence.detach().cpu().numpy()
    stop_codes_np = stop_codes.detach().cpu().numpy()
    seeds_np = seeds.detach().cpu().numpy()
    image_indices_np = image_indices.detach().cpu().numpy()
    stop_names = {
        stop_invalid: "invalid_seed",
        stop_base: "base",
        stop_lost: "lost",
        stop_bounds: "bounds",
        stop_max_length: "max_length",
        stop_loop: "loop",
        stop_max_steps: "max_steps",
    }
    base_peaks_by_image: list[np.ndarray] = []
    for image_index in range(batch_size):
        fields = host_fields[image_index]
        peak_support = peak_support_by_image[image_index]
        base_heat = (
            np.where(peak_support, fields[2], 0.0)
            if peak_support is not None
            else fields[2]
        )
        base_peaks_by_image.append(
            find_peaks(base_heat, params.base_threshold, 3)
        )

    for index in range(count):
        path_count = int(path_counts_np[index])
        if not accepted_np[index] or path_count < 2:
            continue
        stop_reason = stop_names[int(stop_codes_np[index])]
        if stop_reason == "lost" and not params.emit_lost:
            continue
        tip_to_base = path_history_np[index, :path_count].copy()
        length = float(np.linalg.norm(np.diff(tip_to_base, axis=0), axis=1).sum())
        if length < params.min_path_px:
            continue
        base_to_tip = tip_to_base[::-1].copy()
        image_index = int(image_indices_np[index])
        base_peaks = base_peaks_by_image[image_index]
        base_score = 0.0
        if len(base_peaks):
            distances = np.linalg.norm(base_peaks[:, :2] - base_to_tip[0], axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= params.base_hit_radius:
                base_score = float(base_peaks[nearest, 2])
        arm = {
            "polyline": _smooth_resample(
                base_to_tip,
                params.smooth_window,
                params.resample_spacing,
            ),
            "tip_score": float(tips[index, 2]),
            "base_score": base_score,
            "mean_step_cost": float(1.0 - state_confidence_np[index]),
            "mean_identity_dist": 0.0,
            "length_px": length,
            "island_frac": 1.0,
            "identity": seeds_np[index].copy(),
            "state_confidence": float(state_confidence_np[index]),
            "stop_reason": stop_reason,
            "steps": path_count - 1,
        }
        outputs[image_index].append(arm)
    if params.suppress_duplicates and params.embedding_dim:
        outputs = [
            _suppress_duplicates_and_fragments(image_arms, EmbedDecodeParams())
            for image_arms in outputs
        ]
    return outputs


@torch.no_grad()
def trace_arms_noodle(
    raw: np.ndarray | torch.Tensor,
    image: np.ndarray | torch.Tensor,
    model: NoodleTracer,
    params: NoodleTraceParams | None = None,
) -> list[dict]:
    """Trace one image; implemented as a one-item batch for exact API parity."""
    raw_t = torch.as_tensor(raw)
    image_t = torch.as_tensor(image)
    if raw_t.ndim != 3:
        raise ValueError("raw must be (C,H,W)")
    if image_t.ndim != 3:
        raise ValueError("image must be (3,H,W) or (H,W,3)")
    return trace_arms_noodle_batch(
        raw_t[None],
        image_t[None],
        model,
        params,
    )[0]


def save_noodle_tracer_checkpoint(
    path: str,
    model: NoodleTracer,
    *,
    trace_params: NoodleTraceParams,
    metadata: dict,
    training_state: dict | None = None,
) -> None:
    payload = {
        "schema": 2,
        "model": model.state_dict(),
        "model_config": model.checkpoint_config(),
        "trace_params": asdict(trace_params),
        "metadata": metadata,
    }
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, path)


def load_noodle_tracer_checkpoint(
    path: str,
    *,
    device: str | torch.device = "cpu",
) -> tuple[NoodleTracer, NoodleTraceParams, dict, dict | None]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("schema", -1)) != 2:
        raise ValueError("not a noodle tracer checkpoint")
    model_config = dict(checkpoint["model_config"])
    model_config.setdefault("context_mode", "single")
    model_config.setdefault("context_dilation", 2)
    model = NoodleTracer(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    trace_config = dict(checkpoint["trace_params"])
    trace_config.setdefault("context_mode", "single")
    trace_config.setdefault("context_dilation", 2)
    params = NoodleTraceParams(**trace_config)
    return (
        model,
        params,
        dict(checkpoint.get("metadata", {})),
        checkpoint.get("training_state"),
    )
