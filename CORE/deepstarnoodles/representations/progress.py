# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Hydra progress activation, canonical rasterization, and window adapters."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


SOURCE_AXIS_CONVENTION = "pixel_xy:x_right_y_down"
WORLD_AFFINE_CONVENTION = "column_vector:world_xy=A@source_xy1"
IDENTITY_SOURCE_TO_WORLD_AFFINE = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def canonical_recipe_digest(payload: object) -> str:
    """Return a deterministic SHA-256 over a JSON-safe recipe."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_integer(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class ProgressRasterConfig:
    """Frozen recipe for support-masked, world-axis progress rasters.

    ``source_to_world_affine`` is deliberately supplied with each
    :class:`ProgressCanonicalizationInput`: an identity transform means source
    pixel axes, not an assumed camera or aquarium calibration.
    """

    resolution: int = 64
    half_extent_characteristic_scales: float = 1.75
    hard_support_threshold: float = 0.5
    source_axis_convention: str = SOURCE_AXIS_CONVENTION
    world_affine_convention: str = WORLD_AFFINE_CONVENTION

    def __post_init__(self) -> None:
        _strict_integer("resolution", self.resolution, minimum=8)
        if self.resolution & (self.resolution - 1):
            raise ValueError("resolution must be a power of two")
        if _finite_number(
            "half_extent_characteristic_scales",
            self.half_extent_characteristic_scales,
        ) <= 0:
            raise ValueError("half_extent_characteristic_scales must be positive")
        threshold = _finite_number(
            "hard_support_threshold", self.hard_support_threshold
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("hard_support_threshold must lie in [0, 1]")
        if self.source_axis_convention != SOURCE_AXIS_CONVENTION:
            raise ValueError(
                f"source_axis_convention must be {SOURCE_AXIS_CONVENTION!r}"
            )
        if self.world_affine_convention != WORLD_AFFINE_CONVENTION:
            raise ValueError(
                f"world_affine_convention must be {WORLD_AFFINE_CONVENTION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolution": int(self.resolution),
            "half_extent_characteristic_scales": float(
                self.half_extent_characteristic_scales
            ),
            "hard_support_threshold": float(self.hard_support_threshold),
            "source_axis_convention": self.source_axis_convention,
            "world_affine_convention": self.world_affine_convention,
        }

    def recipe_digest(self) -> str:
        return canonical_recipe_digest(
            {"recipe": "progress_raster_v1", "config": self.as_dict()}
        )

    def digest(self) -> str:
        return self.recipe_digest()


@dataclass(frozen=True)
class ProgressWindowConfig:
    """Timestamp-window policy; context length remains model-checkpoint bound."""

    cadence_seconds: float = 1.0
    centered: bool = True
    minimum_valid_fraction: float = 0.9
    nearest_tolerance: float = 0.55
    output_stride: int = 1

    def __post_init__(self) -> None:
        if _finite_number("cadence_seconds", self.cadence_seconds) <= 0:
            raise ValueError("cadence_seconds must be positive")
        if not isinstance(self.centered, bool):
            raise ValueError("centered must be boolean")
        fraction = _finite_number(
            "minimum_valid_fraction", self.minimum_valid_fraction
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError("minimum_valid_fraction must lie in (0, 1]")
        if _finite_number("nearest_tolerance", self.nearest_tolerance) < 0:
            raise ValueError("nearest_tolerance must be nonnegative")
        _strict_integer("output_stride", self.output_stride, minimum=1)

    def offsets(self, context_frames: int) -> torch.Tensor:
        """Return centered or trailing offsets in seconds."""

        _strict_integer("context_frames", context_frames, minimum=2)
        if self.centered:
            indexes = torch.arange(
                -(context_frames // 2),
                context_frames - context_frames // 2,
                dtype=torch.float64,
            )
        else:
            indexes = torch.arange(
                -(context_frames - 1),
                1,
                dtype=torch.float64,
            )
        return indexes * float(self.cadence_seconds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cadence_seconds": float(self.cadence_seconds),
            "centered": self.centered,
            "minimum_valid_fraction": float(self.minimum_valid_fraction),
            "nearest_tolerance": float(self.nearest_tolerance),
            "output_stride": int(self.output_stride),
        }

    def recipe_digest(self) -> str:
        return canonical_recipe_digest(
            {"recipe": "progress_timestamp_window_v1", "config": self.as_dict()}
        )

    def digest(self) -> str:
        return self.recipe_digest()


@dataclass(frozen=True)
class ProgressHydraResult:
    """Activated support/progress tensors, preserving the input batch rank."""

    support: torch.Tensor
    progress: torch.Tensor
    valid: torch.Tensor
    hard_support: torch.Tensor


ProgressActivation = ProgressHydraResult
ProgressRaster = ProgressHydraResult


def progress_from_hydra(
    raw: torch.Tensor,
    config: ProgressRasterConfig = ProgressRasterConfig(),
) -> ProgressHydraResult:
    """Apply the Hydra support/progress contract to channels 0 and 4."""

    if not isinstance(raw, torch.Tensor):
        raise TypeError("raw must be a torch.Tensor")
    unbatched = raw.ndim == 3
    if unbatched:
        raw_batch = raw.unsqueeze(0)
    elif raw.ndim == 4:
        raw_batch = raw
    else:
        raise ValueError("raw must have shape [C,H,W] or [B,C,H,W]")
    if raw_batch.shape[1] < 5:
        raise ValueError("raw must contain at least five channels")
    if not raw_batch.is_floating_point():
        raise TypeError("raw must be floating point")
    if not torch.isfinite(raw_batch[:, (0, 4)]).all():
        raise ValueError("Hydra support and progress logits must be finite")

    support_probability_raw = torch.sigmoid(raw_batch[:, 0:1])
    hard_support = support_probability_raw > config.hard_support_threshold
    support = support_probability_raw * hard_support.to(raw_batch.dtype)
    progress = (
        torch.sigmoid(raw_batch[:, 4:5])
        * hard_support.to(raw_batch.dtype)
        * support
    )
    valid = hard_support.flatten(1).any(dim=1)
    if unbatched:
        return ProgressHydraResult(
            support=support[0],
            progress=progress[0],
            valid=valid[0],
            hard_support=hard_support[0],
        )
    return ProgressHydraResult(
        support=support,
        progress=progress,
        valid=valid,
        hard_support=hard_support,
    )


@dataclass(frozen=True)
class ProgressCanonicalizationInput:
    """Per-frame geometry used to render a fixed-scale canonical raster."""

    crop_left_top_side: Any
    model_size: int
    characteristic_scale: Any
    source_to_world_affine: Any = IDENTITY_SOURCE_TO_WORLD_AFFINE
    world_axis_name: str = "world_xy"

    def __post_init__(self) -> None:
        _strict_integer("model_size", self.model_size, minimum=1)
        if not isinstance(self.world_axis_name, str) or not self.world_axis_name:
            raise ValueError("world_axis_name must be a nonempty string")
        try:
            crops = torch.as_tensor(self.crop_left_top_side)
            scale = torch.as_tensor(self.characteristic_scale)
            affine = torch.as_tensor(
                self.source_to_world_affine, dtype=torch.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("canonicalization geometry must be numeric") from exc
        if crops.shape != (3,) and (
            crops.ndim != 2 or crops.shape[1:] != (3,)
        ):
            raise ValueError(
                "crop_left_top_side must have shape [3] or [B,3]"
            )
        if scale.ndim > 1:
            raise ValueError(
                "characteristic_scale must be scalar or have shape [B]"
            )
        if (
            scale.numel() == 0
            or not torch.isfinite(scale).all()
            or torch.any(scale <= 0)
        ):
            raise ValueError(
                "characteristic_scale must be finite and positive"
            )
        if affine.shape != (3, 3) and (
            affine.ndim != 3 or affine.shape[1:] != (3, 3)
        ):
            raise ValueError(
                "source_to_world_affine must have shape [3,3] or [B,3,3]"
            )
        if not torch.isfinite(affine).all():
            raise ValueError("source_to_world_affine must be finite")
        rows = affine.reshape(-1, 3, 3)
        expected = rows.new_tensor((0.0, 0.0, 1.0)).expand(len(rows), -1)
        if not torch.allclose(
            rows[:, 2], expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "source_to_world_affine must be affine, not projective"
            )
        if torch.any(torch.linalg.det(rows[:, :2, :2]).abs() <= 1e-12):
            raise ValueError(
                "source_to_world_affine must have a nondegenerate linear part"
            )

    @classmethod
    def source_axes(
        cls,
        crop_left_top_side: Any,
        *,
        model_size: int,
        characteristic_scale: Any,
    ) -> "ProgressCanonicalizationInput":
        """Use source pixel axes explicitly; this is not camera calibration."""

        return cls(
            crop_left_top_side=crop_left_top_side,
            model_size=model_size,
            characteristic_scale=characteristic_scale,
            source_to_world_affine=IDENTITY_SOURCE_TO_WORLD_AFFINE,
            world_axis_name="source_xy",
        )

    @classmethod
    def identity(
        cls,
        crop_left_top_side: Any,
        *,
        model_size: int,
        characteristic_scale: Any,
    ) -> "ProgressCanonicalizationInput":
        return cls.source_axes(
            crop_left_top_side,
            model_size=model_size,
            characteristic_scale=characteristic_scale,
        )


ProgressCanonicalizationGeometry = ProgressCanonicalizationInput


@dataclass(frozen=True)
class ProgressCanonicalizationResult:
    progress: torch.Tensor
    support: torch.Tensor
    valid: torch.Tensor
    centroid_model: torch.Tensor
    centroid_source: torch.Tensor
    centroid_world: torch.Tensor
    source_to_world_affine: torch.Tensor
    characteristic_scale: torch.Tensor
    world_axis_name: str
    recipe_digest: str


def _batched_geometry(
    geometry: ProgressCanonicalizationInput,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    crops = torch.as_tensor(
        geometry.crop_left_top_side, dtype=dtype, device=device
    )
    if crops.shape == (3,):
        crops = crops.unsqueeze(0).expand(batch, -1)
    if crops.shape != (batch, 3):
        raise ValueError("crop_left_top_side must have shape [3] or [B,3]")
    crop_valid = torch.isfinite(crops).all(dim=1) & (crops[:, 2] > 0)

    affine = torch.as_tensor(
        geometry.source_to_world_affine, dtype=dtype, device=device
    )
    if affine.shape == (3, 3):
        affine = affine.unsqueeze(0).expand(batch, -1, -1)
    if affine.shape != (batch, 3, 3):
        raise ValueError("source_to_world_affine must have shape [3,3] or [B,3,3]")
    if not torch.isfinite(affine).all():
        raise ValueError("source_to_world_affine must be finite")
    expected_last = affine.new_tensor((0.0, 0.0, 1.0)).expand(batch, -1)
    if not torch.allclose(affine[:, 2], expected_last, rtol=0.0, atol=1e-6):
        raise ValueError("source_to_world_affine must be affine, not projective")
    determinant = torch.linalg.det(affine[:, :2, :2])
    if torch.any(determinant.abs() <= torch.finfo(dtype).eps):
        raise ValueError("source_to_world_affine must have a nondegenerate linear part")

    scale = torch.as_tensor(
        geometry.characteristic_scale, dtype=dtype, device=device
    )
    if scale.ndim == 0:
        scale = scale.expand(batch)
    if scale.shape != (batch,):
        raise ValueError("characteristic_scale must be scalar or have shape [B]")
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("characteristic_scale must be finite and positive")
    return crops, crop_valid, affine, scale


def _canonicalization_geometry(
    canonicalization: ProgressCanonicalizationInput | None,
    *,
    crop_left_top_side: Any | None,
    model_size: int | None,
    source_to_world_affine: Any,
    characteristic_scale: Any | None,
) -> ProgressCanonicalizationInput:
    explicit = (
        crop_left_top_side is not None
        or model_size is not None
        or characteristic_scale is not None
        or source_to_world_affine is not None
    )
    if canonicalization is not None:
        if explicit:
            raise ValueError(
                "pass either canonicalization or explicit geometry arguments"
            )
        return canonicalization
    if crop_left_top_side is None or model_size is None or characteristic_scale is None:
        raise ValueError(
            "crop_left_top_side, model_size, and characteristic_scale are required"
        )
    return ProgressCanonicalizationInput(
        crop_left_top_side=crop_left_top_side,
        model_size=model_size,
        characteristic_scale=characteristic_scale,
        source_to_world_affine=(
            IDENTITY_SOURCE_TO_WORLD_AFFINE
            if source_to_world_affine is None
            else source_to_world_affine
        ),
        world_axis_name=(
            "source_xy" if source_to_world_affine is None else "world_xy"
        ),
    )


def canonicalize_progress(
    progress: torch.Tensor,
    support: torch.Tensor,
    canonicalization: ProgressCanonicalizationInput | None = None,
    *,
    crop_left_top_side: Any | None = None,
    model_size: int | None = None,
    source_to_world_affine: Any | None = None,
    characteristic_scale: Any | None = None,
    config: ProgressRasterConfig = ProgressRasterConfig(),
) -> ProgressCanonicalizationResult:
    """Recenter support and progress at fixed scale without intrinsic rotation."""

    if not isinstance(progress, torch.Tensor) or not isinstance(support, torch.Tensor):
        raise TypeError("progress and support must be torch tensors")
    unbatched = progress.ndim == 3
    if unbatched:
        progress_batch = progress.unsqueeze(0)
        support_batch = support.unsqueeze(0)
    elif progress.ndim == 4:
        progress_batch = progress
        support_batch = support
    else:
        raise ValueError("progress and support must have shape [1,H,W] or [B,1,H,W]")
    if progress_batch.shape != support_batch.shape:
        raise ValueError("progress and support shapes must match")
    if progress_batch.ndim != 4 or progress_batch.shape[1] != 1:
        raise ValueError("progress and support must contain one channel")
    if progress_batch.device != support_batch.device:
        raise ValueError("progress and support must share a device")
    if not progress_batch.is_floating_point() or not support_batch.is_floating_point():
        raise TypeError("progress and support must be floating point")
    if progress_batch.dtype != support_batch.dtype:
        raise TypeError("progress and support must share a dtype")
    if not torch.isfinite(progress_batch).all() or not torch.isfinite(support_batch).all():
        raise ValueError("progress and support must be finite")
    if torch.any(support_batch < 0):
        raise ValueError("support weights must be nonnegative")

    geometry = _canonicalization_geometry(
        canonicalization,
        crop_left_top_side=crop_left_top_side,
        model_size=model_size,
        source_to_world_affine=source_to_world_affine,
        characteristic_scale=characteristic_scale,
    )
    batch, _, height, width = progress_batch.shape
    dtype = progress_batch.dtype
    device = progress_batch.device
    crops, crop_valid, affine, scale = _batched_geometry(
        geometry, batch=batch, dtype=dtype, device=device
    )

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    mass = support_batch[:, 0].sum(dim=(-2, -1))
    safe_mass = mass.clamp_min(torch.finfo(dtype).eps)
    centroid_x = (support_batch[:, 0] * xs).sum(dim=(-2, -1)) / safe_mass
    centroid_y = (support_batch[:, 0] * ys).sum(dim=(-2, -1)) / safe_mass
    centroid_model = torch.stack((centroid_x, centroid_y), dim=1)
    support_valid = mass > 0
    valid = support_valid & crop_valid

    safe_crops = torch.where(
        crop_valid[:, None],
        crops,
        crops.new_tensor((0.0, 0.0, 1.0)),
    )
    crop_scale = safe_crops[:, 2] / float(geometry.model_size)
    centroid_source = safe_crops[:, :2] + centroid_model * crop_scale[:, None]
    homogeneous = torch.cat(
        (centroid_source, torch.ones(batch, 1, dtype=dtype, device=device)),
        dim=1,
    )
    centroid_world = torch.einsum("bij,bj->bi", affine, homogeneous)[:, :2]

    pixels_per_world = (
        0.5
        * float(config.resolution)
        / (float(config.half_extent_characteristic_scales) * scale)
    )
    forward = (
        affine[:, :2, :2]
        * crop_scale[:, None, None]
        * pixels_per_world[:, None, None]
    )
    inverse = torch.linalg.inv(forward)
    output_y, output_x = torch.meshgrid(
        torch.arange(config.resolution, dtype=dtype, device=device),
        torch.arange(config.resolution, dtype=dtype, device=device),
        indexing="ij",
    )
    output_center = 0.5 * (config.resolution - 1)
    output_delta = torch.stack(
        (output_x - output_center, output_y - output_center), dim=-1
    )
    model_delta = torch.einsum("bij,hwj->bhwi", inverse, output_delta)
    source_x = centroid_x[:, None, None] + model_delta[..., 0]
    source_y = centroid_y[:, None, None] + model_delta[..., 1]
    grid = torch.stack(
        (
            2.0 * (source_x + 0.5) / float(width) - 1.0,
            2.0 * (source_y + 0.5) / float(height) - 1.0,
        ),
        dim=-1,
    )
    canonical = F.grid_sample(
        torch.cat((support_batch, progress_batch), dim=1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).masked_fill(~valid[:, None, None, None], 0.0)
    nan_centroid = torch.full_like(centroid_model, float("nan"))
    centroid_model = torch.where(valid[:, None], centroid_model, nan_centroid)
    centroid_source = torch.where(valid[:, None], centroid_source, nan_centroid)
    centroid_world = torch.where(valid[:, None], centroid_world, nan_centroid)
    recipe = canonical_recipe_digest(
        {
            "recipe": "progress_canonicalization_v1",
            "raster": config.as_dict(),
            "world_axis_name": geometry.world_axis_name,
        }
    )

    if unbatched:
        return ProgressCanonicalizationResult(
            progress=canonical[0, 1:2],
            support=canonical[0, 0:1],
            valid=valid[0],
            centroid_model=centroid_model[0],
            centroid_source=centroid_source[0],
            centroid_world=centroid_world[0],
            source_to_world_affine=affine[0],
            characteristic_scale=scale[0],
            world_axis_name=geometry.world_axis_name,
            recipe_digest=recipe,
        )
    return ProgressCanonicalizationResult(
        progress=canonical[:, 1:2],
        support=canonical[:, 0:1],
        valid=valid,
        centroid_model=centroid_model,
        centroid_source=centroid_source,
        centroid_world=centroid_world,
        source_to_world_affine=affine,
        characteristic_scale=scale,
        world_axis_name=geometry.world_axis_name,
        recipe_digest=recipe,
    )


def canonicalize_progress_source_axes(
    progress: torch.Tensor,
    support: torch.Tensor,
    *,
    crop_left_top_side: Any,
    model_size: int,
    characteristic_scale: Any,
    config: ProgressRasterConfig = ProgressRasterConfig(),
) -> ProgressCanonicalizationResult:
    """Canonicalize in declared source axes using an explicit identity affine."""

    return canonicalize_progress(
        progress,
        support,
        ProgressCanonicalizationInput.source_axes(
            crop_left_top_side,
            model_size=model_size,
            characteristic_scale=characteristic_scale,
        ),
        config=config,
    )


canonicalize_progress_raster = canonicalize_progress


def canonicalize_hydra_progress(
    raw: torch.Tensor,
    canonicalization: ProgressCanonicalizationInput | None = None,
    *,
    crop_left_top_side: Any | None = None,
    model_size: int | None = None,
    source_to_world_affine: Any | None = None,
    characteristic_scale: Any | None = None,
    config: ProgressRasterConfig = ProgressRasterConfig(),
) -> ProgressCanonicalizationResult:
    """Activate and canonicalize native Hydra channels 0 and 4 in one call."""

    activated = progress_from_hydra(raw, config)
    return canonicalize_progress(
        activated.progress,
        activated.support,
        canonicalization,
        crop_left_top_side=crop_left_top_side,
        model_size=model_size,
        source_to_world_affine=source_to_world_affine,
        characteristic_scale=characteristic_scale,
        config=config,
    )


def estimate_characteristic_scale(
    support: torch.Tensor,
    *,
    crop_left_top_side: Any,
    model_size: int,
    source_to_world_affine: Any = IDENTITY_SOURCE_TO_WORLD_AFFINE,
    frame_valid: torch.Tensor | None = None,
    quantile: float = 0.9,
    minimum_support_pixels: int = 2,
) -> float:
    """Estimate median hard-support radial-q90 in affine world units.

    The affine contributes its area-equivalent linear scale
    ``sqrt(abs(det(A[:2,:2])))``, matching the research calibration recipe.
    """

    _strict_integer("model_size", model_size, minimum=1)
    _strict_integer(
        "minimum_support_pixels", minimum_support_pixels, minimum=1
    )
    quantile = _finite_number("quantile", quantile)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    if support.ndim == 2:
        support_batch = support.unsqueeze(0).unsqueeze(0)
    elif support.ndim == 3:
        support_batch = support.unsqueeze(1)
    elif support.ndim == 4:
        support_batch = support
    else:
        raise ValueError(
            "support must have shape [H,W], [F,H,W], or [F,1,H,W]"
        )
    if support_batch.shape[1] != 1:
        raise ValueError("support must contain exactly one channel")
    if not torch.isfinite(support_batch).all():
        raise ValueError("support must be finite")
    frames = len(support_batch)
    geometry = ProgressCanonicalizationInput(
        crop_left_top_side=crop_left_top_side,
        model_size=model_size,
        characteristic_scale=1.0,
        source_to_world_affine=source_to_world_affine,
    )
    crops, crop_valid, affine, _ = _batched_geometry(
        geometry,
        batch=frames,
        dtype=torch.float64,
        device=support_batch.device,
    )
    if frame_valid is None:
        selected_frames = torch.ones(
            frames, dtype=torch.bool, device=support_batch.device
        )
    else:
        selected_frames = torch.as_tensor(
            frame_valid, dtype=torch.bool, device=support_batch.device
        )
        if selected_frames.shape != (frames,):
            raise ValueError("frame_valid must have shape [F]")
    selected_frames &= crop_valid

    per_frame: list[torch.Tensor] = []
    for index in torch.nonzero(selected_frames, as_tuple=False).flatten().tolist():
        y, x = torch.where(support_batch[index, 0] > 0)
        if x.numel() < minimum_support_pixels:
            continue
        xy = torch.stack((x, y), dim=1).to(torch.float64) + 0.5
        centroid = xy.mean(dim=0)
        radius = torch.linalg.vector_norm(xy - centroid, dim=1)
        q90_model = torch.quantile(radius, quantile)
        crop_scale = crops[index, 2] / float(model_size)
        area_scale = torch.sqrt(
            torch.abs(torch.linalg.det(affine[index, :2, :2]))
        )
        per_frame.append(q90_model * crop_scale * area_scale)
    if not per_frame:
        raise ValueError("no valid support frame can estimate characteristic scale")
    estimate = torch.quantile(torch.stack(per_frame), 0.5)
    if not torch.isfinite(estimate) or estimate <= 0:
        raise ValueError("estimated characteristic scale is not positive")
    return float(estimate.cpu())


def _as_sequence_mapping(
    progress: torch.Tensor | Mapping[str, torch.Tensor],
    support: torch.Tensor | None,
    frame_valid: torch.Tensor | None,
    timestamps: torch.Tensor | None,
) -> Mapping[str, torch.Tensor]:
    if isinstance(progress, Mapping):
        if support is not None or frame_valid is not None or timestamps is not None:
            raise ValueError("separate tensors cannot accompany a mapping source")
        return progress
    if support is None or frame_valid is None:
        raise ValueError("support and frame_valid are required")
    result = {
        "progress": progress,
        "support": support,
        "frame_valid": frame_valid,
    }
    if timestamps is not None:
        result["timestamps"] = timestamps
    return result


class ProgressWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Adapt one or more canonical sequences into accepted contiguous windows."""

    def __init__(
        self,
        progress: (
            torch.Tensor
            | Mapping[str, torch.Tensor]
            | Sequence[Mapping[str, torch.Tensor]]
        ),
        support: torch.Tensor | None = None,
        frame_valid: torch.Tensor | None = None,
        *,
        timestamps: torch.Tensor | None = None,
        context_frames: int = 60,
        config: ProgressWindowConfig = ProgressWindowConfig(),
    ) -> None:
        _strict_integer("context_frames", context_frames, minimum=2)
        if isinstance(progress, Sequence) and not isinstance(
            progress, (torch.Tensor, Mapping)
        ):
            if support is not None or frame_valid is not None or timestamps is not None:
                raise ValueError("separate tensors cannot accompany sequence mappings")
            sources = tuple(progress)
        else:
            sources = (
                _as_sequence_mapping(
                    progress,  # type: ignore[arg-type]
                    support,
                    frame_valid,
                    timestamps,
                ),
            )
        if not sources:
            raise ValueError("at least one canonical sequence is required")
        self.context_frames = context_frames
        self.config = config
        self.sequences: tuple[dict[str, torch.Tensor], ...] = tuple(
            self._validate_source(source) for source in sources
        )

        positions: list[tuple[int, int]] = []
        minimum = math.ceil(config.minimum_valid_fraction * context_frames)
        for sequence_index, source in enumerate(self.sequences):
            count = len(source["frame_valid"])
            for start in range(
                0,
                max(count - context_frames + 1, 0),
                config.output_stride,
            ):
                stop = start + context_frames
                deltas = (
                    source["timestamps"][start + 1 : stop]
                    - source["timestamps"][start : stop - 1]
                )
                cadence_error = (
                    deltas - float(config.cadence_seconds)
                ).abs()
                if torch.any(cadence_error > config.nearest_tolerance):
                    continue
                if int(source["frame_valid"][start:stop].sum()) >= minimum:
                    positions.append((sequence_index, start))
        self.positions = tuple(positions)

    def _validate_source(
        self, source: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        missing = {"progress", "support", "frame_valid"} - set(source)
        if missing:
            raise ValueError(f"canonical sequence is missing {sorted(missing)}")
        progress = torch.as_tensor(source["progress"])
        support = torch.as_tensor(source["support"])
        valid = torch.as_tensor(source["frame_valid"], dtype=torch.bool)
        if progress.ndim != 4 or progress.shape[1] != 1:
            raise ValueError("progress must have shape [F,1,H,W]")
        if support.shape != progress.shape:
            raise ValueError("support must match progress")
        if valid.shape != (len(progress),):
            raise ValueError("frame_valid must have shape [F]")
        if "timestamps" in source:
            timestamp = torch.as_tensor(source["timestamps"], dtype=torch.float64)
        elif "timestamp" in source:
            timestamp = torch.as_tensor(source["timestamp"], dtype=torch.float64)
        else:
            timestamp = (
                torch.arange(len(progress), dtype=torch.float64)
                * self.config.cadence_seconds
            )
        if timestamp.shape != (len(progress),):
            raise ValueError("timestamps must have shape [F]")
        if not torch.isfinite(timestamp).all():
            raise ValueError("timestamps must be finite")
        if len(timestamp) > 1 and not torch.all(timestamp[1:] > timestamp[:-1]):
            raise ValueError("timestamps must be strictly increasing")
        return {
            "progress": progress,
            "support": support,
            "frame_valid": valid,
            "timestamps": timestamp,
        }

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence_index, start = self.positions[index]
        stop = start + self.context_frames
        source = self.sequences[sequence_index]
        valid = source["frame_valid"][start:stop]
        return {
            "progress": source["progress"][start:stop],
            "support": source["support"][start:stop],
            "frame_valid": valid,
            "timestamps": source["timestamps"][start:stop],
            "window_valid_fraction": valid.to(torch.float32).mean(),
            "sequence_index": torch.tensor(sequence_index, dtype=torch.int64),
            "start_position": torch.tensor(start, dtype=torch.int64),
        }


__all__ = [
    "IDENTITY_SOURCE_TO_WORLD_AFFINE",
    "ProgressActivation",
    "ProgressCanonicalizationGeometry",
    "ProgressCanonicalizationInput",
    "ProgressCanonicalizationResult",
    "ProgressHydraResult",
    "ProgressRaster",
    "ProgressRasterConfig",
    "ProgressWindowConfig",
    "ProgressWindowDataset",
    "SOURCE_AXIS_CONVENTION",
    "WORLD_AFFINE_CONVENTION",
    "canonical_recipe_digest",
    "canonicalize_hydra_progress",
    "canonicalize_progress",
    "canonicalize_progress_raster",
    "canonicalize_progress_source_axes",
    "estimate_characteristic_scale",
    "progress_from_hydra",
]
