# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Canonical 64x64 progress-raster cache extracted from the timelapse archive.

The archive Hydra sidecars keep landmarks and embeddings only, so the temporal
VAE needs a second, lighter pass that re-runs Hydra on each StarSeg crop and
keeps only the support-masked progress raster after canonicalization. One
``.npz`` per clip is written under ``DATA/deepstarnoodles/progress_cache``.
"""
from __future__ import annotations

import json
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from deepstarnoodles.batch_library import Clip, InventoryIssue, inventory, utc_now
from deepstarnoodles.crop import CropConfig
from deepstarnoodles.hydra_append import HydraBatchPredictor, empty_cuda_cache
from deepstarnoodles.paths import DATA_ROOT
from deepstarnoodles.recorder_guard import RecorderGuard
from deepstarnoodles.representations.progress import (
    ProgressCanonicalizationInput,
    ProgressRasterConfig,
    canonical_recipe_digest,
    canonicalize_progress,
    estimate_characteristic_scale,
    progress_from_hydra,
)
from deepstarnoodles.sidecar_crop import (
    SidecarCropper,
    load_starseg_rows,
    starseg_frames_path,
    starseg_sidecar_status,
)

CACHE_SCHEMA_VERSION = 1
DEFAULT_PROGRESS_CACHE_ROOT = DATA_ROOT / "deepstarnoodles" / "progress_cache"


def progress_cache_path(clip: Clip, root: Path = DEFAULT_PROGRESS_CACHE_ROOT) -> Path:
    return root / clip.camera / clip.trial / f"{clip.stem}_progress64.npz"


def cache_recipe(checkpoint_sha: str, config: ProgressRasterConfig) -> dict[str, Any]:
    return {
        "recipe": "archive_progress_cache_v1",
        "schema": CACHE_SCHEMA_VERSION,
        "hydra_checkpoint_sha256": checkpoint_sha,
        "crop": "starseg_sidecar/2x_square/roi_mask",
        "raster": config.as_dict(),
        "characteristic_scale": "per_clip_median_hard_support_q90",
        "world_axis_name": "source_xy",
        "quantization": "uint8_round_255",
    }


def cache_recipe_digest(checkpoint_sha: str, config: ProgressRasterConfig) -> str:
    return canonical_recipe_digest(cache_recipe(checkpoint_sha, config))


def is_cache_complete(clip: Clip, expected_digest: str, root: Path) -> bool:
    path = progress_cache_path(clip, root)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                str(payload["recipe_digest"]) == expected_digest
                and str(payload["status"]) == "complete"
            )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def inventory_cache_sources(
    source_root: Path,
    *,
    expected_digest: str,
    cache_root: Path = DEFAULT_PROGRESS_CACHE_ROOT,
) -> tuple[list[Clip], list[InventoryIssue], list[Clip]]:
    clips, issues = inventory(source_root)
    eligible: list[Clip] = []
    complete: list[Clip] = []
    skipped = list(issues)
    for clip in clips:
        status, detail = starseg_sidecar_status(clip.video)
        if status != "ok":
            skipped.append(InventoryIssue(clip.key, str(clip.video), status, detail))
            continue
        if is_cache_complete(clip, expected_digest, cache_root):
            complete.append(clip)
            continue
        eligible.append(clip)
    return eligible, skipped, complete


@dataclass
class ExtractResult:
    key: str
    status: str
    frames: int
    detected: int
    valid: int
    characteristic_scale: float | None
    path: str
    error: str | None = None


def _quantize(values: torch.Tensor) -> np.ndarray:
    return (
        values.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    )


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_progress_cache(
    clip: Clip,
    predictor: HydraBatchPredictor,
    *,
    checkpoint_sha: str,
    cache_root: Path = DEFAULT_PROGRESS_CACHE_ROOT,
    batch_size: int = 4,
    device: str = "cpu",
    guard: RecorderGuard | None = None,
    max_frames: int | None = None,
    raster_config: ProgressRasterConfig = ProgressRasterConfig(),
    crop_config: CropConfig = CropConfig(),
    canonicalize_batch: int = 64,
) -> ExtractResult:
    """Run Hydra on every StarSeg crop and cache canonical 64x64 rasters.

    Activated support/progress fields are held on the CPU in float16 for the
    whole clip because the characteristic scale is a per-clip statistic that
    must be known before any frame can be canonicalized.
    """

    rows_by_index = load_starseg_rows(starseg_frames_path(clip.video))
    capture = cv2.VideoCapture(str(clip.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {clip.video}")
    cropper = SidecarCropper(crop_config)
    stamps = clip.metadata.get("frame_timestamps") or []
    model_size = int(predictor.image_size)

    frame_ts: list[float] = []
    detected: list[bool] = []
    crop_lts: list[tuple[float, float, float]] = []
    fields: list[torch.Tensor] = []  # [n,2,S,S] float16 (support, progress) for detected frames
    pending: list[Any] = []

    def flush() -> None:
        if not pending:
            return
        crops = [item.crop.crop_rgb for item in pending]
        raw, _activated = predictor.predict_raw_batch(crops)
        activated = progress_from_hydra(raw.float(), raster_config)
        fields.append(
            torch.cat((activated.support, activated.progress), dim=1)
            .to(torch.float16)
            .cpu()
        )
        pending.clear()

    frame_index = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if guard is not None:
                guard.yield_if_needed(empty_cuda_cache)
            row = rows_by_index.get(frame_index)
            ts = float("nan")
            if row is not None and row.get("frame_ts") is not None:
                ts = float(row["frame_ts"])
            elif frame_index < len(stamps):
                ts = float(stamps[frame_index])
            frame_ts.append(ts)
            sidecar = cropper.crop(frame_bgr, row, clip.roi)
            if sidecar is None:
                detected.append(False)
                crop_lts.append((0.0, 0.0, 1.0))
            else:
                detected.append(True)
                crop = sidecar.crop
                crop_lts.append(
                    (float(crop.crop_left), float(crop.crop_top), float(crop.crop_side))
                )
                pending.append(sidecar)
                if len(pending) >= batch_size:
                    flush()
            frame_index += 1
        if pending:
            if guard is not None:
                guard.yield_if_needed(empty_cuda_cache)
            flush()
    finally:
        capture.release()

    frames = len(frame_ts)
    detected_np = np.asarray(detected, dtype=bool)
    crop_np = np.asarray(crop_lts, dtype=np.float64).reshape(frames, 3)
    ts_np = np.asarray(frame_ts, dtype=np.float64)
    resolution = raster_config.resolution
    progress_u8 = np.zeros((frames, resolution, resolution), dtype=np.uint8)
    support_u8 = np.zeros_like(progress_u8)
    valid_np = np.zeros(frames, dtype=bool)
    centroid_np = np.full((frames, 2), np.nan, dtype=np.float64)
    scale: float | None = None

    if fields:
        stacked = torch.cat(fields, dim=0)  # [n,2,S,S] float16
        detected_positions = np.flatnonzero(detected_np)
        detected_crops = torch.from_numpy(crop_np[detected_positions])
        try:
            scale = estimate_characteristic_scale(
                stacked[:, 0:1].float(),
                crop_left_top_side=detected_crops,
                model_size=model_size,
            )
        except ValueError:
            scale = None
        if scale is not None:
            geometry_device = torch.device(device if device.startswith("cuda") else "cpu")
            for start in range(0, len(stacked), canonicalize_batch):
                stop = min(start + canonicalize_batch, len(stacked))
                chunk = stacked[start:stop].to(geometry_device, dtype=torch.float32)
                geometry = ProgressCanonicalizationInput.source_axes(
                    detected_crops[start:stop].to(geometry_device, dtype=torch.float32),
                    model_size=model_size,
                    characteristic_scale=scale,
                )
                result = canonicalize_progress(
                    chunk[:, 1:2], chunk[:, 0:1], geometry, config=raster_config
                )
                positions = detected_positions[start:stop]
                progress_u8[positions] = _quantize(result.progress[:, 0])
                support_u8[positions] = _quantize(result.support[:, 0])
                valid_np[positions] = result.valid.cpu().numpy()
                centroid_np[positions] = result.centroid_source.double().cpu().numpy()
                del chunk, result
        del stacked
    empty_cuda_cache()

    recipe_digest = cache_recipe_digest(checkpoint_sha, raster_config)
    path = progress_cache_path(clip, cache_root)
    status = "complete" if max_frames is None else "smoke"
    _atomic_savez(
        path,
        schema_version=np.int64(CACHE_SCHEMA_VERSION),
        status=np.str_(status),
        key=np.str_(clip.key),
        camera=np.str_(clip.camera),
        trial=np.str_(clip.trial),
        stem=np.str_(clip.stem),
        source=np.str_(str(clip.video)),
        recipe_digest=np.str_(recipe_digest),
        recipe_json=np.str_(json.dumps(cache_recipe(checkpoint_sha, raster_config), sort_keys=True)),
        hydra_checkpoint_sha256=np.str_(checkpoint_sha),
        model_size=np.int64(model_size),
        characteristic_scale=np.float64(np.nan if scale is None else scale),
        updated_at=np.str_(utc_now()),
        frame_index=np.arange(frames, dtype=np.int64),
        frame_ts=ts_np,
        detected=detected_np,
        frame_valid=valid_np,
        crop_left_top_side=crop_np,
        centroid_source=centroid_np,
        progress=progress_u8,
        support=support_u8,
    )
    return ExtractResult(
        key=clip.key,
        status=status,
        frames=frames,
        detected=int(detected_np.sum()),
        valid=int(valid_np.sum()),
        characteristic_scale=scale,
        path=str(path),
    )


@dataclass(frozen=True)
class ProgressCacheClip:
    """Loaded per-clip cache with real timestamps (not yet on a uniform grid)."""

    key: str
    camera: str
    trial: str
    stem: str
    path: Path
    frame_ts: np.ndarray
    frame_valid: np.ndarray
    progress: np.ndarray  # [F,H,W] uint8
    support: np.ndarray  # [F,H,W] uint8
    characteristic_scale: float
    recipe_digest: str

    @property
    def frames(self) -> int:
        return len(self.frame_ts)


def load_progress_cache(path: Path) -> ProgressCacheClip:
    with np.load(path, allow_pickle=False) as payload:
        return ProgressCacheClip(
            key=str(payload["key"]),
            camera=str(payload["camera"]),
            trial=str(payload["trial"]),
            stem=str(payload["stem"]),
            path=Path(path),
            frame_ts=np.asarray(payload["frame_ts"], dtype=np.float64),
            frame_valid=np.asarray(payload["frame_valid"], dtype=bool),
            progress=np.asarray(payload["progress"], dtype=np.uint8),
            support=np.asarray(payload["support"], dtype=np.uint8),
            characteristic_scale=float(payload["characteristic_scale"]),
            recipe_digest=str(payload["recipe_digest"]),
        )


@dataclass(frozen=True)
class GridSequence:
    """A clip resampled onto a uniform cadence grid, ready for windowing."""

    key: str
    camera: str
    trial: str
    timestamps: np.ndarray  # [G] float64, uniform cadence
    source_index: np.ndarray  # [G] int64 nearest cache frame per grid slot
    frame_valid: np.ndarray  # [G] bool

    def tensors(self, cache: ProgressCacheClip) -> dict[str, torch.Tensor]:
        index = torch.from_numpy(self.source_index)
        progress = torch.from_numpy(cache.progress).index_select(0, index).unsqueeze(1)
        support = torch.from_numpy(cache.support).index_select(0, index).unsqueeze(1)
        valid = torch.from_numpy(self.frame_valid)
        zero = ~valid[:, None, None, None]
        return {
            "progress": progress.masked_fill(zero, 0),
            "support": support.masked_fill(zero, 0),
            "frame_valid": valid,
            "timestamps": torch.from_numpy(self.timestamps),
        }


def resample_to_grid(
    cache: ProgressCacheClip,
    *,
    cadence_seconds: float = 1.0,
    nearest_tolerance: float = 0.55,
) -> GridSequence:
    """Snap real frame timestamps onto a uniform grid by nearest neighbour.

    The recorder targets 1 fps but delivers ~1.025 s between frames; a
    centered 60 s window indexed by raw frames would therefore drift ~0.75 s
    at its edges. Resampling on a true-time grid keeps training windows and
    ``ProgressSequenceEncoder`` inference on the same clock.
    """

    finite = np.isfinite(cache.frame_ts)
    if not finite.any():
        raise ValueError(f"{cache.key}: no finite timestamps")
    ts = np.where(finite, cache.frame_ts, np.nan)
    t0 = float(np.nanmin(ts))
    t1 = float(np.nanmax(ts))
    count = int(np.floor((t1 - t0) / cadence_seconds)) + 1
    grid = t0 + cadence_seconds * np.arange(count, dtype=np.float64)
    order = np.flatnonzero(finite)
    sorted_ts = ts[order]
    right = np.searchsorted(sorted_ts, grid)
    left = np.clip(right - 1, 0, len(sorted_ts) - 1)
    right = np.clip(right, 0, len(sorted_ts) - 1)
    choose_right = np.abs(sorted_ts[right] - grid) < np.abs(sorted_ts[left] - grid)
    nearest = np.where(choose_right, right, left)
    source = order[nearest]
    present = np.abs(ts[source] - grid) <= nearest_tolerance
    valid = present & cache.frame_valid[source]
    return GridSequence(
        key=cache.key,
        camera=cache.camera,
        trial=cache.trial,
        timestamps=grid,
        source_index=source.astype(np.int64),
        frame_valid=valid,
    )


def inventory_progress_cache(root: Path = DEFAULT_PROGRESS_CACHE_ROOT) -> list[Path]:
    return sorted(root.glob("*/*/*_progress64.npz"))


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_PROGRESS_CACHE_ROOT",
    "ExtractResult",
    "GridSequence",
    "ProgressCacheClip",
    "cache_recipe",
    "cache_recipe_digest",
    "extract_progress_cache",
    "inventory_cache_sources",
    "inventory_progress_cache",
    "is_cache_complete",
    "load_progress_cache",
    "progress_cache_path",
    "resample_to_grid",
]
