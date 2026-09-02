# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Offline Hydra-only append beside recorder StarSeg sidecars."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from deepstarnoodles.batch_library import Clip, InventoryIssue, atomic_json, inventory, utc_now
from deepstarnoodles.crop import CropConfig
from deepstarnoodles.hydra_landmarks import CHANNEL_CONTRACT, summarize_activated
from deepstarnoodles.paths import DEFAULT_HYDRA_CHECKPOINT
from deepstarnoodles.recorder_guard import RecorderGuard
from deepstarnoodles.sidecar_crop import (
    SidecarCropper,
    load_starseg_rows,
    starseg_frames_path,
    starseg_sidecar_status,
    starseg_summary_path,
)
SCHEMA_VERSION = 1


class HydraBatchPredictor(Protocol):
    image_size: int

    def predict_raw_batch(
        self, crops_rgb: list[np.ndarray]
    ) -> tuple[Any, Any]:
        ...


def hydra_frames_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}_hydra_frames.jsonl")


def hydra_summary_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}_hydra_summary.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hydra_contract(checkpoint_sha: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "channel_contract": CHANNEL_CONTRACT,
        "crop": "starseg_sidecar/2x_square/roi_mask",
        "hydra_checkpoint_sha256": checkpoint_sha,
    }


def hydra_contract_hash(checkpoint_sha: str) -> str:
    body = json.dumps(hydra_contract(checkpoint_sha), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def _jsonl_line_count(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def expected_frame_count(clip: Clip) -> int:
    stamps = clip.metadata.get("frame_timestamps")
    if isinstance(stamps, list) and stamps:
        return len(stamps)
    summary_path = starseg_summary_path(clip.video)
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text())
            return int(payload.get("frames") or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    jsonl = starseg_frames_path(clip.video)
    return _jsonl_line_count(jsonl) if jsonl.is_file() else 0


def is_hydra_complete(clip: Clip, expected_hash: str) -> bool:
    frames_path = hydra_frames_path(clip.video)
    summary_path = hydra_summary_path(clip.video)
    if not frames_path.is_file() or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("status") != "complete":
        return False
    if summary.get("contract_hash") != expected_hash:
        return False
    expected = expected_frame_count(clip)
    if expected <= 0 or _jsonl_line_count(frames_path) != expected:
        return False
    metadata = clip.metadata
    return bool(metadata.get("hydra_frames_jsonl") and metadata.get("hydra_summary_json"))


def inventory_hydra_sources(
    source_root: Path,
    *,
    expected_hash: str,
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
        if is_hydra_complete(clip, expected_hash):
            complete.append(clip)
            continue
        eligible.append(clip)
    return eligible, skipped, complete


def apply_os_isolation(*, nice: int = 10, ionice_idle: bool = True) -> dict[str, Any]:
    applied = {"nice": None, "ionice": False}
    try:
        os.nice(nice)
        applied["nice"] = os.nice(0)
    except OSError:
        pass
    if ionice_idle:
        result = subprocess.run(
            ["ionice", "-c3", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
        )
        applied["ionice"] = result.returncode == 0
    return applied


def configure_cuda(device: str, memory_fraction: float | None) -> None:
    if not device.startswith("cuda"):
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    import torch

    if not torch.cuda.is_available():
        return
    index = 0
    if ":" in device:
        index = int(device.split(":")[1])
    if memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction), device=index)


def empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _flush_batch(
    pending: list[tuple[int, dict[str, Any], Any]],
    predictor: HydraBatchPredictor,
    checkpoint_sha: str,
    device: str,
) -> list[dict[str, Any]]:
    if not pending:
        return []
    crops = [item[2].crop.crop_rgb for item in pending]
    _raw, activated = predictor.predict_raw_batch(crops)
    activated_np = _tensor_to_numpy(activated)
    rows: list[dict[str, Any]] = []
    size = int(getattr(predictor, "image_size", activated_np.shape[-1]))
    for offset, (frame_index, base, sidecar) in enumerate(pending):
        crop = sidecar.crop
        summary = summarize_activated(
            activated_np[offset],
            left=crop.crop_left,
            top=crop.crop_top,
            side=crop.crop_side,
            image_size=size,
        )
        rows.append(
            {
                **base,
                "detected": True,
                "crop": {
                    "bbox_xyxy": list(crop.bbox_xyxy),
                    "left": crop.crop_left,
                    "top": crop.crop_top,
                    "side": crop.crop_side,
                    "confidence": crop.confidence,
                    "source_index": sidecar.source_index,
                },
                "landmarks": summary["landmarks"],
                "embedding": summary["embedding"],
                "embedding_sample": summary["embedding_sample"],
                "device": device,
                "hydra_checkpoint_sha256": checkpoint_sha,
                "channel_contract": CHANNEL_CONTRACT,
            }
        )
        del frame_index
    return rows


def _undetected_row(
    frame_index: int,
    frame_ts: float | None,
    checkpoint_sha: str,
    device: str,
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "frame_ts": frame_ts,
        "detected": False,
        "device": device,
        "hydra_checkpoint_sha256": checkpoint_sha,
        "channel_contract": CHANNEL_CONTRACT,
    }


def update_metadata_pointers(
    metadata_path: Path,
    *,
    frames_path: Path,
    summary_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text())
    metadata["hydra_frames_jsonl"] = str(frames_path)
    metadata["hydra_summary_json"] = str(summary_path)
    metadata["hydra_summary"] = {
        "status": summary.get("status"),
        "frames": summary.get("frames"),
        "detected": summary.get("detected"),
        "contract_hash": summary.get("contract_hash"),
        "channel_contract": summary.get("channel_contract"),
        "hydra_checkpoint_sha256": summary.get("hydra_checkpoint_sha256"),
    }
    atomic_json(metadata_path, metadata)
    return metadata


@dataclass
class AppendResult:
    key: str
    status: str
    frames: int
    detected: int
    error: str | None = None


def append_hydra_clip(
    clip: Clip,
    predictor: HydraBatchPredictor,
    *,
    checkpoint_sha: str,
    batch_size: int = 4,
    device: str = "cpu",
    guard: RecorderGuard | None = None,
    max_frames: int | None = None,
    crop_config: CropConfig = CropConfig(),
) -> AppendResult:
    rows_by_index = load_starseg_rows(starseg_frames_path(clip.video))
    capture = cv2.VideoCapture(str(clip.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {clip.video}")
    cropper = SidecarCropper(crop_config)
    pending: list[tuple[int, dict[str, Any], Any]] = []
    written: list[dict[str, Any]] = []
    stamps = clip.metadata.get("frame_timestamps") or []
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
            frame_ts = None
            if row is not None and row.get("frame_ts") is not None:
                frame_ts = float(row["frame_ts"])
            elif frame_index < len(stamps):
                frame_ts = float(stamps[frame_index])
            base = {"frame_index": frame_index, "frame_ts": frame_ts}
            sidecar = cropper.crop(frame_bgr, row, clip.roi)
            if sidecar is None:
                written.append(_undetected_row(frame_index, frame_ts, checkpoint_sha, device))
            else:
                pending.append((frame_index, base, sidecar))
                if len(pending) >= batch_size:
                    written.extend(_flush_batch(pending, predictor, checkpoint_sha, device))
                    pending.clear()
            frame_index += 1
        if pending:
            if guard is not None:
                guard.yield_if_needed(empty_cuda_cache)
            written.extend(_flush_batch(pending, predictor, checkpoint_sha, device))
    finally:
        capture.release()

    written.sort(key=lambda item: item["frame_index"])
    frames_path = hydra_frames_path(clip.video)
    summary_path = hydra_summary_path(clip.video)
    temporary = frames_path.with_name(f".{frames_path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        for row in written:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(temporary, frames_path)

    detected = sum(1 for row in written if row.get("detected"))
    complete = max_frames is None
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "smoke",
        "source": str(clip.video),
        "frames": len(written),
        "detected": detected,
        "channel_contract": CHANNEL_CONTRACT,
        "hydra_checkpoint_sha256": checkpoint_sha,
        "contract": hydra_contract(checkpoint_sha),
        "contract_hash": hydra_contract_hash(checkpoint_sha),
        "roi": clip.roi.as_dict(),
        "batch_size": batch_size,
        "device": device,
        "updated_at": utc_now(),
        "hydra_frames_jsonl": str(frames_path),
    }
    atomic_json(summary_path, summary)
    if complete:
        clip.metadata.update(
            update_metadata_pointers(
                clip.metadata_path,
                frames_path=frames_path,
                summary_path=summary_path,
                summary=summary,
            )
        )
    return AppendResult(clip.key, summary["status"], len(written), detected)


def default_checkpoint_sha(path: Path = DEFAULT_HYDRA_CHECKPOINT) -> str:
    return sha256_file(path) if path.is_file() else ""
