# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Crop Hydra inputs from recorder StarSeg JSONL. Never runs YOLO."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from deepstarnoodles.crop import (
    SUNFLOWER_NAMES,
    CropConfig,
    CropResult,
    crop_from_xyxy,
    select_focal_detection,
)
from deepstarnoodles.roi import ROI, point_inside


@dataclass(frozen=True)
class SidecarCrop:
    crop: CropResult
    source_index: int


def starseg_frames_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}_starseg_frames.jsonl")


def starseg_summary_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}_starseg_summary.json")


def starseg_sidecar_status(video: Path) -> tuple[str, str]:
    """Return (status, detail). ``ok`` means nonempty per-frame inference exists."""
    jsonl = starseg_frames_path(video)
    summary = starseg_summary_path(video)
    if not jsonl.is_file():
        return "missing_starseg", f"missing {jsonl.name}"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return "malformed_starseg", str(exc)
        frames = int(payload.get("frames") or 0)
        if frames <= 0:
            return "empty_starseg", "starseg summary has frames=0"
        return "ok", ""
    with jsonl.open() as handle:
        for line in handle:
            if line.strip():
                return "ok", ""
    return "empty_starseg", "starseg jsonl is empty"


def iter_starseg_rows(jsonl: Path) -> Iterator[dict[str, Any]]:
    with jsonl.open() as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def load_starseg_rows(jsonl: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in iter_starseg_rows(jsonl):
        rows[int(row["frame_index"])] = row
    return rows


def sunflower_detections(
    row: dict[str, Any],
    roi: ROI | None = None,
) -> list[dict[str, Any]]:
    dets: list[dict[str, Any]] = []
    for index, det in enumerate(row.get("detections") or []):
        name = str(det.get("class_name") or "")
        if name not in SUNFLOWER_NAMES:
            continue
        box = det.get("box_xyxy") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        segment = det.get("segment_xy") or []
        polygon = np.asarray(segment, dtype=np.float32) if len(segment) >= 3 else None
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        test_point = center
        if polygon is not None:
            moments = cv2.moments(polygon)
            if moments["m00"]:
                test_point = (
                    float(moments["m10"] / moments["m00"]),
                    float(moments["m01"] / moments["m00"]),
                )
        if roi is not None and not point_inside(roi, test_point):
            continue
        dets.append(
            {
                "xyxy": (x1, y1, x2, y2),
                "center": center,
                "area": max(1.0, x2 - x1) * max(1.0, y2 - y1),
                "confidence": float(det.get("confidence") or 0.0),
                "polygon": polygon,
                "source_index": index,
            }
        )
    return dets


class SidecarCropper:
    """Focal-animal cropper driven by stored StarSeg boxes."""

    def __init__(self, config: CropConfig = CropConfig()) -> None:
        self.config = config
        self._last_center: tuple[float, float] | None = None

    def reset(self) -> None:
        self._last_center = None

    def crop(
        self,
        frame_bgr: np.ndarray,
        row: dict[str, Any] | None,
        roi: ROI | None = None,
    ) -> SidecarCrop | None:
        if row is None:
            return None
        height, width = frame_bgr.shape[:2]
        dets = sunflower_detections(row, roi)
        chosen = select_focal_detection(
            dets,
            math.hypot(width, height),
            self._last_center,
            self.config.track_max_jump_frac,
        )
        if chosen is None:
            return None
        self._last_center = chosen["center"]
        crop = crop_from_xyxy(
            frame_bgr,
            chosen["xyxy"],
            roi=roi,
            context_scale=self.config.context_scale,
            confidence=chosen["confidence"],
            mask_polygon=chosen["polygon"],
        )
        return SidecarCrop(crop=crop, source_index=int(chosen["source_index"]))
