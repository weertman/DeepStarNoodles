# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Validated source-space ROI polygons and crop-space masking."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ROIError(ValueError):
    """Metadata does not contain a usable StarSeg ROI."""

    def __init__(self, status: str, message: str) -> None:
        self.status = status
        super().__init__(message)


@dataclass(frozen=True)
class ROI:
    vertices: np.ndarray
    sha256: str
    filter_policy: str

    def as_dict(self) -> dict:
        return {
            "absolute": self.vertices.tolist(),
            "sha256": self.sha256,
            "filter_policy": self.filter_policy,
        }


def _hash_vertices(vertices: np.ndarray) -> str:
    canonical = json.dumps(vertices.tolist(), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def parse_roi(metadata: dict[str, Any], width: int, height: int) -> ROI:
    raw_roi = metadata.get("starseg_roi")
    if not isinstance(raw_roi, dict) or "absolute" not in raw_roi:
        raise ROIError("missing_roi", "metadata is missing starseg_roi.absolute")
    raw = raw_roi["absolute"]
    if not isinstance(raw, list) or len(raw) < 3:
        raise ROIError("malformed_roi", "starseg_roi.absolute requires at least 3 vertices")
    try:
        vertices = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ROIError("malformed_roi", "ROI vertices must be numeric pairs") from exc
    if vertices.shape != (len(raw), 2) or not np.isfinite(vertices).all():
        raise ROIError("malformed_roi", "ROI vertices must be finite [x,y] pairs")
    if width <= 0 or height <= 0:
        raise ROIError("malformed_roi", "frame dimensions must be positive")
    if (
        (vertices[:, 0] < 0).any()
        or (vertices[:, 0] >= width).any()
        or (vertices[:, 1] < 0).any()
        or (vertices[:, 1] >= height).any()
    ):
        raise ROIError("out_of_frame_roi", "ROI has vertices outside the source frame")
    area = abs(float(cv2.contourArea(vertices.astype(np.float32))))
    if not math.isfinite(area) or area < 1.0:
        raise ROIError("malformed_roi", "ROI polygon has zero area")
    normalized = vertices.astype(np.float32)
    policy = str(
        raw_roi.get(
            "filter_policy",
            "segmentation_centroid_or_box_center_inside_polygon",
        )
    )
    return ROI(normalized, _hash_vertices(normalized), policy)


def load_roi(metadata_path: Path, width: int | None = None, height: int | None = None) -> tuple[ROI, dict]:
    try:
        metadata = json.loads(metadata_path.read_text())
    except FileNotFoundError as exc:
        raise ROIError("missing_metadata", f"metadata not found: {metadata_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ROIError("malformed_metadata", f"metadata is not valid JSON: {metadata_path}") from exc
    return (
        parse_roi(
            metadata,
            int(width or metadata.get("width") or 0),
            int(height or metadata.get("height") or 0),
        ),
        metadata,
    )


def point_inside(roi: ROI, point: tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(roi.vertices, point, False) >= 0


def mask_source(image: np.ndarray, roi: ROI) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(roi.vertices).astype(np.int32)], 255)
    return cv2.bitwise_and(image, image, mask=mask)


def polygon_in_crop(roi: ROI, left: float, top: float, side: float, size: int) -> np.ndarray:
    scale = float(size) / float(side)
    polygon = roi.vertices.copy()
    polygon[:, 0] = (polygon[:, 0] - left) * scale
    polygon[:, 1] = (polygon[:, 1] - top) * scale
    return polygon


def mask_crop_rgb(crop_rgb: np.ndarray, roi: ROI, left: float, top: float, side: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = crop_rgb.shape[:2]
    polygon = polygon_in_crop(roi, left, top, side, width)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    return cv2.bitwise_and(crop_rgb, crop_rgb, mask=mask), polygon
