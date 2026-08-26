# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""StarSeg focal-animal cropper: YOLO detection to a centered square crop."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from deepstarnoodles.paths import DEFAULT_STARSEG

SUNFLOWER_NAMES = ("Pycnopodia_helianthoides", "sunflower_star")


@dataclass(frozen=True)
class CropConfig:
    # Square crop side = max(bbox_w, bbox_h) * context_scale. Dense training
    # crops used context_scale 2.0.
    context_scale: float = 2.0
    imgsz: int = 1024
    conf: float = 0.25
    iou: float = 0.7
    device: str = "cuda:0"
    track_max_jump_frac: float = 0.25


@dataclass(frozen=True)
class CropResult:
    crop_rgb: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float
    crop_left: float
    crop_top: float
    crop_side: float
    mask_polygon: np.ndarray | None


def _centered_square_crop(
    image_bgr: np.ndarray,
    xyxy: tuple[float, float, float, float],
    context_scale: float,
) -> tuple[np.ndarray, float, float, float]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = int(math.ceil(max(max(1.0, x2 - x1), max(1.0, y2 - y1)) * context_scale))
    side = max(16, side)
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    src_l, src_t = max(0, left), max(0, top)
    src_r, src_b = min(w, left + side), min(h, top + side)
    crop = image_bgr[src_t:src_b, src_l:src_r]
    pad_l, pad_t = src_l - left, src_t - top
    pad_r, pad_b = (left + side) - src_r, (top + side) - src_b
    if any(v > 0 for v in (pad_l, pad_t, pad_r, pad_b)):
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return crop, float(left), float(top), float(side)


class StarCropper:
    def __init__(
        self,
        model_path: Path = DEFAULT_STARSEG,
        config: CropConfig = CropConfig(),
        *,
        model: YOLO | None = None,
    ) -> None:
        self.config = config
        self.model = YOLO(str(model_path)) if model is None else model
        names = {int(k): str(v) for k, v in self.model.names.items()}
        ids = [cid for cid, name in names.items() if name in SUNFLOWER_NAMES]
        if not ids:
            raise RuntimeError(
                f"StarSeg model has no sunflower class {SUNFLOWER_NAMES!r}; names={names}"
            )
        self.sunflower_ids = set(ids)
        self._last_center: tuple[float, float] | None = None

    def reset(self) -> None:
        self._last_center = None

    def _select(self, dets: list[dict], frame_diag: float) -> dict | None:
        if not dets:
            return None
        if self._last_center is not None:
            max_jump = self.config.track_max_jump_frac * frame_diag
            near = [
                (math.dist(d["center"], self._last_center), d)
                for d in dets
                if math.dist(d["center"], self._last_center) <= max_jump
            ]
            if near:
                near.sort(key=lambda t: t[0])
                return near[0][1]
        return max(dets, key=lambda d: (d["area"], d["confidence"]))

    def crop(self, frame_bgr: np.ndarray) -> CropResult | None:
        h, w = frame_bgr.shape[:2]
        result = self.model.predict(
            frame_bgr,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        conf = boxes.conf.detach().cpu().numpy()
        polys = None
        if result.masks is not None and result.masks.xy is not None:
            polys = result.masks.xy

        dets: list[dict] = []
        for i in range(len(cls)):
            if int(cls[i]) not in self.sunflower_ids:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            dets.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "area": max(1.0, (x2 - x1)) * max(1.0, (y2 - y1)),
                    "confidence": float(conf[i]),
                    "polygon": (
                        np.asarray(polys[i], dtype=np.float32) if polys is not None else None
                    ),
                }
            )
        det = self._select(dets, math.hypot(w, h))
        if det is None:
            return None
        self._last_center = det["center"]
        crop_bgr, left, top, side = _centered_square_crop(
            frame_bgr, det["xyxy"], self.config.context_scale
        )
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return CropResult(
            crop_rgb=crop_rgb,
            bbox_xyxy=det["xyxy"],
            center=det["center"],
            confidence=det["confidence"],
            crop_left=left,
            crop_top=top,
            crop_side=side,
            mask_polygon=det["polygon"],
        )
