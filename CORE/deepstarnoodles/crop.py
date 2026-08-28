# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""StarSeg cropping for focal-animal and BoT-SORT multi-animal inference."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.basetrack import BaseTrack

from deepstarnoodles.paths import DEFAULT_BOTSORT_CONFIG, DEFAULT_STARSEG

SUNFLOWER_NAMES = ("Pycnopodia_helianthoides", "sunflower_star")
_BOTSORT_ID_LOCK = Lock()


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
    star_track_id: int | None = None


def _sunflower_class_ids(model: YOLO) -> set[int]:
    raw_names = model.names
    if isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    else:
        names = {index: str(value) for index, value in enumerate(raw_names)}
    ids = {cid for cid, name in names.items() if name in SUNFLOWER_NAMES}
    if not ids:
        raise RuntimeError(
            f"StarSeg model has no sunflower class {SUNFLOWER_NAMES!r}; names={names}"
        )
    return ids


def _result_detections(
    result: object,
    sunflower_ids: set[int],
    *,
    require_track_ids: bool,
) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    track_ids: np.ndarray | None = None
    if require_track_ids:
        raw_ids = getattr(boxes, "id", None)
        if raw_ids is None:
            return []
        track_ids = raw_ids.detach().cpu().numpy().astype(int)

    polygons = None
    masks = getattr(result, "masks", None)
    if masks is not None and masks.xy is not None:
        polygons = masks.xy

    detections: list[dict] = []
    for index, class_id in enumerate(classes):
        if int(class_id) not in sunflower_ids:
            continue
        x1, y1, x2, y2 = (float(value) for value in xyxy[index])
        detections.append(
            {
                "xyxy": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "area": max(1.0, x2 - x1) * max(1.0, y2 - y1),
                "confidence": float(confidences[index]),
                "polygon": (
                    np.asarray(polygons[index], dtype=np.float32)
                    if polygons is not None and index < len(polygons)
                    else None
                ),
                "star_track_id": (
                    int(track_ids[index]) if track_ids is not None else None
                ),
            }
        )
    return detections


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


def _crop_result(
    frame_bgr: np.ndarray,
    detection: dict,
    *,
    context_scale: float,
) -> CropResult:
    crop_bgr, left, top, side = _centered_square_crop(
        frame_bgr,
        detection["xyxy"],
        context_scale,
    )
    return CropResult(
        crop_rgb=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
        bbox_xyxy=detection["xyxy"],
        center=detection["center"],
        confidence=detection["confidence"],
        crop_left=left,
        crop_top=top,
        crop_side=side,
        mask_polygon=detection["polygon"],
        star_track_id=detection["star_track_id"],
    )


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
        self.sunflower_ids = _sunflower_class_ids(self.model)
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
        dets = _result_detections(
            result,
            self.sunflower_ids,
            require_track_ids=False,
        )
        det = self._select(dets, math.hypot(w, h))
        if det is None:
            return None
        self._last_center = det["center"]
        return _crop_result(
            frame_bgr,
            det,
            context_scale=self.config.context_scale,
        )


class TrackedStarCropper:
    """Return every tracked sunflower-star crop in one video stream.

    Ultralytics stores BoT-SORT state on the predictor, so an instance must
    never be shared by interleaved videos. Call :meth:`reset` at each video
    boundary.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_STARSEG,
        config: CropConfig = CropConfig(),
        *,
        tracker_config: Path = DEFAULT_BOTSORT_CONFIG,
        model: YOLO | None = None,
    ) -> None:
        self.config = config
        self.tracker_config = Path(tracker_config).expanduser().resolve()
        if not self.tracker_config.is_file():
            raise FileNotFoundError(
                f"BoT-SORT configuration not found: {self.tracker_config}"
            )
        self.model = YOLO(str(model_path)) if model is None else model
        self.sunflower_ids = _sunflower_class_ids(self.model)
        # Ultralytics 8.4.51 uses a class-level BaseTrack counter. Preserve a
        # private counter per video while calls are serialized through the lock.
        self._base_track_count = 0

    def reset(self) -> None:
        """Clear BoT-SORT state before this predictor is used for another video."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", ()) if predictor is not None else ()
        with _BOTSORT_ID_LOCK:
            process_count = BaseTrack._count
            BaseTrack._count = self._base_track_count
            try:
                for tracker in trackers:
                    tracker.reset()
            finally:
                self._base_track_count = 0
                BaseTrack._count = process_count

    def track_crops(self, frame_bgr: np.ndarray) -> list[CropResult]:
        """Track and crop all confirmed sunflower stars in one BGR frame."""
        with _BOTSORT_ID_LOCK:
            process_count = BaseTrack._count
            BaseTrack._count = self._base_track_count
            try:
                result = self.model.track(
                    frame_bgr,
                    persist=True,
                    tracker=str(self.tracker_config),
                    classes=sorted(self.sunflower_ids),
                    batch=1,
                    imgsz=self.config.imgsz,
                    conf=self.config.conf,
                    iou=self.config.iou,
                    device=self.config.device,
                    verbose=False,
                )[0]
            finally:
                self._base_track_count = BaseTrack._count
                BaseTrack._count = process_count
        detections = _result_detections(
            result,
            self.sunflower_ids,
            require_track_ids=True,
        )
        crops = [
            _crop_result(
                frame_bgr,
                detection,
                context_scale=self.config.context_scale,
            )
            for detection in detections
        ]
        return sorted(
            crops,
            key=lambda crop: (
                crop.star_track_id is None,
                crop.star_track_id if crop.star_track_id is not None else -1,
            ),
        )
