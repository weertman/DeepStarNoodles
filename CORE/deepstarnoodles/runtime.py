# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Shared postprocessing for batched video inference."""
from __future__ import annotations

import cv2
import numpy as np

from deepstarnoodles.crop import CropResult
from deepstarnoodles.decode import find_peaks
from deepstarnoodles.fused_tracker import FusedTracker, tip_anchored_resample


K_TRACK = 12


def body_center(activated: np.ndarray) -> np.ndarray:
    mouth = find_peaks(activated[5], threshold=0.15, nms_radius=8)
    if len(mouth):
        return mouth[0, :2].astype(np.float32)
    support = activated[0] > 0.3
    ys, xs = np.nonzero(support)
    if len(xs):
        return np.asarray([xs.mean(), ys.mean()], dtype=np.float32)
    height, width = activated.shape[-2:]
    return np.asarray([width / 2.0, height / 2.0], dtype=np.float32)


def track_arms(
    tracker: FusedTracker,
    frame_index: int,
    activated: np.ndarray,
    arms: list[dict],
) -> list[int]:
    if not arms:
        return []
    center = body_center(activated)
    sampled: list[np.ndarray] = []
    lengths: list[float] = []
    identities: list[np.ndarray] = []
    for arm in arms:
        points, length = tip_anchored_resample(
            np.asarray(arm["polyline"], dtype=np.float32) - center,
            K_TRACK,
        )
        sampled.append(points)
        lengths.append(length)
        identities.append(np.asarray(arm["identity"], dtype=np.float32))
    return tracker.step(
        frame_index,
        np.asarray(sampled, dtype=np.float32),
        np.asarray(lengths, dtype=np.float32),
        np.asarray(identities, dtype=np.float32),
    )


def source_arm_records(
    arms: list[dict],
    labels: list[int],
    crop: CropResult,
    *,
    model_size: int,
) -> list[dict]:
    scale = crop.crop_side / model_size
    records: list[dict] = []
    for arm, track_id in zip(arms, labels):
        polyline = np.asarray(arm["polyline"], dtype=np.float32)
        source = np.stack(
            [
                polyline[:, 0] * scale + crop.crop_left,
                polyline[:, 1] * scale + crop.crop_top,
            ],
            axis=1,
        )
        records.append(
            {
                "track_id": int(track_id),
                "polyline_crop": polyline.tolist(),
                "polyline_source": source.tolist(),
                "identity": np.asarray(arm["identity"]).tolist(),
            }
        )
    return records


def track_color(track_id: int) -> tuple[int, int, int]:
    hue = int((track_id * 47) % 180)
    pixel = np.uint8([[[hue, 220, 255]]])
    return tuple(
        int(value)
        for value in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    )


def draw_overlay(
    frame_bgr: np.ndarray,
    crop: CropResult | None,
    arms: list[dict],
) -> np.ndarray:
    overlay = frame_bgr.copy()
    if crop is None:
        return overlay
    x1, y1, x2, y2 = (int(value) for value in crop.bbox_xyxy)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    for arm in arms:
        points = np.rint(
            np.asarray(arm["polyline_source"], dtype=np.float32)
        ).astype(np.int32)
        color = track_color(int(arm["track_id"]))
        if len(points) > 1:
            cv2.polylines(
                overlay,
                [points],
                False,
                color,
                2,
                cv2.LINE_AA,
            )
        if len(points):
            tip = tuple(points[-1])
            cv2.circle(overlay, tip, 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                str(arm["track_id"]),
                (tip[0] + 4, tip[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return overlay
