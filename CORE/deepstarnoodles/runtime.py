# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Shared postprocessing for batched video inference."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from deepstarnoodles.crop import CropResult
from deepstarnoodles.decode import find_peaks
from deepstarnoodles.fused_tracker import FusedTracker, tip_anchored_resample


K_TRACK = 12
DEFAULT_STAR_TRACK_MAX_AGE = 30


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


class StarArmTrackerRegistry:
    """Own one arm tracker for each BoT-SORT star identity in a video.

    Arm geometry is body-relative, so sharing one :class:`FusedTracker` across
    animals would permit invalid cross-animal associations. Missing detector
    observations do not age arm tracks; the complete per-star tracker is
    discarded only after BoT-SORT's lost-track horizon has elapsed.
    """

    def __init__(
        self,
        *,
        shape_gate: float = 120.0,
        embed_weight: float = 2.0,
        max_age: int = 16,
        star_max_age: int = DEFAULT_STAR_TRACK_MAX_AGE,
        tracker_factory: Callable[[], FusedTracker] | None = None,
    ) -> None:
        if star_max_age < 0:
            raise ValueError("star_max_age must be non-negative")
        self.star_max_age = int(star_max_age)
        self._tracker_factory = tracker_factory or (
            lambda: FusedTracker(
                shape_gate=shape_gate,
                embed_weight=embed_weight,
                max_age=max_age,
            )
        )
        self._trackers: dict[int, FusedTracker] = {}
        self._last_seen: dict[int, int] = {}

    @property
    def star_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._trackers))

    def track(
        self,
        star_track_id: int,
        frame_index: int,
        activated: np.ndarray,
        arms: list[dict],
    ) -> list[int]:
        star_track_id = int(star_track_id)
        frame_index = int(frame_index)
        previous_frame = self._last_seen.get(star_track_id)
        if previous_frame is not None and frame_index < previous_frame:
            raise ValueError(
                "star observations must be processed in chronological order"
            )
        tracker = self._trackers.get(star_track_id)
        if tracker is None:
            tracker = self._tracker_factory()
            self._trackers[star_track_id] = tracker
        self._last_seen[star_track_id] = frame_index
        return track_arms(tracker, frame_index, activated, arms)

    def prune(self, frame_index: int) -> None:
        frame_index = int(frame_index)
        stale = [
            star_track_id
            for star_track_id, last_seen in self._last_seen.items()
            if frame_index - last_seen > self.star_max_age
        ]
        for star_track_id in stale:
            del self._last_seen[star_track_id]
            del self._trackers[star_track_id]

    def clear(self) -> None:
        self._last_seen.clear()
        self._trackers.clear()


def crop_record(crop: CropResult) -> dict:
    """Serialize crop geometry shared by single- and multi-star schemas."""
    return {
        "left": crop.crop_left,
        "top": crop.crop_top,
        "side": crop.crop_side,
        "confidence": crop.confidence,
        "bbox_xyxy": list(crop.bbox_xyxy),
    }


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


def tracked_star_record(
    crop: CropResult,
    arms: list[dict],
) -> dict:
    """Serialize one tracked animal after arm records are in source pixels."""
    if crop.star_track_id is None:
        raise ValueError("tracked star records require crop.star_track_id")
    return {
        "star_track_id": int(crop.star_track_id),
        "crop": crop_record(crop),
        "arms": arms,
    }


def multi_star_video_payload(
    video: str | Path,
    tracker_config: str | Path,
    frames: list[dict],
) -> dict:
    """Build the versioned JSON envelope shared by both multi-star CLIs."""
    return {
        "schema_version": 2,
        "video": str(video),
        "tracking": {
            "mode": "multi_star_botsort",
            "star_id_scope": "video",
            "arm_id_scope": "star_track_id",
            "star_tracker_config": str(tracker_config),
        },
        "frames": frames,
    }


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


def draw_multi_overlay(
    frame_bgr: np.ndarray,
    stars: Sequence[tuple[CropResult, list[dict]]],
    *,
    support_masks: Sequence[np.ndarray] | None = None,
    support_threshold: float = 0.3,
    support_alpha: float = 0.28,
    support_color: tuple[int, int, int] = (0, 120, 255),
    polyline_color: tuple[int, int, int] = (255, 255, 255),
    draw_arm_ids: bool = False,
) -> np.ndarray:
    """Draw tracked stars with optional support masks and fixed-color arms.

    BoT-SORT identities are labeled on boxes. Arm colors intentionally do not
    encode identity so this renderer is suitable for publication-style output.
    """
    if support_masks is not None and len(support_masks) != len(stars):
        raise ValueError("support_masks must align one-to-one with stars")
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support_threshold must be between 0 and 1")
    if not 0.0 <= support_alpha <= 1.0:
        raise ValueError("support_alpha must be between 0 and 1")

    overlay = frame_bgr.copy()
    if support_masks is not None:
        frame_height, frame_width = overlay.shape[:2]
        combined_support = np.zeros((frame_height, frame_width), dtype=bool)
        for (crop, _arms), support in zip(stars, support_masks):
            support = np.asarray(support, dtype=np.float32)
            if support.ndim != 2:
                raise ValueError("each support mask must be a 2D array")
            side = max(1, int(round(crop.crop_side)))
            left = int(round(crop.crop_left))
            top = int(round(crop.crop_top))
            resized = cv2.resize(
                support,
                (side, side),
                interpolation=cv2.INTER_LINEAR,
            )
            destination_left = max(0, left)
            destination_top = max(0, top)
            destination_right = min(frame_width, left + side)
            destination_bottom = min(frame_height, top + side)
            if (
                destination_left >= destination_right
                or destination_top >= destination_bottom
            ):
                continue
            source_left = destination_left - left
            source_top = destination_top - top
            source_right = source_left + destination_right - destination_left
            source_bottom = source_top + destination_bottom - destination_top
            combined_support[
                destination_top:destination_bottom,
                destination_left:destination_right,
            ] |= (
                resized[source_top:source_bottom, source_left:source_right]
                >= support_threshold
            )
        if np.any(combined_support):
            tint = np.empty_like(overlay)
            tint[:] = support_color
            blended = cv2.addWeighted(
                overlay,
                1.0 - support_alpha,
                tint,
                support_alpha,
                0.0,
            )
            overlay[combined_support] = blended[combined_support]

    for crop, arms in stars:
        if crop.star_track_id is None:
            raise ValueError("multi-star overlays require crop.star_track_id")
        x1, y1, x2, y2 = (int(round(value)) for value in crop.bbox_xyxy)
        box_color = (0, 255, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            overlay,
            f"star {crop.star_track_id}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA,
        )
        for arm in arms:
            points = np.rint(
                np.asarray(arm["polyline_source"], dtype=np.float32)
            ).astype(np.int32)
            if len(points) > 1:
                cv2.polylines(
                    overlay,
                    [points],
                    False,
                    polyline_color,
                    2,
                    cv2.LINE_AA,
                )
            if not len(points):
                continue
            tip = tuple(points[-1])
            cv2.circle(overlay, tip, 4, polyline_color, -1, cv2.LINE_AA)
            if draw_arm_ids:
                cv2.putText(
                    overlay,
                    str(arm["track_id"]),
                    (tip[0] + 4, tip[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    polyline_color,
                    1,
                    cv2.LINE_AA,
                )
    return overlay
