# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import deepstarnoodles.runtime as runtime
from deepstarnoodles.crop import CropResult
from deepstarnoodles.runtime import (
    StarArmTrackerRegistry,
    crop_record,
    draw_multi_overlay,
    multi_star_video_payload,
    source_arm_records,
    tracked_star_record,
)


def _activated(size: int = 64) -> np.ndarray:
    activated = np.zeros((18, size, size), dtype=np.float32)
    activated[0, 28:36, 20:56] = 1.0
    activated[5, 32, 32] = 1.0
    return activated


def _arm(offset: float = 0.0, identity: float = 0.0) -> dict:
    return {
        "polyline": np.asarray(
            [[32 + offset, 32], [42 + offset, 32], [52 + offset, 32]],
            dtype=np.float32,
        ),
        "identity": np.full(8, identity, dtype=np.float32),
    }


def _crop(star_track_id: int | None = 7) -> CropResult:
    return CropResult(
        crop_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        bbox_xyxy=(8.0, 10.0, 48.0, 50.0),
        center=(28.0, 30.0),
        confidence=0.9,
        crop_left=4.0,
        crop_top=6.0,
        crop_side=64.0,
        mask_polygon=None,
        star_track_id=star_track_id,
    )


def test_each_star_owns_an_independent_arm_id_namespace() -> None:
    registry = StarArmTrackerRegistry(star_max_age=30)
    activated = _activated()

    first_star = registry.track(11, 0, activated, [_arm(identity=0.1)])
    second_star = registry.track(22, 0, activated, [_arm(identity=0.9)])
    first_again = registry.track(11, 1, activated, [_arm(offset=1, identity=0.1)])
    second_again = registry.track(22, 1, activated, [_arm(offset=-1, identity=0.9)])

    assert first_star == [0]
    assert second_star == [0]
    assert first_again == [0]
    assert second_again == [0]
    assert registry.star_track_ids == (11, 22)


def test_registry_rejects_out_of_order_observations_and_prunes_after_horizon() -> None:
    registry = StarArmTrackerRegistry(star_max_age=2)
    activated = _activated()
    registry.track(3, 5, activated, [_arm()])

    with pytest.raises(ValueError, match="chronological"):
        registry.track(3, 4, activated, [_arm()])

    registry.prune(7)
    assert registry.star_track_ids == (3,)
    registry.prune(8)
    assert registry.star_track_ids == ()


def test_multi_star_records_scope_arm_ids_under_the_star() -> None:
    crop = _crop(12)
    arms = [_arm()]
    arm_records = source_arm_records(arms, [0], crop, model_size=64)

    record = tracked_star_record(crop, arm_records)
    payload = multi_star_video_payload(
        Path("input.mp4"),
        Path("config/star_tracking/botsort.yaml"),
        [{"frame": 0, "detected": True, "stars": [record]}],
    )

    assert record["star_track_id"] == 12
    assert record["crop"] == crop_record(crop)
    assert record["arms"][0]["track_id"] == 0
    assert record["arms"][0]["polyline_source"][0] == [36.0, 38.0]
    assert payload["schema_version"] == 2
    assert payload["tracking"]["star_id_scope"] == "video"
    assert payload["tracking"]["arm_id_scope"] == "star_track_id"
    assert payload["frames"][0]["stars"][0] is record


def test_tracked_star_record_requires_a_botsort_identity() -> None:
    with pytest.raises(ValueError, match="star_track_id"):
        tracked_star_record(_crop(None), [])


def test_multi_overlay_uses_fixed_polyline_color_not_track_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(_track_id: int) -> tuple[int, int, int]:
        raise AssertionError("multi-star overlay must not color arms by identity")

    monkeypatch.setattr(runtime, "track_color", fail_if_called)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    arms = [
        {
            "track_id": 99,
            "polyline_source": [[12.0, 12.0], [30.0, 30.0], [44.0, 40.0]],
        }
    ]

    overlay = draw_multi_overlay(frame, [(_crop(5), arms)])

    assert overlay.shape == frame.shape
    assert np.any(overlay != frame)
    assert tuple(overlay[30, 30]) == (255, 255, 255)


def test_multi_overlay_projects_support_from_crop_to_source_frame() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    support = np.zeros((64, 64), dtype=np.float32)
    support[10:20, 10:20] = 1.0

    overlay = draw_multi_overlay(
        frame,
        [(_crop(5), [])],
        support_masks=[support],
        support_threshold=0.5,
        support_alpha=0.5,
        support_color=(0, 0, 200),
    )

    assert tuple(overlay[20, 18]) == (0, 0, 100)
    assert tuple(overlay[60, 60]) == (0, 0, 0)


def test_multi_overlay_rejects_misaligned_support_masks() -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        draw_multi_overlay(
            np.zeros((80, 80, 3), dtype=np.uint8),
            [(_crop(5), [])],
            support_masks=[],
        )
