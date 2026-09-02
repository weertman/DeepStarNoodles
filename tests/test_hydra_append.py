# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from deepstarnoodles.batch_library import Clip
from deepstarnoodles.crop import CropConfig, centered_square_crop, crop_from_xyxy
from deepstarnoodles.hydra_append import (
    append_hydra_clip,
    hydra_contract_hash,
    hydra_frames_path,
    inventory_hydra_sources,
    is_hydra_complete,
)
from deepstarnoodles.hydra_budget import choose_batch_size, recommend_memory_fraction
from deepstarnoodles.hydra_landmarks import crop_to_source, peak_argmax, summarize_activated
from deepstarnoodles.recorder_guard import RecorderGuard, RecorderSnapshot, snapshot_from_status
from deepstarnoodles.roi import parse_roi
from deepstarnoodles.sidecar_crop import SidecarCropper, sunflower_detections


def roi_meta(width=20, height=20):
    return {
        "width": width,
        "height": height,
        "starseg_roi": {
            "absolute": [[1, 1], [width - 2, 1], [width - 2, height - 2], [1, height - 2]],
            "filter_policy": "segmentation_centroid_or_box_center_inside_polygon",
        },
    }


def sunflower(box, confidence=0.9, index_name="sunflower_star"):
    return {
        "class_id": 0,
        "class_name": index_name,
        "confidence": confidence,
        "box_xyxy": box,
        "segment_xy": [
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]],
        ],
    }


class FakeHydra:
    image_size = 32

    def predict_raw_batch(self, crops):
        batch = len(crops)
        raw = np.zeros((batch, 18, 32, 32), np.float32)
        activated = np.zeros((batch, 18, 32, 32), np.float32)
        activated[:, 0, 4, 8] = 0.9
        activated[:, 5, 4, 8] = 1.0
        activated[:, 2, 4, 8] = 0.25
        activated[:, 3, 4, 8] = -0.4
        activated[:, 10, 4, 8] = 1.5
        return raw, activated


def test_sidecar_selects_largest_sunflower_and_ignores_prey():
    row = {
        "frame_index": 0,
        "detections": [
            sunflower([2, 2, 6, 6], 0.99),
            {
                "class_name": "live_prey",
                "confidence": 0.99,
                "box_xyxy": [1, 1, 18, 18],
                "segment_xy": [],
            },
            sunflower([8, 8, 16, 16], 0.5),
        ],
    }
    dets = sunflower_detections(row)
    assert [d["source_index"] for d in dets] == [0, 2]
    frame = np.full((20, 20, 3), 200, np.uint8)
    roi = parse_roi(roi_meta(), 20, 20)
    result = SidecarCropper(CropConfig(device="cpu")).crop(frame, row, roi)
    assert result is not None
    assert result.source_index == 2
    assert result.crop.bbox_xyxy == (8.0, 8.0, 16.0, 16.0)


def test_sidecar_continuity_prefers_near_previous_center():
    cropper = SidecarCropper(CropConfig(device="cpu", track_max_jump_frac=0.25))
    roi = parse_roi(roi_meta(), 20, 20)
    frame = np.full((20, 20, 3), 200, np.uint8)
    first = {
        "frame_index": 0,
        "detections": [sunflower([2, 2, 6, 6])],
    }
    second = {
        "frame_index": 1,
        "detections": [sunflower([2, 2, 6, 6], 0.4), sunflower([10, 10, 16, 16], 0.99)],
    }
    assert cropper.crop(frame, first, roi).source_index == 0
    assert cropper.crop(frame, second, roi).source_index == 0


def test_sidecar_crop_matches_2x_geometry_and_masks_roi():
    frame = np.full((20, 20, 3), 255, np.uint8)
    roi = parse_roi(roi_meta(), 20, 20)
    row = {"frame_index": 0, "detections": [sunflower([6, 6, 10, 10])]}
    result = SidecarCropper(CropConfig(device="cpu")).crop(frame, row, roi)
    assert result is not None
    _crop_bgr, left, top, side = centered_square_crop(frame, (6, 6, 10, 10), 2.0)
    assert result.crop.crop_left == left
    assert result.crop.crop_top == top
    assert result.crop.crop_side == side
    direct = crop_from_xyxy(frame, (6, 6, 10, 10), roi=roi, context_scale=2.0)
    assert result.crop.crop_rgb.shape == direct.crop_rgb.shape
    assert result.crop.crop_rgb[0, 0].sum() == 0


def test_no_sunflower_returns_none():
    frame = np.zeros((20, 20, 3), np.uint8)
    roi = parse_roi(roi_meta(), 20, 20)
    row = {
        "frame_index": 0,
        "detections": [
            {
                "class_name": "live_prey",
                "confidence": 0.9,
                "box_xyxy": [4, 4, 8, 8],
                "segment_xy": [],
            }
        ],
    }
    assert SidecarCropper(CropConfig(device="cpu")).crop(frame, row, roi) is None
    assert SidecarCropper(CropConfig(device="cpu")).crop(frame, None, roi) is None


def test_landmarks_peaks_dxdy_and_embedding_sample():
    field = np.zeros((18, 16, 16), np.float32)
    field[0, 3, 5] = 0.8
    field[5, 2, 4] = 0.95
    field[2, 3, 5] = 0.3
    field[3, 3, 5] = -0.2
    field[10, 2, 4] = 2.0
    summary = summarize_activated(field, left=10, top=20, side=32, image_size=16)
    assert summary["landmarks"]["arm_support"]["u"] == 5
    assert summary["landmarks"]["mouth"]["v"] == 2
    x, y = crop_to_source(5, 3, left=10, top=20, side=32, size=16)
    assert summary["landmarks"]["arm_support"]["x"] == pytest.approx(x)
    assert summary["landmarks"]["arm_dx"]["value"] == pytest.approx(0.3)
    assert summary["landmarks"]["arm_dx"]["sample"] == "arm_support"
    assert summary["embedding_sample"] == "mouth"
    assert summary["embedding"][0] == pytest.approx(2.0)
    u, v, score = peak_argmax(field[5])
    assert (u, v) == (4.0, 2.0)
    assert score == pytest.approx(0.95)


def _write_clip(root: Path, *, empty_starseg=False, frames=3, camera="A", stem="seg"):
    folder = root / camera / "trial_104__session" / stem
    folder.mkdir(parents=True)
    video = folder / f"{stem}.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (20, 20))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((20, 20, 3), 180, np.uint8))
    writer.release()
    metadata = {
        **roi_meta(),
        "frame_timestamps": [100.0 + i for i in range(frames)],
        "start_ts": 100.0,
    }
    metadata_path = video.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata))
    summary = {
        "frames": 0 if empty_starseg else frames,
        "detections": 0 if empty_starseg else frames,
        "classes": {},
    }
    (folder / f"{stem}_starseg_summary.json").write_text(json.dumps(summary))
    jsonl = folder / f"{stem}_starseg_frames.jsonl"
    if empty_starseg:
        jsonl.write_text("")
    else:
        with jsonl.open("w") as handle:
            for index in range(frames):
                handle.write(
                    json.dumps(
                        {
                            "frame_index": index,
                            "frame_ts": 100.0 + index,
                            "detections": [sunflower([6, 6, 10, 10])] if index != 1 else [],
                        }
                    )
                    + "\n"
                )
    roi = parse_roi(metadata, 20, 20)
    clip = Clip(
        f"{camera}:{stem}",
        camera,
        folder.parents[0].name,
        "unknown",
        stem,
        video,
        metadata_path,
        metadata,
        roi,
    )
    return clip


def test_append_writes_jsonl_summary_and_metadata_pointers(tmp_path):
    clip = _write_clip(tmp_path, frames=3)
    result = append_hydra_clip(
        clip,
        FakeHydra(),
        checkpoint_sha="abc123",
        batch_size=2,
        device="cpu",
    )
    assert result.status == "complete"
    assert result.frames == 3
    assert result.detected == 2
    rows = [
        json.loads(line)
        for line in hydra_frames_path(clip.video).read_text().splitlines()
        if line
    ]
    assert [row["detected"] for row in rows] == [True, False, True]
    assert rows[0]["crop"]["source_index"] == 0
    assert "arm_support" in rows[0]["landmarks"]
    assert rows[0]["channel_contract"] == "legacy10+embedding8"
    assert "crop" not in rows[1]
    summary = json.loads(clip.video.with_name(f"{clip.stem}_hydra_summary.json").read_text())
    assert summary["contract_hash"] == hydra_contract_hash("abc123")
    metadata = json.loads(clip.metadata_path.read_text())
    assert metadata["hydra_frames_jsonl"].endswith("_hydra_frames.jsonl")
    assert metadata["hydra_summary"]["detected"] == 2
    assert "starseg_roi" in metadata


def test_inventory_skips_empty_starseg_and_complete_hash(tmp_path):
    empty = _write_clip(tmp_path, empty_starseg=True, stem="empty")
    ready = _write_clip(tmp_path, stem="ready", camera="B")
    digest = hydra_contract_hash("abc123")
    eligible, skipped, complete = inventory_hydra_sources(tmp_path, expected_hash=digest)
    assert empty.key not in [clip.key for clip in eligible]
    assert any(item.status == "empty_starseg" for item in skipped)
    assert [clip.key for clip in eligible] == [ready.key]
    append_hydra_clip(ready, FakeHydra(), checkpoint_sha="abc123", batch_size=4, device="cpu")
    assert is_hydra_complete(ready, digest)
    eligible, _skipped, complete = inventory_hydra_sources(tmp_path, expected_hash=digest)
    assert ready.key in [clip.key for clip in complete]
    assert ready.key not in [clip.key for clip in eligible]
    assert not is_hydra_complete(ready, hydra_contract_hash("other"))


def test_recorder_guard_pauses_on_queue_and_wait_increment():
    payloads = iter(
        [
            {"recording": True, "starseg": {"recorded_queue_depth": 1, "recorded_waits": 3}},
            {"recording": True, "starseg": {"recorded_queue_depth": 12, "recorded_waits": 3}},
            {"recording": True, "starseg": {"recorded_queue_depth": 2, "recorded_waits": 4}},
            {"recording": False, "starseg": {"recorded_queue_depth": 0, "recorded_waits": 4}},
        ]
    )

    def fetch(_url):
        return next(payloads)

    guard = RecorderGuard(fetch=fetch, queue_watermark=8, poll_s=0)
    first = guard.poll()
    assert not guard.should_pause(first)
    guard.note(first)
    assert guard.should_pause(guard.poll())
    bumped = guard.poll()
    assert guard.should_pause(bumped)
    pauses: list[int] = []
    # Remaining payload drains the queue.
    guard.wait_until_clear(lambda: pauses.append(1))
    assert pauses == [1]


def test_recorder_down_does_not_pause():
    def fetch(_url):
        raise TimeoutError("offline")

    guard = RecorderGuard(fetch=fetch)
    snap = guard.poll()
    assert snap == RecorderSnapshot(reachable=False)
    assert not guard.should_pause(snap)
    assert snapshot_from_status(
        {"recording": False, "starseg": {"recorded_queue_depth": 0, "recorded_waits": 0}}
    ).reachable


def test_append_yields_to_recorder_guard(tmp_path):
    clip = _write_clip(tmp_path, frames=2, stem="yield")
    calls = {"n": 0}

    class Guard:
        def yield_if_needed(self, on_pause=None):
            calls["n"] += 1
            if on_pause:
                on_pause()

    append_hydra_clip(
        clip,
        FakeHydra(),
        checkpoint_sha="abc",
        batch_size=1,
        device="cpu",
        guard=Guard(),
    )
    assert calls["n"] >= 2


def test_choose_batch_keeps_largest_safe_size():
    results = {
        8: {"images_per_s": 40, "yolo_p99_ms": 180},
        4: {"images_per_s": 30, "yolo_p99_ms": 90},
        1: {"images_per_s": 12, "yolo_p99_ms": 70},
    }
    assert choose_batch_size(results) == 8
    results[8]["yolo_p99_ms"] = 260
    assert choose_batch_size(results) == 4
    results[8] = {"error": "OOM"}
    results[4]["yolo_p99_ms"] = 400
    assert choose_batch_size(results) == 1
    assert choose_batch_size({8: {"images_per_s": 50}, 4: {"images_per_s": 40}}) == 8
    assert recommend_memory_fraction(
        {4: {"hydra_vram_bytes": 2 * 1024**3, "gpu_total_bytes": 8 * 1024**3}},
        4,
    ) == pytest.approx(0.33)
