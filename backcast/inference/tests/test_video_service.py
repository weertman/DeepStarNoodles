# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pytest

from deepstarnoodles_inference import video_service
from test_benchmark_cascade import Boxes, make_pipeline


def write_video(path, frames=6, size=(64, 48), fps=1.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), 20 * index, dtype=np.uint8)
        frame[8:24, 8:24] = 200
        writer.write(frame)
    writer.release()
    return path


def detector(frames, **_kwargs):
    # One sunflower star per frame plus one other class.
    return [type("Result", (), {"boxes": Boxes([[8, 8, 24, 24], [0, 0, 40, 40]], [2, 0]), "masks": None})()
            for _ in frames]


def test_video_frames_decode_every_frame_in_order(tmp_path):
    path = write_video(tmp_path / "clip.avi")
    with video_service.VideoFrames(path, "clip", prefetch=2) as source:
        groups = list(source)
    assert [ids for ids, _ in groups] == [[f"clip#{i:06d}"] for i in range(6)]
    assert all(frames[0].shape == (48, 64, 3) and frames[0].dtype == np.uint8 for _, frames in groups)
    assert source.frames == 6


def test_claim_hands_each_video_to_one_worker(tmp_path):
    ready = tmp_path / "ready"
    ready.mkdir()
    for name in ("000002__b.mp4", "000001__a.mp4", "notes.txt"):
        (ready / name).write_bytes(b"x")
    first, second = video_service.claim(tmp_path, 0), video_service.claim(tmp_path, 1)
    assert (first.parent.name, first.name) == ("0", "000001__a.mp4")
    assert (second.parent.name, second.name) == ("1", "000002__b.mp4")
    assert video_service.claim(tmp_path, 0) is None
    assert sorted(p.name for p in ready.iterdir()) == ["notes.txt"]


def test_claim_respects_session_prefix(tmp_path):
    ready = tmp_path / "ready"
    ready.mkdir()
    for name in ("A1_00001__a.mp4", "B1_00001__a.mp4", "B1_00002__b.mp4"):
        (ready / name).write_bytes(b"x")
    assert video_service.claim(tmp_path, 0, "B1_").name == "B1_00001__a.mp4"
    assert video_service.claim(tmp_path, 1, "B1_").name == "B1_00002__b.mp4"
    assert video_service.claim(tmp_path, 0, "B1_") is None
    assert [p.name for p in ready.iterdir()] == ["A1_00001__a.mp4"]


@pytest.mark.parametrize("ahead", [0, 2])
def test_annotate_video_writes_one_record_per_frame(tmp_path, monkeypatch, ahead):
    path = write_video(tmp_path / "clip.avi")
    pipeline = make_pipeline(monkeypatch, detector_batch=1, crop_batch=1)
    pipeline.args.hydra_output_mode = "minimal"
    pipeline.args.detect_ahead = ahead
    pipeline.args.decode_prefetch_frames = 2
    pipeline.detector = type("Detector", (), {"predict": staticmethod(detector)})()
    sink = io.StringIO()
    summary = video_service.annotate_video(pipeline, pipeline.args, path, "clip", sink, stop=lambda: False)
    records = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert summary["complete"] and summary["frames"] == 6 and summary["star_crops"] == summary["arms"] == 6
    assert [record["frame"] for record in records] == list(range(6))
    assert {record["video"] for record in records} == {"clip"}
    assert [record["image_path"] for record in records] == [f"clip#{i:06d}" for i in range(6)]
    assert all(len(record["stars"]) == 1 for record in records)
    assert [record["time_s"] for record in records] == [float(i) for i in range(6)]
    assert all(list(record)[-4:] == ["video", "time_s", "fps", "schema_version"] for record in records)
    assert all(list(record["stars"][0])[-5:] == ["body_origin_model", "body_origin_code",
                                                 "support_area_model_px", "support_centroid_model", "landmarks_model"]
               for record in records)
    assert summary["fps"] == 1.0


def test_annotate_video_passes_the_worker_stream_to_detect_ahead(tmp_path, monkeypatch):
    from deepstarnoodles_inference import detect_ahead

    seen = []

    class Spy(detect_ahead.DetectAhead):
        def __init__(self, *args, stream=None, **kwargs):
            seen.append(stream)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(detect_ahead, "DetectAhead", Spy)
    pipeline = make_pipeline(monkeypatch, detector_batch=1, crop_batch=1)
    pipeline.args.hydra_output_mode = "minimal"
    pipeline.args.detect_ahead = 2
    pipeline.args.decode_prefetch_frames = 2
    pipeline.detector = type("Detector", (), {"predict": staticmethod(detector)})()
    worker_stream = object()
    for name in ("a", "b"):
        path = write_video(tmp_path / f"{name}.avi", frames=2)
        video_service.annotate_video(pipeline, pipeline.args, path, name, io.StringIO(), stop=lambda: False,
                                     stream=worker_stream)
    assert seen == [worker_stream, worker_stream]


def test_annotate_video_stops_between_frames(tmp_path, monkeypatch):
    path = write_video(tmp_path / "clip.avi")
    pipeline = make_pipeline(monkeypatch, detector_batch=1, crop_batch=1)
    pipeline.args.hydra_output_mode = "minimal"
    pipeline.args.detect_ahead = 0
    pipeline.args.decode_prefetch_frames = 1
    pipeline.detector = type("Detector", (), {"predict": staticmethod(detector)})()
    sink = io.StringIO()
    summary = video_service.annotate_video(pipeline, pipeline.args, path, "clip", sink,
                                           stop=lambda: sink.getvalue().count("\n") >= 2)
    assert not summary["complete"] and summary["frames"] == 2


def test_service_options_validation(tmp_path):
    base = ["--queue", str(tmp_path / "q"), "--output", str(tmp_path / "o"), "--models", "m"]
    args, rest, runtime = video_service.parse_args(base + ["--processes", "8", "--gpu-crops", "--detect-ahead", "2"])
    assert args.processes == 8 and runtime.gpu_crops and runtime.detect_ahead == 2 and "--gpu-crops" in rest
    for extra in (["--processes", "0"], ["--processes", "17"], ["--decode-workers", "2"], ["--fps", "2"],
                  ["--gru-execution", "graph", "--gru-streams", "2"], ["--archive-processes", "2"]):
        with pytest.raises(SystemExit):
            video_service.parse_args(base + extra)


def test_video_frames_expose_the_nominal_frame_rate(tmp_path):
    path = write_video(tmp_path / "clip.avi", frames=3, fps=5.0)
    with video_service.VideoFrames(path, "clip", prefetch=1) as source:
        assert next(source)[0] == ["clip#000000"]
        assert source.fps == 5.0
        list(source)
    record = video_service.frame_record({"frame": 0, "image_path": "clip#000002", "source_shape": [48, 64],
                                         "stars": []}, "clip", source.fps)
    assert record == {"frame": 2, "image_path": "clip#000002", "source_shape": [48, 64], "stars": [],
                      "video": "clip", "time_s": 0.4, "fps": 5.0, "schema_version": 4}
    assert video_service.frame_record({"frame": 0, "image_path": "clip#000002", "stars": []}, "clip")["time_s"] is None
