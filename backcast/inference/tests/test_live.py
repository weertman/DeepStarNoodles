# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from deepstarnoodles_inference import benchmark, cli, live
from test_benchmark_cascade import make_pipeline


def pixels(count=1):
    frame = np.full((20, 30, 3), (0, 20, 30), dtype=np.uint8)
    frame[0, 0, 0] = count
    return frame


def live_pipeline(monkeypatch):
    cascade = make_pipeline(monkeypatch, detector_batch=1, crop_batch=2)
    events = []

    class Cropper:
        def track_crops(self, frame):
            events.append(("detect", int(frame[0, 0, 0])))
            return [replace(benchmark._crop_result(frame, {
                "xyxy": (4, 4, 12, 12), "center": (8, 8), "confidence": 0.9,
                "polygon": None, "star_track_id": None}, context_scale=2.0), star_track_id=7 + index)
                for index in range(int(frame[0, 0, 0]))]

        def reset(self):
            events.append(("reset_detector",))

    class Registry:
        def track(self, star, index, fields, arms):
            events.append(("track", star, index))
            return [42] * len(arms)

        def prune(self, index):
            events.append(("prune", index))

        def clear(self):
            events.append(("reset_registry",))

    cascade.live_cropper, cascade.registry = Cropper(), Registry()
    return live.LivePipeline(cascade, live.LiveConfig(device="cpu", crop_batch_size=2)), events


def test_ndarray_cascade_all_zero_remainder_and_source_gaps(monkeypatch):
    pipeline, events = live_pipeline(monkeypatch)
    monkeypatch.setattr(cv2, "imread", lambda *_args: pytest.fail("ndarray inference must not read files"))
    acquired = time.monotonic_ns() - 1_000_000
    records = [pipeline.predict(pixels(count), source_index=index, acquired_monotonic_ns=acquired,
                                acquisition_clock="opencv_read_completed")
               for count, index in [(3, 10), (0, 14), (1, 19)]]
    assert [r["frame"] for r in records] == [10, 14, 19]
    assert all(r["time_s"] is None and r["fps"] is None and r["schema_version"] == 4 for r in records)
    assert [r["processed_index"] for r in records] == [0, 1, 2]
    assert [r["skipped_source_indices_since_previous"] for r in records] == [0, 3, 4]
    assert [len(r["stars"]) for r in records] == [3, 0, 1]
    assert [event for event in events if event[0] == "prune"] == [("prune", 0), ("prune", 1), ("prune", 2)]
    assert [event[2] for event in events if event[0] == "track"] == [0, 0, 0, 2]
    assert pipeline.cascade.hydra.batches == [2, 1, 1]
    assert np.array_equal(pipeline.cascade.hydra.crops[0], cv2.cvtColor(pixels(3)[:16, :16], cv2.COLOR_BGR2RGB))
    arm = records[0]["stars"][0]["arms"][0]
    assert arm["track_id"] == 42 and arm["polyline_source"] == [[1.0, 2.0], [3.0, 4.0]]
    assert arm["stop_reason"] == "base"
    for record in records:
        assert record["image_path"] is None
        assert record["timing"]["queue_wait_ms"] >= 1
        assert record["inference_started_monotonic_ns"] <= record["inference_ended_monotonic_ns"]
    assert pipeline.report()["processed_frames"] == 3
    assert pipeline.report()["star_crops"] == 4


def test_predict_times_records_from_the_source_index(monkeypatch):
    pipeline, events = live_pipeline(monkeypatch)
    with pytest.raises(ValueError, match="fps"):
        pipeline.predict(pixels(1), source_index=20, fps=0.0)
    assert events == [] and pipeline.processed == 0
    timed = pipeline.predict(pixels(1), source_index=21, fps=2.0)
    assert (timed["time_s"], timed["fps"], list(timed)[-1]) == (10.5, 2.0, "schema_version")
    assert list(timed["stars"][0])[-5:] == ["body_origin_model", "body_origin_code", "support_area_model_px",
                                            "support_centroid_model", "landmarks_model"]


def test_ndarray_and_file_replay_have_identical_geometry_and_ids(tmp_path, monkeypatch):
    first, _ = live_pipeline(monkeypatch)
    second, _ = live_pipeline(monkeypatch)
    path = tmp_path / "source.png"
    assert cv2.imwrite(str(path), pixels(3))
    replay, _, _ = first.cascade.run_group([path], 0, benchmark.Timer(torch.device("cpu"), False))
    actual = second.predict(pixels(3))
    assert actual["stars"] == replay[0]["stars"]
    assert actual["source_shape"] == replay[0]["source_shape"]


def test_live_minimal_and_full_preserve_geometry_and_actual_tracking(monkeypatch):
    from deepstarnoodles_inference.hydra import HydraInferConfig, HydraPredictor
    from deepstarnoodles_inference.model import CHANNEL_NAMES
    from deepstarnoodles_inference.runtime import StarArmTrackerRegistry

    spec = {"geometry": {"slice": [0, 10], "names": list(CHANNEL_NAMES)},
            "embedding": {"slice": [10, 18], "dim": 8}}

    class Model(torch.nn.Module):
        output_spec = spec

        def forward(self, images):
            fields = torch.zeros(len(images), 18, 16, 16)
            fields[:, (0, 5)] = -10
            fields[:, 5, 8, 9] = 6
            return fields

    outputs = []
    for mode in ("full", "minimal"):
        pipeline, _ = live_pipeline(monkeypatch)
        pipeline.config = replace(pipeline.config, hydra_output_mode=mode)
        pipeline.cascade.args.hydra_output_mode = mode
        predictor = object.__new__(HydraPredictor)
        predictor.config = HydraInferConfig(device="cpu", amp=False, image_size=16)
        predictor.device = torch.device("cpu")
        predictor.model = Model()
        predictor.output_spec = spec
        pipeline.cascade.hydra = predictor
        pipeline.cascade.registry = StarArmTrackerRegistry()
        if mode == "minimal":
            monkeypatch.setattr(predictor, "predict_tensor_batch", lambda _images: pytest.fail("minimal live must not activate all fields"))
        records = [pipeline.predict(pixels(count), source_index=index)
                   for count, index in ((3, 0), (0, 8), (3, 9))]
        outputs.append(records)
        assert pipeline.cascade.registry.star_track_ids == (7, 8, 9)
    for full, minimal in zip(*outputs):
        # Includes the body fields: the compact maps give the same origin, area and centroid.
        assert full["stars"] == minimal["stars"]
        assert full["source_shape"] == minimal["source_shape"]
        assert full["processed_index"] == minimal["processed_index"]
    star = outputs[1][0]["stars"][0]
    assert (star["body_origin_model"], star["body_origin_code"]) == ([9.0, 8.0], 0)
    assert (star["support_area_model_px"], star["support_centroid_model"]) == (0, None)
    # Re-observing a star after an empty frame retains its real FusedTracker ID.
    for first, last in zip(outputs[1][0]["stars"], outputs[1][2]["stars"]):
        assert first["arms"][0]["track_id"] == last["arms"][0]["track_id"]


def test_live_config_and_factory_forward_opt_in_output_mode(tmp_path, monkeypatch):
    pipeline, _ = live_pipeline(monkeypatch)
    calls = []

    def construct(args, paths):
        calls.append(args)
        return pipeline.cascade

    monkeypatch.setattr(live, "verify_models", lambda _path: ({"models": {}}, {}))
    monkeypatch.setattr(live, "sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(live, "Cascade", construct)
    config = live.LiveConfig(device="cpu", hydra_output_mode="minimal", gru_execution="fixed", gru_check_interval=4)
    result = live.LivePipeline.from_manifest(tmp_path / "manifest.json", config)
    assert result.config.hydra_output_mode == calls[0].hydra_output_mode == "minimal"
    assert result.config.gru_execution == calls[0].gru_execution == "fixed"
    assert result.config.gru_check_interval == calls[0].gru_check_interval == 4
    assert calls[0].mode == "live-replay" and calls[0].detector_batch_size == 1
    args = ["--models", str(tmp_path), "--output", str(tmp_path / "out"), "--camera", "0", "--device", "cpu"]
    assert live.parse_args(args)[1].hydra_output_mode == "full"
    assert live.parse_args(args + ["--hydra-output-mode", "minimal"])[1].hydra_output_mode == "minimal"
    with pytest.raises(ValueError, match="hydra_output_mode"):
        live.LiveConfig(device="cpu", hydra_output_mode="invalid")


@pytest.mark.parametrize("frame,kwargs", [
    (np.zeros((10, 10), dtype=np.uint8), {}),
    (np.zeros((10, 10, 3), dtype=np.float32), {}),
    (np.zeros((0, 10, 3), dtype=np.uint8), {}),
    (pixels(), {"source_index": -1}),
    (pixels(), {"source_index": True}),
    (pixels(), {"source_timestamp_s": float("nan")}),
    (pixels(), {"acquired_monotonic_ns": 10**25}),
])
def test_invalid_inputs_do_not_advance_trackers(frame, kwargs, monkeypatch):
    pipeline, events = live_pipeline(monkeypatch)
    with pytest.raises(ValueError):
        pipeline.predict(frame, **kwargs)
    assert events == [] and pipeline.processed == 0
    assert pipeline.predict(pixels())["processed_index"] == 0


def test_reset_reuses_models_and_restarts_independent_stream(monkeypatch):
    pipeline, events = live_pipeline(monkeypatch)
    hydra = pipeline.cascade.hydra
    pipeline.predict(pixels(), source_index=14)
    with pytest.raises(ValueError, match="strictly increasing"):
        pipeline.predict(pixels(), source_index=14)
    pipeline.reset()
    result = pipeline.predict(pixels(), source_index=0)
    assert result["processed_index"] == 0
    assert pipeline.cascade.hydra is hydra
    assert ("reset_detector",) in events and ("reset_registry",) in events
    assert pipeline.report()["timing"]["processing_ms"]["samples"] == 1


def test_partial_inference_failure_requires_explicit_reset(monkeypatch):
    pipeline, _ = live_pipeline(monkeypatch)
    trace = benchmark.trace_arms_noodle_batch
    def fail(*_args):
        raise RuntimeError("synthetic decoder failure")
    monkeypatch.setattr(benchmark, "trace_arms_noodle_batch", fail)
    with pytest.raises(RuntimeError, match="decoder failure"):
        pipeline.predict(pixels())
    monkeypatch.setattr(benchmark, "trace_arms_noodle_batch", trace)
    with pytest.raises(RuntimeError, match="reset the stream"):
        pipeline.predict(pixels())
    pipeline.reset()
    assert pipeline.predict(pixels())["processed_index"] == 0


def test_single_stream_api_rejects_concurrent_state_changes(monkeypatch):
    pipeline, _ = live_pipeline(monkeypatch)
    pipeline._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            pipeline.predict(pixels())
        with pytest.raises(RuntimeError, match="already in use"):
            pipeline.reset()
    finally:
        pipeline._lock.release()


class FakeCapture:
    def __init__(self, count=3):
        self.count = count
        self.index = 0
        self.releases = 0
        self.buffer = pixels()

    def isOpened(self):
        return True

    def get(self, prop):
        return 25.0 if prop == cv2.CAP_PROP_FPS else (self.index - 1) * 40.0

    def read(self):
        if self.index >= self.count:
            return False, None
        self.buffer[:] = self.index
        self.index += 1
        return True, self.buffer

    def release(self):
        self.releases += 1


def test_video_sequential_ownership_timestamps_eof_and_release():
    capture = FakeCapture()
    source = live.CaptureSource(video="example.mp4", capture_factory=lambda _source: capture)
    with source:
        first = source.read()
        second = source.read()
        third = source.read()
        with pytest.raises(StopIteration):
            source.read()
    assert [first.source_index, second.source_index, third.source_index] == [0, 1, 2]
    assert first.frame_bgr[0, 0, 0] == 0  # Capture reused/mutated its own buffer.
    assert [p.source_timestamp_s for p in (first, second, third)] == [0.0, 0.04, 0.08]
    assert capture.releases == 1
    report = source.report()
    assert report["captured_frames"] == report["delivered_frames"] == 3
    assert report["overwritten_pending_frames"] == 0 and report["reader_stopped"]
    assert report["queue_policy"] == "sequential" and report["video_end_or_decode_failure"]


def test_latest_queue_is_capacity_one_and_drains_final_frame_before_error():
    queue = live.LatestFrameQueue()
    for index in range(4):
        queue.put(live.FramePacket(pixels(), index, 0, None, 0))
    queue.finish(OSError("capture disconnected"))
    assert queue.take().source_index == 3
    assert queue.overwritten == 3 and queue.delivered == 1
    with pytest.raises(OSError, match="disconnected"):
        queue.take()


def test_latest_capture_overwrite_stats_and_bounded_shutdown():
    blocked = threading.Event()
    release_read = threading.Event()
    released = threading.Event()

    class BlockingCapture(FakeCapture):
        def read(self):
            if self.index == 3:
                blocked.set()
                assert release_read.wait(5)
                return False, None
            return super().read()

        def release(self):
            super().release()
            released.set()

    capture = BlockingCapture()
    source = live.CaptureSource(camera=0, capture_factory=lambda _: capture)
    try:
        source.__enter__()
        assert blocked.wait(5)
        packet = source.read(timeout=1)
        assert packet.source_index == 2 and packet.source_timestamp_s is None
        assert packet.frame_bgr[0, 0, 0] == 2
        source.close(timeout=0)
        assert not source.report()["reader_stopped"]
        assert source.report()["overwritten_pending_frames"] == 2
        assert capture.releases == 0  # Never release concurrently with read().
    finally:
        release_read.set()
        assert released.wait(5)
        source._thread.join(5)
    assert source.report()["reader_stopped"]
    assert capture.releases == 1


def test_camera_failure_propagates_and_releases_capture():
    capture = FakeCapture(count=0)
    source = live.CaptureSource(camera=0, capture_factory=lambda _: capture)
    with source:
        with pytest.raises(OSError, match="camera read failed"):
            source.read(timeout=2)
    assert capture.releases == 1 and source.report()["reader_stopped"]


@pytest.mark.parametrize("failure", ["open", "metadata"])
def test_capture_initialization_failure_releases_resource(failure):
    class BrokenCapture(FakeCapture):
        def isOpened(self):
            return failure != "open"

        def get(self, _prop):
            raise RuntimeError("metadata failed")

    capture = BrokenCapture()
    with pytest.raises((OSError, RuntimeError)):
        with live.CaptureSource(camera=0, capture_factory=lambda _: capture):
            pytest.fail("broken capture must not open")
    assert capture.releases == 1


def test_cancel_discards_pending_and_rejects_inflight_frame():
    queue = live.LatestFrameQueue()
    packet = live.FramePacket(pixels(), 0, 0, None, 0)
    assert queue.put(packet)
    queue.cancel()
    assert not queue.put(packet)
    assert queue.discarded_on_close == 2
    with pytest.raises(StopIteration):
        queue.take()


def test_actual_opencv_video_capture_reaches_ndarray_pipeline(tmp_path, monkeypatch):
    pipeline, _ = live_pipeline(monkeypatch)
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    records = []
    with live.CaptureSource(video=path) as source:
        while True:
            try:
                packet = source.read()
            except StopIteration:
                break
            records.append(pipeline.predict(packet.frame_bgr, source_index=packet.source_index,
                                            acquired_monotonic_ns=packet.acquired_monotonic_ns))
    assert len(records) == 3
    assert [r["source_index"] for r in records] == [0, 1, 2]
    assert all(r["source_shape"] == [24, 32] and r["stars"] == [] for r in records)
    assert source.report()["captured_frames"] == 3
    assert source.report()["statistics_final"]


@pytest.mark.parametrize("exception", [RuntimeError("bad backend"), KeyboardInterrupt()])
def test_sequential_capture_released_on_consumer_exception(exception):
    capture = FakeCapture()
    with pytest.raises(type(exception)):
        with live.CaptureSource(video="example.mp4", capture_factory=lambda _: capture) as source:
            source.read()
            raise exception
    assert capture.releases == 1


@pytest.mark.parametrize("config", [{"crop_batch_size": 0}, {"crop_batch_size": 1.5},
                                     {"ram_limit_gib": float("inf")}, {"allocator_limit_gib": -1},
                                     {"device": "cpu", "starseg_half": True}])
def test_live_config_rejects_unsupported_values(config):
    with pytest.raises(ValueError):
        live.LiveConfig(**config)


def test_video_dropping_is_rejected():
    with pytest.raises(ValueError, match="sequential"):
        live.CaptureSource(video="example.mp4", queue_policy="latest")


def test_timings_have_bounded_storage_and_honest_window_quantiles():
    timing = live.RollingTiming(window=3)
    for value in range(10):
        timing.add(value)
    report = timing.report()
    assert list(timing.values) == [7, 8, 9]
    assert report["samples"] == 10 and report["mean_ms"] == 4.5
    assert report["min_ms"] == 0 and report["max_ms"] == 9
    assert report["quantile_window_samples"] == 3 and report["window_p50_ms"] == 8


def test_timing_report_snapshots_under_lock_while_producer_resumes():
    copying, allow_copy, writer_started = threading.Event(), threading.Event(), threading.Event()
    timing = live.RollingTiming(window=3)
    timing.add(1.0)

    class PausedDeque(deque):
        def __iter__(self):
            copying.set()
            assert allow_copy.wait(5)
            yield from super().__iter__()

    timing.values = PausedDeque(timing.values, maxlen=3)
    reports, errors = [], []

    def report():
        try:
            reports.append(timing.report())
        except BaseException as exc:
            errors.append(exc)

    def producer():
        writer_started.set()
        timing.add(9.0)

    reader = threading.Thread(target=report)
    writer = threading.Thread(target=producer)
    reader.start()
    assert copying.wait(5)
    try:
        # Report owns the same lock that add() uses throughout snapshotting.
        assert not timing._lock.acquire(blocking=False)
        writer.start()
        assert writer_started.wait(5)
    finally:
        allow_copy.set()
        reader.join(5)
        if writer.ident is not None:
            writer.join(5)
    assert not reader.is_alive() and not writer.is_alive() and errors == []
    assert reports[0]["samples"] == 1 and reports[0]["mean_ms"] == 1.0
    assert timing.report()["samples"] == 2 and timing.report()["mean_ms"] == 5.0


@pytest.mark.parametrize("failure", [None, "decoder", "interrupt"])
@pytest.mark.parametrize("gru_execution", ["dynamic", "fixed"])
def test_cli_video_streams_ndarrays_and_reports_cleanup(tmp_path, monkeypatch, failure, gru_execution):
    pipeline, events = live_pipeline(monkeypatch)
    capture = FakeCapture()
    # Zero-star first frame, then one and two; actual crop/preprocess/trace flow.
    def make_pipeline(_manifest, config):
        assert capture.index == 0  # Models load before capture opens.
        assert config.device == "cpu"
        assert config.gru_execution == gru_execution and config.gru_check_interval == 4
        return pipeline
    monkeypatch.setattr(live.LivePipeline, "from_manifest", make_pipeline)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    monkeypatch.setattr(cv2, "imread", lambda *_args: pytest.fail("live CLI must not write/read intermediate image files"))
    video = tmp_path / "clip.mp4"
    video.touch()
    output = tmp_path / "out"
    if failure:
        original = pipeline.predict
        def fail_on_second(*args, **kwargs):
            if pipeline.processed == 1:
                if failure == "interrupt":
                    raise KeyboardInterrupt
                raise RuntimeError("decoder failed")
            return original(*args, **kwargs)
        monkeypatch.setattr(pipeline, "predict", fail_on_second)
    arguments = ["live", "--models", str(tmp_path / "models"), "--output", str(output),
                 "--video", str(video), "--device", "cpu", "--crop-batch-size", "2",
                 "--gru-execution", gru_execution, "--gru-check-interval", "4"]
    if failure == "decoder":
        with pytest.raises(RuntimeError, match="decoder failed"):
            cli.main(arguments)
    else:
        assert cli.main(arguments) == (130 if failure == "interrupt" else 0)
    report = json.loads((output / "report.json").read_text())
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert report["status"] == {None: "complete", "decoder": "failed", "interrupt": "interrupted"}[failure]
    assert report["configuration"]["gru_execution"] == gru_execution
    assert report["configuration"]["gru_check_interval"] == 4
    assert report["measurement"]["processed_frames"] == len(rows) == (1 if failure else 3)
    assert capture.releases == 1 and report["capture"]["reader_stopped"]
    assert report["capture"]["overwritten_pending_frames"] == 0
    assert [row["source_index"] for row in rows] == list(range(len(rows)))
    assert all(row["acquisition_clock"] == "opencv_read_completed" for row in rows)
    # The video's nominal 25 fps times each source index; the time fields stay last after the CLI keys.
    assert [row["time_s"] for row in rows] == [index / 25.0 for index in range(len(rows))]
    assert all(list(row)[-5:] == ["capture_read_ms", "queue_overwritten_total_at_result", "time_s", "fps",
                                  "schema_version"] and row["fps"] == 25.0 for row in rows)
    assert report["output_schema_version"] == 4
    assert report["serialize_write"]["samples"] == len(rows)
    assert report["total_job_seconds"] > 0
