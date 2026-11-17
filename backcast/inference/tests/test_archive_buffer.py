# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""CPU scheduling tests with real crop/PIL transforms and input-sensitive models."""
from __future__ import annotations

import gc
import io
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import cv2
import numpy as np
import pytest
import torch

from deepstarnoodles_inference import archive_buffer as ab, benchmark as b, staged
from deepstarnoodles_inference.crop import CropConfig, _crop_result, _square_crop_geometry
from deepstarnoodles_inference.hydra import HydraPredictor
from deepstarnoodles_inference.noodle_tracer import NoodleTraceParams
from test_benchmark_cascade import Boxes


class FakeHydra:
    image_size = 16
    preprocess_batch = HydraPredictor.preprocess_batch

    def __init__(self):
        self.forward_batches = []

    def predict_tensor_batch_raw(self, images):
        self.forward_batches.append(len(images))
        # Input-sensitive raw fields expose frame/crop mixups without batch
        # dependent neural arithmetic obscuring the scheduling assertion.
        return images[:, 2:3].repeat(1, 18, 1, 1)

    def predict_tensor_batch(self, images):
        raw = self.predict_tensor_batch_raw(images)
        return raw, raw


class ForbiddenRegistry:
    def __getattr__(self, name):
        raise AssertionError(f"archive must not use temporal state: {name}")


def setup_pipeline(tmp_path, monkeypatch, counts, *, batch=4, window=0, workers=0,
                   budget_mib=256, bbox=None, shape=(64, 96), minimal=True, fixed=False):
    paths = []
    for i in range(len(counts)):
        path = tmp_path / f"frame{i}.png"
        assert cv2.imwrite(str(path), np.full((*shape, 3), (i + 1, 50, 100), dtype=np.uint8))
        paths.append(path)
    index = tmp_path / "index.csv"
    index.write_text("image_path\n" + "\n".join(map(str, paths)) + "\n")
    value = object.__new__(b.Cascade)
    value.args = SimpleNamespace(mode="archive", device="cpu", index=index,
        detector_batch_size=1, crop_batch_size=batch, archive_frame_window=window,
        archive_crop_buffer_mib=budget_mib, decode_workers=workers, starseg_half=False,
        hydra_fp32=False, channels_last=False, hydra_batch_size=1,
        hydra_output_mode="minimal" if minimal else "full", ram_limit_gib=24.0,
        gru_execution="fixed" if fixed else "dynamic", gru_check_interval=4)
    value.device = torch.device("cpu")
    value.crop_config = CropConfig(device="cpu", rect=False)
    value.live_cropper, value.registry = None, ForbiddenRegistry()
    value.sunflower_ids = {2}
    value.hydra, value.tracer, value.trace_params = FakeHydra(), object(), NoodleTraceParams()
    value.detector_batches, value.trace_batches = [], []

    def predict(frames, **kwargs):
        assert len(frames) == 1 and kwargs["rect"] is False
        marker = int(frames[0][0, 0, 0])
        value.detector_batches.append(marker)
        boxes = [bbox or (4 + j, 4, 12 + j, 12) for j in range(counts[marker-1])]
        polygons = [np.array([[x, y], [r, y], [r, s]], dtype=np.float32) for x, y, r, s in boxes]
        return [SimpleNamespace(boxes=Boxes(boxes, [2] * len(boxes)), masks=SimpleNamespace(xy=polygons))]

    def trace(raw, images, *_args, **_kwargs):
        value.trace_batches.append(len(raw))
        assert len(raw) == len(images)
        outputs = []
        for fields, image in zip(raw, images):
            torch.testing.assert_close(fields[0], image[2], rtol=0, atol=0)
            marker = float(fields[0, 8, 8])
            outputs.append([{"polyline": np.array([[1.0 + marker, 2.0], [3.0, 4.0]]),
                             "identity": np.asarray([marker]), "stop_reason": "base", "steps": 3,
                             "state_confidence": 0.9, "tip_score": marker, "base_score": 0.7,
                             "length_px": 2.8}])
        return outputs

    value.detector = SimpleNamespace(predict=predict)
    monkeypatch.setattr(b, "trace_arms_noodle_batch", trace)
    from deepstarnoodles_inference import noodle_tracer_fixed
    monkeypatch.setattr(noodle_tracer_fixed, "trace_arms_noodle_batch_fixed_slots", trace)
    return value, paths


def run(pipeline, count, *, profile=True):
    sink = io.StringIO()
    report = b.measure(pipeline, pipeline.args, count, sink, profile)
    return sink.getvalue(), report


@pytest.mark.parametrize("batch", [2, 4, 8])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("minimal,fixed", [(False, False), (True, True)])
def test_cross_frame_batches_preserve_exact_records_and_all_source_mapping(
    tmp_path, monkeypatch, batch, workers, minimal, fixed
):
    counts = [1, 0, 2, 1, 0, 3, 2]
    pipeline, paths = setup_pipeline(tmp_path, monkeypatch, counts, batch=batch,
                                     workers=workers, minimal=minimal, fixed=fixed)
    expected, _ = run(pipeline, len(paths))
    pipeline.args.archive_frame_window = 64
    pipeline.detector_batches.clear()
    pipeline.trace_batches.clear()
    pipeline.hydra.forward_batches.clear()
    actual, report = run(pipeline, len(paths))
    assert actual == expected
    assert pipeline.detector_batches == list(range(1, 8))
    assert pipeline.trace_batches == [batch] * (9 // batch) + ([9 % batch] if 9 % batch else [])
    assert pipeline.hydra.forward_batches == [1] * 9
    records = [json.loads(line) for line in actual.splitlines()]
    assert [r["frame"] for r in records] == list(range(7))
    assert [len(r["stars"]) for r in records] == counts
    assert all([s["detection_index"] for s in record["stars"]] == list(range(count))
               for record, count in zip(records, counts))
    assert report["star_crops"] == report["arms"] == 9
    assert report["empty_source_images"] == 2
    assert report["archive_crop_buffer"]["peak_pending_crops"] <= batch
    assert report["archive_crop_buffer"]["emitted_source_images"] == 7
    assert "not per-image completion latency" in report["source_group_latency_scope"]


def test_crowded_frame_can_span_flushes_and_combine_its_tail_with_later_frames(tmp_path, monkeypatch):
    pipeline, paths = setup_pipeline(tmp_path, monkeypatch, [9, 0, 3], batch=4)
    expected, _ = run(pipeline, 3)
    pipeline.args.archive_frame_window = 4
    pipeline.trace_batches.clear()
    actual, report = run(pipeline, 3)
    assert actual == expected and pipeline.trace_batches == [4, 4, 4]
    assert report["archive_crop_buffer"]["flush_reasons"]["crop_batch"] == 3
    assert [len(json.loads(line)["stars"]) for line in actual.splitlines()] == [9, 0, 3]


def test_frame_window_flushes_before_empty_spans_can_grow_pending_metadata(tmp_path, monkeypatch):
    pipeline, paths = setup_pipeline(tmp_path, monkeypatch, [1, 0, 0, 0, 2], batch=8, window=2)
    _, report = run(pipeline, len(paths))
    stats = report["archive_crop_buffer"]
    assert pipeline.trace_batches == [1, 2]
    assert stats["peak_pending_frames"] == 2
    assert stats["flush_reasons"]["frame_window"] == stats["flush_reasons"]["eof"] == 1


def test_all_empty_sources_are_emitted_without_neural_calls(tmp_path, monkeypatch):
    pipeline, paths = setup_pipeline(tmp_path, monkeypatch, [0] * 20, batch=8, window=2)
    text, report = run(pipeline, len(paths))
    assert len(text.splitlines()) == 20
    assert pipeline.trace_batches == pipeline.hydra.forward_batches == []
    assert report["empty_source_images"] == 20
    assert report["archive_crop_buffer"]["peak_pending_frames"] == 1
    assert report["archive_crop_buffer"]["peak_crop_bytes"] == 0


def test_byte_budget_flushes_before_creating_next_crop_and_preserves_output(tmp_path, monkeypatch):
    pipeline, paths = setup_pipeline(tmp_path, monkeypatch, [1, 1, 1], batch=8,
                                     budget_mib=1, bbox=(64, 64, 320, 320), shape=(384, 384))
    expected, _ = run(pipeline, len(paths))
    pipeline.args.archive_frame_window = 8
    pipeline.trace_batches.clear()
    original, buffers = ab._crop_result, []
    observed = ab.ArchiveCropBuffer
    class Buffer(observed):
        def __init__(self, *args):
            super().__init__(*args)
            buffers.append(self)
    def crop(*args, **kwargs):
        assert buffers[0].queue_bytes == 0  # Previous 0.75MiB crop was flushed.
        return original(*args, **kwargs)
    monkeypatch.setattr(ab, "ArchiveCropBuffer", Buffer)
    monkeypatch.setattr(ab, "_crop_result", crop)
    actual, report = run(pipeline, len(paths))
    assert actual == expected and pipeline.trace_batches == [1, 1, 1]
    stats = report["archive_crop_buffer"]
    assert stats["peak_crop_bytes"] == 512 * 512 * 3 + 3 * 2 * 4 <= 1024**2
    assert stats["flush_reasons"]["byte_budget"] == 2
    assert buffers[0].closed and not buffers[0].records and not buffers[0].queue


def test_oversized_crop_fails_before_allocation(tmp_path, monkeypatch):
    pipeline, _ = setup_pipeline(tmp_path, monkeypatch, [1], batch=8, window=8,
                                 budget_mib=1, bbox=(0, 0, 400, 400), shape=(512, 512))
    monkeypatch.setattr(ab, "_crop_result", lambda *a, **k: pytest.fail("must reject before allocation"))
    with pytest.raises(MemoryError, match="before allocation"):
        run(pipeline, 1)
    assert pipeline.trace_batches == []


@pytest.mark.parametrize("xyxy,context", [((1.5, 2.7, 14.3, 17.6), 2.0),
    ((0, 0, 2, 3), 2.0), ((-5, -2, 20, 30), 1.5)])
def test_reservation_geometry_matches_actual_crop_pixels(xyxy, context):
    frame = np.zeros((80, 100, 3), np.uint8)
    detection = dict(xyxy=xyxy, center=(0, 0), confidence=0.9, polygon=None, star_track_id=None)
    left, top, side = _square_crop_geometry(xyxy, context)
    crop = _crop_result(frame, detection, context_scale=context)
    assert (crop.crop_left, crop.crop_top, crop.crop_side) == (left, top, side)
    assert crop.crop_rgb.nbytes == side * side * 3


def test_full_source_pixels_are_released_before_reading_next_source(tmp_path, monkeypatch):
    pipeline, _ = setup_pipeline(tmp_path, monkeypatch, [1, 1, 1, 1], batch=8, window=8)
    original, references = cv2.imread, []
    def read(*args, **kwargs):
        gc.collect()
        assert all(ref() is None for ref in references)
        frame = original(*args, **kwargs)
        references.append(weakref.ref(frame))
        return frame
    monkeypatch.setattr(cv2, "imread", read)
    run(pipeline, 4)
    gc.collect()
    assert all(ref() is None for ref in references)


def test_failure_keeps_completed_prefix_and_releases_queued_pixels(tmp_path, monkeypatch):
    pipeline, _ = setup_pipeline(tmp_path, monkeypatch, [1] * 5, batch=2, window=8)
    original_trace = b.trace_arms_noodle_batch
    calls, references, buffers = [], [], []
    original_crop, original_buffer = ab._crop_result, ab.ArchiveCropBuffer
    def trace(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("GRU failure")
        return original_trace(*args, **kwargs)
    def crop(*args, **kwargs):
        result = original_crop(*args, **kwargs)
        references.append(weakref.ref(result.crop_rgb))
        return result
    class Buffer(original_buffer):
        def __init__(self, *args):
            super().__init__(*args)
            buffers.append(self)
    monkeypatch.setattr(b, "trace_arms_noodle_batch", trace)
    monkeypatch.setattr(ab, "_crop_result", crop)
    monkeypatch.setattr(ab, "ArchiveCropBuffer", Buffer)
    sink = io.StringIO()
    with pytest.raises(RuntimeError, match="GRU failure"):
        b.measure(pipeline, pipeline.args, 5, sink, False)
    assert [json.loads(line)["frame"] for line in sink.getvalue().splitlines()] == [0, 1]
    gc.collect()
    assert buffers[0].closed and not buffers[0].queue and not buffers[0].records
    assert all(ref() is None for ref in references)


def test_warmup_and_measurement_have_separate_buffers_and_exact_row_limits(tmp_path, monkeypatch):
    pipeline, _ = setup_pipeline(tmp_path, monkeypatch, [1] * 5, batch=8, window=8, workers=2)
    zero, stats = run(pipeline, 0)
    assert zero == "" and stats["archive_crop_buffer"]["emitted_source_images"] == 0
    warmup, _ = run(pipeline, 2)
    measured, report = run(pipeline, 3)
    assert [json.loads(line)["frame"] for line in warmup.splitlines()] == [0, 1]
    assert [json.loads(line)["frame"] for line in measured.splitlines()] == [0, 1, 2]
    assert pipeline.trace_batches == [2, 3]
    assert pipeline.detector_batches == [1, 2, 1, 2, 3]
    assert report["archive_decode"]["submitted"] == report["source_images"] == 3


@pytest.mark.parametrize("extra", [["--archive-frame-window", "-1"], ["--archive-frame-window", "257"],
    ["--archive-crop-buffer-mib", "0"], ["--archive-crop-buffer-mib", "1025"],
    ["--archive-frame-window", "2", "--mode", "live-replay"],
    ["--archive-frame-window", "2", "--detector-batch-size", "2"],
    ["--archive-frame-window", "2", "--crop-batch-size", "129"]])
def test_invalid_archive_buffer_options_are_rejected(extra):
    argv = ["--models", "models", "--index", "images.csv", "--output", "out", "--mode", "archive"]
    with pytest.raises(SystemExit):
        b.parse_args(argv + extra)


def test_default_is_disabled_and_staged_schedule_rejects_buffering():
    argv = ["--models", "models", "--index", "images.csv", "--output", "out", "--mode", "archive", "--warmup", "0"]
    assert b.parse_args(argv).archive_frame_window == 0
    with pytest.raises(ValueError, match="resident archive-only"):
        staged.parse_args(argv + ["--archive-frame-window", "4"])


def test_queued_polygon_has_owned_storage_and_is_charged_to_budget(tmp_path, monkeypatch):
    pipeline, _ = setup_pipeline(tmp_path, monkeypatch, [1], batch=8, window=8)
    timer = b.Timer(torch.device("cpu"), False)
    buffer = ab.ArchiveCropBuffer(pipeline, pipeline.args, timer, io.StringIO())
    backing = np.zeros((1000, 2), dtype=np.float32)
    polygon = backing[:3]
    reference = weakref.ref(backing)
    detection = dict(xyxy=(4, 4, 12, 12), center=(8, 8), confidence=.9,
                     polygon=polygon, star_track_id=None)
    buffer.admit({"frame": 0, "image_path": "x", "source_shape": [32, 32], "stars": []}, 1)
    buffer.add_crop(np.zeros((32, 32, 3), np.uint8), 0, 0, detection)
    crop = buffer.queue[0][2]
    assert crop.mask_polygon.flags.owndata
    assert not np.shares_memory(crop.mask_polygon, backing)
    assert buffer.queue_bytes == crop.crop_rgb.nbytes + crop.mask_polygon.nbytes
    del backing, polygon, detection
    gc.collect()
    assert reference() is None
    buffer.close()


def test_main_reports_buffer_configuration_and_separate_warmup(tmp_path, monkeypatch):
    from deepstarnoodles_inference.hydra import HydraInferConfig
    pipeline, _ = setup_pipeline(tmp_path, monkeypatch, [1, 1, 1], batch=8, window=4)
    reset_calls = []
    pipeline.registry = SimpleNamespace(clear=lambda: reset_calls.append(True))
    pipeline.hydra.config = HydraInferConfig(device="cpu")
    models = {}
    for name in ("starseg", "hydra", "tracer"):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        models[name] = {"path": str(path), "sha256": b.sha256(path)}
    models["hydra"]["architecture"] = "dinov3_convnext_base"
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": models}))
    def factory(args, paths):
        pipeline.args = args
        return pipeline
    monkeypatch.setattr(b, "Cascade", factory)
    output = tmp_path / "output"
    assert b.main(["--models", str(manifest), "--index", str(pipeline.args.index),
                   "--output", str(output), "--mode", "archive", "--device", "cpu",
                   "--images", "3", "--warmup", "1", "--crop-batch-size", "8",
                   "--hydra-batch-size", "1", "--archive-frame-window", "4"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete" and reset_calls == [True]
    assert report["configuration"]["archive_frame_window"] == 4
    assert report["configuration"]["archive_crop_buffer_mib"] == 256
    assert report["warmup"]["archive_crop_buffer"]["crop_batch_histogram"] == {"1": 1}
    assert report["measurement"]["archive_crop_buffer"]["crop_batch_histogram"] == {"3": 1}
    assert report["total_job_seconds"] >= report["measurement"]["elapsed_seconds"]


def test_original_detection_backing_released_before_later_empty_sources(tmp_path, monkeypatch):
    pipeline, _ = setup_pipeline(tmp_path, monkeypatch, [1, 0, 0], batch=8, window=8)
    original = pipeline.detect_frames
    references, checked_sources = [], []

    def detect(frames, start_index, timer, *, source_paths=None):
        gc.collect()
        if start_index:
            # The buffered crop owns its small polygon copy; the original
            # descriptor must not pin this backing array across empty frames.
            assert references[0]() is None
            checked_sources.append(start_index)
        result = original(frames, start_index, timer, source_paths=source_paths)
        if start_index == 0:
            backing = np.zeros((100_000, 2), dtype=np.float32)
            references.append(weakref.ref(backing))
            result[2][0][2]["polygon"] = backing[:3]
        return result

    pipeline.detect_frames = detect
    output, report = run(pipeline, 3)
    assert checked_sources == [1, 2]
    assert [len(json.loads(line)["stars"]) for line in output.splitlines()] == [1, 0, 0]
    assert report["archive_crop_buffer"]["peak_crop_bytes"] == 16 * 16 * 3 + 3 * 2 * 4
    gc.collect()
    assert references[0]() is None
