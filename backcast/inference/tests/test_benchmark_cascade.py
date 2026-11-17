# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from deepstarnoodles_inference import benchmark
from deepstarnoodles_inference.body_fields import host_body_fields, host_landmarks
from deepstarnoodles_inference.crop import CropConfig
from deepstarnoodles_inference.hydra import HydraInferConfig
from deepstarnoodles_inference.noodle_tracer import NoodleTraceParams
from deepstarnoodles_inference.records import FRAME_TAIL, OUTPUT_SCHEMA_VERSION, STAR_FIELDS


class Boxes:
    def __init__(self, coordinates, classes):
        self.xyxy = torch.tensor(coordinates, dtype=torch.float32).reshape(-1, 4)
        self.cls = torch.tensor(classes, dtype=torch.float32)
        self.conf = torch.full((len(classes),), 0.9)

    def __len__(self):
        return len(self.cls)


class Hydra:
    image_size = 16

    def __init__(self):
        self.crops = []
        self.batches = []

    def preprocess_batch(self, crops):
        self.crops.extend(crop.copy() for crop in crops)
        self.batches.append(len(crops))
        return torch.zeros(len(crops), 3, 16, 16)

    def predict_tensor_batch(self, images):
        fields = self.predict_tensor_batch_raw(images)
        return fields, fields

    def predict_tensor_batch_raw(self, images):
        return torch.zeros(len(images), 18, 16, 16)

    def tracking_maps(self, raw):
        return raw[:, (0, 5)]


def make_pipeline(monkeypatch, *, detector_batch=3, crop_batch=2):
    pipeline = object.__new__(benchmark.Cascade)
    pipeline.args = SimpleNamespace(
        detector_batch_size=detector_batch, crop_batch_size=crop_batch,
        device="cpu", starseg_half=False, channels_last=False,
        ram_limit_gib=24.0,
    )
    pipeline.device = torch.device("cpu")
    pipeline.crop_config = CropConfig(device="cpu")
    pipeline.hydra = Hydra()
    pipeline.tracer = object()
    pipeline.trace_params = NoodleTraceParams()
    pipeline.sunflower_ids = {2}
    pipeline.live_cropper = None

    def trace(raw, images, *_args):
        assert len(raw) == len(images)
        return [[{"polyline": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                  "identity": np.zeros(8), "stop_reason": "base", "steps": 3,
                  "state_confidence": 0.9, "tip_score": 0.8, "base_score": 0.7,
                  "length_px": 2.8}] for _ in raw]

    monkeypatch.setattr(benchmark, "trace_arms_noodle_batch", trace)
    return pipeline


def write_images(tmp_path: Path, count=4):
    paths = []
    for index in range(count):
        path = tmp_path / f"image{index}.png"
        pixels = np.empty((20, 30, 3), dtype=np.uint8)
        pixels[:] = (index + 1, 20, 30)
        pixels[4:12, 4:12] = (index + 1, 40, 60)
        assert cv2.imwrite(str(path), pixels)
        paths.append(path)
    index = tmp_path / "images.csv"
    index.write_text("image_path\n" + "\n".join(map(str, paths)) + "\n")
    return paths, index


@pytest.mark.parametrize("hydra_output_mode", ["full", "minimal"])
def test_archive_detects_crops_all_stars_without_temporal_tracking(tmp_path, monkeypatch, hydra_output_mode):
    paths, index = write_images(tmp_path)
    pipeline = make_pipeline(monkeypatch)
    pipeline.args.index = index
    pipeline.args.hydra_output_mode = hydra_output_mode
    pipeline.args.fps = 4.0
    if hydra_output_mode == "minimal":
        def forbidden(_images):
            raise AssertionError("archive minimal mode must not request full activations")
        pipeline.hydra.predict_tensor_batch = forbidden

    class ForbiddenRegistry:
        def __getattr__(self, name):
            raise AssertionError(f"archive must not use temporal registry: {name}")

    pipeline.registry = ForbiddenRegistry()
    detector_batches = []

    def predict(frames, **kwargs):
        detector_batches.append(len(frames))
        assert kwargs["rect"] is False
        results = []
        for frame in frames:
            source = int(frame[0, 0, 0]) - 1
            count = {0: 2, 1: 0, 2: 3, 3: 0}[source]
            coordinates = [[4, 4, 12, 12]] * count + [[0, 0, 20, 20]]
            results.append(SimpleNamespace(boxes=Boxes(coordinates, [2] * count + [0]), masks=None))
        return results

    pipeline.detector = SimpleNamespace(predict=predict)
    sink = io.StringIO()
    report = benchmark.measure(pipeline, pipeline.args, len(paths), sink, profile=True)
    records = [json.loads(line) for line in sink.getvalue().splitlines()]

    assert detector_batches == [3, 1]
    assert pipeline.hydra.batches == [2, 2, 1]
    assert [len(record["stars"]) for record in records] == [2, 0, 3, 0]
    assert [record["frame"] for record in records] == [0, 1, 2, 3]
    assert report["source_images"] == 4
    assert report["star_crops"] == report["arms"] == 5
    assert report["empty_source_images"] == 2
    assert report["source_group_latency"]["samples"] == 2
    assert report["stages"]["hydra"]["samples"] == 3
    expected = cv2.cvtColor(cv2.imread(str(paths[0]))[:16, :16], cv2.COLOR_BGR2RGB)
    assert np.array_equal(pipeline.hydra.crops[0], expected)
    assert all(crop.shape == (16, 16, 3) for crop in pipeline.hydra.crops)
    arm = records[0]["stars"][0]["arms"][0]
    assert arm["polyline_source"] == [[1.0, 2.0], [3.0, 4.0]]
    assert arm["arm_index"] == 0 and "track_id" not in arm
    assert arm["stop_reason"] == "base" and arm["steps"] == 3
    assert all(star["star_track_id"] is None for record in records for star in record["stars"])
    # Schema 3 and 4: body fields and landmarks after the arms of every star, time fields last in every record.
    maps = np.full((16, 16), 0.5 if hydra_output_mode == "minimal" else 0.0, np.float32)
    # Landmarks always come from the raw fields (all zero logits: p 0.5 everywhere, so the first pixel).
    expected = {**host_body_fields(maps, maps), "landmarks_model": host_landmarks(np.full((3, 16, 16), 0.5, np.float32))}
    assert expected["landmarks_model"]["butthole"] == [0.0, 0.0, 0.5]
    for record in records:
        assert list(record)[-3:] == list(FRAME_TAIL)
        assert (record["time_s"], record["fps"], record["schema_version"]) == (record["frame"] / 4, 4.0,
                                                                              OUTPUT_SCHEMA_VERSION)
        for star in record["stars"]:
            assert list(star)[-len(STAR_FIELDS) - 1:] == ["arms", *STAR_FIELDS]
            assert {key: star[key] for key in STAR_FIELDS} == expected
    assert expected["body_origin_code"] == (0 if hydra_output_mode == "minimal" else 2)


@pytest.mark.parametrize("hydra_output_mode", ["full", "minimal"])
def test_live_replay_tracks_each_frame_before_reading_next_and_resets(tmp_path, monkeypatch, hydra_output_mode):
    paths, index = write_images(tmp_path, count=2)
    pipeline = make_pipeline(monkeypatch, detector_batch=1)
    pipeline.args.index = index
    pipeline.args.hydra_output_mode = hydra_output_mode
    events = []
    imread = cv2.imread

    def read(path, flags):
        events.append(("read", Path(path).name))
        return imread(path, flags)

    class Cropper:
        def track_crops(self, frame):
            from dataclasses import replace
            crop = benchmark._crop_result(frame, {"xyxy": (4, 4, 12, 12),
                "center": (8, 8), "confidence": 0.9, "polygon": None,
                "star_track_id": None}, context_scale=2.0)
            return [replace(crop, star_track_id=7)]

        def reset(self):
            events.append(("reset_cropper",))

    class Registry:
        def track(self, star, frame, _activated, arms, **kwargs):
            assert _activated.shape[0] == (2 if hydra_output_mode == "minimal" else 18)
            assert kwargs == ({"support_mouth_only": True} if hydra_output_mode == "minimal" else {})
            events.append(("track", star, frame))
            return [42] * len(arms)

        def prune(self, frame):
            events.append(("prune", frame))

        def clear(self):
            events.append(("clear_registry",))

    pipeline.live_cropper, pipeline.registry = Cropper(), Registry()
    monkeypatch.setattr(benchmark.cv2, "imread", read)
    with pytest.raises(ValueError, match="exactly one source frame"):
        pipeline.run_group(paths, 0, benchmark.Timer(torch.device("cpu"), False))
    assert events == []
    sink = io.StringIO()
    benchmark.measure(pipeline, pipeline.args, 2, sink, profile=False)
    pipeline.reset()
    assert events == [("read", paths[0].name), ("track", 7, 0), ("prune", 0),
                      ("read", paths[1].name), ("track", 7, 1), ("prune", 1),
                      ("reset_cropper",), ("clear_registry",)]
    for line in sink.getvalue().splitlines():
        record = json.loads(line)
        star = record["stars"][0]
        assert star["star_track_id"] == 7
        assert star["arms"][0]["track_id"] == 42
        assert "arm_index" not in star["arms"][0]
        assert list(star)[-len(STAR_FIELDS):] == list(STAR_FIELDS)
        assert record["time_s"] is None and record["fps"] is None


@pytest.mark.parametrize("mode", ["archive", "live-replay"])
def test_cascade_forwards_same_detector_geometry_and_precision(mode, tmp_path, monkeypatch):
    import ultralytics

    paths, _ = write_images(tmp_path, count=1)
    calls = []

    class Detector:
        names = {2: "Pycnopodia_helianthoides"}

        def __init__(self, _path):
            pass

        def predict(self, _frames, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=Boxes([], []), masks=None)]

        track = predict

    monkeypatch.setattr(ultralytics, "YOLO", Detector)
    monkeypatch.setattr(benchmark, "HydraPredictor", lambda *_args: Hydra())
    monkeypatch.setattr(benchmark, "load_noodle_tracer_checkpoint",
                        lambda *_args, **_kwargs: (SimpleNamespace(hidden_size=128), NoodleTraceParams(), {}, None))
    args = SimpleNamespace(device="cuda:0", mode=mode, starseg_half=True,
                           hydra_fp32=False, channels_last=False, crop_batch_size=2)
    pipeline = benchmark.Cascade(args, {key: tmp_path / key for key in ("starseg", "hydra", "tracer")})
    # Empty detections exercise the production cropper and forwarding path
    # without creating CUDA tensors or loading model weights.
    records, crops, arms = pipeline.run_group(paths, 0, benchmark.Timer(torch.device("cpu"), False))
    assert crops == arms == 0 and records[0]["stars"] == []
    assert len(calls) == 1
    assert calls[0]["half"] is True and calls[0]["rect"] is False
    assert calls[0]["imgsz"] == 1024
    assert pipeline.hydra.batches == []


@pytest.mark.parametrize("fail_measurement", [False, True])
@pytest.mark.parametrize("gru_execution", ["dynamic", "fixed"])
def test_main_separates_warmup_resets_state_and_records_outcome(tmp_path, monkeypatch, fail_measurement, gru_execution):
    _, index = write_images(tmp_path, count=2)
    manifest = tmp_path / "models.json"
    models = {}
    for name in ("starseg", "hydra", "tracer"):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        models[name] = {"path": str(path), "sha256": benchmark.sha256(path)}
    models["hydra"]["architecture"] = "dinov3_convnext_base"
    manifest.write_text(json.dumps({"schema_version": 1, "models": models}))
    events = []

    class Pipeline:
        device = torch.device("cpu")
        trace_params = NoodleTraceParams()
        crop_config = CropConfig(device="cpu", rect=False)
        hydra = SimpleNamespace(config=HydraInferConfig(device="cpu"))

        def __init__(self, *_args):
            self.reset_done = False

        def run_group(self, paths, start_index, _timer):
            events.append(("group", start_index, len(paths), self.reset_done))
            if fail_measurement and self.reset_done:
                raise RuntimeError("mock inference failed")
            return [{"frame": start_index + i, "stars": []} for i, _ in enumerate(paths)], 0, 0

        def reset(self):
            events.append(("reset",))
            self.reset_done = True

    monkeypatch.setattr(benchmark, "Cascade", Pipeline)
    output = tmp_path / "output"
    argv = ["--models", str(manifest), "--index", str(index), "--output", str(output),
            "--mode", "live-replay", "--device", "cpu", "--images", "2", "--warmup", "1",
            "--gru-execution", gru_execution, "--gru-check-interval", "4"]
    if fail_measurement:
        with pytest.raises(RuntimeError, match="mock inference failed"):
            benchmark.main(argv)
    else:
        assert benchmark.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    assert events[:3] == [("group", 0, 1, False), ("reset",), ("group", 0, 1, True)]
    assert report["warmup"]["source_images"] == 1
    assert len(report["input_files"]) == 2
    assert all(item["sha256"] == benchmark.sha256(item["path"]) for item in report["input_files"])
    assert report["status"] == ("failed" if fail_measurement else "complete")
    assert report["configuration"]["gru_execution"] == gru_execution
    assert report["configuration"]["gru_check_interval"] == 4
    if fail_measurement:
        assert report["error"] == "RuntimeError: mock inference failed"
    else:
        records = (output / "predictions.jsonl").read_text().splitlines()
        assert [json.loads(line)["frame"] for line in records] == [0, 1]
        assert report["measurement"]["source_images"] == 2
        assert report["measurement"]["stages"] == {}

    # The checkpoint hashes alone must not authorize a different preprocessing
    # or weight selection than the runner can actually execute.
    payload = json.loads(manifest.read_text())
    payload["models"]["hydra"]["prefer_ema"] = False
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="EMA"):
        benchmark.verify_models(manifest)


def test_compute_crops_returns_body_fields_without_changing_tracking_input(monkeypatch):
    pipeline = make_pipeline(monkeypatch, crop_batch=3)
    pipeline.args.hydra_output_mode = "minimal"
    pipeline.live_cropper = object()
    images = torch.zeros(3, 3, 16, 16)
    activated, decoded, bodies = pipeline.compute_crops([None] * 3, images, benchmark.Timer(torch.device("cpu"), False))
    assert activated.shape == (3, 2, 16, 16) and len(decoded) == len(bodies) == 3
    # The fake's compact tracking maps are zeros (not activated), and the body fields read those maps.
    maps = np.zeros((16, 16), np.float32)
    # Landmarks come from the fake's raw fields: zero logits, p 0.5 everywhere.
    landmarks = host_landmarks(np.full((3, 16, 16), 0.5, np.float32))
    assert bodies == [{**host_body_fields(maps, maps), "landmarks_model": landmarks}] * 3


def test_fps_option_is_validated():
    base = ["--models", "m", "--index", "i.csv", "--output", "o", "--mode", "archive"]
    assert benchmark.parse_args(base).fps is None
    assert benchmark.parse_args(base + ["--fps", "2"]).fps == 2.0
    for bad in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            benchmark.parse_args(base + ["--fps", bad])
