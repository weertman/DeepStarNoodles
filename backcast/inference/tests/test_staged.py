"""CPU orchestration tests using real crop transforms and deterministic fake models.

These verify spool fidelity and scheduling, not numerical GPU model parity.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import cv2
import numpy as np
import pytest
import torch
import ultralytics

from deepstarnoodles_inference import benchmark as b
from deepstarnoodles_inference import staged
from deepstarnoodles_inference import noodle_tracer_fixed
from deepstarnoodles_inference.hydra import HydraPredictor as RealHydraPredictor


def arguments(**overrides):
    values = dict(device="cpu", mode="archive", detector_batch_size=2, crop_batch_size=2,
                  channels_last=False, starseg_half=False, hydra_fp32=False,
                  ram_limit_gib=1024.0, max_spool_gib=0.1)
    return SimpleNamespace(**(values | overrides))


def spool_state():
    return {"live_bytes": 0, "written_bytes": 0, "peak_bytes": 0}


@pytest.fixture
def frames(tmp_path):
    result = []
    # Markers determine detections; distinct RGB channels expose BGR/RGB errors.
    for index, marker in enumerate((51, 79, 0)):
        image = np.full((48, 64, 3), (marker, marker + 10, marker + 20), dtype=np.uint8)
        path = tmp_path / f"frame_{index}.png"
        assert cv2.imwrite(str(path), image)
        result.append(path)
    return result


@pytest.fixture
def fake_models(monkeypatch):
    witness = SimpleNamespace(events=[], detector_refs=[], result_refs=[], hydra_refs=[],
                              tracer_refs=[], traced_centers=[], enforce_exclusive=True,
                              forbid_full_activations=False, fixed_intervals=[])

    def assert_released(refs, stage):
        if witness.enforce_exclusive:
            gc.collect()
            assert all(ref() is None for ref in refs), f"prior {stage} object survived stage unload"

    class Boxes:
        def __init__(self, marker):
            if marker == 51:
                boxes = [(0, 0, 16, 16), (32, 16, 48, 32), (2, 2, 6, 6)]
                classes = [0, 0, 1]
            elif marker == 79:
                boxes, classes = [(20, 12, 36, 28)], [0]
            else:
                boxes, classes = [], []
            self.xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            self.cls = torch.tensor(classes, dtype=torch.float32)
            self.conf = torch.full((len(classes),), 0.875)

        def __len__(self):
            return len(self.cls)

    class Result:
        def __init__(self, detector, marker):
            # A surviving Result deliberately keeps its detector alive; this
            # catches final-loop-variable references, not just `del detector`.
            self.owner = detector
            self.boxes = Boxes(marker)
            self.masks = SimpleNamespace(xy=[np.array([[x, y], [r, y], [r, s]], dtype=np.float32)
                                            for x, y, r, s in self.boxes.xyxy.tolist()])
            witness.result_refs.append(weakref.ref(self))

    class Detector:
        names = {0: "sunflower_star", 1: "unrelated_species"}

        def __init__(self, path):
            assert_released(witness.tracer_refs, "tracer")
            witness.events.append("detector_load")
            witness.detector_refs.append(weakref.ref(self))

        def predict(self, images, **kwargs):
            assert kwargs["rect"] is False
            assert kwargs["imgsz"] == 1024
            witness.events.append(("detect", len(images)))
            return [Result(self, int(image[0, 0, 0])) for image in images]

    class Hydra:
        image_size = 512
        preprocess_batch = RealHydraPredictor.preprocess_batch

        def __init__(self, path, config):
            assert_released(witness.detector_refs, "detector")
            assert_released(witness.result_refs, "detector result")
            witness.events.append("hydra_load")
            witness.hydra_refs.append(weakref.ref(self))
            self.config = config
            self.model = torch.nn.Identity()

        def predict_tensor_batch(self, images):
            assert not witness.forbid_full_activations, "minimal archive must not request full activations"
            raw = self.predict_tensor_batch_raw(images)
            return raw, raw + 1.0

        def predict_tensor_batch_raw(self, images):
            witness.events.append(("hydra_forward", len(images)))
            # Eighteen channels like Hydra: support (0) and mouth (5) feed the body fields.
            raw = (images[:, :1] + 0.125).repeat(1, 18, 1, 1)
            raw._model_owner = self
            return raw

    class Tracer:
        hidden_size = 128

        def __init__(self):
            witness.tracer_refs.append(weakref.ref(self))

    def load_tracer(path, *, device):
        assert_released(witness.hydra_refs, "hydra")
        witness.events.append("tracer_load")
        return Tracer(), SimpleNamespace(), {}, {}

    def trace(raw, images, tracer, params):
        assert raw.dtype == images.dtype == torch.float32
        assert tuple(images.shape[1:]) == (3, 512, 512)
        torch.testing.assert_close(raw, (images[:, :1] + 0.125).repeat(1, 18, 1, 1), rtol=0, atol=0)
        witness.events.append(("trace", len(images)))
        answer = []
        for field, image in zip(raw, images):
            center = tuple(float(x) for x in image[:, 256, 256])
            witness.traced_centers.append(center)
            answer.append([{"polyline": np.array([[0, 0], [256, 256], [511, 511]], dtype=np.float32),
                            "identity": np.array([float(field[0, 256, 256]), *center], dtype=np.float32),
                            "stop_reason": "base", "steps": 7, "state_confidence": 0.9,
                            "tip_score": 0.8, "base_score": 0.7, "length_px": 12.5}])
        return answer

    monkeypatch.setattr(ultralytics, "YOLO", Detector)
    def fixed(raw, images, tracer, params, *, check_interval):
        witness.fixed_intervals.append(check_interval)
        return trace(raw, images, tracer, params)
    monkeypatch.setattr(noodle_tracer_fixed, "trace_arms_noodle_batch_fixed_slots", fixed)
    for module in (staged, b):
        monkeypatch.setattr(module, "HydraPredictor", Hydra)
        monkeypatch.setattr(module, "load_noodle_tracer_checkpoint", load_tracer)
        monkeypatch.setattr(module, "trace_arms_noodle_batch", trace)
    return witness


@pytest.fixture
def model_paths():
    return {name: Path(f"fake_{name}.pt") for name in ("starseg", "hydra", "tracer")}


@pytest.mark.parametrize("crop_batch_size", [1, 2, 4])
@pytest.mark.parametrize("hydra_output_mode", ["full", "minimal"])
@pytest.mark.parametrize("gru_execution", ["dynamic", "fixed"])
def test_staged_matches_resident_archive_and_releases_each_model(
    tmp_path, frames, fake_models, model_paths, crop_batch_size, hydra_output_mode, gru_execution
):
    args = arguments(crop_batch_size=crop_batch_size, hydra_output_mode=hydra_output_mode,
                     gru_execution=gru_execution, gru_check_interval=4)
    fake_models.forbid_full_activations = hydra_output_mode == "minimal"
    state = spool_state()
    spool = tmp_path / "spool"
    spool.mkdir()
    timer = b.Timer(torch.device("cpu"), False)
    records, crops, arms = staged.run_shard(args, model_paths, frames, 10, spool, timer, state)
    assert (crops, arms) == (3, 3)
    assert [r["frame"] for r in records] == [10, 11, 12]
    assert [len(r["stars"]) for r in records] == [2, 1, 0]
    assert [s["detection_index"] for s in records[0]["stars"]] == [0, 1]
    first = records[0]["stars"][0]
    assert first["crop"]["left"] == first["crop"]["top"] == -8.0
    assert first["arms"][0]["polyline_source"][1] == [8.0, 8.0]
    assert first["mask_polygon_source"] == [[0.0, 0.0], [16.0, 0.0], [16.0, 16.0]]
    assert first["arms"][0]["steps"] == 7
    assert state["live_bytes"] == 0 and state["peak_bytes"] > 0
    assert state["written_bytes"] > state["peak_bytes"]
    assert list(spool.iterdir()) == []
    gc.collect()
    assert all(ref() is None for ref in fake_models.detector_refs + fake_models.hydra_refs + fake_models.tracer_refs)
    assert fake_models.traced_centers[0] == pytest.approx((71 / 255, 61 / 255, 51 / 255))
    assert fake_models.fixed_intervals == ([4] * ((3 + crop_batch_size - 1) // crop_batch_size)
                                          if gru_execution == "fixed" else [])

    fake_models.enforce_exclusive = False
    args.gru_execution = "dynamic"
    resident = b.Cascade(args, model_paths)
    expected, expected_crops, expected_arms = [], 0, 0
    for group in b.chunks(frames, args.detector_batch_size):
        group_records, group_crops, group_arms = resident.run_group(group, 10 + len(expected), timer)
        expected.extend(group_records)
        expected_crops += group_crops
        expected_arms += group_arms
    assert (records, crops, arms) == (expected, expected_crops, expected_arms)


def test_empty_detections_skip_hydra_and_tracer(tmp_path, frames, fake_models, model_paths):
    state = spool_state()
    records, crops, arms = staged.run_shard(arguments(), model_paths, [frames[-1]], 20,
                                          tmp_path, b.Timer(torch.device("cpu"), False), state)
    assert (crops, arms) == (0, 0)
    assert records == [{"frame": 20, "image_path": str(frames[-1]), "source_shape": [48, 64], "stars": []}]
    assert "hydra_load" not in fake_models.events and "tracer_load" not in fake_models.events
    assert state == spool_state()
    assert all(ref() is None for ref in fake_models.detector_refs + fake_models.result_refs)


def test_spool_budget_rejects_write_before_exceeding_limit(tmp_path):
    value = np.ones((3, 16, 16), dtype=np.float32)
    args = arguments(max_spool_gib=(value.nbytes + 256) / 1024**3)
    state = spool_state()
    first, second = tmp_path / "first.npy", tmp_path / "second.npy"
    staged.save_array(first, value, state, args)
    before = dict(state)
    with pytest.raises(MemoryError, match="max-spool-gib"):
        staged.save_array(second, value, state, args)
    assert not second.exists()
    assert state == before
    assert state["live_bytes"] == first.stat().st_size <= args.max_spool_gib * 1024**3


def test_hydra_spool_limit_stops_before_loading_tracer(tmp_path, frames, fake_models, model_paths):
    # Enough for uint8 crops, insufficient for a float32 dense field.
    args = arguments(max_spool_gib=20_000 / 1024**3)
    state = spool_state()
    with pytest.raises(MemoryError, match="max-spool-gib"):
        staged.run_shard(args, model_paths, frames[:1], 0, tmp_path,
                         b.Timer(torch.device("cpu"), False), state)
    assert "hydra_load" in fake_models.events
    assert "tracer_load" not in fake_models.events
    assert sum(path.stat().st_size for path in tmp_path.glob("*.npy")) <= 20_000


def write_cli_inputs(tmp_path, frames):
    models = {}
    for name in ("starseg", "hydra", "tracer"):
        path = tmp_path / f"{name}.pt"
        path.write_text(f"fake {name}")
        models[name] = {"path": path.name, "sha256": b.sha256(path)}
    models["hydra"]["architecture"] = "dinov3_convnext_base"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": models}))
    index = tmp_path / "frames.csv"
    index.write_text("image_path\n" + "".join(f"{p.name}\n" for p in frames))
    output = tmp_path / "results"
    argv = ["--models", str(manifest), "--index", str(index), "--output", str(output),
            "--mode", "archive", "--device", "cpu", "--images", str(len(frames)), "--warmup", "0",
            "--shard-size", "2", "--detector-batch-size", "2", "--crop-batch-size", "2"]
    return argv, output


@pytest.mark.parametrize("gru_execution", ["dynamic", "fixed"])
def test_main_streams_multiple_shards_and_removes_intermediates(tmp_path, frames, fake_models, gru_execution):
    argv, output = write_cli_inputs(tmp_path, frames)
    argv += ["--gru-execution", gru_execution, "--gru-check-interval", "4"]
    assert staged.main(argv) == 0
    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [record["frame"] for record in records] == [0, 1, 2]
    assert [len(record["stars"]) for record in records] == [2, 1, 0]
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["configuration"]["gru_execution"] == gru_execution
    assert report["configuration"]["gru_check_interval"] == 4
    assert report["measurement"]["source_images"] == 3
    assert report["measurement"]["star_crops"] == 3
    assert report["measurement"]["empty_source_images"] == 1
    assert report["spool"]["live_bytes"] == 0
    assert not list(output.glob("spool_*"))
    assert fake_models.events.count("detector_load") == 2
    assert fake_models.events.count("hydra_load") == fake_models.events.count("tracer_load") == 1


def test_failed_shard_records_failure_and_removes_temporary_files(tmp_path, frames, fake_models):
    argv, output = write_cli_inputs(tmp_path, frames)
    argv += ["--max-spool-gib", str(20_000 / 1024**3)]
    with pytest.raises(MemoryError, match="max-spool-gib"):
        staged.main(argv)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert "MemoryError" in report["error"]
    assert not list(output.glob("spool_*"))
    assert (output / "predictions.jsonl").read_text() == ""


@pytest.mark.parametrize("bad_args", [["--mode", "live-replay"], ["--warmup", "1"],
                                     ["--shard-size", "0"], ["--max-spool-gib", "nan"]])
def test_cli_rejects_incompatible_staged_settings(bad_args):
    argv = ["--models", "manifest.json", "--index", "frames.csv", "--output", "new-results",
            "--mode", "archive", "--warmup", "0", "--device", "cpu"]
    with pytest.raises(ValueError):
        staged.parse_args(argv + bad_args)
