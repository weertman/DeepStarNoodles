# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Exactness of device peak candidates, copy-free patches, CUDA-graph replay and detect-ahead."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json

import numpy as np
import pytest
import torch

from deepstarnoodles_inference import benchmark, live
from deepstarnoodles_inference.decode import find_peaks
from deepstarnoodles_inference.device_peaks import find_peaks_device
from deepstarnoodles_inference.noodle_tracer import (
    NoodleTraceParams, NoodleTracer, extract_centered_patches,
)
from deepstarnoodles_inference.noodle_tracer_fixed import trace_arms_noodle_batch_fixed_slots
from deepstarnoodles_inference.noodle_tracer_graph import dense_patches_one_image
from test_benchmark_cascade import Boxes, make_pipeline, write_images
from test_fixed_slots import ScriptedTracer, assert_outputs_close, fields_with_seeds, params

CUDA = torch.cuda.is_available()


def heatmaps(seed):
    rng = np.random.default_rng(seed)
    heat = rng.random((47, 61), dtype=np.float32)
    heat[10:14, 20:25] = np.float32(0.8)      # exact plateau and ties
    heat[0, :] = np.float32(1.0)              # saturated border
    heat[30, 40] = np.float32(0.2)            # exactly at a threshold
    heat[31, 40] = np.nextafter(np.float32(0.2), np.float32(0))
    heat[5, 5] = heat[5, 7] = np.float32(0.95)  # equal peaks within suppression radius
    return heat


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("threshold,radius", [(0.2, 4), (0.25, 3), (0.0, 1), (0.97, 2)])
def test_device_peaks_match_host_find_peaks(seed, threshold, radius):
    heat = heatmaps(seed)
    expected = find_peaks(heat, threshold, radius)
    observed = find_peaks_device(torch.from_numpy(heat), threshold, radius)
    assert observed.dtype == expected.dtype and np.array_equal(observed, expected)
    if CUDA:
        assert np.array_equal(find_peaks_device(torch.from_numpy(heat).cuda(), threshold, radius), expected)


@pytest.mark.parametrize("override", [{}, {"support_gated_peaks": True}, {"max_tips": 2},
                                      {"tip_seed_separation": 20.0}, {"suppress_duplicates": True}])
def test_device_seed_inputs_match_host_fixed_slots(override):
    raw, images = fields_with_seeds()
    config = replace(params(), **override)
    host = trace_arms_noodle_batch_fixed_slots(raw, images, ScriptedTracer(), config, check_interval=1)
    device = trace_arms_noodle_batch_fixed_slots(raw, images, ScriptedTracer(), config, check_interval=1,
                                                device_peaks=True)
    assert_outputs_close(host, device)
    assert sum(len(arms) for arms in device) > 0


def neural_fields(seed, batch=2, size=72):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn((batch, 18, size, size), generator=generator) * 2
    images = torch.rand((batch, 3, size, size), generator=generator)
    for image in range(batch):
        for k in range(5 + image):
            y, x = 6 + 11 * k % (size - 12), 6 + 17 * k % (size - 12)
            raw[image, 9, y, x] = 6.0
            raw[image, 0, y - 3:y + 4, x - 3:x + 4] = 4.0
    return raw, images


def neural_tracer(device):
    torch.manual_seed(0)
    return NoodleTracer(pretrained_rgb=False, dense_patch_size=17).to(device).eval()


def neural_params(**kwargs):
    return NoodleTraceParams(dense_patch_size=17, max_steps=9, min_path_px=0.0, loop_history=3,
                             emit_lost=True, **kwargs)


def test_device_peaks_match_host_with_real_layers_on_cpu():
    raw, images = neural_fields(1)
    model = neural_tracer("cpu")
    for config in (neural_params(), neural_params(support_gated_peaks=True)):
        host = trace_arms_noodle_batch_fixed_slots(raw, images, model, config, check_interval=1)
        device = trace_arms_noodle_batch_fixed_slots(raw, images, model, config, check_interval=1,
                                                    device_peaks=True)
        assert_outputs_close(host, device)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if CUDA else ["cpu"])
def test_one_image_dense_patches_are_identical(device):
    generator = torch.Generator().manual_seed(3)
    fields = torch.randn((1, 18, 40, 52), generator=generator).to(device)
    positions = (torch.rand((9, 2), generator=generator) * 70 - 9).to(device)
    positions[0] = torch.tensor([0.0, 0.0])
    positions[1] = torch.tensor([51.0, 39.0])
    for count in (1, 9):
        expected = extract_centered_patches(fields, positions[:count], None, 17)
        observed = dense_patches_one_image(fields, positions[:count], 17)
        assert torch.equal(expected, observed) and observed.is_contiguous()
        assert observed.stride() == expected.contiguous().stride() or count == 1


def digest(outputs):
    value = hashlib.sha256()
    for arms in outputs:
        for arm in arms:
            for key in sorted(arm):
                item = arm[key]
                value.update(key.encode())
                value.update(item.tobytes() if isinstance(item, np.ndarray) else repr(item).encode())
    return value.hexdigest()


@pytest.mark.skipif(not CUDA, reason="CUDA graphs require CUDA")
@pytest.mark.parametrize("graph_first_step", [False, True])
@pytest.mark.parametrize("steps_per_replay", [1, 3])
def test_graph_rollout_reproduces_fixed_slots_bytes(steps_per_replay, graph_first_step):
    from deepstarnoodles_inference.noodle_tracer_graph import FixedSlotGraphRollout

    model = neural_tracer("cuda")
    config = neural_params()
    rollout = FixedSlotGraphRollout(model, config, steps_per_replay=steps_per_replay, max_graphs=2,
                                    graph_first_step=graph_first_step)
    with torch.inference_mode():
        for seed in range(4):
            raw, images = neural_fields(seed, batch=1 + seed % 2)
            raw, images = raw.cuda(), images.cuda()
            reference = trace_arms_noodle_batch_fixed_slots(raw, images, model, config, check_interval=1)
            diagnostics = {}
            observed = trace_arms_noodle_batch_fixed_slots(
                raw, images, model, config, check_interval=steps_per_replay, diagnostics=diagnostics,
                graph_rollout=rollout, device_peaks=True)
            assert digest(observed) == digest(reference)
            assert diagnostics["execution"] == "fixed_slots_cuda_graph"
            assert diagnostics["rollout_steps"] <= config.max_steps
    assert rollout.captures >= 3 and rollout.evictions >= 1 and len(rollout._entries) <= 2
    assert all((entry.first_graph is not None) == graph_first_step for entry in rollout._entries.values())
    assert rollout.summary()["graph_first_step"] is graph_first_step


@pytest.mark.skipif(not CUDA, reason="CUDA graphs require CUDA")
def test_hydra_graph_replays_identical_raw_fields():
    from deepstarnoodles_inference.hydra_graph import HydraGraph

    class Predictor:
        device = torch.device("cuda")

        def __init__(self):
            torch.manual_seed(0)
            self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.GELU(),
                                             torch.nn.Conv2d(8, 18, 1)).cuda().eval()

        def predict_tensor_batch_raw(self, images):
            with torch.autocast("cuda"):
                return self.model(images).float()

    predictor = Predictor()
    graph = HydraGraph(predictor)
    with torch.inference_mode():
        for seed in range(3):
            images = torch.rand((1, 3, 32, 32), generator=torch.Generator().manual_seed(seed)).cuda()
            assert torch.equal(graph.raw(images), predictor.predict_tensor_batch_raw(images))
    assert graph.summary()["captures"] == 1 and graph.summary()["replays"] == 3


def test_cli_validates_graph_and_detect_ahead_options():
    base = ["--models", "m", "--index", "i.csv", "--output", "o", "--mode", "archive"]
    parsed = benchmark.parse_args(base + ["--gru-execution", "graph", "--hydra-output-mode", "minimal",
                                          "--hydra-execution", "graph", "--detect-ahead", "2"])
    assert (parsed.gru_execution, parsed.hydra_execution, parsed.detect_ahead) == ("graph", "graph", 2)
    for extra in (["--gru-execution", "graph", "--device", "cpu"],
                  ["--hydra-execution", "graph"],
                  ["--hydra-execution", "graph", "--hydra-output-mode", "minimal", "--device", "cpu"],
                  ["--detect-ahead", "9"], ["--detect-ahead", "1", "--profile-stages"],
                  ["--detect-ahead", "1", "--archive-frame-window", "4"]):
        with pytest.raises(SystemExit):
            benchmark.parse_args(base + extra)
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--models", "m", "--index", "i.csv", "--output", "o", "--mode", "live-replay",
                              "--detect-ahead", "1"])
    with pytest.raises(ValueError, match="graph requires CUDA"):
        live.LiveConfig(device="cpu", gru_execution="graph")
    with pytest.raises(ValueError, match="hydra_execution graph"):
        live.LiveConfig(device="cuda:0", hydra_execution="graph")


def detector(frames, **_kwargs):
    results = []
    for frame in frames:
        source = int(frame[0, 0, 0]) - 1
        count = {0: 2, 1: 0, 2: 3, 3: 1}[source]
        coordinates = [[4, 4, 12, 12]] * count + [[0, 0, 20, 20]]
        results.append(type("Result", (), {"boxes": Boxes(coordinates, [2] * count + [0]), "masks": None})())
    return results


@pytest.mark.parametrize("depth", [1, 2])
def test_detect_ahead_preserves_order_and_records(tmp_path, monkeypatch, depth):
    paths, index = write_images(tmp_path)
    outputs = []
    for ahead in (0, depth):
        pipeline = make_pipeline(monkeypatch, detector_batch=1, crop_batch=2)
        pipeline.args.index = index
        pipeline.args.hydra_output_mode = "minimal"
        pipeline.args.detect_ahead = ahead
        pipeline.detector = type("Detector", (), {"predict": staticmethod(detector)})()
        sink = io.StringIO()
        report = benchmark.measure(pipeline, pipeline.args, len(paths), sink, profile=False)
        outputs.append(sink.getvalue())
        assert report["source_images"] == 4 and report["star_crops"] == report["arms"] == 6
        if ahead:
            assert report["detect_ahead"]["produced_groups"] == 4
            assert report["detect_ahead"]["peak_queued"] <= depth
    assert outputs[0] == outputs[1]
    assert [json.loads(line)["frame"] for line in outputs[1].splitlines()] == [0, 1, 2, 3]


def test_detect_ahead_surfaces_producer_failure(tmp_path, monkeypatch):
    paths, index = write_images(tmp_path)
    pipeline = make_pipeline(monkeypatch, detector_batch=1, crop_batch=1)
    pipeline.args.index = index
    pipeline.args.detect_ahead = 1

    def broken(frames, **_kwargs):
        raise RuntimeError("detector failed")

    pipeline.detector = type("Detector", (), {"predict": staticmethod(broken)})()
    with pytest.raises(RuntimeError, match="detector failed"):
        benchmark.measure(pipeline, pipeline.args, len(paths), io.StringIO(), profile=False)


def test_shard_child_argv_replaces_parent_only_flags(tmp_path):
    from deepstarnoodles_inference.archive_shards import child_argv

    argv = ["--models", "m.json", "--index", "i.csv", "--output", "out", "--mode", "archive",
            "--archive-processes", "3", "--images=10", "--archive-processes=3"]
    child = child_argv(argv, 1, 3, tmp_path / "sync", tmp_path / "out/shard_1")
    assert "--archive-processes" not in " ".join(child) and child.count("--output") == 1
    assert child[-8:] == ["--output", str(tmp_path / "out/shard_1"), "--shard-index", "1",
                          "--shard-count", "3", "--shard-sync", str(tmp_path / "sync")]
    assert "--images=10" in child and child[:4] == ["--models", "m.json", "--index", "i.csv"]


def test_shard_merge_restores_global_order_and_bytes(tmp_path):
    from deepstarnoodles_inference.archive_shards import merge_predictions

    records = [{"frame": i, "image_path": f"/x/{i}.png", "source_shape": [2, 3],
                "stars": [{"arms": [{"length_px": 0.1 * i + 1e-7, "polyline_crop": [[1.5, 2.25]]}]}]} for i in range(7)]
    expected = "".join(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n" for r in records)
    shards = []
    for k in range(3):
        path = tmp_path / f"shard{k}.jsonl"
        local = [dict(r, frame=j) for j, r in enumerate(records[k::3])]
        path.write_text("".join(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n" for r in local))
        shards.append(path)
    out = tmp_path / "merged"
    out.mkdir()
    assert merge_predictions(out, shards, 7) == 7
    assert (out / "predictions.jsonl").read_text() == expected
    out2 = tmp_path / "short"
    out2.mkdir()
    with pytest.raises(ValueError, match="ended before"):
        merge_predictions(out2, shards, 8)


def test_cli_validates_archive_processes():
    base = ["--models", "m", "--index", "i.csv", "--output", "o", "--mode", "archive"]
    assert benchmark.parse_args(base + ["--archive-processes", "3"]).archive_processes == 3
    assert benchmark.parse_args(base + ["--archive-processes", "16"]).archive_processes == 16
    for extra in (["--archive-processes", "17"], ["--archive-processes", "0"], ["--archive-processes", "2", "--device", "cpu"],
                  ["--archive-processes", "2", "--profile-stages"],
                  ["--shard-index", "0"], ["--shard-count", "2", "--shard-index", "2", "--shard-sync", "s"]):
        with pytest.raises(SystemExit):
            benchmark.parse_args(base + extra)
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--models", "m", "--index", "i.csv", "--output", "o", "--mode", "live-replay",
                              "--archive-processes", "2"])


def test_graph_gate_separates_capture_from_replays():
    import threading
    import time
    from deepstarnoodles_inference.capture_lock import _ReplayCaptureGate

    gate = _ReplayCaptureGate()
    events = []
    replay_entered = threading.Event()
    release_replay = threading.Event()

    def replay():
        with gate.replay():
            events.append("replay-start")
            replay_entered.set()
            release_replay.wait(5)
            events.append("replay-end")

    def capture():
        with gate.capture():
            events.append("capture")

    first = threading.Thread(target=replay)
    first.start()
    replay_entered.wait(5)
    capturing = threading.Thread(target=capture)
    capturing.start()
    time.sleep(0.05)
    late = threading.Thread(target=lambda: (gate.replay().__enter__(), events.append("late-replay")))
    late.start()
    time.sleep(0.05)
    assert events == ["replay-start"]  # capture waits; the late replay yields to it
    release_replay.set()
    for thread in (first, capturing, late):
        thread.join(5)
    assert events == ["replay-start", "replay-end", "capture", "late-replay"]


@pytest.mark.skipif(not CUDA, reason="crop streams require CUDA")
def test_crop_streams_yield_in_order_with_exclusive_workspaces():
    import random
    import threading
    import time
    from types import SimpleNamespace
    from deepstarnoodles_inference.crop_streams import CropStreams

    busy = set()
    lock = threading.Lock()

    class Cascade:
        device = torch.device("cuda:0")
        args = SimpleNamespace(hydra_execution="eager", gru_check_interval=1)

        def compute_crops(self, crop_group, images, timer, workspace):
            with lock:
                assert id(workspace) not in busy
                busy.add(id(workspace))
            assert torch.cuda.current_stream() == workspace.stream
            time.sleep(random.random() * 0.01)
            with lock:
                busy.discard(id(workspace))
            return images * 2, [crop_group], [{"body": crop_group}]

    streams = CropStreams(Cascade(), 3)
    jobs = [(lambda value: (lambda workspace: (value, torch.full((1,), float(value), device="cuda"))))(n)
            for n in range(12)]
    results = list(streams.map(jobs))
    streams.close()
    assert [group for group, _, _, _ in results] == list(range(12))
    assert [float(activated) for _, activated, _, _ in results] == [2.0 * n for n in range(12)]
    assert [bodies for _, _, _, bodies in results] == [[{"body": n}] for n in range(12)]


@pytest.mark.skipif(not CUDA, reason="concurrent GRU rollouts require CUDA")
@pytest.mark.parametrize("graph_first_step", [False, True])
@pytest.mark.parametrize("slots", [1, 3])
def test_concurrent_rollouts_reproduce_sequential_bytes(slots, graph_first_step):
    from deepstarnoodles_inference.noodle_tracer_graph import (
        ConcurrentGraphRollouts, trace_arms_fixed_slots_concurrent,
    )

    model = neural_tracer("cuda")
    config = neural_params()
    groups = []
    for seed in range(5):
        raw, images = neural_fields(seed, batch=1 + seed % 2)
        groups.append((raw.cuda(), images.cuda()))
    empty = torch.full_like(groups[0][0], -10.0)
    groups.insert(2, (empty, groups[0][1]))  # a group with no seeds
    with torch.inference_mode():
        expected = [trace_arms_noodle_batch_fixed_slots(raw, images, model, config, check_interval=1)
                    for raw, images in groups]
        scheduler = ConcurrentGraphRollouts(model, config, slots=slots, graph_first_step=graph_first_step)
        for _ in range(2):  # second pass replays cached graphs
            observed = trace_arms_fixed_slots_concurrent(groups, model, config, scheduler)
            assert [digest(item) for item in observed] == [digest(item) for item in expected]
    assert observed[2] == [[]]
    assert scheduler.summary()["jobs"] == 2 * 5


def test_cli_validates_gru_streams():
    base = ["--models", "m", "--index", "i.csv", "--output", "o", "--mode", "archive"]
    parsed = benchmark.parse_args(base + ["--gru-execution", "graph", "--gru-streams", "4"])
    assert parsed.gru_streams == 4
    for extra in (["--gru-streams", "4"], ["--gru-execution", "graph", "--gru-streams", "9"],
                  ["--gru-execution", "graph", "--gru-streams", "2", "--crop-streams", "2"]):
        with pytest.raises(SystemExit):
            benchmark.parse_args(base + extra)
    with pytest.raises(ValueError, match="gru_streams"):
        live.LiveConfig(device="cuda:0", gru_streams=2)


@pytest.mark.parametrize("gru_streams", [1, 3])
def test_detect_ahead_gathers_frames_without_reordering(tmp_path, monkeypatch, gru_streams):
    paths, index = write_images(tmp_path)
    pipeline = make_pipeline(monkeypatch, detector_batch=1, crop_batch=1)
    pipeline.args.index = index
    pipeline.args.hydra_output_mode = "minimal"
    pipeline.args.detect_ahead = 4
    pipeline.args.gru_streams = gru_streams
    pipeline.detector = type("Detector", (), {"predict": staticmethod(detector)})()
    calls = []

    def infer_many(jobs, timer):
        calls.append(len(jobs))
        return sum(pipeline.infer_crops(crop_group, images, records, start, timer)
                   for records, start, crop_group, images in jobs)

    pipeline.infer_many = infer_many
    sink = io.StringIO()
    report = benchmark.measure(pipeline, pipeline.args, len(paths), sink, profile=False)
    frames = [json.loads(line)["frame"] for line in sink.getvalue().splitlines()]
    assert frames == [0, 1, 2, 3] and report["star_crops"] == report["arms"] == 6
    assert (sum(calls) == 6) if gru_streams > 1 else not calls


def test_cli_validates_gru_compile():
    base = ["--models", "m", "--index", "i.csv", "--output", "o", "--mode", "archive"]
    parsed = benchmark.parse_args(base + ["--gru-execution", "graph", "--gru-compile"])
    assert parsed.gru_compile is True
    assert benchmark.parse_args(base).gru_compile is False
    for execution in ("dynamic", "fixed"):
        with pytest.raises(SystemExit):
            benchmark.parse_args(base + ["--gru-execution", execution, "--gru-compile"])
    with pytest.raises(ValueError, match="gru_compile"):
        live.LiveConfig(device="cuda:0", gru_execution="fixed", gru_compile=True)
    with pytest.raises(ValueError, match="gru_compile"):
        live.LiveConfig(device="cuda:0", gru_execution="graph", gru_compile=1)


def test_prebuilt_triton_modules_require_matching_source_and_binary(tmp_path, monkeypatch):
    from deepstarnoodles_inference import gru_compile

    source = tmp_path / "cuda_utils.c"
    source.write_text("int helper(void) { return 0; }\n")
    binary = tmp_path / "prebuilt" / "cuda_utils.cpython-311-x86_64-linux-gnu.so"
    binary.parent.mkdir()
    binary.write_bytes(b"\x7fELF prebuilt")
    manifest = binary.parent / "manifest.json"
    manifest.write_text(json.dumps({"modules": {"cuda_utils": {
        "file": binary.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}}}))
    build = gru_compile.prebuilt_build(manifest)
    out = tmp_path / "build"
    out.mkdir()
    built = build("cuda_utils", str(source), str(out), [], [], [])
    assert open(built, "rb").read() == binary.read_bytes()
    with pytest.raises(RuntimeError, match="no prebuilt"):
        build("__triton_launcher", str(source), str(out), [], [], [])
    source.write_text("int helper(void) { return 1; }\n")
    with pytest.raises(RuntimeError, match="no prebuilt"):
        build("cuda_utils", str(source), str(out), [], [], [])
    # A compiler with Python headers on the host always wins over prebuilt modules.
    headers = tmp_path / "include"
    headers.mkdir()
    (headers / "Python.h").write_text("")
    monkeypatch.setattr(gru_compile.sysconfig, "get_paths", lambda: {"include": str(headers)})
    monkeypatch.setenv(gru_compile.PREBUILT_ENV, str(manifest))
    monkeypatch.setattr(gru_compile.shutil, "which", lambda name: "/usr/bin/gcc" if name == "gcc" else None)
    assert gru_compile.install_prebuilt_build() is False
    # Without the headers the compiler cannot build the helper, so the prebuilt module is used.
    knobs = pytest.importorskip("triton.knobs")
    monkeypatch.setattr(knobs.build, "impl", knobs.build.impl)
    (headers / "Python.h").unlink()
    assert gru_compile.install_prebuilt_build() is True
    monkeypatch.delenv(gru_compile.PREBUILT_ENV)
    monkeypatch.setattr(gru_compile.shutil, "which", lambda name: None)
    assert gru_compile.install_prebuilt_build() is False


@pytest.mark.skipif(not CUDA, reason="compiled CUDA-graph rollouts require CUDA")
def test_compiled_graph_rollout_stays_within_strict_tolerances():
    pytest.importorskip("triton")
    from deepstarnoodles_inference.noodle_tracer_graph import FixedSlotGraphRollout

    model = neural_tracer("cuda")
    config = neural_params()
    eager = FixedSlotGraphRollout(model, config)
    compiled = FixedSlotGraphRollout(model, config, compile_step=True)
    with torch.inference_mode():
        for seed in range(3):
            raw, images = neural_fields(seed, batch=1)
            raw, images = raw.cuda(), images.cuda()
            reference = trace_arms_noodle_batch_fixed_slots(raw, images, model, config, check_interval=1,
                                                            graph_rollout=eager, device_peaks=True)
            observed = trace_arms_noodle_batch_fixed_slots(raw, images, model, config, check_interval=1,
                                                           graph_rollout=compiled, device_peaks=True)
            # Same arms, steps and stop reasons; geometry and confidences within strict gates.
            assert_outputs_close(reference, observed, atol=0.01)
    assert compiled.summary()["compiled_step"] is True and compiled.captures >= 1


@pytest.mark.skipif(not CUDA, reason="graph rollouts require CUDA")
def test_compiled_steps_outside_replay_hold_the_capture_lock(monkeypatch):
    # Compilation synchronizes the device, so it must never overlap another thread's capture.
    from deepstarnoodles_inference import gru_compile
    from deepstarnoodles_inference.capture_lock import CAPTURE_LOCK
    from deepstarnoodles_inference.noodle_tracer_graph import FixedSlotGraphRollout, rollout_step

    calls = []

    def guarded_step(*args, **kwargs):
        calls.append(CAPTURE_LOCK._is_owned())
        return rollout_step(*args, **kwargs)

    monkeypatch.setattr(gru_compile, "compiled_rollout_step", lambda: guarded_step)
    model = neural_tracer("cuda")
    config = neural_params()
    rollout = FixedSlotGraphRollout(model, config, compile_step=True)
    raw, images = neural_fields(0, batch=1)
    with torch.inference_mode():
        trace_arms_noodle_batch_fixed_slots(raw.cuda(), images.cuda(), model, config, check_interval=1,
                                            graph_rollout=rollout, device_peaks=True)
    assert calls and all(calls)
