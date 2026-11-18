# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from pathlib import Path
import gc
import threading
from types import SimpleNamespace
import weakref

import cv2
import numpy as np
import pytest
import torch

from deepstarnoodles_inference import archive_decode as decode
from deepstarnoodles_inference import benchmark
from test_benchmark_cascade import Boxes, make_pipeline, write_images


def test_real_images_and_archive_records_are_exactly_unchanged(tmp_path, monkeypatch):
    paths, _ = write_images(tmp_path, count=5)
    def pipeline():
        value = make_pipeline(monkeypatch, detector_batch=2, crop_batch=2)
        def predict(frames, **kwargs):
            return [SimpleNamespace(boxes=Boxes([[4, 4, 12, 12]] * (int(frame[0, 0, 0]) % 3),
                                                [2] * (int(frame[0, 0, 0]) % 3)), masks=None)
                    for frame in frames]
        value.detector = SimpleNamespace(predict=predict)
        return value
    reference, candidate = pipeline(), pipeline()
    timer = benchmark.Timer(torch.device("cpu"), False)
    expected = []
    for group in benchmark.chunks(paths, 2):
        expected.extend(reference.run_group(group, len(expected), timer)[0])
    actual = []
    with decode.OrderedDecoder(paths, workers=2, max_pending=3) as reader:
        for group in benchmark.chunks(reader, 2):
            source_paths, frames = zip(*group)
            for path, frame in group:
                np.testing.assert_array_equal(frame, cv2.imread(str(path), cv2.IMREAD_COLOR))
            actual.extend(candidate.run_frames(frames, len(actual), timer, source_paths=source_paths)[0])
    assert actual == expected
    assert [record["frame"] for record in actual] == list(range(5))
    assert reader.report()["delivered"] == 5 and reader.report()["peak_pending"] <= 3


def test_completion_order_cannot_reorder_source_frames(monkeypatch):
    second_done, release_first = threading.Event(), threading.Event()
    monkeypatch.setattr(decode, "expected_bytes", lambda _path: 3)
    def read(path, _size):
        if path.name == "0":
            assert release_first.wait(5)
        else:
            second_done.set()
        return np.full((1, 1, 3), int(path.name), np.uint8), 1.0
    monkeypatch.setattr(decode, "read_frame", read)
    answers, errors = [], []
    with decode.OrderedDecoder(["0", "1"], workers=2, max_pending=2) as reader:
        def consume():
            try:
                answers.append(next(reader))
            except BaseException as exc:
                errors.append(exc)
        consumer = threading.Thread(target=consume)
        consumer.start()
        assert second_done.wait(5)
        assert answers == []
        release_first.set()
        consumer.join(5)
        assert not consumer.is_alive() and errors == []
        assert answers[0][0].name == "0"
        assert next(reader)[0].name == "1"


def test_byte_reservations_limit_pending_work_below_item_limit(monkeypatch):
    monkeypatch.setattr(decode, "expected_bytes", lambda _path: 60)
    monkeypatch.setattr(decode, "read_frame", lambda path, size: (np.zeros((10, 2, 3), np.uint8), 1.0))
    with decode.OrderedDecoder(map(str, range(20)), workers=2, max_pending=8,
                               byte_budget=120, max_frame_bytes=60) as reader:
        names = []
        for path, frame in reader:
            names.append(path.name)
            assert reader.reserved_bytes <= 120
        assert names == list(map(str, range(20)))
    assert reader.report()["peak_reserved_bytes"] == 120
    assert reader.report()["peak_pending"] == 2


def test_oversize_is_rejected_before_any_decode(monkeypatch):
    monkeypatch.setattr(decode, "expected_bytes", lambda _path: 301)
    monkeypatch.setattr(decode, "read_frame", lambda *_args: pytest.fail("oversize must not be decoded"))
    with decode.OrderedDecoder(["huge.png"], byte_budget=300, max_frame_bytes=300) as reader:
        with pytest.raises(MemoryError, match="before decoding"):
            next(reader)
    assert reader.submitted == 0


@pytest.mark.parametrize("where", ["header", "decode"])
def test_error_propagates_at_its_source_position_and_closes_workers(monkeypatch, where):
    def header(path):
        if path.name == "1" and where == "header":
            raise ValueError("bad header")
        return 3
    def read(path, _size):
        if path.name == "1" and where == "decode":
            raise ValueError("bad decode")
        return np.zeros((1, 1, 3), np.uint8), 1.0
    monkeypatch.setattr(decode, "expected_bytes", header)
    monkeypatch.setattr(decode, "read_frame", read)
    with decode.OrderedDecoder(["0", "1", "2"], workers=2, max_pending=2) as reader:
        assert next(reader)[0].name == "0"
        with pytest.raises(ValueError, match="bad"):
            next(reader)
    assert reader._closed and reader.reserved_bytes == 0 and not reader._pending


def test_early_consumer_stop_releases_reader_arrays(monkeypatch):
    references = []
    monkeypatch.setattr(decode, "expected_bytes", lambda _path: 3)
    def read(path, size):
        array = np.zeros((1, 1, 3), np.uint8)
        references.append(weakref.ref(array))
        return array, 1.0
    monkeypatch.setattr(decode, "read_frame", read)
    with decode.OrderedDecoder(map(str, range(50)), workers=2, max_pending=4) as reader:
        next(reader)
    gc.collect()
    assert reader.submitted <= 5
    assert all(reference() is None for reference in references)


def test_changed_decoded_size_is_rejected(monkeypatch):
    monkeypatch.setattr(cv2, "imread", lambda *_args: np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(ValueError, match="reserved header size"):
        decode.read_frame(Path("changed.png"), 3)


@pytest.mark.parametrize("options", [{"workers": 0}, {"workers": 5}, {"max_pending": 1},
                                    {"workers": True}, {"byte_budget": 0},
                                    {"byte_budget": 100, "max_frame_bytes": 101}])
def test_bounds_are_validated(options):
    with pytest.raises(ValueError):
        decode.OrderedDecoder([], **options)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_measure_prefetch_preserves_jsonl_batches_and_all_rows(tmp_path, monkeypatch, workers):
    import io
    import json
    paths, index = write_images(tmp_path, count=5)
    outputs, reports, batches = [], [], []
    for enabled in (0, workers):
        pipeline = make_pipeline(monkeypatch, detector_batch=2, crop_batch=2)
        pipeline.args.index = index
        pipeline.args.decode_workers = enabled
        pipeline.args.mode = "archive"
        observed = []
        def predict(frames, **kwargs):
            observed.append([int(frame[0, 0, 0]) for frame in frames])
            return [SimpleNamespace(boxes=Boxes([[4, 4, 12, 12]] * (int(frame[0, 0, 0]) % 3),
                                                [2] * (int(frame[0, 0, 0]) % 3)), masks=None)
                    for frame in frames]
        pipeline.detector = SimpleNamespace(predict=predict)
        sink = io.StringIO()
        reports.append(benchmark.measure(pipeline, pipeline.args, len(paths), sink, profile=True))
        outputs.append(sink.getvalue())
        batches.append(observed)
    assert outputs[0] == outputs[1]
    assert batches == [[[1, 2], [3, 4], [5]]] * 2
    assert [len(json.loads(line)["stars"]) for line in outputs[1].splitlines()] == [1, 2, 0, 1, 2]
    assert reports[0]["archive_decode"] is None
    assert reports[1]["archive_decode"]["submitted"] == reports[1]["archive_decode"]["delivered"] == 5
    assert "read_decode" not in reports[1]["stages"]
    assert reports[1]["stages"]["archive_decode_wait"]["samples"] == 3
    assert reports[1]["source_group_latency"]["samples"] == 3


def test_zero_warmup_and_separate_measures_do_not_prefetch_unselected_rows(tmp_path, monkeypatch):
    import io
    paths, index = write_images(tmp_path, count=4)
    args = SimpleNamespace(index=index, detector_batch_size=2, decode_workers=2, ram_limit_gib=24.0)
    pipeline = SimpleNamespace(device=torch.device("cpu"), run_frames=lambda frames, start, timer, source_paths:
                               ([{"frame": start+i, "stars": []} for i in range(len(frames))], 0, 0))
    requested = []
    original = decode.read_frame
    def read(path, size):
        requested.append(path)
        return original(path, size)
    monkeypatch.setattr(decode, "read_frame", read)
    zero = benchmark.measure(pipeline, args, 0, io.StringIO(), profile=False)
    assert requested == [] and zero["archive_decode"]["submitted"] == 0
    warmup = benchmark.measure(pipeline, args, 1, io.StringIO(), profile=False)
    assert requested == paths[:1] and warmup["archive_decode"]["delivered"] == 1
    measured = benchmark.measure(pipeline, args, 3, io.StringIO(), profile=False)
    assert sorted(requested[1:]) == sorted(paths[:3])
    assert measured["archive_decode"]["submitted"] == measured["source_images"] == 3


def test_inference_error_closes_prefetch_and_releases_arrays(tmp_path, monkeypatch):
    import io
    _, index = write_images(tmp_path, count=20)
    references, readers = [], []
    original = decode.read_frame
    def read(path, size):
        frame, elapsed = original(path, size)
        references.append(weakref.ref(frame))
        return frame, elapsed
    class Observed(decode.ArchiveSourceGroups):
        def __init__(self, *args):
            super().__init__(*args)
            readers.append(self.reader)
    def fail(*args, **kwargs):
        raise RuntimeError("inference failure")
    monkeypatch.setattr(decode, "read_frame", read)
    monkeypatch.setattr(benchmark, "ArchiveSourceGroups", Observed)
    args = SimpleNamespace(index=index, detector_batch_size=1, decode_workers=2, ram_limit_gib=24.0)
    with pytest.raises(RuntimeError, match="inference failure"):
        benchmark.measure(SimpleNamespace(device=torch.device("cpu"), run_frames=fail), args, 20, io.StringIO(), False)
    gc.collect()
    assert readers[0]._closed and readers[0].reserved_bytes == 0
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("extra", [
    ["--decode-workers", "3"], ["--decode-workers", "-1"],
    ["--decode-workers", "4", "--decode-prefetch-frames", "3"],
    ["--decode-prefetch-frames", "0"], ["--decode-prefetch-frames", "65"],
    ["--decode-prefetch-mib", "0"], ["--decode-prefetch-mib", "1025"],
    ["--decode-max-frame-mib", "0"], ["--decode-max-frame-mib", "257"],
    ["--mode", "live-replay", "--decode-workers", "1"],
])
def test_cli_rejects_invalid_or_temporal_prefetch(extra):
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--models", "models", "--index", "images.csv", "--output", "out",
                              "--mode", "archive", "--device", "cpu", *extra])


def test_cli_default_preserves_sync_and_staged_accepts_prefetch():
    from deepstarnoodles_inference import staged
    argv = ["--models", "models", "--index", "images.csv", "--output", "out",
            "--mode", "archive", "--device", "cpu"]
    assert benchmark.parse_args(argv).decode_workers == 0
    assert staged.parse_args(argv + ["--warmup", "0", "--decode-workers", "2"]).decode_workers == 2


# Reuse deterministic model fixtures to exercise actual staged crop/spool code.
from test_staged import arguments, fake_models, frames, model_paths, spool_state


@pytest.mark.parametrize("empty", [False, True])
def test_staged_prefetch_keeps_source_order_spool_results_and_model_release(
    tmp_path, frames, fake_models, model_paths, empty
):
    from deepstarnoodles_inference import staged
    selected = frames[-1:] if empty else frames
    outputs, states = [], []
    for workers in (0, 2):
        spool = tmp_path / f"spool{workers}"
        spool.mkdir()
        state = spool_state()
        outputs.append(staged.run_shard(arguments(decode_workers=workers), model_paths, selected, 10,
                                        spool, benchmark.Timer(torch.device("cpu"), False), state))
        states.append(state)
        assert list(spool.iterdir()) == []
    assert outputs[0] == outputs[1]
    assert [r["frame"] for r in outputs[1][0]] == list(range(10, 10 + len(selected)))
    assert states[1]["archive_decode"]["delivered"] == len(selected)
    assert states[1]["archive_decode"]["submitted"] == len(selected)
    gc.collect()
    assert all(ref() is None for ref in fake_models.detector_refs + fake_models.hydra_refs + fake_models.tracer_refs)


def test_shard_summary_is_bounded_and_does_not_add_overlapping_peaks():
    sample = dict(workers=2, max_pending=4, byte_budget=100, max_frame_bytes=50,
                  peak_reserved_bytes=80, peak_pending=4, submitted=5, delivered=5,
                  sum_worker_decode_ms=20.0, consumer_wait_ms=2.0, scope="reader")
    result = None
    for _ in range(100):
        result = decode.accumulate_decode_report(result, sample)
    assert result["shards"] == 100
    assert result["submitted"] == result["delivered"] == 500
    assert result["peak_reserved_bytes"] == 80 and result["peak_pending"] == 4
    assert result["sum_worker_decode_ms"] == 2000 and result["consumer_wait_ms"] == 200
    assert len(result) == len(sample) + 1
