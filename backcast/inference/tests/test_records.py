# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest

from deepstarnoodles_inference import archive_shards
from deepstarnoodles_inference.records import FRAME_TAIL, OUTPUT_SCHEMA_VERSION, finalize_frame


def test_finalize_frame_appends_tail_last_and_is_idempotent():
    record = {"frame": 7, "image_path": None, "source_shape": [4, 5], "stars": [], "video": "clip"}
    finalize_frame(record, 2.0)
    assert list(record)[-3:] == list(FRAME_TAIL)
    assert (record["time_s"], record["fps"], record["schema_version"]) == (3.5, 2.0, OUTPUT_SCHEMA_VERSION)
    record["capture_read_ms"] = 1.0
    finalize_frame(record, 2.0)
    assert list(record) == ["frame", "image_path", "source_shape", "stars", "video", "capture_read_ms",
                            "time_s", "fps", "schema_version"]
    once = json.dumps(record)
    assert json.dumps(finalize_frame(record, 2.0)) == once


def test_finalize_frame_without_fps_writes_null_never_nan():
    record = finalize_frame({"frame": 3, "stars": []}, None)
    assert record["time_s"] is None and record["fps"] is None
    json.dumps(record, allow_nan=False)
    assert finalize_frame({"frame": 1, "stars": []}, 29.97)["time_s"] == 1 / 29.97
    for bad in (0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            finalize_frame({"frame": 0}, bad)


def test_sharded_merge_recomputes_time_from_the_global_frame(tmp_path):
    shards = []
    for index in range(2):
        path = tmp_path / f"shard{index}.jsonl"
        lines = [finalize_frame({"frame": local, "image_path": f"{index}-{local}", "stars": []}, 4.0)
                 for local in range(2)]
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        shards.append(path)
    output = tmp_path / "merged"
    output.mkdir()
    assert archive_shards.merge_predictions(output, shards, 4) == 4
    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [r["frame"] for r in records] == [0, 1, 2, 3]
    assert [r["image_path"] for r in records] == ["0-0", "1-0", "0-1", "1-1"]
    assert [r["time_s"] for r in records] == [0.0, 0.25, 0.5, 0.75]
    assert all(list(r)[-3:] == list(FRAME_TAIL) for r in records)
