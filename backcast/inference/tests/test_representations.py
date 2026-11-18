# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Inference records -> noodle_polylines containers and the representations command (torch-free)."""
from __future__ import annotations

import bz2
import csv
import gzip
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from deepstarnoodles_inference import cli, representations as R
from deepstarnoodles_inference._vendor.noodle_polylines import PRESETS, PolylineContainer, compute_all
from deepstarnoodles_inference._vendor.noodle_polylines import adapters
from deepstarnoodles_inference._vendor.noodle_polylines.container import (ARM_FIELDS, FRAME_FIELDS, STOP_REASON_CODES,
                                                                       ContainerError)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "polylines"
EXAMPLE = FIXTURES / "flip_example_01.npz"
STOP_NAMES = {code: name for name, code in STOP_REASON_CODES.items()}

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def example():
    return PolylineContainer.load(EXAMPLE, sidecar=True)


def contexts(c):
    bouts = c.meta["bouts"]
    return {"fig4d": {"windows": [(b["upside_down_start_s"], b["right_side_up_start_s"]) for b in bouts]},
            "fig4e": {"bout": bouts[0]}, "fig5": {"onset_s": bouts[0]["flip_start_s"]},
            "fig5_example": {"onset_s": bouts[0]["flip_start_s"]}}


def star_entry(c, f, *, star_id=3, schema3=True, archive=False, passthrough=True):
    left, top, side = (float(v) for v in c.crop[f])
    arms = []
    for k, a in enumerate(c.frame_arms(f)):
        arm = {"arm_index" if archive else "track_id": k if archive else int(c.track_id[a]),
               "polyline_crop": c.arm(a).tolist(),
               "polyline_source": (c.arm(a).astype(np.float64) * side / 512 + [left, top]).tolist(),
               "identity": c.identity[a].tolist()}
        if passthrough:
            arm.update(stop_reason=STOP_NAMES[int(c.stop_reason[a])], steps=int(c.steps[a]),
                       state_confidence=float(c.state_confidence[a]), tip_score=float(c.tip_score[a]),
                       base_score=float(c.base_score[a]), length_px=12.5)
        arms.append(arm)
    star = {"detection_index": 0, "star_track_id": None if archive else star_id,
            "crop": {"left": left, "top": top, "side": side, "confidence": 0.9,
                     "bbox_xyxy": [left, top, left + side / 2, top + side / 2]},
            "mask_polygon_source": [[left, top], [left + 10.0, top], [left + 10.0, top + 10.0]], "arms": arms}
    if schema3:
        centroid = c.support_centroid_model[f]
        star.update(body_origin_model=[float(v) for v in c.body_origin_model[f]],
                    body_origin_code=int(c.body_origin_code[f]),
                    support_area_model_px=int(c.support_area_model_px[f]),
                    support_centroid_model=None if np.isnan(centroid).any() else [float(v) for v in centroid])
    return star


def records_from(c, *, schema3=True, archive=False, passthrough=True, drop=(), fps=2.0, extra_star=None):
    out = []
    for f in range(c.n_frames):
        frame = int(c.frame_index[f])
        stars = [] if f in drop else [star_entry(c, f, schema3=schema3, archive=archive, passthrough=passthrough)]
        if extra_star is not None and f in extra_star:
            second = star_entry(c, f, star_id=9, schema3=schema3, archive=archive, passthrough=passthrough)
            second["detection_index"] = 1
            stars.append(second)
        record = {"frame": frame, "image_path": f"clip#{frame:06d}", "source_shape": [2128, 2600], "stars": stars}
        if schema3:
            record.update(time_s=None if fps is None else frame / fps, fps=fps, schema_version=3)
        out.append(record)
    return out


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n" for r in records))
    return path


def same_array(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def same_outputs(left, right):
    assert set(left) == set(right)
    for key in left:
        same_array(left[key], right[key])


def test_vendored_helpers_this_adapter_relies_on_exist():
    for name in ("_drop_unused_arrays", "_stack_polylines", "_stack_identity", "_stop_codes", "_zstd_decompress"):
        assert callable(getattr(adapters, name))
    from deepstarnoodles_inference._vendor.noodle_polylines import container
    assert callable(container._offsets)
    upstream = json.loads((Path(__file__).resolve().parents[1] / "UPSTREAM.json").read_text())
    assert R.VENDORED_COMMIT == upstream["noodle_polylines"]["source_commit"]


def test_schema3_round_trip_reproduces_the_container_and_every_preset(tmp_path, example):
    path = write_jsonl(tmp_path / "clip.jsonl", records_from(example))
    loaded = R.load_inference(path)
    assert list(loaded) == [3]
    c = loaded[3]
    for name in (*FRAME_FIELDS, *ARM_FIELDS, "points", "arm_point_offsets", "frame_arm_offsets"):
        if name == "source_frame_index":
            assert c.source_frame_index is None and example.source_frame_index is None
            continue
        same_array(getattr(c, name), getattr(example, name))
    assert c.meta["fps"] == 2.0 and c.meta["missing_fields"] == [] and c.meta["body_origin"] == "runtime"
    assert c.meta["record_schema"] == [3] and c.meta["gap_rows"] == 0
    for preset in PRESETS:
        kw = contexts(example).get(preset, {})
        same_outputs(compute_all(c, preset, **kw), compute_all(example, preset, **kw))


def test_gaps_become_no_star_rows(tmp_path, example):
    part = example.select_frames(slice(0, 12))
    path = write_jsonl(tmp_path / "gaps.jsonl", records_from(part, drop={0, 4, 5, 11}))
    c = R.load_inference(path)[3]
    assert c.n_frames == 10 and c.meta["gap_rows"] == 2       # frames 1..10, 4 and 5 missing
    assert c.has_star.tolist() == [i not in (4, 5) for i in range(1, 11)]
    gap = ~c.has_star
    assert np.isnan(c.crop[gap]).all() and np.isnan(c.body_origin_model[gap]).all()
    assert (c.body_origin_code[gap] == -1).all() and (c.support_area_model_px[gap] == 0).all()
    assert (c.arms_per_frame()[gap] == 0).all()
    same_array(c.time_s, part.time_s[1:11])
    plain = R.load_inference(path, fill_gaps=False)[3]
    assert plain.n_frames == 8 and plain.has_star.all() and "gap_rows" not in plain.meta


def test_matches_upstream_adapter_on_legacy_files(tmp_path, example):
    part = example.select_frames(slice(0, 30))
    legacy = records_from(part, schema3=False, passthrough=False, drop={7})
    path = write_jsonl(tmp_path / "legacy.jsonl", legacy)
    ours, theirs = R.load_inference(path, fill_gaps=False), adapters.from_inference_json(path)
    assert list(ours) == list(theirs) == [3]
    for name in (*FRAME_FIELDS, *ARM_FIELDS, "points", "arm_point_offsets", "frame_arm_offsets"):
        left, right = getattr(ours[3], name), getattr(theirs[3], name)
        assert (left is None) == (right is None), name
        if left is not None:
            same_array(left, right)
    assert ours[3].meta["missing_fields"] == ["time_s", *R.BODY_FIELDS]
    # Single-star records keep frames without a detection, as upstream does.
    single = [{"frame": i, "detected": i != 2, "crop": None if i == 2 else {"left": 0.0, "top": 1.0, "side": 512.0},
               "arms": [] if i == 2 else [{"polyline_crop": [[1.0, 2.0], [3.0, 4.0]], "identity": [0.0] * 8}]}
              for i in range(4)]
    path = tmp_path / "single.json"
    path.write_text(json.dumps(single))
    ours, theirs = R.load_inference(path)[None], adapters.from_inference_json(path)[None]
    for name in ("points", "frame_arm_offsets", "frame_index", "crop", "has_star", "identity", "track_id"):
        same_array(getattr(ours, name), getattr(theirs, name))


def test_untracked_archive_records(tmp_path, example):
    part = example.select_frames(slice(0, 20))
    path = write_jsonl(tmp_path / "archive.jsonl", records_from(part, archive=True, drop={3}))
    with pytest.raises(ContainerError, match="frame 640.*untracked='single'.*untracked='detections'"):
        R.load_inference(path)
    single = R.load_inference(path, untracked="single")
    assert list(single) == ["single"]
    c = single["single"]
    assert c.n_frames == 20 and c.has_star.tolist() == [i != 3 for i in range(20)]
    assert (c.track_id == -1).all() and c.identity is not None and c.meta["untracked"] == "single"
    same_array(c.body_origin_model[c.has_star], part.body_origin_model[c.has_star])
    outputs = compute_all(c, "fig6_flipping")
    assert outputs["track_id_source"] == "relink" and (outputs["track_id"] >= 0).any()

    crowded = write_jsonl(tmp_path / "crowded.jsonl", records_from(part, archive=True, extra_star={2, 5}))
    with pytest.raises(ContainerError, match="2 untracked stars"):
        R.load_inference(crowded, untracked="single")
    d = R.load_inference(crowded, untracked="detections")["untracked"]
    assert d.n_frames == 22 and d.frame_index.tolist() == list(range(22))
    assert d.source_frame_index.tolist() == sorted([*part.frame_index.tolist(), *part.frame_index[[2, 5]].tolist()])
    assert d.meta["detection_index"][:4] == [0, 0, 0, 1] and d.identity is None and d.track_id is None
    same_array(d.time_s, d.source_frame_index / 2.0)
    assert "track_id" not in compute_all(d, "fig6_flipping")

    mixed = records_from(part, archive=True)
    mixed[4]["stars"][0]["star_track_id"] = 5
    with pytest.raises(ContainerError, match="tracked and untracked"):
        R.load_inference(write_jsonl(tmp_path / "mixed.jsonl", mixed), untracked="detections")


def test_fps_sources_and_conflicts(tmp_path, example):
    part = example.select_frames(slice(0, 5))
    path = write_jsonl(tmp_path / "a.jsonl", records_from(part))
    assert R.load_inference(path, fps=2.0)[3].fps == 2.0
    with pytest.raises(ContainerError, match="disagrees"):
        R.load_inference(path, fps=4.0)
    records = records_from(part)
    records[2]["fps"] = 3.0
    with pytest.raises(ContainerError, match="disagree on fps"):
        R.load_inference(write_jsonl(tmp_path / "b.jsonl", records))
    unknown = write_jsonl(tmp_path / "c.jsonl", records_from(part, fps=None))
    c = R.load_inference(unknown)[3]
    assert c.time_s is None and c.fps is None and c.meta["missing_fields"] == ["time_s"]
    timed = R.load_inference(unknown, fps=2.0)[3]
    same_array(timed.times(), part.time_s)
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"schema_version": 2, "fps": 2.0, "frames": records_from(part, fps=None)}))
    assert R.load_inference(payload)[3].fps == 2.0
    with pytest.raises(ContainerError, match="disagrees"):
        R.load_inference(payload, fps=5.0)


def test_legacy_records_give_body_free_outputs_only(tmp_path, example):
    part = example.select_frames(slice(0, 20))
    path = write_jsonl(tmp_path / "legacy.jsonl", records_from(part, schema3=False))
    c = R.load_inference(path, fps=2.0)[3]
    assert c.body_origin_model is None and c.support_area_model_px is None and c.fps == 2.0
    assert c.tip_score is not None and c.stop_reason.dtype == np.int8
    outputs = compute_all(c, "fig5", onset_s=330.0)
    assert set(outputs) == {"preset", "arc_length", "curvature", "tip_source", "n_arms"}


def test_partial_body_fields_are_dropped_with_a_warning(tmp_path, example):
    records = records_from(example.select_frames(slice(0, 4)))
    del records[1]["stars"][0]["support_area_model_px"]
    with pytest.warns(UserWarning, match="support_area_model_px present on 3 of 4"):
        c = R.load_inference(write_jsonl(tmp_path / "p.jsonl", records))[3]
    assert c.support_area_model_px is None and c.meta["missing_fields"] == ["support_area_model_px"]


def test_inference_only_stop_reasons_map_to_minus_one(tmp_path, example):
    records = records_from(example.select_frames(slice(0, 2)))
    records[0]["stars"][0]["arms"][0]["stop_reason"] = "bounds"
    c = R.load_inference(write_jsonl(tmp_path / "s.jsonl", records))[3]
    assert c.stop_reason[0] == -1 and (c.stop_reason[1:] >= 0).all()


def test_compressed_files_and_streams(tmp_path, example):
    records = records_from(example.select_frames(slice(0, 6)))
    text = "".join(json.dumps(r) + "\n" for r in records)
    reference = R.load_inference(write_jsonl(tmp_path / "plain.jsonl", records))[3]
    (tmp_path / "c.jsonl.gz").write_bytes(gzip.compress(text.encode()))
    (tmp_path / "c.jsonl.bz2").write_bytes(bz2.compress(text.encode()))
    (tmp_path / "pretty.json").write_text(json.dumps({"frames": records}, indent=2))
    (tmp_path / "pretty.jsonl").write_text(json.dumps({"frames": records}, indent=2))
    for source in ("c.jsonl.gz", "c.jsonl.bz2", "pretty.json", "pretty.jsonl"):
        same_array(R.load_inference(tmp_path / source)[3].points, reference.points)
    same_array(R.load_inference(io.StringIO(text))[3].body_origin_model, reference.body_origin_model)
    for skip in (True, False):
        same_array(R.load_inference(tmp_path / "plain.jsonl", skip_unused=skip)[3].identity, reference.identity)


def test_zstd_files(tmp_path, example, monkeypatch):
    zstd = shutil.which("zstd")
    records = records_from(example.select_frames(slice(0, 6)))
    plain = write_jsonl(tmp_path / "z.jsonl", records)
    reference = R.load_inference(plain)[3]
    try:
        import zstandard  # noqa: F401
        native = True
    except ImportError:
        native = sys.version_info >= (3, 14)
    if zstd is None and not native:
        pytest.skip("neither zstd nor a zstd module is available")
    if zstd is not None:
        subprocess.run([zstd, "-q", "-f", str(plain), "-o", str(tmp_path / "z.jsonl.zst")], check=True)
    else:
        import zstandard
        (tmp_path / "z.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(plain.read_bytes()))
    if not native:
        with pytest.raises(ImportError, match="zstandard"):
            R.load_inference(tmp_path / "z.jsonl.zst")
        pytest.skip("no zstd module: only the error path is checked")
    same_array(R.load_inference(tmp_path / "z.jsonl.zst")[3].points, reference.points)


def run_cli(*argv):
    return cli.main(["representations", *map(str, argv)])


def test_cli_parallel_equals_serial_and_writes_csv(tmp_path, example):
    inputs = tmp_path / "inputs"
    (inputs / "run_a").mkdir(parents=True)
    write_jsonl(inputs / "run_a" / "predictions.jsonl", records_from(example.select_frames(slice(0, 60)), drop={10}))
    write_jsonl(inputs / "b.jsonl", records_from(example.select_frames(slice(60, 120))))
    (inputs / "run_a" / "report.json").write_text("{}")
    (inputs / "partial.jsonl.part").write_text("{")
    windows = tmp_path / "windows.json"
    windows.write_text(json.dumps([[331.0, 334.0]]))
    for workers in (1, 2):
        # spawn: the test process may hold torch threads, which fork would copy mid-flight
        assert run_cli(inputs, "--output", tmp_path / f"out{workers}", "--preset", "all", "--csv",
                       "--workers", workers, "--mp-context", "spawn", "--windows-json", windows) == 0
    serial, parallel = tmp_path / "out1", tmp_path / "out2"
    files = sorted(p.relative_to(serial) for p in serial.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(parallel) for p in parallel.rglob("*") if p.is_file())
    for rel in files:
        if rel.name.startswith("manifest"):
            continue
        if rel.suffix == ".npz":
            with np.load(serial / rel) as a, np.load(parallel / rel) as b:
                assert a.files == b.files and all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")
                                                  for k in a.files)
        else:
            assert (serial / rel).read_bytes() == (parallel / rel).read_bytes(), rel
    manifest = json.loads((serial / "manifest.json").read_text())
    assert [item["name"] for item in manifest["inputs"]] == ["b", "run_a"]
    assert manifest["status"] == "complete" and manifest["versions"]["noodle_polylines_commit"] == R.VENDORED_COMMIT
    rows = {(row["sequence"], row["preset"]): row for row in manifest["sequences"]}
    assert len(rows) == 2 * len(PRESETS) and all(row["status"] == "ok" for row in rows.values())
    fig4d = rows[("run_a__star3", "fig4d")]
    assert fig4d["r0_context"] == {"windows": [[331.0, 334.0]]} and "arm_deformation_window" in fig4d["keys_written"]
    c = PolylineContainer.load(serial / "containers" / "run_a__star3.npz")
    assert c.n_frames == 60 and c.meta["gap_rows"] == 1
    with (serial / "fig4d" / "run_a__star3__frames.csv").open() as fh:
        frames = list(csv.DictReader(fh))
    with (serial / "fig4d" / "run_a__star3__arms.csv").open() as fh:
        arms = list(csv.DictReader(fh))
    assert len(frames) == c.n_frames and len(arms) == c.n_arms
    assert frames[10]["has_star"] == "0" and frames[10]["crop_0"] == "" and frames[10]["body_origin_code"] == "-1"
    assert float(frames[0]["body_origin_model_0"]) == c.body_origin_model[0, 0]
    assert {"arm_deformation_window_0", "fold_asymmetry", "time_s"} <= set(frames[0])
    assert [int(row["arm"]) for row in arms[:3]] == [0, 1, 2] and "track_id" in arms[0]
    assert float(arms[0]["arc_length"]) == compute_all(c, "fig4d", windows=[(331.0, 334.0)])["arc_length"][0]
    with (serial / "manifest.csv").open() as fh:
        assert len(list(csv.DictReader(fh))) == len(rows)


def test_cli_require_complete_and_legacy_degradation(tmp_path, example):
    legacy = write_jsonl(tmp_path / "legacy.jsonl", records_from(example.select_frames(slice(0, 10)), schema3=False))
    assert run_cli(legacy, "--output", tmp_path / "strict", "--require-complete") == 1
    manifest = json.loads((tmp_path / "strict" / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "missing fields" in manifest["sequences"][0]["error"]
    assert run_cli(legacy, "--output", tmp_path / "loose", "--preset", "fig4b", "--fps", "2") == 0
    row = json.loads((tmp_path / "loose" / "manifest.json").read_text())["sequences"][0]
    assert row["keys_written"] == ["arc_length", "curvature", "n_arms", "preset", "tip_source"]
    assert row["missing_fields"] == list(R.BODY_FIELDS)
    with pytest.raises(SystemExit):
        run_cli(legacy, "--output", tmp_path / "loose")      # existing output directory


def test_cli_untracked_detections_refuse_windowed_r0_and_relink(tmp_path, example):
    path = write_jsonl(tmp_path / "arch.jsonl", records_from(example.select_frames(slice(0, 40)), archive=True))
    assert run_cli(path, "--output", tmp_path / "o", "--untracked", "detections", "--preset", "fig4b", "fig5",
                   "--onset-s", "330") == 1
    rows = {row["preset"]: row for row in json.loads((tmp_path / "o" / "manifest.json").read_text())["sequences"]}
    assert rows["fig4b"]["status"] == "ok" and rows["fig5"]["status"] == "error"
    assert run_cli(path, "--output", tmp_path / "p", "--untracked", "detections", "--preset", "fig5") == 0
    assert run_cli(path, "--output", tmp_path / "q", "--untracked", "single", "--preset", "fig6_flipping",
                   "--csv") == 0
    assert (tmp_path / "q" / "fig6_flipping" / "arch__starsingle__arms.csv").is_file()


def test_cli_reads_stdin(tmp_path, example, monkeypatch):
    text = "".join(json.dumps(r) + "\n" for r in records_from(example.select_frames(slice(0, 5))))
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    assert run_cli("-", "--output", tmp_path / "s", "--name", "piped", "--preset", "fig6_flipping") == 0
    assert (tmp_path / "s" / "fig6_flipping" / "piped__star3.npz").is_file()


def test_cli_name_applies_to_one_file_and_is_refused_otherwise(tmp_path, example):
    records = records_from(example.select_frames(slice(0, 5)))
    for sub in ("a", "b"):
        (tmp_path / "in" / sub).mkdir(parents=True)
    write_jsonl(tmp_path / "in" / "a" / "predictions.jsonl", records)
    write_jsonl(tmp_path / "in" / "b" / "predictions.jsonl", records)
    one = tmp_path / "in" / "a" / "predictions.jsonl"
    assert run_cli(one, "--output", tmp_path / "s", "--name", "clip7", "--preset", "fig6_flipping") == 0
    assert (tmp_path / "s" / "fig6_flipping" / "clip7__star3.npz").is_file()
    assert not (tmp_path / "s" / "fig6_flipping" / "a__star3.npz").exists()
    with pytest.raises(SystemExit):          # several inputs: --name would be ambiguous, not silently dropped
        run_cli(tmp_path / "in", "--output", tmp_path / "t", "--name", "clip7")
    assert not (tmp_path / "t").exists()
    for bad in ("../escape", "x/y", "..", ""):
        with pytest.raises(SystemExit):
            run_cli(one, "--output", tmp_path / "u", "--name", bad)


def test_input_names_and_discovery(tmp_path):
    for rel in ("a/predictions.jsonl", "b/predictions.jsonl.zst", "c.jsonl.gz", "d/worker.0.json",
                "d/x.jsonl.part", "d/report.json", "_manifest.jsonl", "d/_queue.jsonl", "camA/C0001.mp4.jsonl.zst",
                "camB/C0001.mp4.jsonl"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("")
    found = R.collect_named_inputs([str(tmp_path)])
    assert [(Path(p).relative_to(tmp_path).as_posix(), name) for p, name in found] == [
        ("a/predictions.jsonl", "a"), ("b/predictions.jsonl.zst", "b"), ("c.jsonl.gz", "c"),
        ("camA/C0001.mp4.jsonl.zst", "camA/C0001.mp4"), ("camB/C0001.mp4.jsonl", "camB/C0001.mp4")]
    assert R.collect_inputs([str(tmp_path)]) == [path for path, _ in found]
    assert R.collect_named_inputs([str(tmp_path / "camA")])[0][1] == "C0001.mp4"
    assert R.input_names([p for p, _ in found]) == ["a", "b", "c", "C0001.mp4", "C0001.mp4__2"]
    with pytest.raises(ValueError, match="stdin"):
        R.collect_inputs(["-", str(tmp_path / "c.jsonl.gz")])
    # the same name from two places is refused, not numbered in sorted order
    with pytest.raises(ValueError, match="would both be named 'C0001.mp4'"):
        R.collect_named_inputs([str(tmp_path / "camA"), str(tmp_path / "camB")])
    (tmp_path / "c.jsonl").write_text("")
    with pytest.raises(ValueError, match="same video in two formats"):
        R.collect_named_inputs([str(tmp_path)])


def test_axis_table_matches_compute_all(example):
    F, A, P = example.n_frames, example.n_arms, example.n_points
    found = {}
    for preset in PRESETS:
        for key, value in compute_all(example, preset, shapes=True, **contexts(example).get(preset, {})).items():
            shape = np.shape(value)
            axis = ("scalar" if shape == () else "frame" if shape[0] == F else "arm" if shape[0] == A
                    else "point" if shape[0] == P else "window" if len(shape) == 2 and shape[1] == F else None)
            found.setdefault(key, set()).add(axis)
    assert all(len(axes) == 1 for axes in found.values()), found
    assert {key: axes.pop() for key, axes in found.items()} == R.AXIS


def test_representations_import_without_torch():
    import deepstarnoodles_inference
    parent = Path(deepstarnoodles_inference.__file__).resolve().parents[1]
    code = ("import sys; import deepstarnoodles_inference.representations, deepstarnoodles_inference.cli; "
            "bad = [m for m in ('torch', 'cv2', 'pandas', 'noodle_polylines', 'deepstarnoodles') if m in sys.modules]; "
            "print(','.join(bad)); sys.exit(1 if bad else 0)")
    result = subprocess.run([sys.executable, "-s", "-c", code], cwd=parent, capture_output=True, text=True,
                            env={"PYTHONPATH": str(parent), "PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("zstd") is None or sys.version_info >= (3, 14), reason="needs the zstd CLI")
def test_zstd_stream_glue_with_a_stand_in_module(tmp_path, example, monkeypatch):
    # Exercises the zstandard stream_reader path (text wrapper over a byte reader) without the package.
    import types

    calls = []

    class Frame:
        """One frame's decompressobj, through the zstd CLI: finished once the bytes so far decompress."""

        def __init__(self):
            self.seen, self.eof, self.unused_data = b"", False, b""

        def decompress(self, data):
            calls.append(len(data))
            self.seen += data
            done = subprocess.run(["zstd", "-q", "-d", "-c"], input=self.seen, capture_output=True)
            if done.returncode:
                return b""
            self.eof = True
            return done.stdout

    class Decompressor:
        def decompressobj(self):
            return Frame()

    monkeypatch.setitem(sys.modules, "zstandard", types.SimpleNamespace(ZstdDecompressor=Decompressor))
    plain = write_jsonl(tmp_path / "z.jsonl", records_from(example.select_frames(slice(0, 4))))
    subprocess.run(["zstd", "-q", "-f", str(plain), "-o", str(tmp_path / "z.jsonl.zst")], check=True)
    same_array(R.load_inference(tmp_path / "z.jsonl.zst")[3].points, R.load_inference(plain)[3].points)
    assert calls == [(tmp_path / "z.jsonl.zst").stat().st_size]
    # A file cut inside its frame is an error, not a shorter video.
    whole = (tmp_path / "z.jsonl.zst").read_bytes()
    (tmp_path / "cut.jsonl.zst").write_bytes(whole[:len(whole) // 2])
    with pytest.raises(EOFError, match="truncated"):
        R.load_inference(tmp_path / "cut.jsonl.zst")


# ------------------------------------------------------------------ robustness (backcast rehearsal, 0.5.0.dev13)
def manifest_of(directory):
    return json.loads((directory / "manifest.json").read_text())


def test_csv_tables_of_a_sequence_without_arms(tmp_path, example):
    records = records_from(example.select_frames(slice(0, 3)))
    for record in records:
        record["stars"][0]["arms"] = []
    path = write_jsonl(tmp_path / "noarms.jsonl", records)
    assert run_cli(path, "--output", tmp_path / "o", "--csv", "--preset", "fig4b", "fig6_flipping", "fig6c") == 0
    rows = manifest_of(tmp_path / "o")["sequences"]
    assert [row["status"] for row in rows] == ["ok"] * 3 and all(row["arms"] == 0 for row in rows)
    with (tmp_path / "o" / "fig4b" / "noarms__star3__arms.csv").open() as fh:
        assert list(csv.DictReader(fh)) == []
    with (tmp_path / "o" / "fig4b" / "noarms__star3__frames.csv").open() as fh:
        assert len(list(csv.DictReader(fh))) == 3


def test_zstd_truncation_with_zstandard(tmp_path, example):
    zstandard = pytest.importorskip("zstandard")
    records = records_from(example.select_frames(slice(0, 30)))
    text = "".join(json.dumps(r) + "\n" for r in records).encode()
    # two frames, as appending zstd outputs (or a streaming writer) produces
    middle = text.index(b"\n", len(text) // 2) + 1
    whole = zstandard.ZstdCompressor().compress(text[:middle]) + zstandard.ZstdCompressor().compress(text[middle:])
    (tmp_path / "two.jsonl.zst").write_bytes(whole)
    assert R.load_inference(tmp_path / "two.jsonl.zst")[3].n_frames == 30
    for cut in (len(whole) - 7, len(whole) * 3 // 4, len(whole) // 3):
        (tmp_path / "cut.jsonl.zst").write_bytes(whole[:cut])
        with pytest.raises((EOFError, ContainerError), match="truncated"):
            R.load_inference(tmp_path / "cut.jsonl.zst")
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "cut.jsonl.zst").write_bytes(whole[:len(whole) * 3 // 4])
    assert run_cli(tmp_path / "in", "--output", tmp_path / "o", "--preset", "fig4b") == 1
    assert manifest_of(tmp_path / "o")["inputs"][0]["status"] == "error"


def test_runner_manifest_is_not_an_input(tmp_path, example):
    (tmp_path / "ann" / "cam1").mkdir(parents=True)
    write_jsonl(tmp_path / "ann" / "cam1" / "C0001.mp4.jsonl", records_from(example.select_frames(slice(0, 5))))
    (tmp_path / "ann" / "_manifest.jsonl").write_text('{"relative": "cam1/C0001.mp4", "frames": 5}\n')
    assert run_cli(tmp_path / "ann", "--output", tmp_path / "o", "--preset", "fig4b") == 0
    assert [item["name"] for item in manifest_of(tmp_path / "o")["inputs"]] == ["cam1/C0001.mp4"]


def test_cameras_with_the_same_file_names_stay_apart_and_overwrite_clears_stale_outputs(tmp_path, example):
    tracked = tmp_path / "tracked"
    for cam, part in (("camA", slice(0, 5)), ("camB", slice(5, 12))):
        (tracked / cam).mkdir(parents=True)
        write_jsonl(tracked / cam / "C0001.mp4.jsonl", records_from(example.select_frames(part)))
    out = tmp_path / "o"
    assert run_cli(tracked / "camB", "--output", out, "--preset", "fig4b", "--csv") == 0
    assert (out / "fig4b" / "C0001.mp4__star3.npz").is_file()
    assert run_cli(tracked, "--output", out, "--preset", "fig4b", "--csv", "--overwrite") == 0
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == ["containers/camA/C0001.mp4__star3.npz", "containers/camB/C0001.mp4__star3.npz",
                     "fig4b/camA/C0001.mp4__star3.npz", "fig4b/camA/C0001.mp4__star3__arms.csv",
                     "fig4b/camA/C0001.mp4__star3__frames.csv", "fig4b/camB/C0001.mp4__star3.npz",
                     "fig4b/camB/C0001.mp4__star3__arms.csv", "fig4b/camB/C0001.mp4__star3__frames.csv",
                     "manifest.csv", "manifest.json"]
    assert PolylineContainer.load(out / "containers" / "camA" / "C0001.mp4__star3.npz").n_frames == 5
    assert PolylineContainer.load(out / "containers" / "camB" / "C0001.mp4__star3.npz").n_frames == 7
    rows = manifest_of(out)["sequences"]
    assert [(row["sequence"], row["outputs"], row["frames_csv"]) for row in rows] == [
        ("camA/C0001.mp4__star3", "fig4b/camA/C0001.mp4__star3.npz", "camA/C0001.mp4__star3__frames.csv"),
        ("camB/C0001.mp4__star3", "fig4b/camB/C0001.mp4__star3.npz", "camB/C0001.mp4__star3__frames.csv")]
    # --overwrite clears only a representations output directory
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "notes.txt").write_text("keep")
    with pytest.raises(SystemExit):
        run_cli(tracked, "--output", tmp_path / "other", "--overwrite")
    assert (tmp_path / "other" / "notes.txt").read_text() == "keep"
    with pytest.raises(SystemExit):     # two directories with the same relative names
        run_cli(tracked / "camA", tracked / "camB", "--output", tmp_path / "x")


def test_frames_without_a_record_become_no_star_rows(tmp_path, example):
    part = example.select_frames(slice(0, 40))
    dropped = [r for i, r in enumerate(records_from(part)) if not 10 <= i < 20]
    zero = records_from(part, drop=set(range(10, 20)))
    with pytest.warns(UserWarning, match="10 frames between 640 and 679 have no record"):
        a = R.load_inference(write_jsonl(tmp_path / "dropped.jsonl", dropped))[3]
    b = R.load_inference(write_jsonl(tmp_path / "zero.jsonl", zero))[3]
    assert a.n_frames == b.n_frames == 40 and a.meta["missing_records"] == 10 and "missing_records" not in b.meta
    for name in ("frame_index", "time_s", "has_star", "crop", "body_origin_model", "points", "frame_arm_offsets"):
        same_array(getattr(a, name), getattr(b, name))
    assert "no record" in " ".join(a.meta["warnings"])
    for name, path in (("a", tmp_path / "dropped.jsonl"), ("b", tmp_path / "zero.jsonl")):
        assert run_cli(path, "--output", tmp_path / name, "--preset", "fig4e", "--r0", "100") == 0
    with np.load(tmp_path / "a" / "fig4e" / "dropped__star3.npz") as x, \
            np.load(tmp_path / "b" / "fig4e" / "zero__star3.npz") as y:
        assert x.files == y.files and all(np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f")
                                          for k in x.files)
    assert manifest_of(tmp_path / "a")["status"] == "complete_with_warnings"
    assert manifest_of(tmp_path / "a")["inputs"][0]["status"] == "warning"
    assert manifest_of(tmp_path / "b")["status"] == "complete"
    # without a frame rate the times of missing frames are unknown: they are left out, with a warning
    with pytest.warns(UserWarning, match="frame rate is unknown"):
        no_fps = R.load_inference(write_jsonl(tmp_path / "nofps.jsonl", [
            r for i, r in enumerate(records_from(part, fps=None)) if not 10 <= i < 20]))[3]
    assert no_fps.n_frames == 30 and "frame rate is unknown" in no_fps.meta["warnings"][0]
    # an untracked single star fills by frame number too
    archive = [r for i, r in enumerate(records_from(part, archive=True)) if not 10 <= i < 20]
    with pytest.warns(UserWarning, match="no record"):
        single = R.load_inference(write_jsonl(tmp_path / "arch.jsonl", archive), untracked="single")["single"]
    assert single.n_frames == 40 and single.has_star.sum() == 30


def test_untracked_single_points_at_track_videos(tmp_path, example):
    crowded = write_jsonl(tmp_path / "crowded.jsonl", records_from(example.select_frames(slice(0, 6)), archive=True,
                                                                   extra_star={2}))
    with pytest.raises(ContainerError, match="track-videos"):
        R.load_inference(crowded, untracked="single")


def test_single_star_and_multi_star_records_in_one_file_are_refused(tmp_path, example):
    records = records_from(example.select_frames(slice(0, 6)), archive=True)
    for record in records[2:4]:
        del record["stars"]
    with pytest.raises(ContainerError, match="single-star records .* and multi-star records"):
        R.load_inference(write_jsonl(tmp_path / "mixed.jsonl", records), untracked="single")


def test_a_truncated_last_line_names_the_line(tmp_path, example):
    text = "".join(json.dumps(r) + "\n" for r in records_from(example.select_frames(slice(0, 4))))
    (tmp_path / "cut.jsonl").write_text(text[:-50])
    with pytest.raises(ContainerError, match=r"line 4 \(after frame 642\) is not a JSON record.*truncated"):
        R.load_inference(tmp_path / "cut.jsonl")
    (tmp_path / "blank.jsonl").write_text("\n" + text.replace("\n", "\n\n") + "  \n")
    assert R.load_inference(tmp_path / "blank.jsonl")[3].n_frames == 4


def test_empty_inputs_are_reported(tmp_path, example):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "zero.jsonl").write_text("")
    write_jsonl(tmp_path / "in" / "nostars.jsonl", [{**r, "stars": []} for r in records_from(
        example.select_frames(slice(0, 3)))])
    write_jsonl(tmp_path / "in" / "good.jsonl", records_from(example.select_frames(slice(0, 3))))
    with pytest.raises(ContainerError, match="holds no records"):
        R.load_inference(tmp_path / "in" / "zero.jsonl")
    assert run_cli(tmp_path / "in", "--output", tmp_path / "o", "--preset", "fig4b") == 1
    status = {item["name"]: item["status"] for item in manifest_of(tmp_path / "o")["inputs"]}
    assert status == {"good": "ok", "nostars": "empty", "zero": "error"}
    with (tmp_path / "o" / "manifest.csv").open() as fh:
        assert {(row["input"], row["status"]) for row in csv.DictReader(fh)} == {
            ("good", "ok"), ("nostars", "empty"), ("zero", "error")}
    (tmp_path / "in" / "zero.jsonl").unlink()
    assert run_cli(tmp_path / "in", "--output", tmp_path / "p", "--preset", "fig4b") == 0   # empty is not a failure


def test_a_field_missing_on_some_star_rows_marks_the_input(tmp_path, example):
    records = records_from(example.select_frames(slice(0, 6)))
    for record in records[3:]:
        del record["stars"][0]["body_origin_model"]
    path = write_jsonl(tmp_path / "mixed.jsonl", records)
    assert run_cli(path, "--output", tmp_path / "o", "--preset", "fig4b") == 0
    manifest = manifest_of(tmp_path / "o")
    assert manifest["status"] == "complete_with_warnings" and manifest["inputs"][0]["status"] == "warning"
    row = manifest["sequences"][0]
    assert row["status"] == "warning" and "body_origin_model present on 3 of 6" in row["warnings"][0]
    assert run_cli(path, "--output", tmp_path / "p", "--preset", "fig4b", "--require-complete") == 1


# ------------------------------------------------------------------ input discovery and checks (verification of dev13)
def test_a_symlinked_camera_directory_is_read(tmp_path, example):
    for directory in ("in/camA", "real/camB"):
        (tmp_path / directory).mkdir(parents=True)
    write_jsonl(tmp_path / "in" / "camA" / "C0001.mp4.jsonl", records_from(example.select_frames(slice(0, 3))))
    write_jsonl(tmp_path / "real" / "camB" / "C0001.mp4.jsonl", records_from(example.select_frames(slice(0, 4))))
    (tmp_path / "in" / "camB").symlink_to(tmp_path / "real" / "camB", target_is_directory=True)
    assert run_cli(tmp_path / "in", "--output", tmp_path / "o", "--preset", "fig4b") == 0
    assert [item["name"] for item in manifest_of(tmp_path / "o")["inputs"]] == ["camA/C0001.mp4", "camB/C0001.mp4"]


def test_a_track_videos_output_inside_a_raw_tree_is_skipped(tmp_path, example, capsys):
    raw = tmp_path / "raw"
    (raw / "tracked").mkdir(parents=True)
    write_jsonl(raw / "flip1.jsonl", records_from(example.select_frames(slice(0, 3)), archive=True))
    write_jsonl(raw / "tracked" / "flip1.jsonl", records_from(example.select_frames(slice(0, 3))))
    (raw / "tracked" / "_track-videos.log").write_text("{}\n")
    capsys.readouterr()
    assert run_cli(raw, "--output", tmp_path / "o", "--untracked", "single", "--preset", "fig4b") == 0
    assert [item["name"] for item in manifest_of(tmp_path / "o")["inputs"]] == ["flip1"]
    assert "it holds track-videos output" in capsys.readouterr().err
    # the tracked tree on its own is read as usual
    assert run_cli(raw / "tracked", "--output", tmp_path / "t", "--preset", "fig4b") == 0
    assert [item["name"] for item in manifest_of(tmp_path / "t")["inputs"]] == ["flip1"]


def test_an_underscore_record_file_is_skipped_with_a_note(tmp_path, example, capsys):
    (tmp_path / "mix").mkdir()
    for name in ("_C0002.mp4.jsonl", "C0003.mp4.jsonl"):
        write_jsonl(tmp_path / "mix" / name, records_from(example.select_frames(slice(0, 3))))
    capsys.readouterr()
    assert run_cli(tmp_path / "mix", "--output", tmp_path / "o", "--preset", "fig4b") == 0
    assert [item["name"] for item in manifest_of(tmp_path / "o")["inputs"]] == ["C0003.mp4"]
    assert "_C0002.mp4.jsonl" in capsys.readouterr().err
    assert run_cli(tmp_path / "mix" / "_C0002.mp4.jsonl", "--output", tmp_path / "p", "--preset", "fig4b") == 0


def test_a_file_shorter_than_the_fleet_manifest_says_is_an_error(tmp_path, example):
    (tmp_path / "ann" / "cam").mkdir(parents=True)
    write_jsonl(tmp_path / "ann" / "cam" / "C0001.mp4.jsonl", records_from(example.select_frames(slice(0, 4))))
    (tmp_path / "ann" / "_manifest.jsonl").write_text('{"relative": "cam/C0001.mp4", "frames": 6}\n')
    with pytest.raises(ContainerError, match="holds 4 records, but .* lists 6"):
        R.load_inference(tmp_path / "ann" / "cam" / "C0001.mp4.jsonl",
                         expected_records=(6, tmp_path / "ann" / "_manifest.jsonl"))
    assert run_cli(tmp_path / "ann", "--output", tmp_path / "o", "--preset", "fig4b") == 1
    item = manifest_of(tmp_path / "o")["inputs"][0]
    assert item["status"] == "error" and "lists 6 for it" in item["error"]
    assert run_cli(tmp_path / "ann" / "cam" / "C0001.mp4.jsonl", "--output", tmp_path / "p", "--preset", "fig4b") == 1


def test_a_bad_first_line_names_the_line(tmp_path, example):
    lines = [json.dumps(r) for r in records_from(example.select_frames(slice(0, 3)))]
    (tmp_path / "cut1.jsonl").write_text(lines[0][:-40] + "\n" + "\n".join(lines[1:]) + "\n")
    (tmp_path / "only.jsonl").write_text(lines[0][:-40])
    for name in ("cut1.jsonl", "only.jsonl"):
        with pytest.raises(ContainerError, match=r"line 1 \(before any record\) is not a JSON record.*truncated"):
            R.load_inference(tmp_path / name)
    (tmp_path / "number.jsonl").write_text(lines[0] + "\n5\n" + lines[1] + "\n")
    with pytest.raises(ContainerError, match="line 2 is not a frame record"):
        R.load_inference(tmp_path / "number.jsonl")
    (tmp_path / "noframe.jsonl").write_text(lines[0] + '\n{"stars": []}\n')
    with pytest.raises(ContainerError, match="record 2 is not a frame record"):
        R.load_inference(tmp_path / "noframe.jsonl")
    (tmp_path / "pretty.json.txt").write_text(json.dumps({"frames": json.loads("[" + ",".join(lines) + "]")},
                                                         indent=1))
    with (tmp_path / "pretty.json.txt").open() as stream:     # a pretty-printed document is still parsed whole
        assert R.load_inference(stream)[3].n_frames == 3


def test_one_record_off_the_frame_grid_does_not_change_the_spacing():
    assert R._frame_step([0, 2, 4, 6, 8, 41, 42, 44]) == 2
    assert R._frame_span([0, 2, 4, 11, 12, 14], 0, 14) == [0, 2, 4, 6, 8, 10, 11, 12, 14]
    assert R._frame_span([0, 1, 2, 5, 6], 1, 6) == [1, 2, 3, 4, 5, 6]
    assert R._frame_span([7], 7, 7) == [7] and R._frame_step([7]) == 1
