# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import pytest

from deepstarnoodles_inference import __version__, video_tracking
from deepstarnoodles_inference.runtime import StarArmTrackerRegistry
from deepstarnoodles_inference.star_tracking import HungarianParams

ARMS = 5
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "track_videos"
REAL = ("flip_r1_f030-089.jsonl.gz", "lab_stress_f000-039.jsonl.gz")


def real_records(name):
    """Real dev12 serve-videos records, cut down (tests/fixtures/track_videos/README.md)."""
    with gzip.open(FIXTURES / name, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def dumps(records):
    return b"".join((json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
                    for record in records)


def star(center_source, angle, *, seed, confidence=0.9, side=400.0):
    """One star record: arms radiating from the crop's center, in shuffled order, as serve-videos writes them."""
    left, top = center_source[0] - side / 2, center_source[1] - side / 2
    center = np.asarray([256.0, 256.0])
    arms = []
    for k in range(ARMS):
        theta = angle + 2 * np.pi * k / ARMS
        radius = np.linspace(30, 180, 10)[:, None]
        polyline = center + radius * np.asarray([np.cos(theta), np.sin(theta)])
        identity = np.zeros(8)
        identity[k] = 1.0
        arms.append({"polyline_crop": polyline.tolist(), "identity": identity.tolist(), "true_arm": k})
    random.Random(seed).shuffle(arms)
    for index, arm in enumerate(arms):
        arm["arm_index"] = index
    return {"star_track_id": None, "crop": {"left": left, "top": top, "side": side, "confidence": confidence,
                                            "bbox_xyxy": [left + 40, top + 40, left + side - 40, top + side - 40]},
            "arms": arms, "body_origin_model": center.tolist(), "body_origin_code": 0}


def side(item):
    return "left" if item["crop"]["left"] < 800 else "right"


def video(frames=12, late=None):
    """Two stars throughout, and optionally a third from frame ``late`` on."""
    records = []
    for frame in range(frames):
        stars = [star((300 + 3 * frame, 400.0), 0.02 * frame, seed=frame),
                 star((1400.0, 900 - 2 * frame), 0.4 - 0.03 * frame, seed=100 + frame)]
        if late is not None and frame >= late:
            stars.append(star((900.0, 1500.0), 0.1, seed=200 + frame))
        records.append({"frame": frame, "video": "clip", "stars": stars[::-1] if frame % 2 else stars})
    return records


def test_stars_and_arms_keep_their_identities_through_a_video():
    tracked = list(video_tracking.track_records(video()))
    by_position = {}
    for record in tracked:
        assert record["untracked_stars"] == []
        for item in record["stars"]:
            assert item["star_track_id"] is not None
            by_position.setdefault(side(item), set()).add(item["star_track_id"])
            # Each arm keeps one ID whatever its position in the list.
            labels = {arm["true_arm"]: arm["track_id"] for arm in item["arms"]}
            by_position.setdefault((side(item), "arms"), []).append(labels)
    assert len(by_position["left"]) == len(by_position["right"]) == 1
    assert by_position["left"] != by_position["right"]
    for name in ("left", "right"):
        first, *rest = by_position[(name, "arms")]
        assert len(set(first.values())) == ARMS
        assert all(labels == first for labels in rest)
        # The per-frame order really was shuffled, so arm_index alone would not have done this.
    orders = [[arm["true_arm"] for arm in record["stars"][0]["arms"]] for record in tracked]
    assert len({tuple(order) for order in orders}) > 1


def test_with_botsort_a_star_appearing_later_waits_in_untracked_stars_for_one_frame():
    tracked = list(video_tracking.track_records(video(8, late=3), star_tracker="botsort"))
    assert [len(record["stars"]) for record in tracked] == [2, 2, 2, 2, 3, 3, 3, 3]
    assert [len(record["untracked_stars"]) for record in tracked] == [0, 0, 0, 1, 0, 0, 0, 0]
    newcomer = tracked[3]["untracked_stars"][0]
    assert newcomer["star_track_id"] is None and all(arm["track_id"] is None for arm in newcomer["arms"])
    assert len({item["star_track_id"] for record in tracked for item in record["stars"]}) == 3
    assert all("star_tracking" not in record for record in tracked)


def test_by_default_a_star_is_tracked_in_the_first_frame_it_appears_in():
    tracked = list(video_tracking.track_records(video(8, late=3)))
    assert [len(record["stars"]) for record in tracked] == [2, 2, 2, 3, 3, 3, 3, 3]
    assert all(record["untracked_stars"] == [] for record in tracked)
    assert {item["star_track_id"] for item in tracked[3]["stars"]} == {1, 2, 3}
    assert len({item["star_track_id"] for record in tracked for item in record["stars"]}) == 3
    for record in tracked:
        assert list(record)[-2:] == ["untracked_stars", "star_tracking"]
        assert record["star_tracking"]["tracker"] == "hungarian"
        assert record["star_tracking"]["params"] == HungarianParams().as_dict()
        assert record["star_tracking"]["version"] == __version__


def test_a_saved_center_tracks_arms_exactly_as_the_hydra_maps_would():
    def maps_for(center):
        maps = np.zeros((2, 512, 512), dtype=np.float32)
        maps[1, int(center[1]), int(center[0])] = 0.9  # a mouth peak where the saved center is
        return maps

    by_maps, by_center = StarArmTrackerRegistry(), StarArmTrackerRegistry()
    for frame, record in enumerate(video()):
        for item in record["stars"]:
            arms = [{"polyline": arm["polyline_crop"], "identity": arm["identity"]} for arm in item["arms"]]
            center = item["body_origin_model"]
            star_id = 1 if side(item) == "left" else 2
            assert (by_maps.track(star_id, frame, maps_for(center), arms, support_mouth_only=True)
                    == by_center.track(star_id, frame, None, arms, center=center))


def test_frames_out_of_order_are_refused():
    records = video(3)
    records[2]["frame"] = 0
    with pytest.raises(ValueError, match="frames must increase"):
        list(video_tracking.track_records(records))


def test_records_without_a_body_origin_are_refused():
    records = video(2)
    del records[0]["stars"][0]["body_origin_model"]
    with pytest.raises(ValueError, match="no body origin"):
        list(video_tracking.track_records(records))


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record) + "\n" for record in records)
    if path.name.endswith(".zst"):
        subprocess.run(["zstd", "-q", "-o", str(path)], input=text.encode(), check=True)
    else:
        path.write_text(text)


def read(path):
    return list(video_tracking.read_records(path))


@pytest.mark.parametrize("suffix", [".jsonl", pytest.param(".jsonl.zst", marks=pytest.mark.skipif(
    shutil.which("zstd") is None, reason="zstd not installed"))])
def test_command_mirrors_a_directory_and_resumes(tmp_path, suffix, capsys):
    source = tmp_path / "annotations"
    write(source / "cam1" / f"day1.mp4{suffix}", video(6))
    write(source / "cam2" / f"day1.mp4{suffix}", video(4))
    (source / "_manifest.jsonl").write_text('{"relative": "cam1/day1.mp4"}\n')
    (source / "cam1" / "worker.0.json").write_text("{}\n")
    out = tmp_path / "tracked"
    assert video_tracking.main([str(source), "--output", str(out), "--jobs", "1"]) == 0
    tracked = read(out / "cam1" / f"day1.mp4{suffix}")
    assert len(tracked) == 6 and all(item["star_track_id"] for item in tracked[0]["stars"])
    assert not (out / "_manifest.jsonl").exists()
    assert video_tracking.main([str(source), "--output", str(out), "--jobs", "1"]) == 0
    assert '"already_tracked": 2' in capsys.readouterr().out


def test_tracked_output_feeds_representations_without_untracked_modes(tmp_path):
    from deepstarnoodles_inference.representations import ContainerError, load_inference

    raw, tracked = tmp_path / "clip.jsonl", tmp_path / "tracked" / "clip.jsonl"
    write(raw, video(8, late=3))
    with pytest.raises(ContainerError, match="without a star_track_id"):
        load_inference(raw, fps=1.0)
    video_tracking.track_file(raw, tracked)
    stars = load_inference(tracked, fps=1.0)
    assert sorted(stars) == [1, 2, 3]


def test_command_refuses_to_overwrite_its_input(tmp_path):
    write(tmp_path / "day1.mp4.jsonl", video(2))
    with pytest.raises(SystemExit):
        video_tracking.main([str(tmp_path), "--output", str(tmp_path)])


@pytest.mark.parametrize("name", REAL)
def test_botsort_output_is_byte_identical_to_dev12(name):
    golden = json.loads((FIXTURES / "dev12_botsort.json").read_text())[name]
    tracked = list(video_tracking.track_records(real_records(name), star_tracker="botsort"))
    ids = [[record["frame"], [[item["star_track_id"], [arm["track_id"] for arm in item["arms"]]]
                              for item in record["stars"]], len(record["untracked_stars"])] for record in tracked]
    assert ids == golden["ids"]
    assert hashlib.sha256(dumps(tracked)).hexdigest() == golden["sha256"]


def degenerate_arms(frame):
    """Arms with one point, zero length, and lengths that cross, to reach every branch of the shape distance."""
    arms = []
    for k in range(6):
        theta = 0.3 * frame + k
        if k == 0:
            polyline = [[256.0 + 40 * np.cos(theta), 256.0 + 40 * np.sin(theta)]]
        elif k == 1:
            polyline = [[300.0, 300.0]] * 4
        else:
            radius = np.linspace(20, 60 + 25 * k + 10 * np.sin(frame), 6 + k)[:, None]
            polyline = (256.0 + radius * np.asarray([np.cos(theta), np.sin(theta)])).tolist()
        identity = np.zeros(8)
        identity[k] = 1.0 - 0.05 * (frame % 3)
        arms.append({"polyline_crop": polyline, "identity": identity.tolist()})
    return arms[frame % 3:] if frame % 4 else []


@pytest.mark.parametrize("star_tracker", video_tracking.STAR_TRACKERS)
@pytest.mark.parametrize("name", [*REAL, "degenerate"])
def test_track_videos_arm_ids_match_the_runtime_registry(name, star_tracker):
    """The vendored array tracker gives the arm IDs runtime.StarArmTrackerRegistry (package fused_tracker) gives."""
    if name == "degenerate":
        sequences = [(frame, [(1, degenerate_arms(frame), [256.0, 250.0])]) for frame in range(40)]
    else:
        sequences = [(record["frame"], [(item["star_track_id"], item["arms"], item["body_origin_model"])
                                        for item in record["stars"]])
                     for record in video_tracking.track_records(real_records(name), star_tracker=star_tracker)]
    runtime, fast = StarArmTrackerRegistry(), video_tracking.ArmTrackers()
    compared = 0
    for frame, stars in sequences:
        for star_id, arms, center in stars:
            expected = runtime.track(star_id, frame, None, [{"polyline": arm["polyline_crop"],
                                                             "identity": arm["identity"]} for arm in arms],
                                     center=center)
            assert fast.track(star_id, frame, arms, center) == expected
            compared += len(arms)
        runtime.prune(frame)
        fast.prune(frame)
        assert fast.star_track_ids == runtime.star_track_ids
    assert compared > 100


def test_arm_trackers_refuse_non_finite_input():
    arms = degenerate_arms(1)
    arms[2]["identity"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        video_tracking.ArmTrackers().track(1, 0, arms, [256.0, 256.0])


def test_by_default_the_real_flip_clip_is_one_star_and_only_the_duplicate_box_is_untracked():
    records = real_records(REAL[0])
    tracked = list(video_tracking.track_records(records))
    assert {item["star_track_id"] for record in tracked for item in record["stars"]} == {1}
    untracked = [(record["frame"], len(record["untracked_stars"])) for record in tracked if record["untracked_stars"]]
    assert untracked == [(75, 1)]
    assert sum(len(record["stars"]) + len(record["untracked_stars"]) for record in tracked) == 61
    frame75 = next(record for record in tracked if record["frame"] == 75)
    assert frame75["untracked_stars"][0]["crop"]["confidence"] < frame75["stars"][0]["crop"]["confidence"]
    # botsort splits the same star and holds back two detections
    botsort = list(video_tracking.track_records(real_records(REAL[0]), star_tracker="botsort"))
    assert len({item["star_track_id"] for record in botsort for item in record["stars"]}) == 2


def test_by_default_every_lab_detection_is_tracked_and_ids_stay_within_a_hold():
    tracked = list(video_tracking.track_records(real_records(REAL[1])))
    assert all(record["untracked_stars"] == [] for record in tracked)
    for record in tracked:
        ids = [item["star_track_id"] for item in record["stars"]]
        assert len(set(ids)) == len(ids)
    # stills are held 5 frames: inside a hold an ID stays on one box (the detector may miss a star for a frame)
    for start in range(0, 40, 5):
        boxes = {}
        for record in tracked[start:start + 5]:
            for item in record["stars"]:
                first = boxes.setdefault(item["star_track_id"], item["crop"]["bbox_xyxy"])
                assert np.abs(np.subtract(item["crop"]["bbox_xyxy"], first)).max() < 3


def gap_video(seconds_missing, fps=1.0):
    """One star crawling 0.03 box widths per record, with ``seconds_missing`` seconds of records removed."""
    records = []
    resume = 10 + int(seconds_missing * fps)
    for index, frame in enumerate([*range(10), *range(resume, resume + 10)]):
        records.append({"frame": frame, "stars": [star((300 + 10 * index, 400.0), 0.05 * index, seed=frame)],
                        "time_s": frame / fps, "fps": fps})
    return records


def test_ids_continue_across_missing_records_and_end_after_max_gap_s():
    tracked = list(video_tracking.track_records(gap_video(15)))
    assert {item["star_track_id"] for record in tracked for item in record["stars"]} == {1}
    before, after = tracked[9]["stars"][0]["arms"], tracked[10]["stars"][0]["arms"]
    labels = lambda arms: {arm["true_arm"]: arm["track_id"] for arm in arms}  # noqa: E731
    assert labels(before) == labels(after)          # the star's arm tracker survives the gap
    tracked = list(video_tracking.track_records(gap_video(31)))
    assert [item["star_track_id"] for record in tracked for item in record["stars"]] == [1] * 10 + [2] * 10
    tracked = list(video_tracking.track_records(gap_video(31), star_params={"max_gap_s": "90"}))
    assert {item["star_track_id"] for record in tracked for item in record["stars"]} == {1}


def test_an_ended_star_drops_its_arm_tracker(monkeypatch):
    discarded = []
    original = video_tracking.ArmTrackers.discard
    monkeypatch.setattr(video_tracking.ArmTrackers, "discard",
                        lambda self, star_id: (discarded.append(star_id), original(self, star_id)))
    list(video_tracking.track_records(gap_video(31)))
    assert discarded == [1]


def test_record_time_falls_back_to_fps_then_to_the_frame_number():
    assert video_tracking._record_time({"time_s": 2.5, "fps": 1.0}, 7) == 2.5
    assert video_tracking._record_time({"time_s": None, "fps": 2.0}, 7) == 3.5
    assert video_tracking._record_time({}, 7) == 7.0
    # records without time_s are gated on frame / fps: 12 frames at 4 fps is 3 s, within max_gap_s=5
    records = [{"frame": frame, "fps": 4.0, "stars": [star((300.0, 400.0), 0.0, seed=frame)]} for frame in (0, 12)]
    tracked = list(video_tracking.track_records(records, star_params={"max_gap_s": 5}))
    assert [record["stars"][0]["star_track_id"] for record in tracked] == [1, 1]
    records = [{"frame": frame, "fps": 1.0, "stars": [star((300.0, 400.0), 0.0, seed=frame)]} for frame in (0, 12)]
    tracked = list(video_tracking.track_records(records, star_params={"max_gap_s": 5}))
    assert [record["stars"][0]["star_track_id"] for record in tracked] == [1, 2]


def test_tracker_options_are_checked(tmp_path, capsys):
    write(tmp_path / "in" / "day1.mp4.jsonl", video(2))
    out = str(tmp_path / "out")
    for bad in (["--star-param", "tau_s=1"], ["--star-param", "jump_gate=-1"], ["--star-param", "jump_gate"],
                ["--star-tracker", "botsort", "--star-param", "jump_gate=3"],
                ["--tracker-config", str(video_tracking.DEFAULT_BOTSORT_CONFIG)],
                ["--star-tracker", "botsort", "--tracker-config", str(tmp_path / "missing.yaml")]):
        with pytest.raises(SystemExit):
            video_tracking.main([str(tmp_path / "in"), "--output", out, "--jobs", "1", *bad])
    assert not (tmp_path / "out").exists()       # refused before any job started
    with pytest.raises(ValueError, match="botsort only"):
        list(video_tracking.track_records(video(2), tracker_config=video_tracking.DEFAULT_BOTSORT_CONFIG))
    with pytest.raises(ValueError, match="star_tracker must be"):
        list(video_tracking.track_records(video(2), star_tracker="bytetrack"))


def test_command_records_the_tracker_in_summaries_and_the_run_log(tmp_path, capsys):
    write(tmp_path / "in" / "day1.mp4.jsonl", video(3))
    out = tmp_path / "out"
    assert video_tracking.main([str(tmp_path / "in"), "--output", str(out), "--jobs", "1",
                                "--star-param", "jump_gate=2"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["star_tracker"] == "hungarian" and lines[0]["star_params"]["jump_gate"] == 2.0
    assert lines[1]["star_ids"] == 2 and lines[1]["star_params"]["jump_gate"] == 2.0
    assert read(out / "day1.mp4.jsonl")[0]["star_tracking"]["params"]["jump_gate"] == 2.0
    assert video_tracking.main([str(tmp_path / "in"), "--output", str(tmp_path / "bs"), "--jobs", "1",
                                "--star-tracker", "botsort", "--overwrite"]) == 0
    log = [json.loads(line) for line in (tmp_path / "bs" / video_tracking.RUN_LOG).read_text().splitlines()]
    assert log[0]["star_tracker"] == "botsort" and log[0]["version"] == __version__
    assert "star_tracking" not in read(tmp_path / "bs" / "day1.mp4.jsonl")[0]


@pytest.mark.parametrize("fuse_score", [True, False])
def test_botsort_maps_low_confidence_matches_back_to_their_detections(tmp_path, fuse_score):
    """BoT-SORT's second pass indexes its low-confidence subset; a 0.2 detection must not take another star's ID."""
    config = tmp_path / "botsort.yaml"
    text = video_tracking.DEFAULT_BOTSORT_CONFIG.read_text()
    config.write_text(text.replace("fuse_score: true", f"fuse_score: {str(fuse_score).lower()}"))
    records = []
    for frame in range(8):
        weak = 0.2 if frame >= 5 else 0.9
        records.append({"frame": frame, "stars": [star((300.0, 400.0), 0.0, seed=frame),
                                                  star((1400.0, 900.0), 0.4, seed=100 + frame, confidence=weak)]})
    tracked = list(video_tracking.track_records(records, star_tracker="botsort", tracker_config=config))
    for record in tracked:
        ids = {side(item): item["star_track_id"] for item in record["stars"]}
        assert ids.get("left") == 1 and ids.get("right", 2) == 2
        assert all(side(item) == "right" for item in record["untracked_stars"])
    if not fuse_score:   # without score fusion the 0.2 detections match star 2 in the second pass
        assert all({item["star_track_id"] for item in record["stars"]} == {1, 2} for record in tracked)


needs_zstd = pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd not installed")


def run_command(capsys, *argv):
    code = video_tracking.main([*map(str, argv), "--jobs", "1"])
    captured = capsys.readouterr()
    return code, [json.loads(line) for line in captured.out.splitlines()], captured.err


@pytest.mark.parametrize("star_tracker", video_tracking.STAR_TRACKERS)
def test_tracked_records_are_refused_rather_than_tracked_again(tmp_path, capsys, star_tracker):
    write(tmp_path / "raw" / "day1.mp4.jsonl", video(6, late=2))
    code, _, _ = run_command(capsys, tmp_path / "raw", "--output", tmp_path / "once", "--star-tracker", star_tracker)
    assert code == 0
    code, _, err = run_command(capsys, tmp_path / "once", "--output", tmp_path / "twice")
    assert code == 1 and "already have star IDs" in err
    assert not (tmp_path / "twice" / "day1.mp4.jsonl").exists()
    assert list((tmp_path / "twice").glob("*.part")) == []
    live = video(2)
    live[1]["stars"][0]["star_track_id"] = 4          # live-replay style records carry IDs but no untracked_stars
    with pytest.raises(ValueError, match="already have star IDs"):
        list(video_tracking.track_records(live, star_tracker=star_tracker))


def test_an_output_directory_inside_the_input_is_not_read_back(tmp_path, capsys):
    raw = tmp_path / "annotations"
    write(raw / "cam1" / "day1.mp4.jsonl", video(4))
    for _ in range(2):
        code, lines, _ = run_command(capsys, raw, "--output", raw / "tracked", "--overwrite")
        assert code == 0 and lines[0]["videos"] == 1
    assert sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*") if p.is_file()) == [
        "cam1/day1.mp4.jsonl", f"tracked/{video_tracking.RUN_LOG}", "tracked/cam1/day1.mp4.jsonl"]
    with pytest.raises(SystemExit):
        video_tracking.main([str(raw), "--output", str(raw)])


def test_records_without_a_stars_list_are_refused():
    records = video(3)
    del records[1]["stars"]
    with pytest.raises(ValueError, match="no 'stars' list"):
        list(video_tracking.track_records(records))


@pytest.mark.parametrize("suffix", [".jsonl", pytest.param(".jsonl.zst", marks=needs_zstd)])
def test_a_truncated_file_names_the_line_and_leaves_no_part_file(tmp_path, capsys, suffix):
    text = "".join(json.dumps(record) + "\n" for record in video(5))
    cut = text[:text.rindex("\n", 0, len(text) - 1) + 40]          # the last record is cut 40 bytes in
    source = tmp_path / "in" / f"day1.mp4{suffix}"
    source.parent.mkdir()
    if suffix == ".jsonl":
        source.write_text(cut)
    else:
        whole = subprocess.run(["zstd", "-q", "-c"], input=text.encode(), capture_output=True, check=True).stdout
        source.write_bytes(whole[:len(whole) * 3 // 4])
    code, _, err = run_command(capsys, tmp_path / "in", "--output", tmp_path / "out")
    assert code == 1
    if suffix == ".jsonl":
        assert "line 5 (after frame 3) is not a JSON record" in err and "truncated" in err
    else:
        assert "zstd could not read" in err
    assert [p.name for p in (tmp_path / "out").iterdir()] == [video_tracking.RUN_LOG]


def test_blank_lines_are_skipped(tmp_path):
    lines = [json.dumps(record) for record in video(4)]
    clean, blank = tmp_path / "clean.jsonl", tmp_path / "blank.jsonl"
    clean.write_text("\n".join(lines) + "\n")
    blank.write_text("\n\n".join(lines) + "\n  \n\n")
    video_tracking.track_file(clean, tmp_path / "a" / "x.jsonl")
    video_tracking.track_file(blank, tmp_path / "b" / "x.jsonl")
    assert (tmp_path / "a" / "x.jsonl").read_bytes() == (tmp_path / "b" / "x.jsonl").read_bytes()


def test_an_empty_file_is_an_error(tmp_path, capsys):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "day1.mp4.jsonl").write_text("")
    (tmp_path / "in" / "day2.mp4.jsonl").write_text("\n")
    write(tmp_path / "in" / "day3.mp4.jsonl", [{"frame": 0, "stars": []}])
    code, lines, err = run_command(capsys, tmp_path / "in", "--output", tmp_path / "out")
    assert code == 1 and err.count("holds no records") == 2
    assert lines[-1]["video"].endswith("day3.mp4.jsonl") and lines[-1]["frames"] == 1
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [video_tracking.RUN_LOG, "day3.mp4.jsonl"]


@needs_zstd
def test_one_video_in_two_formats_is_refused(tmp_path, capsys):
    write(tmp_path / "in" / "day1.mp4.jsonl", video(2))
    write(tmp_path / "in" / "day1.mp4.jsonl.zst", video(2))
    with pytest.raises(SystemExit):
        video_tracking.main([str(tmp_path / "in"), "--output", str(tmp_path / "out")])
    assert "are the same video" in capsys.readouterr().err
    write(tmp_path / "camA" / "day9.mp4.jsonl", video(2))
    write(tmp_path / "camB" / "day9.mp4.jsonl", video(2))
    with pytest.raises(SystemExit):
        video_tracking.main([str(tmp_path / "camA" / "day9.mp4.jsonl"), str(tmp_path / "camB" / "day9.mp4.jsonl"),
                             "--output", str(tmp_path / "out")])
    assert "would both be tracked into" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@needs_zstd
def test_a_symlinked_zst_file_is_read(tmp_path):
    write(tmp_path / "store" / "day1.mp4.jsonl.zst", video(3))
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "day1.mp4.jsonl.zst").symlink_to(tmp_path / "store" / "day1.mp4.jsonl.zst")
    assert video_tracking.main([str(tmp_path / "in"), "--output", str(tmp_path / "out"), "--jobs", "1"]) == 0
    assert len(read(tmp_path / "out" / "day1.mp4.jsonl.zst")) == 3


# ------------------------------------------------------------------ input discovery (verification of 0.5.0.dev13)
def test_a_symlinked_camera_directory_is_followed(tmp_path, capsys):
    write(tmp_path / "in" / "camA" / "day1.mp4.jsonl", video(3))
    write(tmp_path / "elsewhere" / "camB" / "day1.mp4.jsonl", video(4))
    (tmp_path / "in" / "camB").symlink_to(tmp_path / "elsewhere" / "camB", target_is_directory=True)
    (tmp_path / "in" / "loop").symlink_to(tmp_path / "in", target_is_directory=True)   # a cycle is entered once
    code, lines, _ = run_command(capsys, tmp_path / "in", "--output", tmp_path / "out")
    assert code == 0 and lines[0]["videos"] == 2
    assert len(read(tmp_path / "out" / "camB" / "day1.mp4.jsonl")) == 4


def test_another_track_videos_output_inside_the_input_is_skipped(tmp_path, capsys):
    raw = tmp_path / "annotations"
    write(raw / "cam1" / "day1.mp4.jsonl", video(4))
    assert run_command(capsys, raw, "--output", raw / "tracked_old")[0] == 0
    code, lines, err = run_command(capsys, raw, "--output", tmp_path / "tracked_new")
    assert code == 0 and lines[0]["videos"] == 1
    assert f"skipped {raw / 'tracked_old'}: it holds track-videos output" in err


def test_no_input_files_is_an_error(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    (tmp_path / "only").mkdir()
    (tmp_path / "only" / "_manifest.jsonl").write_text("")
    for directory in ("empty", "only"):
        with pytest.raises(SystemExit) as stop:
            video_tracking.main([str(tmp_path / directory), "--output", str(tmp_path / "out")])
        assert stop.value.code == 2 and "no input files found" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_record_file_starting_with_an_underscore_is_skipped_with_a_note(tmp_path, capsys):
    write(tmp_path / "in" / "_C0002.mp4.jsonl", video(2))
    write(tmp_path / "in" / "C0003.mp4.jsonl", video(2))
    (tmp_path / "in" / "_manifest.jsonl").write_text("")
    code, lines, err = run_command(capsys, tmp_path / "in", "--output", tmp_path / "out")
    assert code == 0 and lines[0]["videos"] == 1
    assert "skipped 1 record file(s) whose names start with '_'" in err and "_C0002.mp4.jsonl" in err
    assert "_manifest" not in err
    code, lines, _ = run_command(capsys, tmp_path / "in" / "_C0002.mp4.jsonl", "--output", tmp_path / "by_name")
    assert code == 0 and lines[1]["frames"] == 2


def test_a_file_given_twice_is_tracked_once(tmp_path, capsys):
    write(tmp_path / "in" / "day1.mp4.jsonl", video(3))
    for argv in ([tmp_path / "in" / "day1.mp4.jsonl"] * 2, [tmp_path / "in", tmp_path / "in" / "day1.mp4.jsonl"]):
        code = video_tracking.main([*map(str, argv), "--output", str(tmp_path / "out"), "--jobs", "2", "--overwrite"])
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert code == 0 and lines[0]["videos"] == 1 and len(lines) == 2


@pytest.mark.parametrize("suffix", [".jsonl", pytest.param(".jsonl.zst", marks=needs_zstd)])
def test_a_file_shorter_than_the_fleet_manifest_says_is_an_error(tmp_path, capsys, suffix):
    # A cut exactly at a record boundary reads as a complete, shorter video; the fleet's manifest knows better.
    write(tmp_path / "ann" / "cam" / f"short.mp4{suffix}", video(6)[:4])
    write(tmp_path / "ann" / "cam" / f"whole.mp4{suffix}", video(6))
    (tmp_path / "ann" / "_manifest.jsonl").write_text(
        '{"relative": "cam/short.mp4", "frames": 6}\n{"relative": "cam/whole.mp4", "frames": 5}\n'
        '{"relative": "cam/whole.mp4", "frames": 6}\n{"relative": "cam/half')  # the last row wins; a cut row is skipped
    code, lines, err = run_command(capsys, tmp_path / "ann", "--output", tmp_path / "out")
    assert code == 1 and "holds 4 records, but" in err and "lists 6 for it" in err
    assert [line["frames"] for line in lines[1:]] == [6]
    assert not (tmp_path / "out" / "cam" / f"short.mp4{suffix}").exists()
    assert list((tmp_path / "out").rglob("*.part")) == []
