# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import random
import shutil
import subprocess

import numpy as np
import pytest

from deepstarnoodles_inference import video_tracking
from deepstarnoodles_inference.runtime import StarArmTrackerRegistry

ARMS = 5


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


def test_a_star_appearing_later_waits_in_untracked_stars_for_one_frame():
    tracked = list(video_tracking.track_records(video(8, late=3)))
    assert [len(record["stars"]) for record in tracked] == [2, 2, 2, 2, 3, 3, 3, 3]
    assert [len(record["untracked_stars"]) for record in tracked] == [0, 0, 0, 1, 0, 0, 0, 0]
    newcomer = tracked[3]["untracked_stars"][0]
    assert newcomer["star_track_id"] is None and all(arm["track_id"] is None for arm in newcomer["arms"])
    assert len({item["star_track_id"] for record in tracked for item in record["stars"]}) == 3


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
