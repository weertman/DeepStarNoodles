# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections import Counter, defaultdict
import itertools

import pytest

from deepstarnoodles_inference.star_tracking import HungarianParams, HungarianStarTracker

WIDTH = 200.0


def box(cx, cy, *, width=WIDTH, confidence=0.9):
    """A star whose box is ``width`` square around (cx, cy) in source px."""
    half = width / 2
    return {"crop": {"bbox_xyxy": [cx - half, cy - half, cx + half, cy + half], "confidence": confidence}}


def run(frames, params=None):
    """IDs per frame for ``[(time_s, [star, ...]), ...]``."""
    tracker = HungarianStarTracker(params)
    return [tracker.update(stars, time_s) for time_s, stars in frames], tracker


def animals_per_id(frames, names, params=None):
    """``{ID: Counter(animal)}`` for frames whose stars are labelled by ``names`` (one list per frame)."""
    ids, _ = run(frames, params)
    carried = defaultdict(Counter)
    for frame_ids, frame_names in zip(ids, names):
        for star_id, name in zip(frame_ids, frame_names):
            carried[star_id][name] += 1
    return carried


def test_a_lone_jump_within_jump_gate_keeps_its_id():
    # A star moved by hand: 2 widths in 1 s, nothing else nearby.
    ids, _ = run([(0.0, [box(1000, 1000)]), (1.0, [box(1000 + 2 * WIDTH, 1000)])])
    assert ids == [[1], [1]]
    ids, _ = run([(0.0, [box(1000, 1000)]), (1.0, [box(1000 + 2.7 * WIDTH, 1000)])])
    assert ids == [[1], [2]]          # beyond jump_gate (2.5)


def test_a_jump_next_to_another_star_starts_a_new_id():
    # The same 2-width jump, but it lands 1 width from a second star: it may as well be another animal.
    other = box(1000 + 2 * WIDTH, 1000 + WIDTH)
    ids, _ = run([(0.0, [box(1000, 1000), other]),
                  (1.0, [box(1000 + 2 * WIDTH, 1000), box(1000 + 2 * WIDTH, 1000 + WIDTH)])])
    assert ids == [[1, 2], [3, 2]]


def test_a_lost_star_comes_back_only_near_where_it_was():
    for seconds in (10.0, 25.0):
        ids, _ = run([(0.0, [box(1000, 1000)]), (seconds, [box(1000 + 0.5 * WIDTH, 1000)])])
        assert ids == [[1], [1]]
    ids, _ = run([(0.0, [box(1000, 1000)]), (10.0, [box(1000 + 1.2 * WIDTH, 1000)])])
    assert ids == [[1], [2]]          # 0.75 + 0.02 * 10 = 0.95 widths
    ids, _ = run([(0.0, [box(1000, 1000)]), (30.0, [box(1000 + 1.2 * WIDTH, 1000)])])
    assert ids == [[1], [1]]          # 0.75 + 0.02 * 30 = 1.35 widths
    ids, _ = run([(0.0, [box(1000, 1000)]), (30.0, [box(1000 + 1.6 * WIDTH, 1000)])])
    assert ids == [[1], [2]]          # capped at 1.5 widths


def test_a_lost_star_is_not_taken_back_next_to_another_track():
    # The newcomer is where the lost star was, but a tracked star is 1 width away: no re-acquisition.
    still = box(1000, 1000 + WIDTH)
    frames = [(0.0, [box(1000, 1000), still])] + [(float(t), [still]) for t in range(1, 10)]
    frames.append((10.0, [box(1000, 1000), still]))
    ids, _ = run(frames)
    assert ids[-1] == [3, 2]


def test_a_track_unseen_longer_than_max_gap_s_ends():
    ids, tracker = run([(0.0, [box(1000, 1000)]), (30.0, [box(1000, 1000)])])
    assert ids == [[1], [1]] and tracker.ended == []
    tracker.update([], 60.0)
    assert tracker.ended == [] and tracker.live_ids == [1]   # exactly max_gap_s: still open
    assert tracker.update([box(1000, 1000)], 60.5) == [2]
    assert tracker.ended == [1] and tracker.live_ids == [2]
    ids, _ = run([(0.0, [box(1000, 1000)]), (100.0, [box(1000, 1000)])], HungarianParams(max_gap_s=120.0))
    assert ids == [[1], [1]]


def test_the_less_confident_of_two_duplicate_boxes_gets_no_id():
    first, twin = box(1000, 1000, confidence=0.6), box(1001, 1000, confidence=0.9)
    ids, _ = run([(0.0, [first, twin]), (0.5, [box(1002, 1000)])])
    assert ids == [[None, 1], [1]]
    ids, _ = run([(0.0, [first, twin])], HungarianParams(dup_iou=0.0))    # 0 turns it off
    assert ids == [[1, 2]]
    near = box(1000 + 0.2 * WIDTH, 1000)                                  # IoU ~0.67: two stars
    ids, _ = run([(0.0, [box(1000, 1000), near])])
    assert ids == [[1, 2]]


def test_a_size_change_over_size_gate_is_refused():
    ids, _ = run([(0.0, [box(1000, 1000)]), (0.5, [box(1000, 1000, width=2.1 * WIDTH)])])
    assert ids == [[1], [2]]
    ids, _ = run([(0.0, [box(1000, 1000)]), (0.5, [box(1000, 1000, width=1.9 * WIDTH)])])
    assert ids == [[1], [1]]
    ids, _ = run([(0.0, [box(1000, 1000)]), (0.5, [box(1000, 1000, width=WIDTH / 2.1)])])
    assert ids == [[1], [2]]


def test_two_stars_passing_each_other_keep_their_ids():
    frames = []
    for step in range(20):
        offset = 0.3 * WIDTH * step
        frames.append((float(step), [box(1000 + offset, 1000), box(1000 + 5.7 * WIDTH - offset, 1000 + WIDTH)]))
    ids, _ = run(frames)
    assert all(frame == [1, 2] for frame in ids)


@pytest.mark.parametrize("lateral", [0.0, 0.3, 0.6])
def test_two_stars_crossing_head_on_never_swap(lateral):
    # 0.2 widths/s each, at 1 fps, both detected throughout (cross_v0.2_y0.0_both): the motion model follows each
    # star through the crossing; where the boxes coincide the tracker may split an ID, but no ID changes animal.
    frames, names = [], []
    for step in range(60):
        offset = 0.2 * WIDTH * step
        frames.append((float(step), [box(1000 + offset, 1000),
                                     box(1000 + 12 * WIDTH - offset, 1000 + lateral * WIDTH)]))
        names.append(["A", "B"])
    carried = animals_per_id(frames, names)
    assert all(len(animals) == 1 for animals in carried.values())
    if lateral >= 0.3:
        assert len(carried) == 2


def test_a_tracked_star_keeps_its_id_when_a_lost_one_returns_where_it_was():
    # leave20_bswap: A leaves for 20 s; B walks into A's spot; A comes back where B used to be. B was seen a
    # second ago, A twenty seconds ago, so B keeps its ID and A gets a new one.
    frames, names = [], []
    for t in range(100):
        b_x = 1000 + 1.5 * WIDTH if t < 60 else max(1000.0, 1000 + 1.5 * WIDTH - 1.5 * WIDTH * (t - 60) / 20)
        stars, labels = [box(b_x, 1000)], ["B"]
        if t < 60:
            stars.append(box(1000, 1000)); labels.append("A")
        elif t >= 80:
            stars.append(box(1000 + 1.5 * WIDTH, 1000)); labels.append("A")
        frames.append((float(t), stars)); names.append(labels)
    carried = animals_per_id(frames, names)
    assert all(len(animals) == 1 for animals in carried.values())
    assert carried[1] == Counter({"B": 100})


@pytest.mark.parametrize("gap, offset", [(5.0, 1.0), (20.0, 1.3), (45.0, 0.0)])
def test_a_newcomer_near_where_a_star_left_gets_a_new_id(gap, offset):
    # subst: A leaves at t = 60; a different star appears ``gap`` s later ``offset`` widths away, beyond the
    # re-acquisition reach (0.75 + 0.02 widths per second unseen) or after max_gap_s. (Closer than that, or at the
    # same spot within 30 s, it takes A's ID: from boxes alone it could be A.)
    frames = [(float(t), [box(1000, 1000)]) for t in range(60)]
    frames += [(60 + gap + t, [box(1000 + offset * WIDTH, 1000)]) for t in range(20)]
    ids, _ = run(frames)
    assert {i for frame in ids[60:] for i in frame} == {2}


def test_one_detection_between_two_tracks_is_ambiguous():
    # Two stars 0.3 widths apart; next second only one box, halfway: either could be it, so neither ID goes on.
    ids, tracker = run([(0.0, [box(1000, 1000), box(1000 + 0.3 * WIDTH, 1000)]),
                        (1.0, [box(1000 + 0.15 * WIDTH, 1000)])])
    assert ids == [[1, 2], [3]]
    assert sorted(tracker.ended) == [1, 2] and tracker.live_ids == [3]
    ids, _ = run([(0.0, [box(1000, 1000), box(1000 + 0.3 * WIDTH, 1000)]),
                  (1.0, [box(1000 + 0.15 * WIDTH, 1000)])], HungarianParams(ambiguity=0.0))   # 0 turns it off
    assert ids[1] in ([1], [2])


def test_ids_are_numbered_from_one_in_order_of_birth():
    frames = [(0.0, [box(3000, 1000)]), (1.0, [box(3000, 1000), box(500, 500)]),
              (2.0, [box(8000, 8000), box(3000, 1000), box(500, 500)])]
    ids, _ = run(frames)
    assert ids == [[1], [1, 2], [3, 1, 2]]


def test_the_order_of_detections_in_a_record_does_not_change_the_partition():
    rows = [[box(1000 + 40 * t, 1000), box(2000, 1500 - 30 * t), box(400, 3000 + 25 * t, confidence=0.5)]
            for t in range(8)]
    reference = None
    for order in itertools.permutations(range(3)):
        tracker = HungarianStarTracker()
        groups: dict = {}
        for t, stars in enumerate(rows):
            for k, star_id in zip(order, tracker.update([stars[k] for k in order], float(t))):
                groups.setdefault(star_id, set()).add((t, k))
        partition = sorted(sorted(group) for group in groups.values())
        reference = reference or partition
        assert partition == reference
    assert len(reference) == 3


def test_min_conf_new_keeps_low_confidence_detections_from_starting_tracks():
    params = HungarianParams(min_conf_new=0.5)
    ids, _ = run([(0.0, [box(1000, 1000, confidence=0.3)]), (1.0, [box(1000, 1000, confidence=0.6)]),
                  (2.0, [box(1000, 1000, confidence=0.3)])], params)
    assert ids == [[None], [1], [1]]


def test_times_must_not_decrease():
    tracker = HungarianStarTracker()
    tracker.update([box(1000, 1000)], 5.0)
    with pytest.raises(ValueError, match="must not decrease"):
        tracker.update([box(1000, 1000)], 4.0)


def test_parameters_are_checked():
    assert (HungarianParams.from_mapping({"jump_gate": "2", "size_hist": "5"})
            == HungarianParams(jump_gate=2.0, size_hist=5))
    assert HungarianParams.from_mapping(None) == HungarianParams()
    with pytest.raises(ValueError, match="unknown star-tracker parameters: gate"):
        HungarianParams.from_mapping({"gate": 1})
    for bad in ({"reacquire_gate": "2"}, {"reacquire_gate": 0}, {"size_gate": 1.0}, {"max_gap_s": 0},
                {"dup_iou": 1.5}, {"size_hist": "2.5"}, {"size_hist": 0}, {"sigma": "nan"}, {"sigma": 0},
                {"new_cost": -1}, {"ambiguity": -0.1}, {"velocity_alpha": 1.5}, {"jump_gate": "wide"},
                {"min_conf_new": 2}, {"recent_s": 0}):
        with pytest.raises(ValueError):
            HungarianParams.from_mapping(bad)
    assert HungarianParams().as_dict() == {
        "sigma": 0.25, "sigma_growth": 0.05, "new_cost": 3.5, "ambiguity": 0.4, "velocity_alpha": 0.3,
        "velocity_horizon_s": 5.0, "size_weight": 0.5, "size_gate": 2.0, "size_hist": 3, "recent_s": 4.5,
        "reacquire_gate": 0.75, "reacquire_gate_per_s": 0.02, "reacquire_gate_max": 1.5, "reacquire_clear": 1.5,
        "jump_gate": 2.5, "max_gap_s": 30.0, "dup_iou": 0.95, "min_conf_new": 0.0}
