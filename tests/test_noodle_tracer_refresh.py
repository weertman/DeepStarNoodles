"""Refreshed Noodle Tracer data: augmentation geometry, cohorts, and locks."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepstarnoodles.noodle_tracer_cache import (
    EXPECTED_COHORT_COUNTS,
    cohort_rows,
    validate_cohort_splits,
)
from deepstarnoodles.noodle_tracer_refresh import augment_image_and_arms, draw_augmentation
from cache_fixed_tracer_split import check_test_lock
from train_noodle_tracer_refreshed import parse_args

REPO = Path(__file__).resolve().parents[1]


def test_draw_augmentation_is_deterministic_and_bounded():
    first = draw_augmentation(20261011, 3, 17)
    assert first == draw_augmentation(20261011, 3, 17)
    assert first["rotate_k"] in range(4)
    for key in ("brightness", "contrast", "saturation"):
        assert 0.85 <= first[key] <= 1.15
    draws = [draw_augmentation(20261011, cycle, 17) for cycle in range(1, 41)]
    assert len({json.dumps(draw, sort_keys=True) for draw in draws}) == 40
    assert {draw["rotate_k"] for draw in draws} == {0, 1, 2, 3}


@pytest.mark.parametrize(
    ("flip_x", "flip_y", "rotate_k"),
    list(itertools.product((False, True), (False, True), range(4))),
)
def test_augmentation_moves_polylines_with_pixels(flip_x: bool, flip_y: bool, rotate_k: int):
    size = 16
    points = np.asarray([[2.0, 3.0], [9.0, 5.0], [13.0, 12.0]], dtype=np.float32)
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    for value, (x, y) in zip((80, 160, 240), points.astype(int)):
        pixels[y, x] = value
    params = {"flip_x": flip_x, "flip_y": flip_y, "rotate_k": rotate_k,
              "brightness": 1.0, "contrast": 1.0, "saturation": 1.0}
    augmented, (moved,) = augment_image_and_arms(Image.fromarray(pixels), [points], params, size=size)
    assert augmented.shape == (size, size, 3)
    for value, (x, y) in zip((80, 160, 240), moved.astype(int)):
        assert augmented[y, x, 0] == value
    assert np.count_nonzero(augmented[..., 0]) == 3


def test_photometric_changes_leave_geometry_and_clip_to_the_crop():
    size = 8
    image = Image.fromarray(np.full((size, size, 3), 100, dtype=np.uint8))
    path = np.asarray([[-2.0, 3.0], [9.5, 7.0]], dtype=np.float32)
    params = {"flip_x": False, "flip_y": False, "rotate_k": 0,
              "brightness": 1.15, "contrast": 0.85, "saturation": 1.1}
    augmented, (moved,) = augment_image_and_arms(image, [path], params, size=size)
    assert not np.array_equal(augmented, np.asarray(image))
    np.testing.assert_array_equal(moved, [[0.0, 3.0], [7.0, 7.0]])


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "source_family", "split_group_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_cohort_rows_select_source_families_in_index_order(tmp_path: Path):
    rows = [
        {"image_path": "a", "source_family": "starseg", "split_group_id": "g1"},
        {"image_path": "b", "source_family": "sleepy_stars", "split_group_id": "g2"},
        {"image_path": "c", "source_family": "arm_pit", "split_group_id": "g3"},
    ]
    write_index(tmp_path / "index.csv", rows)
    assert [row["image_path"] for row in cohort_rows(tmp_path / "index.csv", "all")] == ["a", "b", "c"]
    assert [row["image_path"] for row in cohort_rows(tmp_path / "index.csv", "sleepy")] == ["b"]
    assert [row["image_path"] for row in cohort_rows(tmp_path / "index.csv", "non_sleepy")] == ["a", "c"]
    with pytest.raises(ValueError, match="unknown tracer cohort"):
        cohort_rows(tmp_path / "index.csv", "everything")


def split(group: str, family: str = "starseg") -> list[dict[str, str]]:
    return [{"image_path": group, "source_family": family, "split_group_id": group}]


def test_validate_cohort_splits_rejects_leakage_membership_and_counts():
    good = {"train": split("t"), "validation": split("v"), "test": split("x")}
    validate_cohort_splits(good, "all")
    with pytest.raises(ValueError, match="leakage"):
        validate_cohort_splits({**good, "test": split("t")}, "all")
    with pytest.raises(ValueError, match="missing split_group_id"):
        validate_cohort_splits({**good, "test": [{"source_family": "starseg", "split_group_id": ""}]}, "all")
    with pytest.raises(ValueError, match="entered the non-Sleepy cohort"):
        validate_cohort_splits({**good, "test": split("x", "sleepy_stars")}, "non_sleepy",
                               expected_counts={"train": 1, "validation": 1, "test": 1})
    with pytest.raises(ValueError, match="split counts changed"):
        validate_cohort_splits(good, "sleepy")
    assert EXPECTED_COHORT_COUNTS["sleepy"] == {"train": 403, "validation": 34, "test": 63}


def test_test_split_cache_requires_a_matching_analysis_lock(tmp_path: Path):
    (tmp_path / "test_index.csv").write_text("image_path\nx\n", encoding="utf-8")
    digest = hashlib.sha256((tmp_path / "test_index.csv").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="requires --analysis-lock"):
        check_test_lock(None, tmp_path)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"test_source_index_sha256": digest}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        check_test_lock(lock, tmp_path)
    lock.write_text(json.dumps({"test_access_locked": True, "test_source_index_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="test index differs"):
        check_test_lock(lock, tmp_path)
    lock.write_text(json.dumps({"test_access_locked": True, "test_source_index_sha256": digest}))
    assert check_test_lock(lock, tmp_path)["test_access_locked"] is True


def test_refreshed_driver_defaults_are_the_released_recipe():
    args = parse_args(["--output", "RUNS/x"])
    assert (args.cohort, args.cycles, args.steps_per_cycle) == ("all", 30, 4000)
    assert (args.seed, args.augmentation_seed, args.validation_cohort) == (20261011, 20261011, "sleepy")
    assert args.checkpoint.name == "epoch_060.pt"
    assert args.normalization_json == REPO / "config/recipes/noodle_tracer_fullcorpus_normalization.json"
    assert args.init_checkpoint is None
    adapted = parse_args(["--output", "RUNS/x", "--draws-per-image", "10",
                          "--init-checkpoint", "MODELS/tracer/released.pt"])
    assert adapted.draws_per_image == 10 and adapted.init_checkpoint.name == "released.pt"
    with pytest.raises(SystemExit):
        parse_args(["--output", "RUNS/x", "--cycles", "0"])


def test_shipped_normalization_matches_the_released_checkpoint():
    normalization = json.loads(
        (REPO / "config/recipes/noodle_tracer_fullcorpus_normalization.json").read_text(encoding="utf-8")
    )
    assert normalization["channels"] == 18
    assert len(normalization["mean"]) == len(normalization["std"]) == 18
    checkpoint = REPO / "MODELS/tracer/noodle_tracer_fullcorpus_seed20261011_calibrated.pt"
    if not checkpoint.is_file():
        pytest.skip("released tracer checkpoint not downloaded")
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    assert torch.equal(state["dense_mean"].flatten(), torch.tensor(normalization["mean"], dtype=torch.float32))
    assert torch.equal(state["dense_std"].flatten(), torch.tensor(normalization["std"], dtype=torch.float32))
