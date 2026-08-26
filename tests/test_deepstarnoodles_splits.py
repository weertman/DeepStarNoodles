# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
import json
from pathlib import Path

from deepstarnoodles.paths import (
    DEFAULT_DENSE_INDEX,
    DEFAULT_TEST_INDEX,
    DEFAULT_TRAIN_INDEX,
    DEFAULT_VALIDATION_INDEX,
)
from deepstarnoodles.train import split_indices


def test_legacy_training_split_is_deterministic_and_disjoint() -> None:
    first_train, first_validation = split_indices(100, seed=7)
    second_train, second_validation = split_indices(100, seed=7)
    assert (first_train, first_validation) == (second_train, second_validation)
    assert len(first_train) == 80
    assert len(first_validation) == 20
    assert set(first_train).isdisjoint(first_validation)
    assert set(first_train) | set(first_validation) == set(range(100))


def test_grouped_splits_exist_and_are_disjoint() -> None:
    if not DEFAULT_TRAIN_INDEX.exists():
        import pytest
        pytest.skip("local DeepStarNoodles splits are not present")
    counts = {}
    paths = {
        "train": DEFAULT_TRAIN_INDEX,
        "validation": DEFAULT_VALIDATION_INDEX,
        "test": DEFAULT_TEST_INDEX,
    }
    names: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    for split, path in paths.items():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        counts[split] = len(rows)
        names[split] = {row["image_name"] for row in rows}
        for row in rows:
            groups.setdefault(row["split_group_id"], set()).add(split)
    assert counts == {"train": 4062, "validation": 478, "test": 502}
    assert names["train"].isdisjoint(names["validation"])
    assert names["train"].isdisjoint(names["test"])
    assert names["validation"].isdisjoint(names["test"])
    assert all(len(splits) == 1 for splits in groups.values())


def test_dense_index_points_at_local_files() -> None:
    if not DEFAULT_DENSE_INDEX.exists():
        import pytest
        pytest.skip("local DeepStarNoodles index is not present")
    with DEFAULT_DENSE_INDEX.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert Path(row["image_path"]).exists()
    assert Path(row["shapes_path"]).exists()


def test_metadata_summary_matches_split_counts() -> None:
    summary_path = DEFAULT_DENSE_INDEX.parent / "metadata_summary.json"
    if not summary_path.exists():
        import pytest
        pytest.skip("local metadata summary is not present")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["split_counts"]["train"] == 4062
    assert summary["split_counts"]["validation"] == 478
    assert summary["split_counts"]["test"] == 502
