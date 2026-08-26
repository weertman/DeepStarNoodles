# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from deepstarnoodles.paths import resolve_repo_path, to_repo_relative


def test_empty_path_stays_empty() -> None:
    assert to_repo_relative("") == ""
    with pytest.raises(ValueError, match="empty path"):
        resolve_repo_path("")


def test_already_relative_project_path_is_unchanged() -> None:
    stored = "DATA/deepstarnoodles/images/star.jpg"
    assert to_repo_relative(stored) == stored


def test_foreign_deepstarnoodles_prefix_is_stripped(tmp_path: Path) -> None:
    foreign = "/home/other/DeepStarNoodles/DATA/deepstarnoodles/shapes/image_000001.json"
    assert (
        to_repo_relative(foreign, root=tmp_path)
        == "DATA/deepstarnoodles/shapes/image_000001.json"
    )


def test_starseg_three_class_prefix_maps_into_data(tmp_path: Path) -> None:
    foreign = "/home/other/StarSeg/yolo26_three_class_train_dataset/images/train/a.jpg"
    assert to_repo_relative(foreign, root=tmp_path) == "DATA/starseg/images/train/a.jpg"
    assert (
        to_repo_relative("/home/other/StarSeg/yolo26_three_class_train_dataset", root=tmp_path)
        == "DATA/starseg"
    )


def test_cvat_export_prefix_maps_to_dense_images(tmp_path: Path) -> None:
    foreign = (
        "/home/other/DeepNoodleTracker/DATA/DOWNLOAD/sunflower_star/"
        "latest_cvat_export/images/default/frame_01.jpg"
    )
    assert (
        to_repo_relative(foreign, root=tmp_path)
        == "DATA/deepstarnoodles/images/frame_01.jpg"
    )


def test_resolve_repo_path_joins_the_checkout(tmp_path: Path) -> None:
    image = tmp_path / "DATA" / "deepstarnoodles" / "images" / "star.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    assert (
        resolve_repo_path("DATA/deepstarnoodles/images/star.jpg", root=tmp_path)
        == image.resolve()
    )
    assert (
        resolve_repo_path(
            "/machine/DeepStarNoodles/DATA/deepstarnoodles/images/star.jpg",
            root=tmp_path,
        )
        == image.resolve()
    )
