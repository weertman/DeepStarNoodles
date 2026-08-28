# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics.trackers.basetrack import BaseTrack

from deepstarnoodles.crop import CropConfig, StarCropper, TrackedStarCropper


class _FakeBoxes:
    def __init__(
        self,
        xyxy: list[list[float]],
        classes: list[int],
        confidences: list[float],
        track_ids: list[int] | None = None,
    ) -> None:
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
        self.cls = torch.tensor(classes, dtype=torch.float32)
        self.conf = torch.tensor(confidences, dtype=torch.float32)
        self.id = (
            None
            if track_ids is None
            else torch.tensor(track_ids, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _FakeTracker:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _CounterTracker(_FakeTracker):
    def reset(self) -> None:
        super().reset()
        BaseTrack.reset_id()


class _FakeModel:
    def __init__(
        self,
        results: list[SimpleNamespace],
        *,
        names: dict[int, str] | list[str] | None = None,
    ) -> None:
        self.names = names or {
            0: "Other_species",
            1: "Prey",
            2: "Pycnopodia_helianthoides",
        }
        self._results = iter(results)
        self.calls: list[tuple[str, dict]] = []
        self.tracker = _FakeTracker()
        self.predictor = SimpleNamespace(trackers=[self.tracker])

    def track(self, _frame: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(("track", dict(kwargs)))
        return [next(self._results)]

    def predict(self, _frame: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(("predict", dict(kwargs)))
        return [next(self._results)]


class _CounterModel:
    """Mimic Ultralytics' process-global ID behavior without running BoT-SORT."""

    names = {2: "Pycnopodia_helianthoides"}

    def __init__(self) -> None:
        self.initialized = False
        self.tracker = _CounterTracker()
        self.predictor = SimpleNamespace(trackers=[self.tracker])

    def track(self, _frame: np.ndarray, **_kwargs: object) -> list[SimpleNamespace]:
        if not self.initialized:
            BaseTrack.reset_id()
            self.initialized = True
        track_id = BaseTrack.next_id()
        return [
            _result(
                [[10, 10, 30, 30]],
                [2],
                [0.9],
                track_ids=[track_id],
            )
        ]


def _result(
    xyxy: list[list[float]],
    classes: list[int],
    confidences: list[float],
    *,
    track_ids: list[int] | None,
    polygons: list[np.ndarray] | None = None,
) -> SimpleNamespace:
    masks = None if polygons is None else SimpleNamespace(xy=polygons)
    return SimpleNamespace(
        boxes=_FakeBoxes(xyxy, classes, confidences, track_ids),
        masks=masks,
    )


def _tracker_config(tmp_path: Path) -> Path:
    path = tmp_path / "botsort.yaml"
    path.write_text("tracker_type: botsort\n")
    return path


def test_tracked_cropper_filters_classes_preserves_masks_and_sorts_ids(
    tmp_path: Path,
) -> None:
    polygons = [
        np.asarray([[10, 10], [30, 30]], dtype=np.float32),
        np.asarray([[35, 35], [45, 45]], dtype=np.float32),
        np.asarray([[60, 20], [90, 50]], dtype=np.float32),
    ]
    model = _FakeModel(
        [
            _result(
                [[10, 10, 30, 30], [35, 35, 45, 45], [60, 20, 90, 50]],
                [2, 0, 2],
                [0.8, 0.99, 0.7],
                track_ids=[9, 4, 2],
                polygons=polygons,
            )
        ]
    )
    cropper = TrackedStarCropper(
        config=CropConfig(context_scale=1.0, device="cpu"),
        tracker_config=_tracker_config(tmp_path),
        model=model,
    )
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[:] = (1, 2, 3)

    crops = cropper.track_crops(frame)

    assert [crop.star_track_id for crop in crops] == [2, 9]
    assert np.array_equal(crops[0].mask_polygon, polygons[2])
    assert np.array_equal(crops[1].mask_polygon, polygons[0])
    assert tuple(crops[0].crop_rgb[0, 0]) == (3, 2, 1)
    assert crops[0].crop_side == 30.0
    method, kwargs = model.calls[0]
    assert method == "track"
    assert kwargs["persist"] is True
    assert kwargs["batch"] == 1
    assert kwargs["classes"] == [2]
    assert kwargs["tracker"] == str(cropper.tracker_config)


def test_tracked_cropper_keeps_ids_across_order_changes_and_misses(
    tmp_path: Path,
) -> None:
    model = _FakeModel(
        [
            _result(
                [[60, 10, 80, 30], [10, 10, 30, 30]],
                [2, 2],
                [0.8, 0.9],
                track_ids=[7, 3],
            ),
            _result([], [], [], track_ids=[]),
            _result(
                [[12, 10, 32, 30], [58, 10, 78, 30]],
                [2, 2],
                [0.85, 0.75],
                track_ids=[3, 7],
            ),
        ]
    )
    cropper = TrackedStarCropper(
        config=CropConfig(device="cpu"),
        tracker_config=_tracker_config(tmp_path),
        model=model,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    first = cropper.track_crops(frame)
    missing = cropper.track_crops(frame)
    third = cropper.track_crops(frame)

    assert [crop.star_track_id for crop in first] == [3, 7]
    assert missing == []
    assert [crop.star_track_id for crop in third] == [3, 7]


def test_tracked_cropper_skips_unconfirmed_boxes_and_resets_only_its_model(
    tmp_path: Path,
) -> None:
    config = _tracker_config(tmp_path)
    first_model = _FakeModel(
        [
            _result(
                [[10, 10, 30, 30]],
                [2],
                [0.9],
                track_ids=None,
            )
        ]
    )
    second_model = _FakeModel([])
    first = TrackedStarCropper(
        config=CropConfig(device="cpu"),
        tracker_config=config,
        model=first_model,
    )
    second = TrackedStarCropper(
        config=CropConfig(device="cpu"),
        tracker_config=config,
        model=second_model,
    )

    assert first.track_crops(np.zeros((50, 50, 3), dtype=np.uint8)) == []
    first.reset()

    assert first_model.tracker.reset_calls == 1
    assert second_model.tracker.reset_calls == 0
    second.reset()
    assert second_model.tracker.reset_calls == 1


def test_tracked_croppers_isolate_ultralytics_global_id_counters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(BaseTrack, "_count", 99)
    config = _tracker_config(tmp_path)
    first = TrackedStarCropper(
        config=CropConfig(device="cpu"),
        tracker_config=config,
        model=_CounterModel(),
    )
    second = TrackedStarCropper(
        config=CropConfig(device="cpu"),
        tracker_config=config,
        model=_CounterModel(),
    )
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    assert first.track_crops(frame)[0].star_track_id == 1
    assert BaseTrack._count == 99
    assert second.track_crops(frame)[0].star_track_id == 1
    assert first.track_crops(frame)[0].star_track_id == 2
    assert second.track_crops(frame)[0].star_track_id == 2
    first.reset()
    assert first.track_crops(frame)[0].star_track_id == 1
    assert second.track_crops(frame)[0].star_track_id == 3
    assert BaseTrack._count == 99


def test_single_star_cropper_keeps_legacy_untracked_selection() -> None:
    model = _FakeModel(
        [
            _result(
                [[5, 5, 15, 15], [20, 20, 60, 60]],
                [2, 2],
                [0.99, 0.6],
                track_ids=None,
            )
        ],
        names=["Other_species", "Prey", "sunflower_star"],
    )
    cropper = StarCropper(
        config=CropConfig(context_scale=1.0, device="cpu"),
        model=model,
    )

    crop = cropper.crop(np.zeros((80, 80, 3), dtype=np.uint8))

    assert crop is not None
    assert crop.bbox_xyxy == (20.0, 20.0, 60.0, 60.0)
    assert crop.star_track_id is None
    assert model.calls[0][0] == "predict"
