# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from deepstarnoodles.fused_tracker import (
    FusedTracker,
    TemporalTrackerConfig,
    reconcile_forward_reverse,
)


def _poly(tip_x: float, branch_y: float = 0.0) -> np.ndarray:
    """Tip-first test polyline with a shape-disambiguating base branch."""
    return np.asarray(
        [[tip_x, 0.0], [tip_x - 2.0, branch_y], [tip_x - 5.0, branch_y]],
        dtype=np.float32,
    )


def _batch(*polys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.stack(polys) if polys else np.empty((0, 3, 2), dtype=np.float32)
    lengths = np.full(len(array), 5.0, dtype=np.float32)
    embeds = np.zeros((len(array), 2), dtype=np.float32)
    return array, lengths, embeds


def _step(
    tracker: FusedTracker,
    frame: int,
    *polys: np.ndarray,
    qualities: list[float] | None = None,
    observable: bool = True,
    timestamp: float | None = None,
) -> list[int]:
    array, lengths, embeds = _batch(*polys)
    return tracker.step(
        frame,
        array,
        lengths,
        embeds,
        qualities=None if qualities is None else np.asarray(qualities),
        observable=observable,
        timestamp=timestamp,
    )


def test_legacy_path_preserves_exact_labels_and_retirement() -> None:
    tracker = FusedTracker(shape_gate=3.0, max_age=1)

    assert _step(tracker, 0, _poly(0), _poly(20)) == [0, 1]
    assert _step(tracker, 1, _poly(20.5), _poly(0.5)) == [1, 0]
    assert _step(tracker, 2) == []
    assert _step(tracker, 3) == []
    assert _step(tracker, 4, _poly(0.5)) == [2]
    assert tracker.detection_diagnostics()[0]["stage"] == "legacy"


def test_tentative_promotion_and_rejection() -> None:
    config = TemporalTrackerConfig(confirmation_hits=2, tentative_max_misses=0)
    promoted = FusedTracker(shape_gate=4.0, temporal=config)

    assert _step(promoted, 0, _poly(0)) == [0]
    assert promoted.detection_diagnostics()[0]["visible"] is False
    assert _step(promoted, 1, _poly(0.5)) == [0]
    assert promoted.detection_diagnostics()[0]["visible"] is True
    assert promoted.track_diagnostics()[0]["state"] == "confirmed"

    rejected = FusedTracker(temporal=config)
    assert _step(rejected, 0, _poly(0)) == [0]
    assert _step(rejected, 1) == []
    assert rejected.track_diagnostics()[0]["state"] == "retired"
    assert _step(rejected, 2, _poly(0)) == [1]


def test_low_confidence_recovers_but_cannot_birth() -> None:
    tracker = FusedTracker(
        shape_gate=4.0,
        temporal=TemporalTrackerConfig(
            confirmation_hits=1, high_quality=0.7, low_quality=0.2
        ),
    )

    assert _step(tracker, 0, _poly(0), qualities=[0.4]) == [-1]
    assert _step(tracker, 1, _poly(0), qualities=[0.9]) == [0]
    assert _step(tracker, 2, _poly(0.4), qualities=[0.3]) == [0]
    assert tracker.detection_diagnostics()[0]["stage"] == "low_recovery"


def test_lost_track_recovers_same_id() -> None:
    tracker = FusedTracker(
        shape_gate=5.0,
        temporal=TemporalTrackerConfig(
            confirmation_hits=1, lost_after_misses=1, retire_after_misses=3
        ),
    )
    assert _step(tracker, 0, _poly(0)) == [0]
    _step(tracker, 1)
    assert tracker.track_diagnostics()[0]["state"] == "lost"
    assert _step(tracker, 2, _poly(1)) == [0]
    assert tracker.track_diagnostics()[0]["state"] == "confirmed"


@pytest.mark.parametrize("mode", ["hysteresis", "log_odds", "hmm"])
def test_detector_dropout_is_not_a_miss_when_unobservable(mode: str) -> None:
    tracker = FusedTracker(
        temporal=TemporalTrackerConfig(
            confirmation_hits=1,
            existence_mode=mode,
            retire_after_misses=1,
        )
    )
    assert _step(tracker, 0, _poly(0)) == [0]
    before = tracker.track_diagnostics()[0]["existence"]
    _step(tracker, 5, observable=False)
    after = tracker.track_diagnostics()[0]
    assert after["misses"] == 0
    assert after["existence"] == before
    assert _step(tracker, 6, _poly(0.5)) == [0]


def test_shape_keeps_identity_during_tip_crossing() -> None:
    tracker = FusedTracker(
        shape_gate=8.0,
        temporal=TemporalTrackerConfig(confirmation_hits=1),
    )
    upper = _poly(-2, 6)
    lower = _poly(2, -6)
    assert _step(tracker, 0, upper, lower) == [0, 1]

    # Detection order is swapped and tips cross; branch shape carries identity.
    crossed_lower = _poly(-1, -6)
    crossed_upper = _poly(1, 6)
    assert _step(tracker, 1, crossed_lower, crossed_upper) == [1, 0]


def test_irregular_timestamps_drive_state_prediction_deterministically() -> None:
    config = TemporalTrackerConfig(
        confirmation_hits=1,
        motion_mode="state",
        mahalanobis_gate=10.0,
    )

    def run() -> tuple[list[list[int]], np.ndarray]:
        tracker = FusedTracker(shape_gate=20.0, vel_alpha=1.0, temporal=config)
        labels = [
            _step(tracker, 0, _poly(0), timestamp=0.0),
            _step(tracker, 1, _poly(2), timestamp=2.0),
            _step(tracker, 9, _poly(5), timestamp=5.0),
        ]
        return labels, tracker.predict_gaps(12, timestamp=7.0)[0]["poly"]

    labels_a, prediction_a = run()
    labels_b, prediction_b = run()
    assert labels_a == labels_b == [[0], [0], [0]]
    np.testing.assert_array_equal(prediction_a, prediction_b)
    assert prediction_a[0, 0] > 5.0


def test_gap_prediction_never_overwrites_observed_geometry() -> None:
    tracker = FusedTracker(
        shape_gate=10.0,
        vel_alpha=1.0,
        temporal=TemporalTrackerConfig(confirmation_hits=1, motion_mode="ema"),
    )
    _step(tracker, 0, _poly(0))
    observed = _poly(2)
    _step(tracker, 1, observed)
    stored_before = tracker.tracks[0]["poly"].copy()

    product = tracker.predict_gaps(4)[0]
    product["poly"][:] = 999

    np.testing.assert_array_equal(tracker.tracks[0]["poly"], stored_before)
    np.testing.assert_array_equal(stored_before, observed)
    assert product["source"] == "prediction"


def test_forward_reverse_reconciliation_is_deterministic() -> None:
    forward = [[0, 1], [0, 1], [0]]
    reverse = [[8, 9], [8, 9], [8]]
    assert reconcile_forward_reverse(forward, reverse) == {0: 8, 1: 9}
