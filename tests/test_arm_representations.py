# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from deepstarnoodles.representations.arm_bag import (
    ArmBagConfig,
    arm_bag,
    arm_bag_sequence,
    arm_feature_names,
    pool_arm_states,
)
from deepstarnoodles.representations.arm_motion import (
    ArmMotionConfig,
    TrackedArmFrame,
    TrackedArmObservation,
    arm_motion,
)


def _static_arms() -> list[np.ndarray]:
    return [
        np.asarray([[0.0, 0.0], [2.0, 0.3], [5.0, 1.0]]),
        np.asarray([[0.0, 2.0], [1.5, 2.8], [4.0, 4.0]]),
        np.asarray([[1.0, -1.0], [2.0, -2.0], [3.0, -4.0]]),
    ]


def _motion_arm(
    track_id: int,
    extension: float,
    *,
    identity: np.ndarray | None,
) -> dict[str, object]:
    direction = 1.0 if track_id == 0 else -1.0
    base = np.asarray([-1.0 if track_id == 0 else 1.0, 0.0])
    polyline = np.asarray(
        [
            base,
            [base[0] + direction * 4.0, 0.4 * direction],
            [base[0] + direction * (10.0 + extension), 0.0],
        ],
        dtype=np.float64,
    )
    record: dict[str, object] = {
        "track_id": track_id,
        "polyline_source": polyline,
    }
    if identity is not None:
        record["identity_embedding"] = identity
    return record


def _motion_frame(
    timestamp: float,
    extension: float,
    *,
    identities: tuple[np.ndarray | None, np.ndarray | None] = (
        np.zeros(2),
        np.zeros(2),
    ),
    tracks: tuple[int, ...] = (0, 1),
    observed: bool = True,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "observed": observed,
        "arms": [
            _motion_arm(
                track_id,
                extension,
                identity=identities[track_id],
            )
            for track_id in tracks
        ],
    }


def _motion_config(**overrides: object) -> ArmMotionConfig:
    values: dict[str, object] = {
        "lags_seconds": (1,),
        "pair_tolerance_seconds": 0.01,
        "min_track_observations": 2,
        "min_matched_arms": 2,
    }
    values.update(overrides)
    return ArmMotionConfig(**values)


def test_default_arm_bag_is_102d_float32_and_immutable() -> None:
    result = arm_bag(_static_arms())

    assert result.valid
    assert result.arm_count == 3
    assert result.values.shape == (102,)
    assert result.values.dtype == np.float32
    assert result.feature_names == arm_feature_names(ArmBagConfig())
    assert len(result.feature_names) == 4 * 24 + 6
    assert not result.values.flags.writeable
    with pytest.raises(ValueError):
        result.values[0] = 12.0
    with pytest.raises(FrozenInstanceError):
        result.valid = False  # type: ignore[misc]


def test_arm_bag_config_controls_dimension_and_recipe_digest() -> None:
    small = ArmBagConfig(samples=8)
    again = ArmBagConfig(samples=8)
    large = ArmBagConfig(samples=12)

    first = arm_bag(_static_arms(), small)
    second = arm_bag(_static_arms(), again)
    changed = arm_bag(_static_arms(), large)

    assert first.values.shape == (4 * 8 + 6,)
    assert arm_feature_names(8) == arm_feature_names(samples=8)
    assert first.recipe_digest == second.recipe_digest
    assert first.recipe_digest != changed.recipe_digest
    assert first.config_metadata["samples"] == 8
    with pytest.raises(TypeError):
        first.config_metadata["samples"] = 4  # type: ignore[index]


def test_arm_bag_is_permutation_and_rigid_transform_invariant() -> None:
    arms = _static_arms()
    baseline = arm_bag(arms)
    permuted = arm_bag([arms[2], arms[0], arms[1]])

    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    translation = np.asarray([17.0, -9.0])
    transformed = arm_bag([arm @ rotation.T + translation for arm in arms])

    np.testing.assert_allclose(permuted.values, baseline.values, atol=1e-7)
    np.testing.assert_allclose(transformed.values, baseline.values, atol=2e-7)


def test_weighted_pooling_uses_population_mean_and_deviation() -> None:
    states = np.asarray([[1.0, 3.0], [5.0, 7.0]])
    pooled = pool_arm_states(states, weights=np.asarray([1.0, 3.0]))

    np.testing.assert_allclose(pooled[:2], [4.0, 6.0])
    np.testing.assert_allclose(pooled[2:], np.sqrt([3.0, 3.0]))


def test_native_arm_mapping_adapter_and_user_scales() -> None:
    arms = _static_arms()
    records = [
        {"polyline": arms[0]},
        {"polyline_crop": arms[1]},
        {"polyline_source": arms[2]},
    ]
    from_records = arm_bag(
        records,
        length_scale=5.0,
        radius_scale=5.0,
    )
    from_arrays = arm_bag(
        arms,
        length_scale=5.0,
        radius_scale=5.0,
    )

    np.testing.assert_array_equal(from_records.values, from_arrays.values)
    provided = arm_bag(
        arms,
        ArmBagConfig(scale_policy="provided"),
        length_scale=5.0,
        radius_scale=5.0,
    )
    assert provided.valid
    with pytest.raises(ValueError):
        arm_bag(arms, ArmBagConfig(scale_policy="provided"), length_scale=5.0)


def test_empty_and_degenerate_arm_frames_are_invalid_zeros() -> None:
    degenerate = np.repeat(np.asarray([[2.0, 3.0]]), 4, axis=0)
    empty, collapsed = arm_bag_sequence([[], [degenerate]])

    assert not empty.valid
    assert empty.arm_count == 0
    assert not collapsed.valid
    assert collapsed.arm_count == 1
    np.testing.assert_array_equal(empty.values, np.zeros(102, dtype=np.float32))
    np.testing.assert_array_equal(
        collapsed.values, np.zeros(102, dtype=np.float32)
    )


def test_motion_lag_pairing_honors_tolerance() -> None:
    frames = [_motion_frame(0.0, 0.0), _motion_frame(1.4, 2.0)]
    narrow = arm_motion(
        frames,
        _motion_config(pair_tolerance_seconds=0.39),
    )
    wide = arm_motion(
        frames,
        _motion_config(pair_tolerance_seconds=0.41),
    )

    assert not narrow.valid.any()
    assert wide.valid[1, 0]
    assert wide.matched_counts[1, 0] == 2
    assert wide.values.shape == (2, 1, 5)


def test_motion_never_bridges_a_same_id_continuity_gap() -> None:
    frames = [
        _motion_frame(0.0, 0.0),
        _motion_frame(1.0, 1.0, tracks=()),
        _motion_frame(2.0, 2.0),
    ]
    result = arm_motion(
        frames,
        _motion_config(lags_seconds=(2,), pair_tolerance_seconds=0.01),
    )

    assert not result.valid[2, 0]
    assert result.matched_counts[2, 0] == 0
    assert np.isnan(result.values[2, 0]).all()


def test_motion_strict_identity_gate_reports_retained_tracks() -> None:
    frames = [
        _motion_frame(0.0, 0.0),
        _motion_frame(
            1.0,
            2.0,
            identities=(np.asarray([0.5, 0.0]), np.asarray([2.0, 0.0])),
        ),
    ]
    result = arm_motion(frames, _motion_config(identity_gate=1.0))

    assert result.valid[1, 0]
    assert not result.strict_valid[1, 0]
    assert result.strict_matched_counts[1, 0] == 1
    assert result.retained_fraction[1, 0] == pytest.approx(0.5)
    assert result.identity_distance[1, 0] == pytest.approx(1.25)


def test_motion_missing_identity_never_fabricates_strict_validity() -> None:
    frames = [
        _motion_frame(0.0, 0.0, identities=(None, None)),
        _motion_frame(1.0, 2.0, identities=(None, None)),
    ]
    result = arm_motion(frames, _motion_config())

    assert result.valid[1, 0]
    assert not result.strict_valid[1, 0]
    assert result.strict_matched_counts[1, 0] == 0
    assert np.isnan(result.identity_distance[1, 0])


def test_motion_provided_scale_controls_displacement_units() -> None:
    frames = [_motion_frame(0.0, 0.0), _motion_frame(1.0, 2.0)]
    config = _motion_config(scale_policy="provided")
    scale_ten = arm_motion(frames, config, scale=10.0)
    scale_twenty = arm_motion(frames, config, scale=20.0)

    assert scale_ten.sequence_scale == 10.0
    np.testing.assert_allclose(
        scale_twenty.values[1, 0, :3],
        scale_ten.values[1, 0, :3] / 2.0,
    )
    np.testing.assert_allclose(
        scale_twenty.values[1, 0, 3:],
        scale_ten.values[1, 0, 3:],
    )
    with pytest.raises(ValueError):
        arm_motion(frames, config)


def test_motion_distal_points_are_final_base_to_tip_samples() -> None:
    frames = [_motion_frame(0.0, 0.0), _motion_frame(1.0, 2.0)]
    tip_only = arm_motion(
        frames,
        _motion_config(
            scale_policy="provided",
            resample_points=8,
            distal_points=1,
        ),
        scale=10.0,
    )
    distal_region = arm_motion(
        frames,
        _motion_config(
            scale_policy="provided",
            resample_points=8,
            distal_points=4,
        ),
        scale=10.0,
    )

    assert tip_only.values[1, 0, 0] > distal_region.values[1, 0, 0]
    assert tip_only.recipe_digest != distal_region.recipe_digest


def test_typed_motion_records_and_quality_defaults_remain_usable() -> None:
    typed_frames = [
        TrackedArmFrame(
            timestamp=float(frame),
            observed=True,
            arms=tuple(
                TrackedArmObservation(
                    track_id=track_id,
                    polyline=_motion_arm(
                        track_id,
                        float(frame),
                        identity=np.zeros(2),
                    )["polyline_source"],
                    identity=np.zeros(2),
                )
                for track_id in (0, 1)
            ),
        )
        for frame in range(2)
    ]
    result = arm_motion(typed_frames, _motion_config())

    assert result.valid[1, 0]
    assert result.strict_valid[1, 0]


def test_default_motion_metadata_marks_offline_sequence_scale() -> None:
    result = arm_motion([], ArmMotionConfig())

    assert result.values.shape == (0, 3, 5)
    assert result.recipe_metadata["causal_lag_comparisons"] is True
    assert result.recipe_metadata["normalization"]["offline"] is True
    assert result.recipe_metadata["normalization_is_offline"] is True
    assert result.recipe_metadata["distal_selection"] == "final_resampled_points"
    assert np.isnan(result.sequence_scale)
    assert not result.values.flags.writeable


def test_motion_requires_chronological_unique_timestamps() -> None:
    frames = [_motion_frame(1.0, 0.0), _motion_frame(1.0, 1.0)]
    with pytest.raises(ValueError, match="chronological and unique"):
        arm_motion(frames, _motion_config())
