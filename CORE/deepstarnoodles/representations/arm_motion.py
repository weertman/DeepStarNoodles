# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Dense tracked arm-motion features without tabular or storage dependencies.

Lag pairs always compare the current frame with an earlier frame.  The default
``sequence_median`` length normalization is nevertheless an offline statistic:
it is computed from quality arms across the complete supplied sequence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ARM_MOTION_REPRESENTATION = "deepstarnoodles_tracked_arm_motion@1"
MOTION_REPRESENTATION = ARM_MOTION_REPRESENTATION
MOTION_FEATURE_NAMES = (
    "distal_mean",
    "distal_q90",
    "distal_max",
    "participation_fraction",
    "top2_share",
)
_POLYLINE_KEYS = ("polyline", "polyline_crop", "polyline_source")
_SCALE_POLICIES = frozenset(("sequence_median", "provided"))
_MISSING = object()

__all__ = [
    "ARM_MOTION_REPRESENTATION",
    "MOTION_FEATURE_NAMES",
    "MOTION_REPRESENTATION",
    "ArmMotionConfig",
    "ArmMotionResult",
    "TrackedArmFrame",
    "TrackedArmObservation",
    "arm_motion",
    "arm_motion_recipe",
    "tracked_arm_frame",
]


class _FrozenDict(dict[str, Any]):
    """A JSON-serializable dictionary with ordinary mutation disabled."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _real(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    return result


def _finite_float(name: str, value: Any, *, minimum: float | None = None) -> float:
    result = _real(name, value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _positive_scale(name: str, value: Any, epsilon: float) -> float:
    result = _finite_float(name, value)
    if result <= epsilon:
        raise ValueError(f"{name} must be greater than epsilon")
    return result


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict(
            {str(key): _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _canonical_digest(metadata: Mapping[str, Any]) -> str:
    payload = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any],
    ndim: int,
    name: str,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    result.setflags(write=False)
    return result


def _point(value: Any, *, name: str) -> np.ndarray:
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite (2,) point")
    return result


def _polyline(value: Any) -> np.ndarray:
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("polyline coordinates must be numeric") from exc
    if result.ndim != 2 or result.shape[1:] != (2,) or len(result) < 2:
        raise ValueError("polyline must have shape [point, 2] with at least two points")
    if not np.isfinite(result).all():
        raise ValueError("polyline coordinates must be finite")
    return result


def _optional_real(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _real(name, value)


@dataclass(frozen=True)
class ArmMotionConfig:
    """Recipe and quality controls for tracked distal motion."""

    lags_seconds: tuple[float, ...] = (1, 3, 5)
    distal_points: int = 6
    resample_points: int = 24
    pair_tolerance_seconds: float = 0.55
    min_track_observations: int = 10
    min_matched_arms: int = 2
    identity_gate: float = 1.0
    detection_quality_min: float = 0.6
    tracker_existence_min: float = 0.6
    scale_policy: str = "sequence_median"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if isinstance(self.lags_seconds, (str, bytes)):
            raise TypeError("lags_seconds must be a sequence of positive numbers")
        try:
            raw_lags = tuple(self.lags_seconds)
        except TypeError as exc:
            raise TypeError(
                "lags_seconds must be a sequence of positive numbers"
            ) from exc
        if not raw_lags:
            raise ValueError("lags_seconds must not be empty")
        lags = tuple(
            sorted(
                {
                    _finite_float("lag", lag, minimum=0.0)
                    for lag in raw_lags
                }
            )
        )
        if not lags or lags[0] <= 0.0:
            raise ValueError("lags_seconds must contain positive values")
        object.__setattr__(self, "lags_seconds", lags)

        resample_points = _integer(
            "resample_points", self.resample_points, minimum=2
        )
        distal_points = _integer("distal_points", self.distal_points, minimum=1)
        if distal_points > resample_points:
            raise ValueError("distal_points must not exceed resample_points")
        object.__setattr__(self, "resample_points", resample_points)
        object.__setattr__(self, "distal_points", distal_points)
        object.__setattr__(
            self,
            "pair_tolerance_seconds",
            _finite_float(
                "pair_tolerance_seconds",
                self.pair_tolerance_seconds,
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "min_track_observations",
            _integer(
                "min_track_observations",
                self.min_track_observations,
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "min_matched_arms",
            _integer("min_matched_arms", self.min_matched_arms, minimum=1),
        )
        object.__setattr__(
            self,
            "identity_gate",
            _finite_float("identity_gate", self.identity_gate, minimum=0.0),
        )
        for name in ("detection_quality_min", "tracker_existence_min"):
            threshold = _finite_float(name, getattr(self, name), minimum=0.0)
            if threshold > 1.0:
                raise ValueError(f"{name} must be <= 1")
            object.__setattr__(self, name, threshold)
        if not isinstance(self.scale_policy, str):
            raise TypeError("scale_policy must be a string")
        if self.scale_policy not in _SCALE_POLICIES:
            choices = ", ".join(sorted(_SCALE_POLICIES))
            raise ValueError(f"scale_policy must be one of: {choices}")
        object.__setattr__(
            self,
            "epsilon",
            _finite_float("epsilon", self.epsilon, minimum=0.0),
        )
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class TrackedArmObservation:
    """One native base-to-tip arm observation and optional tracking evidence."""

    track_id: int
    polyline: np.ndarray
    identity: np.ndarray | None = None
    visibility: bool = True
    quality: float | None = None
    existence: float | None = None
    stop_reason: str | None = None
    match_state: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.track_id, (bool, np.bool_)) or not isinstance(
            self.track_id, (int, np.integer)
        ):
            raise TypeError("track_id must be an integer")
        if not isinstance(self.visibility, (bool, np.bool_)):
            raise TypeError("visibility must be boolean")
        points = _readonly_array(
            _polyline(self.polyline),
            dtype=np.float64,
            ndim=2,
            name="polyline",
        )
        identity: np.ndarray | None
        if self.identity is None:
            identity = None
        else:
            identity = _readonly_array(
                self.identity, dtype=np.float64, ndim=1, name="identity"
            )
            if not len(identity):
                raise ValueError("identity must not be empty")
        for name in ("quality", "existence"):
            object.__setattr__(
                self, name, _optional_real(name, getattr(self, name))
            )
        for name in ("stop_reason", "match_state"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")

        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "polyline", points)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "visibility", bool(self.visibility))


@dataclass(frozen=True)
class TrackedArmFrame:
    """A timestamped frame of tracked native arm observations."""

    timestamp: float
    observed: bool
    arms: tuple[TrackedArmObservation, ...]
    body_center: np.ndarray | None = None

    def __post_init__(self) -> None:
        timestamp = _finite_float("timestamp", self.timestamp)
        if not isinstance(self.observed, (bool, np.bool_)):
            raise TypeError("observed must be boolean")
        if isinstance(self.arms, (str, bytes)):
            raise TypeError("arms must be an iterable of TrackedArmObservation")
        try:
            arms = tuple(self.arms)
        except TypeError as exc:
            raise TypeError(
                "arms must be an iterable of TrackedArmObservation"
            ) from exc
        if not all(isinstance(arm, TrackedArmObservation) for arm in arms):
            raise TypeError("typed frames require TrackedArmObservation arms")
        center = (
            None
            if self.body_center is None
            else _readonly_array(
                _point(self.body_center, name="body_center"),
                dtype=np.float64,
                ndim=1,
                name="body_center",
            )
        )
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "observed", bool(self.observed))
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "body_center", center)


@dataclass(frozen=True)
class ArmMotionResult:
    """Immutable dense frame-by-lag motion tensors and diagnostics."""

    values: np.ndarray
    strict_values: np.ndarray
    valid: np.ndarray
    strict_valid: np.ndarray
    matched_counts: np.ndarray
    strict_matched_counts: np.ndarray
    identity_distance: np.ndarray
    retained_fraction: np.ndarray
    timestamps: np.ndarray
    lags_seconds: np.ndarray
    feature_names: tuple[str, ...]
    sequence_scale: float
    representation: str
    recipe_digest: str
    recipe_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = _readonly_array(
            self.values, dtype=np.float32, ndim=3, name="values"
        )
        strict_values = _readonly_array(
            self.strict_values,
            dtype=np.float32,
            ndim=3,
            name="strict_values",
        )
        valid = _readonly_array(self.valid, dtype=bool, ndim=2, name="valid")
        strict_valid = _readonly_array(
            self.strict_valid, dtype=bool, ndim=2, name="strict_valid"
        )
        matched = _readonly_array(
            self.matched_counts,
            dtype=np.int32,
            ndim=2,
            name="matched_counts",
        )
        strict_matched = _readonly_array(
            self.strict_matched_counts,
            dtype=np.int32,
            ndim=2,
            name="strict_matched_counts",
        )
        identity_distance = _readonly_array(
            self.identity_distance,
            dtype=np.float32,
            ndim=2,
            name="identity_distance",
        )
        retained = _readonly_array(
            self.retained_fraction,
            dtype=np.float32,
            ndim=2,
            name="retained_fraction",
        )
        timestamps = _readonly_array(
            self.timestamps, dtype=np.float64, ndim=1, name="timestamps"
        )
        lags = _readonly_array(
            self.lags_seconds, dtype=np.float64, ndim=1, name="lags_seconds"
        )
        names = tuple(str(name) for name in self.feature_names)
        expected = (len(timestamps), len(lags), len(names))
        if values.shape != expected or strict_values.shape != expected:
            raise ValueError("motion values must have shape [frame, lag, feature]")
        for name, array in (
            ("valid", valid),
            ("strict_valid", strict_valid),
            ("matched_counts", matched),
            ("strict_matched_counts", strict_matched),
            ("identity_distance", identity_distance),
            ("retained_fraction", retained),
        ):
            if array.shape != expected[:2]:
                raise ValueError(f"{name} must have shape [frame, lag]")
        if np.any(matched < 0) or np.any(strict_matched < 0):
            raise ValueError("matched counts must be non-negative")
        if not np.isfinite(timestamps).all() or not np.isfinite(lags).all():
            raise ValueError("timestamps and lags_seconds must be finite")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("timestamps must be strictly increasing")
        if len(lags) and np.any(lags <= 0.0):
            raise ValueError("lags_seconds must be positive")
        sequence_scale = _real("sequence_scale", self.sequence_scale)
        if not (
            np.isnan(sequence_scale)
            or (np.isfinite(sequence_scale) and sequence_scale > 0.0)
        ):
            raise ValueError("sequence_scale must be finite and positive or NaN")
        if not isinstance(self.representation, str) or not self.representation:
            raise ValueError("representation must be a non-empty string")
        if (
            not isinstance(self.recipe_digest, str)
            or len(self.recipe_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.recipe_digest)
        ):
            raise ValueError("recipe_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.recipe_metadata, Mapping):
            raise TypeError("recipe_metadata must be a mapping")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "strict_values", strict_values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "strict_valid", strict_valid)
        object.__setattr__(self, "matched_counts", matched)
        object.__setattr__(self, "strict_matched_counts", strict_matched)
        object.__setattr__(self, "identity_distance", identity_distance)
        object.__setattr__(self, "retained_fraction", retained)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "lags_seconds", lags)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "sequence_scale", sequence_scale)
        object.__setattr__(
            self, "recipe_metadata", _freeze_metadata(dict(self.recipe_metadata))
        )

    @property
    def representation_id(self) -> str:
        """Versioned representation identifier."""

        return self.representation

    @property
    def config_metadata(self) -> Mapping[str, Any]:
        """Immutable normalized configuration metadata."""

        return self.recipe_metadata["config"]

    @property
    def valid_mask(self) -> np.ndarray:
        return self.valid

    @property
    def strict_valid_mask(self) -> np.ndarray:
        return self.strict_valid

    @property
    def matched_arms(self) -> np.ndarray:
        return self.matched_counts

    @property
    def strict_matched_arms(self) -> np.ndarray:
        return self.strict_matched_counts

    @property
    def identity_distance_median(self) -> np.ndarray:
        return self.identity_distance

    @property
    def strict_retained_fraction(self) -> np.ndarray:
        return self.retained_fraction

    @property
    def lags(self) -> np.ndarray:
        return self.lags_seconds

    @property
    def scale(self) -> float:
        return self.sequence_scale


def _coerce_config(config: ArmMotionConfig) -> ArmMotionConfig:
    if not isinstance(config, ArmMotionConfig):
        raise TypeError("config must be an ArmMotionConfig")
    return config


def arm_motion_recipe(config: ArmMotionConfig = ArmMotionConfig()) -> dict[str, Any]:
    """Return the JSON-serializable deterministic motion recipe."""

    config = _coerce_config(config)
    offline_scale = config.scale_policy == "sequence_median"
    return {
        "representation": ARM_MOTION_REPRESENTATION,
        "config": {
            "lags_seconds": list(config.lags_seconds),
            "distal_points": config.distal_points,
            "resample_points": config.resample_points,
            "pair_tolerance_seconds": config.pair_tolerance_seconds,
            "min_track_observations": config.min_track_observations,
            "min_matched_arms": config.min_matched_arms,
            "identity_gate": config.identity_gate,
            "detection_quality_min": config.detection_quality_min,
            "tracker_existence_min": config.tracker_existence_min,
            "scale_policy": config.scale_policy,
            "epsilon": config.epsilon,
        },
        "feature_names": list(MOTION_FEATURE_NAMES),
        "polyline_order": "base_to_tip",
        "resampling": "uniform_arclength",
        "distal_selection": "final_resampled_points",
        "displacement": "body_relative_distal_point_rms",
        "lag_comparisons": "causal_previous_frame",
        "causal_lag_comparisons": True,
        "track_continuity": "same_id_present_every_intervening_frame",
        "track_observation_scope": "whole_sequence",
        "normalization": {
            "policy": config.scale_policy,
            "statistic": (
                "median_quality_arm_arclength"
                if offline_scale
                else "caller_provided"
            ),
            "scope": "whole_sequence" if offline_scale else "provided_constant",
            "offline": offline_scale,
        },
        "normalization_is_offline": offline_scale,
        "strict_identity": "finite_equal_shape_endpoint_identity_distance_gate",
    }


def _mapping_value(
    record: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: Any = _MISSING,
) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    if default is _MISSING:
        choices = ", ".join(repr(key) for key in keys)
        raise ValueError(f"record must contain one of {choices}")
    return default


def _observation(value: Any) -> TrackedArmObservation:
    if isinstance(value, TrackedArmObservation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("each arm must be a TrackedArmObservation or mapping")
    polyline = _mapping_value(value, _POLYLINE_KEYS)
    return TrackedArmObservation(
        track_id=_mapping_value(value, ("track_id", "id")),
        polyline=polyline,
        identity=_mapping_value(
            value,
            ("identity", "identity_embedding", "embedding"),
            default=None,
        ),
        visibility=_mapping_value(
            value,
            ("visibility", "visible", "detection_visible"),
            default=True,
        ),
        quality=_mapping_value(
            value,
            ("quality", "detection_quality", "decode_quality"),
            default=None,
        ),
        existence=_mapping_value(
            value,
            ("existence", "tracker_existence"),
            default=None,
        ),
        stop_reason=_mapping_value(value, ("stop_reason",), default=None),
        match_state=_mapping_value(value, ("match_state",), default=None),
    )


def _frame(value: Any) -> TrackedArmFrame:
    if isinstance(value, TrackedArmFrame):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("each frame must be a TrackedArmFrame or mapping")
    raw_arms = _mapping_value(value, ("arms",))
    if isinstance(raw_arms, (str, bytes)) or not isinstance(raw_arms, Iterable):
        raise TypeError("frame arms must be an iterable")
    arms = tuple(_observation(arm) for arm in raw_arms)
    return TrackedArmFrame(
        timestamp=_mapping_value(value, ("timestamp",)),
        observed=_mapping_value(value, ("observed", "frame_observed")),
        arms=arms,
        body_center=_mapping_value(
            value,
            ("body_center", "body_center_source", "center"),
            default=None,
        ),
    )


def tracked_arm_frame(value: TrackedArmFrame | Mapping[str, Any]) -> TrackedArmFrame:
    """Normalize a typed or mapping arm-frame record."""

    return _frame(value)


def _resample(
    polyline: np.ndarray, count: int, epsilon: float
) -> tuple[np.ndarray, float]:
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= epsilon:
        return np.repeat(polyline[:1], count, axis=0), 0.0
    keep = np.concatenate(([True], np.diff(cumulative) > epsilon))
    cumulative = cumulative[keep]
    unique = polyline[keep]
    targets = np.linspace(0.0, total, count)
    sampled = np.column_stack(
        [
            np.interp(targets, cumulative, unique[:, axis])
            for axis in range(polyline.shape[1])
        ]
    )
    return sampled, total


def _passes_quality(arm: TrackedArmObservation, config: ArmMotionConfig) -> bool:
    if arm.track_id < 0 or not arm.visibility:
        return False
    if arm.quality is not None and (
        not np.isfinite(arm.quality)
        or arm.quality < config.detection_quality_min
    ):
        return False
    if arm.existence is not None and (
        not np.isfinite(arm.existence)
        or arm.existence < config.tracker_existence_min
    ):
        return False
    stop_reason = "" if arm.stop_reason is None else arm.stop_reason.strip().lower()
    if stop_reason not in ("", "base"):
        return False
    match_state = "" if arm.match_state is None else arm.match_state.strip().lower()
    return match_state in ("", "confirmed")


@dataclass(frozen=True)
class _PreparedArm:
    distal: np.ndarray
    arclength: float
    identity: np.ndarray | None


def _prepare_frames(
    frames: Sequence[TrackedArmFrame],
    config: ArmMotionConfig,
) -> tuple[list[dict[int, _PreparedArm]], np.ndarray]:
    tracks: list[dict[int, _PreparedArm]] = []
    scale_lengths: list[float] = []
    for frame_index, frame in enumerate(frames):
        rows: dict[int, _PreparedArm] = {}
        if not frame.observed:
            tracks.append(rows)
            continue
        if frame.body_center is None:
            if not frame.arms:
                tracks.append(rows)
                continue
            center = np.mean(
                np.stack([arm.polyline[0] for arm in frame.arms]), axis=0
            )
        else:
            center = frame.body_center
        for arm in frame.arms:
            if not _passes_quality(arm, config):
                continue
            sampled, arclength = _resample(
                arm.polyline, config.resample_points, config.epsilon
            )
            if arclength <= config.epsilon:
                continue
            if arm.track_id in rows:
                raise ValueError(
                    f"frame {frame_index} contains duplicate usable track_id "
                    f"{arm.track_id}"
                )
            relative = sampled - center
            distal = np.array(
                relative[-config.distal_points :], dtype=np.float64, copy=True
            )
            distal.setflags(write=False)
            rows[arm.track_id] = _PreparedArm(
                distal=distal,
                arclength=arclength,
                identity=arm.identity,
            )
            scale_lengths.append(arclength)
        tracks.append(rows)
    return tracks, np.asarray(scale_lengths, dtype=np.float64)


def _nearest_previous_frame(
    timestamps: np.ndarray,
    current: int,
    lag_seconds: float,
    tolerance_seconds: float,
) -> int | None:
    target = float(timestamps[current]) - lag_seconds
    position = int(np.searchsorted(timestamps, target))
    candidates = [
        index for index in (position - 1, position) if 0 <= index < current
    ]
    if not candidates:
        return None
    previous = min(
        candidates,
        key=lambda index: abs(float(timestamps[index]) - target),
    )
    if abs(float(timestamps[previous]) - target) > tolerance_seconds:
        return None
    return previous


def _identity_distance(
    first: np.ndarray | None, second: np.ndarray | None
) -> float | None:
    if first is None or second is None or first.shape != second.shape:
        return None
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    distance = float(np.linalg.norm(second - first))
    return distance if np.isfinite(distance) else None


def _motion_features(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 1e-12:
        participation = 1.0
        top2_share = 0.0
    else:
        effective = total * total / max(
            float(np.square(values).sum()), 1e-12
        )
        participation = effective / len(values)
        top2_share = float(
            np.sort(values)[-min(2, len(values)) :].sum() / total
        )
    return np.asarray(
        (
            float(values.mean()),
            float(np.quantile(values, 0.9)),
            float(values.max()),
            participation,
            top2_share,
        ),
        dtype=np.float32,
    )


def _empty_arrays(
    frame_count: int, lag_count: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    shape = (frame_count, lag_count, len(MOTION_FEATURE_NAMES))
    grid = shape[:2]
    return (
        np.full(shape, np.nan, dtype=np.float32),
        np.full(shape, np.nan, dtype=np.float32),
        np.zeros(grid, dtype=bool),
        np.zeros(grid, dtype=bool),
        np.zeros(grid, dtype=np.int32),
        np.zeros(grid, dtype=np.int32),
        np.full(grid, np.nan, dtype=np.float32),
        np.full(grid, np.nan, dtype=np.float32),
    )


def _result(
    arrays: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    *,
    timestamps: np.ndarray,
    config: ArmMotionConfig,
    sequence_scale: float,
) -> ArmMotionResult:
    metadata = arm_motion_recipe(config)
    (
        values,
        strict_values,
        valid,
        strict_valid,
        matched,
        strict_matched,
        identity_distance,
        retained,
    ) = arrays
    return ArmMotionResult(
        values=values,
        strict_values=strict_values,
        valid=valid,
        strict_valid=strict_valid,
        matched_counts=matched,
        strict_matched_counts=strict_matched,
        identity_distance=identity_distance,
        retained_fraction=retained,
        timestamps=timestamps,
        lags_seconds=np.asarray(config.lags_seconds, dtype=np.float64),
        feature_names=MOTION_FEATURE_NAMES,
        sequence_scale=sequence_scale,
        representation=ARM_MOTION_REPRESENTATION,
        recipe_digest=_canonical_digest(metadata),
        recipe_metadata=metadata,
    )


def arm_motion(
    frames: Sequence[TrackedArmFrame | Mapping[str, Any]]
    | Iterable[TrackedArmFrame | Mapping[str, Any]],
    config: ArmMotionConfig = ArmMotionConfig(),
    *,
    scale: float | None = None,
) -> ArmMotionResult:
    """Compute dense tracked distal motion on a frame-by-lag grid.

    Geometry comparisons are causal: each value uses only a current frame and
    an earlier frame near the requested lag.  Same-ID observations must pass
    quality gates in every intervening frame.  Track-observation eligibility,
    and the default ``sequence_median`` normalization, are computed over the
    complete supplied sequence; the default normalization is therefore an
    offline statistic.

    Native polylines are base-to-tip.  They are arc-length resampled internally
    and the final ``distal_points`` samples are used, avoiding the tip-first
    assumption of legacy storage formats.
    """

    config = _coerce_config(config)
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Iterable):
        raise TypeError("frames must be an iterable")
    typed_frames = tuple(_frame(frame) for frame in frames)
    timestamps = np.asarray(
        [frame.timestamp for frame in typed_frames], dtype=np.float64
    )
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("frame timestamps must be chronological and unique")

    if config.scale_policy == "provided":
        if scale is None:
            raise ValueError("scale_policy='provided' requires scale")
        sequence_scale = _positive_scale("scale", scale, config.epsilon)
    else:
        if scale is not None:
            raise ValueError(
                "scale may only be supplied when scale_policy='provided'"
            )
        sequence_scale = math.nan

    frame_tracks, scale_lengths = _prepare_frames(typed_frames, config)
    if config.scale_policy == "sequence_median" and len(scale_lengths):
        candidate = float(np.median(scale_lengths))
        if np.isfinite(candidate) and candidate > config.epsilon:
            sequence_scale = candidate

    arrays = _empty_arrays(len(typed_frames), len(config.lags_seconds))
    if not np.isfinite(sequence_scale):
        return _result(
            arrays,
            timestamps=timestamps,
            config=config,
            sequence_scale=math.nan,
        )

    (
        values,
        strict_values,
        valid,
        strict_valid,
        matched,
        strict_matched,
        identity_distance,
        retained,
    ) = arrays
    frequency = Counter(
        track_id for frame_rows in frame_tracks for track_id in frame_rows
    )

    for current in range(len(typed_frames)):
        if not typed_frames[current].observed:
            continue
        for lag_index, lag_seconds in enumerate(config.lags_seconds):
            previous = _nearest_previous_frame(
                timestamps,
                current,
                lag_seconds,
                config.pair_tolerance_seconds,
            )
            if previous is None or not typed_frames[previous].observed:
                continue
            common = set(frame_tracks[previous]) & set(frame_tracks[current])
            common = {
                track_id
                for track_id in common
                if frequency[track_id] >= config.min_track_observations
                and all(
                    track_id in frame_tracks[position]
                    for position in range(previous, current + 1)
                )
            }

            displacement: list[float] = []
            strict_displacement: list[float] = []
            distances: list[float] = []
            for track_id in sorted(common):
                first = frame_tracks[previous][track_id]
                second = frame_tracks[current][track_id]
                delta = second.distal - first.distal
                motion = float(
                    np.sqrt(np.square(delta).sum(axis=1).mean())
                    / sequence_scale
                )
                if not np.isfinite(motion):
                    continue
                displacement.append(motion)
                distance = _identity_distance(first.identity, second.identity)
                if distance is not None:
                    distances.append(distance)
                    if distance <= config.identity_gate:
                        strict_displacement.append(motion)

            if len(displacement) < config.min_matched_arms:
                continue
            primary = np.asarray(displacement, dtype=np.float64)
            values[current, lag_index] = _motion_features(primary)
            valid[current, lag_index] = True
            matched[current, lag_index] = len(primary)
            retained[current, lag_index] = (
                len(strict_displacement) / len(primary)
            )
            if distances:
                identity_distance[current, lag_index] = float(
                    np.median(distances)
                )

            strict_matched[current, lag_index] = len(strict_displacement)
            if len(strict_displacement) >= config.min_matched_arms:
                strict = np.asarray(strict_displacement, dtype=np.float64)
                strict_values[current, lag_index] = _motion_features(strict)
                strict_valid[current, lag_index] = True

    return _result(
        arrays,
        timestamps=timestamps,
        config=config,
        sequence_scale=sequence_scale,
    )
