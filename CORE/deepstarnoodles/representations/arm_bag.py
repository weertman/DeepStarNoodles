# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Configurable, permutation-invariant static arm representations.

Native polylines are interpreted as base-to-tip.  Each arm is resampled by
arc length, translated to its base, rotated so its chord points along +x, and
normalized by its own arc length.  Frame-level length and radius scalars are
normalized independently before arm states are pooled by mean and population
standard deviation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ARM_BAG_REPRESENTATION = "deepstarnoodles_static_arm_bag@1"
REPRESENTATION = ARM_BAG_REPRESENTATION
_SCALAR_NAMES = ("arclen_n", "radius_n", "curv")
_POLYLINE_KEYS = ("polyline", "polyline_crop", "polyline_source")
_SCALE_POLICIES = frozenset(("frame_median", "provided"))

__all__ = [
    "ARM_BAG_REPRESENTATION",
    "REPRESENTATION",
    "ArmBagConfig",
    "ArmBagResult",
    "arm_bag",
    "arm_bag_recipe",
    "arm_bag_sequence",
    "arm_feature_names",
    "arm_state",
    "pool_arm_states",
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


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
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


@dataclass(frozen=True)
class ArmBagConfig:
    """Recipe controls for :func:`arm_bag`.

    ``frame_median`` derives any scale not supplied to :func:`arm_bag` from
    the current frame.  ``provided`` requires both scales to be supplied.
    """

    samples: int = 24
    epsilon: float = 1e-8
    scale_policy: str = "frame_median"

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _integer("samples", self.samples, minimum=2))
        object.__setattr__(
            self, "epsilon", _positive_float("epsilon", self.epsilon)
        )
        if not isinstance(self.scale_policy, str):
            raise TypeError("scale_policy must be a string")
        if self.scale_policy not in _SCALE_POLICIES:
            choices = ", ".join(sorted(_SCALE_POLICIES))
            raise ValueError(f"scale_policy must be one of: {choices}")


@dataclass(frozen=True)
class ArmBagResult:
    """One immutable static bag result with auditable recipe provenance."""

    values: np.ndarray
    valid: bool
    arm_count: int
    feature_names: tuple[str, ...]
    representation: str
    recipe_digest: str
    recipe_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = _readonly_array(
            self.values, dtype=np.float32, ndim=1, name="values"
        )
        if not np.isfinite(values).all():
            raise ValueError("values must be finite")
        names = tuple(str(name) for name in self.feature_names)
        if values.shape != (len(names),):
            raise ValueError("values and feature_names must have matching lengths")
        count = _integer("arm_count", self.arm_count, minimum=0)
        if not isinstance(self.valid, (bool, np.bool_)):
            raise TypeError("valid must be boolean")
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
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "arm_count", count)
        object.__setattr__(self, "feature_names", names)
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


def _coerce_config(config: ArmBagConfig) -> ArmBagConfig:
    if not isinstance(config, ArmBagConfig):
        raise TypeError("config must be an ArmBagConfig")
    return config


def arm_feature_names(
    config: ArmBagConfig | int | None = None,
    *,
    samples: int | None = None,
) -> tuple[str, ...]:
    """Return feature names for a config or an explicit sample count."""

    if isinstance(config, ArmBagConfig):
        if samples is not None:
            raise ValueError("samples cannot be supplied with an ArmBagConfig")
        count = config.samples
    elif config is None:
        count = 24 if samples is None else _integer("samples", samples, minimum=2)
    else:
        if samples is not None:
            raise ValueError("samples was supplied twice")
        count = _integer("samples", config, minimum=2)

    shape = tuple(
        f"shape_p{index:02d}_{axis}"
        for index in range(count)
        for axis in ("x", "y")
    )
    per_arm = shape + _SCALAR_NAMES
    return tuple(f"mean_{name}" for name in per_arm) + tuple(
        f"std_{name}" for name in per_arm
    )


def arm_bag_recipe(config: ArmBagConfig = ArmBagConfig()) -> dict[str, Any]:
    """Return the JSON-serializable deterministic static-bag recipe."""

    config = _coerce_config(config)
    return {
        "representation": ARM_BAG_REPRESENTATION,
        "config": {
            "samples": config.samples,
            "epsilon": config.epsilon,
            "scale_policy": config.scale_policy,
        },
        "feature_names": list(arm_feature_names(config)),
        "polyline_order": "base_to_tip",
        "shape": {
            "sampling": "uniform_arclength",
            "origin": "arm_base",
            "orientation": "chord_to_positive_x",
            "normalization": "per_arm_arclength",
            "turning": "mean_absolute_sampled_turn_angle",
        },
        "scalars": {
            "length": "arm_arclength",
            "radius": "tip_distance_from_mean_bases_or_body_center",
            "argument_scales_override_policy": True,
        },
        "pooling": "weighted_population_mean_std",
    }


def _polyline(value: Any) -> np.ndarray:
    source = value
    if isinstance(value, Mapping):
        for key in _POLYLINE_KEYS:
            if key in value and value[key] is not None:
                source = value[key]
                break
        else:
            choices = ", ".join(repr(key) for key in _POLYLINE_KEYS)
            raise ValueError(f"arm mapping must contain one of {choices}")
    try:
        points = np.array(source, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("polyline coordinates must be numeric") from exc
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        raise ValueError("each polyline must have shape [point, 2] with at least two points")
    if not np.isfinite(points).all():
        raise ValueError("polyline coordinates must be finite")
    return points


def _arms(values: Any) -> list[np.ndarray]:
    if isinstance(values, Mapping):
        if "arms" in values:
            values = values["arms"]
        else:
            return [_polyline(values)]

    if isinstance(values, np.ndarray):
        if values.ndim == 2:
            return [_polyline(values)]
        if values.ndim == 3:
            return [_polyline(row) for row in values]
        if values.size == 0:
            return []
        raise ValueError("polyline arrays must have shape [point, 2] or [arm, point, 2]")

    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError("arms must be an iterable of polylines or arm mappings")
    items = list(values)
    if not items:
        return []

    # A plain nested numeric [point, 2] sequence is one polyline.
    try:
        numeric = np.asarray(items, dtype=np.float64)
    except (TypeError, ValueError):
        numeric = np.empty(0)
    if numeric.ndim == 2 and numeric.shape[1:] == (2,):
        return [_polyline(numeric)]
    return [_polyline(item) for item in items]


def _point(value: Any, *, name: str) -> np.ndarray:
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite (2,) point")
    return result


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


def _arm_axis(polyline: np.ndarray, epsilon: float) -> np.ndarray:
    chord = polyline[-1] - polyline[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= epsilon:
        offsets = polyline - polyline[0]
        distances = np.linalg.norm(offsets, axis=1)
        farthest = int(np.argmax(distances))
        chord = offsets[farthest]
        chord_length = float(distances[farthest])
    if chord_length <= epsilon:
        return np.asarray((1.0, 0.0), dtype=np.float64)
    return chord / chord_length


def _rotate_to_axis(vectors: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            vectors[..., 0] * axis[0] + vectors[..., 1] * axis[1],
            -vectors[..., 0] * axis[1] + vectors[..., 1] * axis[0],
        )
    )


def _turning(polyline: np.ndarray, epsilon: float) -> np.ndarray:
    steps = np.diff(polyline, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    steps = steps[lengths > epsilon]
    if len(steps) < 2:
        return np.empty(0, dtype=np.float64)
    cross = steps[:-1, 0] * steps[1:, 1] - steps[:-1, 1] * steps[1:, 0]
    dot = np.sum(steps[:-1] * steps[1:], axis=1)
    return np.arctan2(cross, dot)


def arm_state(
    polyline_or_arm: np.ndarray | Mapping[str, Any] | Sequence[Sequence[float]],
    *,
    length_scale: float,
    radius_scale: float,
    body_center: Sequence[float] | np.ndarray,
    config: ArmBagConfig = ArmBagConfig(),
) -> np.ndarray:
    """Return one aligned arm shape and its three normalized scalars."""

    config = _coerce_config(config)
    length_scale = _positive_float("length_scale", length_scale)
    radius_scale = _positive_float("radius_scale", radius_scale)
    points = _polyline(polyline_or_arm)
    center = _point(body_center, name="body_center")
    sampled, arclength = _resample(points, config.samples, config.epsilon)
    if arclength <= config.epsilon:
        aligned = np.zeros((config.samples, 2), dtype=np.float64)
        curvature = 0.0
    else:
        aligned = (
            _rotate_to_axis(sampled - points[0], _arm_axis(points, config.epsilon))
            / arclength
        )
        turns = _turning(sampled, config.epsilon)
        curvature = float(np.mean(np.abs(turns))) if len(turns) else 0.0
    tip_radius = float(np.linalg.norm(points[-1] - center))
    scalars = np.asarray(
        (
            arclength / length_scale,
            tip_radius / radius_scale,
            curvature,
        ),
        dtype=np.float64,
    )
    return np.concatenate((aligned.ravel(), scalars))


def _weights(weights: Any, count: int) -> np.ndarray | None:
    if weights is None:
        return None
    try:
        result = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be numeric") from exc
    if result.shape != (count,):
        raise ValueError(f"weights must have shape ({count},)")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if count and float(result.sum()) <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return result.copy()


def pool_arm_states(states: Any, weights: Any = None) -> np.ndarray:
    """Pool arm rows by (optionally weighted) mean and population deviation."""

    try:
        values = np.asarray(states, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("states must be numeric") from exc
    if values.ndim != 2 or not len(values) or values.shape[1] == 0:
        raise ValueError("states must be a non-empty [arm, feature] array")
    if not np.isfinite(values).all():
        raise ValueError("states must be finite")
    arm_weights = _weights(weights, len(values))
    if arm_weights is None:
        mean = values.mean(axis=0)
        spread = values.std(axis=0)
    else:
        normalized = arm_weights / float(arm_weights.sum())
        mean = normalized @ values
        spread = np.sqrt(
            np.sum(normalized[:, None] * np.square(values - mean), axis=0)
        )
    return np.concatenate((mean, spread)).astype(np.float32)


def _bag_result(
    values: np.ndarray,
    *,
    valid: bool,
    arm_count: int,
    config: ArmBagConfig,
) -> ArmBagResult:
    metadata = arm_bag_recipe(config)
    return ArmBagResult(
        values=values,
        valid=valid,
        arm_count=arm_count,
        feature_names=arm_feature_names(config),
        representation=ARM_BAG_REPRESENTATION,
        recipe_digest=_canonical_digest(metadata),
        recipe_metadata=metadata,
    )


def arm_bag(
    polylines_or_arm_dicts: Any,
    config: ArmBagConfig = ArmBagConfig(),
    *,
    weights: Any = None,
    body_center: Sequence[float] | np.ndarray | None = None,
    length_scale: float | None = None,
    radius_scale: float | None = None,
) -> ArmBagResult:
    """Build one static bag from native base-to-tip arms.

    Empty frames and frames whose data-derived scale is degenerate return a
    zero vector with ``valid=False``.  Malformed coordinates, weights, centers,
    and explicitly supplied scales raise ``ValueError``.
    """

    config = _coerce_config(config)
    if config.scale_policy == "provided" and (
        length_scale is None or radius_scale is None
    ):
        raise ValueError(
            "scale_policy='provided' requires length_scale and radius_scale"
        )
    supplied_length = (
        None
        if length_scale is None
        else _positive_float("length_scale", length_scale)
    )
    supplied_radius = (
        None
        if radius_scale is None
        else _positive_float("radius_scale", radius_scale)
    )
    center = None if body_center is None else _point(body_center, name="body_center")
    arms = _arms(polylines_or_arm_dicts)
    arm_weights = _weights(weights, len(arms))
    names = arm_feature_names(config)
    zeros = np.zeros(len(names), dtype=np.float32)
    if not arms:
        return _bag_result(zeros, valid=False, arm_count=0, config=config)

    if center is None:
        center = np.mean(np.stack([arm[0] for arm in arms]), axis=0)
    lengths = np.asarray(
        [
            float(np.linalg.norm(np.diff(arm, axis=0), axis=1).sum())
            for arm in arms
        ],
        dtype=np.float64,
    )
    radii = np.asarray(
        [float(np.linalg.norm(arm[-1] - center)) for arm in arms],
        dtype=np.float64,
    )
    if not np.any(lengths > config.epsilon):
        return _bag_result(
            zeros, valid=False, arm_count=len(arms), config=config
        )

    effective_length = (
        supplied_length if supplied_length is not None else float(np.median(lengths))
    )
    effective_radius = (
        supplied_radius if supplied_radius is not None else float(np.median(radii))
    )
    if (
        effective_length <= config.epsilon
        or effective_radius <= config.epsilon
        or not np.isfinite(effective_length)
        or not np.isfinite(effective_radius)
    ):
        return _bag_result(
            zeros, valid=False, arm_count=len(arms), config=config
        )

    states = np.stack(
        [
            arm_state(
                arm,
                length_scale=effective_length,
                radius_scale=effective_radius,
                body_center=center,
                config=config,
            )
            for arm in arms
        ]
    )
    values = pool_arm_states(states, arm_weights)
    return _bag_result(values, valid=True, arm_count=len(arms), config=config)


def _per_frame_scalar(value: Any, count: int, *, name: str) -> list[Any]:
    if value is None:
        return [None] * count
    if np.isscalar(value):
        return [value] * count
    items = list(value)
    if len(items) != count:
        raise ValueError(f"{name} must be scalar or contain one value per frame")
    return items


def _per_frame_centers(value: Any, count: int) -> list[Any]:
    if value is None:
        return [None] * count
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        array = np.empty(0)
    if array.shape == (2,):
        return [value] * count
    if array.shape == (count, 2):
        return [array[index] for index in range(count)]
    items = list(value)
    if len(items) != count:
        raise ValueError("body_center must be one point or one point per frame")
    return items


def _per_frame_weights(value: Any, count: int) -> list[Any]:
    if value is None:
        return [None] * count
    if count == 1:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            array = np.empty(0)
        if array.ndim == 1:
            return [value]
    items = list(value)
    if len(items) != count:
        raise ValueError("weights must contain one weight vector per frame")
    if count > 1 and any(np.isscalar(item) for item in items):
        raise ValueError("each frame's weights must be a vector or None")
    return items


def arm_bag_sequence(
    frames: Sequence[Any] | Iterable[Any],
    config: ArmBagConfig = ArmBagConfig(),
    *,
    weights: Any = None,
    body_center: Any = None,
    length_scale: Any = None,
    radius_scale: Any = None,
) -> tuple[ArmBagResult, ...]:
    """Build bags for a frame sequence, returning one result per frame.

    A scalar scale or one body-center point is shared by all frames.  Otherwise
    those arguments, and ``weights``, must contain one item per frame.
    """

    config = _coerce_config(config)
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Iterable):
        raise TypeError("frames must be an iterable")
    frame_list = list(frames)
    count = len(frame_list)
    frame_weights = _per_frame_weights(weights, count)
    centers = _per_frame_centers(body_center, count)
    length_scales = _per_frame_scalar(length_scale, count, name="length_scale")
    radius_scales = _per_frame_scalar(radius_scale, count, name="radius_scale")
    return tuple(
        arm_bag(
            frame,
            config,
            weights=frame_weights[index],
            body_center=centers[index],
            length_scale=length_scales[index],
            radius_scale=radius_scales[index],
        )
        for index, frame in enumerate(frame_list)
    )
