# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Decode-quality scoring and lightweight calibration for temporal arm tracking.

``decode_quality`` is a ranking score, not a probability.  It is the weighted
geometric mean of available component scores in [0, 1].  Higher tip, base,
island, and detector scores can only increase it; higher path and identity
costs can only decrease it.  Length plausibility is maximal inside a documented
body-radius-relative interval and decreases outside it.

Only fitted calibration models expose ``p_real``.  This naming distinction is
intentional: raw hand or logistic scores should not be interpreted as
probabilities before calibration on representative accepted and rejected arms.
The module is NumPy-only so quality scoring can be used in decoder workers
without adding a runtime dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np


_EPS = 1e-8
DEFAULT_FEATURE_NAMES = (
    "tip_score",
    "base_score",
    "mean_step_cost",
    "mean_identity_dist",
    "island_frac",
    "length_radius_plausibility",
    "detector_confidence",
)


@dataclass(frozen=True)
class DecodeQualityConfig:
    """Shape and weights of the deterministic hand score.

    Cost scales are soft e-folding scales.  Arm lengths between
    ``min_length_radius`` and ``max_length_radius`` body radii receive full
    plausibility credit; outside that interval the component decays
    exponentially with distance to the nearest bound.
    """

    tip_weight: float = 1.5
    base_weight: float = 1.0
    step_cost_weight: float = 1.0
    identity_weight: float = 1.0
    island_weight: float = 0.75
    length_plausibility_weight: float = 0.75
    detector_weight: float = 1.0
    step_cost_scale: float = 1.0
    identity_dist_scale: float = 1.0
    min_length_radius: float = 0.5
    max_length_radius: float = 8.0
    length_decay_radius: float = 1.0


@dataclass(frozen=True)
class QualityResult:
    """A quality score together with auditable component provenance."""

    decode_quality: float
    quality_components: dict[str, dict[str, Any]]
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "decode_quality": self.decode_quality,
            "quality_components": self.quality_components,
        }
        if self.rejection_reason is not None:
            result["rejection_reason"] = self.rejection_reason
        return result


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _positive_scale(value: float) -> float:
    value = _finite(value)
    return max(value if value is not None else 1.0, _EPS)


def _length_plausibility(
    length_px: float,
    body_radius_px: float,
    config: DecodeQualityConfig,
) -> tuple[float, float]:
    ratio = max(length_px, 0.0) / max(body_radius_px, _EPS)
    if ratio < config.min_length_radius:
        distance = config.min_length_radius - ratio
    elif ratio > config.max_length_radius:
        distance = ratio - config.max_length_radius
    else:
        distance = 0.0
    quality = np.exp(-distance / _positive_scale(config.length_decay_radius))
    return float(np.clip(quality, _EPS, 1.0)), float(ratio)


def compute_decode_quality(
    arm: Mapping[str, Any] | None = None,
    *,
    tip_score: float | None = None,
    base_score: float | None = None,
    mean_step_cost: float | None = None,
    mean_identity_dist: float | None = None,
    island_frac: float | None = None,
    length_px: float | None = None,
    body_radius_px: float | None = None,
    detector_confidence: float | None = None,
    rejection_reason: str | None = None,
    config: DecodeQualityConfig = DecodeQualityConfig(),
) -> QualityResult:
    """Compute a finite monotonic ranking score from all available evidence.

    Values explicitly passed as keywords take precedence over fields in
    ``arm``.  Missing or non-finite optional components are omitted and the
    remaining weights are renormalized.  Provenance records the raw value and
    transformation for every included component.  ``rejection_reason`` is
    carried as metadata only and never changes the score or decoder filtering.
    """
    record = {} if arm is None else arm

    def resolve(explicit: Any, key: str, *aliases: str) -> tuple[float | None, str]:
        if explicit is not None:
            return _finite(explicit), "argument"
        for candidate in (key, *aliases):
            if candidate in record:
                return _finite(record.get(candidate)), f"arm.{candidate}"
        return None, f"arm.{key}"

    tip_score, tip_source = resolve(tip_score, "tip_score")
    base_score, base_source = resolve(base_score, "base_score")
    mean_step_cost, step_source = resolve(mean_step_cost, "mean_step_cost")
    mean_identity_dist, identity_source = resolve(mean_identity_dist, "mean_identity_dist")
    island_frac, island_source = resolve(island_frac, "island_frac")
    length_px, length_source = resolve(length_px, "length_px")
    body_radius_px, radius_source = resolve(
        body_radius_px,
        "body_radius_px",
        "body_radius",
        "body_radius_source_px",
    )
    detector_confidence, detector_source = resolve(
        detector_confidence,
        "detector_confidence",
        "detector_score",
    )

    components: dict[str, dict[str, Any]] = {}
    weighted: list[tuple[float, float]] = []

    def add(name: str, raw: float, score: float, weight: float, source: str, transform: str) -> None:
        score = float(np.clip(score, _EPS, 1.0))
        weight = max(float(weight), 0.0)
        components[name] = {
            "raw_value": float(raw),
            "score": score,
            "weight": weight,
            "source": source,
            "transform": transform,
        }
        if weight > 0:
            weighted.append((score, weight))

    if tip_score is not None:
        add("tip_score", tip_score, _unit(tip_score), config.tip_weight, tip_source, "clip(raw, 0, 1)")
    if base_score is not None:
        add("base_score", base_score, _unit(base_score), config.base_weight, base_source, "clip(raw, 0, 1)")
    if mean_step_cost is not None:
        score = np.exp(-max(mean_step_cost, 0.0) / _positive_scale(config.step_cost_scale))
        add("mean_step_cost", mean_step_cost, score, config.step_cost_weight, step_source, "exp(-max(raw, 0)/scale)")
    if mean_identity_dist is not None:
        score = np.exp(-max(mean_identity_dist, 0.0) / _positive_scale(config.identity_dist_scale))
        add(
            "mean_identity_dist",
            mean_identity_dist,
            score,
            config.identity_weight,
            identity_source,
            "exp(-max(raw, 0)/scale)",
        )
    if island_frac is not None:
        add("island_frac", island_frac, _unit(island_frac), config.island_weight, island_source, "clip(raw, 0, 1)")
    if (
        length_px is not None
        and body_radius_px is not None
        and length_px >= 0
        and body_radius_px > 0
    ):
        plausibility, ratio = _length_plausibility(length_px, body_radius_px, config)
        add(
            "length_radius_plausibility",
            ratio,
            plausibility,
            config.length_plausibility_weight,
            f"{length_source} / {radius_source}",
            f"1 inside [{config.min_length_radius:g}, {config.max_length_radius:g}], exponential decay outside",
        )
        components["length_radius_plausibility"]["length_px"] = float(length_px)
        components["length_radius_plausibility"]["body_radius_px"] = float(body_radius_px)
    if detector_confidence is not None:
        add(
            "detector_confidence",
            detector_confidence,
            _unit(detector_confidence),
            config.detector_weight,
            detector_source,
            "clip(raw, 0, 1)",
        )

    if not weighted:
        quality = 0.0
    else:
        denominator = sum(weight for _, weight in weighted)
        quality = float(np.exp(sum(weight * np.log(score) for score, weight in weighted) / denominator))
    quality = float(np.clip(np.nan_to_num(quality, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
    return QualityResult(quality, components, rejection_reason)


def hand_score(
    records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    config: DecodeQualityConfig = DecodeQualityConfig(),
) -> float | np.ndarray:
    """Return deterministic ``decode_quality`` for one record or a record list."""
    if isinstance(records, Mapping):
        return compute_decode_quality(records, config=config).decode_quality
    return np.asarray([compute_decode_quality(record, config=config).decode_quality for record in records])


@dataclass
class HandScoreModel:
    """Serializable wrapper around :func:`hand_score`."""

    config: DecodeQualityConfig = DecodeQualityConfig()

    def predict_decode_quality(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(hand_score(records, self.config), dtype=np.float64)

    def predict(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Alias for estimators expecting a conventional ``predict`` method."""
        return self.predict_decode_quality(records)

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": "hand_score", "version": 1, "config": asdict(self.config)}

    def to_json(self, **json_kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HandScoreModel":
        return cls(DecodeQualityConfig(**dict(payload["config"])))

    @classmethod
    def from_json(cls, payload: str) -> "HandScoreModel":
        return cls.from_dict(json.loads(payload))


def _feature_value(record: Mapping[str, Any], name: str) -> float:
    if name == "length_radius_plausibility":
        length = _finite(record.get("length_px"))
        radius = _finite(record.get("body_radius_px"))
        if length is None or radius is None or radius <= 0:
            return 0.5
        return _length_plausibility(length, radius, DecodeQualityConfig())[0]
    value = _finite(record.get(name))
    return 0.0 if value is None else value


def quality_feature_matrix(
    records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
) -> np.ndarray:
    """Create a finite feature matrix; unavailable scalar evidence maps to zero."""
    return np.asarray(
        [[_feature_value(record, name) for name in feature_names] for record in records],
        dtype=np.float64,
    )


@dataclass
class LogisticQualityModel:
    """Small L2-regularized logistic ranker implemented with NumPy Newton steps.

    Its sigmoid output is named ``decode_quality`` rather than ``p_real`` because
    regularized in-sample logistic scores are not assumed calibrated.
    """

    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES
    coefficients: np.ndarray | None = None
    intercept: float = 0.0
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None

    def fit(
        self,
        records: Sequence[Mapping[str, Any]] | np.ndarray,
        labels: Sequence[int | bool],
        *,
        l2: float = 1e-3,
        max_iter: int = 100,
        tolerance: float = 1e-8,
    ) -> "LogisticQualityModel":
        x = (
            quality_feature_matrix(records, self.feature_names)
            if not isinstance(records, np.ndarray)
            else np.asarray(records, dtype=np.float64)
        )
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or len(y) == 0:
            raise ValueError("records and labels must be non-empty with matching rows")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("features must be finite and labels must be binary")
        self.feature_mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self.feature_scale = np.where(scale > _EPS, scale, 1.0)
        z = (x - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(z)), z])
        beta = np.zeros(design.shape[1], dtype=np.float64)
        penalty = np.eye(len(beta), dtype=np.float64) * max(float(l2), 0.0)
        penalty[0, 0] = 0.0
        for _ in range(max(int(max_iter), 1)):
            logits = np.clip(design @ beta, -40.0, 40.0)
            prediction = 1.0 / (1.0 + np.exp(-logits))
            gradient = design.T @ (prediction - y) + penalty @ beta
            weights = np.maximum(prediction * (1.0 - prediction), _EPS)
            hessian = design.T @ (design * weights[:, None]) + penalty
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            beta -= step
            if np.linalg.norm(step) <= tolerance * (1.0 + np.linalg.norm(beta)):
                break
        self.intercept = float(beta[0])
        self.coefficients = beta[1:]
        return self

    def predict_decode_quality(
        self,
        records: Sequence[Mapping[str, Any]] | np.ndarray,
    ) -> np.ndarray:
        if self.coefficients is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("fit the model before prediction")
        x = (
            quality_feature_matrix(records, self.feature_names)
            if not isinstance(records, np.ndarray)
            else np.asarray(records, dtype=np.float64)
        )
        z = (x - self.feature_mean) / self.feature_scale
        logits = np.clip(self.intercept + z @ self.coefficients, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, records: Sequence[Mapping[str, Any]] | np.ndarray) -> np.ndarray:
        """Return uncalibrated ``decode_quality`` scores."""
        return self.predict_decode_quality(records)

    def to_dict(self) -> dict[str, Any]:
        if self.coefficients is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("fit the model before serialization")
        return {
            "model_type": "logistic_quality",
            "version": 1,
            "feature_names": list(self.feature_names),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
        }

    def to_json(self, **json_kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogisticQualityModel":
        return cls(
            feature_names=tuple(payload["feature_names"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
        )

    @classmethod
    def from_json(cls, payload: str) -> "LogisticQualityModel":
        return cls.from_dict(json.loads(payload))


def _fit_scalar_logistic(
    scores: np.ndarray,
    labels: np.ndarray,
    l2: float,
    max_iter: int,
) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(scores)), scores])
    beta = np.array([np.log((labels.mean() + _EPS) / (1.0 - labels.mean() + _EPS)), 1.0])
    penalty = np.diag([0.0, max(float(l2), 0.0)])
    for _ in range(max(max_iter, 1)):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -40.0, 40.0)))
        gradient = x.T @ (p - labels) + penalty @ beta
        weight = np.maximum(p * (1.0 - p), _EPS)
        hessian = x.T @ (x * weight[:, None]) + penalty
        step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta -= step
        if np.linalg.norm(step) < 1e-9 * (1.0 + np.linalg.norm(beta)):
            break
    # Calibration must preserve the meaning "higher quality => no less real".
    # At the constrained optimum a negative slope becomes a constant prevalence
    # model rather than silently reversing the upstream ranking.
    if beta[1] < 0:
        prevalence = float(np.clip(labels.mean(), _EPS, 1.0 - _EPS))
        return float(np.log(prevalence / (1.0 - prevalence))), 0.0
    return float(beta[0]), float(beta[1])


@dataclass
class PlattCalibrator:
    """Sigmoid calibration mapping arbitrary quality scores to ``p_real``."""

    intercept: float | None = None
    slope: float | None = None

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int | bool],
        *,
        l2: float = 1e-6,
        max_iter: int = 100,
    ) -> "PlattCalibrator":
        x, y = _validated_calibration_data(scores, labels)
        self.intercept, self.slope = _fit_scalar_logistic(x, y, l2, max_iter)
        return self

    def predict_p_real(self, scores: Sequence[float]) -> np.ndarray:
        if self.intercept is None or self.slope is None:
            raise RuntimeError("fit the calibrator before prediction")
        x = np.asarray(scores, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(self.intercept + self.slope * x, -40.0, 40.0)))

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        """Return calibrated ``p_real`` values."""
        return self.predict_p_real(scores)

    def to_dict(self) -> dict[str, Any]:
        if self.intercept is None or self.slope is None:
            raise RuntimeError("fit the calibrator before serialization")
        return {"model_type": "platt", "version": 1, "intercept": self.intercept, "slope": self.slope}

    def to_json(self, **json_kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlattCalibrator":
        return cls(float(payload["intercept"]), float(payload["slope"]))

    @classmethod
    def from_json(cls, payload: str) -> "PlattCalibrator":
        return cls.from_dict(json.loads(payload))


def _validated_calibration_data(
    scores: Sequence[float],
    labels: Sequence[int | bool],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("scores and labels must be non-empty with matching lengths")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("scores must be finite and labels must be binary")
    if np.unique(y).size < 2:
        raise ValueError("calibration requires both label classes")
    return x, y


@dataclass
class IsotonicCalibrator:
    """Non-decreasing piecewise-linear calibration implemented with PAVA."""

    thresholds: np.ndarray | None = None
    probabilities: np.ndarray | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[int | bool]) -> "IsotonicCalibrator":
        x, y = _validated_calibration_data(scores, labels)
        order = np.argsort(x, kind="mergesort")
        unique_x, inverse = np.unique(x[order], return_inverse=True)
        sums = np.bincount(inverse, weights=y[order]).astype(np.float64)
        counts = np.bincount(inverse).astype(np.float64)
        values = sums / counts

        blocks: list[list[float]] = []
        for start, (value, weight) in enumerate(zip(values, counts)):
            blocks.append([float(start), float(start), float(value), float(weight)])
            while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
                right = blocks.pop()
                left = blocks.pop()
                total = left[3] + right[3]
                blocks.append([left[0], right[1], (left[2] * left[3] + right[2] * right[3]) / total, total])

        fitted = np.empty(len(unique_x), dtype=np.float64)
        for start, end, value, _ in blocks:
            fitted[int(start):int(end) + 1] = value
        self.thresholds = unique_x
        self.probabilities = np.clip(fitted, 0.0, 1.0)
        return self

    def predict_p_real(self, scores: Sequence[float]) -> np.ndarray:
        if self.thresholds is None or self.probabilities is None:
            raise RuntimeError("fit the calibrator before prediction")
        x = np.asarray(scores, dtype=np.float64)
        return np.interp(x, self.thresholds, self.probabilities)

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        """Return calibrated ``p_real`` values."""
        return self.predict_p_real(scores)

    def to_dict(self) -> dict[str, Any]:
        if self.thresholds is None or self.probabilities is None:
            raise RuntimeError("fit the calibrator before serialization")
        return {
            "model_type": "isotonic",
            "version": 1,
            "thresholds": self.thresholds.tolist(),
            "probabilities": self.probabilities.tolist(),
        }

    def to_json(self, **json_kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IsotonicCalibrator":
        return cls(
            np.asarray(payload["thresholds"], dtype=np.float64),
            np.asarray(payload["probabilities"], dtype=np.float64),
        )

    @classmethod
    def from_json(cls, payload: str) -> "IsotonicCalibrator":
        return cls.from_dict(json.loads(payload))


def model_from_json(payload: str) -> HandScoreModel | LogisticQualityModel | PlattCalibrator | IsotonicCalibrator:
    """Deserialize any model emitted by this module."""
    data = json.loads(payload)
    model_type = data.get("model_type")
    classes = {
        "hand_score": HandScoreModel,
        "logistic_quality": LogisticQualityModel,
        "platt": PlattCalibrator,
        "isotonic": IsotonicCalibrator,
    }
    if model_type not in classes:
        raise ValueError(f"unknown quality model type: {model_type!r}")
    return classes[model_type].from_dict(data)


def brier_score(labels: Sequence[int | bool], p_real: Sequence[float]) -> float:
    """Mean squared calibration error for calibrated probabilities."""
    y, p = _metric_arrays(labels, p_real)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    labels: Sequence[int | bool],
    p_real: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Equal-width expected calibration error."""
    y, p = _metric_arrays(labels, p_real)
    n_bins = max(int(n_bins), 1)
    index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_index in range(n_bins):
        mask = index == bin_index
        if mask.any():
            error += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(error)


def precision_recall_summary(
    labels: Sequence[int | bool],
    p_real: Sequence[float],
) -> dict[str, float]:
    """Compact threshold-free and best-F1 precision/recall summary."""
    y, p = _metric_arrays(labels, p_real)
    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1.0 - sorted_y)
    positives = float(sorted_y.sum())
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(positives, 1.0)
    average_precision = float((precision * sorted_y).sum() / max(positives, 1.0))
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, _EPS)
    best = int(np.argmax(f1))
    return {
        "average_precision": average_precision,
        "best_f1": float(f1[best]),
        "precision_at_best_f1": float(precision[best]),
        "recall_at_best_f1": float(recall[best]),
        "threshold_at_best_f1": float(p[order[best]]),
    }


def calibration_summary(
    labels: Sequence[int | bool],
    p_real: Sequence[float],
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    """Return Brier, ECE, and precision/recall diagnostics."""
    return {
        "brier": brier_score(labels, p_real),
        "ece": expected_calibration_error(labels, p_real, n_bins=n_bins),
        **precision_recall_summary(labels, p_real),
    }


def _metric_arrays(
    labels: Sequence[int | bool],
    p_real: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(p_real, dtype=np.float64).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("labels and probabilities must be non-empty with matching lengths")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("labels must be binary and probabilities finite")
    return y, np.clip(p, 0.0, 1.0)
