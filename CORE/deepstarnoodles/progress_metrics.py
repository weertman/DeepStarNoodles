# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Behavior metrics from progress temporal VAE tokens and window latents.

Frame tokens (frozen frame encoder, z-scored on training statistics) feed a
60-frame sliding diagonal-Gaussian estimate of marginal entropy and AR(1)
entropy rate. Window latents feed standardized latent speed and posterior KL.
Everything is per frame on the uniform cadence grid; segment reduction is a
median over frames with enough finite samples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

LOG_2PI_E = math.log(2.0 * math.pi * math.e)
VARIANCE_FLOOR = 1e-8


@dataclass(frozen=True)
class FeatureStandardization:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def as_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureStandardization":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )


def fit_standardization(values: np.ndarray, *, minimum_std: float = 1e-6) -> FeatureStandardization:
    """Per-dimension mean/std over finite rows only."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must have shape [N, D]")
    finite = np.isfinite(values).all(axis=1)
    if finite.sum() < 2:
        raise ValueError("need at least two finite rows to standardize")
    selected = values[finite]
    std = selected.std(axis=0, ddof=1)
    return FeatureStandardization(
        mean=selected.mean(axis=0), std=np.maximum(std, minimum_std)
    )


def valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, stop) spans of consecutive True values."""

    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 1:
        raise ValueError("valid must be one-dimensional")
    padded = np.concatenate(([False], valid, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _gaussian_entropy(variance: np.ndarray) -> np.ndarray:
    return 0.5 * (LOG_2PI_E + np.log(np.maximum(variance, VARIANCE_FLOOR)))


def token_entropy_metrics(
    tokens: np.ndarray,
    valid: np.ndarray,
    *,
    window: int = 60,
    centered: bool = True,
) -> dict[str, np.ndarray]:
    """Sliding diagonal-Gaussian entropy metrics in nats per dimension.

    ``tokens`` must already be standardized. Windows never span an invalid
    frame; each window's value is attributed to its center (or last) frame.
    Returns per-frame ``marginal_entropy``, ``entropy_rate`` (AR(1) residual),
    and ``predictable_information``; frames without a full window are NaN.
    """

    tokens = np.asarray(tokens, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if tokens.ndim != 2 or valid.shape != (len(tokens),):
        raise ValueError("tokens must be [F, D] with matching valid [F]")
    if window < 3:
        raise ValueError("window must be at least 3 frames")
    frames = len(tokens)
    marginal = np.full(frames, np.nan)
    rate = np.full(frames, np.nan)
    anchor = window // 2 if centered else window - 1
    for start, stop in valid_runs(valid):
        if stop - start < window:
            continue
        run = tokens[start:stop]
        if not np.isfinite(run).all():
            continue
        views = sliding_window_view(run, window, axis=0)  # [W, D, window]
        values = np.moveaxis(views, -1, 1)  # [W, window, D]
        variance = values.var(axis=1, ddof=1)
        marginal_w = _gaussian_entropy(variance).mean(axis=1)

        previous = values[:, :-1, :]
        following = values[:, 1:, :]
        previous_centered = previous - previous.mean(axis=1, keepdims=True)
        following_centered = following - following.mean(axis=1, keepdims=True)
        covariance = (previous_centered * following_centered).sum(axis=1)
        previous_ss = (previous_centered**2).sum(axis=1)
        slope = covariance / np.maximum(previous_ss, VARIANCE_FLOOR)
        residual_ss = (following_centered**2).sum(axis=1) - slope * covariance
        residual_variance = np.maximum(residual_ss, 0.0) / max(window - 3, 1)
        rate_w = _gaussian_entropy(residual_variance).mean(axis=1)

        centers = np.arange(start + anchor, start + anchor + len(marginal_w))
        marginal[centers] = marginal_w
        rate[centers] = rate_w
    predictable = np.where(
        np.isfinite(marginal) & np.isfinite(rate),
        np.maximum(marginal - rate, 0.0),
        np.nan,
    )
    return {
        "marginal_entropy": marginal,
        "entropy_rate": rate,
        "predictable_information": predictable,
    }


def latent_speed(
    standardized_mean: np.ndarray,
    valid: np.ndarray,
    *,
    lag_frames: int,
) -> np.ndarray:
    """Per-frame ||mu_t - mu_{t-lag}|| where both endpoints are valid."""

    mean = np.asarray(standardized_mean, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if mean.ndim != 2 or valid.shape != (len(mean),):
        raise ValueError("standardized_mean must be [F, D] with matching valid [F]")
    if lag_frames < 1:
        raise ValueError("lag_frames must be positive")
    speed = np.full(len(mean), np.nan)
    if len(mean) > lag_frames:
        both = valid[lag_frames:] & valid[:-lag_frames]
        delta = mean[lag_frames:] - mean[:-lag_frames]
        norms = np.linalg.norm(delta, axis=1)
        speed[lag_frames:] = np.where(both, norms, np.nan)
    return speed


def posterior_kl(mean: np.ndarray, logvar: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """KL(q(z|x) || N(0, I)) in nats per window, NaN where invalid."""

    mean = np.asarray(mean, dtype=np.float64)
    logvar = np.asarray(logvar, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if mean.shape != logvar.shape or mean.ndim != 2 or valid.shape != (len(mean),):
        raise ValueError("mean/logvar must be [F, D] with matching valid [F]")
    kl = 0.5 * (mean**2 + np.exp(logvar) - 1.0 - logvar).sum(axis=1)
    return np.where(valid, kl, np.nan)


def segment_median(values: np.ndarray, *, minimum_samples: int = 30) -> tuple[float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < minimum_samples:
        return float("nan"), int(len(finite))
    return float(np.median(finite)), int(len(finite))


__all__ = [
    "FeatureStandardization",
    "LOG_2PI_E",
    "fit_standardization",
    "latent_speed",
    "posterior_kl",
    "segment_median",
    "token_entropy_metrics",
    "valid_runs",
]
