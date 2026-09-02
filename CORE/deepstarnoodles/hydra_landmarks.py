# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Compact landmarks from an activated 18-channel Hydra field. No tensors kept."""
from __future__ import annotations

import numpy as np

from deepstarnoodles.model import CHANNEL_NAMES

CHANNEL_CONTRACT = "legacy10+embedding8"
GEOMETRY_PEAK_NAMES = (
    "arm_support",
    "arm_centerline",
    "arm_progress",
    "mouth",
    "madreporite",
    "butthole",
    "arm_base",
    "arm_tip",
)
DIRECTION_NAMES = ("arm_dx", "arm_dy")
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNEL_NAMES)}


def peak_argmax(channel: np.ndarray) -> tuple[float, float, float]:
    """Return (u, v, score) for the brightest pixel. u is x, v is y."""
    heat = np.asarray(channel, dtype=np.float32)
    if heat.ndim != 2 or heat.size == 0:
        return 0.0, 0.0, 0.0
    index = int(np.argmax(heat))
    v, u = divmod(index, heat.shape[1])
    return float(u), float(v), float(heat[v, u])


def crop_to_source(
    u: float,
    v: float,
    *,
    left: float,
    top: float,
    side: float,
    size: int,
) -> tuple[float, float]:
    scale = float(side) / float(size)
    return left + (u + 0.5) * scale, top + (v + 0.5) * scale


def _sample(field: np.ndarray, u: float, v: float) -> float:
    height, width = field.shape[-2], field.shape[-1]
    x = int(np.clip(round(u), 0, width - 1))
    y = int(np.clip(round(v), 0, height - 1))
    return float(field[y, x])


def summarize_activated(
    activated: np.ndarray,
    *,
    left: float,
    top: float,
    side: float,
    image_size: int | None = None,
) -> dict:
    field = np.asarray(activated, dtype=np.float32)
    if field.ndim != 3 or field.shape[0] < 18:
        raise ValueError("activated field must be (18,H,W)")
    size = int(image_size or field.shape[-1])
    landmarks: dict[str, dict[str, float]] = {}
    peaks: dict[str, tuple[float, float, float]] = {}
    for name in GEOMETRY_PEAK_NAMES:
        u, v, score = peak_argmax(field[CHANNEL_INDEX[name]])
        peaks[name] = (u, v, score)
        x, y = crop_to_source(u, v, left=left, top=top, side=side, size=size)
        landmarks[name] = {"u": u, "v": v, "score": score, "x": x, "y": y}

    sample_name = "arm_support" if peaks["arm_support"][2] > 0 else "arm_centerline"
    su, sv, _ = peaks[sample_name]
    for name in DIRECTION_NAMES:
        landmarks[name] = {
            "value": _sample(field[CHANNEL_INDEX[name]], su, sv),
            "sample": sample_name,
        }

    embed_name = "mouth" if peaks["mouth"][2] > 0 else sample_name
    eu, ev, _ = peaks[embed_name]
    embedding = [_sample(field[10 + dim], eu, ev) for dim in range(8)]
    return {
        "landmarks": landmarks,
        "embedding": embedding,
        "embedding_sample": embed_name,
    }
