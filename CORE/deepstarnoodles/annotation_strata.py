# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Resolution-independent strata derived from source-coordinate annotations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from deepstarnoodles.dataset import _chaikin_smooth, _resample_polyline, _shape_points

CANONICAL_SIZE = 512
CLOSE_ARM_THRESHOLD_CANONICAL_PX = 4.0
NEARBY_TIP_THRESHOLD_CANONICAL_PX = 32.0


def _minimum_pairwise_separation(point_sets: Sequence[np.ndarray]) -> float | None:
    minimum: float | None = None
    for left in range(len(point_sets)):
        a = point_sets[left]
        if not len(a):
            continue
        for right in range(left + 1, len(point_sets)):
            b = point_sets[right]
            if not len(b):
                continue
            distance = float(
                np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).min()
            )
            minimum = distance if minimum is None else min(minimum, distance)
    return minimum


def annotation_strata(
    shapes: Sequence[dict[str, Any]],
    *,
    width: int | float,
    height: int | float,
) -> dict[str, Any]:
    """Measure hard annotation strata after mapping source points to 512 square.

    Arms are smoothed and sampled at one canonical pixel. Two points at each end
    are excluded from the close-arm measurement so shared/nearby bases and tips
    do not masquerade as an interior crossing.
    """
    source_width = float(width)
    source_height = float(height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("annotation width and height must be positive")
    scale = np.asarray(
        [CANONICAL_SIZE / source_width, CANONICAL_SIZE / source_height],
        dtype=np.float32,
    )
    arms: list[np.ndarray] = []
    for shape in shapes:
        if (
            shape.get("outside")
            or shape.get("label") != "arm"
            or shape.get("type") != "polyline"
        ):
            continue
        source_points = _shape_points(shape)
        if len(source_points) < 2:
            continue
        canonical = source_points * scale
        arms.append(
            _resample_polyline(_chaikin_smooth(canonical, iterations=2), spacing=1.0)
        )

    interiors = [arm[2:-2] for arm in arms]
    interior_separation = _minimum_pairwise_separation(interiors)
    tips = [arm[-1:] for arm in arms]
    tip_separation = _minimum_pairwise_separation(tips)
    close_arm = (
        interior_separation is not None
        and interior_separation <= CLOSE_ARM_THRESHOLD_CANONICAL_PX
    )
    nearby_tip = (
        tip_separation is not None
        and tip_separation < NEARBY_TIP_THRESHOLD_CANONICAL_PX
    )
    return {
        "flags": {
            "close_arm": bool(close_arm),
            "crossing": bool(close_arm),
            "nearby_tip": bool(nearby_tip),
        },
        "measurements": {
            "minimum_interior_arm_separation_canonical_px": interior_separation,
            "minimum_tip_separation_canonical_px": tip_separation,
        },
        "canonical_size": [CANONICAL_SIZE, CANONICAL_SIZE],
        "source_size": [source_width, source_height],
        "thresholds_canonical_px": {
            "close_arm": CLOSE_ARM_THRESHOLD_CANONICAL_PX,
            "crossing": CLOSE_ARM_THRESHOLD_CANONICAL_PX,
            "nearby_tip": NEARBY_TIP_THRESHOLD_CANONICAL_PX,
        },
        "arm_count": len(arms),
    }


def annotation_strata_from_index_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Load an index row's source annotation and return canonical strata."""
    shapes_path = row.get("shapes_path")
    if not shapes_path:
        raise ValueError("index row has no shapes_path")
    shapes = json.loads(Path(str(shapes_path)).read_text(encoding="utf-8"))
    if not isinstance(shapes, list):
        raise ValueError("shapes annotation must contain a JSON list")
    return annotation_strata(
        shapes,
        width=float(row["width"]),
        height=float(row["height"]),
    )


def native_tolerance_metadata(
    image_size: int,
    *,
    match_tolerance_canonical_px: float,
    point_tolerance_canonical_px: float,
) -> dict[str, Any]:
    """Describe canonical tolerances and their square-raster native values."""
    scale = float(image_size) / CANONICAL_SIZE
    return {
        "canonical_size": [CANONICAL_SIZE, CANONICAL_SIZE],
        "native_size": [int(image_size), int(image_size)],
        "native_scale_from_canonical": scale,
        "match_tolerance_canonical_px": float(match_tolerance_canonical_px),
        "match_tolerance_native_px": float(match_tolerance_canonical_px) * scale,
        "point_tolerance_canonical_px": float(point_tolerance_canonical_px),
        "point_tolerance_native_px": float(point_tolerance_canonical_px) * scale,
        "stratum_thresholds_canonical_px": {
            "close_arm": CLOSE_ARM_THRESHOLD_CANONICAL_PX,
            "crossing": CLOSE_ARM_THRESHOLD_CANONICAL_PX,
            "nearby_tip": NEARBY_TIP_THRESHOLD_CANONICAL_PX,
        },
        "stratum_thresholds_native_px": {
            "close_arm": CLOSE_ARM_THRESHOLD_CANONICAL_PX * scale,
            "crossing": CLOSE_ARM_THRESHOLD_CANONICAL_PX * scale,
            "nearby_tip": NEARBY_TIP_THRESHOLD_CANONICAL_PX * scale,
        },
        "stratum_membership_space": "canonical_512x512",
    }
