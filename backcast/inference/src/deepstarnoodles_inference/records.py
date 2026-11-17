# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Output record schema constants and the per-frame tail every writer appends (torch-free).

Schema 3 adds, after every existing key, per star ``body_origin_model``, ``body_origin_code``,
``support_area_model_px`` and ``support_centroid_model``, and per frame ``time_s``, ``fps`` and
``schema_version``. Schema 4 adds per star ``landmarks_model``: ``[x, y, p]`` for the mouth, madreporite
and butthole. Readers detect the new fields by key presence: a record without them is a legacy
(0.5.0.dev10 or earlier) record and stays valid.
"""
from __future__ import annotations

import math

# 1 = single-star records, 2 = the multi-star payload (runtime.multi_star_video_payload),
# 3 = per-record body and time fields, 4 = per-star landmarks. Absent from a frame record = legacy.
OUTPUT_SCHEMA_VERSION = 4

# Body origin source codes; equal to noodle_polylines.container.BODY_ORIGIN_CODES.
BODY_ORIGIN_MOUTH, BODY_ORIGIN_SUPPORT, BODY_ORIGIN_CENTRE = 0, 1, 2
MOUTH_PEAK_THRESHOLD, MOUTH_PEAK_NMS = 0.15, 8
ORIGIN_SUPPORT_THRESHOLD = 0.3   # body-origin fallback (runtime.body_center_from_maps)
AREA_SUPPORT_THRESHOLD = 0.5     # support area and centroid (the paper re-trace's frame landmarks)

# Hydra's point heatmaps, in channel order from FIRST_LANDMARK_CHANNEL (model.CHANNEL_NAMES).
LANDMARKS = ("mouth", "madreporite", "butthole")
FIRST_LANDMARK_CHANNEL = 5

STAR_FIELDS = ("body_origin_model", "body_origin_code", "support_area_model_px", "support_centroid_model",
               "landmarks_model")
FRAME_TAIL = ("time_s", "fps", "schema_version")


def validate_fps(fps) -> float | None:
    """``None``, or ``fps`` as a finite positive float."""
    if fps is None:
        return None
    if isinstance(fps, bool):
        raise ValueError("fps must be a finite positive number")
    value = float(fps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("fps must be a finite positive number")
    return value


def finalize_frame(record: dict, fps) -> dict:
    """Append ``time_s``, ``fps`` and ``schema_version`` as the record's last keys, in place.

    ``time_s = frame / fps`` (Python float) or ``None`` without a frame rate. Idempotent: the keys are
    popped and re-appended, so a writer that adds keys after an earlier call keeps them last by
    calling it again.
    """
    fps = validate_fps(fps)
    for key in FRAME_TAIL:
        record.pop(key, None)
    record["time_s"] = None if fps is None else int(record["frame"]) / fps
    record["fps"] = fps
    record["schema_version"] = OUTPUT_SCHEMA_VERSION
    return record
