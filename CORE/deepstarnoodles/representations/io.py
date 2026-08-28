# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Adapters and compact sidecar I/O for representation outputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any, Literal
import uuid

import numpy as np

from .arm_bag import ArmBagConfig, ArmBagResult, arm_bag_sequence
from .arm_motion import ArmMotionResult, TrackedArmFrame, tracked_arm_frame
from .temporal_vae import ProgressSequenceEncoding


SIDECAR_SCHEMA_VERSION = 1
_POLYLINE_SPACES = ("crop", "source")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _selected_star(
    frame: Mapping[str, Any],
    star_track_id: int | None,
) -> Mapping[str, Any] | None:
    stars = frame.get("stars")
    if stars is None:
        return frame
    if not isinstance(stars, Sequence) or isinstance(stars, (str, bytes)):
        raise ValueError("frame stars must be a sequence")
    if star_track_id is None:
        raise ValueError(
            "star_track_id is required for multi-star payload frames"
        )
    matches = [
        star
        for star in stars
        if isinstance(star, Mapping)
        and int(star.get("star_track_id", -1)) == int(star_track_id)
    ]
    if len(matches) > 1:
        raise ValueError("frame contains a duplicate star_track_id")
    return None if not matches else matches[0]


def _arm_records(
    frame: Mapping[str, Any],
    *,
    star_track_id: int | None,
    polyline_space: Literal["crop", "source"],
) -> tuple[Mapping[str, Any], ...]:
    selected = _selected_star(frame, star_track_id)
    if selected is None:
        return ()
    arms = selected.get("arms", ())
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)):
        raise ValueError("frame arms must be a sequence")
    preferred = f"polyline_{polyline_space}"
    result = []
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise ValueError("each arm record must be a mapping")
        if preferred not in arm and "polyline" not in arm:
            raise ValueError(f"arm record lacks {preferred!r} or 'polyline'")
        copied = dict(arm)
        copied["polyline"] = copied.get(preferred, copied.get("polyline"))
        result.append(copied)
    return tuple(result)


def tracked_frames_from_video_payload(
    payload: Mapping[str, Any],
    *,
    fps: float | None = None,
    star_track_id: int | None = None,
    polyline_space: Literal["crop", "source"] = "crop",
) -> tuple[TrackedArmFrame, ...]:
    """Adapt a DeepStarNoodles tracking payload for :func:`arm_motion`.

    Existing payloads do not always contain timestamps. In that case ``fps``
    must be supplied (or present at the payload top level) and timestamps are
    derived from ``frame / fps``.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if polyline_space not in _POLYLINE_SPACES:
        raise ValueError(f"polyline_space must be one of {_POLYLINE_SPACES}")
    frames = payload.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("payload frames must be a sequence")
    selected_fps = payload.get("fps") if fps is None else fps
    if selected_fps is not None:
        selected_fps = float(selected_fps)
        if not np.isfinite(selected_fps) or selected_fps <= 0:
            raise ValueError("fps must be finite and positive")

    adapted: list[TrackedArmFrame] = []
    for position, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError("each frame must be a mapping")
        if "timestamp" in frame:
            timestamp = float(frame["timestamp"])
        else:
            if selected_fps is None:
                raise ValueError("fps is required when frame timestamps are absent")
            frame_index = int(frame.get("frame", position))
            timestamp = frame_index / selected_fps
        selected = _selected_star(frame, star_track_id)
        arms = _arm_records(
            frame,
            star_track_id=star_track_id,
            polyline_space=polyline_space,
        )
        body_center = None
        if selected is not None:
            for key in (
                f"body_center_{polyline_space}",
                "body_center",
                "center",
            ):
                if key in selected:
                    body_center = selected[key]
                    break
        record = {
            "timestamp": timestamp,
            "observed": bool(selected is not None and frame.get("detected", True)),
            "arms": arms,
            "body_center": body_center,
        }
        adapted.append(tracked_arm_frame(record))
    return tuple(adapted)


def arm_bags_from_video_payload(
    payload: Mapping[str, Any],
    config: ArmBagConfig = ArmBagConfig(),
    *,
    star_track_id: int | None = None,
    polyline_space: Literal["crop", "source"] = "crop",
) -> tuple[ArmBagResult, ...]:
    """Build one static arm bag per frame from a tracking payload."""

    frames = payload.get("frames") if isinstance(payload, Mapping) else None
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("payload frames must be a sequence")
    arms = [
        _arm_records(
            frame,
            star_track_id=star_track_id,
            polyline_space=polyline_space,
        )
        for frame in frames
    ]
    return arm_bag_sequence(arms, config)


def _frame_count(
    arm_bags: Sequence[ArmBagResult] | None,
    arm_motion: ArmMotionResult | None,
    temporal: ProgressSequenceEncoding | None,
) -> int:
    counts = []
    if arm_bags is not None:
        counts.append(len(arm_bags))
    if arm_motion is not None:
        counts.append(len(arm_motion.timestamps))
    if temporal is not None:
        counts.append(len(temporal.timestamps))
    if not counts:
        raise ValueError("at least one representation result is required")
    if len(set(counts)) != 1:
        raise ValueError("representation results must share a frame count")
    return counts[0]


def save_representation_sidecar(
    path: str | Path,
    *,
    arm_bags: Sequence[ArmBagResult] | None = None,
    arm_motion: ArmMotionResult | None = None,
    temporal: ProgressSequenceEncoding | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save aligned representation arrays as a pickle-free NPZ."""

    frame_count = _frame_count(arm_bags, arm_motion, temporal)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SIDECAR_SCHEMA_VERSION, dtype=np.int16),
        "frames": np.asarray(frame_count, dtype=np.int64),
    }
    manifest: dict[str, Any] = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "frames": frame_count,
        "metadata": _json_value(dict(metadata or {})),
        "representations": {},
    }

    if arm_bags is not None:
        bags = tuple(arm_bags)
        if bags:
            feature_names = bags[0].feature_names
            representation = bags[0].representation
            if any(
                bag.feature_names != feature_names
                or bag.representation != representation
                for bag in bags
            ):
                raise ValueError("arm bags must share a feature contract")
            arrays["arm_bag_values"] = np.stack([bag.values for bag in bags])
            arrays["arm_bag_valid"] = np.asarray(
                [bag.valid for bag in bags], dtype=bool
            )
            arrays["arm_bag_count"] = np.asarray(
                [bag.arm_count for bag in bags], dtype=np.int16
            )
            arrays["arm_bag_feature_names"] = np.asarray(feature_names)
            arrays["arm_bag_recipe_digest"] = np.asarray(
                [bag.recipe_digest for bag in bags]
            )
            manifest["representations"]["arm_bag"] = {
                "representation": representation,
                "feature_names": list(feature_names),
                "recipes": sorted({bag.recipe_digest for bag in bags}),
                "recipe_metadata": _json_value(bags[0].recipe_metadata),
            }
        else:
            manifest["representations"]["arm_bag"] = {"empty": True}

    if arm_motion is not None:
        arrays.update(
            {
                "arm_motion_values": np.asarray(arm_motion.values),
                "arm_motion_strict_values": np.asarray(arm_motion.strict_values),
                "arm_motion_valid": np.asarray(arm_motion.valid),
                "arm_motion_strict_valid": np.asarray(arm_motion.strict_valid),
                "arm_motion_matched_counts": np.asarray(
                    arm_motion.matched_counts
                ),
                "arm_motion_strict_matched_counts": np.asarray(
                    arm_motion.strict_matched_counts
                ),
                "arm_motion_identity_distance": np.asarray(
                    arm_motion.identity_distance
                ),
                "arm_motion_retained_fraction": np.asarray(
                    arm_motion.retained_fraction
                ),
                "arm_motion_timestamps": np.asarray(arm_motion.timestamps),
                "arm_motion_lags_seconds": np.asarray(
                    arm_motion.lags_seconds
                ),
                "arm_motion_feature_names": np.asarray(
                    arm_motion.feature_names
                ),
                "arm_motion_sequence_scale": np.asarray(
                    arm_motion.sequence_scale, dtype=np.float64
                ),
            }
        )
        manifest["representations"]["arm_motion"] = {
            "representation": arm_motion.representation,
            "recipe_digest": arm_motion.recipe_digest,
            "recipe_metadata": _json_value(arm_motion.recipe_metadata),
        }

    if temporal is not None:
        arrays.update(
            {
                "temporal_mean": temporal.mean.detach().cpu().numpy(),
                "temporal_logvar": temporal.logvar.detach().cpu().numpy(),
                "temporal_valid": temporal.valid.detach().cpu().numpy(),
                "temporal_window_valid_fraction": (
                    temporal.window_valid_fraction.detach().cpu().numpy()
                ),
                "temporal_timestamps": temporal.timestamps.detach().cpu().numpy(),
                "temporal_offsets": temporal.offsets.detach().cpu().numpy(),
            }
        )
        manifest["representations"]["progress_temporal_vae"] = {
            "representation": temporal.representation,
            "recipe_digest": temporal.recipe_digest,
            "checkpoint_sha256": temporal.checkpoint_sha256,
            "metadata": _json_value(temporal.checkpoint_metadata),
        }

    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def load_representation_sidecar(path: str | Path) -> dict[str, np.ndarray]:
    """Load and minimally validate a representation NPZ sidecar."""

    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    version = int(np.asarray(arrays.get("schema_version", -1)).item())
    if version != SIDECAR_SCHEMA_VERSION:
        raise ValueError("unsupported representation sidecar schema")
    if "manifest_json" not in arrays:
        raise ValueError("representation sidecar lacks a manifest")
    manifest = json.loads(str(np.asarray(arrays["manifest_json"]).item()))
    if int(manifest.get("schema_version", -1)) != SIDECAR_SCHEMA_VERSION:
        raise ValueError("representation sidecar manifest mismatch")
    return arrays


__all__ = [
    "SIDECAR_SCHEMA_VERSION",
    "arm_bags_from_video_payload",
    "load_representation_sidecar",
    "save_representation_sidecar",
    "tracked_frames_from_video_payload",
]
