# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from deepstarnoodles.representations import (
    ArmMotionConfig,
    ProgressCanonicalizationInput,
    ProgressSequenceEncoder,
    ProgressSequenceEncoding,
    ProgressTemporalVAE,
    ProgressTemporalVAEConfig,
    ProgressWindowConfig,
    ProgressWindowDataset,
    arm_bags_from_video_payload,
    arm_motion,
    load_representation_sidecar,
    save_representation_sidecar,
    tracked_frames_from_video_payload,
)


def _arm(track_id: int, offset: float = 0.0) -> dict:
    return {
        "track_id": track_id,
        "polyline_crop": [
            [0.0, float(track_id)],
            [5.0, float(track_id)],
            [10.0 + offset, float(track_id)],
        ],
        "polyline_source": [
            [100.0, 200.0 + track_id],
            [105.0, 200.0 + track_id],
            [110.0 + offset, 200.0 + track_id],
        ],
        "identity": [float(track_id), 0.0],
    }


def _payload() -> dict:
    return {
        "video": "example.mp4",
        "frames": [
            {"frame": index, "detected": True, "arms": [_arm(0, index), _arm(1)]}
            for index in range(3)
        ],
    }


def test_tracking_payload_adapters_supply_timestamps_and_native_arms() -> None:
    payload = _payload()
    frames = tracked_frames_from_video_payload(payload, fps=2.0)
    bags = arm_bags_from_video_payload(payload)

    assert [frame.timestamp for frame in frames] == [0.0, 0.5, 1.0]
    assert all(frame.observed for frame in frames)
    assert frames[0].arms[0].polyline[0].tolist() == [0.0, 0.0]
    assert len(bags) == 3
    assert all(bag.values.shape == (102,) for bag in bags)


def test_multi_star_payload_requires_and_respects_star_scope() -> None:
    payload = {
        "schema_version": 2,
        "frames": [
            {
                "frame": 0,
                "detected": True,
                "stars": [
                    {"star_track_id": 3, "arms": [_arm(0)]},
                    {"star_track_id": 9, "arms": [_arm(4)]},
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="star_track_id"):
        tracked_frames_from_video_payload(payload, fps=1.0)
    selected = tracked_frames_from_video_payload(
        payload, fps=1.0, star_track_id=9
    )
    assert [arm.track_id for arm in selected[0].arms] == [4]


def test_sidecar_round_trip_is_pickle_free_and_versioned(tmp_path) -> None:
    payload = _payload()
    bags = arm_bags_from_video_payload(payload)
    frames = tracked_frames_from_video_payload(payload, fps=1.0)
    motion = arm_motion(
        frames,
        ArmMotionConfig(
            lags_seconds=(1,),
            min_track_observations=2,
            min_matched_arms=2,
        ),
    )
    temporal = ProgressSequenceEncoding(
        mean=torch.zeros(3, 4),
        logvar=torch.zeros(3, 4),
        valid=torch.tensor([False, True, False]),
        window_valid_fraction=torch.tensor([0.5, 1.0, 0.5]),
        timestamps=torch.arange(3, dtype=torch.float64),
        offsets=torch.tensor([-1.0, 0.0]),
        recipe_digest="a" * 64,
        checkpoint_sha256="b" * 64,
        checkpoint_metadata={"purpose": "test"},
    )
    path = save_representation_sidecar(
        tmp_path / "representations.npz",
        arm_bags=bags,
        arm_motion=motion,
        temporal=temporal,
        metadata={"video": "example.mp4"},
    )
    arrays = load_representation_sidecar(path)
    manifest = json.loads(str(arrays["manifest_json"].item()))

    assert int(arrays["schema_version"]) == 1
    assert arrays["arm_bag_values"].shape == (3, 102)
    assert arrays["arm_motion_values"].shape == (3, 1, 5)
    assert arrays["temporal_mean"].shape == (3, 4)
    assert manifest["metadata"]["video"] == "example.mp4"


def test_sequence_encoder_accepts_raw_hydra_frames_directly() -> None:
    config = ProgressTemporalVAEConfig(
        context_frames=4,
        resolution=16,
        latent_dim=3,
        frame_token_dim=6,
        temporal_dim=8,
        base_channels=2,
        max_channels=4,
        transformer_layers=1,
        transformer_heads=2,
        dropout=0.0,
    )
    encoder = ProgressSequenceEncoder(
        ProgressTemporalVAE(config),
        window_config=ProgressWindowConfig(
            minimum_valid_fraction=1.0,
            nearest_tolerance=0.1,
        ),
        amp=False,
    )
    raw = torch.full((8, 5, 16, 16), -10.0)
    raw[:, 0, 4:12, 4:12] = 10.0
    raw[:, 4, 4:12, 4:12] = 0.0
    geometry = ProgressCanonicalizationInput.source_axes(
        [[0.0, 0.0, 16.0]] * 8,
        model_size=16,
        characteristic_scale=[6.0] * 8,
    )
    encoded = encoder.encode_hydra(
        raw,
        torch.arange(8, dtype=torch.float64),
        geometry,
    )

    assert encoded.mean.shape == (8, 3)
    assert encoded.offsets.tolist() == [-2.0, -1.0, 0.0, 1.0]
    assert encoded.valid.sum() == 5


def test_training_windows_reject_nonchronological_timestamps() -> None:
    progress = torch.zeros(5, 1, 16, 16)
    with pytest.raises(ValueError, match="strictly increasing"):
        ProgressWindowDataset(
            progress,
            torch.ones_like(progress),
            torch.ones(5, dtype=torch.bool),
            timestamps=torch.tensor([0.0, 1.0, 1.0, 3.0, 4.0]),
            context_frames=4,
        )

    split_at_gap = ProgressWindowDataset(
        progress,
        torch.ones_like(progress),
        torch.ones(5, dtype=torch.bool),
        timestamps=torch.tensor([0.0, 1.0, 10.0, 11.0, 12.0]),
        context_frames=3,
        config=ProgressWindowConfig(
            cadence_seconds=1.0,
            nearest_tolerance=0.1,
            minimum_valid_fraction=1.0,
        ),
    )
    assert len(split_at_gap) == 1
    assert split_at_gap[0]["start_position"].item() == 2
