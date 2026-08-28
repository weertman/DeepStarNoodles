# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""DeepStarNoodles: dense sunflower-star pose, Noodle Tracer, and video tracking."""

from deepstarnoodles.model import CHANNEL_NAMES, HydraModel
from deepstarnoodles.noodle_tracer import NoodleTracer, load_noodle_tracer_checkpoint
from deepstarnoodles.paths import REPO_ROOT
from deepstarnoodles.representations import (
    ArmBagConfig,
    ArmMotionConfig,
    ProgressSequenceEncoder,
    ProgressTemporalVAE,
    ProgressTemporalVAEConfig,
    arm_bag,
    arm_motion,
    load_progress_temporal_vae_checkpoint,
)

__all__ = [
    "ArmBagConfig",
    "ArmMotionConfig",
    "CHANNEL_NAMES",
    "HydraModel",
    "NoodleTracer",
    "ProgressSequenceEncoder",
    "ProgressTemporalVAE",
    "ProgressTemporalVAEConfig",
    "REPO_ROOT",
    "arm_bag",
    "arm_motion",
    "load_noodle_tracer_checkpoint",
    "load_progress_temporal_vae_checkpoint",
]
