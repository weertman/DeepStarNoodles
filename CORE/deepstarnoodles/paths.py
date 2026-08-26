# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Repository-relative defaults shared by supported command-line tools."""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get(
        "DEEPSTARNOODLES_ROOT",
        Path(__file__).resolve().parents[2],
    )
).expanduser().resolve()

DATA_ROOT = REPO_ROOT / "DATA"
RUNS_ROOT = REPO_ROOT / "RUNS"
MODELS_ROOT = REPO_ROOT / "MODELS"

DEFAULT_DENSE_INDEX = (
    DATA_ROOT / "deepstarnoodles" / "deepstarnoodles_dense_train_index.csv"
)
DEFAULT_SPLITS_ROOT = DATA_ROOT / "deepstarnoodles" / "splits"
DEFAULT_TRAIN_INDEX = DEFAULT_SPLITS_ROOT / "train_index.csv"
DEFAULT_VALIDATION_INDEX = DEFAULT_SPLITS_ROOT / "validation_index.csv"
DEFAULT_TEST_INDEX = DEFAULT_SPLITS_ROOT / "test_index.csv"
DEFAULT_ANNOTATED_INDEX = DEFAULT_DENSE_INDEX
DEFAULT_IGNORE_SHAPES_DIR = DATA_ROOT / "deepstarnoodles" / "ignore_masks"
DEFAULT_STARSEG_DATA = DATA_ROOT / "starseg" / "data.yaml"

DEFAULT_DINOV3_REPO = REPO_ROOT / "external" / "dinov3"
DEFAULT_DINOV3_CONVNEXT_BASE_CHECKPOINT = (
    MODELS_ROOT / "dinov3" / "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
)
DEFAULT_STARSEG = MODELS_ROOT / "starseg" / "starseg_best.pt"
DEFAULT_HYDRA_PARENT_CHECKPOINT = MODELS_ROOT / "hydra" / "epoch_060.pt"
DEFAULT_HYDRA_CHECKPOINT = MODELS_ROOT / "hydra" / "endpoint_mse_best.pt"
DEFAULT_TRACER_CHECKPOINT = (
    MODELS_ROOT
    / "tracer"
    / "noodle_tracer_seed20260807_endpoint_mse_calibrated.pt"
)
DEFAULT_COMBINED_CHECKPOINT = DEFAULT_HYDRA_CHECKPOINT
