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

_PROJECT_PREFIXES = (
    "DATA/",
    "MODELS/",
    "RUNS/",
    "config/",
    "CORE/",
    "SCRIPTS/",
    "external/",
    "tests/",
    "huggingface/",
)
_THREE_CLASS_MARKER = "yolo26_three_class_train_dataset/"
_TRAIN_DATASET_MARKER = "yolo26_train_dataset/"
_CVAT_IMAGE_MARKER = "latest_cvat_export/images/default/"


def _posix(value: str | Path) -> str:
    return str(value).strip().replace("\\", "/")


def to_repo_relative(value: str | Path, *, root: Path | None = None) -> str:
    """Return the stored project path (posix, relative to the repo root).

    Accepts already-relative ``DATA/...`` strings, absolute checkouts of this
    tree, and the historical foreign prefixes used before the public extract.
    Empty input stays empty so unannotated rows can omit ``shapes_path``.
    """
    text = _posix(value)
    if not text:
        return ""
    root = (root or REPO_ROOT).resolve()
    stripped = text.lstrip("./")
    if stripped.startswith(_PROJECT_PREFIXES):
        return stripped

    candidate = Path(text).expanduser()
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        pass

    root_posix = root.as_posix().rstrip("/")
    if text == root_posix:
        return ""
    if text.startswith(root_posix + "/"):
        return text[len(root_posix) + 1 :]

    for marker in ("/DATA/deepstarnoodles/", "/DATA/starseg/"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx + 1 :]

    idx = text.find(_THREE_CLASS_MARKER)
    if idx >= 0:
        rest = text[idx + len(_THREE_CLASS_MARKER) :]
        return f"DATA/starseg/{rest}" if rest else "DATA/starseg"
    if text.rstrip("/").endswith("yolo26_three_class_train_dataset"):
        return "DATA/starseg"

    idx = text.find(_TRAIN_DATASET_MARKER)
    if idx >= 0:
        rest = text[idx + len(_TRAIN_DATASET_MARKER) :]
        if rest.startswith(("images/", "labels/")):
            return f"DATA/starseg/{rest}"
        return "yolo26_train_dataset"

    if text.rstrip("/").endswith("yolo26_train_dataset"):
        return "yolo26_train_dataset"

    idx = text.find(_CVAT_IMAGE_MARKER)
    if idx >= 0:
        name = text[idx + len(_CVAT_IMAGE_MARKER) :]
        return f"DATA/deepstarnoodles/images/{name}"

    raise ValueError(f"cannot map path into the repository: {value!r}")


def resolve_repo_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a stored project path to an absolute filesystem location."""
    relative = to_repo_relative(value, root=root)
    if not relative:
        raise ValueError("empty path")
    root = (root or REPO_ROOT).resolve()
    if relative == "yolo26_train_dataset":
        raise ValueError("upstream StarSeg train set is not shipped in this extract")
    return (root / relative).resolve()

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
DEFAULT_BOTSORT_CONFIG = REPO_ROOT / "config" / "star_tracking" / "botsort.yaml"

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
    / "noodle_tracer_fullcorpus_seed20261011_calibrated.pt"
)
DEFAULT_TEMPORAL_VAE_CHECKPOINT = (
    MODELS_ROOT
    / "temporal_vae"
    / "dnt_progress_temporal_vae_60s_sleepy.pt"
)
DEFAULT_COMBINED_CHECKPOINT = DEFAULT_HYDRA_CHECKPOINT
