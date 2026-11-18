# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Packaged configuration and externally mounted model defaults."""
from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = Path(os.environ.get("DEEPSTARNOODLES_MODELS", "/models")).expanduser()
DEFAULT_BOTSORT_CONFIG = PACKAGE_ROOT / "config" / "botsort.yaml"
DEFAULT_STARSEG = MODELS_ROOT / "starseg.pt"
DEFAULT_HYDRA_CHECKPOINT = MODELS_ROOT / "hydra.pt"
DEFAULT_TRACER_CHECKPOINT = MODELS_ROOT / "tracer.pt"
