#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Train or fine-tune the 18-channel DeepStarNoodles dense model."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

from deepstarnoodles.train import main

if __name__ == "__main__":
    raise SystemExit(main() or 0)
