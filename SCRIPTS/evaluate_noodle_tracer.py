#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Evaluate locked noodle tracer against a frozen-output study cache."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

from deepstarnoodles.evaluate_noodle_tracer import main

if __name__ == "__main__":
    raise SystemExit(main())
