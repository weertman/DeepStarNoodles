# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""The installed vendored noodle_polylines and its test data match the hashes pinned in UPSTREAM.json.

In CI the tests run against the installed wheel, so this also checks that the wheel carries the
complete, unmodified copy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from deepstarnoodles_inference._vendor import noodle_polylines

TESTS = Path(__file__).resolve().parent
UPSTREAM = json.loads((TESTS.parent / "UPSTREAM.json").read_text())["noodle_polylines"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_installed_vendored_files_match_the_pinned_upstream_set():
    installed = Path(noodle_polylines.__file__).resolve().parent
    pinned = {Path(item["source"]).name: item["sha256"] for item in UPSTREAM["files"]}
    assert all(item["source"].startswith("CORE/noodle_polylines/") for item in UPSTREAM["files"])
    assert sorted(path.name for path in installed.glob("*.py")) == sorted(pinned)
    assert {name: sha256(installed / name) for name in pinned} == pinned
    assert len(UPSTREAM["source_commit"]) == 40


def test_polyline_fixtures_match_the_pinned_upstream_data():
    for item in UPSTREAM["test_data"]:
        path = TESTS.parent / item["destination"]
        assert sha256(path) == item["sha256"], item["destination"]
    assert sorted(path.name for path in (TESTS / "fixtures" / "polylines").iterdir()) == sorted(
        Path(item["destination"]).name for item in UPSTREAM["test_data"])
