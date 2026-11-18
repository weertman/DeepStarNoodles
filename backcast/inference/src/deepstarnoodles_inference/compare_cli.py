# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Stream prediction parity checks without loading inference models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def _sha256(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _failure(kind: str) -> dict:
    from .validation import DEFAULT_TOLERANCES
    return {"pass": False, "frames": 0, "issue_count": 1,
            "issues_first_100": [{"kind": kind}],
            "tolerances": dict(DEFAULT_TOLERANCES),
            "scope": "runtime output parity screening, not task accuracy"}


def _without_local_details(summary: dict) -> dict:
    # Validation's API keeps exception text useful to local Python callers.
    # The portable CLI report contains no source paths or malformed JSON text.
    result = dict(summary)
    result["issues_first_100"] = []
    for issue in summary.get("issues_first_100", [])[:100]:
        item = {"kind": issue["kind"]}
        if type(issue.get("frame")) is int:
            item["frame"] = issue["frame"]
        result["issues_first_100"].append(item)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "Compare streamed prediction records using fixed runtime parity gates; "
        "this does not measure task accuracy."))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional new JSON report; never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and os.path.lexists(args.output):
        print("Comparison output already exists; choose a new output.", file=sys.stderr)
        return 1

    paths = {"reference": args.reference, "candidate": args.candidate}
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if any(value is None for value in hashes.values()):
        parity = _failure("unreadable_input")
    else:
        from .validation import compare
        parity = _without_local_details(compare(args.reference, args.candidate))
        # Hashing and validation remain streaming. Refuse to qualify a pair
        # that changed while being checked instead of attaching stale hashes.
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            parity = _failure("input_changed_during_comparison")
    report = {"schema_version": 1, "input_sha256": hashes, "parity": parity}
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        try:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(serialized)
        except OSError:
            print("Could not create the comparison report; output was not overwritten.", file=sys.stderr)
            return 1
    print(serialized, end="")
    return 0 if parity["pass"] is True else 1
