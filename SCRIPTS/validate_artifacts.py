#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Validate required local artifacts against config/artifacts.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "artifacts.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--roles",
        nargs="*",
        default=None,
        help="Validate only these roles (default: every manifest entry).",
    )
    parser.add_argument(
        "--require-local",
        action="store_true",
        help="Do not accept an entry's external archive_path fallback.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Fail when optional entries such as yolo26l-seg.pt are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    failures: list[str] = []
    checked = 0

    for item in manifest["artifacts"]:
        if args.roles and item["role"] not in args.roles:
            continue
        local_path = REPO_ROOT / item["path"]
        archive_path = (
            Path(item["archive_path"]).expanduser()
            if item.get("archive_path")
            else None
        )
        if item["role"] == "external-archive" and archive_path and not args.require_local:
            candidates = [archive_path, local_path]
        else:
            candidates = [local_path]
            if archive_path and not args.require_local:
                candidates.append(archive_path)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            if item.get("optional") and not args.include_optional:
                print(f"skip {item['id']}: optional, missing")
                continue
            failures.append(f"{item['id']}: missing ({', '.join(map(str, candidates))})")
            continue
        checked += 1
        expected = item.get("sha256")
        if expected and path.is_file():
            actual = sha256(path)
            if actual != expected:
                failures.append(
                    f"{item['id']}: checksum mismatch ({actual} != {expected})"
                )
                continue
        if item["role"] == "external-archive" and path.is_dir():
            files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
            expected_count = item.get("file_count")
            expected_bytes = item.get("bytes")
            actual_bytes = sum(candidate.stat().st_size for candidate in files)
            if expected_count is not None and len(files) != expected_count:
                failures.append(
                    f"{item['id']}: file count mismatch ({len(files)} != {expected_count})"
                )
                continue
            if expected_bytes is not None and actual_bytes != expected_bytes:
                failures.append(
                    f"{item['id']}: size mismatch ({actual_bytes} != {expected_bytes})"
                )
                continue
        print(f"ok  {item['id']}: {path}")

    if failures:
        print("\nArtifact validation failed:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print(f"\nValidated {checked} artifact entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
