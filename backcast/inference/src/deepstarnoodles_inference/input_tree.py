# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Finding record files in directories, for ``track-videos`` and ``representations``.

Both commands search their directory arguments the same way:

- Directory symlinks are followed (each real directory once), so a camera linked into the tree is not left out.
- A subdirectory holding a ``track-videos`` run log is another run's tracked output, and is skipped with a note; pass
  it on its own to read it. The directory given on the command line is never skipped this way.
- Files whose names start with ``_`` are the fleet's and ``track-videos``' own files (``_manifest.jsonl``,
  ``_track-videos.log``). Any other record file starting with ``_`` is skipped with a note.

The fleet's ``_manifest.jsonl`` lists how many records it fetched for each video. :func:`expected_records` finds that
count, so a file cut exactly at a record boundary, which reads as a complete shorter video, is still caught.
"""
from __future__ import annotations

from collections.abc import Callable
import functools
import json
import os
from pathlib import Path

FLEET_MANIFEST = "_manifest.jsonl"
RUN_LOG = "_track-videos.log"       # one JSON line per video tracked into an --output directory
RUN_LOGS = (RUN_LOG, "_track-videos.jsonl")     # and the name 0.5.0.dev13 pre-releases gave it
OWN_FILES = (FLEET_MANIFEST, *RUN_LOGS)
RECORD_SUFFIXES = (".jsonl", ".jsonl.gz", ".jsonl.bz2", ".jsonl.xz", ".jsonl.zst", ".json", ".json.gz",
                   ".json.bz2", ".json.xz", ".json.zst")


def walk(root: Path, *, exclude: Path | None = None) -> tuple[list[Path], list[str]]:
    """Every file under ``root`` (as reached through ``root``, sorted) and notes on what was skipped.

    Directory symlinks are followed once per real directory. ``exclude`` (an output directory) is never entered.
    """
    root = Path(root)
    excluded = os.path.realpath(exclude) if exclude is not None else None
    seen = {os.path.realpath(root)}
    files: list[Path] = []
    notes: list[str] = []
    for directory, subdirectories, names in os.walk(root, followlinks=True):
        keep = []
        for name in sorted(subdirectories):
            path = Path(directory) / name
            real = os.path.realpath(path)
            if real == excluded or real in seen:
                continue
            log = next((name for name in RUN_LOGS if (path / name).is_file()), None)
            if log is not None:
                notes.append(f"skipped {path}: it holds track-videos output ({log}); pass it on its own to read it")
                continue
            seen.add(real)
            keep.append(name)
        subdirectories[:] = keep
        files.extend(Path(directory) / name for name in names)
    return sorted(files), notes


def record_files(root: Path, accept: Callable[[Path], bool], *,
                 exclude: Path | None = None) -> tuple[list[Path], list[str]]:
    """Record files under ``root`` that ``accept`` takes, skipping ``_``-prefixed ones; and notes."""
    files, notes = walk(root, exclude=exclude)
    found, underscored = [], []
    for path in files:
        if not path.name.lower().endswith(RECORD_SUFFIXES) or not accept(path):
            continue
        if path.name.startswith("_"):
            if path.name not in OWN_FILES:
                underscored.append(path)
            continue
        found.append(path)
    if underscored:
        listed = ", ".join(str(p) for p in underscored[:5]) + (" ..." if len(underscored) > 5 else "")
        notes.append(f"skipped {len(underscored)} record file(s) whose names start with '_' (reserved for the "
                     f"fleet's and track-videos' own files): {listed}; pass a file by name to read it")
    return found, notes


def _strip_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in sorted(RECORD_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix):
            return name[:-len(suffix)]
    return name


@functools.lru_cache(maxsize=64)
def _manifest_rows(path: str, mtime_ns: int, size: int) -> dict[str, int]:
    rows: dict[str, int] = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                     # a half-written last line while the fleet is still running
            if isinstance(row, dict) and isinstance(row.get("relative"), str) and isinstance(row.get("frames"), int):
                rows[row["relative"]] = row["frames"]     # a later row for the same video wins
    return rows


def expected_records(path: Path) -> tuple[int, Path] | None:
    """``(records, manifest)`` from the nearest ``_manifest.jsonl`` above ``path`` that lists it, else ``None``."""
    path = Path(path).absolute()
    for directory in path.parents:
        manifest = directory / FLEET_MANIFEST
        if not manifest.is_file():
            continue
        stat = manifest.stat()
        relative = _strip_suffix(path.relative_to(directory).as_posix())
        count = _manifest_rows(str(manifest), stat.st_mtime_ns, stat.st_size).get(relative)
        if count is not None:
            return count, manifest
    return None


def check_record_count(path: Path, records: int) -> None:
    """Raise ``ValueError`` when the fleet's manifest lists another record count for ``path``."""
    expected = expected_records(path)
    if expected is not None and expected[0] != records:
        raise ValueError(f"{path} holds {records} records, but {expected[1]} lists {expected[0]} for it; the file "
                         "is truncated or incomplete (move the manifest aside if the video was annotated again "
                         "elsewhere)")
