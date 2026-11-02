# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Streaming over many sequences: load one container at a time, compute, yield, forget.

Memory stays bounded by the largest single sequence. For a camera-year (~3e8 arms) split the work by
file and run :func:`compute_many` in several processes (the tracker is sequential within a sequence,
independent across sequences).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .adapters import load_any
from .container import PolylineContainer
from .presets import Preset, compute_all


def iter_sequences(paths: Iterable[str | Path], **load_kw) -> Iterator[tuple[str, PolylineContainer]]:
    """Yield ``(name, container)`` for container ``.npz`` files, Sleepy segments or flipping clip folders."""
    for p in paths:
        p = Path(p)
        yield p.stem if p.is_file() else p.name, load_any(p, **load_kw)


def compute_many(paths: Iterable[str | Path], preset: str | Preset = 'fig5', context: dict | None = None,
                 **compute_kw) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    """Stream :func:`compute_all` over many inputs. ``context`` maps a sequence name to its R0 context
    (e.g. ``{'seg_x': {'onset_s': 1.78e9}}``)."""
    context = context or {}
    for name, c in iter_sequences(paths):
        yield name, compute_all(c, preset, **{**compute_kw, **context.get(name, {})})


def save_outputs(path: str | Path, outputs: dict[str, np.ndarray]) -> Path:
    """Write :func:`compute_all` outputs as a pickle-free ``.npz`` (ragged dicts are skipped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in outputs.items() if not isinstance(v, dict)}
    arrays = {k: v for k, v in arrays.items() if v.dtype != object}
    np.savez_compressed(path, **arrays)
    return path
