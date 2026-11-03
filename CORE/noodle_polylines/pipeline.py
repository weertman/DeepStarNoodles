# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Streaming over many sequences: load one container at a time, compute, yield, forget.

Memory stays bounded by the largest single sequence (times the number of workers). A camera-year
(~3e8 arms) is many files: convert each once to a container ``.npz`` (``save(..., compressed=False)``
lets later loads memory-map it), then run :func:`compute_many` with ``workers`` > 1. The tracker is
sequential within a sequence and independent across sequences, so the work splits by file and the
results equal a serial run's exactly (each sequence is computed by the same code in one process).
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .adapters import load_any
from .container import PolylineContainer
from .presets import Preset, compute_all


def sequence_name(path: str | Path) -> str:
    """The name a sequence is reported under: the file stem, or the clip folder's name."""
    p = Path(path)
    return p.stem if p.is_file() else p.name


def iter_sequences(paths: Iterable[str | Path], **load_kw) -> Iterator[tuple[str, PolylineContainer]]:
    """Yield ``(name, container)`` for container ``.npz`` files, Sleepy segments, inference JSON or
    flipping clip folders, one at a time (``load_kw`` goes to :func:`~noodle_polylines.adapters.load_any`,
    e.g. ``mmap=True`` for uncompressed container files)."""
    for p in paths:
        yield sequence_name(p), load_any(p, **load_kw)


def _compute_one(path: str | Path, preset, compute_kw: dict, load_kw: dict, out_dir: str | None):
    """Load, compute and (optionally) write one sequence: the unit of work of serial and pooled runs."""
    name = sequence_name(path)
    out = compute_all(load_any(path, **load_kw), preset, **compute_kw)
    if out_dir is not None:
        return name, save_outputs(Path(out_dir) / f'{name}.npz', out)
    return name, out


def compute_many(paths: Iterable[str | Path], preset: str | Preset = 'fig5', context: dict | None = None,
                 workers: int = 1, max_pending: int | None = None, out_dir: str | Path | None = None,
                 load_kw: dict | None = None, mp_context: str | None = None,
                 **compute_kw) -> Iterator[tuple[str, dict[str, np.ndarray] | Path]]:
    """Stream :func:`compute_all` over many inputs, yielding ``(name, outputs)`` in input order.

    ``context`` maps a sequence name to extra :func:`compute_all` keywords, its R0 context for example
    (``{'seg_x': {'onset_s': 1.78e9}}``). ``load_kw`` goes to the loader (``mmap=True``, ``fields=...``,
    ``fps=...`` for inference JSON).

    ``workers`` > 1 runs sequences in a process pool (``mp_context``: 'fork', 'spawn' or 'forkserver';
    default the platform's). Results are those of the serial run, bit for bit, and come back in input
    order; at most ``max_pending`` (default ``2 * workers``) sequences are in flight or waiting to be
    yielded, which bounds memory. Pin BLAS threads (``OMP_NUM_THREADS=1``) in the environment before
    starting Python, so ``workers`` processes do not oversubscribe the cores.

    With ``out_dir`` each sequence's outputs are written to ``<out_dir>/<name>.npz``
    (:func:`save_outputs`, pickle-free) by the process that computed them, and the path is yielded
    instead of the arrays, so the parent never holds them.
    """
    context = context or {}
    load_kw = dict(load_kw or {})
    out = None if out_dir is None else str(out_dir)

    def job(p):
        return (p, preset, {**compute_kw, **context.get(sequence_name(p), {})}, load_kw, out)

    if workers <= 1:
        for p in paths:
            yield _compute_one(*job(p))
        return
    window = max(1, max_pending or 2 * workers)
    ctx = multiprocessing.get_context(mp_context) if mp_context else None
    pending: deque = deque()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        try:
            for p in paths:
                pending.append(ex.submit(_compute_one, *job(p)))
                if len(pending) >= window:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:                      # an abandoned or failed stream does not wait for queued work
            for f in pending:
                f.cancel()


def save_outputs(path: str | Path, outputs: dict[str, np.ndarray]) -> Path:
    """Write :func:`compute_all` outputs as a pickle-free ``.npz`` (ragged dicts are skipped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in outputs.items() if not isinstance(v, dict)}
    arrays = {k: v for k, v in arrays.items() if v.dtype != object}
    np.savez_compressed(path, **arrays)
    return path
