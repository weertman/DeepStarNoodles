# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Session-disjoint splits and out-of-core window datasets over the progress cache.

All four cameras of one recording session observe the same animals at the same
time, so the split unit is the session directory (``trial_104__recording_start-…``)
and every camera of a session lands in the same split. Windows are enumerated
from cheap metadata (timestamps and validity) and rasters are decompressed per
clip on demand, because the full cache does not fit in RAM.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from deepstarnoodles.paths import DATA_ROOT
from deepstarnoodles.progress_cache import (
    DEFAULT_PROGRESS_CACHE_ROOT,
    GridSequence,
    ProgressCacheClip,
    inventory_progress_cache,
    load_progress_cache,
    resample_to_grid,
)
from deepstarnoodles.representations.progress import ProgressWindowConfig

DEFAULT_SPLIT_PATH = DATA_ROOT / "deepstarnoodles" / "progress_vae_splits.json"
_TRIAL_NUMBER = re.compile(r"^(trial_\d+)")


def trial_number(session: str) -> str:
    match = _TRIAL_NUMBER.match(session)
    return match.group(1) if match else session


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    key: str
    camera: str
    session: str
    frames: int
    valid_frames: int

    @property
    def trial(self) -> str:
        return trial_number(self.session)


def scan_cache(root: Path = DEFAULT_PROGRESS_CACHE_ROOT) -> list[CacheEntry]:
    """Read only scalar/1-D members of each npz; rasters stay compressed."""

    entries: list[CacheEntry] = []
    for path in inventory_progress_cache(root):
        try:
            with np.load(path, allow_pickle=False) as payload:
                if str(payload["status"]) != "complete":
                    continue
                valid = np.asarray(payload["frame_valid"], dtype=bool)
                entries.append(
                    CacheEntry(
                        path=path,
                        key=str(payload["key"]),
                        camera=str(payload["camera"]),
                        session=str(payload["trial"]),
                        frames=int(len(valid)),
                        valid_frames=int(valid.sum()),
                    )
                )
        except Exception:  # noqa: BLE001 - a half-written file must not kill a scan
            continue
    return entries


def _session_hash(session: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{session}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def make_session_splits(
    entries: Sequence[CacheEntry],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 20260902,
) -> dict[str, Any]:
    """Greedy, hash-ordered assignment of whole sessions to train/val/test.

    Sessions are visited in a deterministic pseudo-random order and assigned to
    validation, then test, until each holds its share of clips; the remainder is
    training. Every trial number keeps at least one session in training so the
    model sees all animal groups.
    """

    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("fractions must lie in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum below 1")
    per_session: dict[str, int] = {}
    for entry in entries:
        per_session[entry.session] = per_session.get(entry.session, 0) + 1
    total = sum(per_session.values())
    ordered = sorted(per_session, key=lambda name: _session_hash(name, seed))
    trials = {trial_number(name) for name in ordered}
    reserved_for_train = {
        trial: max(
            (name for name in ordered if trial_number(name) == trial),
            key=lambda name: per_session[name],
        )
        for trial in trials
    }
    assignment: dict[str, str] = {}
    counts = {"train": 0, "validation": 0, "test": 0}
    targets = {
        "validation": validation_fraction * total,
        "test": test_fraction * total,
    }
    for name in ordered:
        if name in reserved_for_train.values():
            assignment[name] = "train"
            counts["train"] += per_session[name]
            continue
        placed = False
        for split in ("validation", "test"):
            if counts[split] + per_session[name] * 0.5 <= targets[split]:
                assignment[name] = split
                counts[split] += per_session[name]
                placed = True
                break
        if not placed:
            assignment[name] = "train"
            counts["train"] += per_session[name]
    return {
        "schema_version": 1,
        "unit": "recording_session_all_cameras",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "clip_counts": counts,
        "sessions": {
            split: sorted(name for name, chosen in assignment.items() if chosen == split)
            for split in ("train", "validation", "test")
        },
    }


def save_splits(payload: dict[str, Any], path: Path = DEFAULT_SPLIT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_splits(path: Path = DEFAULT_SPLIT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def entries_for_split(
    entries: Sequence[CacheEntry], splits: dict[str, Any], split: str
) -> list[CacheEntry]:
    wanted = set(splits["sessions"][split])
    return [entry for entry in entries if entry.session in wanted]


class _ClipLRU:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._items: OrderedDict[Path, tuple[ProgressCacheClip, GridSequence]] = OrderedDict()

    def get(
        self, path: Path, window_config: ProgressWindowConfig
    ) -> tuple[ProgressCacheClip, GridSequence]:
        hit = self._items.get(path)
        if hit is not None:
            self._items.move_to_end(path)
            return hit
        cache = load_progress_cache(path)
        grid = resample_to_grid(
            cache,
            cadence_seconds=window_config.cadence_seconds,
            nearest_tolerance=window_config.nearest_tolerance,
        )
        self._items[path] = (cache, grid)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return cache, grid


class ProgressCacheWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Contiguous windows over grid-resampled clips, decompressed per clip."""

    def __init__(
        self,
        entries: Sequence[CacheEntry],
        *,
        context_frames: int = 60,
        window_config: ProgressWindowConfig = ProgressWindowConfig(),
        enumeration_stride: int = 1,
        lru_clips: int = 32,
    ) -> None:
        if context_frames < 2:
            raise ValueError("context_frames must be >= 2")
        if enumeration_stride < 1:
            raise ValueError("enumeration_stride must be >= 1")
        self.entries = list(entries)
        self.context_frames = context_frames
        self.window_config = window_config
        self._lru = _ClipLRU(lru_clips)
        minimum = math.ceil(window_config.minimum_valid_fraction * context_frames)
        clip_of: list[int] = []
        start_of: list[int] = []
        self.clip_window_ranges: list[tuple[int, int]] = []
        for clip_index, entry in enumerate(self.entries):
            grid = self._grid_only(entry.path)
            valid = grid.frame_valid.astype(np.int64)
            count = len(valid)
            first = len(clip_of)
            if count >= context_frames:
                cumulative = np.concatenate(([0], np.cumsum(valid)))
                for start in range(0, count - context_frames + 1, enumeration_stride):
                    if cumulative[start + context_frames] - cumulative[start] >= minimum:
                        clip_of.append(clip_index)
                        start_of.append(start)
            self.clip_window_ranges.append((first, len(clip_of)))
        self.clip_of = np.asarray(clip_of, dtype=np.int64)
        self.start_of = np.asarray(start_of, dtype=np.int64)

    def _grid_only(self, path: Path) -> GridSequence:
        with np.load(path, allow_pickle=False) as payload:
            stub = ProgressCacheClip(
                key=str(payload["key"]),
                camera=str(payload["camera"]),
                trial=str(payload["trial"]),
                stem=str(payload["stem"]),
                path=path,
                frame_ts=np.asarray(payload["frame_ts"], dtype=np.float64),
                frame_valid=np.asarray(payload["frame_valid"], dtype=bool),
                progress=np.zeros((0, 1, 1), dtype=np.uint8),
                support=np.zeros((0, 1, 1), dtype=np.uint8),
                characteristic_scale=float(payload["characteristic_scale"]),
                recipe_digest=str(payload["recipe_digest"]),
            )
        return resample_to_grid(
            stub,
            cadence_seconds=self.window_config.cadence_seconds,
            nearest_tolerance=self.window_config.nearest_tolerance,
        )

    def __len__(self) -> int:
        return len(self.clip_of)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        clip_index = int(self.clip_of[index])
        start = int(self.start_of[index])
        stop = start + self.context_frames
        cache, grid = self._lru.get(self.entries[clip_index].path, self.window_config)
        source = torch.from_numpy(grid.source_index[start:stop])
        valid = torch.from_numpy(grid.frame_valid[start:stop])
        zero = ~valid[:, None, None, None]
        progress = torch.from_numpy(cache.progress).index_select(0, source).unsqueeze(1)
        support = torch.from_numpy(cache.support).index_select(0, source).unsqueeze(1)
        return {
            "progress": progress.masked_fill(zero, 0),
            "support": support.masked_fill(zero, 0),
            "frame_valid": valid,
            "timestamps": torch.from_numpy(grid.timestamps[start:stop]),
            "window_valid_fraction": valid.to(torch.float32).mean(),
            "sequence_index": torch.tensor(clip_index, dtype=torch.int64),
            "start_position": torch.tensor(start, dtype=torch.int64),
        }


class ClipMixBatchSampler(Sampler[list[int]]):
    """Shuffle clips, draw a few windows from each, mix across a clip buffer.

    Keeps the per-clip LRU hot while still giving each batch windows from many
    different clips. ``set_epoch`` reseeds so cycling loaders see new draws.
    """

    def __init__(
        self,
        dataset: ProgressCacheWindowDataset,
        *,
        batch_size: int,
        windows_per_clip: int = 8,
        clips_per_buffer: int = 16,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 1 or windows_per_clip < 1 or clips_per_buffer < 1:
            raise ValueError("sampler sizes must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.windows_per_clip = windows_per_clip
        self.clips_per_buffer = clips_per_buffer
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.nonempty_clips = [
            index
            for index, (first, last) in enumerate(dataset.clip_window_ranges)
            if last > first
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        total = len(self.nonempty_clips) * self.windows_per_clip
        return total // self.batch_size if self.drop_last else math.ceil(total / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        clips = np.asarray(self.nonempty_clips)
        generator.shuffle(clips)
        batch: list[int] = []
        for buffer_start in range(0, len(clips), self.clips_per_buffer):
            pool: list[int] = []
            for clip_index in clips[buffer_start : buffer_start + self.clips_per_buffer]:
                first, last = self.dataset.clip_window_ranges[int(clip_index)]
                draws = generator.integers(first, last, size=self.windows_per_clip)
                pool.extend(int(value) for value in draws)
            generator.shuffle(pool)
            for index in pool:
                batch.append(index)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch and not self.drop_last:
            yield batch


def fixed_validation_indices(
    dataset: ProgressCacheWindowDataset,
    *,
    windows: int,
    seed: int = 0,
) -> list[int]:
    """Deterministic, clip-spread subset ordered by clip so the LRU stays hot."""

    generator = np.random.default_rng(seed)
    nonempty = [
        index for index, (first, last) in enumerate(dataset.clip_window_ranges) if last > first
    ]
    if not nonempty:
        return []
    per_clip = max(1, math.ceil(windows / len(nonempty)))
    chosen: list[int] = []
    for clip_index in nonempty:
        first, last = dataset.clip_window_ranges[clip_index]
        draws = generator.integers(first, last, size=per_clip)
        chosen.extend(int(value) for value in draws)
        if len(chosen) >= windows:
            break
    return chosen[:windows]


__all__ = [
    "CacheEntry",
    "ClipMixBatchSampler",
    "DEFAULT_SPLIT_PATH",
    "ProgressCacheWindowDataset",
    "entries_for_split",
    "fixed_validation_indices",
    "load_splits",
    "make_session_splits",
    "save_splits",
    "scan_cache",
    "trial_number",
]
