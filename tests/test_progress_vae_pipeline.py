# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Tests for the archive progress cache, session splits, and behavior metrics."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from deepstarnoodles.progress_cache import ProgressCacheClip, resample_to_grid
from deepstarnoodles.progress_metrics import (
    LOG_2PI_E,
    fit_standardization,
    latent_speed,
    posterior_kl,
    segment_median,
    token_entropy_metrics,
    valid_runs,
)
from deepstarnoodles.progress_vae_data import (
    CacheEntry,
    ClipMixBatchSampler,
    ProgressCacheWindowDataset,
    fixed_validation_indices,
    make_session_splits,
    trial_number,
)


def _clip(frames: int, cadence: float, *, invalid: set[int] = frozenset()) -> ProgressCacheClip:
    valid = np.ones(frames, dtype=bool)
    for index in invalid:
        valid[index] = False
    return ProgressCacheClip(
        key="A:clip",
        camera="A",
        trial="trial_104__recording_start-20260823T123820688",
        stem="clip",
        path=Path("clip.npz"),
        frame_ts=1000.0 + cadence * np.arange(frames),
        frame_valid=valid,
        progress=np.full((frames, 8, 8), 200, dtype=np.uint8),
        support=np.full((frames, 8, 8), 255, dtype=np.uint8),
        characteristic_scale=100.0,
        recipe_digest="digest",
    )


def test_resample_to_grid_absorbs_recorder_drift() -> None:
    clip = _clip(578, 1.025)
    grid = resample_to_grid(clip)
    assert len(grid.timestamps) == 592
    assert np.allclose(np.diff(grid.timestamps), 1.0)
    assert grid.frame_valid.all()
    assert np.all(np.abs(clip.frame_ts[grid.source_index] - grid.timestamps) <= 0.55)
    assert grid.source_index[-1] == 577


def test_resample_to_grid_marks_gaps_and_invalid_frames() -> None:
    frame_ts = np.concatenate((np.arange(0, 10.0), np.arange(15.0, 25.0)))
    clip = _clip(20, 1.0, invalid={3})
    clip = ProgressCacheClip(**{**clip.__dict__, "frame_ts": frame_ts})
    grid = resample_to_grid(clip)
    assert len(grid.timestamps) == 25
    assert not grid.frame_valid[3]
    assert not grid.frame_valid[10:15].any()
    assert grid.frame_valid[9] and grid.frame_valid[15]


def test_trial_number_strips_session_suffix() -> None:
    assert trial_number("trial_104__recording_start-20260823T123820688") == "trial_104"
    assert trial_number("odd") == "odd"


def _entries() -> list[CacheEntry]:
    entries: list[CacheEntry] = []
    for trial, sessions in (("trial_103", 3), ("trial_104", 8), ("trial_110", 2)):
        for session in range(sessions):
            name = f"{trial}__recording_start-2026080{session}T000000000"
            for camera in "ABCD":
                for clip in range(5 + session):
                    entries.append(
                        CacheEntry(
                            path=Path(f"{camera}/{name}/{clip}.npz"),
                            key=f"{camera}:{clip}",
                            camera=camera,
                            session=name,
                            frames=600,
                            valid_frames=590,
                        )
                    )
    return entries


def test_session_splits_keep_cameras_together_and_trials_in_train() -> None:
    entries = _entries()
    splits = make_session_splits(entries, validation_fraction=0.15, test_fraction=0.15)
    assigned = {name: split for split, names in splits["sessions"].items() for name in names}
    assert set(assigned) == {entry.session for entry in entries}
    for split in ("validation", "test"):
        assert splits["clip_counts"][split] > 0
    train_trials = {trial_number(name) for name in splits["sessions"]["train"]}
    assert train_trials == {"trial_103", "trial_104", "trial_110"}
    total = len(entries)
    assert splits["clip_counts"]["validation"] <= 0.15 * total + 0.5 * max(
        sum(1 for entry in entries if entry.session == name) for name in assigned
    )
    again = make_session_splits(entries, validation_fraction=0.15, test_fraction=0.15)
    assert again["sessions"] == splits["sessions"]


def _write_cache(path: Path, frames: int, cadence: float, *, seed: int) -> CacheEntry:
    generator = np.random.default_rng(seed)
    valid = np.ones(frames, dtype=bool)
    valid[frames // 2] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        status=np.str_("complete"),
        key=np.str_(path.stem),
        camera=np.str_("A"),
        trial=np.str_("trial_104__recording_start-20260823T123820688"),
        stem=np.str_(path.stem),
        frame_ts=1000.0 + cadence * np.arange(frames),
        frame_valid=valid,
        progress=generator.integers(0, 255, size=(frames, 8, 8), dtype=np.uint8),
        support=generator.integers(0, 255, size=(frames, 8, 8), dtype=np.uint8),
        characteristic_scale=np.float64(90.0),
        recipe_digest=np.str_("digest"),
    )
    return CacheEntry(
        path=path,
        key=path.stem,
        camera="A",
        session="trial_104__recording_start-20260823T123820688",
        frames=frames,
        valid_frames=int(valid.sum()),
    )


def test_window_dataset_and_sampler(tmp_path: Path) -> None:
    entries = [
        _write_cache(tmp_path / "A" / "s" / f"clip{index}_progress64.npz", 80, 1.025, seed=index)
        for index in range(3)
    ]
    dataset = ProgressCacheWindowDataset(entries, context_frames=20, lru_clips=2)
    assert len(dataset) > 0
    assert all(last > first for first, last in dataset.clip_window_ranges)
    item = dataset[0]
    assert item["progress"].shape == (20, 1, 8, 8)
    assert item["progress"].dtype.is_floating_point is False
    assert item["support"].shape == item["progress"].shape
    assert item["frame_valid"].shape == (20,)
    assert item["timestamps"].shape == (20,)
    zeroed = item["progress"][~item["frame_valid"]]
    assert int(zeroed.sum()) == 0
    assert item["window_valid_fraction"] >= 0.9

    sampler = ClipMixBatchSampler(dataset, batch_size=4, windows_per_clip=3, clips_per_buffer=2, seed=1)
    batches = list(iter(sampler))
    assert len(batches) == len(sampler) == (3 * 3) // 4
    assert all(len(batch) == 4 for batch in batches)
    assert all(0 <= index < len(dataset) for batch in batches for index in batch)
    sampler.set_epoch(1)
    assert list(iter(sampler)) != batches

    chosen = fixed_validation_indices(dataset, windows=5, seed=0)
    assert len(chosen) == 5
    assert chosen == fixed_validation_indices(dataset, windows=5, seed=0)


def test_valid_runs() -> None:
    assert valid_runs(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)) == [(0, 2), (3, 4), (6, 9)]
    assert valid_runs(np.zeros(3, dtype=bool)) == []


def test_token_entropy_white_noise_versus_autocorrelated() -> None:
    generator = np.random.default_rng(0)
    frames, dims = 400, 16
    white = generator.standard_normal((frames, dims))
    valid = np.ones(frames, dtype=bool)
    metrics = token_entropy_metrics(white, valid, window=60)
    finite = np.isfinite(metrics["marginal_entropy"])
    assert finite.sum() == frames - 60 + 1
    assert not np.isfinite(metrics["marginal_entropy"][:30]).any()
    assert abs(np.nanmean(metrics["marginal_entropy"]) - 0.5 * LOG_2PI_E) < 0.15
    assert np.nanmean(metrics["predictable_information"]) < 0.15

    smooth = np.empty_like(white)
    smooth[0] = white[0]
    for index in range(1, frames):
        smooth[index] = 0.97 * smooth[index - 1] + math.sqrt(1 - 0.97**2) * white[index]
    smooth_metrics = token_entropy_metrics(smooth, valid, window=60)
    assert np.nanmean(smooth_metrics["entropy_rate"]) < np.nanmean(metrics["entropy_rate"]) - 0.5
    assert np.nanmean(smooth_metrics["predictable_information"]) > 0.5


def test_token_entropy_respects_invalid_frames() -> None:
    tokens = np.random.default_rng(1).standard_normal((100, 4))
    valid = np.ones(100, dtype=bool)
    valid[50] = False
    metrics = token_entropy_metrics(tokens, valid, window=20)
    assert np.isnan(metrics["marginal_entropy"][50])
    assert np.isfinite(metrics["marginal_entropy"][10:40]).all()


def test_latent_speed_and_posterior_kl() -> None:
    mean = np.zeros((10, 3))
    mean[:, 0] = np.arange(10)
    valid = np.ones(10, dtype=bool)
    valid[4] = False
    speed = latent_speed(mean, valid, lag_frames=5)
    assert np.isnan(speed[:5]).all()
    assert np.isnan(speed[9])  # depends on invalid frame 4
    assert np.allclose(speed[5:9], 5.0)
    kl = posterior_kl(np.zeros((3, 4)), np.zeros((3, 4)), np.array([True, False, True]))
    assert kl[0] == 0.0 and np.isnan(kl[1])
    kl_shift = posterior_kl(np.ones((1, 4)), np.zeros((1, 4)), np.array([True]))
    assert abs(kl_shift[0] - 2.0) < 1e-9


def test_standardization_and_segment_median() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 1.0], [5.0, 6.0]])
    stats = fit_standardization(values)
    assert np.allclose(stats.mean, [3.0, 4.0])
    standardized = stats.apply(values[[0, 1, 3]])
    assert abs(standardized.mean()) < 1e-9
    median, count = segment_median(np.array([1.0, np.nan, 3.0, 2.0]), minimum_samples=3)
    assert median == 2.0 and count == 3
    median, count = segment_median(np.array([1.0, np.nan]), minimum_samples=3)
    assert math.isnan(median) and count == 1
    with pytest.raises(ValueError):
        fit_standardization(np.full((1, 2), 1.0))
