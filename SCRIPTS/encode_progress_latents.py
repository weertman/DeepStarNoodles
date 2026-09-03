#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Encode cached progress rasters to VAE latents and per-frame behavior metrics.

Writes one ``{stem}_latents.npz`` per clip (tokens, mu, logvar, validity,
per-frame metrics) plus ``segments.csv`` with one median row per ~10 minute
clip. Standardization statistics are fitted on training-split clips only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

import numpy as np
import torch

from deepstarnoodles.hydra_append import apply_os_isolation, configure_cuda
from deepstarnoodles.paths import DATA_ROOT, DEFAULT_TEMPORAL_VAE_CHECKPOINT, MODELS_ROOT
from deepstarnoodles.progress_cache import (
    DEFAULT_PROGRESS_CACHE_ROOT,
    load_progress_cache,
    resample_to_grid,
)
from deepstarnoodles.progress_metrics import (
    FeatureStandardization,
    fit_standardization,
    latent_speed,
    posterior_kl,
    segment_median,
    token_entropy_metrics,
)
from deepstarnoodles.progress_vae_data import (
    DEFAULT_SPLIT_PATH,
    CacheEntry,
    entries_for_split,
    load_splits,
    scan_cache,
    trial_number,
)
from deepstarnoodles.representations import ProgressSequenceEncoder
from deepstarnoodles.representations.progress import canonical_recipe_digest

DEFAULT_LATENTS_ROOT = DATA_ROOT / "deepstarnoodles" / "progress_latents"
METRIC_NAMES = (
    "marginal_entropy",
    "entropy_rate",
    "predictable_information",
    "latent_speed_1s",
    "latent_speed_5s",
    "latent_speed_30s",
    "posterior_kl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_PROGRESS_CACHE_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LATENTS_ROOT)
    parser.add_argument("--stats-clips", type=int, default=240)
    parser.add_argument("--entropy-window", type=int, default=60)
    parser.add_argument("--minimum-samples", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.2)
    parser.add_argument("--no-isolate", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--camera", action="append", choices=list("ABCD"))
    parser.add_argument("--trial")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def default_checkpoint() -> Path:
    candidates = sorted(
        (MODELS_ROOT / "temporal_vae").glob("dnt_progress_temporal_vae_60s_archive_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else DEFAULT_TEMPORAL_VAE_CHECKPOINT


def output_path(entry: CacheEntry, root: Path) -> Path:
    return root / entry.camera / entry.session / entry.path.name.replace("_progress64.npz", "_latents.npz")


@torch.no_grad()
def frame_tokens(encoder: ProgressSequenceEncoder, progress: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    frames = len(progress)
    tokens = np.full((frames, encoder.model.config.frame_token_dim), np.nan, dtype=np.float32)
    positions = torch.nonzero(valid, as_tuple=False).flatten()
    for start in range(0, len(positions), encoder.frame_batch_size):
        selected = positions[start : start + encoder.frame_batch_size]
        values = progress.index_select(0, selected).to(encoder.device).to(torch.float32).mul_(1.0 / 255.0)
        with encoder._autocast():
            encoded = encoder.model.encoder.frame_encoder(values)
        tokens[selected.numpy()] = encoded.float().cpu().numpy()
    return tokens


def encode_entry(encoder: ProgressSequenceEncoder, entry: CacheEntry) -> dict[str, np.ndarray]:
    cache = load_progress_cache(entry.path)
    grid = resample_to_grid(
        cache,
        cadence_seconds=encoder.window_config.cadence_seconds,
        nearest_tolerance=encoder.window_config.nearest_tolerance,
    )
    tensors = grid.tensors(cache)
    tokens = frame_tokens(encoder, tensors["progress"], tensors["frame_valid"])
    encoding = encoder.encode(tensors["progress"], tensors["timestamps"], tensors["frame_valid"])
    return {
        "timestamps": grid.timestamps,
        "frame_valid": grid.frame_valid,
        "source_index": grid.source_index,
        "tokens": tokens,
        "mean": encoding.mean.numpy(),
        "logvar": encoding.logvar.numpy(),
        "window_valid": encoding.valid.numpy(),
        "window_valid_fraction": encoding.window_valid_fraction.numpy(),
    }


def fit_statistics(
    encoder: ProgressSequenceEncoder, train_entries: list[CacheEntry], *, count: int, seed: int
) -> dict[str, FeatureStandardization]:
    generator = np.random.default_rng(seed)
    chosen = list(train_entries)
    generator.shuffle(chosen)
    tokens: list[np.ndarray] = []
    means: list[np.ndarray] = []
    for entry in chosen[:count]:
        encoded = encode_entry(encoder, entry)
        tokens.append(encoded["tokens"][encoded["frame_valid"]])
        means.append(encoded["mean"][encoded["window_valid"]])
    return {
        "tokens": fit_standardization(np.concatenate(tokens)),
        "mean": fit_standardization(np.concatenate(means)),
    }


def compute_metrics(
    encoded: dict[str, np.ndarray],
    stats: dict[str, FeatureStandardization],
    *,
    entropy_window: int,
) -> dict[str, np.ndarray]:
    token_valid = encoded["frame_valid"] & np.isfinite(encoded["tokens"]).all(axis=1)
    standardized_tokens = np.where(
        token_valid[:, None], stats["tokens"].apply(encoded["tokens"].astype(np.float64)), np.nan
    )
    entropy = token_entropy_metrics(standardized_tokens, token_valid, window=entropy_window)
    window_valid = encoded["window_valid"] & np.isfinite(encoded["mean"]).all(axis=1)
    standardized_mean = np.where(
        window_valid[:, None], stats["mean"].apply(encoded["mean"].astype(np.float64)), np.nan
    )
    metrics = {
        **entropy,
        "latent_speed_1s": latent_speed(standardized_mean, window_valid, lag_frames=1),
        "latent_speed_5s": latent_speed(standardized_mean, window_valid, lag_frames=5),
        "latent_speed_30s": latent_speed(standardized_mean, window_valid, lag_frames=30),
        "posterior_kl": posterior_kl(
            np.nan_to_num(encoded["mean"]), np.nan_to_num(encoded["logvar"]), window_valid
        ),
    }
    return metrics


def segment_row(
    entry: CacheEntry,
    encoded: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    *,
    minimum_samples: int,
) -> dict[str, object]:
    timestamps = encoded["timestamps"]
    row: dict[str, object] = {
        "key": entry.key,
        "camera": entry.camera,
        "session": entry.session,
        "trial": trial_number(entry.session),
        "start_ts": float(timestamps[0]),
        "midpoint_ts": float(0.5 * (timestamps[0] + timestamps[-1])),
        "end_ts": float(timestamps[-1]),
        "grid_frames": int(len(timestamps)),
        "valid_frames": int(encoded["frame_valid"].sum()),
        "valid_windows": int(encoded["window_valid"].sum()),
    }
    for name in METRIC_NAMES:
        median, count = segment_median(metrics[name], minimum_samples=minimum_samples)
        row[name] = median
        row[f"{name}_n"] = count
    return row


def aggregate(root: Path, destination: Path) -> int:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/*/*_latents.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                rows.append(json.loads(str(payload["segment_json"])))
        except Exception:  # noqa: BLE001
            continue
    if not rows:
        return 0
    fieldnames = list(rows[0].keys())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (str(item["camera"]), float(item["start_ts"]))):
            writer.writerow(row)
    return len(rows)


def main() -> int:
    args = parse_args()
    if args.aggregate_only:
        count = aggregate(args.output_root, args.output_root / "segments.csv")
        print(json.dumps({"event": "aggregated", "segments": count}), flush=True)
        return 0
    if not args.no_isolate:
        apply_os_isolation()
        configure_cuda(args.device, args.memory_fraction)
    checkpoint = args.checkpoint or default_checkpoint()
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    encoder = ProgressSequenceEncoder(checkpoint, device=device)
    entries = scan_cache(args.cache_root)
    splits = load_splits(args.splits)
    train_entries = entries_for_split(entries, splits, "train")
    if not train_entries:
        print("no training-split clips available for statistics", file=sys.stderr)
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_root / f"standardization_{encoder.checkpoint_sha256[:16]}.json"
    if stats_path.is_file():
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        stats = {name: FeatureStandardization.from_dict(payload[name]) for name in ("tokens", "mean")}
    else:
        began = time.perf_counter()
        stats = fit_statistics(encoder, train_entries, count=args.stats_clips, seed=args.seed)
        stats_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": encoder.checkpoint_sha256,
                    "encoder_recipe_digest": encoder.recipe_digest,
                    "stats_clips": min(args.stats_clips, len(train_entries)),
                    "seed": args.seed,
                    "tokens": stats["tokens"].as_dict(),
                    "mean": stats["mean"].as_dict(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps({"event": "statistics", "path": str(stats_path), "seconds": round(time.perf_counter() - began, 1)}),
            flush=True,
        )
    recipe_digest = canonical_recipe_digest(
        {
            "recipe": "progress_behavior_metrics_v1",
            "encoder_recipe_digest": encoder.recipe_digest,
            "standardization_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "entropy_window": args.entropy_window,
            "minimum_samples": args.minimum_samples,
        }
    )

    selected = [
        entry
        for entry in entries
        if (not args.camera or entry.camera in args.camera)
        and (not args.trial or args.trial in entry.session)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    print(
        json.dumps(
            {
                "event": "start",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": encoder.checkpoint_sha256,
                "device": device,
                "clips": len(selected),
                "recipe_digest": recipe_digest,
                "output_root": str(args.output_root),
            }
        ),
        flush=True,
    )
    done = skipped = errors = 0
    for index, entry in enumerate(selected, start=1):
        destination = output_path(entry, args.output_root)
        if destination.is_file() and not args.force:
            try:
                with np.load(destination, allow_pickle=False) as payload:
                    if str(payload["recipe_digest"]) == recipe_digest:
                        skipped += 1
                        continue
            except Exception:  # noqa: BLE001
                pass
        began = time.perf_counter()
        try:
            encoded = encode_entry(encoder, entry)
            metrics = compute_metrics(encoded, stats, entropy_window=args.entropy_window)
            row = segment_row(entry, encoded, metrics, minimum_samples=args.minimum_samples)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp.npz")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    recipe_digest=np.str_(recipe_digest),
                    checkpoint_sha256=np.str_(encoder.checkpoint_sha256),
                    key=np.str_(entry.key),
                    segment_json=np.str_(json.dumps(row)),
                    timestamps=encoded["timestamps"],
                    frame_valid=encoded["frame_valid"],
                    source_index=encoded["source_index"],
                    tokens=encoded["tokens"].astype(np.float16),
                    mean=encoded["mean"],
                    logvar=encoded["logvar"],
                    window_valid=encoded["window_valid"],
                    window_valid_fraction=encoded["window_valid_fraction"],
                    **{name: metrics[name].astype(np.float32) for name in METRIC_NAMES},
                )
            temporary.replace(destination)
            done += 1
            if index % 25 == 0 or index == len(selected):
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "index": index,
                            "total": len(selected),
                            "done": done,
                            "skipped": skipped,
                            "errors": errors,
                            "last_seconds": round(time.perf_counter() - began, 2),
                            "last": {name: row[name] for name in ("marginal_entropy", "entropy_rate", "latent_speed_5s", "posterior_kl")},
                        }
                    ),
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(json.dumps({"event": "error", "key": entry.key, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    count = aggregate(args.output_root, args.output_root / "segments.csv")
    print(
        json.dumps({"event": "done", "done": done, "skipped": skipped, "errors": errors, "segments": count}),
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
