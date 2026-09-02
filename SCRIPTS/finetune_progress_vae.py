#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Fine-tune the progress temporal VAE on the archive progress cache.

Warm-starts from the released 60 s checkpoint by default, trains with the
package's masked-reconstruction + KL-rate objective on session-disjoint splits,
and exports an inference checkpoint under MODELS/temporal_vae.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

from torch.utils.data import DataLoader, Subset

from deepstarnoodles.hydra_append import apply_os_isolation, configure_cuda
from deepstarnoodles.paths import DEFAULT_TEMPORAL_VAE_CHECKPOINT, MODELS_ROOT, RUNS_ROOT
from deepstarnoodles.progress_cache import DEFAULT_PROGRESS_CACHE_ROOT
from deepstarnoodles.progress_vae_data import (
    DEFAULT_SPLIT_PATH,
    ClipMixBatchSampler,
    ProgressCacheWindowDataset,
    entries_for_split,
    fixed_validation_indices,
    load_splits,
    make_session_splits,
    save_splits,
    scan_cache,
)
from deepstarnoodles.representations import (
    ProgressTemporalTrainingConfig,
    ProgressTemporalVAE,
    ProgressTemporalVAEConfig,
    ProgressWindowConfig,
    export_inference_checkpoint,
    fine_tune_progress_temporal_vae,
    load_progress_temporal_vae_checkpoint,
    train_progress_temporal_vae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="archive_ft")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_PROGRESS_CACHE_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--remake-splits", action="store_true")
    parser.add_argument("--split-seed", type=int, default=20260902)
    parser.add_argument("--source", type=Path, default=DEFAULT_TEMPORAL_VAE_CHECKPOINT)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--kl-warmup-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=5e-6)
    parser.add_argument("--kl-target-nats", type=float, default=16.0)
    parser.add_argument("--mask-ratio", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--windows-per-clip", type=int, default=8)
    parser.add_argument("--clips-per-buffer", type=int, default=16)
    parser.add_argument("--validation-windows", type=int, default=1024)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.3)
    parser.add_argument("--no-isolate", action="store_true")
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--export", type=Path, help="Inference checkpoint destination.")
    parser.add_argument("--dry-run", action="store_true", help="Build splits/datasets and report only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or (RUNS_ROOT / "progress_vae" / args.name)
    export_path = args.export or (
        MODELS_ROOT / "temporal_vae" / f"dnt_progress_temporal_vae_60s_{args.name}.pt"
    )
    if not args.no_isolate:
        apply_os_isolation()
        configure_cuda(args.device, args.memory_fraction)

    entries = scan_cache(args.cache_root)
    if not entries:
        print("no complete progress caches found", file=sys.stderr)
        return 2
    if args.remake_splits or not args.splits.is_file():
        splits = make_session_splits(entries, seed=args.split_seed)
        save_splits(splits, args.splits)
    else:
        splits = load_splits(args.splits)
        known = {name for names in splits["sessions"].values() for name in names}
        unknown = sorted({entry.session for entry in entries} - known)
        if unknown:
            print(
                json.dumps({"event": "warning", "unassigned_sessions": unknown}),
                flush=True,
            )
    train_entries = entries_for_split(entries, splits, "train")
    val_entries = entries_for_split(entries, splits, "validation")
    test_entries = entries_for_split(entries, splits, "test")

    if args.from_scratch:
        model = ProgressTemporalVAE(ProgressTemporalVAEConfig())
        window_config = ProgressWindowConfig()
        preprocess_config = None
        source_info = None
    else:
        loaded = load_progress_temporal_vae_checkpoint(args.source, "cpu", return_info=True)
        model = loaded.model
        window_config = loaded.window_config or ProgressWindowConfig()
        preprocess_config = loaded.preprocess_config
        source_info = {
            "path": str(args.source),
            "checkpoint_sha256": loaded.checkpoint_sha256,
        }
    context = model.config.context_frames
    config = ProgressTemporalTrainingConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        warmup_steps=args.warmup_steps,
        kl_warmup_steps=args.kl_warmup_steps,
        kl_target_nats=args.kl_target_nats,
        mask_ratio=args.mask_ratio,
        validation_interval=args.validation_interval,
        checkpoint_interval=args.checkpoint_interval,
        amp_dtype=args.amp_dtype,
        seed=args.seed,
    )

    began = time.perf_counter()
    train_dataset = ProgressCacheWindowDataset(
        train_entries, context_frames=context, window_config=window_config, lru_clips=2 * args.clips_per_buffer
    )
    val_dataset = ProgressCacheWindowDataset(
        val_entries, context_frames=context, window_config=window_config, lru_clips=8
    )
    val_indices = fixed_validation_indices(
        val_dataset, windows=args.validation_windows, seed=args.seed
    )
    report = {
        "event": "datasets",
        "cache_clips": len(entries),
        "train_clips": len(train_entries),
        "validation_clips": len(val_entries),
        "test_clips": len(test_entries),
        "train_windows": len(train_dataset),
        "validation_windows_available": len(val_dataset),
        "validation_windows_used": len(val_indices),
        "context_frames": context,
        "split_counts": splits["clip_counts"],
        "enumeration_seconds": round(time.perf_counter() - began, 1),
        "output": str(output),
        "export": str(export_path),
        "source": source_info,
        "model_config": model.config.as_dict(),
        "window_config": window_config.as_dict(),
        "preprocess_config": None if preprocess_config is None else preprocess_config.as_dict(),
        "training_config": config.as_dict(),
    }
    print(json.dumps(report, default=str), flush=True)
    if args.dry_run:
        return 0
    if not len(train_dataset) or not val_indices:
        print("insufficient windows for training", file=sys.stderr)
        return 2

    sampler = ClipMixBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        windows_per_clip=args.windows_per_clip,
        clips_per_buffer=args.clips_per_buffer,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.workers, 2),
        pin_memory=args.device.startswith("cuda"),
    )
    config = ProgressTemporalTrainingConfig(
        **{
            **config.as_dict(),
            "validation_batches": max(1, len(val_indices) // args.batch_size),
        }
    )
    metadata = {
        "name": args.name,
        "training_dataset": "startracked_loop_archive_progress_cache",
        "cache_root": str(args.cache_root),
        "splits_path": str(args.splits),
        "split_unit": splits["unit"],
        "split_seed": splits["seed"],
        "train_clips": len(train_entries),
        "validation_clips": len(val_entries),
        "test_clips": len(test_entries),
        "train_sessions": splits["sessions"]["train"],
        "validation_sessions": splits["sessions"]["validation"],
        "test_sessions": splits["sessions"]["test"],
        "input_cadence_hz": 1.0 / window_config.cadence_seconds,
        "grid_resampling": "nearest_within_tolerance_on_uniform_cadence_grid",
        "hydra_checkpoint_sha256": "dcf7140c9ef0129d8f66b33b05af6d0ebb064f84b41ce61e51e67061799e3f02",
    }
    common = {
        "device": args.device,
        "config": config,
        "resume": args.resume,
        "preprocess_config": preprocess_config,
        "window_config": window_config,
        "metadata": metadata,
    }
    if args.from_scratch:
        result = train_progress_temporal_vae(model, train_loader, val_loader, output, **common)
    else:
        result = fine_tune_progress_temporal_vae(model, train_loader, val_loader, output, **common)
    print(
        json.dumps(
            {
                "event": "trained",
                "best_validation_loss": result["best_validation_loss"],
                "best_checkpoint": result["best_checkpoint"],
            }
        ),
        flush=True,
    )
    exported = export_inference_checkpoint(
        result["best_checkpoint"],
        export_path,
        metadata={
            **metadata,
            "best_validation_loss": result["best_validation_loss"],
            "acausal": window_config.centered,
        },
    )
    (output / "export.json").write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "exported", **exported}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
