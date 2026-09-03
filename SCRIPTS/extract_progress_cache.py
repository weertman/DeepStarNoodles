#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Extract canonical 64x64 progress rasters from the archive for the temporal VAE.

Separate process from the live recorder. Re-runs Hydra on StarSeg crops and
keeps only channels 0/4 after canonicalization (see progress_cache.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

from deepstarnoodles.batch_library import SOURCE_ROOT, balanced
from deepstarnoodles.hydra import HydraInferConfig, HydraPredictor
from deepstarnoodles.hydra_append import (
    apply_os_isolation,
    configure_cuda,
    default_checkpoint_sha,
)
from deepstarnoodles.hydra_budget import load_budget
from deepstarnoodles.paths import DEFAULT_HYDRA_CHECKPOINT
from deepstarnoodles.progress_cache import (
    DEFAULT_PROGRESS_CACHE_ROOT,
    cache_recipe_digest,
    extract_progress_cache,
    inventory_cache_sources,
)
from deepstarnoodles.recorder_guard import DEFAULT_STATUS_URL, RecorderGuard
from deepstarnoodles.representations.progress import ProgressRasterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_PROGRESS_CACHE_ROOT)
    parser.add_argument("--run", action="store_true", help="Write caches (default is inventory).")
    parser.add_argument("--camera", action="append", choices=list("ABCD"))
    parser.add_argument("--key", action="append")
    parser.add_argument("--trial")
    parser.add_argument("--date")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument("--budget-json", type=Path)
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    parser.add_argument("--queue-watermark", type=int, default=8)
    parser.add_argument("--no-yield", action="store_true")
    parser.add_argument("--no-isolate", action="store_true")
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA_CHECKPOINT)
    parser.add_argument(
        "--wait-for-pid",
        type=int,
        help="Poll until this PID exits before touching the GPU (queue behind another pass).",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def select(args: argparse.Namespace, clips: list) -> list:
    selected = [
        clip
        for clip in clips
        if (not args.key or clip.key in args.key)
        and (not args.camera or clip.camera in args.camera)
        and (not args.trial or args.trial in clip.trial)
        and (not args.date or clip.date == args.date)
    ]
    selected = balanced(selected)
    return selected[: args.limit] if args.limit is not None else selected


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    args = parse_args()
    if args.wait_for_pid is not None and args.run:
        waited = 0.0
        while pid_alive(args.wait_for_pid):
            if waited == 0.0:
                print(
                    json.dumps({"event": "waiting", "pid": args.wait_for_pid}),
                    flush=True,
                )
            time.sleep(args.poll_seconds)
            waited += args.poll_seconds
        print(json.dumps({"event": "wait_done", "pid": args.wait_for_pid, "seconds": waited}), flush=True)

    budget = load_budget(args.budget_json)
    batch_size = args.batch_size or int(budget.get("recommended_batch_size") or 4)
    memory_fraction = (
        args.memory_fraction
        if args.memory_fraction is not None
        else float(budget.get("memory_fraction") or 0.45)
    )
    isolation = {} if args.no_isolate else apply_os_isolation()
    if not args.no_isolate:
        configure_cuda(args.device, memory_fraction)

    raster_config = ProgressRasterConfig()
    checkpoint_sha = default_checkpoint_sha(args.hydra)
    expected_digest = cache_recipe_digest(checkpoint_sha, raster_config)
    eligible, skipped, complete = inventory_cache_sources(
        args.source_root, expected_digest=expected_digest, cache_root=args.cache_root
    )
    selected = select(args, eligible)
    skipped_counts: dict[str, int] = {}
    for item in skipped:
        skipped_counts[item.status] = skipped_counts.get(item.status, 0) + 1
    report = {
        "mode": "run" if args.run else "dry-run",
        "eligible": len(eligible),
        "selected": len(selected),
        "complete": len(complete),
        "skipped_counts": skipped_counts,
        "batch_size": batch_size,
        "memory_fraction": memory_fraction,
        "isolation": isolation,
        "recipe_digest": expected_digest,
        "hydra_checkpoint_sha256": checkpoint_sha,
        "cache_root": str(args.cache_root),
    }
    if not args.run:
        report["keys"] = [clip.key for clip in selected][:50]
        print(json.dumps(report, indent=2), flush=True)
        return 0

    import torch

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(json.dumps({"event": "start", **report, "device": device}, default=str), flush=True)
    predictor = HydraPredictor(args.hydra, HydraInferConfig(device=device))
    guard = None if args.no_yield else RecorderGuard(
        args.status_url, queue_watermark=args.queue_watermark
    )
    errors = 0
    for index, clip in enumerate(selected, start=1):
        began = time.perf_counter()
        try:
            result = extract_progress_cache(
                clip,
                predictor,
                checkpoint_sha=checkpoint_sha,
                cache_root=args.cache_root,
                batch_size=batch_size,
                device=device,
                guard=guard,
                max_frames=args.max_frames,
                raster_config=raster_config,
            )
            row = {
                "event": "clip",
                "index": index,
                "total": len(selected),
                "key": result.key,
                "status": result.status,
                "frames": result.frames,
                "detected": result.detected,
                "valid": result.valid,
                "characteristic_scale": result.characteristic_scale,
                "seconds": round(time.perf_counter() - began, 2),
            }
        except Exception as exc:  # noqa: BLE001 - keep the pass going, report per clip
            errors += 1
            row = {
                "event": "clip",
                "index": index,
                "total": len(selected),
                "key": clip.key,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.perf_counter() - began, 2),
            }
        print(json.dumps(row, default=str), flush=True)
    print(json.dumps({"event": "done", "selected": len(selected), "errors": errors}), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
