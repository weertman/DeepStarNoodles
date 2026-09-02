#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Offline Hydra append over the timelapse archive. Separate process; no recorder import."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

from deepstarnoodles.batch_library import SOURCE_ROOT, balanced
from deepstarnoodles.hydra import HydraInferConfig, HydraPredictor
from deepstarnoodles.hydra_append import (
    append_hydra_clip,
    apply_os_isolation,
    configure_cuda,
    default_checkpoint_sha,
    hydra_contract_hash,
    inventory_hydra_sources,
)
from deepstarnoodles.hydra_budget import load_budget
from deepstarnoodles.paths import DEFAULT_HYDRA_CHECKPOINT
from deepstarnoodles.recorder_guard import DEFAULT_STATUS_URL, RecorderGuard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--run", action="store_true", help="Write sidecars (default is inventory).")
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
        "--verbose-skipped",
        action="store_true",
        help="Include every skipped clip in the JSON report.",
    )
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


def main() -> int:
    args = parse_args()
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

    checkpoint_sha = default_checkpoint_sha(args.hydra)
    expected_hash = hydra_contract_hash(checkpoint_sha) if checkpoint_sha else ""
    eligible, skipped, complete = inventory_hydra_sources(
        args.source_root, expected_hash=expected_hash
    )
    selected = select(args, eligible)
    skipped_counts: dict[str, int] = {}
    for item in skipped:
        skipped_counts[item.status] = skipped_counts.get(item.status, 0) + 1
    skipped_rows = [
        {"key": item.key, "status": item.status, "error": item.error} for item in skipped
    ]
    report = {
        "mode": "run" if args.run else "dry-run",
        "eligible": len(eligible),
        "selected": len(selected),
        "complete": len(complete),
        "skipped_counts": skipped_counts,
        "skipped_sample": skipped_rows[:8],
        "batch_size": batch_size,
        "memory_fraction": memory_fraction,
        "budget_status": budget.get("status"),
        "isolation": isolation,
        "contract_hash": expected_hash,
    }
    if args.verbose_skipped:
        report["skipped"] = skipped_rows
    if not args.run:
        report["keys"] = [clip.key for clip in selected]
        print(json.dumps(report, indent=2), flush=True)
        return 0

    import time

    import torch

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(json.dumps({"event": "start", **report, "device": device}, default=str), flush=True)
    predictor = HydraPredictor(args.hydra, HydraInferConfig(device=device))
    guard = None if args.no_yield else RecorderGuard(
        args.status_url, queue_watermark=args.queue_watermark
    )
    results = []
    for index, clip in enumerate(selected, start=1):
        began = time.perf_counter()
        try:
            result = append_hydra_clip(
                clip,
                predictor,
                checkpoint_sha=checkpoint_sha,
                batch_size=batch_size,
                device=device,
                guard=guard,
                max_frames=args.max_frames,
            )
            row = {
                "event": "clip",
                "index": index,
                "total": len(selected),
                "key": result.key,
                "status": result.status,
                "frames": result.frames,
                "detected": result.detected,
                "seconds": round(time.perf_counter() - began, 2),
            }
        except Exception as exc:
            row = {
                "event": "clip",
                "index": index,
                "total": len(selected),
                "key": clip.key,
                "status": "error",
                "error": str(exc),
                "seconds": round(time.perf_counter() - began, 2),
            }
        results.append({k: row[k] for k in row if k not in {"event", "index", "total", "seconds"}})
        print(json.dumps(row), flush=True)
    report["results"] = results
    print(json.dumps({"event": "done", **report}, default=str), flush=True)
    return 0 if all(item.get("status") != "error" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
