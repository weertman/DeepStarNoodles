#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Calibrate noodle tracer seed and termination thresholds on validation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / 'CORE'
SCRIPTS = Path(__file__).resolve().parent
for path in (CORE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepstarnoodles.noodle_tracer_cache import FrozenOutputDataset, OutputAugment
from deepstarnoodles.noodle_tracer import (
    load_noodle_tracer_checkpoint,
    save_noodle_tracer_checkpoint,
)
from train_noodle_tracer import free_rollout_metrics


DEFAULT_STUDY = REPO / "RUNS/noodle_tracer_sleepy18"
DEFAULT_CHECKPOINT = DEFAULT_STUDY / "training/primary/best.pt"


def parse_thresholds(value: str) -> list[float | None]:
    output: list[float | None] = []
    for token in value.split(","):
        token = token.strip()
        output.append(None if token.lower() in {"none", "argmax"} else float(token))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--invalid-seed-thresholds",
        default="none,0.2,0.3,0.4,0.5,0.6",
    )
    parser.add_argument(
        "--base-state-thresholds",
        default="none,0.2,0.3,0.4,0.5,0.6",
    )
    parser.add_argument(
        "--lost-state-thresholds",
        default="none",
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--write-calibrated-checkpoint",
        type=Path,
        help="Write a checkpoint copy with the selected validation thresholds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study = args.study_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint.parent / "validation_state_calibration.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model, base_params, metadata, training_state = load_noodle_tracer_checkpoint(
        str(checkpoint),
        device=args.device,
    )
    dataset = FrozenOutputDataset(
        study / "cache/validation_manifest.json",
        augment=OutputAugment(enabled=False),
    )
    rows = []
    for invalid_threshold in parse_thresholds(args.invalid_seed_thresholds):
        for base_threshold in parse_thresholds(args.base_state_thresholds):
            for lost_threshold in parse_thresholds(args.lost_state_thresholds):
                params = replace(
                    base_params,
                    invalid_seed_threshold=invalid_threshold,
                    base_state_threshold=base_threshold,
                    lost_state_threshold=lost_threshold,
                )
                metrics = free_rollout_metrics(
                    model,
                    dataset,
                    params=params,
                    limit=args.max_images,
                )
                row = {
                    "invalid_seed_threshold": invalid_threshold,
                    "base_state_threshold": base_threshold,
                    "lost_state_threshold": lost_threshold,
                    **metrics,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True))
    rows.sort(
        key=lambda row: (
            float(row["group_macro_f1"]),
            float(row["f1"]),
            -float(row["chamfer_mean"]),
        ),
        reverse=True,
    )
    fieldnames = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    best = rows[0]
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": metadata.get("epoch"),
        "selection_metric": "validation_group_macro_f1",
        "best": best,
        "calibrated_trace_params": {
            **asdict(base_params),
            "invalid_seed_threshold": best["invalid_seed_threshold"],
            "base_state_threshold": best["base_state_threshold"],
            "lost_state_threshold": best["lost_state_threshold"],
        },
        "candidates": len(rows),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.write_calibrated_checkpoint is not None:
        calibrated_path = args.write_calibrated_checkpoint.expanduser().resolve()
        calibrated_path.parent.mkdir(parents=True, exist_ok=True)
        calibrated_params = replace(
            base_params,
            invalid_seed_threshold=best["invalid_seed_threshold"],
            base_state_threshold=best["base_state_threshold"],
            lost_state_threshold=best["lost_state_threshold"],
        )
        calibrated_metadata = {
            **metadata,
            "validation_calibration": {
                "selection_metric": "validation_group_macro_f1",
                **best,
            },
        }
        save_noodle_tracer_checkpoint(
            str(calibrated_path),
            model,
            trace_params=calibrated_params,
            metadata=calibrated_metadata,
            training_state=training_state,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
