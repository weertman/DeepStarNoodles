#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Calibrate noodle tracer seed and termination thresholds on validation.

Every threshold combination is scored on a fixed validation cache. With several
``--checkpoint`` values (for example, training seeds of one recipe), each
combination is scored for every checkpoint and selected by the mean metric; the
first checkpoint is the one written by ``--write-calibrated-checkpoint``.

The released tracer's threshold was selected with ``--selection-metric f1`` on
the non-Sleepy validation cache built with the endpoint-refined Hydra, over
three training seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
    parser.add_argument(
        "--validation-root",
        type=Path,
        help="study root holding cache/validation_manifest.json (default: --study-root)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        help="checkpoint to calibrate; repeat to select by the mean over checkpoints",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("group_macro_f1", "f1"),
        default="group_macro_f1",
        help="validation metric that selects the thresholds (default: group_macro_f1)",
    )
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


METRIC_KEYS = (
    "precision", "recall", "f1", "group_macro_f1", "tp", "fp", "fn",
    "chamfer_mean", "base_completion_rate", "images",
)


def main() -> int:
    args = parse_args()
    study = args.study_root.expanduser().resolve()
    validation_root = (
        args.validation_root.expanduser().resolve() if args.validation_root is not None else study
    )
    checkpoints = [path.expanduser().resolve() for path in (args.checkpoint or [DEFAULT_CHECKPOINT])]
    checkpoint = checkpoints[0]
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint.parent / "validation_state_calibration.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    loaded = [load_noodle_tracer_checkpoint(str(path), device=args.device) for path in checkpoints]
    model, base_params, metadata, training_state = loaded[0]
    dataset = FrozenOutputDataset(
        validation_root / "cache/validation_manifest.json",
        augment=OutputAugment(enabled=False),
    )
    rows = []
    for invalid_threshold in parse_thresholds(args.invalid_seed_thresholds):
        for base_threshold in parse_thresholds(args.base_state_thresholds):
            for lost_threshold in parse_thresholds(args.lost_state_thresholds):
                per_checkpoint = []
                for candidate, candidate_params, _, _ in loaded:
                    params = replace(
                        candidate_params,
                        invalid_seed_threshold=invalid_threshold,
                        base_state_threshold=base_threshold,
                        lost_state_threshold=lost_threshold,
                    )
                    per_checkpoint.append(free_rollout_metrics(
                        candidate,
                        dataset,
                        params=params,
                        limit=args.max_images,
                    ))
                row = {
                    "invalid_seed_threshold": invalid_threshold,
                    "base_state_threshold": base_threshold,
                    "lost_state_threshold": lost_threshold,
                    **{
                        key: sum(float(metrics[key]) for metrics in per_checkpoint) / len(per_checkpoint)
                        for key in METRIC_KEYS
                    },
                    "checkpoints": len(per_checkpoint),
                }
                if len(per_checkpoint) > 1:
                    row["per_checkpoint_f1"] = json.dumps([float(m["f1"]) for m in per_checkpoint])
                rows.append(row)
                print(json.dumps(row, sort_keys=True))
    secondary = "f1" if args.selection_metric == "group_macro_f1" else "group_macro_f1"
    def chamfer_key(row: dict) -> float:
        value = float(row["chamfer_mean"])
        return -value if math.isfinite(value) else -math.inf

    rows.sort(
        key=lambda row: (
            float(row[args.selection_metric]),
            float(row[secondary]),
            chamfer_key(row),
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
        "checkpoints": [str(path) for path in checkpoints],
        "validation_manifest": str(validation_root / "cache/validation_manifest.json"),
        "checkpoint_epoch": metadata.get("epoch"),
        "selection_metric": f"validation_{args.selection_metric}",
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
                "selection_metric": f"validation_{args.selection_metric}",
                "validation_manifest": str(validation_root / "cache/validation_manifest.json"),
                "checkpoints": [str(path) for path in checkpoints],
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
