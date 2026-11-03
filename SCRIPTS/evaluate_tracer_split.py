#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Score Noodle Tracer checkpoints on a fixed validation or locked test cache.

The cache comes from ``cache_fixed_tracer_split.py``. Arms are matched one to
one when their symmetric Chamfer distance is at most 8 model-crop pixels;
precision, recall and F1 pool all images, and group-macro F1 averages the
``split_group_id`` groups. With several checkpoints, F1 differences against the
first are given with a paired bootstrap over groups. This is the procedure of the
released tracer's locked test readout.

``--split test`` requires ``--analysis-lock`` with ``test_access_locked``. When the
lock records checkpoint hashes (keys ending in ``checkpoint_sha256``), every
scored checkpoint must be one of them.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "CORE"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from deepstarnoodles.evaluate_decode import match_arms  # noqa: E402
from deepstarnoodles.noodle_tracer import load_noodle_tracer_checkpoint, trace_arms_noodle_batch  # noqa: E402
from deepstarnoodles.noodle_tracer_cache import FrozenOutputDataset, sha256_file  # noqa: E402

MATCH_TOLERANCE_PX = 8.0


def lock_checkpoint_hashes(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("checkpoint_sha256") and isinstance(item, str) and key != "frozen_checkpoint_sha256":
                found.add(item)
            else:
                found |= lock_checkpoint_hashes(item)
    elif isinstance(value, list):
        for item in value:
            found |= lock_checkpoint_hashes(item)
    return found


def check_lock(lock_path: Path | None, checkpoints: Iterable[Path]) -> dict:
    if lock_path is None:
        raise SystemExit("scoring the held-out test split requires --analysis-lock")
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if not lock.get("test_access_locked"):
        raise SystemExit("analysis lock does not authorize held-out test access")
    allowed = lock_checkpoint_hashes(lock)
    if allowed:
        for checkpoint in checkpoints:
            if sha256_file(checkpoint) not in allowed:
                raise SystemExit(f"{checkpoint} is not a checkpoint recorded in the analysis lock")
    return lock


def per_image(decode: Callable, dataset: FrozenOutputDataset, batch_size: int = 4) -> list[dict]:
    rows = []
    for start in range(0, len(dataset), batch_size):
        items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        raw = torch.stack([item["raw"][:18] for item in items])
        image = torch.stack([item["image"] for item in items])
        for item, arms in zip(items, decode(raw, image)):
            gt = [np.asarray(arm, dtype=np.float32) for arm in item["arms"]]
            matched = match_arms([arm["polyline"] for arm in arms], gt, MATCH_TOLERANCE_PX)
            rows.append({
                "tp": int(matched["tp"]), "fp": int(matched["fp"]), "fn": int(matched["fn"]),
                "chamfer": [float(value[2]) for value in matched["matches"]],
                "group": str(item["record"].get("split_group_id", "unknown")),
            })
    return rows


def summarize(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    groups: dict[str, list[int]] = {}
    for row in rows:
        counts = groups.setdefault(row["group"], [0, 0, 0])
        counts[0] += row["tp"]; counts[1] += row["fp"]; counts[2] += row["fn"]
    group_f1 = []
    for gtp, gfp, gfn in groups.values():
        gp, gr = gtp / max(gtp + gfp, 1), gtp / max(gtp + gfn, 1)
        group_f1.append(2 * gp * gr / max(gp + gr, 1e-9))
    chamfers = [value for row in rows for value in row["chamfer"]]
    return {
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "precision": precision, "recall": recall,
        "group_macro_f1": float(np.mean(group_f1)) if group_f1 else 0.0,
        "chamfer_mean": float(np.mean(chamfers)) if chamfers else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "images": len(rows), "groups": len(groups),
    }


def pooled_f1(rows: list[dict], index: np.ndarray) -> float:
    tp = sum(rows[i]["tp"] for i in index)
    fp = sum(rows[i]["fp"] for i in index)
    fn = sum(rows[i]["fn"] for i in index)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def paired_group_bootstrap(a: list[dict], b: list[dict], *, samples: int, seed: int) -> dict:
    """F1(a) − F1(b) with a 95% interval from resampling split groups."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(a):
        groups.setdefault(row["group"], []).append(index)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    every = np.arange(len(a))
    differences = []
    for _ in range(samples):
        index = np.concatenate([groups[keys[k]] for k in rng.choice(len(keys), len(keys))])
        differences.append(pooled_f1(a, index) - pooled_f1(b, index))
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": pooled_f1(a, every) - pooled_f1(b, every),
            "ci95": [float(low), float(high)], "groups": len(keys), "samples": samples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-root", type=Path, required=True,
                        help="output directory of cache_fixed_tracer_split.py")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="checkpoint to score; repeat to compare against the first")
    parser.add_argument("--invalid-seed-threshold", type=float,
                        help="override the threshold stored in every checkpoint")
    parser.add_argument("--analysis-lock", type=Path, help="required for --split test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    checkpoints = [path.expanduser().resolve() for path in args.checkpoint]
    if args.split == "test":
        check_lock(args.analysis_lock, checkpoints)
    dataset = FrozenOutputDataset(args.split_root / "cache" / f"{args.split}_manifest.json")
    device = torch.device(args.device)
    results: dict[str, Any] = {"split_root": str(args.split_root.resolve()), "split": args.split,
                               "match_tolerance_px": MATCH_TOLERANCE_PX, "checkpoints": []}
    rows_by_checkpoint = []
    for checkpoint in checkpoints:
        model, params, _, _ = load_noodle_tracer_checkpoint(str(checkpoint), device=device)
        if args.invalid_seed_threshold is not None:
            params = replace(params, invalid_seed_threshold=args.invalid_seed_threshold)
        rows = per_image(
            lambda raw, image: trace_arms_noodle_batch(raw.to(device), image.to(device), model, params),
            dataset, batch_size=args.batch_size,
        )
        rows_by_checkpoint.append(rows)
        summary = summarize(rows)
        results["checkpoints"].append({"checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint),
                                       "invalid_seed_threshold": params.invalid_seed_threshold, **summary})
        print(json.dumps({"checkpoint": checkpoint.name, **{k: summary[k] for k in
                          ("f1", "group_macro_f1", "precision", "recall", "chamfer_mean", "images")}}),
              flush=True)
    results["contrasts_vs_first"] = [
        {"checkpoint": str(checkpoints[i]),
         **paired_group_bootstrap(rows_by_checkpoint[i], rows_by_checkpoint[0],
                                  samples=args.bootstrap_samples, seed=args.seed)}
        for i in range(1, len(checkpoints))
    ]
    for contrast in results["contrasts_vs_first"]:
        print(json.dumps(contrast), flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
