#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Write a fixed tracer normalization file from a cached training split.

The released recipe normalizes the 18 dense channels with statistics of the raw
``epoch_060.pt`` outputs over every pixel of the 3,659 non-Sleepy training
images. ``config/recipes/noodle_tracer_fullcorpus_normalization.json`` holds
those values. To recompute them (results can differ in the last digits across
hardware), cache that split and convert its manifest:

    python SCRIPTS/cache_frozen_outputs.py --stage all --cohort non_sleepy \\
      --cache-splits train --output RUNS/tracer_normalization_non_sleepy
    python SCRIPTS/write_tracer_normalization.py \\
      RUNS/tracer_normalization_non_sleepy/cache/train_manifest.json \\
      RUNS/tracer_normalization_non_sleepy/normalization.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def write_normalization(manifest_path: Path, output_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    statistics = manifest["provenance"]["channel_statistics"]
    if manifest.get("split") != "train" or statistics.get("source_split") != "train":
        raise SystemExit("the normalization source must be a train manifest of train pixels")
    if len(statistics.get("mean", [])) != 18 or len(statistics.get("std", [])) != 18:
        raise SystemExit("expected 18 dense channel statistics")
    if not all(float(value) > 0 for value in statistics["std"]):
        raise SystemExit("dense standard deviations must be positive")
    value = {
        "channels": 18,
        "mean": statistics["mean"],
        "std": statistics["std"],
        "source_split": "train",
        "train_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "contract_id": manifest["provenance"]["contract_id"],
        "frozen_checkpoint_sha256": manifest["provenance"]["frozen_checkpoint_sha256"],
        "images": len(manifest.get("records", [])),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = write_normalization(args.train_manifest, args.output)
    print(json.dumps({"output": str(args.output), "images": value["images"]}))


if __name__ == "__main__":
    main()
