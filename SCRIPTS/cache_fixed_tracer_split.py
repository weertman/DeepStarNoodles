#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Cache a fixed (unaugmented) evaluation split for one tracer cohort.

Refreshed training validates on a separate, fixed cache: by default the 34
Sleepy validation images, cached with the training Hydra. Threshold calibration
of the released tracer used the 444 non-Sleepy validation images cached with the
endpoint-refined Hydra. The manifest records the fixed training normalization.

Held-out test images are cached only with ``--split test --analysis-lock``. The
lock must set ``test_access_locked`` and record the SHA-256 of the test index,
of the Hydra checkpoint (``frozen_checkpoint_sha256``), and of the normalization
file (``normalization_sha256``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "CORE"
for path in (CORE, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepstarnoodles.noodle_tracer_cache import FROZEN_CACHE_SCHEMA, TRACER_COHORTS, sha256_file  # noqa: E402
from cache_frozen_outputs import DEFAULT_SPLITS, atomic_json, cache_split, prepare_contract  # noqa: E402
from generate_refreshed_tracer_cache import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_NORMALIZATION,
    load_normalization,
)


def check_test_lock(lock_path: Path | None, split_dir: Path) -> dict:
    if lock_path is None:
        raise ValueError("caching held-out test images requires --analysis-lock")
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if not lock.get("test_access_locked"):
        raise ValueError("analysis lock does not authorize held-out test access")
    if lock.get("test_source_index_sha256") != sha256_file(split_dir / "test_index.csv"):
        raise ValueError("test index differs from the locked analysis")
    return lock


def cache_fixed_split(
    *,
    output: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    split_dir: Path = DEFAULT_SPLITS,
    normalization_json: Path = DEFAULT_NORMALIZATION,
    cohort: str = "sleepy",
    split: str = "validation",
    analysis_lock: Path | None = None,
    device: str = "cuda:0",
    batch_size: int = 4,
    num_workers: int = 2,
) -> Path:
    output = Path(output).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    split_dir = Path(split_dir).expanduser().resolve()
    lock = check_test_lock(analysis_lock, split_dir) if split == "test" else None
    contract = prepare_contract(output=output, checkpoint=checkpoint, split_dir=split_dir, cohort=cohort)
    if lock is not None:
        if lock.get("frozen_checkpoint_sha256") != contract["frozen_checkpoint_sha256"]:
            raise ValueError("frozen dense checkpoint differs from the analysis lock")
        if lock.get("normalization_sha256") != sha256_file(normalization_json):
            raise ValueError("normalization differs from the analysis lock")
    mean, std = load_normalization(normalization_json)
    records, _, _, pixel_count = cache_split(
        output=output,
        split=split,
        checkpoint=checkpoint,
        checkpoint_sha256=contract["frozen_checkpoint_sha256"],
        index_sha256=contract["splits"][split]["materialized_sha256"],
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        force=False,
        cohort=cohort,
    )
    manifest = {
        "schema_version": FROZEN_CACHE_SCHEMA,
        "split": split,
        "records": records,
        "provenance": {
            "contract_id": contract["contract_id"],
            "frozen_checkpoint_sha256": contract["frozen_checkpoint_sha256"],
            "index_sha256": contract["splits"][split]["materialized_sha256"],
            "raw_output_dtype": "float16",
            "rgb_dtype": "uint8",
            "cached_pixels": pixel_count,
            "channel_statistics": {"source_split": "train", "mean": mean.tolist(), "std": std.tolist()},
        },
    }
    manifest_path = output / "cache" / f"{split}_manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"Fixed {cohort} {split}: {len(records)} images -> {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Hydra checkpoint for the cached fields (default: epoch_060.pt)")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--normalization-json", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--cohort", choices=TRACER_COHORTS, default="sleepy")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--analysis-lock", type=Path, help="required for --split test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    cache_fixed_split(
        output=args.output, checkpoint=args.checkpoint, split_dir=args.split_dir,
        normalization_json=args.normalization_json, cohort=args.cohort, split=args.split,
        analysis_lock=args.analysis_lock, device=args.device, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
