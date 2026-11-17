#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Generate one refreshed training generation for the Noodle Tracer.

Each training image of the prepared cohort is augmented in pixel space
(flips, quarter turns, brightness, contrast, saturation), its annotated arm
polylines are moved with the same transform, and the frozen Hydra model is rerun
on the augmented image. The cached dense outputs, augmented RGB, and transformed
arms become ``cache/generations/gNNNN/`` and the study's active
``cache/train_manifest.json``.

Prepare the study first with ``cache_frozen_outputs.py --stage prepare``.
``--limit`` writes a pilot generation that is never published for training.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "CORE"
for path in (CORE, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepstarnoodles.noodle_tracer_cache import (  # noqa: E402
    FROZEN_CACHE_SCHEMA,
    cache_record_identity,
    prepared_arms_from_row,
    sha256_file,
    write_cache_record,
)
from deepstarnoodles.noodle_tracer_refresh import (  # noqa: E402
    REFRESH_IMAGE_SIZE,
    augment_image_and_arms,
    draw_augmentation,
)
from deepstarnoodles.paths import resolve_repo_path, to_repo_relative  # noqa: E402
from cache_frozen_outputs import (  # noqa: E402
    atomic_json,
    canonical_hash,
    index_name,
    load_frozen_model,
    validate_cache_contract,
)

DEFAULT_CHECKPOINT = REPO / "MODELS/hydra/epoch_060.pt"
DEFAULT_NORMALIZATION = REPO / "config/recipes/noodle_tracer_fullcorpus_normalization.json"
RELEASED_AUGMENTATION_SEED = 20261011


def display_path(path: Path) -> str:
    """Repo-relative path when inside the checkout, else the absolute path."""
    try:
        return to_repo_relative(Path(path).resolve())
    except ValueError:
        return str(Path(path).resolve())


def load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    normalization = json.loads(Path(path).read_text(encoding="utf-8"))
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    std = np.asarray(normalization["std"], dtype=np.float64)
    if mean.shape != (18,) or std.shape != (18,) or not np.isfinite(mean).all() or not np.all(std > 0):
        raise ValueError("normalization must contain finite 18-channel mean and positive std")
    return mean, std


def generate(
    *,
    study_root: Path,
    checkpoint: Path,
    normalization_json: Path,
    cycle: int,
    seed: int = RELEASED_AUGMENTATION_SEED,
    device: str = "cuda:0",
    batch_size: int = 4,
    limit: int | None = None,
    draws_per_image: int = 1,
    cpu_workers: int = 0,
) -> Path:
    """Write one generation and return its directory."""
    if cycle < 1 or batch_size < 1 or draws_per_image < 1 or (limit is not None and limit < 1):
        raise ValueError("cycle, batch size, draws, and pilot limit must be positive")
    if cpu_workers not in (0,) and not 3 <= cpu_workers <= 32:
        raise ValueError("cpu_workers must be zero (serial), or between 3 and 32")
    root = Path(study_root).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    contract = json.loads((root / "experiment_contract.json").read_text(encoding="utf-8"))
    cohort = contract.get("cohort", "sleepy")
    validate_cache_contract(output=root, checkpoint=checkpoint, contract=contract, splits=("train",))
    with (root / "indexes" / index_name(cohort, "train")).open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    if len(all_rows) != int(contract["splits"]["train"]["records"]):
        raise ValueError("training index count differs from contract")
    pilot = limit is not None
    rows = all_rows[:limit] if pilot else all_rows
    expanded_rows = [
        (draw, source_index, row)
        for draw in range(draws_per_image)
        for source_index, row in enumerate(rows)
    ]
    generation = f"g{cycle:04d}"
    generation_dir = root / "cache" / "generations" / generation
    if pilot:
        generation_dir = root / "cache" / "pilots" / f"{generation}_n{len(rows)}"
    record_dir = generation_dir / "train"
    record_dir.mkdir(parents=True, exist_ok=True)
    mean, std = load_normalization(normalization_json)
    torch_device = torch.device(device)
    model = load_frozen_model(checkpoint, torch_device)
    records: list[dict] = []
    pending: list[dict] = []
    started = time.monotonic()
    progress_path = generation_dir / "progress.json"

    write_futures = deque()
    write_pool = None
    max_pending_writes = max(batch_size, 2 * max(cpu_workers - 2, 1))

    def drain_writes() -> None:
        while write_futures:
            write_futures.popleft().result()

    def flush() -> None:
        if not pending:
            return
        batch = torch.stack([item["tensor"] for item in pending]).to(torch_device)
        with torch.inference_mode(), torch.autocast(
            device_type=torch_device.type, enabled=torch_device.type == "cuda"
        ):
            raw_batch = model(batch).float().cpu().numpy()
        if raw_batch.shape[1:] != (18, REFRESH_IMAGE_SIZE, REFRESH_IMAGE_SIZE):
            raise ValueError(f"unexpected frozen output shape {raw_batch.shape}")
        for item, raw in zip(pending, raw_batch):
            if write_pool is None:
                write_cache_record(item["path"], raw, item["image"], item["arms"],
                                   cache_identity=item["identity"])
            else:
                # Bound retained dense arrays; workers never touch CUDA or RNG.
                if len(write_futures) >= max_pending_writes:
                    write_futures.popleft().result()
                write_futures.append(write_pool.submit(
                    write_cache_record, item["path"], raw, item["image"], item["arms"],
                    cache_identity=item["identity"]))
        pending.clear()

    def prepare_row(indexed_row):
        row_index, (draw, source_index, row) = indexed_row
        image_path = resolve_repo_path(row["image_path"])
        shapes_path = resolve_repo_path(row["shapes_path"])
        image_hash = sha256_file(image_path)
        shapes_hash = sha256_file(shapes_path)
        params = draw_augmentation(seed, cycle, row_index)
        identity = canonical_hash({
            "schema": FROZEN_CACHE_SCHEMA,
            "contract_id": contract["contract_id"],
            "checkpoint_sha256": contract["frozen_checkpoint_sha256"],
            "index_sha256": contract["splits"]["train"]["materialized_sha256"],
            "image_sha256": image_hash, "shapes_sha256": shapes_hash,
            "row_index": row_index, "source_row_index": source_index,
            "draw": draw, "augmentation": params,
        })
        path = record_dir / f"{row_index:05d}_{identity[:12]}.npz"
        record = {
            "row_index": row_index,
            "source_row_index": source_index,
            "draw": draw,
            "cache_file": path.relative_to(root / "cache").as_posix(),
            "image_path": to_repo_relative(image_path),
            "shapes_path": to_repo_relative(shapes_path),
            "source_family": row.get("source_family", ""),
            "split_group_id": row.get("split_group_id", ""),
            "arm_count": int(row.get("arm_count", "0") or 0),
            "image_sha256": image_hash, "shapes_sha256": shapes_hash,
            "cache_identity": identity, "augmentation": params,
        }
        prepared = None
        if cache_record_identity(path) != identity:
            with Image.open(image_path) as image:
                pixels, arms = augment_image_and_arms(image, prepared_arms_from_row(row), params)
            record["arm_count"] = len(arms)
            prepared = {
                "path": path, "image": pixels, "arms": arms, "identity": identity,
            }
        return row_index, record, prepared

    # Submit only a small ordered window, rather than Executor.map's eager
    # submission of the full corpus. Completion order cannot affect batches.
    with ExitStack() as stack:
        prepare_pool = None
        if cpu_workers:
            prepare_pool = stack.enter_context(ThreadPoolExecutor(max_workers=2))
            write_pool = stack.enter_context(ThreadPoolExecutor(max_workers=cpu_workers - 2))
        source = iter(enumerate(expanded_rows))
        prepared_futures = deque()
        if prepare_pool:
            for _ in range(8):
                entry = next(source, None)
                if entry is None:
                    break
                prepared_futures.append(prepare_pool.submit(prepare_row, entry))
        for _ in range(len(expanded_rows)):
            if prepare_pool:
                row_index, record, item = prepared_futures.popleft().result()
                entry = next(source, None)
                if entry is not None:
                    prepared_futures.append(prepare_pool.submit(prepare_row, entry))
            else:
                row_index, record, item = prepare_row(next(source))
            records.append(record)
            if item is not None:
                # Preserve the original CPU tensor arithmetic and GPU call order.
                item["tensor"] = torch.from_numpy(item["image"].copy()).permute(2, 0, 1).float().div_(255)
                pending.append(item)
                if len(pending) >= batch_size:
                    flush()
            if (row_index + 1) % 20 == 0 or row_index + 1 == len(expanded_rows):
                flush()
                # Publication/progress never gets ahead of durable cache writes.
                drain_writes()
                progress = {
                    "stage": "generating", "cycle": cycle, "pilot": pilot,
                    "done": row_index + 1, "total": len(expanded_rows),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
                atomic_json(progress_path, progress)
                print(json.dumps(progress), flush=True)
        drain_writes()
    manifest = {
        "schema_version": FROZEN_CACHE_SCHEMA,
        "split": "train",
        "records": records,
        "provenance": {
            "contract_id": contract["contract_id"],
            "frozen_checkpoint_sha256": contract["frozen_checkpoint_sha256"],
            "index_sha256": contract["splits"]["train"]["materialized_sha256"],
            "generation": generation,
            "augmentation_seed": seed,
            "draws_per_image": draws_per_image,
            "raw_output_dtype": "float16",
            "rgb_dtype": "uint8",
            "channel_statistics": {
                "source_split": "train",
                "mean": mean.tolist(),
                "std": std.tolist(),
                "fixed_reference": display_path(normalization_json),
            },
        },
    }
    atomic_json(generation_dir / "manifest.json", manifest)
    atomic_json(root / "cache" / ("pilot_train_manifest.json" if pilot else "train_manifest.json"), manifest)
    atomic_json(progress_path, {
        "stage": "complete", "cycle": cycle, "pilot": pilot,
        "done": len(expanded_rows), "total": len(expanded_rows),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    })
    performance = {
        "schema_version": 1, "cycle": cycle, "pilot": pilot,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cpu_workers": cpu_workers, "preprocessing_workers": 2 if cpu_workers else 0,
        "compression_workers": cpu_workers - 2 if cpu_workers else 0,
        "max_pending_writes": max_pending_writes if cpu_workers else 0,
        "batch_size": batch_size, "draws_per_image": draws_per_image,
        "augmentation_seed": seed, "records": len(records),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_epoch": time.time(), "status": "complete",
    }
    (root / "performance").mkdir(exist_ok=True)
    atomic_json(root / "performance" / (generation + ("_pilot" if pilot else "") + ".json"), performance)
    print(f"Published {'pilot' if pilot else 'training'} generation: {generation_dir}", flush=True)
    return generation_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="frozen Hydra used for training (default: MODELS/hydra/epoch_060.pt)")
    parser.add_argument("--normalization-json", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--seed", type=int, default=RELEASED_AUGMENTATION_SEED,
                        help="augmentation seed (released tracer: 20261011)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, help="pilot only; never publishes the training manifest")
    parser.add_argument("--draws-per-image", type=int, default=1,
                        help="augmented draws of each training image per generation")
    parser.add_argument("--cpu-workers", type=int, default=None,
                        help="CPU preparation/compression threads; zero preserves serial execution")
    args = parser.parse_args()
    workers = args.cpu_workers
    if workers is None:
        workers = int(os.environ.get("NOODLE_CACHE_CPU_WORKERS", "0"))
        settings = args.study_root / "cache_parallelism.json"
        if settings.is_file():
            config = json.loads(settings.read_text())
            if config.get("schema_version") != 1:
                raise ValueError("unsupported cache parallelism settings")
            if args.cycle >= config.get("effective_from_cycle", 1):
                workers = config["cpu_workers"]
    generate(
        study_root=args.study_root, checkpoint=args.checkpoint,
        normalization_json=args.normalization_json, cycle=args.cycle, seed=args.seed,
        device=args.device, batch_size=args.batch_size, limit=args.limit,
        draws_per_image=args.draws_per_image,
        cpu_workers=workers,
    )


if __name__ == "__main__":
    main()
