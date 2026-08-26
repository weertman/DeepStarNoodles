#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Prepare and cache the frozen 18-channel Noodle Tracer experiment.

This command intentionally separates immutable experiment preparation from
training and held-out evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / 'CORE'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from deepstarnoodles.dataset import HydraDenseDataset
from deepstarnoodles.model import HydraModel
from deepstarnoodles.recurrent_tracer_data import (
    FROZEN_CACHE_SCHEMA,
    cache_record_identity,
    prepared_arms_from_row,
    sha256_file,
    sleepy_rows,
    validate_sleepy_split_contract,
    write_cache_record,
)
from deepstarnoodles.train import collate


DEFAULT_SPLITS = REPO / "DATA/deepstarnoodles/splits"
DEFAULT_CHECKPOINT = REPO / "MODELS/hydra/epoch_060.pt"
DEFAULT_OUTPUT = REPO / "RUNS/noodle_tracer_sleepy18"
EXPECTED_COUNTS = {"train": 403, "validation": 34, "test": 63}
CACHE_SPLITS = ("train", "validation", "test")
CHANNEL_NAMES = (
    "arm_support_logit",
    "arm_centerline_logit",
    "arm_dx_raw",
    "arm_dy_raw",
    "arm_progress_logit",
    "mouth_logit",
    "madreporite_logit",
    "butthole_logit",
    "arm_base_logit",
    "arm_tip_logit",
    *(f"arm_embedding_{index}" for index in range(8)),
)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def parse_cache_splits(value: str) -> tuple[str, ...]:
    requested = [item.strip() for item in value.split(",")]
    if not requested or any(not item for item in requested):
        raise argparse.ArgumentTypeError(
            "--cache-splits must be a comma-separated list of train,validation,test"
        )
    unknown = sorted(set(requested) - set(CACHE_SPLITS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown cache split(s): {','.join(unknown)}; "
            "choose from train,validation,test"
        )
    if len(requested) != len(set(requested)):
        raise argparse.ArgumentTypeError("--cache-splits may not contain duplicates")
    return tuple(split for split in CACHE_SPLITS if split in requested)


def write_index(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty index")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def prepare_contract(
    *,
    output: Path,
    checkpoint: Path,
    split_dir: Path,
) -> dict[str, Any]:
    source_paths = {
        "train": split_dir / "train_index.csv",
        "validation": split_dir / "validation_index.csv",
        "test": split_dir / "test_index.csv",
    }
    rows = {name: sleepy_rows(path) for name, path in source_paths.items()}
    validate_sleepy_split_contract(rows["train"], rows["validation"], rows["test"])
    index_dir = output / "indexes"
    for name, split_rows in rows.items():
        write_index(index_dir / f"sleepy_{name}.csv", split_rows)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    embedding_dim = int(checkpoint_payload.get("config", {}).get("embedding_dim", -1))
    if embedding_dim != 8:
        raise ValueError(f"expected frozen 8-D embedding checkpoint, got {embedding_dim}")
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    materialized_hashes = {
        name: sha256_file(index_dir / f"sleepy_{name}.csv") for name in rows
    }
    contract = {
        "schema_version": 1,
        "experiment": "recurrent_local_tracer_sleepy18",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frozen_checkpoint": str(checkpoint.resolve()),
        "frozen_checkpoint_sha256": sha256_file(checkpoint),
        "dense_output_contract": {
            "channels": list(CHANNEL_NAMES),
            "geometry_slice": [0, 10],
            "embedding_slice": [10, 18],
            "geometry_representation": "legacy10_raw_logits",
            "embedding_normalized": False,
            "image_size": 512,
            "cache_schema": FROZEN_CACHE_SCHEMA,
            "aligned_rgb": {
                "channels": 3,
                "range": [0.0, 1.0],
                "storage_dtype": "uint8",
            },
        },
        "splits": {
            name: {
                "source": str(source_paths[name].resolve()),
                "source_sha256": source_hashes[name],
                "materialized": str((index_dir / f"sleepy_{name}.csv").resolve()),
                "materialized_sha256": materialized_hashes[name],
                "records": len(rows[name]),
            }
            for name in rows
        },
        "invariants": {
            "split_group_id_disjoint": True,
            "test_excluded_from_training_and_selection": True,
            "dense_checkpoint_frozen": True,
            "test_single_use_after_configuration_lock": True,
        },
        "baseline": {
            "decoder": "deepstarnoodles.decode_embed.decode_image_embed",
            "parameters": "EmbedDecodeParams defaults",
            "match_tolerance_px": 8.0,
        },
        "primary_model": {
            "patch_size": 33,
            "step_size_px": 1.0,
            "dense_input_channels": 18,
            "image_input_channels": 3,
            "input_representation": "aligned raw dense output plus RGB",
            "policy": "multimodal CNN-GRU",
        },
    }
    contract["contract_id"] = canonical_hash(
        {key: value for key, value in contract.items() if key != "created_at"}
    )
    atomic_json(output / "experiment_contract.json", contract)
    return contract


def validate_cache_contract(
    *,
    output: Path,
    checkpoint: Path,
    contract: dict[str, Any],
    splits: Sequence[str],
) -> None:
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != contract.get("frozen_checkpoint_sha256"):
        raise ValueError("frozen checkpoint hash differs from prepared contract")
    expected_contract_id = canonical_hash(
        {
            key: value
            for key, value in contract.items()
            if key not in {"created_at", "contract_id"}
        }
    )
    if contract.get("contract_id") != expected_contract_id:
        raise ValueError("prepared contract hash is invalid")
    for split in splits:
        split_contract = contract.get("splits", {}).get(split)
        if not isinstance(split_contract, dict):
            raise ValueError(f"prepared contract has no {split} split")
        index_path = output / "indexes" / f"sleepy_{split}.csv"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        if sha256_file(index_path) != split_contract.get("materialized_sha256"):
            raise ValueError(f"{split} index hash differs from prepared contract")


def load_train_statistics(
    *,
    output: Path,
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = output / "cache" / "train_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "train must be included in --cache-splits unless a validated "
            "train cache manifest already exists"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest.get("provenance", {})
    expected = contract["splits"]["train"]
    if (
        manifest.get("schema_version") != FROZEN_CACHE_SCHEMA
        or manifest.get("split") != "train"
        or provenance.get("contract_id") != contract["contract_id"]
        or provenance.get("frozen_checkpoint_sha256")
        != contract["frozen_checkpoint_sha256"]
        or provenance.get("index_sha256") != expected["materialized_sha256"]
        or len(manifest.get("records", [])) != expected["records"]
    ):
        raise ValueError(
            "stored train cache manifest does not match the prepared contract; "
            "include train in --cache-splits to regenerate it"
        )
    channel_statistics = provenance.get("channel_statistics", {})
    try:
        mean = np.asarray(channel_statistics["mean"], dtype=np.float64)
        std = np.asarray(channel_statistics["std"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("stored train cache manifest has invalid channel statistics") from error
    if (
        channel_statistics.get("source_split") != "train"
        or mean.shape != (18,)
        or std.shape != (18,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or not (std > 0).all()
    ):
        raise ValueError("stored train cache manifest has invalid channel statistics")
    return mean, std


def load_frozen_model(checkpoint_path: Path, device: torch.device) -> HydraModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = HydraModel(embedding_dim=8)
    state = next(
        (
            checkpoint[key]
            for key in ("model_ema", "ema", "ema_model")
            if isinstance(checkpoint.get(key), dict)
        ),
        checkpoint["model"],
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def cache_split(
    *,
    output: Path,
    split: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    index_sha256: str,
    device: str,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, int]:
    index_path = output / "indexes" / f"sleepy_{split}.csv"
    dataset = HydraDenseDataset(index_path, image_size=(512, 512), augment=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )
    torch_device = torch.device(device)
    model = load_frozen_model(checkpoint, torch_device)
    records: list[dict[str, Any]] = []
    channel_sum = np.zeros(18, dtype=np.float64)
    channel_sq_sum = np.zeros(18, dtype=np.float64)
    pixel_count = 0
    row_offset = 0
    cache_dir = output / "cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch in loader:
            image_batch = batch["image"].numpy()
            images = batch["image"].to(torch_device, non_blocking=True)
            with torch.autocast(
                device_type=torch_device.type,
                enabled=torch_device.type == "cuda",
            ):
                raw_batch = model(images).float().cpu().numpy()
            if raw_batch.shape[1] != 18:
                raise RuntimeError(f"frozen model emitted {raw_batch.shape[1]} channels")
            for local_index, raw in enumerate(raw_batch):
                row_index = row_offset + local_index
                row = dataset.rows[row_index]
                cache_name = f"{row_index:04d}_{hashlib.sha1(row['image_path'].encode()).hexdigest()[:12]}.npz"
                cache_path = cache_dir / cache_name
                arms = prepared_arms_from_row(row)
                image_sha256 = sha256_file(row["image_path"])
                shapes_sha256 = sha256_file(row["shapes_path"])
                cache_identity = canonical_hash(
                    {
                        "schema": FROZEN_CACHE_SCHEMA,
                        "checkpoint_sha256": checkpoint_sha256,
                        "index_sha256": index_sha256,
                        "row_index": row_index,
                        "image_path": row["image_path"],
                        "image_sha256": image_sha256,
                        "shapes_path": row["shapes_path"],
                        "shapes_sha256": shapes_sha256,
                    }
                )
                if (
                    force
                    or not cache_path.is_file()
                    or cache_record_identity(cache_path) != cache_identity
                ):
                    write_cache_record(
                        cache_path,
                        raw,
                        np.uint8(
                            np.clip(
                                image_batch[local_index].transpose(1, 2, 0),
                                0,
                                1,
                            )
                            * 255
                        ),
                        arms,
                        cache_identity=cache_identity,
                    )
                records.append(
                    {
                        "row_index": row_index,
                        "cache_file": cache_path.relative_to(output / "cache").as_posix(),
                        "image_path": row["image_path"],
                        "shapes_path": row["shapes_path"],
                        "split_group_id": row.get("split_group_id", ""),
                        "arm_count": len(arms),
                        "image_sha256": image_sha256,
                        "shapes_sha256": shapes_sha256,
                        "cache_identity": cache_identity,
                    }
                )
                flattened = raw.reshape(18, -1).astype(np.float64)
                channel_sum += flattened.sum(axis=1)
                channel_sq_sum += np.square(flattened).sum(axis=1)
                pixel_count += flattened.shape[1]
            row_offset += len(raw_batch)
    mean = channel_sum / max(pixel_count, 1)
    variance = channel_sq_sum / max(pixel_count, 1) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-8))
    return records, mean, std, pixel_count


def cache_all(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    selected_splits = tuple(getattr(args, "cache_splits", CACHE_SPLITS))
    required_contract_splits = set(selected_splits)
    if "train" not in selected_splits:
        required_contract_splits.add("train")
    validate_cache_contract(
        output=args.output,
        checkpoint=args.checkpoint,
        contract=contract,
        splits=tuple(split for split in CACHE_SPLITS if split in required_contract_splits),
    )
    statistics = (
        None
        if "train" in selected_splits
        else load_train_statistics(output=args.output, contract=contract)
    )
    pending_manifests: dict[str, tuple[list[dict[str, Any]], np.ndarray, np.ndarray, int]] = {}
    for split in selected_splits:
        result = cache_split(
            output=args.output,
            split=split,
            checkpoint=args.checkpoint,
            checkpoint_sha256=contract["frozen_checkpoint_sha256"],
            index_sha256=contract["splits"][split]["materialized_sha256"],
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force=args.force,
        )
        pending_manifests[split] = result
        if split == "train":
            statistics = (result[1], result[2])
    assert statistics is not None
    train_mean, train_std = statistics
    for split, (records, _, _, pixel_count) in pending_manifests.items():
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
                "channel_statistics": {
                    "source_split": "train",
                    "mean": train_mean.tolist(),
                    "std": train_std.tolist(),
                },
            },
        }
        atomic_json(args.output / "cache" / f"{split}_manifest.json", manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "cache", "all"), default="prepare"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--cache-splits",
        type=parse_cache_splits,
        default=CACHE_SPLITS,
        metavar="SPLITS",
        help="comma-separated cache splits (default: train,validation,test)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.split_dir = args.split_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    contract_path = args.output / "experiment_contract.json"
    if args.stage in {"prepare", "all"} or not contract_path.is_file():
        contract = prepare_contract(
            output=args.output,
            checkpoint=args.checkpoint,
            split_dir=args.split_dir,
        )
    else:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if sha256_file(args.checkpoint) != contract["frozen_checkpoint_sha256"]:
            raise ValueError("frozen checkpoint hash differs from prepared contract")
    if args.stage in {"cache", "all"}:
        cache_all(args, contract)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "output": str(args.output),
                "contract_id": contract["contract_id"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
