# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Cached frozen-output data and supervision for the Noodle Tracer."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from deepstarnoodles.dataset import _chaikin_smooth, _resample_polyline, _shape_points


FROZEN_CACHE_SCHEMA = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sleepy_rows(index_path: str | Path) -> list[dict[str, str]]:
    with Path(index_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row.get("source_family") == "sleepy_stars"]


def split_group_ids(rows: Iterable[dict[str, str]]) -> set[str]:
    return {row["split_group_id"] for row in rows if row.get("split_group_id")}


def validate_sleepy_split_contract(
    train_rows: Sequence[dict[str, str]],
    validation_rows: Sequence[dict[str, str]],
    test_rows: Sequence[dict[str, str]],
    *,
    expected_counts: tuple[int, int, int] = (403, 34, 63),
) -> None:
    actual = (len(train_rows), len(validation_rows), len(test_rows))
    if actual != expected_counts:
        raise ValueError(f"Sleepy split counts changed: {actual} != {expected_counts}")
    groups = [split_group_ids(rows) for rows in (train_rows, validation_rows, test_rows)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("split_group_id leakage in Sleepy tracer indexes")


def prepared_arms_from_row(row: dict[str, str], image_size: int = 512) -> list[np.ndarray]:
    shapes = json.loads(Path(row["shapes_path"]).read_text(encoding="utf-8"))
    sx = image_size / float(row["width"])
    sy = image_size / float(row["height"])
    arms = []
    for shape in shapes:
        if (
            shape.get("outside")
            or shape.get("label") != "arm"
            or shape.get("type") != "polyline"
        ):
            continue
        points = _shape_points(shape)
        if len(points) < 2:
            continue
        points[:, 0] *= sx
        points[:, 1] *= sy
        arms.append(
            _resample_polyline(_chaikin_smooth(points, iterations=2), spacing=1.0)
        )
    return arms


def pack_polylines(polylines: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    points = []
    for polyline in polylines:
        value = np.asarray(polyline, dtype=np.float32).reshape(-1, 2)
        points.append(value)
        offsets.append(offsets[-1] + len(value))
    packed = np.concatenate(points, axis=0) if points else np.zeros((0, 2), dtype=np.float32)
    return packed, np.asarray(offsets, dtype=np.int32)


def unpack_polylines(points: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(points[int(start):int(stop)], dtype=np.float32).copy()
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]


def write_cache_record(
    path: str | Path,
    raw: np.ndarray,
    image: np.ndarray,
    arms: Sequence[np.ndarray],
    *,
    cache_identity: str,
) -> None:
    points, offsets = pack_polylines(arms)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray([FROZEN_CACHE_SCHEMA], dtype=np.int16),
            cache_identity=np.asarray([cache_identity]),
            raw=np.asarray(raw, dtype=np.float16),
            image=np.asarray(image, dtype=np.uint8),
            arm_points=points,
            arm_offsets=offsets,
        )
    temporary.replace(path)


def cache_record_identity(path: str | Path) -> str | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            value = payload.get("cache_identity")
            return str(value[0]) if value is not None and len(value) else None
    except (OSError, ValueError, KeyError):
        return None


def read_cache_record(
    path: str | Path,
    *,
    expected_identity: str | None = None,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    raw, _, arms = read_cache_record_full(
        path,
        expected_identity=expected_identity,
    )
    return raw, arms


def read_cache_record_full(
    path: str | Path,
    *,
    expected_identity: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        schema = int(payload["schema"][0])
        if schema != FROZEN_CACHE_SCHEMA:
            raise ValueError(f"unsupported recurrent tracer cache schema {schema}")
        identity_value = payload.get("cache_identity")
        identity = (
            str(identity_value[0])
            if identity_value is not None and len(identity_value)
            else None
        )
        if expected_identity is not None and identity != expected_identity:
            raise ValueError(
                f"cache record identity mismatch: {identity!r} != {expected_identity!r}"
            )
        raw = torch.from_numpy(payload["raw"].astype(np.float32))
        image = torch.from_numpy(payload["image"].astype(np.float32) / 255.0)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("cached RGB image must be (H,W,3)")
        image = image.permute(2, 0, 1).contiguous()
        arms = unpack_polylines(payload["arm_points"], payload["arm_offsets"])
    return raw, image, arms


@dataclass(frozen=True)
class OutputAugment:
    enabled: bool = False
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate90: bool = True
    noise_std: float = 0.02
    dropout_p: float = 0.2
    dropout_size: int = 12


def transform_field_and_arms(
    raw: torch.Tensor,
    arms: Sequence[np.ndarray],
    *,
    flip_x: bool = False,
    flip_y: bool = False,
    rotate_k: int = 0,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """Apply exact square-grid transforms, including tangent-vector semantics."""
    if raw.ndim != 3 or raw.shape[-2] != raw.shape[-1]:
        raise ValueError("raw output must be square (C,H,W)")
    output = raw.clone()
    paths = [np.asarray(path, dtype=np.float32).copy() for path in arms]
    size = int(raw.shape[-1])
    if flip_x:
        output = output.flip(-1)
        output[2] = -output[2]
        for path in paths:
            path[:, 0] = size - 1 - path[:, 0]
    if flip_y:
        output = output.flip(-2)
        output[3] = -output[3]
        for path in paths:
            path[:, 1] = size - 1 - path[:, 1]
    for _ in range(int(rotate_k) % 4):
        output = torch.rot90(output, 1, (-2, -1))
        dx = output[2].clone()
        output[2] = output[3]
        output[3] = -dx
        for path in paths:
            old_x = path[:, 0].copy()
            path[:, 0] = path[:, 1]
            path[:, 1] = size - 1 - old_x
    return output, paths


def augment_field_and_arms(
    raw: torch.Tensor,
    arms: Sequence[np.ndarray],
    config: OutputAugment,
    *,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    output, _, paths = augment_sample(raw, None, arms, config, rng=rng)
    return output, paths


def augment_sample(
    raw: torch.Tensor,
    image: torch.Tensor | None,
    arms: Sequence[np.ndarray],
    config: OutputAugment,
    *,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, list[np.ndarray]]:
    if not config.enabled:
        return (
            raw,
            image,
            [np.asarray(path, dtype=np.float32).copy() for path in arms],
        )
    rng = rng or random
    flip_x = rng.random() < config.horizontal_flip_p
    flip_y = rng.random() < config.vertical_flip_p
    rotate_k = rng.randrange(4) if config.rotate90 else 0
    output, paths = transform_field_and_arms(
        raw,
        arms,
        flip_x=flip_x,
        flip_y=flip_y,
        rotate_k=rotate_k,
    )
    image_output = image
    if image_output is not None:
        if flip_x:
            image_output = image_output.flip(-1)
        if flip_y:
            image_output = image_output.flip(-2)
        if rotate_k:
            image_output = torch.rot90(image_output, rotate_k, (-2, -1))
    if config.noise_std > 0:
        output = output + torch.randn_like(output) * config.noise_std
    if (
        config.dropout_p > 0
        and config.dropout_size > 0
        and rng.random() < config.dropout_p
    ):
        height, width = output.shape[-2:]
        side = min(config.dropout_size, height, width)
        x0 = rng.randrange(max(1, width - side + 1))
        y0 = rng.randrange(max(1, height - side + 1))
        output[:, y0:y0 + side, x0:x0 + side] = 0
    return output, image_output, paths


class FrozenOutputDataset(Dataset):
    """One cached dense field and all annotated arms per item."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        augment: OutputAugment | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != FROZEN_CACHE_SCHEMA:
            raise ValueError("unsupported cache manifest schema")
        self.records = list(manifest["records"])
        self.provenance = dict(manifest["provenance"])
        self.augment = augment or OutputAugment(enabled=False)
        if not self.records:
            raise ValueError("empty frozen-output cache manifest")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.manifest_path.parent / record["cache_file"]
        raw, image, arms = read_cache_record_full(
            path,
            expected_identity=record.get("cache_identity"),
        )
        raw, image, arms = augment_sample(raw, image, arms, self.augment)
        return {"raw": raw, "image": image, "arms": arms, "record": record}


def collate_frozen_outputs(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "raw": torch.stack([item["raw"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "arms": [item["arms"] for item in batch],
        "records": [item["record"] for item in batch],
    }


def channel_statistics(manifest_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = manifest["provenance"].get("channel_statistics")
    if not isinstance(stats, dict):
        raise ValueError("cache manifest has no training channel statistics")
    return np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["std"], dtype=np.float32)


def validate_cache_manifest(
    manifest_path: str | Path,
    contract: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != FROZEN_CACHE_SCHEMA:
        raise ValueError(f"unsupported cache manifest schema: {path}")
    if manifest.get("split") != split:
        raise ValueError(f"cache manifest split mismatch: {path}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"cache manifest lacks provenance: {path}")
    expected = {
        "contract_id": contract["contract_id"],
        "frozen_checkpoint_sha256": contract["frozen_checkpoint_sha256"],
        "index_sha256": contract["splits"][split]["materialized_sha256"],
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"cache manifest {key} mismatch: {provenance.get(key)!r} != {value!r}"
            )
    records = manifest.get("records", [])
    if len(records) != int(contract["splits"][split]["records"]):
        raise ValueError(f"cache manifest record count differs from contract: {path}")
    required_record_fields = {
        "cache_identity",
        "image_sha256",
        "shapes_sha256",
    }
    if any(not required_record_fields <= set(record) for record in records):
        raise ValueError(f"cache manifest records lack source identities: {path}")
    return manifest
