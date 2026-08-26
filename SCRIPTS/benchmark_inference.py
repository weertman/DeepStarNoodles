#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Benchmark synchronized DeepStarNoodles inference stages on fixed test images."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

from deepstarnoodles.decode import find_peaks
from deepstarnoodles.fused_tracker import FusedTracker, tip_anchored_resample
from deepstarnoodles.hydra import HydraInferConfig, HydraPredictor
from deepstarnoodles.paths import (
    DEFAULT_HYDRA_CHECKPOINT,
    DEFAULT_STARSEG,
    DEFAULT_TEST_INDEX,
    DEFAULT_TRACER_CHECKPOINT,
)
from deepstarnoodles.noodle_tracer import (
    load_noodle_tracer_checkpoint,
    trace_arms_noodle,
)


K_TRACK = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_TEST_INDEX)
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA_CHECKPOINT)
    parser.add_argument("--tracer", type=Path, default=DEFAULT_TRACER_CHECKPOINT)
    parser.add_argument("--starseg", type=Path, default=DEFAULT_STARSEG)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), default=1)
    parser.add_argument(
        "--engine",
        choices=("sequential", "batch"),
        default="sequential",
        help="Sequential is the pre-optimization reference; batch requires B>1 APIs.",
    )
    parser.add_argument("--include-starseg", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StageTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.values: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.values[name].append(1000.0 * (time.perf_counter() - started))


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean_ms": None, "median_ms": None, "p90_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(len(array)),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.percentile(array, 90)),
    }


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))


def body_center(activated: np.ndarray) -> np.ndarray:
    mouth = find_peaks(activated[5], threshold=0.15, nms_radius=8)
    if len(mouth):
        return mouth[0, :2].astype(np.float32)
    support = activated[0] > 0.3
    ys, xs = np.nonzero(support)
    if len(xs):
        return np.asarray([xs.mean(), ys.mean()], dtype=np.float32)
    height, width = activated.shape[-2:]
    return np.asarray([width / 2.0, height / 2.0], dtype=np.float32)


def track_frame(
    tracker: FusedTracker,
    frame_index: int,
    activated: np.ndarray,
    arms: list[dict],
) -> list[int]:
    if not arms:
        return []
    center = body_center(activated)
    sampled: list[np.ndarray] = []
    lengths: list[float] = []
    identities: list[np.ndarray] = []
    for arm in arms:
        points, length = tip_anchored_resample(
            np.asarray(arm["polyline"], dtype=np.float32) - center,
            K_TRACK,
        )
        sampled.append(points)
        lengths.append(length)
        identities.append(np.asarray(arm["identity"], dtype=np.float32))
    return tracker.step(
        frame_index,
        np.asarray(sampled, dtype=np.float32),
        np.asarray(lengths, dtype=np.float32),
        np.asarray(identities, dtype=np.float32),
    )


def gpu_inventory(device: torch.device) -> dict:
    inventory = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        inventory.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return inventory


def load_rows(path: Path, count: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # Avoid a benchmark made only of tiny StarSeg cluster crops.
    preferred = [
        row
        for row in rows
        if min(int(row.get("width", 0)), int(row.get("height", 0))) >= 256
    ]
    selected = preferred[:count]
    if len(selected) < count:
        selected.extend(rows[: count - len(selected)])
    return selected


def main() -> int:
    args = parse_args()
    if args.images <= 0 or args.warmup < 0:
        raise ValueError("images must be positive and warmup non-negative")
    if args.engine == "batch" and args.batch_size == 1:
        raise ValueError("batch engine is only useful with --batch-size 4")

    device = torch.device(args.device)
    rows = load_rows(args.index, args.images + args.warmup)
    if len(rows) < args.images + args.warmup:
        raise RuntimeError("benchmark index does not contain enough rows")
    images = [read_rgb(Path(row["image_path"])) for row in rows]

    predictor = HydraPredictor(
        args.hydra,
        HydraInferConfig(device=args.device, amp=True),
    )
    tracer, trace_params, tracer_metadata, _training = load_noodle_tracer_checkpoint(
        str(args.tracer),
        device=args.device,
    )
    tracker = FusedTracker(shape_gate=120.0, embed_weight=2.0, max_age=16)
    starseg = None
    if args.include_starseg:
        from ultralytics import YOLO

        starseg = YOLO(str(args.starseg))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def infer_one(rgb: np.ndarray, frame_index: int, timer: StageTimer) -> int:
        if starseg is not None:
            with timer.stage("starseg"):
                starseg.predict(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    imgsz=1024,
                    conf=0.25,
                    iou=0.7,
                    device=args.device,
                    verbose=False,
                )
        with timer.stage("hydra"):
            aligned_batch = predictor.preprocess(rgb)
            raw_batch, activated_batch = predictor.predict_tensor_batch(
                aligned_batch
            )
            raw = raw_batch[0]
            activated = activated_batch[0]
            aligned = aligned_batch[0].to(predictor.device)
        with timer.stage("noodle_tracer"):
            arms = trace_arms_noodle(
                raw,
                aligned,
                tracer,
                trace_params,
            )
        with timer.stage("fused_tracker"):
            track_frame(tracker, frame_index, activated.cpu().numpy(), arms)
        return len(arms)

    # Baseline warmup uses the exact same single-image path regardless of the
    # requested measurement engine.
    warmup_timer = StageTimer(device)
    with torch.inference_mode():
        for index, rgb in enumerate(images[: args.warmup]):
            infer_one(rgb, index, warmup_timer)

    measured = images[args.warmup :]
    timer = StageTimer(device)
    group_ms: list[float] = []
    arm_counts: list[int] = []
    with torch.inference_mode():
        if args.engine == "sequential":
            for start in range(0, len(measured), args.batch_size):
                group = measured[start : start + args.batch_size]
                synchronize(device)
                started = time.perf_counter()
                for offset, rgb in enumerate(group):
                    arm_counts.append(infer_one(rgb, start + offset, timer))
                synchronize(device)
                group_ms.append(1000.0 * (time.perf_counter() - started))
        else:
            if not hasattr(predictor, "predict_tensor_batch"):
                raise RuntimeError(
                    "batch engine requires HydraPredictor.predict_tensor_batch"
                )
            from deepstarnoodles.noodle_tracer import trace_arms_noodle_batch

            for start in range(0, len(measured), args.batch_size):
                group = measured[start : start + args.batch_size]
                synchronize(device)
                started = time.perf_counter()
                if starseg is not None:
                    with timer.stage("starseg"):
                        starseg.predict(
                            [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in group],
                            imgsz=1024,
                            conf=0.25,
                            iou=0.7,
                            device=args.device,
                            verbose=False,
                        )
                with timer.stage("hydra"):
                    aligned_batch = predictor.preprocess_batch(group)
                    raw_batch, activated_batch = predictor.predict_tensor_batch(
                        aligned_batch
                    )
                with timer.stage("noodle_tracer"):
                    decoded_batch = trace_arms_noodle_batch(
                        raw_batch,
                        aligned_batch,
                        tracer,
                        trace_params,
                    )
                for offset, arms in enumerate(decoded_batch):
                    with timer.stage("fused_tracker"):
                        track_frame(
                            tracker,
                            start + offset,
                            activated_batch[offset].cpu().numpy(),
                            arms,
                        )
                    arm_counts.append(len(arms))
                synchronize(device)
                group_ms.append(1000.0 * (time.perf_counter() - started))

    total_seconds = sum(group_ms) / 1000.0
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inventory": gpu_inventory(device),
        "configuration": {
            "engine": args.engine,
            "batch_size": args.batch_size,
            "images": len(measured),
            "warmup": args.warmup,
            "include_starseg": args.include_starseg,
            "index": str(args.index.resolve()),
            "hydra_checkpoint": str(args.hydra.resolve()),
            "hydra_sha256": sha256_file(args.hydra),
            "tracer_checkpoint": str(args.tracer.resolve()),
            "tracer_sha256": sha256_file(args.tracer),
            "starseg_checkpoint": str(args.starseg.resolve()),
            "starseg_sha256": sha256_file(args.starseg),
            "trace_params": {
                "max_steps": trace_params.max_steps,
                "invalid_seed_threshold": trace_params.invalid_seed_threshold,
                "dense_patch_size": trace_params.dense_patch_size,
            },
            "tracer_metadata": jsonable(tracer_metadata),
            "image_paths": [str(Path(row["image_path"]).resolve()) for row in rows[args.warmup :]],
        },
        "stages": {
            name: summarize(values)
            for name, values in sorted(timer.values.items())
        },
        "groups": summarize(group_ms),
        "throughput_fps": len(measured) / max(total_seconds, 1e-12),
        "arm_count": {
            "mean": statistics.fmean(arm_counts) if arm_counts else 0.0,
            "min": min(arm_counts, default=0),
            "max": max(arm_counts, default=0),
        },
        "peak_memory": {
            "allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
