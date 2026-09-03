#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Measure Hydra B=1/4/8 throughput and VRAM versus YOLO-1024 archival latency."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

from deepstarnoodles.hydra import HydraInferConfig, HydraPredictor
from deepstarnoodles.hydra_budget import (
    CANDIDATES,
    MEASURED_BUDGET_PATH,
    YOLO_P99_LIMIT_MS,
    budget_payload,
    write_budget,
)
from deepstarnoodles.paths import DEFAULT_HYDRA_CHECKPOINT, DEFAULT_STARSEG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--output", type=Path, default=MEASURED_BUDGET_PATH)
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA_CHECKPOINT)
    parser.add_argument("--yolo", type=Path, default=DEFAULT_STARSEG)
    parser.add_argument("--skip-yolo", action="store_true")
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_vram(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _yolo_p99_ms(model, images: list[np.ndarray], device: str) -> float:
    latencies: list[float] = []
    for image in images:
        start = time.perf_counter()
        model.predict(image, imgsz=1024, conf=0.25, device=device, verbose=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return float(np.percentile(latencies, 99))


def measure_batch(
    predictor: HydraPredictor,
    batch: int,
    *,
    warmup: int,
    repeats: int,
    yolo,
    yolo_images: list[np.ndarray] | None,
    yolo_device: str,
) -> dict:
    device = predictor.device
    size = predictor.image_size
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    crop[:, :, 1] = 128
    crops = [crop] * batch
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        predictor.predict_raw_batch(crops)
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        predictor.predict_raw_batch(crops)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    images = batch * repeats
    result = {
        "batch_size": batch,
        "images": images,
        "seconds": elapsed,
        "images_per_s": images / elapsed if elapsed else 0.0,
        "hydra_vram_bytes": _peak_vram(device),
        "gpu_total_bytes": int(torch.cuda.get_device_properties(device).total_memory)
        if device.type == "cuda"
        else None,
        "yolo_p99_ms": None,
    }
    if yolo is not None and yolo_images:
        result["yolo_p99_ms"] = _yolo_p99_ms(yolo, yolo_images, yolo_device)
    return result


def main() -> int:
    args = parse_args()
    cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    if not cuda:
        payload = budget_payload(
            {},
            status="deferred_no_cuda",
            device=str(args.device),
            note="nvidia/CUDA unavailable; keep default B=4 until this script runs on GPU.",
        )
        payload["recommended_batch_size"] = 4
        write_budget(payload, args.output)
        print(json_dump(payload))
        return 0

    predictor = HydraPredictor(args.hydra, HydraInferConfig(device=args.device))
    yolo = None
    yolo_images = None
    if not args.skip_yolo and args.yolo.is_file():
        from ultralytics import YOLO

        yolo = YOLO(str(args.yolo))
        yolo_images = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(8)]
    results: dict[int, dict] = {}
    for batch in reversed(CANDIDATES):
        try:
            results[batch] = measure_batch(
                predictor,
                batch,
                warmup=args.warmup,
                repeats=args.repeats,
                yolo=yolo,
                yolo_images=yolo_images,
                yolo_device=args.device,
            )
        except Exception as exc:
            results[batch] = {"error": str(exc)}
    payload = budget_payload(
        results,
        status="measured",
        device=args.device,
        yolo_p99_limit_ms=YOLO_P99_LIMIT_MS,
        note="Largest B whose concurrent YOLO p99 stays under the 4 fps archival budget.",
    )
    write_budget(payload, args.output)
    print(json_dump(payload))
    return 0


def json_dump(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
