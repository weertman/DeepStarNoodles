#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Deploy DeepStarNoodles on a video: StarSeg crop, 18-ch Hydra, Noodle Tracer, fused tracks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

from deepstarnoodles.crop import CropConfig, StarCropper
from deepstarnoodles.fused_tracker import FusedTracker
from deepstarnoodles.hydra import HydraInferConfig, HydraPredictor
from deepstarnoodles.paths import (
    DEFAULT_HYDRA_CHECKPOINT,
    DEFAULT_STARSEG,
    DEFAULT_TRACER_CHECKPOINT,
)
from deepstarnoodles.noodle_tracer import (
    load_noodle_tracer_checkpoint,
    trace_arms_noodle_batch,
)
from deepstarnoodles.runtime import (
    draw_overlay,
    source_arm_records,
    track_arms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--starseg", type=Path, default=DEFAULT_STARSEG)
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA_CHECKPOINT)
    parser.add_argument("--tracer", type=Path, default=DEFAULT_TRACER_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dense/tracer micro-batch. Keep 1 for minimum live latency.",
    )
    parser.add_argument("--shape-gate", type=float, default=120.0)
    parser.add_argument("--embed-weight", type=float, default=2.0)
    parser.add_argument("--max-age", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = args.device
    cropper = StarCropper(
        args.starseg,
        CropConfig(device=device if device.startswith("cuda") else "cpu"),
    )
    hydra = HydraPredictor(args.hydra, HydraInferConfig(device=device))
    tracer, tracer_params, _meta, _train = load_noodle_tracer_checkpoint(
        str(args.tracer), device=device
    )
    tracker = FusedTracker(
        shape_gate=args.shape_gate,
        embed_weight=args.embed_weight,
        max_age=args.max_age,
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = None
    if args.overlay is not None:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)

    frames: list[dict] = []
    pending: list[dict] = []
    next_frame = 0

    def flush_pending() -> None:
        nonlocal writer
        detected = [item for item in pending if item["crop"] is not None]
        decoded_by_frame: dict[int, tuple[np.ndarray, list[dict]]] = {}
        if detected:
            crops_rgb = [item["crop"].crop_rgb for item in detected]
            image_batch = hydra.preprocess_batch(crops_rgb).to(hydra.device)
            raw_batch, activated_batch = hydra.predict_tensor_batch(image_batch)
            decoded_batch = trace_arms_noodle_batch(
                raw_batch,
                image_batch,
                tracer,
                tracer_params,
            )
            for item, activated, arms in zip(
                detected,
                activated_batch,
                decoded_batch,
            ):
                decoded_by_frame[item["frame"]] = (
                    activated.cpu().numpy(),
                    arms,
                )

        for item in pending:
            frame_index = int(item["frame"])
            frame_bgr = item["frame_bgr"]
            crop = item["crop"]
            record: dict = {
                "frame": frame_index,
                "detected": crop is not None,
                "crop": None,
                "arms": [],
            }
            if crop is not None:
                activated, arms = decoded_by_frame[frame_index]
                labels = track_arms(
                    tracker,
                    frame_index,
                    activated,
                    arms,
                )
                record["crop"] = {
                    "left": crop.crop_left,
                    "top": crop.crop_top,
                    "side": crop.crop_side,
                    "confidence": crop.confidence,
                    "bbox_xyxy": list(crop.bbox_xyxy),
                }
                record["arms"] = source_arm_records(
                    arms,
                    labels,
                    crop,
                    model_size=hydra.image_size,
                )
            if args.overlay is not None:
                overlay = draw_overlay(frame_bgr, crop, record["arms"])
                if writer is None:
                    height, width = overlay.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.overlay),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                writer.write(overlay)
            frames.append(record)
        pending.clear()

    with torch.inference_mode():
        while True:
            if args.max_frames is not None and next_frame >= args.max_frames:
                break
            ok, frame_bgr = capture.read()
            if not ok:
                break
            crop = cropper.crop(frame_bgr)
            pending.append(
                {
                    "frame": next_frame,
                    "frame_bgr": frame_bgr,
                    "crop": crop,
                }
            )
            next_frame += 1
            if len(pending) >= args.batch_size:
                flush_pending()
        if pending:
            flush_pending()
    capture.release()
    if writer is not None:
        writer.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"video": str(args.video), "frames": frames}, indent=2) + "\n")
    print(f"wrote {len(frames)} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
