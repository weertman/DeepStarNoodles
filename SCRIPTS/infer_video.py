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

from deepstarnoodles.crop import (
    CropConfig,
    CropResult,
    StarCropper,
    TrackedStarCropper,
)
from deepstarnoodles.fused_tracker import FusedTracker
from deepstarnoodles.hydra import HydraInferConfig, HydraPredictor
from deepstarnoodles.paths import (
    DEFAULT_BOTSORT_CONFIG,
    DEFAULT_HYDRA_CHECKPOINT,
    DEFAULT_STARSEG,
    DEFAULT_TRACER_CHECKPOINT,
)
from deepstarnoodles.noodle_tracer import (
    load_noodle_tracer_checkpoint,
    trace_arms_noodle_batch,
)
from deepstarnoodles.runtime import (
    StarArmTrackerRegistry,
    crop_record,
    draw_multi_overlay,
    draw_overlay,
    multi_star_video_payload,
    source_arm_records,
    track_arms,
    tracked_star_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--starseg", type=Path, default=DEFAULT_STARSEG)
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA_CHECKPOINT)
    parser.add_argument("--tracer", type=Path, default=DEFAULT_TRACER_CHECKPOINT)
    parser.add_argument(
        "--multi-star",
        action="store_true",
        help="Track and trace every sunflower star with BoT-SORT.",
    )
    parser.add_argument(
        "--star-tracker-config",
        type=Path,
        default=DEFAULT_BOTSORT_CONFIG,
        help="Pinned Ultralytics BoT-SORT configuration for --multi-star.",
    )
    parser.add_argument(
        "--overlay-support",
        action="store_true",
        help="Blend Hydra support masks into a multi-star overlay.",
    )
    parser.add_argument("--support-threshold", type=float, default=0.3)
    parser.add_argument("--support-alpha", type=float, default=0.28)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dense/tracer crop micro-batch. Keep 1 for minimum live latency.",
    )
    parser.add_argument("--shape-gate", type=float, default=120.0)
    parser.add_argument("--embed-weight", type=float, default=2.0)
    parser.add_argument("--max-age", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.overlay_support and not args.multi_star:
        raise ValueError("--overlay-support requires --multi-star")
    if not 0.0 <= args.support_threshold <= 1.0:
        raise ValueError("--support-threshold must be between 0 and 1")
    if not 0.0 <= args.support_alpha <= 1.0:
        raise ValueError("--support-alpha must be between 0 and 1")
    device = args.device
    crop_config = CropConfig(
        device=device if device.startswith("cuda") else "cpu"
    )
    cropper: StarCropper | TrackedStarCropper
    if args.multi_star:
        cropper = TrackedStarCropper(
            args.starseg,
            crop_config,
            tracker_config=args.star_tracker_config,
        )
    else:
        cropper = StarCropper(args.starseg, crop_config)
    hydra = HydraPredictor(args.hydra, HydraInferConfig(device=device))
    tracer, tracer_params, _meta, _train = load_noodle_tracer_checkpoint(
        str(args.tracer), device=device
    )
    tracker = (
        None
        if args.multi_star
        else FusedTracker(
            shape_gate=args.shape_gate,
            embed_weight=args.embed_weight,
            max_age=args.max_age,
        )
    )
    star_trackers = StarArmTrackerRegistry(
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
        observations: list[tuple[dict, int, CropResult]] = [
            (item, crop_index, crop)
            for item in pending
            for crop_index, crop in enumerate(item["crops"])
        ]
        decoded: dict[tuple[int, int], tuple[np.ndarray, list[dict]]] = {}
        for start in range(0, len(observations), args.batch_size):
            batch = observations[start : start + args.batch_size]
            crops_rgb = [crop.crop_rgb for _item, _index, crop in batch]
            image_batch = hydra.preprocess_batch(crops_rgb).to(hydra.device)
            raw_batch, activated_batch = hydra.predict_tensor_batch(image_batch)
            decoded_batch = trace_arms_noodle_batch(
                raw_batch,
                image_batch,
                tracer,
                tracer_params,
            )
            for (item, crop_index, _crop), activated, arms in zip(
                batch,
                activated_batch,
                decoded_batch,
            ):
                decoded[(int(item["frame"]), crop_index)] = (
                    activated.cpu().numpy(),
                    arms,
                )

        for item in pending:
            frame_index = int(item["frame"])
            frame_bgr = item["frame_bgr"]
            crops: list[CropResult] = item["crops"]
            if args.multi_star:
                star_records: list[dict] = []
                overlay_stars: list[tuple[CropResult, list[dict]]] = []
                support_masks: list[np.ndarray] = []
                for crop_index, crop in enumerate(crops):
                    if crop.star_track_id is None:
                        raise RuntimeError("BoT-SORT crop is missing a star track id")
                    activated, arms = decoded[(frame_index, crop_index)]
                    labels = star_trackers.track(
                        crop.star_track_id,
                        frame_index,
                        activated,
                        arms,
                    )
                    arm_records = source_arm_records(
                        arms,
                        labels,
                        crop,
                        model_size=hydra.image_size,
                    )
                    star_records.append(tracked_star_record(crop, arm_records))
                    overlay_stars.append((crop, arm_records))
                    support_masks.append(activated[0])
                star_trackers.prune(frame_index)
                record = {
                    "frame": frame_index,
                    "detected": bool(crops),
                    "stars": star_records,
                }
                overlay = draw_multi_overlay(
                    frame_bgr,
                    overlay_stars,
                    support_masks=support_masks if args.overlay_support else None,
                    support_threshold=args.support_threshold,
                    support_alpha=args.support_alpha,
                )
            else:
                crop = crops[0] if crops else None
                record = {
                    "frame": frame_index,
                    "detected": crop is not None,
                    "crop": None,
                    "arms": [],
                }
                if crop is not None:
                    if tracker is None:
                        raise RuntimeError("single-star arm tracker is not initialized")
                    activated, arms = decoded[(frame_index, 0)]
                    labels = track_arms(
                        tracker,
                        frame_index,
                        activated,
                        arms,
                    )
                    record["crop"] = crop_record(crop)
                    record["arms"] = source_arm_records(
                        arms,
                        labels,
                        crop,
                        model_size=hydra.image_size,
                    )
                overlay = draw_overlay(frame_bgr, crop, record["arms"])
            if args.overlay is not None:
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

    pending_crop_count = 0
    try:
        with torch.inference_mode():
            while True:
                if args.max_frames is not None and next_frame >= args.max_frames:
                    break
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if args.multi_star:
                    crops = cropper.track_crops(frame_bgr)
                else:
                    crop = cropper.crop(frame_bgr)
                    crops = [] if crop is None else [crop]
                pending.append(
                    {
                        "frame": next_frame,
                        "frame_bgr": frame_bgr,
                        "crops": crops,
                    }
                )
                pending_crop_count += len(crops)
                next_frame += 1
                if (
                    len(pending) >= args.batch_size
                    or pending_crop_count >= args.batch_size
                ):
                    flush_pending()
                    pending_crop_count = 0
            if pending:
                flush_pending()
    finally:
        capture.release()
        cropper.reset()
        if writer is not None:
            writer.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"video": str(args.video), "frames": frames}
    if args.multi_star:
        payload = multi_star_video_payload(
            args.video,
            cropper.tracker_config,
            frames,
        )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(frames)} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
