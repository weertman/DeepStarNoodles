#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Round-robin multiple videos through one batched DeepStarNoodles process."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
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


@dataclass
class VideoState:
    index: int
    path: Path
    capture: cv2.VideoCapture
    cropper: StarCropper | TrackedStarCropper
    tracker: FusedTracker | None
    star_trackers: StarArmTrackerRegistry | None
    fps: float
    frames: list[dict] = field(default_factory=list)
    next_frame: int = 0
    done: bool = False
    writer: cv2.VideoWriter | None = None


@dataclass
class PendingFrame:
    stream: VideoState
    frame_index: int
    frame_bgr: np.ndarray
    crops: list[CropResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--starseg", type=Path, default=DEFAULT_STARSEG)
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA_CHECKPOINT)
    parser.add_argument("--tracer", type=Path, default=DEFAULT_TRACER_CHECKPOINT)
    parser.add_argument(
        "--multi-star",
        action="store_true",
        help="Track and trace every sunflower star with per-stream BoT-SORT.",
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
        help="Blend Hydra support masks into multi-star overlays.",
    )
    parser.add_argument("--support-threshold", type=float, default=0.3)
    parser.add_argument("--support-alpha", type=float, default=0.28)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum Hydra/tracer crops per GPU micro-batch.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--shape-gate", type=float, default=120.0)
    parser.add_argument("--embed-weight", type=float, default=2.0)
    parser.add_argument("--max-age", type=int, default=16)
    return parser.parse_args()


def output_stem(state: VideoState) -> str:
    return f"{state.index:02d}_{state.path.stem}"


def write_overlay(
    state: VideoState,
    overlay_dir: Path,
    image: np.ndarray,
) -> None:
    if state.writer is None:
        height, width = image.shape[:2]
        state.writer = cv2.VideoWriter(
            str(overlay_dir / f"{output_stem(state)}_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            state.fps,
            (width, height),
        )
    state.writer.write(image)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    if args.overlay_support and not args.multi_star:
        raise ValueError("--overlay-support requires --multi-star")
    if not 0.0 <= args.support_threshold <= 1.0:
        raise ValueError("--support-threshold must be between 0 and 1")
    if not 0.0 <= args.support_alpha <= 1.0:
        raise ValueError("--support-alpha must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir is not None:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    crop_config = CropConfig(
        device=args.device if args.device.startswith("cuda") else "cpu"
    )
    primary_cropper = (
        None if args.multi_star else StarCropper(args.starseg, crop_config)
    )
    states: list[VideoState] = []
    for index, path in enumerate(args.video):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise SystemExit(f"could not open video: {path}")
        if args.multi_star:
            # Ultralytics stores persistent BoT-SORT state on the predictor.
            # Interleaved streams therefore require independent YOLO instances.
            cropper: StarCropper | TrackedStarCropper = TrackedStarCropper(
                args.starseg,
                crop_config,
                tracker_config=args.star_tracker_config,
            )
        else:
            if primary_cropper is None:
                raise RuntimeError("single-star cropper is not initialized")
            cropper = (
                primary_cropper
                if index == 0
                else StarCropper(
                    args.starseg,
                    crop_config,
                    model=primary_cropper.model,
                )
            )
        states.append(
            VideoState(
                index=index,
                path=path,
                capture=capture,
                cropper=cropper,
                tracker=(
                    None
                    if args.multi_star
                    else FusedTracker(
                        shape_gate=args.shape_gate,
                        embed_weight=args.embed_weight,
                        max_age=args.max_age,
                    )
                ),
                star_trackers=(
                    StarArmTrackerRegistry(
                        shape_gate=args.shape_gate,
                        embed_weight=args.embed_weight,
                        max_age=args.max_age,
                    )
                    if args.multi_star
                    else None
                ),
                fps=float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
            )
        )

    hydra = HydraPredictor(args.hydra, HydraInferConfig(device=args.device))
    tracer, tracer_params, _metadata, _training = load_noodle_tracer_checkpoint(
        str(args.tracer),
        device=args.device,
    )

    def process_batch(pending: list[PendingFrame]) -> None:
        observations: list[tuple[PendingFrame, int, CropResult]] = [
            (item, crop_index, crop)
            for item in pending
            for crop_index, crop in enumerate(item.crops)
        ]
        decoded: dict[tuple[int, int, int], tuple[np.ndarray, list[dict]]] = {}
        for start in range(0, len(observations), args.batch_size):
            batch = observations[start : start + args.batch_size]
            crops_rgb = [crop.crop_rgb for _item, _index, crop in batch]
            image_batch = hydra.preprocess_batch(crops_rgb).to(hydra.device)
            raw_batch, activated_batch = hydra.predict_tensor_batch(image_batch)
            arms_batch = trace_arms_noodle_batch(
                raw_batch,
                image_batch,
                tracer,
                tracer_params,
            )
            for (item, crop_index, _crop), activated, arms in zip(
                batch,
                activated_batch,
                arms_batch,
            ):
                decoded[(item.stream.index, item.frame_index, crop_index)] = (
                    activated.cpu().numpy(),
                    arms,
                )

        for item in pending:
            state = item.stream
            if args.multi_star:
                if state.star_trackers is None:
                    raise RuntimeError("multi-star arm trackers are not initialized")
                star_records: list[dict] = []
                overlay_stars: list[tuple[CropResult, list[dict]]] = []
                support_masks: list[np.ndarray] = []
                for crop_index, crop in enumerate(item.crops):
                    if crop.star_track_id is None:
                        raise RuntimeError("BoT-SORT crop is missing a star track id")
                    activated, arms = decoded[
                        (state.index, item.frame_index, crop_index)
                    ]
                    labels = state.star_trackers.track(
                        crop.star_track_id,
                        item.frame_index,
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
                state.star_trackers.prune(item.frame_index)
                record = {
                    "frame": item.frame_index,
                    "detected": bool(item.crops),
                    "stars": star_records,
                }
                overlay = draw_multi_overlay(
                    item.frame_bgr,
                    overlay_stars,
                    support_masks=support_masks if args.overlay_support else None,
                    support_threshold=args.support_threshold,
                    support_alpha=args.support_alpha,
                )
            else:
                crop = item.crops[0] if item.crops else None
                record = {
                    "frame": item.frame_index,
                    "detected": crop is not None,
                    "crop": None,
                    "arms": [],
                }
                if crop is not None:
                    if state.tracker is None:
                        raise RuntimeError("single-star arm tracker is not initialized")
                    activated, arms = decoded[(state.index, item.frame_index, 0)]
                    labels = track_arms(
                        state.tracker,
                        item.frame_index,
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
                overlay = draw_overlay(item.frame_bgr, crop, record["arms"])
            state.frames.append(record)
            if args.overlay_dir is not None:
                write_overlay(
                    state,
                    args.overlay_dir,
                    overlay,
                )

    try:
        with torch.inference_mode():
            while True:
                round_frames: list[PendingFrame] = []
                for state in states:
                    if state.done:
                        continue
                    if (
                        args.max_frames is not None
                        and state.next_frame >= args.max_frames
                    ):
                        state.done = True
                        continue
                    ok, frame_bgr = state.capture.read()
                    if not ok:
                        state.done = True
                        continue
                    frame_index = state.next_frame
                    state.next_frame += 1
                    if args.multi_star:
                        crops = state.cropper.track_crops(frame_bgr)
                    else:
                        crop = state.cropper.crop(frame_bgr)
                        crops = [] if crop is None else [crop]
                    round_frames.append(
                        PendingFrame(
                            stream=state,
                            frame_index=frame_index,
                            frame_bgr=frame_bgr,
                            crops=crops,
                        )
                    )
                if not round_frames:
                    break
                for start in range(0, len(round_frames), args.batch_size):
                    process_batch(round_frames[start : start + args.batch_size])
    finally:
        for state in states:
            state.capture.release()
            state.cropper.reset()
            if state.writer is not None:
                state.writer.release()

    for state in states:
        output = args.output_dir / f"{output_stem(state)}_tracks.json"
        payload: dict = {
            "video": str(state.path),
            "frames": state.frames,
        }
        if args.multi_star:
            if not isinstance(state.cropper, TrackedStarCropper):
                raise RuntimeError("multi-star output has an invalid cropper")
            payload = multi_star_video_payload(
                state.path,
                state.cropper.tracker_config,
                state.frames,
            )
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {len(state.frames)} frames -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
