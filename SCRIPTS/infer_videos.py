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

from deepstarnoodles.crop import CropConfig, CropResult, StarCropper
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


@dataclass
class VideoState:
    index: int
    path: Path
    capture: cv2.VideoCapture
    cropper: StarCropper
    tracker: FusedTracker
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
    crop: CropResult | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--starseg", type=Path, default=DEFAULT_STARSEG)
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA_CHECKPOINT)
    parser.add_argument("--tracer", type=Path, default=DEFAULT_TRACER_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir is not None:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    crop_config = CropConfig(
        device=args.device if args.device.startswith("cuda") else "cpu"
    )
    primary_cropper = StarCropper(args.starseg, crop_config)
    states: list[VideoState] = []
    for index, path in enumerate(args.video):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise SystemExit(f"could not open video: {path}")
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
                tracker=FusedTracker(
                    shape_gate=args.shape_gate,
                    embed_weight=args.embed_weight,
                    max_age=args.max_age,
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
        detected = [item for item in pending if item.crop is not None]
        decoded: dict[tuple[int, int], tuple[np.ndarray, list[dict]]] = {}
        if detected:
            crops_rgb = [item.crop.crop_rgb for item in detected if item.crop is not None]
            image_batch = hydra.preprocess_batch(crops_rgb).to(hydra.device)
            raw_batch, activated_batch = hydra.predict_tensor_batch(image_batch)
            arms_batch = trace_arms_noodle_batch(
                raw_batch,
                image_batch,
                tracer,
                tracer_params,
            )
            for item, activated, arms in zip(
                detected,
                activated_batch,
                arms_batch,
            ):
                decoded[(item.stream.index, item.frame_index)] = (
                    activated.cpu().numpy(),
                    arms,
                )

        for item in pending:
            state = item.stream
            crop = item.crop
            record: dict = {
                "frame": item.frame_index,
                "detected": crop is not None,
                "crop": None,
                "arms": [],
            }
            if crop is not None:
                activated, arms = decoded[(state.index, item.frame_index)]
                labels = track_arms(
                    state.tracker,
                    item.frame_index,
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
            state.frames.append(record)
            if args.overlay_dir is not None:
                write_overlay(
                    state,
                    args.overlay_dir,
                    draw_overlay(item.frame_bgr, crop, record["arms"]),
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
                    round_frames.append(
                        PendingFrame(
                            stream=state,
                            frame_index=frame_index,
                            frame_bgr=frame_bgr,
                            crop=state.cropper.crop(frame_bgr),
                        )
                    )
                if not round_frames:
                    break
                for start in range(0, len(round_frames), args.batch_size):
                    process_batch(round_frames[start : start + args.batch_size])
    finally:
        for state in states:
            state.capture.release()
            if state.writer is not None:
                state.writer.release()

    for state in states:
        output = args.output_dir / f"{output_stem(state)}_tracks.json"
        output.write_text(
            json.dumps(
                {
                    "video": str(state.path),
                    "frames": state.frames,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {len(state.frames)} frames -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
