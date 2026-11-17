#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Measure the real resident StarSeg -> crop -> Hydra -> GRU cascade.

Archive mode processes independent stills. Live replay processes one ordered
stream with BoT-SORT and arm association; it does not simulate camera arrivals.
Inputs are streamed from a CSV with an image_path column (paths relative to the
CSV directory, or absolute). Outputs are JSONL; no full-video output is retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import resource
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import torch

from deepstarnoodles_inference.archive_decode import ArchiveSourceGroups
from deepstarnoodles_inference.body_fields import summarize as summarize_bodies
from deepstarnoodles_inference.crop import (
    CropConfig, TrackedStarCropper, _crop_geometry, _crop_result, _result_detections,
    _sunflower_class_ids,
)
from deepstarnoodles_inference.hydra import HydraInferConfig, HydraPredictor, predict_hydra_crops, store_autocast_weights
from deepstarnoodles_inference.noodle_tracer import load_noodle_tracer_checkpoint, trace_arms_noodle_batch
from deepstarnoodles_inference.records import OUTPUT_SCHEMA_VERSION, finalize_frame, validate_fps
from deepstarnoodles_inference.runtime import StarArmTrackerRegistry, crop_record, source_arm_records


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, required=True, help="Pinned model manifest with paths and SHA-256 hashes")
    p.add_argument("--index", type=Path, required=True, help="Representative full-frame CSV; live replay must be chronological")
    p.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are rejected")
    p.add_argument("--mode", choices=("live-replay", "archive"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--images", type=int, default=128)
    p.add_argument("--warmup", type=int, default=8, help="First N inputs, replayed separately before measurement")
    p.add_argument("--detector-batch-size", type=int, default=1)
    p.add_argument("--crop-batch-size", type=int, default=1)
    p.add_argument("--archive-frame-window", type=int, default=0,
                   help="Experimental resident archive crop accumulation: max pending source records (0 disables, 1-256)")
    p.add_argument("--archive-crop-buffer-mib", type=int, default=256,
                   help="Maximum retained RGB crop/polygon arrays in buffered archives (1-1024 MiB); oversized crops fail before allocation")
    p.add_argument("--hydra-batch-size", type=int, default=None,
                   help="Experimental: cap Hydra forward batch independently of the GRU crop batch; omitted uses the full crop batch")
    p.add_argument("--archive-processes", type=int, default=1,
                   help="Experimental: shard archive frames round-robin across N resident worker processes on one "
                        "device (1-16); merged output keeps input order. The allocator limit applies per process; "
                        "run more than a few under CUDA MPS so their kernels overlap")
    p.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--shard-sync", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--gru-streams", type=int, default=1,
                   help="Experimental: roll out up to N crops' GRU graphs concurrently on separate CUDA streams "
                        "from one thread (1-8; requires --gru-execution graph); archive detect-ahead gathers crops "
                        "across queued frames")
    p.add_argument("--crop-streams", type=int, default=1,
                   help="Experimental: run up to N crop groups of one frame concurrently on separate CUDA streams "
                        "(1-4); tracking and output keep crop order")
    p.add_argument("--detect-ahead", type=int, default=0,
                   help="Experimental archive overlap: run StarSeg and crop preparation up to N source groups "
                        "ahead of Hydra/GRU on a separate CUDA stream (0 disables, 1-8)")
    p.add_argument("--decode-workers", type=int, choices=(0, 1, 2, 4), default=0,
                   help="Experimental archive-only ordered decode workers; 0 preserves synchronous decode")
    p.add_argument("--decode-prefetch-frames", type=int, default=4,
                   help="Maximum reader-owned queued/inflight frames (workers through 64)")
    p.add_argument("--decode-prefetch-mib", type=int, default=256,
                   help="Decoded-array reservation budget in MiB, excluding consumer batch and codec scratch (1-1024)")
    p.add_argument("--decode-max-frame-mib", type=int, default=64,
                   help="Reject larger decoded frames from their headers before decode when workers are enabled")
    p.add_argument("--allocator-limit-gib", type=float, default=7.0, help="PyTorch allocator guard only; not total VRAM or GPU emulation")
    p.add_argument("--ram-limit-gib", type=float, default=24.0, help="Fail when sampled process high-water RSS exceeds this budget")
    p.add_argument("--profile-stages", action="store_true", help="Synchronize each stage; use separate runs for throughput")
    p.add_argument("--starseg-half", action="store_true")
    p.add_argument("--hydra-fp32", action="store_true")
    p.add_argument("--hydra-output-mode", choices=("full", "minimal"), default="full",
                   help="Experimental: skip unused activated fields; live tracking activates only support/mouth")
    p.add_argument("--channels-last", action="store_true", help="Experimental Hydra memory layout")
    p.add_argument("--hydra-fp16-weights", action="store_true",
                   help="Hold Hydra Conv2d/Linear weights in float16 instead of casting them on every autocast "
                        "forward; same outputs, less GPU work")
    p.add_argument("--hydra-compile", action="store_true",
                   help="Experimental: torch.compile the Hydra forward before graph capture (requires "
                        "--hydra-execution graph); not byte-identical to the eager forward")
    p.add_argument("--hydra-execution", choices=("eager", "graph"), default="eager",
                   help="Experimental: replay minimal-mode Hydra forwards from CUDA graphs at repeated shapes")
    p.add_argument("--gru-execution", choices=("dynamic", "fixed", "graph"), default="dynamic",
                   help="Experimental fixed mode keeps all seed slots; graph replays fixed-slot steps from CUDA "
                        "graphs with device peak candidates; dynamic remains the reference")
    p.add_argument("--gru-check-interval", type=int, default=8,
                   help="Fixed/graph modes check aggregate termination every N steps (1-128); ignored by dynamic mode")
    p.add_argument("--gru-compile", action="store_true",
                   help="Experimental: torch.compile the GRU step before graph capture (requires --gru-execution "
                        "graph); not byte-identical to the eager step")
    p.add_argument("--gru-graph-first-step", action="store_true",
                   help="With --gru-execution graph, also replay each rollout's first step from a captured graph")
    p.add_argument("--cuda-blocking-sync", action="store_true",
                   help="Threads waiting for the GPU sleep instead of spinning a CPU core; frees CPU quota for "
                        "decode and crop preparation when several archive processes share a container")
    p.add_argument("--starseg-execution", choices=("eager", "graph"), default="eager",
                   help="graph replays StarSeg's backbone and neck from CUDA graphs per input shape; the head and "
                        "Ultralytics pre/postprocessing stay eager")
    p.add_argument("--gpu-crops", action="store_true",
                   help="Archive: cut and resize Hydra crops on the GPU with Pillow's fixed-point bilinear "
                        "arithmetic (byte-identical inputs); uploads each source frame that has detections")
    p.add_argument("--fps", type=float, default=None,
                   help="Source frame rate: each record gets time_s = frame / fps (null when omitted)")
    args = p.parse_args(argv)
    try:
        args.fps = validate_fps(args.fps)
    except ValueError:
        p.error("--fps must be finite and positive")
    if not 1 <= args.gru_check_interval <= 128:
        p.error("--gru-check-interval must be between 1 and 128")
    if min(args.images, args.detector_batch_size, args.crop_batch_size) < 1 or args.warmup < 0:
        p.error("image/batch counts must be positive and warmup non-negative")
    if not 0 <= args.archive_frame_window <= 256 or not 1 <= args.archive_crop_buffer_mib <= 1024:
        p.error("archive frame window must be 0-256 and crop buffer MiB must be 1-1024")
    if args.archive_frame_window and (args.mode != "archive" or args.detector_batch_size != 1):
        p.error("archive crop accumulation requires --mode archive --detector-batch-size 1")
    if args.archive_frame_window and args.crop_batch_size > 128:
        p.error("archive crop accumulation requires --crop-batch-size <= 128")
    if args.hydra_batch_size is not None and args.hydra_batch_size < 1:
        p.error("--hydra-batch-size must be positive")
    if (not 1 <= args.decode_prefetch_frames <= 64
            or args.decode_prefetch_frames < args.decode_workers
            or not 1 <= args.decode_prefetch_mib <= 1024
            or not 1 <= args.decode_max_frame_mib <= args.decode_prefetch_mib):
        p.error("decode queue requires workers <= prefetch frames <= 64 and 1 <= max-frame MiB <= prefetch MiB <= 1024")
    if not 1 <= args.archive_processes <= 16:
        p.error("--archive-processes must be between 1 and 16")
    if args.archive_processes > 1 and (args.mode != "archive" or args.archive_frame_window or args.profile_stages
                                       or torch.device(args.device).type != "cuda"):
        p.error("--archive-processes requires CUDA --mode archive without --archive-frame-window or --profile-stages")
    sharded = args.shard_count > 1 or args.shard_index is not None or args.shard_sync is not None
    if sharded and (args.archive_processes != 1 or args.shard_index is None or args.shard_sync is None
                    or not 0 <= args.shard_index < args.shard_count <= 16 or args.mode != "archive"
                    or args.archive_frame_window or args.images < args.shard_count):
        p.error("internal shard options are invalid")
    if not 1 <= args.crop_streams <= 4 or (args.crop_streams > 1 and (torch.device(args.device).type != "cuda"
                                                                       or args.archive_frame_window
                                                                       or args.profile_stages)):
        p.error("--crop-streams must be 1-4; above 1 requires CUDA without --archive-frame-window or --profile-stages")
    if not 1 <= args.gru_streams <= 8 or (args.gru_streams > 1 and (args.gru_execution != "graph"
                                                                     or args.crop_streams > 1
                                                                     or args.archive_frame_window)):
        p.error("--gru-streams must be 1-8; above 1 requires --gru-execution graph without --crop-streams "
                "or --archive-frame-window")
    if not 0 <= args.detect_ahead <= 8:
        p.error("--detect-ahead must be between 0 and 8")
    if args.detect_ahead and (args.mode != "archive" or args.archive_frame_window or args.profile_stages):
        p.error("--detect-ahead requires --mode archive without --archive-frame-window or --profile-stages")
    if args.mode != "archive" and args.decode_workers:
        p.error("--decode-workers is archive-only; live replay keeps synchronous ordered input")
    if args.mode == "live-replay" and args.detector_batch_size != 1:
        p.error("live replay requires --detector-batch-size 1")
    if not (np.isfinite(args.allocator_limit_gib) and args.allocator_limit_gib > 0
            and np.isfinite(args.ram_limit_gib) and args.ram_limit_gib > 0):
        p.error("memory budgets must be finite and positive")
    if args.starseg_half and torch.device(args.device).type != "cuda":
        p.error("--starseg-half requires CUDA")
    if args.hydra_fp16_weights and (args.hydra_fp32 or torch.device(args.device).type != "cuda"):
        p.error("--hydra-fp16-weights requires CUDA autocast")
    if args.hydra_compile and args.hydra_execution != "graph":
        p.error("--hydra-compile requires --hydra-execution graph")
    if args.gru_execution == "graph" and torch.device(args.device).type != "cuda":
        p.error("--gru-execution graph requires CUDA")
    if args.gru_compile and args.gru_execution != "graph":
        p.error("--gru-compile requires --gru-execution graph")
    if args.gru_graph_first_step and args.gru_execution != "graph":
        p.error("--gru-graph-first-step requires --gru-execution graph")
    if args.starseg_execution == "graph" and torch.device(args.device).type != "cuda":
        p.error("--starseg-execution graph requires CUDA")
    if args.cuda_blocking_sync and torch.device(args.device).type != "cuda":
        p.error("--cuda-blocking-sync requires CUDA")
    if args.gpu_crops and (args.mode != "archive" or args.archive_frame_window
                           or torch.device(args.device).type != "cuda"):
        p.error("--gpu-crops requires CUDA and --mode archive without --archive-frame-window")
    if args.gpu_crops and args.crop_streams > 1:
        p.error("--gpu-crops shares uploaded frames on one stream; use --crop-streams 1")
    if args.hydra_execution == "graph" and (torch.device(args.device).type != "cuda" or args.hydra_output_mode != "minimal"):
        p.error("--hydra-execution graph requires CUDA and --hydra-output-mode minimal")
    return args


def resolve(path, base=None):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else ((Path.cwd() if base is None else Path(base)) / path).resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_models(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported model manifest schema")
    if manifest.get("path_base", "manifest_directory") != "manifest_directory":
        raise ValueError("portable manifests require path_base=manifest_directory")
    paths = {}
    for key in ("starseg", "hydra", "tracer"):
        record = manifest["models"][key]
        paths[key] = resolve(record["path"], path.parent)
        if sha256(paths[key]) != record["sha256"]:
            raise ValueError(f"{key} checkpoint hash does not match the pinned manifest")
    if manifest["models"]["hydra"]["architecture"] != "dinov3_convnext_base":
        raise ValueError("current HydraPredictor supports the ConvNeXt-Base checkpoint")
    expected = {"detector_image_size": 1024, "crop_image_size": 512, "context_scale": 2.0}
    for key, value in expected.items():
        if manifest.get("preprocessing", {}).get(key, value) != value:
            raise ValueError(f"unsupported manifest preprocessing: {key}")
    if manifest["models"]["hydra"].get("prefer_ema", True) is not True:
        raise ValueError("this benchmark requires Hydra EMA weights")
    for key, size in (("starseg", 1024), ("hydra", 512)):
        if manifest["models"][key].get("image_size", size) != size:
            raise ValueError(f"unsupported {key} image size")
    tracer = manifest["models"]["tracer"]
    if tracer.get("hidden_size", 128) != 128:
        raise ValueError("this benchmark requires GRU128")
    if tracer.get("normalization_source", "checkpoint_buffers") != "checkpoint_buffers":
        raise ValueError("tracer normalization must come from its checkpoint buffers")
    if tracer.get("parent_checkpoint_sha256", manifest["models"]["hydra"]["sha256"]) != manifest["models"]["hydra"]["sha256"]:
        raise ValueError("tracer parent does not match the Hydra checkpoint")
    return manifest, paths


def input_paths(index, count):
    with index.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError("input CSV must have an image_path column")
        for row in islice(reader, count):
            yield resolve(row["image_path"], index.resolve().parent)


def measured_paths(args, count, shard):
    """Global inputs, or this child's round-robin shard of the timed inputs."""
    paths = input_paths(args.index, count)
    shards = getattr(args, "shard_count", 1) or 1
    if shard and shards > 1:
        return islice(paths, args.shard_index, None, shards), len(range(args.shard_index, count, shards))
    return paths, count


def chunks(items, size):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def summary(values):
    if not values:
        return {"samples": 0}
    return {"samples": len(values), "mean_ms": float(np.mean(values)),
            **{f"p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)}}


def host_peak_bytes():
    # Linux reports KiB; macOS reports bytes. Includes model loading/warmup.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)


class Timer:
    def __init__(self, device, profile):
        self.device, self.profile = device, profile
        self.values = defaultdict(list)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name):
        if not self.profile:
            yield
            return
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.values[name].append((time.perf_counter() - start) * 1000)


class Cascade:
    def __init__(self, args, paths):
        from ultralytics import YOLO

        self.args = args
        self.device = torch.device(args.device)
        self.crop_config = CropConfig(device=args.device, half=args.starseg_half, rect=False)
        self.detector = YOLO(str(paths["starseg"]))
        self.sunflower_ids = _sunflower_class_ids(self.detector)
        self.starseg_graph = None
        if getattr(args, "starseg_execution", "eager") == "graph":
            from .starseg_graph import StarSegGraph
            self.starseg_graph = StarSegGraph(self.detector.model).install()
        self.hydra = HydraPredictor(paths["hydra"], HydraInferConfig(device=args.device, amp=not args.hydra_fp32))
        if getattr(args, "hydra_fp16_weights", False):
            store_autocast_weights(self.hydra.model)
        if args.channels_last:
            self.hydra.model.to(memory_format=torch.channels_last)
        if getattr(args, "hydra_compile", False):
            from .gru_compile import compile_hydra
            compile_hydra(self.hydra)
        self.hydra_graph = None
        if getattr(args, "hydra_execution", "eager") == "graph":
            from .hydra_graph import HydraGraph
            self.hydra_graph = HydraGraph(self.hydra)
        self.tracer, self.trace_params, self.tracer_metadata, _ = load_noodle_tracer_checkpoint(str(paths["tracer"]), device=args.device)
        if self.tracer.hidden_size != 128:
            raise ValueError("expected the pinned GRU128 checkpoint")
        for name in ("dense_mean", "dense_std"):
            value = getattr(self.tracer, name, None)
            if value is not None and (not torch.isfinite(value).all() or (name == "dense_std" and not (value > 0).all())):
                raise ValueError(f"invalid checkpoint normalization buffer: {name}")
        self.live_cropper = (TrackedStarCropper(config=self.crop_config, model=self.detector)
                             if args.mode == "live-replay" else None)
        self.registry = StarArmTrackerRegistry()
        self.gpu_crops = None
        if getattr(args, "gpu_crops", False):
            from .crop_gpu import GpuCropPreparer
            self.gpu_crops = GpuCropPreparer(self.hydra.image_size, self.device)

    def graph_rollout(self):
        """Lazily create the CUDA-graph cache used by ``--gru-execution graph``."""
        rollout = getattr(self, "gru_graph", None)
        if rollout is None:
            from .noodle_tracer_graph import FixedSlotGraphRollout
            rollout = FixedSlotGraphRollout(
                self.tracer, self.trace_params,
                steps_per_replay=getattr(self.args, "gru_check_interval", 8),
                compile_step=getattr(self.args, "gru_compile", False),
                graph_first_step=getattr(self.args, "gru_graph_first_step", False))
            self.gru_graph = rollout
        return rollout

    def reset(self):
        if self.live_cropper:
            self.live_cropper.reset()
        self.registry.clear()

    def run_group(self, paths, start_index, timer):
        if self.live_cropper and len(paths) != 1:
            raise ValueError("live replay must process exactly one source frame at a time")
        with timer.stage("read_decode"):
            frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
            if any(frame is None for frame in frames):
                raise ValueError("could not decode one or more source images")
        return self.run_frames(frames, start_index, timer, source_paths=paths)

    def run_frames(self, frames, start_index, timer, *, source_paths=None):
        """Process BGR uint8 arrays without file I/O; caller owns frame lifetime.

        Temporal trackers age by processed observations. A dropping capture source
        must supply a contiguous processed index here, and record its source
        acquisition sequence separately.
        """
        frames, records, observations = self.detect_frames(frames, start_index, timer, source_paths=source_paths)
        total_arms = 0
        groups = list(chunks(observations, self.args.crop_batch_size))
        uploads = None
        if getattr(self, "gpu_crops", None) is not None and observations:
            with timer.stage("frame_upload"):
                uploads = self.upload_frames(frames, observations)
        if self.gru_streams() > 1 and len(groups) > 1:
            jobs = []
            for group in groups:
                with timer.stage("crop_preprocess_transfer"):
                    crop_group = self.crop_group(frames, group)
                    jobs.append((records, start_index, crop_group, self.preprocess_crops(crop_group, uploads)))
            total_arms = self.infer_many(jobs, timer)
            groups = []
        if self.crop_streams() > 1 and len(groups) > 1:
            def job(group):
                def prepare(_workspace):
                    crop_group = self.crop_group(frames, group)
                    return crop_group, self.preprocess_crops(crop_group)
                return prepare
            total_arms = self.infer_parallel([job(group) for group in groups], records, start_index, timer)
            groups = []
        for group in groups:
            with timer.stage("crop_preprocess_transfer"):
                crop_group = self.crop_group(frames, group)
                images = self.preprocess_crops(crop_group, uploads)
            total_arms += self.infer_crops(crop_group, images, records, start_index, timer)
            del images, crop_group
        if self.live_cropper:
            self.registry.prune(start_index)
        return records, len(observations), total_arms

    def detect_frames(self, frames, start_index, timer, *, source_paths=None):
        """Validate and detect; observations use local source indices as before."""
        if not frames:
            raise ValueError("at least one source frame is required")
        if self.live_cropper and len(frames) != 1:
            raise ValueError("live replay must process exactly one source frame at a time")
        for frame in frames:
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
                raise ValueError("source frames must be nonempty HxWx3 BGR uint8 arrays")
        if source_paths is not None and len(source_paths) != len(frames):
            raise ValueError("source_paths must match source frames")
        frames = [np.ascontiguousarray(frame) for frame in frames]
        records = [{"frame": start_index + i,
                    "image_path": None if source_paths is None else str(source_paths[i]),
                    "source_shape": list(frame.shape[:2]), "stars": []}
                   for i, frame in enumerate(frames)]
        with timer.stage("starseg"):
            if self.live_cropper:
                crops = self.live_cropper.track_crops(frames[0])
                observations = [(0, i, crop) for i, crop in enumerate(crops)]
            else:
                predictions = self.detector.predict(
                    frames, imgsz=self.crop_config.imgsz, conf=self.crop_config.conf,
                    iou=self.crop_config.iou, device=self.args.device, verbose=False,
                    half=self.args.starseg_half,
                    # Fixed detector geometry prevents batch membership from
                    # changing Ultralytics' rectangular-padding path.
                    rect=False,
                )
                observations = [(i, j, det) for i, result in enumerate(predictions)
                                for j, det in enumerate(_result_detections(result, self.sunflower_ids, require_track_ids=False))]
                del predictions
        return frames, records, observations

    def crop_group(self, frames, group):
        """``(i, j, CropResult)`` for one observation group; GPU crops leave the pixels to ``preprocess_crops``."""
        if self.live_cropper:
            return list(group)
        make = _crop_geometry if getattr(self, "gpu_crops", None) is not None else _crop_result
        return [(i, j, make(frames[i], item, context_scale=self.crop_config.context_scale)) for i, j, item in group]

    def upload_frames(self, frames, observations):
        """Device copies of the source frames with detections, keyed by local index (GPU crops only)."""
        if getattr(self, "gpu_crops", None) is None:
            return None
        return {i: self.gpu_crops.upload(frames[i]) for i in sorted({i for i, _j, _item in observations})}

    def preprocess_crops(self, crop_group, uploads=None):
        """Training-time crop preparation shared by immediate/buffered archives."""
        if uploads is not None:
            images = self.gpu_crops.prepare(uploads, crop_group)
        else:
            images = self.hydra.preprocess_batch([c.crop_rgb for _, _, c in crop_group]).to(self.device)
        if self.args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        return images

    def compute_crops(self, crop_group, images, timer, workspace=None):
        """Hydra and GRU for one ordered crop group; returns ``(activated, decoded, bodies)``.

        ``bodies`` holds each crop's body origin and support fields (see ``body_fields``).
        ``workspace`` supplies per-stream graph state for concurrent crop streams.
        """
        hydra_graph = getattr(self, "hydra_graph", None) if workspace is None else workspace.hydra_graph
        with timer.stage("hydra"):
            minimal = getattr(self.args, "hydra_output_mode", "full") == "minimal"
            raw, activated = predict_hydra_crops(
                self.hydra, images, batch_size=getattr(self.args, "hydra_batch_size", None),
                minimal=minimal, tracking=self.live_cropper is not None, graph=hydra_graph)
            # Queued on this stream, outside any graph or compiled region; read after the GRU.
            pending = summarize_bodies(raw, activated)
        with timer.stage("gru"):
            execution = getattr(self.args, "gru_execution", "dynamic")
            if execution in ("fixed", "graph"):
                from .noodle_tracer_fixed import trace_arms_noodle_batch_fixed_slots
                graph_options = {}
                if execution == "graph":
                    rollout = self.graph_rollout() if workspace is None else workspace.rollout()
                    graph_options = {"graph_rollout": rollout, "device_peaks": True}
                decoded = trace_arms_noodle_batch_fixed_slots(
                    raw, images, self.tracer, self.trace_params,
                    check_interval=getattr(self.args, "gru_check_interval", 8), **graph_options)
            else:
                decoded = trace_arms_noodle_batch(raw, images, self.tracer, self.trace_params)
        # The tracers end with host reads, so this copy waits on no further GPU work.
        return activated, decoded, pending.finish()

    def record_crops(self, crop_group, activated, decoded, records, start_index, timer, bodies=None):
        """Track (live) and serialize decoded arms in crop order; returns arm count.

        ``bodies`` (one dict per crop) is appended to each star after its arms.
        """
        total_arms = 0
        minimal = getattr(self.args, "hydra_output_mode", "full") == "minimal"
        with timer.stage("postprocess_track"):
            for k, ((i, j, crop), arms) in enumerate(zip(crop_group, decoded)):
                if self.live_cropper:
                    tracking_options = {"support_mouth_only": True} if minimal else {}
                    labels = self.registry.track(crop.star_track_id, start_index + i,
                                                 activated[k].cpu().numpy(), arms, **tracking_options)
                else:
                    labels = list(range(len(arms)))
                arm_records = source_arm_records(arms, labels, crop, model_size=self.hydra.image_size)
                for record, arm in zip(arm_records, arms):
                    for key in ("stop_reason", "steps", "state_confidence", "tip_score", "base_score", "length_px"):
                        if key in arm:
                            record[key] = arm[key]
                if not self.live_cropper:
                    for record in arm_records:
                        record["arm_index"] = record.pop("track_id")
                star = {"detection_index": j, "star_track_id": crop.star_track_id,
                        "crop": crop_record(crop),
                        "mask_polygon_source": crop.mask_polygon.tolist() if crop.mask_polygon is not None else None,
                        "arms": arm_records}
                if bodies is not None:
                    star.update(bodies[k])
                records[i]["stars"].append(star)
                total_arms += len(arms)
        return total_arms

    def infer_crops(self, crop_group, images, records, start_index, timer):
        """One ordered Hydra/GRU group; update indexed source records in place."""
        activated, decoded, bodies = self.compute_crops(crop_group, images, timer)
        return self.record_crops(crop_group, activated, decoded, records, start_index, timer, bodies)

    def crop_streams(self):
        return getattr(self.args, "crop_streams", 1) or 1

    def gru_streams(self):
        return getattr(self.args, "gru_streams", 1) or 1

    def gru_concurrent(self):
        scheduler = getattr(self, "_gru_concurrent", None)
        if scheduler is None:
            from .noodle_tracer_graph import ConcurrentGraphRollouts
            scheduler = self._gru_concurrent = ConcurrentGraphRollouts(
                self.tracer, self.trace_params, slots=self.gru_streams(),
                compile_step=getattr(self.args, "gru_compile", False),
                graph_first_step=getattr(self.args, "gru_graph_first_step", False))
        return scheduler

    def infer_many(self, jobs, timer):
        """Hydra each ``(records, start_index, crop_group, images)`` job, then roll out
        all of their GRUs concurrently; tracking and records keep job order."""
        from .noodle_tracer_graph import trace_arms_fixed_slots_concurrent

        minimal = getattr(self.args, "hydra_output_mode", "full") == "minimal"
        hydra_out, pending = [], []
        with timer.stage("hydra"):
            for _records, _start, _crop_group, images in jobs:
                hydra_out.append(predict_hydra_crops(
                    self.hydra, images, batch_size=getattr(self.args, "hydra_batch_size", None),
                    minimal=minimal, tracking=self.live_cropper is not None,
                    graph=getattr(self, "hydra_graph", None)))
                pending.append(summarize_bodies(*hydra_out[-1]))
        with timer.stage("gru"):
            decoded = trace_arms_fixed_slots_concurrent(
                [(raw, job[3]) for (raw, _activated), job in zip(hydra_out, jobs)],
                self.tracer, self.trace_params, self.gru_concurrent())
        bodies = [item.finish() for item in pending]
        total = 0
        for (records, start, crop_group, _images), (_raw, activated), arms, body in zip(jobs, hydra_out, decoded,
                                                                                       bodies):
            total += self.record_crops(crop_group, activated, arms, records, start, timer, body)
        return total

    def infer_parallel(self, jobs, records, start_index, timer):
        """Run crop-group jobs concurrently on per-stream workspaces; record in order.

        Each job maps a workspace to ``(crop_group, images)`` on that workspace's
        current stream. Hydra/GRU work for different groups overlaps; tracking
        and output records keep the original group order.
        """
        from .crop_streams import CropStreams

        streams = getattr(self, "_crop_stream_pool", None)
        if streams is None:
            streams = self._crop_stream_pool = CropStreams(self, self.crop_streams())
        total_arms = 0
        for crop_group, activated, decoded, bodies in streams.map(jobs):
            total_arms += self.record_crops(crop_group, activated, decoded, records, start_index, timer, bodies)
        return total_arms

    def close(self):
        streams = getattr(self, "_crop_stream_pool", None)
        if streams is not None:
            streams.close()
            self._crop_stream_pool = None
        if getattr(self, "_gru_concurrent", None) is not None:
            self.gru_concurrent_summary = self._gru_concurrent.summary()


def measure_detect_ahead(pipeline, args, count, sink, shard=False):
    """Archive measurement with StarSeg/crop preparation overlapped ahead."""
    from .detect_ahead import DetectAhead

    paths, count = measured_paths(args, count, shard)

    timer = Timer(pipeline.device, False)
    stream_sync = (torch.cuda.current_stream(pipeline.device).synchronize
                   if pipeline.device.type == "cuda" else (lambda: None))
    groups_ms, source_count, crop_count, arm_count, empty_count = [], 0, 0, 0, 0
    timer.sync()
    run_start = time.perf_counter()
    with ArchiveSourceGroups(paths, args.detector_batch_size, args, timer) as source:
        with DetectAhead(pipeline, source, depth=args.detect_ahead,
                         crop_batch_size=args.crop_batch_size, timer=timer) as ahead:
            gather = pipeline.gru_streams() > 1
            while True:
                start = time.perf_counter()
                item = ahead.get()
                if item is None:
                    break
                if gather:
                    # Concurrent GRUs: take every already-queued frame (bounded)
                    # so independent crops from several frames overlap.
                    batch = [item]
                    while sum(len(entry[3]) for entry in batch) < 2 * pipeline.gru_streams() and len(batch) < 8:
                        extra = ahead.poll()
                        if extra is None or extra is ahead.EMPTY:
                            break
                        batch.append(extra)
                    jobs, offsets = [], []
                    for paths_b, records_b, _crops_b, groups_b in batch:
                        offsets.append(source_count + sum(len(entry[0]) for entry in batch[:len(offsets)]))
                        for crop_group, images in ahead.consume(groups_b):
                            jobs.append((records_b, offsets[-1], crop_group, images))
                    arms = pipeline.infer_many(jobs, timer) if jobs else 0
                    del jobs
                    for paths_b, records_b, crops_b, _groups_b in batch:
                        for record in records_b:
                            finalize_frame(record, getattr(args, "fps", None))
                            sink.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                        source_count += len(paths_b)
                        crop_count += crops_b
                        empty_count += sum(not record["stars"] for record in records_b)
                    sink.flush()
                    arm_count += arms
                    stream_sync()
                    groups_ms.append((time.perf_counter() - start) * 1000)
                    del batch
                    if host_peak_bytes() > args.ram_limit_gib * 1024**3:
                        raise MemoryError("process high-water RSS exceeded --ram-limit-gib")
                    continue
                paths, records, crops, groups = item
                arms = 0
                if pipeline.crop_streams() > 1 and len(groups) > 1:
                    arms = pipeline.infer_parallel([ahead.job(group) for group in groups], records,
                                                   source_count, timer)
                else:
                    for crop_group, images in ahead.consume(groups):
                        arms += pipeline.infer_crops(crop_group, images, records, source_count, timer)
                        del images
                del groups
                for record in records:
                    finalize_frame(record, getattr(args, "fps", None))
                    sink.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                sink.flush()
                # Consumer-stream completion only: the producer stream keeps running.
                stream_sync()
                groups_ms.append((time.perf_counter() - start) * 1000)
                source_count += len(paths)
                crop_count += crops
                arm_count += arms
                empty_count += sum(not record["stars"] for record in records)
                if host_peak_bytes() > args.ram_limit_gib * 1024**3:
                    raise MemoryError("process high-water RSS exceeded --ram-limit-gib")
            ahead_report = ahead.report()
    timer.sync()
    elapsed = time.perf_counter() - run_start
    if source_count != count:
        raise ValueError(f"requested {count} images but input CSV supplied {source_count}")
    return {"source_images": source_count, "star_crops": crop_count, "arms": arm_count,
            "empty_source_images": empty_count, "elapsed_seconds": elapsed,
            "source_images_per_second": source_count / elapsed if elapsed else None,
            "star_crops_per_second": crop_count / elapsed if elapsed else None,
            "source_group_latency": summary(groups_ms), "group_ms": groups_ms,
            "group_latency_scope": "consumer dequeue through output flush; detection ran ahead",
            "stages": {}, "archive_decode": source.report(), "detect_ahead": ahead_report}


def measure(pipeline, args, count, sink, profile, *, shard=False):
    if getattr(args, "archive_frame_window", 0):
        from .archive_buffer import measure_buffered_archive
        return measure_buffered_archive(pipeline, args, count, sink, profile)
    if getattr(args, "detect_ahead", 0):
        return measure_detect_ahead(pipeline, args, count, sink, shard)
    paths, count = measured_paths(args, count, shard)
    timer = Timer(pipeline.device, profile)
    groups_ms, source_count, crop_count, arm_count, empty_count = [], 0, 0, 0, 0
    timer.sync()
    run_start = time.perf_counter()
    with ArchiveSourceGroups(paths, args.detector_batch_size, args, timer) as source:
        while True:
            timer.sync()
            start = time.perf_counter()
            try:
                paths, frames = next(source)
            except StopIteration:
                break
            if frames is None:
                records, crops, arms = pipeline.run_group(paths, source_count, timer)
            else:
                records, crops, arms = pipeline.run_frames(frames, source_count, timer, source_paths=paths)
                del frames
            with timer.stage("serialize_write"):
                for record in records:
                    finalize_frame(record, getattr(args, "fps", None))
                    sink.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                sink.flush()
            timer.sync()
            groups_ms.append((time.perf_counter() - start) * 1000)
            source_count += len(paths)
            crop_count += crops
            arm_count += arms
            empty_count += sum(not record["stars"] for record in records)
            if host_peak_bytes() > args.ram_limit_gib * 1024**3:
                raise MemoryError("process high-water RSS exceeded --ram-limit-gib")
    elapsed = time.perf_counter() - run_start
    if source_count != count:
        raise ValueError(f"requested {count} images but input CSV supplied {source_count}")
    return {"source_images": source_count, "star_crops": crop_count, "arms": arm_count,
            "empty_source_images": empty_count, "elapsed_seconds": elapsed,
            "source_images_per_second": source_count / elapsed if elapsed else None,
            "star_crops_per_second": crop_count / elapsed if elapsed else None,
            "source_group_latency": summary(groups_ms), "group_ms": groups_ms,
            "stages": {k: summary(v) for k, v in timer.values.items()},
            "archive_decode": source.report()}


def main(argv=None):
    job_started = time.perf_counter()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.models.is_dir():
        args.models = args.models / "manifest.json"
    if args.archive_processes > 1:
        if sum(1 for _ in input_paths(args.index, max(args.images, args.warmup))) < max(args.images, args.warmup):
            raise ValueError("input CSV does not contain enough images")
        verify_models(args.models)
        from .archive_shards import run_sharded
        return run_sharded(args, argv, job_started)
    manifest, paths = verify_models(args.models)
    # Check input syntax before allocating GPU memory or creating results.
    if sum(1 for _ in input_paths(args.index, max(args.images, args.warmup))) < max(args.images, args.warmup):
        raise ValueError("input CSV does not contain enough images")
    device = torch.device(args.device)
    blocking_sync = None
    if args.cuda_blocking_sync:
        from .cuda_sched import request_blocking_sync
        blocking_sync = request_blocking_sync(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.allocator_limit_gib * 1024**3 / total, 1.0), device)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "output_schema_version": OUTPUT_SCHEMA_VERSION, "status": "running", "models": manifest,
              "models_manifest_sha256": sha256(args.models), "index_sha256": sha256(args.index),
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "inventory": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                            "device": str(device), "python": sys.version,
                            "ultralytics": importlib.metadata.version("ultralytics"),
                            "numpy": np.__version__, "opencv": cv2.__version__,
                            "cudnn": torch.backends.cudnn.version(), "torch_cpu_threads": torch.get_num_threads()},
              "scope": "resident cascade; image files -> serialized JSONL; no overlay; inputs prehashed before timing",
              "limitations": ["Live replay excludes camera arrival/queue latency and does not establish a live FPS SLA.",
                              "Stage profiling inserts synchronizations; compare throughput with profiling disabled.",
                              "Archive decode workers overlap inference; queue wait and summed worker decode are not additive serial stages. Decode reservations exclude consumer batches, codec scratch and page cache.",
                              "PyTorch peaks exclude external allocations. RSS is process high-water including load/warmup.",
                              "Allocator guard does not emulate RTX 5060 performance or certify total 8GB VRAM fit.",
                              "JSONL flush reaches the OS page cache; fsync/storage durability latency is excluded.",
                              "Input hashing warms the filesystem cache; this is not a cold-storage throughput measurement. Keep inputs immutable during runs.",
                              "Archive mode here is resident microbatching; compare staged-archive separately for disk-spooled scheduling."]}
    if device.type == "cuda":
        report["inventory"].update(gpu_name=torch.cuda.get_device_name(device), gpu_total_memory_bytes=total,
                                    gpu_capability=list(torch.cuda.get_device_capability(device)),
                                    cuda_blocking_sync=blocking_sync)
    try:
        report["input_files"] = [{"path": str(path), "sha256": sha256(path)}
                                 for path in input_paths(args.index, max(args.images, args.warmup))]
        start = time.perf_counter()
        pipeline = Cascade(args, paths)
        Timer(device, False).sync()
        report["model_load_seconds"] = time.perf_counter() - start
        if host_peak_bytes() > args.ram_limit_gib * 1024**3:
            raise MemoryError("model loading exceeded --ram-limit-gib")
        report["crop_config"] = asdict(pipeline.crop_config)
        report["hydra_config"] = asdict(pipeline.hydra.config)
        report["trace_params"] = asdict(pipeline.trace_params)
        report["hydra_state_dict_key"] = getattr(pipeline.hydra, "selected_state_dict_key", None)
        tracer = getattr(pipeline, "tracer", None)
        if tracer is not None and hasattr(tracer, "dense_mean") and hasattr(tracer, "dense_std"):
            normalization = {name: getattr(tracer, name).detach().cpu().numpy().astype("<f4")
                             for name in ("dense_mean", "dense_std")}
            report["normalization"] = {
                "source": "checkpoint_buffers", "digest_encoding": "dense_mean then dense_std, little-endian float32",
                "sha256": hashlib.sha256(b"".join(normalization[name].tobytes() for name in ("dense_mean", "dense_std"))).hexdigest(),
                **{name: value.reshape(-1).tolist() for name, value in normalization.items()},
            }
        with torch.inference_mode():
            # Warmup runs the actual selected batch path and writes to a discard sink.
            with open("/dev/null", "w") as sink:
                report["warmup"] = measure(pipeline, args, args.warmup, sink, False)
                if not report["warmup"]["star_crops"]:
                    report["limitations"].append("Warmup had no star crops; Hydra/GRU first-call cost remains in measurement.")
            pipeline.reset()
            if device.type == "cuda":
                report["startup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                report["startup_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
            if args.shard_sync is not None:
                from .archive_shards import wait_for_start
                report["shard_start_epoch"] = wait_for_start(args.shard_sync, args.shard_index)
            with (args.output / "predictions.jsonl").open("x") as sink:
                report["measurement"] = measure(pipeline, args, args.images, sink, args.profile_stages,
                                                shard=args.shard_sync is not None)
            report["measurement"]["finished_epoch"] = time.time()
        graph = getattr(pipeline, "gru_graph", None)
        if graph is not None:
            report["gru_graph"] = graph.summary()
        if getattr(pipeline, "_gru_concurrent", None) is not None:
            report["gru_concurrent"] = pipeline._gru_concurrent.summary()
        if getattr(pipeline, "hydra_graph", None) is not None:
            report["hydra_graph"] = pipeline.hydra_graph.summary()
        if getattr(pipeline, "gpu_crops", None) is not None:
            report["gpu_crops"] = pipeline.gpu_crops.summary()
        if getattr(pipeline, "starseg_graph", None) is not None:
            report["starseg_graph"] = pipeline.starseg_graph.summary()
        report["status"] = "complete"
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if "pipeline" in locals() and hasattr(pipeline, "close"):
            pipeline.close()
        report["total_job_seconds"] = time.perf_counter() - job_started
        report["host_peak_rss_bytes"] = host_peak_bytes()
        if device.type == "cuda":
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
