#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Measure the real resident StarSeg -> crop -> Hydra -> GRU cascade.

Archive mode processes independent stills. Live replay processes one ordered
stream with BoT-SORT and arm association; it does not simulate camera arrivals.
Inputs are streamed from a CSV with an image_path column (paths relative to the
repository, or absolute). Outputs are JSONL; no full-video output is retained.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

from deepstarnoodles.crop import (
    CropConfig, TrackedStarCropper, _crop_result, _result_detections,
    _sunflower_class_ids,
)
from deepstarnoodles.hydra import HydraInferConfig, HydraPredictor
from deepstarnoodles.noodle_tracer import load_noodle_tracer_checkpoint, trace_arms_noodle_batch
from deepstarnoodles.runtime import StarArmTrackerRegistry, crop_record, source_arm_records


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
    p.add_argument("--allocator-limit-gib", type=float, default=7.0, help="PyTorch allocator guard only; not total VRAM or GPU emulation")
    p.add_argument("--ram-limit-gib", type=float, default=24.0, help="Fail when sampled process high-water RSS exceeds this budget")
    p.add_argument("--profile-stages", action="store_true", help="Synchronize each stage; use separate runs for throughput")
    p.add_argument("--starseg-half", action="store_true")
    p.add_argument("--hydra-fp32", action="store_true")
    p.add_argument("--channels-last", action="store_true", help="Experimental Hydra memory layout")
    args = p.parse_args(argv)
    if min(args.images, args.detector_batch_size, args.crop_batch_size) < 1 or args.warmup < 0:
        p.error("image/batch counts must be positive and warmup non-negative")
    if args.mode == "live-replay" and args.detector_batch_size != 1:
        p.error("live replay requires --detector-batch-size 1")
    if not (np.isfinite(args.allocator_limit_gib) and args.allocator_limit_gib > 0
            and np.isfinite(args.ram_limit_gib) and args.ram_limit_gib > 0):
        p.error("memory budgets must be finite and positive")
    if args.starseg_half and torch.device(args.device).type != "cuda":
        p.error("--starseg-half requires CUDA")
    return args


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO / path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_models(path):
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported model manifest schema")
    paths = {}
    for key in ("starseg", "hydra", "tracer"):
        record = manifest["models"][key]
        paths[key] = resolve(record["path"])
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
    if tracer.get("parent_checkpoint_sha256", manifest["models"]["hydra"]["sha256"]) != manifest["models"]["hydra"]["sha256"]:
        raise ValueError("tracer parent does not match the Hydra checkpoint")
    return manifest, paths


def input_paths(index, count):
    with index.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError("input CSV must have an image_path column")
        for row in islice(reader, count):
            yield resolve(row["image_path"])


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
        self.hydra = HydraPredictor(paths["hydra"], HydraInferConfig(device=args.device, amp=not args.hydra_fp32))
        if args.channels_last:
            self.hydra.model.to(memory_format=torch.channels_last)
        self.tracer, self.trace_params, self.tracer_metadata, _ = load_noodle_tracer_checkpoint(str(paths["tracer"]), device=args.device)
        if self.tracer.hidden_size != 128:
            raise ValueError("expected the pinned GRU128 checkpoint")
        self.live_cropper = (TrackedStarCropper(config=self.crop_config, model=self.detector)
                             if args.mode == "live-replay" else None)
        self.registry = StarArmTrackerRegistry()

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
        records = [{"frame": start_index + i, "image_path": str(path),
                    "source_shape": list(frame.shape[:2]), "stars": []}
                   for i, (path, frame) in enumerate(zip(paths, frames))]
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
        total_arms = 0
        for group in chunks(observations, self.args.crop_batch_size):
            with timer.stage("crop_preprocess_transfer"):
                crop_group = [(i, j, (item if self.live_cropper else _crop_result(
                    frames[i], item, context_scale=self.crop_config.context_scale))) for i, j, item in group]
                images = self.hydra.preprocess_batch([c.crop_rgb for _, _, c in crop_group]).to(self.device)
                if self.args.channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
            with timer.stage("hydra"):
                raw, activated = self.hydra.predict_tensor_batch(images)
            with timer.stage("gru"):
                decoded = trace_arms_noodle_batch(raw, images, self.tracer, self.trace_params)
            with timer.stage("postprocess_track"):
                for k, ((i, j, crop), arms) in enumerate(zip(crop_group, decoded)):
                    if self.live_cropper:
                        labels = self.registry.track(crop.star_track_id, start_index + i, activated[k].cpu().numpy(), arms)
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
                    records[i]["stars"].append({"detection_index": j, "star_track_id": crop.star_track_id,
                                                "crop": crop_record(crop),
                                                "mask_polygon_source": crop.mask_polygon.tolist() if crop.mask_polygon is not None else None,
                                                "arms": arm_records})
                    total_arms += len(arms)
            # Release tensors before the next micro-batch forward is evaluated.
            del images, raw, activated, decoded, crop_group
        if self.live_cropper:
            self.registry.prune(start_index)
        return records, len(observations), total_arms


def measure(pipeline, args, count, sink, profile):
    timer = Timer(pipeline.device, profile)
    groups_ms, source_count, crop_count, arm_count, empty_count = [], 0, 0, 0, 0
    timer.sync()
    run_start = time.perf_counter()
    for paths in chunks(input_paths(args.index, count), args.detector_batch_size):
        timer.sync()
        start = time.perf_counter()
        records, crops, arms = pipeline.run_group(paths, source_count, timer)
        with timer.stage("serialize_write"):
            for record in records:
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
            "stages": {k: summary(v) for k, v in timer.values.items()}}


def main(argv=None):
    args = parse_args(argv)
    manifest, paths = verify_models(args.models)
    # Check input syntax before allocating GPU memory or creating results.
    if sum(1 for _ in input_paths(args.index, max(args.images, args.warmup))) < max(args.images, args.warmup):
        raise ValueError("input CSV does not contain enough images")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.allocator_limit_gib * 1024**3 / total, 1.0), device)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "running", "models": manifest,
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
                              "PyTorch peaks exclude external allocations. RSS is process high-water including load/warmup.",
                              "Allocator guard does not emulate RTX 5060 performance or certify total 8GB VRAM fit.",
                              "JSONL flush reaches the OS page cache; fsync/storage durability latency is excluded.",
                              "Input hashing warms the filesystem cache; this is not a cold-storage throughput measurement. Keep inputs immutable during runs.",
                              "Archive mode here is resident microbatching; stage-wise spooling is a separate planned candidate."]}
    if device.type == "cuda":
        report["inventory"].update(gpu_name=torch.cuda.get_device_name(device), gpu_total_memory_bytes=total,
                                    gpu_capability=list(torch.cuda.get_device_capability(device)))
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
            with (args.output / "predictions.jsonl").open("x") as sink:
                report["measurement"] = measure(pipeline, args, args.images, sink, args.profile_stages)
        report["status"] = "complete"
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["host_peak_rss_bytes"] = host_peak_bytes()
        if device.type == "cuda":
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
