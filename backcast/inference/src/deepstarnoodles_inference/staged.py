"""Experimental archive scheduling with bounded disk shards and one model at a time.

The whole-job measurement includes stage reloads and uncompressed float32 spool
I/O. This is a scheduling experiment; it preserves detector/crop/tracer settings.
"""
from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from . import benchmark as b
from .archive_decode import ArchiveSourceGroups, accumulate_decode_report
from .body_fields import summarize as summarize_bodies
from .crop import CropConfig, _crop_result, _result_detections, _sunflower_class_ids
from .hydra import HydraInferConfig, HydraPredictor, predict_hydra_crops
from .noodle_tracer import load_noodle_tracer_checkpoint, trace_arms_noodle_batch
from .records import OUTPUT_SCHEMA_VERSION, finalize_frame
from .runtime import crop_record, source_arm_records


def parse_args(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--shard-size", type=int, default=64)
    p.add_argument("--max-spool-gib", type=float, default=8.0)
    extra, rest = p.parse_known_args(argv)
    args = b.parse_args(rest)
    if args.archive_frame_window:
        raise ValueError("--archive-frame-window is resident archive-only; staged scheduling already buffers disk shards")
    if (getattr(args, "detect_ahead", 0) or getattr(args, "archive_processes", 1) != 1
            or getattr(args, "crop_streams", 1) != 1 or getattr(args, "gru_streams", 1) != 1):
        raise ValueError("--detect-ahead, --archive-processes and --crop-streams are resident-only")
    if getattr(args, "hydra_execution", "eager") != "eager":
        raise ValueError("--hydra-execution graph is resident-only; staged Hydra forwards stay eager")
    if args.mode != "archive" or args.warmup != 0:
        raise ValueError("staged-archive requires --mode archive --warmup 0; stage startup is part of its job cost")
    if extra.shard_size < 1 or not np.isfinite(extra.max_spool_gib) or extra.max_spool_gib <= 0:
        raise ValueError("shard size and spool budget must be positive")
    args.shard_size, args.max_spool_gib = extra.shard_size, extra.max_spool_gib
    return args


def release(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def check_ram(args):
    if b.host_peak_bytes() > args.ram_limit_gib * 1024**3:
        raise MemoryError("process high-water RSS exceeded --ram-limit-gib")


def save_array(path, value, state, args):
    # np.save's small header is charged as well; reject before large writes.
    if state["live_bytes"] + value.nbytes + 256 > args.max_spool_gib * 1024**3:
        raise MemoryError("shard exceeds --max-spool-gib; lower --shard-size")
    np.save(path, value, allow_pickle=False)
    size = path.stat().st_size
    state["live_bytes"] += size
    state["written_bytes"] += size
    state["peak_bytes"] = max(state["peak_bytes"], state["live_bytes"])


def run_shard(args, model_paths, paths, frame_start, spool, timer, state):
    from ultralytics import YOLO

    records, entries = [], []
    cfg = CropConfig(device=args.device, rect=False, half=args.starseg_half)
    with timer.stage("detector_load_detect_crop_spool"):
        detector = YOLO(str(model_paths["starseg"]))
        ids = _sunflower_class_ids(detector)
        source_offset = 0
        source_group = []
        with ArchiveSourceGroups(paths, args.detector_batch_size, args, timer) as input_groups:
            for source_paths, frames in input_groups:
                source_group = list(enumerate(source_paths, start=source_offset))
                source_offset += len(source_paths)
                if frames is None:
                    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in source_paths]
                    if any(frame is None for frame in frames):
                        raise ValueError("could not decode source image")
                predictions = detector.predict(frames, imgsz=cfg.imgsz, conf=cfg.conf,
                    iou=cfg.iou, device=args.device, half=args.starseg_half, rect=False, verbose=False)
                for (source, path), frame, prediction in zip(source_group, frames, predictions):
                    records.append({"frame": frame_start + source, "image_path": str(path),
                                    "source_shape": list(frame.shape[:2]), "stars": []})
                    for detection_index, detection in enumerate(_result_detections(prediction, ids, require_track_ids=False)):
                        crop = _crop_result(frame, detection, context_scale=cfg.context_scale)
                        stem = str(len(entries))
                        crop_path = spool / f"{stem}_crop.npy"
                        save_array(crop_path, crop.crop_rgb, state, args)
                        entries.append({"source": source, "detection_index": detection_index,
                                        "crop": crop_record(crop), "stem": stem,
                                        "mask_polygon_source": crop.mask_polygon.tolist() if crop.mask_polygon is not None else None})
                del predictions, frames
                check_ram(args)
        if input_groups.report() is not None:
            state["archive_decode"] = accumulate_decode_report(state.get("archive_decode"), input_groups.report())
        del detector
        # Loop variables can otherwise retain an Ultralytics GPU Result.
        if source_group:
            del prediction, frame
        release(timer.device)
    if not entries:
        return records, 0, 0

    with timer.stage("hydra_load_forward_spool"):
        hydra = HydraPredictor(model_paths["hydra"], HydraInferConfig(device=args.device, amp=not args.hydra_fp32))
        if args.channels_last:
            hydra.model.to(memory_format=torch.channels_last)
        for group in b.chunks(entries, args.crop_batch_size):
            crops = [np.load(spool / f"{entry['stem']}_crop.npy", allow_pickle=False) for entry in group]
            images = hydra.preprocess_batch(crops).to(timer.device)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            raw, activated = predict_hydra_crops(
                hydra, images, batch_size=getattr(args, "hydra_batch_size", None),
                minimal=getattr(args, "hydra_output_mode", "full") == "minimal")
            pending = summarize_bodies(raw, activated)
            raw_cpu, images_cpu = raw.cpu().numpy(), images.cpu().numpy()
            for entry, body in zip(group, pending.finish()):
                entry["body"] = body
            del pending
            for entry, fields, image in zip(group, raw_cpu, images_cpu):
                save_array(spool / f"{entry['stem']}_raw.npy", fields, state, args)
                save_array(spool / f"{entry['stem']}_image.npy", image, state, args)
                path = spool / f"{entry['stem']}_crop.npy"
                state["live_bytes"] -= path.stat().st_size
                path.unlink()
            del raw, activated, images, raw_cpu, images_cpu, crops
            check_ram(args)
        del hydra, fields, image
        release(timer.device)

    with timer.stage("tracer_load_decode_output"):
        tracer, trace_params, _, _ = load_noodle_tracer_checkpoint(str(model_paths["tracer"]), device=args.device)
        if tracer.hidden_size != 128:
            raise ValueError("expected GRU128")
        arm_count = 0
        graph_rollout = None
        for group in b.chunks(entries, args.crop_batch_size):
            raws = torch.from_numpy(np.stack([np.load(spool / f"{e['stem']}_raw.npy", allow_pickle=False) for e in group])).to(timer.device)
            images = torch.from_numpy(np.stack([np.load(spool / f"{e['stem']}_image.npy", allow_pickle=False) for e in group])).to(timer.device)
            execution = getattr(args, "gru_execution", "dynamic")
            if execution in ("fixed", "graph"):
                from .noodle_tracer_fixed import trace_arms_noodle_batch_fixed_slots
                if execution == "graph" and graph_rollout is None:
                    from .noodle_tracer_graph import FixedSlotGraphRollout
                    graph_rollout = FixedSlotGraphRollout(
                        tracer, trace_params, steps_per_replay=getattr(args, "gru_check_interval", 8))
                graph_options = ({"graph_rollout": graph_rollout, "device_peaks": True}
                                 if execution == "graph" else {})
                decoded = trace_arms_noodle_batch_fixed_slots(
                    raws, images, tracer, trace_params,
                    check_interval=getattr(args, "gru_check_interval", 8), **graph_options)
            else:
                decoded = trace_arms_noodle_batch(raws, images, tracer, trace_params)
            for entry, arms in zip(group, decoded):
                c = entry["crop"]
                crop = SimpleNamespace(crop_side=c["side"], crop_left=c["left"], crop_top=c["top"])
                arm_records = source_arm_records(arms, list(range(len(arms))), crop, model_size=512)
                for record, arm in zip(arm_records, arms):
                    record["arm_index"] = record.pop("track_id")
                    for key in ("stop_reason", "steps", "state_confidence", "tip_score", "base_score", "length_px"):
                        if key in arm:
                            record[key] = arm[key]
                records[entry["source"]]["stars"].append({"detection_index": entry["detection_index"],
                    "star_track_id": None, "crop": c, "mask_polygon_source": entry["mask_polygon_source"], "arms": arm_records,
                    **entry["body"]})
                arm_count += len(arms)
                for suffix in ("raw", "image"):
                    path = spool / f"{entry['stem']}_{suffix}.npy"
                    state["live_bytes"] -= path.stat().st_size
                    path.unlink()
            del raws, images, decoded
            check_ram(args)
        del tracer
        release(timer.device)
    return records, len(entries), arm_count


def main(argv=None):
    job_start = time.perf_counter()
    args = parse_args(argv)
    if args.models.is_dir():
        args.models = args.models / "manifest.json"
    manifest, model_paths = b.verify_models(args.models)
    inputs = list(b.input_paths(args.index, args.images))
    if len(inputs) != args.images:
        raise ValueError("input CSV does not contain requested images")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.allocator_limit_gib * 1024**3 / total, 1.0), device)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "output_schema_version": OUTPUT_SCHEMA_VERSION, "status": "running",
              "schedule": "staged-archive",
              "models": manifest, "models_manifest_sha256": b.sha256(args.models), "index_sha256": b.sha256(args.index),
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "inventory": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": str(device)},
              "limitations": ["Experimental disk-sharded schedule; no camera or temporal tracking.",
                               "Inputs are prehashed; filesystem cache state is not cold-storage controlled.",
                               "Spools are float32 .npy, uncompressed, buffered filesystem I/O without fsync.",
                               "All stage reloads and shard cleanup are included in pipeline elapsed time.",
                               "Allocator/RSS guards are not hard total-device or cgroup limits.",
                               "Archive decode queues exist only during each detector stage; reported worker decode overlaps its consumer, and reserved bytes exclude consumer batches, codec scratch and page cache.",
                               "Incomplete temporary shards are discarded; completed JSONL remains available, automatic resume not implemented."]}
    state = {"live_bytes": 0, "peak_bytes": 0, "written_bytes": 0}
    timer = b.Timer(device, True)
    if device.type == "cuda":
        report["inventory"].update(gpu_name=torch.cuda.get_device_name(device), gpu_total_memory_bytes=total,
                                    gpu_capability=list(torch.cuda.get_device_capability(device)))
    try:
        report["input_files"] = [{"path": str(p), "sha256": b.sha256(p)} for p in inputs]
        timer.sync()
        start = time.perf_counter()
        sources = crops = arms = empty = 0
        with torch.inference_mode(), (args.output / "predictions.jsonl").open("x") as sink:
            for shard in b.chunks(inputs, args.shard_size):
                with tempfile.TemporaryDirectory(prefix="spool_", dir=args.output) as directory:
                    records, crop_count, arm_count = run_shard(args, model_paths, shard, sources, Path(directory), timer, state)
                    for record in records:
                        finalize_frame(record, getattr(args, "fps", None))
                        sink.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                    sink.flush()
                    sources += len(shard)
                    crops += crop_count
                    arms += arm_count
                    empty += sum(not r["stars"] for r in records)
                    check_ram(args)
        timer.sync()
        elapsed = time.perf_counter() - start
        report.update(status="complete", spool=state,
            measurement={"source_images": sources, "star_crops": crops, "arms": arms, "empty_source_images": empty,
                         "elapsed_seconds": elapsed, "source_images_per_second": sources / elapsed,
                         "star_crops_per_second": crops / elapsed,
                         "stages": {k: b.summary(v) for k, v in timer.values.items()},
                         "archive_decode": state.get("archive_decode")})
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", spool=state)
        raise
    finally:
        # TemporaryDirectory has removed any incomplete current shard.
        report["spool_bytes_before_cleanup"] = state["live_bytes"]
        state["live_bytes"] = 0
        report["total_job_seconds"] = time.perf_counter() - job_start
        report["host_peak_rss_bytes"] = b.host_peak_bytes()
        if device.type == "cuda":
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
