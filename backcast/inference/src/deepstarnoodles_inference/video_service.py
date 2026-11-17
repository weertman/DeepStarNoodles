# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Resident annotation service for videos streamed into a rolling cache.

A sender writes each video elsewhere and renames it into ``<queue>/ready/`` once
complete. Every worker process loads the cascade once, then claims whole videos
by renaming them into ``<queue>/work/<worker>/`` (a rename succeeds for exactly
one claimant), decodes every frame and runs the archive cascade on each frame
independently. One JSON record per frame is appended to
``<output>/<video>.jsonl.part`` and flushed, so a reader can stream records as
they appear; ``image_path`` is ``<video>#<frame index>``, and ``time_s`` is the frame index over the
video's nominal frame rate (``fps``; both null when the container reports none). A finished video's
records are renamed to ``.jsonl`` and the video is deleted, so the queue acts as
a rolling cache of not-yet-annotated videos. A ``STOP`` file in the queue ends
every worker after its current frame and leaves partial records in place.

Runtime flags after the service options are the ``benchmark`` flags (archive
mode, one process per worker), validated by the same parser.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from .records import finalize_frame

SUFFIXES = (".mp4", ".mkv", ".mov", ".avi")


class VideoFrames:
    """Decode one video on a thread; yields ``([frame id], [BGR frame])`` groups in order."""

    def __init__(self, path: Path, name: str, *, prefetch: int = 4, stop=None) -> None:
        self.path, self.name = Path(path), name
        self.stop = stop or (lambda: False)
        self.frames = 0
        self.fps: float | None = None  # nominal frame rate, set by the decoder before the first frame
        self.decode_seconds = 0.0
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(prefetch)))
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._decode, name="video-decode", daemon=True)
        self.error: BaseException | None = None

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._halt.set()
        while self._thread.is_alive():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._thread.join(timeout=0.05)

    def _put(self, item) -> bool:
        while not self._halt.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _decode(self) -> None:
        import cv2

        capture = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        try:
            if not capture.isOpened():
                raise ValueError(f"could not open video: {self.path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            # Written before the first frame is queued; the queue orders it before any consumer read.
            self.fps = float(fps) if math.isfinite(fps) and fps > 0 else None
            index = 0
            while not self._halt.is_set():
                started = time.perf_counter()
                ok, frame = capture.read()
                self.decode_seconds += time.perf_counter() - started
                if not ok:
                    break
                if not self._put(([f"{self.name}#{index:06d}"], [frame])):
                    return
                index += 1
            self.frames = index
        except BaseException as exc:  # surfaced to the consumer
            self.error = exc
        finally:
            capture.release()
            self._put(None)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.stop():
                raise StopIteration
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                if self.error is not None:
                    raise self.error
                raise StopIteration
            return item


def claim(queue_dir: Path, worker: int, prefix: str = ""):
    """Move the oldest ready video (named ``prefix*``) into this worker's directory; None when none is ready."""
    ready = queue_dir / "ready"
    mine = queue_dir / "work" / str(worker)
    mine.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in ready.iterdir() if p.suffix.lower() in SUFFIXES and p.name.startswith(prefix)):
        target = mine / path.name
        try:
            os.rename(path, target)
        except FileNotFoundError:
            continue  # another worker claimed it first
        return target
    return None


def frame_record(record: dict, name: str, fps: float | None = None) -> dict:
    """Archive record keyed by video frame: ``frame`` is the index within the video.

    The record ends with ``time_s`` (``frame / fps``, or null without a frame rate), ``fps`` and the
    output ``schema_version`` (see ``records``).
    """
    index = int(record["image_path"].rsplit("#", 1)[1])
    return finalize_frame({**record, "frame": index, "video": name}, fps)


def annotate_video(pipeline, args, path: Path, name: str, sink, *, stop, stream=None) -> dict:
    """Annotate every frame of one video; returns per-video counts (``complete`` False if stopped)."""
    from .benchmark import Timer

    timer = Timer(pipeline.device, False)
    frames = crops = arms = 0
    started = time.perf_counter()

    def write(records):
        for record in records:
            sink.write(json.dumps(frame_record(record, name, source.fps), separators=(",", ":"),
                                  allow_nan=False) + "\n")
        sink.flush()

    with VideoFrames(path, name, prefetch=args.decode_prefetch_frames, stop=stop) as source:
        if args.detect_ahead:
            from .detect_ahead import DetectAhead
            with DetectAhead(pipeline, source, depth=args.detect_ahead, crop_batch_size=args.crop_batch_size,
                             timer=timer, stream=stream) as ahead:
                while not stop():
                    item = ahead.get()
                    if item is None:
                        break
                    paths, records, count, groups = item
                    if pipeline.crop_streams() > 1 and len(groups) > 1:
                        arms += pipeline.infer_parallel([ahead.job(group) for group in groups], records, frames, timer)
                    else:
                        for crop_group, images in ahead.consume(groups):
                            arms += pipeline.infer_crops(crop_group, images, records, frames, timer)
                            del images
                    del groups
                    write(records)
                    frames += len(paths)
                    crops += count
        else:
            for paths, batch in source:
                records, count, arm_count = pipeline.run_frames(batch, frames, timer, source_paths=paths)
                write(records)
                frames += len(paths)
                crops += count
                arms += arm_count
                if stop():
                    break
        complete = not stop() and source.frames == frames
    return {"video": name, "frames": frames, "star_crops": crops, "arms": arms, "complete": complete,
            "seconds": time.perf_counter() - started, "decode_seconds": source.decode_seconds, "fps": source.fps}


def run_worker(args, runtime) -> int:
    import torch
    from . import benchmark

    loaded = time.perf_counter()
    if runtime.models.is_dir():
        runtime.models = runtime.models / "manifest.json"
    manifest, paths = benchmark.verify_models(runtime.models)
    device = torch.device(runtime.device)
    if runtime.cuda_blocking_sync:
        from .cuda_sched import request_blocking_sync
        request_blocking_sync(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(runtime.allocator_limit_gib * 1024**3 / total, 1.0), device)
    pipeline = benchmark.Cascade(runtime, paths)
    # One detect-ahead stream for the worker's whole life. The allocator reuses freed blocks only within their stream
    # and cuBLAS keeps a workspace per stream, so a new stream per video would strand about 500 MiB of cache and 32 MiB
    # of live memory each time, until the allocator limit forces a release.
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" and runtime.detect_ahead else None
    output, queue_dir = args.output, args.queue
    report = {"worker": args.worker_index, "model_load_seconds": time.perf_counter() - loaded, "videos": [],
              "started_epoch": time.time()}
    (output / f"ready.{args.worker_index}").write_text(json.dumps(report) + "\n")
    stop_file = queue_dir / "STOP"
    stop = stop_file.exists
    with torch.inference_mode():
        while not stop():
            path = claim(queue_dir, args.worker_index, args.claim_prefix)
            if path is None:
                time.sleep(0.2)
                continue
            name = path.name.rsplit(".", 1)[0]
            partial = output / f"{name}.jsonl.part"
            try:
                with partial.open("w", encoding="utf-8") as sink:
                    summary = annotate_video(pipeline, runtime, path, name, sink, stop=stop, stream=stream)
            except (ValueError, OSError) as exc:
                # An undecodable or truncated video is set aside; the worker continues.
                failed = queue_dir / "failed"
                failed.mkdir(exist_ok=True)
                os.replace(path, failed / path.name)
                report["videos"].append({"video": name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            report["videos"].append(summary)
            if summary["complete"]:
                os.replace(partial, output / f"{name}.jsonl")
                path.unlink()
            if benchmark.host_peak_bytes() > runtime.ram_limit_gib * 1024**3:
                raise MemoryError("process high-water RSS exceeded --ram-limit-gib")
    for key in ("gru_graph", "hydra_graph", "starseg_graph", "gpu_crops"):
        value = getattr(pipeline, key, None)
        if value is not None and hasattr(value, "summary"):
            report[key] = value.summary()
    report["finished_epoch"] = time.time()
    (output / f"worker.{args.worker_index}.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queue", type=Path, required=True, help="rolling cache with ready/ (and work/) subdirectories")
    p.add_argument("--output", type=Path, required=True, help="directory for per-video record files")
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--processes", type=int, default=1, help="resident worker processes (1-16)")
    p.add_argument("--claim-prefix", default="", help="claim only ready videos whose names start with this prefix")
    p.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    args, rest = p.parse_known_args(argv)
    if not 1 <= args.processes <= 16:
        p.error("--processes must be between 1 and 16")
    from . import benchmark
    runtime = benchmark.parse_args(["--models", str(args.models), "--index", str(args.queue / "unused.csv"),
                                    "--output", str(args.output), "--mode", "archive", "--images", "1",
                                    "--warmup", "0", *rest])
    if runtime.fps is not None:
        p.error("serve-videos reads each video's own frame rate; --fps is for image-index commands")
    if (runtime.archive_processes != 1 or runtime.gru_streams != 1 or runtime.detector_batch_size != 1
            or runtime.profile_stages or runtime.decode_workers):
        p.error("serve-videos takes --processes; workers use one archive process, one GRU stream, detector "
                "batch 1, no stage profiling and their own video decoder (no --decode-workers)")
    return args, rest, runtime


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, rest, runtime = parse_args(argv)
    (args.queue / "ready").mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker_index is not None:
        return run_worker(args, runtime)
    threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "8")) // args.processes)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    children = []
    for index in range(args.processes):
        log = (args.output / f"worker.{index}.log").open("w")
        children.append((subprocess.Popen(
            [sys.executable, "-m", "deepstarnoodles_inference", "serve-videos", "--queue", str(args.queue),
             "--output", str(args.output), "--models", str(args.models), "--processes", str(args.processes),
             "--claim-prefix", args.claim_prefix, "--worker-index", str(index), *rest], stdout=log, stderr=subprocess.STDOUT, env=env), log))
    codes = [child.wait() for child, _ in children]
    for _, log in children:
        log.close()
    (args.output / "service.json").write_text(json.dumps({"processes": args.processes, "exit_codes": codes,
                                                          "argv": argv}, indent=1) + "\n")
    return 0 if not any(codes) else 1
