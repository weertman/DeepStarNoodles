# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Experimental: one archive sharded across resident worker processes on one GPU.

Frames are assigned round-robin to N child benchmark processes. Every child
loads the same pinned models, warms up on the same leading inputs, then waits
for a common start signal so their timed phases overlap. Each frame is
processed exactly as in a single process; the parent only restores global frame
indices and input order. Timing covers the first start signal through the last
child's final flush. Device memory is shared, so the per-child allocator limit
does not bound the device total.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .records import OUTPUT_SCHEMA_VERSION, finalize_frame

SHARD_FLAGS = ("--archive-processes", "--shard-index", "--shard-count", "--shard-sync", "--output")


def child_argv(argv: list[str], index: int, count: int, sync: Path, output: Path) -> list[str]:
    """Drop parent-only flags from argv and add one child's shard identity."""
    kept, skip = [], False
    for item in argv:
        if skip:
            skip = False
            continue
        name = item.split("=", 1)[0]
        if name in SHARD_FLAGS:
            skip = "=" not in item
            continue
        kept.append(item)
    return [*kept, "--output", str(output), "--shard-index", str(index), "--shard-count", str(count),
            "--shard-sync", str(sync)]


def wait_for_start(sync: Path, index: int, timeout: float = 600.0) -> float:
    """Child side: announce readiness, then block until the parent's start file."""
    ready = sync / f"ready.{index}"
    ready.write_text(str(time.time()))
    deadline = time.monotonic() + timeout
    go = sync / "go"
    while not go.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("shard start signal not received")
        time.sleep(0.002)
    return float(go.read_text())


def merge_predictions(output: Path, shards: list[Path], count: int) -> int:
    """Interleave shard records back to global order and global frame indices."""
    streams = [path.open(encoding="utf-8") for path in shards]
    written = 0
    try:
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as sink:
            for frame in range(count):
                line = streams[frame % len(streams)].readline()
                if not line:
                    raise ValueError(f"shard {frame % len(streams)} ended before frame {frame}")
                record = json.loads(line)
                record["frame"] = frame
                if "schema_version" in record:
                    # time_s follows the global frame index, not the child's local one.
                    finalize_frame(record, record.get("fps"))
                sink.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                written += 1
        if any(stream.readline() for stream in streams):
            raise ValueError("a shard produced more records than assigned")
    finally:
        for stream in streams:
            stream.close()
    return written


def run_sharded(args, argv: list[str], job_started: float) -> int:
    from .benchmark import sha256

    count = args.archive_processes
    args.output.mkdir(parents=True, exist_ok=False)
    sync = args.output / "sync"
    sync.mkdir()
    report = {"schema_version": 1, "output_schema_version": OUTPUT_SCHEMA_VERSION, "status": "running",
              "schedule": "sharded-archive",
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "models_manifest_sha256": sha256(args.models), "index_sha256": sha256(args.index),
              "limitations": ["Children share one GPU; the per-child allocator guard does not bound the device total.",
                              "Each child repeats the same warmup inputs; whole-job time includes N model loads."]}
    threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "8")) // count)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    children, outputs = [], []
    try:
        for index in range(count):
            output = args.output / f"shard_{index}"
            outputs.append(output)
            log = (args.output / f"shard_{index}.log").open("x")
            children.append((subprocess.Popen(
                [sys.executable, "-m", "deepstarnoodles_inference", "benchmark",
                 *child_argv(list(argv), index, count, sync, output)],
                stdout=log, stderr=subprocess.STDOUT, env=env), log))
        deadline = time.monotonic() + 900
        while not all((sync / f"ready.{index}").exists() for index in range(count)):
            failed = [index for index, (child, _) in enumerate(children) if child.poll() is not None]
            if failed:
                raise RuntimeError(f"shard process {failed[0]} exited before starting")
            if time.monotonic() > deadline:
                raise TimeoutError("shard processes did not become ready")
            time.sleep(0.01)
        started = time.time()
        temporary = sync / "go.tmp"
        temporary.write_text(repr(started))
        os.replace(temporary, sync / "go")
        codes = [child.wait() for child, _ in children]
        if any(codes):
            raise RuntimeError(f"shard process exit codes {codes}")
        reports = [json.loads((output / "report.json").read_text()) for output in outputs]
        finished = max(item["measurement"]["finished_epoch"] for item in reports)
        elapsed = finished - started
        written = merge_predictions(args.output, [output / "predictions.jsonl" for output in outputs], args.images)
        measurements = [item["measurement"] for item in reports]
        total = {key: sum(item[key] for item in measurements) for key in ("source_images", "star_crops", "arms",
                                                                         "empty_source_images")}
        if total["source_images"] != args.images or written != args.images:
            raise ValueError("sharded archive did not process every requested input exactly once")
        report["measurement"] = {
            **total, "elapsed_seconds": elapsed,
            "source_images_per_second": args.images / elapsed if elapsed > 0 else None,
            "star_crops_per_second": total["star_crops"] / elapsed if elapsed > 0 else None,
            "timing_scope": "common start signal after every child's warmup through the last child's final flush",
            "shards": [{"index": index, "source_images": item["source_images"], "star_crops": item["star_crops"],
                        "elapsed_seconds": item["elapsed_seconds"],
                        "finished_after_start_seconds": item["finished_epoch"] - started}
                       for index, item in enumerate(measurements)],
        }
        report["warmup"] = reports[0].get("warmup")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "host_peak_rss_bytes"):
            values = [item.get(key) for item in reports if item.get(key) is not None]
            report[key + "_per_shard"] = values
            report[key + "_sum"] = sum(values) if values else None
        report["model_load_seconds"] = max(item.get("model_load_seconds") or 0 for item in reports)
        for key in ("gru_graph", "hydra_graph", "starseg_graph", "inventory", "trace_params", "crop_config",
                    "hydra_config"):
            if key in reports[0]:
                report[key] = reports[0][key]
        if "gpu_crops" in reports[0]:
            report["gpu_crops_per_shard"] = [item.get("gpu_crops") for item in reports]
        report["status"] = "complete"
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for child, log in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            log.close()
        report["total_job_seconds"] = time.perf_counter() - job_started
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
