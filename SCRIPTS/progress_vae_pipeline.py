#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Unattended sequencing for the temporal VAE stages behind the archive extraction.

1. Once the progress cache holds ``--stage1-min-clips`` complete clips and no
   archive Hydra pass is running, fine-tune a preliminary model and encode.
2. Once the extraction process has exited, rebuild session splits over the full
   cache, fine-tune the final model, encode all clips, aggregate metrics.
3. Mirror caches, latents, splits, runs, and checkpoints to the LaCie.

Every stage is a subprocess of the sibling scripts, so any stage can be re-run
by hand with the same arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CORE"))

from deepstarnoodles.paths import DATA_ROOT, MODELS_ROOT, RUNS_ROOT
from deepstarnoodles.progress_cache import DEFAULT_PROGRESS_CACHE_ROOT, inventory_progress_cache

PYTHON = REPO / ".venv" / "bin" / "python"
SCRIPTS = REPO / "SCRIPTS"
DEFAULT_LACIE = Path("/media/starbot/7479-B887/deepstarnoodles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_PROGRESS_CACHE_ROOT)
    parser.add_argument("--stage1-min-clips", type=int, default=2500)
    parser.add_argument("--stage1-steps", type=int, default=4000)
    parser.add_argument("--final-steps", type=int, default=8000)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--memory-fraction", type=float, default=0.3)
    parser.add_argument("--lacie-root", type=Path, default=DEFAULT_LACIE)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--state", type=Path, default=RUNS_ROOT / "progress_vae" / "pipeline_state.json")
    return parser.parse_args()


def log(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}, default=str), flush=True)


def process_running(pattern: str) -> bool:
    """True when a Python interpreter (not a shell or editor sandbox quoting the
    command line) is running the given script invocation."""
    result = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, comm, args = parts
        if int(pid) == os.getpid() or not comm.startswith("python"):
            continue
        if pattern in args:
            return True
    return False


def cache_count(root: Path) -> int:
    return len(inventory_progress_cache(root))


def run(command: list[str]) -> int:
    log("run", command=" ".join(command))
    completed = subprocess.run(command)
    log("finished", returncode=completed.returncode, command=command[1:3])
    return completed.returncode


def load_state(path: Path) -> dict[str, object]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finetune(name: str, *, splits: Path, steps: int, memory_fraction: float, remake: bool) -> int:
    command = [
        str(PYTHON), "-u", str(SCRIPTS / "finetune_progress_vae.py"),
        "--name", name,
        "--splits", str(splits),
        "--steps", str(steps),
        "--memory-fraction", str(memory_fraction),
    ]
    if remake:
        command.append("--remake-splits")
    return run(command)


def encode(name: str, *, splits: Path) -> int:
    checkpoint = MODELS_ROOT / "temporal_vae" / f"dnt_progress_temporal_vae_60s_{name}.pt"
    return run(
        [
            str(PYTHON), "-u", str(SCRIPTS / "encode_progress_latents.py"),
            "--checkpoint", str(checkpoint),
            "--splits", str(splits),
        ]
    )


def backup(lacie_root: Path) -> int:
    if not lacie_root.parent.is_dir() or shutil.which("rsync") is None:
        log("backup_skipped", reason="LaCie not mounted or rsync missing", root=lacie_root)
        return 1
    lacie_root.mkdir(parents=True, exist_ok=True)
    sources = [
        DATA_ROOT / "deepstarnoodles" / "progress_cache",
        DATA_ROOT / "deepstarnoodles" / "progress_latents",
        MODELS_ROOT / "temporal_vae",
        RUNS_ROOT / "progress_vae",
    ]
    sources += list((DATA_ROOT / "deepstarnoodles").glob("progress_vae_splits*.json"))
    existing = [str(source) for source in sources if source.exists()]
    return run(["rsync", "-rlt", "--update", "--info=stats2", *existing, str(lacie_root) + "/"])


def main() -> int:
    args = parse_args()
    state = load_state(args.state)
    stage1_splits = DATA_ROOT / "deepstarnoodles" / "progress_vae_splits_stage1.json"
    final_splits = DATA_ROOT / "deepstarnoodles" / "progress_vae_splits.json"
    log("pipeline_start", state=state, cache_clips=cache_count(args.cache_root))

    while True:
        extracting = process_running("SCRIPTS/extract_progress_cache.py --run")
        hydra_pass = process_running("SCRIPTS/append_hydra_archive.py --run")
        count = cache_count(args.cache_root)

        if (
            not args.skip_stage1
            and not state.get("stage1_done")
            and count >= args.stage1_min_clips
            and not hydra_pass
        ):
            log("stage1_begin", cache_clips=count)
            code = finetune(
                "archive_ft_stage1",
                splits=stage1_splits,
                steps=args.stage1_steps,
                memory_fraction=args.memory_fraction,
                remake=True,
            )
            if code == 0:
                encode("archive_ft_stage1", splits=stage1_splits)
                backup(args.lacie_root)
            state["stage1_done"] = code == 0
            state["stage1_cache_clips"] = count
            save_state(args.state, state)

        if not extracting and not hydra_pass:
            log("final_begin", cache_clips=count)
            code = finetune(
                "archive_ft",
                splits=final_splits,
                steps=args.final_steps,
                memory_fraction=args.memory_fraction,
                remake=True,
            )
            state["final_finetune_returncode"] = code
            if code == 0:
                state["final_encode_returncode"] = encode("archive_ft", splits=final_splits)
            state["backup_returncode"] = backup(args.lacie_root)
            state["final_done"] = code == 0
            save_state(args.state, state)
            log("pipeline_done", state=state)
            return 0 if code == 0 else 1

        log("waiting", cache_clips=count, extracting=extracting, hydra_pass=hydra_pass)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
