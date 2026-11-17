# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Lightweight command routing; model imports are lazy."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Self-contained StarSeg → Hydra → GRU inference")
    parser.add_argument("command", choices=("benchmark", "archive", "live", "live-replay", "staged-archive", "serve-videos",
                                            "track-videos", "verify-models", "fetch-models", "compare", "representations",
                                            "smoke"))
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    remaining = argv[1:]
    if args.command in ("benchmark", "archive", "live-replay"):
        from .benchmark import main as run
        if args.command != "benchmark":
            if any(value == "--mode" or value.startswith("--mode=") for value in remaining):
                parser.error("archive/live-replay select --mode automatically")
            if not any(value == "--images" or value.startswith("--images=") for value in remaining) and not any(value in ("-h", "--help") for value in remaining):
                from pathlib import Path
                from .benchmark import input_paths
                index_parser = argparse.ArgumentParser(add_help=False)
                index_parser.add_argument("--index", type=Path, required=True)
                index_args, _ = index_parser.parse_known_args(remaining)
                remaining += ["--images", str(sum(1 for _ in input_paths(index_args.index, None)))]
            if not any(value == "--warmup" or value.startswith("--warmup=") for value in remaining):
                remaining += ["--warmup", "0"]
            remaining += ["--mode", args.command]
        return run(remaining)
    if args.command == "serve-videos":
        from .video_service import main as run
        return run(remaining)
    if args.command == "compare":
        from .compare_cli import main as run
        return run(remaining)
    if args.command == "track-videos":
        # Needs Ultralytics and the arm tracker, but no GPU, models or video.
        from .video_tracking import main as run
        return run(remaining)
    if args.command == "representations":
        # Torch-free: reads inference records and writes the paper's polyline representations.
        from .representations import main as run
        return run(remaining)
    if args.command == "live":
        from .live import main as run
        return run(remaining)
    if args.command == "staged-archive":
        from .staged import main as run
        if any(value in ("-h", "--help") for value in remaining):
            print("Staged archive options: --shard-size N (default 64), "
                  "--max-spool-gib GIB (default 8). "
                  "This command defaults to --mode archive --warmup 0.\n")
        if not any(value == "--mode" or value.startswith("--mode=") for value in remaining):
            remaining += ["--mode", "archive"]
        if not any(value == "--warmup" or value.startswith("--warmup=") for value in remaining):
            remaining += ["--warmup", "0"]
        return run(remaining)
    if args.command in ("verify-models", "fetch-models"):
        from .artifacts import main as run
        return run([args.command, *remaining])
    from .smoke import main as run
    return run(remaining)
