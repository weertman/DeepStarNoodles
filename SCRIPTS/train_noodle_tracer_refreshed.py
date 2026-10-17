#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Train the Noodle Tracer with the released refreshed full-corpus recipe.

The defaults reproduce the released tracer's training:

1. prepare a study of every training image (``--cohort all``) with the frozen
   parent Hydra (``MODELS/hydra/epoch_060.pt``);
2. cache the fixed Sleepy validation split used for in-training validation;
3. for each of 30 cycles, generate a refreshed generation (fresh pixel-space
   augmentation of every image, Hydra rerun on the augmented image), then train
   exactly 4,000 successful updates from the previous cycle's checkpoint, with
   the fixed normalization in ``config/recipes/``.

The learning rate warms up over two cycles and then follows a cosine decay from
3e-4 to 3e-5 at the final cycle; rollout training ramps to 30% by cycle 12.
Re-running the same command resumes after the last completed cycle. Select the
invalid-seed threshold afterwards with ``calibrate_noodle_tracer.py``.

``--cohort sleepy --draws-per-image 10`` gives the matched-update Sleepy-only
comparison: ten augmented draws of each Sleepy image per cycle and the same
update budget.

To adapt the tracer to new scenes, point ``--split-dir`` at your own split
indexes, use ``--cohort all --validation-cohort all``, and optionally start from
the released weights with ``--init-checkpoint``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
CORE = REPO / "CORE"
for path in (CORE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepstarnoodles.noodle_tracer_cache import TRACER_COHORTS, sha256_file  # noqa: E402
from cache_frozen_outputs import DEFAULT_SPLITS, prepare_contract  # noqa: E402
from cache_fixed_tracer_split import cache_fixed_split  # noqa: E402
from generate_refreshed_tracer_cache import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_NORMALIZATION,
    RELEASED_AUGMENTATION_SEED,
)

RELEASED_SEED = 20261011
RELEASED_CYCLES = 30
RELEASED_STEPS_PER_CYCLE = 4000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True, help="study directory under RUNS/")
    parser.add_argument("--cohort", choices=TRACER_COHORTS, default="all")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="frozen training Hydra (default: MODELS/hydra/epoch_060.pt)")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--normalization-json", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--validation-cohort", choices=TRACER_COHORTS, default="sleepy",
                        help="fixed validation cohort scored after every cycle (default: sleepy)")
    parser.add_argument("--cycles", type=int, default=RELEASED_CYCLES)
    parser.add_argument("--steps-per-cycle", type=int, default=RELEASED_STEPS_PER_CYCLE,
                        help="successful optimizer updates per cycle")
    parser.add_argument("--seed", type=int, default=RELEASED_SEED, help="tracer training seed")
    parser.add_argument("--augmentation-seed", type=int, default=RELEASED_AUGMENTATION_SEED)
    parser.add_argument("--draws-per-image", type=int, default=1,
                        help="augmented draws of each training image per cycle")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="start cycle 1 from these tracer weights (for example the released "
                             "tracer) instead of a random initialization")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4, help="training data-loader workers")
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--no-prefetch", action="store_true",
                        help="generate each cycle only after the previous cycle has trained")
    parser.add_argument("--keep-generations", action="store_true",
                        help="keep each cycle's cached generation (about 27 GB for the full corpus)")
    parser.add_argument("--trainer-args", default="",
                        help="extra arguments for train_noodle_tracer.py, e.g. \"--lr 1e-4\"")
    args = parser.parse_args(argv)
    if args.cycles < 1 or args.steps_per_cycle < 1 or args.draws_per_image < 1:
        parser.error("--cycles, --steps-per-cycle and --draws-per-image must be positive")
    return args


def log(output: Path, message: str) -> None:
    line = f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {message}"
    print(line, flush=True)
    with (output / "logs" / "driver.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def generation_name(cycle: int) -> str:
    return f"g{cycle:04d}"


def run_dir(output: Path, cycle: int) -> Path:
    return output / "training" / generation_name(cycle)


def published_generation(output: Path) -> str | None:
    manifest = output / "cache" / "train_manifest.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8"))["provenance"].get("generation")


def cycle_complete(output: Path, cycle: int) -> bool:
    return (run_dir(output, cycle) / "last.pt.sha256").is_file()


def start_generation(args: argparse.Namespace, cycle: int) -> subprocess.Popen | None:
    if published_generation(args.output) == generation_name(cycle):
        return None
    log(args.output, f"{generation_name(cycle)} generation start")
    command = [
        sys.executable, str(SCRIPTS / "generate_refreshed_tracer_cache.py"),
        "--study-root", str(args.output), "--checkpoint", str(args.checkpoint),
        "--normalization-json", str(args.normalization_json), "--cycle", str(cycle),
        "--seed", str(args.augmentation_seed), "--draws-per-image", str(args.draws_per_image),
        "--device", args.device, "--batch-size", str(args.generation_batch_size),
    ]
    stream = (args.output / "logs" / f"{generation_name(cycle)}_generation.log").open("w")
    return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)


def trainer_command(args: argparse.Namespace, cycle: int) -> list[str]:
    command = [
        sys.executable, str(SCRIPTS / "train_noodle_tracer.py"),
        "--study-root", str(args.output),
        "--validation-study-root", str(args.output / "validation" / args.validation_cohort),
        "--out-dir", str(run_dir(args.output, cycle)),
        "--device", args.device, "--seed", str(args.seed),
        "--epochs", str(cycle), "--lr-decay-epochs", str(args.cycles),
        "--steps-per-epoch", str(args.steps_per_cycle),
        "--batch-size", "1", "--num-workers", str(args.num_workers),
        "--context-mode", "single", "--normalization-json", str(args.normalization_json),
        "--pretrained-rgb", "--freeze-rgb", "--no-augment", "--amp",
    ]
    if args.draws_per_image > 1:
        command.append("--repeated-train-cache")
    if cycle > 1:
        command += ["--resume", str(run_dir(args.output, cycle - 1) / "last.pt")]
    elif args.init_checkpoint is not None:
        command += ["--init-checkpoint", str(args.init_checkpoint)]
    return command + shlex.split(args.trainer_args)


def wait(process: subprocess.Popen, label: str, output: Path) -> None:
    if process.wait() != 0:
        raise SystemExit(f"{label} failed with exit code {process.returncode}; see {output / 'logs'}")


def write_recipe(args: argparse.Namespace, contract: dict) -> None:
    path = args.output / "recipe.json"
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    recipe = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "trainer_command_cycle_1": trainer_command(args, 1),
        "contract_id": contract["contract_id"],
        "cohort_counts": {split: {k: v for k, v in values.items()
                                  if k in ("images", "annotated_arm_polylines", "source_families")}
                          for split, values in contract["splits"].items()},
        "hydra_sha256": sha256_file(args.checkpoint),
        "init_checkpoint_sha256": (sha256_file(args.init_checkpoint)
                                   if args.init_checkpoint is not None else None),
        "normalization_sha256": sha256_file(args.normalization_json),
        "split_index_sha256": {split: sha256_file(args.split_dir / f"{split}_index.csv")
                               for split in ("train", "validation", "test")},
        "git_commit": commit or None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for key in ("hydra_sha256", "normalization_sha256", "split_index_sha256", "contract_id",
                    "init_checkpoint_sha256"):
            if previous.get(key) != recipe[key]:
                raise SystemExit(f"{path} records a different {key}; use a new --output directory")
        for key in ("cohort", "cycles", "steps_per_cycle", "seed", "augmentation_seed",
                    "draws_per_image", "validation_cohort", "trainer_args"):
            if previous["arguments"].get(key) != recipe["arguments"][key]:
                raise SystemExit(f"{path} records a different --{key.replace('_', '-')}; "
                                 "use a new --output directory")
        return
    path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.split_dir = args.split_dir.expanduser().resolve()
    args.normalization_json = args.normalization_json.expanduser().resolve()
    if args.init_checkpoint is not None:
        args.init_checkpoint = args.init_checkpoint.expanduser().resolve()
    (args.output / "logs").mkdir(parents=True, exist_ok=True)
    contract_path = args.output / "experiment_contract.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract.get("cohort") != args.cohort:
            raise SystemExit(f"{args.output} was prepared for cohort {contract.get('cohort')!r}")
    else:
        contract = prepare_contract(output=args.output, checkpoint=args.checkpoint,
                                    split_dir=args.split_dir, cohort=args.cohort)
    write_recipe(args, contract)
    validation_root = args.output / "validation" / args.validation_cohort
    if not (validation_root / "cache" / "validation_manifest.json").is_file():
        log(args.output, f"caching fixed {args.validation_cohort} validation split")
        cache_fixed_split(output=validation_root, checkpoint=args.checkpoint,
                          split_dir=args.split_dir, normalization_json=args.normalization_json,
                          cohort=args.validation_cohort, split="validation", device=args.device)
    pending: subprocess.Popen | None = None
    for cycle in range(1, args.cycles + 1):
        if cycle_complete(args.output, cycle):
            continue
        if pending is not None:
            wait(pending, f"{generation_name(cycle)} generation", args.output)
            pending = None
        process = start_generation(args, cycle)
        if process is not None:
            wait(process, f"{generation_name(cycle)} generation", args.output)
        if published_generation(args.output) != generation_name(cycle):
            raise SystemExit(f"published training manifest is not {generation_name(cycle)}")
        output_dir = run_dir(args.output, cycle)
        if output_dir.exists():
            output_dir.rename(output_dir.with_name(f"{output_dir.name}.incomplete.{int(time.time())}"))
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        log(args.output, f"{generation_name(cycle)} training start")
        stream = (args.output / "logs" / f"{generation_name(cycle)}_training.log").open("w")
        trainer = subprocess.Popen(trainer_command(args, cycle), stdout=stream, stderr=subprocess.STDOUT)
        if cycle < args.cycles and not args.no_prefetch:
            # The trainer reads the published manifest at start-up (config.json is
            # written afterwards); only then may the next generation replace it.
            while not (output_dir / "config.json").is_file() and trainer.poll() is None:
                time.sleep(5)
            if trainer.poll() in (None, 0):
                pending = start_generation(args, cycle + 1)
        wait(trainer, f"{generation_name(cycle)} training", args.output)
        checkpoint = output_dir / "last.pt"
        (output_dir / "last.pt.sha256").write_text(f"{sha256_file(checkpoint)}  last.pt\n")
        generation_dir = args.output / "cache" / "generations" / generation_name(cycle)
        if (generation_dir / "manifest.json").is_file():
            shutil.copy(generation_dir / "manifest.json", output_dir / "train_manifest.json")
        if not args.keep_generations:
            shutil.rmtree(generation_dir, ignore_errors=True)
        last_row = (output_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()[-1]
        log(args.output, f"{generation_name(cycle)} complete: {last_row}")
    if pending is not None:
        wait(pending, "final prefetch", args.output)
    final = run_dir(args.output, args.cycles) / "last.pt"
    summary = {"checkpoint": str(final), "sha256": sha256_file(final), "cycles": args.cycles}
    (args.output / "final.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(args.output, f"training complete: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
