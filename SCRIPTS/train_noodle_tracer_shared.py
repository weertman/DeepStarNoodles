#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Serial refreshed-cache fanout for final-corpus GRU sizes and controls.

Each consumer starts from random tracer weights, uses the same parent Hydra and
normalizer, and validates on the entire fixed validation split. A Sleepy-only
consumer filters rows of that same refreshed all-corpus generation. It never
regenerates a separate augmentation stream. Checkpoints and completion receipts
are independent per consumer; a generation is pruned only after all consumers
finish and its original manifest identity is rechecked.

Consumer JSON is an array (or {"consumers": [...]}) of objects, for example:
[{"id":"gru64", "hidden_size":64, "control":"full"},
 {"id":"sleepy", "hidden_size":128, "control":"sleepy_only"}].

Use a fresh output after the separately completed seed-0 baseline, listing only
its six remaining consumers. Seeds 1 and 2 each list all seven consumers. This
baseline-first schedule regenerates seed 0's caches; it does not retain 30 large
cache generations. No test cache, released tracer warm start, or endpoint-refined
Hydra is accepted by this driver.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "SCRIPTS"
for path in (REPO / "CORE", SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cache_frozen_outputs import atomic_json, prepare_contract, validate_cache_contract
from deepstarnoodles.noodle_tracer_cache import sha256_file, validate_cache_manifest
from train_noodle_tracer import validate_normalization_source
from train_noodle_tracer_refreshed import generation_name, log, trainer_command

CONTROL_FLAGS = {
    "full": [], "sleepy_only": [], "no_rgb": ["--no-rgb-features"],
    "no_embedding": ["--no-embedding-features"],
    "reset_state": ["--reset-recurrent-state"],
    "no_rollout": ["--rollout-probability", "0"],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def consumers_from(path: Path) -> list[dict]:
    value = read_json(path)
    consumers = value.get("consumers") if isinstance(value, dict) else value
    if not isinstance(consumers, list) or not consumers:
        raise ValueError("consumer config must contain a nonempty array")
    result, ids = [], set()
    for row in consumers:
        if not isinstance(row, dict) or set(row) - {"id", "hidden_size", "control", "source_filter"}:
            raise ValueError("unknown consumer settings")
        name, hidden, control = row.get("id"), row.get("hidden_size"), row.get("control", "full")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("consumer id must be a safe directory name")
        if name in ids or hidden not in (64, 128) or control not in CONTROL_FLAGS:
            raise ValueError("duplicate consumer or unsupported hidden size/control")
        expected_source = "sleepy" if control == "sleepy_only" else "all"
        source = row.get("source_filter", expected_source)
        if source != expected_source or (control != "full" and hidden != 128):
            raise ValueError("controls require hidden128 and their prespecified source filter")
        ids.add(name)
        result.append(dict(id=name, hidden_size=hidden, control=control, source_filter=source))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("output", "checkpoint", "split-dir", "normalization-json", "consumer-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--augmentation-seed", type=int, required=True)
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--first-cycle", type=int, default=1)
    parser.add_argument("--last-cycle", type=int)
    parser.add_argument("--steps-per-cycle", type=int, default=4000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="create this file to stop after the running step (default: <output>/STOP)")
    parser.add_argument("--keep-generations", action="store_true")
    args = parser.parse_args(argv)
    args.last_cycle = args.last_cycle or args.cycles
    if args.stop_file is None:
        args.stop_file = args.output / "STOP"
    if not 1 <= args.first_cycle <= args.last_cycle <= args.cycles:
        parser.error("require 1 <= first-cycle <= last-cycle <= cycles")
    if min(args.steps_per_cycle, args.generation_batch_size) < 1 or args.num_workers < 0:
        parser.error("step and batch counts must be positive; workers must be nonnegative")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    return args


class Stopped(RuntimeError):
    pass


class ChildRunner:
    """One child process group, always reaped before returning or stopping."""
    def __init__(self, stop_file: Path):
        self.stop_file = stop_file
        self.stopped = False
        self.process = None

    def stop(self, signum=None, frame=None):
        self.stopped = True

    def check(self):
        if self.stopped or self.stop_file.exists():
            raise Stopped("STOP requested; no further generation or consumer will start")

    def run(self, command: list[str], logfile: Path):
        self.check()
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as stream:
            self.process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            try:
                while self.process.poll() is None:
                    self.check()
                    time.sleep(0.2)
                self.check()
                if self.process.returncode:
                    raise RuntimeError(f"child failed ({self.process.returncode}); see {logfile}")
            finally:
                # Kill the complete group, including loader workers, even if its
                # direct child has already exited after a failing generation.
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    # The outer campaign runner grants this driver ten seconds
                    # on TERM; finish our own child-group cleanup before that.
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process = None


def validate_parent(args) -> dict:
    digest = sha256_file(args.checkpoint)
    cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("config", {})
    if cfg.get("endpoint_only") is not False or cfg.get("init_checkpoint") is not None:
        raise ValueError("shared training requires the clean unrefined parent Hydra")
    if cfg.get("train_subset") is not None or cfg.get("val_subset") is not None:
        raise ValueError("parent Hydra must use the complete training and validation indexes")
    if cfg.get("train_index_sha256") != sha256_file(args.split_dir / "train_index.csv"):
        raise ValueError("Hydra was not trained on this full mixed training index")
    if cfg.get("val_index_sha256") != sha256_file(args.split_dir / "validation_index.csv"):
        raise ValueError("Hydra validation index differs from the frozen validation split")
    normalizer = read_json(args.normalization_json)
    validate_normalization_source(normalizer, digest)
    if normalizer.get("source_split") != "train" or normalizer.get("frozen_checkpoint_sha256") != digest:
        raise ValueError("normalizer must identify this parent checkpoint and training-only source")
    for key in ("mean", "std"):
        values = normalizer.get(key, [])
        if len(values) != 18 or any(not math.isfinite(float(x)) for x in values):
            raise ValueError("normalizer must contain 18 finite channels")
    if any(float(x) <= 0 for x in normalizer["std"]):
        raise ValueError("normalizer standard deviations must be positive")
    return {"checkpoint_sha256": digest, "normalization_sha256": sha256_file(args.normalization_json),
            "split_index_sha256": {s: sha256_file(args.split_dir / f"{s}_index.csv")
                                   for s in ("train", "validation", "test")}}


def prepare(args, consumers, runner) -> tuple[dict, str]:
    identity = validate_parent(args)
    contract_path = args.output / "experiment_contract.json"
    contract = read_json(contract_path) if contract_path.exists() else prepare_contract(
        output=args.output, checkpoint=args.checkpoint, split_dir=args.split_dir, cohort="all")
    validate_cache_contract(output=args.output, checkpoint=args.checkpoint, contract=contract,
                            splits=("train", "validation"))
    if contract.get("cohort") != "all":
        raise ValueError("shared generations must cover the full mixed corpus")
    families = contract["splits"]["train"].get("source_families", {})
    if families.get("sleepy_stars", 0) < 1 or sum(families.values()) <= families.get("sleepy_stars", 0):
        raise ValueError("full mixed training must include Sleepy and other source families")
    for split, digest in identity["split_index_sha256"].items():
        if contract["splits"][split].get("source_sha256") != digest:
            raise ValueError("prepared contract does not match current source indexes")
    recipe = {"schema_version": 1, **identity, "contract_id": contract["contract_id"],
              "consumers": consumers, "seed": args.seed, "augmentation_seed": args.augmentation_seed,
              "cycles": args.cycles, "steps_per_cycle": args.steps_per_cycle,
              "draws_per_image": 1, "generation_cohort": "all", "validation_cohort": "all",
              "context_mode": "single", "initialization": "random_tracer_pretrained_frozen_rgb"}
    recipe_path = args.output / "shared_recipe.json"
    if recipe_path.exists() and read_json(recipe_path) != recipe:
        raise ValueError("shared recipe differs; use a new output directory")
    if not recipe_path.exists():
        atomic_json(recipe_path, recipe)
    validation_root = args.output / "validation" / "all"
    val_manifest = validation_root / "cache" / "validation_manifest.json"
    if not val_manifest.exists():
        runner.run([sys.executable, str(SCRIPTS / "cache_fixed_tracer_split.py"),
                    "--output", str(validation_root), "--checkpoint", str(args.checkpoint),
                    "--split-dir", str(args.split_dir), "--normalization-json", str(args.normalization_json),
                    "--cohort", "all", "--split", "validation", "--device", args.device,
                    "--batch-size", str(args.generation_batch_size), "--num-workers", str(args.num_workers)],
                   args.output / "logs" / "validation_cache.log")
    val_contract = read_json(validation_root / "experiment_contract.json")
    validate_cache_contract(output=validation_root, checkpoint=args.checkpoint, contract=val_contract,
                            splits=("validation",))
    if val_contract.get("cohort") != "all" or val_contract["splits"]["validation"]["source_sha256"] != identity["split_index_sha256"]["validation"]:
        raise ValueError("validation must cover the entire frozen validation index")
    val_data = validate_cache_manifest(val_manifest, val_contract, split="validation")
    verify_statistics(val_data, args.normalization_json)
    return contract, sha256_file(recipe_path)


def verify_statistics(manifest, normalizer_path):
    normalizer = read_json(normalizer_path)
    stats = manifest["provenance"].get("channel_statistics", {})
    if stats.get("source_split") != "train":
        raise ValueError("cache statistics must come from training data")
    for key in ("mean", "std"):
        actual = stats.get(key, [])
        if len(actual) != 18 or any(not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-7)
                                   for a, b in zip(actual, normalizer[key])):
            raise ValueError("cache statistics differ from the fixed parent normalizer")


def generation_path(args, cycle):
    return args.output / "cache" / "generations" / generation_name(cycle)


def generation_identity(args, contract, cycle, *, check_files=True) -> str:
    directory = generation_path(args, cycle)
    active = args.output / "cache" / "train_manifest.json"
    original = directory / "manifest.json"
    if directory.is_symlink() or directory.resolve() != directory:
        raise ValueError("generation directory must be owned directly by this study")
    digest = sha256_file(original)
    if sha256_file(active) != digest:
        raise ValueError("published manifest changed from original generation identity")
    data = validate_cache_manifest(active, contract, split="train")
    provenance = data["provenance"]
    if (provenance.get("generation") != generation_name(cycle)
            or provenance.get("augmentation_seed") != args.augmentation_seed
            or provenance.get("draws_per_image") != 1):
        raise ValueError("generation augmentation identity differs")
    verify_statistics(data, args.normalization_json)
    for record in data["records"]:
        path = args.output / "cache" / record["cache_file"]
        if not path.resolve().is_relative_to(directory) or path.is_symlink():
            raise ValueError("generation references a cache outside its owned directory")
        if check_files and not path.is_file():
            raise ValueError("generation cache record is missing")
    return digest


def consumer_dir(args, consumer, cycle):
    return args.output / "consumers" / consumer["id"] / "training" / generation_name(cycle)


def command_for(args, consumer, cycle):
    adapted = SimpleNamespace(**vars(args))
    adapted.output = args.output / "consumers" / consumer["id"]
    adapted.validation_cohort, adapted.draws_per_image = "all", 1
    adapted.init_checkpoint, adapted.trainer_args = None, ""
    command = trainer_command(adapted, cycle)
    command[command.index("--study-root") + 1] = str(args.output)
    command[command.index("--validation-study-root") + 1] = str(args.output / "validation" / "all")
    return command + ["--hidden-size", str(consumer["hidden_size"]),
                      "--train-source-cohort", consumer["source_filter"], *CONTROL_FLAGS[consumer["control"]]]


def completion(args, consumer, cycle, recipe_sha, *, required=False):
    directory = consumer_dir(args, consumer, cycle)
    receipt_path = directory / "complete.json"
    if not receipt_path.exists():
        if required:
            raise ValueError(f"missing completed predecessor for {consumer['id']} cycle {cycle}")
        return None
    receipt = read_json(receipt_path)
    if (receipt.get("recipe_sha256") != recipe_sha or receipt.get("consumer") != consumer
            or receipt.get("cycle") != cycle
            or receipt.get("checkpoint_sha256") != sha256_file(directory / "last.pt")
            or receipt.get("generation_manifest_sha256") != sha256_file(directory / "train_manifest.json")):
        raise ValueError(f"completion receipt identity changed: {directory}")
    return receipt


def seal_consumer(args, consumer, cycle, recipe_sha, generation_sha):
    directory = consumer_dir(args, consumer, cycle)
    payload = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("epoch") != cycle:
        raise ValueError("trainer did not produce the requested completed cycle")
    expected = {
        "hidden_size": consumer["hidden_size"], "train_source_cohort": consumer["source_filter"],
        "rgb_features": consumer["control"] != "no_rgb",
        "embedding_features": consumer["control"] != "no_embedding",
        "reset_recurrent_state": consumer["control"] == "reset_state",
        "rollout_probability": 0.0 if consumer["control"] == "no_rollout" else 0.3,
        "seed": args.seed, "steps_per_epoch": args.steps_per_cycle, "lr_decay_epochs": args.cycles,
        "max_train_images": None, "max_val_images": None,
    }
    signature = metadata.get("training_signature", {})
    if any(key not in signature or signature[key] != value for key, value in expected.items()):
        raise ValueError("trainer checkpoint settings differ from the requested consumer")
    with (directory / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    if (len(metrics) != 1 or int(metrics[0]["epoch"]) != cycle
            or float(metrics[0]["train_optimizer_steps"]) != args.steps_per_cycle):
        raise ValueError("trainer did not complete exactly the prescribed successful updates")
    atomic_text(directory / "train_manifest.json",
                (generation_path(args, cycle) / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(directory / "train_manifest.json") != generation_sha:
        raise ValueError("generation changed while recording consumer provenance")
    digest = sha256_file(directory / "last.pt")
    atomic_text(directory / "last.pt.sha256", f"{digest}  last.pt\n")
    atomic_json(directory / "complete.json", {
        "schema_version": 1, "consumer": consumer, "cycle": cycle,
        "recipe_sha256": recipe_sha, "checkpoint_sha256": digest,
        "generation_manifest_sha256": generation_sha,
    })


def run_cycles(args, consumers, contract, recipe_sha, runner):
    for cycle in range(args.first_cycle, args.last_cycle + 1):
        runner.check()
        finished = [completion(args, c, cycle, recipe_sha) for c in consumers]
        if all(finished):
            directory = generation_path(args, cycle)
            if directory.exists() and not args.keep_generations:
                digest = generation_identity(args, contract, cycle)
                if any(done["generation_manifest_sha256"] != digest for done in finished):
                    raise ValueError("completed consumers do not own the generation to prune")
                runner.check()
                shutil.rmtree(directory)
            continue
        for consumer, done in zip(consumers, finished):
            if not done and cycle > 1:
                completion(args, consumer, cycle - 1, recipe_sha, required=True)
        directory = generation_path(args, cycle)
        if not (directory / "manifest.json").exists():
            log(args.output, f"{generation_name(cycle)} shared generation start")
            runner.run([sys.executable, str(SCRIPTS / "generate_refreshed_tracer_cache.py"),
                        "--study-root", str(args.output), "--checkpoint", str(args.checkpoint),
                        "--normalization-json", str(args.normalization_json), "--cycle", str(cycle),
                        "--seed", str(args.augmentation_seed), "--draws-per-image", "1",
                        "--device", args.device, "--batch-size", str(args.generation_batch_size)],
                       args.output / "logs" / f"{generation_name(cycle)}_generation.log")
        generation_sha = generation_identity(args, contract, cycle)
        # A completed consumer and a resumed consumer must have seen precisely
        # the same manifest, including if a pruned generation was regenerated.
        if any(done and done["generation_manifest_sha256"] != generation_sha for done in finished):
            raise ValueError("regenerated manifest differs from a completed consumer's inputs")
        for consumer, done in zip(consumers, finished):
            if done:
                continue
            runner.check()
            if generation_identity(args, contract, cycle, check_files=False) != generation_sha:
                raise ValueError("generation changed between consumers")
            output = consumer_dir(args, consumer, cycle)
            if output.exists():
                output.rename(output.with_name(f"{output.name}.incomplete.{time.time_ns()}"))
            output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output / "progress.json", {
                "stage": "training", "cycle": cycle, "consumer": consumer["id"],
                "generation_manifest_sha256": generation_sha,
            })
            log(args.output, f"{generation_name(cycle)} {consumer['id']} training start")
            runner.run(command_for(args, consumer, cycle),
                       args.output / "logs" / f"{generation_name(cycle)}_{consumer['id']}.log")
            if generation_identity(args, contract, cycle, check_files=False) != generation_sha:
                raise ValueError("generation changed during consumer training")
            seal_consumer(args, consumer, cycle, recipe_sha, generation_sha)
        for consumer in consumers:
            completion(args, consumer, cycle, recipe_sha, required=True)
        if not args.keep_generations:
            runner.check()
            if generation_identity(args, contract, cycle, check_files=False) != generation_sha:
                raise ValueError("refusing to prune a changed generation")
            shutil.rmtree(directory)
        atomic_json(args.output / "progress.json", {"stage": "cycle_complete", "cycle": cycle})
    summary = {"stage": "complete" if args.last_cycle == args.cycles else "range_complete",
               "last_cycle": args.last_cycle, "schedule_cycles": args.cycles,
               "recipe_sha256": recipe_sha, "consumers": {
                   c["id"]: completion(args, c, args.last_cycle, recipe_sha, required=True) for c in consumers}}
    atomic_json(args.output / "shared_summary.json", summary)
    atomic_json(args.output / "progress.json", summary)


def main(argv=None) -> int:
    args = parse_args(argv)
    consumers = consumers_from(args.consumer_config)
    (args.output / "logs").mkdir(parents=True, exist_ok=True)
    runner = ChildRunner(args.stop_file)
    previous_handlers = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (args.output / ".shared_driver.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another shared driver owns this output") from error
            runner.check()
            contract, recipe_sha = prepare(args, consumers, runner)
            run_cycles(args, consumers, contract, recipe_sha, runner)
    except Stopped as error:
        atomic_json(args.output / "progress.json", {"stage": "stopped", "reason": str(error)})
        return 130
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
