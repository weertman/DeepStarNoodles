"""Shared-cache fanout ownership, independent resume, protocol and stop tests."""
from __future__ import annotations

import json
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_noodle_tracer_shared as shared
from deepstarnoodles.noodle_tracer_cache import FROZEN_CACHE_SCHEMA


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def study(tmp_path):
    args = SimpleNamespace(output=tmp_path / "study", checkpoint=tmp_path / "parent.pt",
        split_dir=tmp_path / "splits", normalization_json=tmp_path / "normalizer.json",
        seed=0, augmentation_seed=100, cycles=2, first_cycle=1, last_cycle=2,
        steps_per_cycle=2, device="cpu", num_workers=0, generation_batch_size=1,
        keep_generations=False, stop_file=tmp_path / "STOP")
    (args.output / "logs").mkdir(parents=True)
    write_json(args.normalization_json, {"mean": [0.0] * 18, "std": [1.0] * 18})
    consumers = [dict(id="gru64", hidden_size=64, control="full", source_filter="all"),
                 dict(id="sleepy", hidden_size=128, control="sleepy_only", source_filter="sleepy")]
    contract = {"contract_id": "contract", "frozen_checkpoint_sha256": "parent",
                "cohort": "all", "splits": {"train": {"records": 2, "materialized_sha256": "index"}}}
    return args, consumers, contract


def publish(args, contract, cycle):
    directory = shared.generation_path(args, cycle)
    records = []
    for index, family in enumerate(("sleepy_stars", "other")):
        path = directory / "train" / f"{index}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable cached tensor")
        records.append({"cache_file": str(path.relative_to(args.output / "cache")),
                        "source_family": family, "cache_identity": str(index),
                        "image_sha256": f"image{index}", "shapes_sha256": f"shape{index}"})
    manifest = {"schema_version": FROZEN_CACHE_SCHEMA, "split": "train", "records": records,
                "provenance": {"contract_id": contract["contract_id"],
                    "frozen_checkpoint_sha256": contract["frozen_checkpoint_sha256"],
                    "index_sha256": "index", "generation": shared.generation_name(cycle),
                    "augmentation_seed": args.augmentation_seed, "draws_per_image": 1,
                    "channel_statistics": {"source_split": "train", "mean": [0.0] * 18, "std": [1.0] * 18}}}
    write_json(directory / "manifest.json", manifest)
    write_json(args.output / "cache" / "train_manifest.json", manifest)


class FakeRunner:
    def __init__(self, args, contract, fail=None, mutate=False):
        self.args, self.contract = args, contract
        self.commands, self.fail, self.mutate = [], fail, mutate

    def check(self):
        pass

    def run(self, command, logfile):
        self.commands.append(command)
        if command[1].endswith("generate_refreshed_tracer_cache.py"):
            publish(self.args, self.contract, int(command[command.index("--cycle") + 1]))
            return
        cycle = int(command[command.index("--epochs") + 1])
        output = Path(command[command.index("--out-dir") + 1])
        output.mkdir(parents=True)
        if self.fail == (output.parents[1].name, cycle):
            raise RuntimeError("mock trainer failure")
        signature = {"hidden_size": int(command[command.index("--hidden-size") + 1]),
            "train_source_cohort": command[command.index("--train-source-cohort") + 1],
            "rgb_features": "--no-rgb-features" not in command,
            "embedding_features": "--no-embedding-features" not in command,
            "reset_recurrent_state": "--reset-recurrent-state" in command,
            "rollout_probability": 0.0 if "--rollout-probability" in command else 0.3,
            "seed": self.args.seed, "steps_per_epoch": self.args.steps_per_cycle,
            "lr_decay_epochs": self.args.cycles, "max_train_images": None, "max_val_images": None}
        torch.save({"metadata": {"epoch": cycle, "training_signature": signature}}, output / "last.pt")
        (output / "metrics.csv").write_text(
            f"epoch,train_optimizer_steps\n{cycle},{self.args.steps_per_cycle}\n")
        if self.mutate:
            active = self.args.output / "cache" / "train_manifest.json"
            active.write_text(active.read_text() + "\n")


def test_shared_generation_once_per_cycle_no_data_or_normalizer_overrides(study):
    args, consumers, contract = study
    runner = FakeRunner(args, contract)
    shared.run_cycles(args, consumers, contract, "recipe", runner)
    generation = [c for c in runner.commands if "--cycle" in c]
    training = [c for c in runner.commands if "--epochs" in c]
    assert len(generation) == 2 and len(training) == 4
    for c in generation:
        assert c[c.index("--draws-per-image") + 1] == "1"
    for c in training:
        assert c[c.index("--study-root") + 1] == str(args.output)
        assert c[c.index("--validation-study-root") + 1] == str(args.output / "validation" / "all")
        assert c[c.index("--normalization-json") + 1] == str(args.normalization_json)
        assert "--init-checkpoint" not in c and "--max-val-images" not in c
        cycle = int(c[c.index("--epochs") + 1])
        if cycle == 2:
            assert Path(c[c.index("--resume") + 1]).parts[-3:] == ("training", "g0001", "last.pt")
    assert not list((args.output / "cache" / "generations").iterdir())
    before = len(runner.commands)
    shared.run_cycles(args, consumers, contract, "recipe", runner)
    assert len(runner.commands) == before


def test_failure_keeps_cache_and_successful_consumer_resumes_independently(study):
    args, consumers, contract = study
    failed = FakeRunner(args, contract, fail=("sleepy", 1))
    with pytest.raises(RuntimeError, match="mock trainer failure"):
        shared.run_cycles(args, consumers, contract, "recipe", failed)
    assert shared.generation_path(args, 1).exists()
    assert shared.completion(args, consumers[0], 1, "recipe")
    assert shared.completion(args, consumers[1], 1, "recipe") is None
    resumed = FakeRunner(args, contract)
    shared.run_cycles(args, consumers, contract, "recipe", resumed)
    assert len([c for c in resumed.commands if "--cycle" in c]) == 1
    assert len([c for c in resumed.commands if "--epochs" in c]) == 3
    assert list((args.output / "consumers" / "sleepy" / "training").glob("g0001.incomplete.*"))


def test_changed_manifest_never_seals_or_deletes_generation(study):
    args, consumers, contract = study
    with pytest.raises(ValueError, match="published manifest changed"):
        shared.run_cycles(args, consumers, contract, "recipe", FakeRunner(args, contract, mutate=True))
    assert shared.generation_path(args, 1).is_dir()
    assert not (shared.consumer_dir(args, consumers[0], 1) / "complete.json").exists()


def test_corrupt_checkpoint_receipt_fails_resume(study):
    args, consumers, contract = study
    shared.run_cycles(args, consumers, contract, "recipe", FakeRunner(args, contract))
    (shared.consumer_dir(args, consumers[0], 1) / "last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="completion receipt identity changed"):
        shared.run_cycles(args, consumers, contract, "recipe", FakeRunner(args, contract))


def test_missing_predecessor_and_foreign_cache_rejected(study, tmp_path):
    args, consumers, contract = study
    args.first_cycle = args.last_cycle = 2
    with pytest.raises(ValueError, match="missing completed predecessor"):
        shared.run_cycles(args, consumers, contract, "recipe", FakeRunner(args, contract))
    publish(args, contract, 2)
    manifest = shared.read_json(args.output / "cache" / "train_manifest.json")
    manifest["records"][0]["cache_file"] = "../../outside.npz"
    write_json(args.output / "cache" / "train_manifest.json", manifest)
    write_json(shared.generation_path(args, 2) / "manifest.json", manifest)
    with pytest.raises(ValueError, match="outside its owned directory"):
        shared.generation_identity(args, contract, 2)


def test_all_completed_before_pruning_interruption_can_resume_cleanup(study):
    args, consumers, contract = study
    args.keep_generations = True
    args.last_cycle = 1
    shared.run_cycles(args, consumers, contract, "recipe", FakeRunner(args, contract))
    args.keep_generations = False
    runner = FakeRunner(args, contract)
    shared.run_cycles(args, consumers, contract, "recipe", runner)
    assert not runner.commands and not shared.generation_path(args, 1).exists()


@pytest.mark.parametrize("control, flag", [
    ("no_rgb", "--no-rgb-features"), ("no_embedding", "--no-embedding-features"),
    ("reset_state", "--reset-recurrent-state"), ("no_rollout", "--rollout-probability"),
])
def test_control_commands_and_consumer_validation(study, tmp_path, control, flag):
    args, _, _ = study
    cfg = tmp_path / "consumers.json"
    write_json(cfg, [{"id": "control", "hidden_size": 128, "control": control}])
    consumer, = shared.consumers_from(cfg)
    command = shared.command_for(args, consumer, 1)
    assert flag in command and consumer["source_filter"] == "all"
    if control == "no_rollout":
        assert command[command.index(flag) + 1] == "0"
    write_json(cfg, [{"id": "../escape", "hidden_size": 64}])
    with pytest.raises(ValueError, match="safe directory"):
        shared.consumers_from(cfg)
    write_json(cfg, [{"id": "bad", "hidden_size": 64, "control": control}])
    with pytest.raises(ValueError, match="controls require"):
        shared.consumers_from(cfg)


def test_parent_requires_clean_full_mixed_source_and_matching_normalizer(study):
    args, _, _ = study
    args.split_dir.mkdir()
    for split in ("train", "validation", "test"):
        (args.split_dir / f"{split}_index.csv").write_text(split)
    cfg = {"endpoint_only": False, "init_checkpoint": None,
           "train_index_sha256": shared.sha256_file(args.split_dir / "train_index.csv"),
           "val_index_sha256": shared.sha256_file(args.split_dir / "validation_index.csv")}
    torch.save({"config": cfg}, args.checkpoint)
    normalizer = {"mean": [0.0] * 18, "std": [1.0] * 18, "source_split": "train",
                  "frozen_checkpoint_sha256": shared.sha256_file(args.checkpoint)}
    write_json(args.normalization_json, normalizer)
    assert shared.validate_parent(args)["checkpoint_sha256"] == shared.sha256_file(args.checkpoint)
    torch.save({"config": {**cfg, "endpoint_only": True}}, args.checkpoint)
    with pytest.raises(ValueError, match="unrefined parent"):
        shared.validate_parent(args)
    torch.save({"config": {**cfg, "train_index_sha256": "sleepy-only-index"}}, args.checkpoint)
    with pytest.raises(ValueError, match="full mixed training index"):
        shared.validate_parent(args)


def test_stop_reaps_child_and_does_not_start_another(tmp_path):
    runner = shared.ChildRunner(tmp_path / "STOP")
    timer = threading.Timer(0.25, runner.stop, args=(signal.SIGTERM, None))
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(shared.Stopped):
            runner.run([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "child.log")
    finally:
        timer.join()
    assert runner.process is None and time.monotonic() - started < 5
    with pytest.raises(shared.Stopped):
        runner.run([sys.executable, "-c", "raise Exception('must not start')"], tmp_path / "next.log")
    assert not (tmp_path / "next.log").exists()
