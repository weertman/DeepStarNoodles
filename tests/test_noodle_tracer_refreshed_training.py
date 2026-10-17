"""End-to-end trainer behaviour used by refreshed training, on a synthetic study."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from deepstarnoodles.noodle_tracer_cache import FROZEN_CACHE_SCHEMA, write_cache_record
import train_noodle_tracer

SIZE = 48


def straight_star(offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.full((18, SIZE, SIZE), -10.0, dtype=np.float32)
    y = SIZE // 2 + offset
    raw[0, y - 2:y + 3, 6:SIZE - 6] = 10.0
    raw[1, y, 6:SIZE - 6] = 10.0
    raw[2] = 1.0
    raw[8, y, 7] = 10.0
    raw[9, y, SIZE - 8] = 10.0
    raw[10:] = 0.1
    image = np.full((SIZE, SIZE, 3), 40, dtype=np.uint8)
    image[y - 2:y + 3, 6:SIZE - 6] = 200
    arm = np.asarray([[7.0, y], [20.0, y], [SIZE - 8.0, y]], dtype=np.float32)
    return raw, image, arm


def write_study(root: Path, counts: dict[str, int]) -> None:
    contract = {
        "contract_id": "synthetic-contract",
        "frozen_checkpoint_sha256": "synthetic-hydra",
        "cohort": "all",
        "splits": {split: {"records": n, "materialized_sha256": f"{split}-index"}
                   for split, n in counts.items()},
    }
    (root / "cache").mkdir(parents=True)
    (root / "experiment_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    for split, n in counts.items():
        records = []
        for index in range(n):
            raw, image, arm = straight_star(offset=index - n // 2)
            name = f"{split}/{index:03d}.npz"
            write_cache_record(root / "cache" / name, raw, image, [arm], cache_identity=f"{split}{index}")
            records.append({"cache_file": name, "cache_identity": f"{split}{index}",
                            "image_sha256": "i", "shapes_sha256": "s",
                            "split_group_id": f"group-{split}-{index % 2}"})
        manifest = {
            "schema_version": FROZEN_CACHE_SCHEMA,
            "split": split,
            "records": records,
            "provenance": {
                "contract_id": contract["contract_id"],
                "frozen_checkpoint_sha256": contract["frozen_checkpoint_sha256"],
                "index_sha256": f"{split}-index",
                "channel_statistics": {"source_split": "train", "mean": [0.0] * 18, "std": [1.0] * 18},
            },
        }
        (root / "cache" / f"{split}_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture()
def study(tmp_path: Path) -> dict[str, Path]:
    train = tmp_path / "train_study"
    validation = tmp_path / "validation_study"
    write_study(train, {"train": 3, "validation": 2})
    write_study(validation, {"train": 3, "validation": 2})
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({
        "mean": [float(i) / 10.0 for i in range(18)],
        "std": [1.0 + float(i) / 100.0 for i in range(18)],
    }), encoding="utf-8")
    return {"train": train, "validation": validation, "normalization": normalization, "root": tmp_path}


def run_trainer(monkeypatch, study: dict[str, Path], out: str, epochs: int, *extra: str) -> Path:
    out_dir = study["root"] / out
    argv = [
        "train_noodle_tracer.py",
        "--study-root", str(study["train"]),
        "--validation-study-root", str(study["validation"]),
        "--out-dir", str(out_dir),
        "--device", "cpu", "--num-workers", "0", "--seed", "7",
        "--epochs", str(epochs), "--lr-decay-epochs", "3", "--steps-per-epoch", "5",
        "--hidden-size", "16", "--max-policy-steps", "8",
        "--normalization-json", str(study["normalization"]),
        "--no-pretrained-rgb", "--no-augment", "--no-amp",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert train_noodle_tracer.main() == 0
    return out_dir


def last_row(out_dir: Path) -> dict[str, str]:
    with (out_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))[-1]


def test_fixed_update_budget_normalization_and_separate_validation(monkeypatch, study):
    out_dir = run_trainer(monkeypatch, study, "cycle1", 1)
    row = last_row(out_dir)
    # Three training images, five updates: the loader cycles until the budget is met.
    assert float(row["train_optimizer_steps"]) == 5.0
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert config["validation_study_root"] == str(study["validation"].resolve())
    normalization = json.loads(study["normalization"].read_text(encoding="utf-8"))
    state = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert torch.equal(state["dense_mean"].flatten(), torch.tensor(normalization["mean"]))
    assert torch.equal(state["dense_std"].flatten(), torch.tensor(normalization["std"]))


def test_resume_guards_schedule_and_records_explicit_override(monkeypatch, study):
    first = run_trainer(monkeypatch, study, "cycle1", 1)
    resume = ("--resume", str(first / "last.pt"))
    with pytest.raises(ValueError, match="resume training signature differs"):
        run_trainer(monkeypatch, study, "cycle2_refused", 2, "--lr", "1e-4", *resume)
    second = run_trainer(monkeypatch, study, "cycle2", 2, "--lr", "1e-4", "--min-lr", "1e-4",
                         "--override-schedule-on-resume", *resume)
    metadata = torch.load(second / "last.pt", map_location="cpu", weights_only=False)["metadata"]
    assert metadata["epoch"] == 2
    (override,) = metadata["schedule_override"]
    assert override["after_epoch"] == 1
    assert set(override["changed"]) == {"lr", "min_lr"}
    assert float(last_row(second)["lr"]) == pytest.approx(1e-4)


def test_relocated_resume_requires_identical_normalization(monkeypatch, study):
    first = run_trainer(monkeypatch, study, "cycle1", 1)
    moved = study["root"] / "moved_train_study"
    shutil.copytree(study["train"], moved)
    relocated = {**study, "train": moved}
    resume = ("--resume", str(first / "last.pt"))
    with pytest.raises(ValueError, match="resume training signature differs"):
        run_trainer(monkeypatch, relocated, "cycle2_refused", 2, *resume)
    second = run_trainer(monkeypatch, relocated, "cycle2", 2, "--relocate-resume-paths", *resume)
    metadata = torch.load(second / "last.pt", map_location="cpu", weights_only=False)["metadata"]
    assert set(metadata["path_relocation"][0]["paths"]) == {"study_root"}
    other = study["root"] / "other_normalization.json"
    other.write_text(json.dumps({"mean": [0.0] * 18, "std": [1.0] * 18}), encoding="utf-8")
    with pytest.raises(ValueError, match="relocated normalization values differ"):
        run_trainer(monkeypatch, {**relocated, "normalization": other}, "cycle2_bad", 2,
                    "--relocate-resume-paths", *resume)


def test_split_evaluator_scores_compares_and_gates_test(monkeypatch, study, tmp_path):
    import evaluate_tracer_split

    checkpoint = run_trainer(monkeypatch, study, "cycle1", 1) / "last.pt"
    out_json = tmp_path / "evaluation.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_tracer_split.py", "--split-root", str(study["validation"]),
        "--checkpoint", str(checkpoint), "--checkpoint", str(checkpoint),
        "--device", "cpu", "--bootstrap-samples", "50", "--out-json", str(out_json),
    ])
    assert evaluate_tracer_split.main() == 0
    result = json.loads(out_json.read_text(encoding="utf-8"))
    first, second = result["checkpoints"]
    assert first["images"] == 2 and first["f1"] == second["f1"]
    (contrast,) = result["contrasts_vs_first"]
    assert contrast["difference"] == 0.0 and contrast["groups"] == 2
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"test_access_locked": False}), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not authorize"):
        evaluate_tracer_split.check_lock(lock, [checkpoint])
    lock.write_text(json.dumps({"test_access_locked": True,
                                "chosen": {"checkpoint_sha256": "0" * 64}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a checkpoint recorded"):
        evaluate_tracer_split.check_lock(lock, [checkpoint])
    digest = evaluate_tracer_split.sha256_file(checkpoint)
    lock.write_text(json.dumps({"test_access_locked": True, "frozen_checkpoint_sha256": "x",
                                "chosen": {"checkpoint_sha256": digest}}), encoding="utf-8")
    assert evaluate_tracer_split.check_lock(lock, [checkpoint])["test_access_locked"] is True
