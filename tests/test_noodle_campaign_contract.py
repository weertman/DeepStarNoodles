"""A new frozen split must pass its own counts and hashes, not legacy counts."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "SCRIPTS/cache_frozen_outputs.py"
spec = importlib.util.spec_from_file_location("campaign_cache_outputs", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_versioned_counts_reject_changed_index(tmp_path):
    hashes = {}
    counts = {c: {} for c in ("all", "sleepy", "non_sleepy")}
    for split in ("train", "validation", "test"):
        index = tmp_path / f"{split}_index.csv"
        with index.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source_family", "split_group_id"])
            w.writeheader()
            w.writerows([{"source_family": "sleepy_stars", "split_group_id": split + "_sleepy"},
                         {"source_family": "stardome", "split_group_id": split + "_dome"}])
        hashes[split] = hashlib.sha256(index.read_bytes()).hexdigest()
        counts["all"][split] = 2
        counts["sleepy"][split] = counts["non_sleepy"][split] = 1
    contract = {"schema_version": 1, "dataset_version": "fixture", "cohort_counts": counts, "index_sha256": hashes}
    (tmp_path / "split_contract.json").write_text(json.dumps(contract))
    actual, digest = module.versioned_split_counts(tmp_path)
    assert actual == counts and len(digest) == 64
    with (tmp_path / "train_index.csv").open("a") as f:
        f.write("startrack_usb2,extra\n")
    with pytest.raises(ValueError, match="index differs"):
        module.versioned_split_counts(tmp_path)


def test_legacy_counts_still_apply_without_contract(tmp_path):
    assert module.versioned_split_counts(tmp_path) == (None, None)
    from deepstarnoodles.noodle_tracer_cache import validate_cohort_splits
    with pytest.raises(ValueError, match="counts changed"):
        validate_cohort_splits({s: [] for s in ("train", "validation", "test")}, "non_sleepy")
