# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Validation-study evaluation for noodle tracer context experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from deepstarnoodles.decode import activate_logits
from deepstarnoodles.decode_embed import decode_image_embed
from deepstarnoodles.evaluate_decode import (
    endpoint_errors,
    match_arms,
    orientation_correct,
)
from deepstarnoodles.noodle_tracer import (
    _INFERENCE_ATTENTION_BRANCH_ATTRIBUTE,
    NoodleTraceParams,
    NoodleTracer,
    load_noodle_tracer_checkpoint,
    trace_arms_noodle,
)
from deepstarnoodles.recurrent_tracer_data import (
    read_cache_record_full,
    sha256_file,
    validate_cache_manifest,
)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _first_divergence(decoded: np.ndarray, gt: np.ndarray, tolerance: float = 8.0) -> float:
    """Tip-origin arc distance before a decoded path first leaves GT tolerance."""
    path = np.asarray(decoded, dtype=np.float32)[::-1]
    truth = np.asarray(gt, dtype=np.float32)
    if len(path) < 2 or not len(truth):
        return 0.0
    nearest = np.sqrt(((path[:, None] - truth[None]) ** 2).sum(-1)).min(axis=1)
    bad = np.nonzero(nearest > tolerance)[0]
    stop = int(bad[0]) if len(bad) else len(path)
    if stop <= 1:
        return 0.0
    return float(np.linalg.norm(np.diff(path[:stop], axis=0), axis=1).sum())


def _image_flags(gt: Sequence[np.ndarray], activated: np.ndarray) -> dict[str, bool]:
    crossing = False
    for left in range(len(gt)):
        for right in range(left + 1, len(gt)):
            a = np.asarray(gt[left])[2:-2]
            b = np.asarray(gt[right])[2:-2]
            if len(a) and len(b):
                separation = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))
                crossing |= bool((separation <= 4.0).any())
    lengths = [
        float(np.linalg.norm(np.diff(arm, axis=0), axis=1).sum())
        for arm in gt
        if len(arm) >= 2
    ]
    support_gap = False
    for arm in gt:
        xy = np.rint(arm).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, activated.shape[2] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, activated.shape[1] - 1)
        support_gap |= bool((activated[0, xy[:, 1], xy[:, 0]] < 0.5).any())
    tips = np.stack([arm[-1] for arm in gt]) if gt else np.zeros((0, 2))
    nearby = False
    if len(tips) > 1:
        distance = np.linalg.norm(tips[:, None] - tips[None], axis=2)
        distance += np.eye(len(tips)) * 1e6
        nearby = bool((distance < 32.0).any())
    return {
        "crossing": crossing,
        "long_arm": bool(lengths and max(lengths) >= 160.0),
        "support_gap": support_gap,
        "nearby_tip": nearby,
    }


def score_decoder(decoded: list[dict], gt: Sequence[np.ndarray]) -> dict[str, Any]:
    polylines = [np.asarray(arm["polyline"], dtype=np.float32) for arm in decoded]
    matching = match_arms(polylines, list(gt), 8.0)
    chamfers = []
    base_errors = []
    tip_errors = []
    orientation = []
    divergence = []
    gt_chamfer: list[float | None] = [None] * len(gt)
    for decoded_index, gt_index, distance in matching["matches"]:
        predicted = polylines[decoded_index]
        truth = np.asarray(gt[gt_index], dtype=np.float32)
        chamfers.append(float(distance))
        gt_chamfer[gt_index] = float(distance)
        base, tip = endpoint_errors(predicted, truth)
        base_errors.append(base)
        tip_errors.append(tip)
        orientation.append(float(orientation_correct(predicted, truth)))
        divergence.append(_first_divergence(predicted, truth))
    return {
        "tp": int(matching["tp"]),
        "fp": int(matching["fp"]),
        "fn": int(matching["fn"]),
        "chamfers": chamfers,
        "base_errors": base_errors,
        "tip_errors": tip_errors,
        "orientation": orientation,
        "first_divergence_px": divergence,
        "base_arrivals": [
            float(arm.get("stop_reason") == "base") for arm in decoded
        ],
        "steps": [float(arm.get("steps", math.nan)) for arm in decoded],
        "gt_chamfer": gt_chamfer,
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    def values(name: str) -> list[float]:
        return [
            float(value)
            for row in rows
            for value in row.get(name, [])
            if np.isfinite(value)
        ]

    chamfers = values("chamfers")
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "chamfer_mean": float(np.mean(chamfers)) if chamfers else None,
        "chamfer_p90": float(np.percentile(chamfers, 90)) if chamfers else None,
        "base_err_mean": float(np.mean(values("base_errors"))) if values("base_errors") else None,
        "tip_err_mean": float(np.mean(values("tip_errors"))) if values("tip_errors") else None,
        "orientation_accuracy": float(np.mean(values("orientation"))) if values("orientation") else None,
        "first_divergence_px_mean": (
            float(np.mean(values("first_divergence_px")))
            if values("first_divergence_px")
            else None
        ),
        "base_arrival_rate": float(np.mean(values("base_arrivals"))) if values("base_arrivals") else 0.0,
        "mean_steps": float(np.mean(values("steps"))) if values("steps") else None,
    }


def paired_bootstrap(
    per_image: Sequence[dict[str, Any]],
    *,
    baseline_name: str,
    candidate_name: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in per_image:
        groups.setdefault(str(row["split_group_id"]), []).append(row)
    names = sorted(groups)
    if not names or samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    f1_delta = []
    chamfer_improvement = []
    for _ in range(samples):
        chosen = rng.choice(names, size=len(names), replace=True)
        sampled = [row for name in chosen for row in groups[str(name)]]
        baseline = aggregate([row[baseline_name] for row in sampled])
        candidate = aggregate([row[candidate_name] for row in sampled])
        f1_delta.append(candidate["f1"] - baseline["f1"])
        common_improvements = []
        for row in sampled:
            baseline_gt = row[baseline_name].get("gt_chamfer", [])
            candidate_gt = row[candidate_name].get("gt_chamfer", [])
            for baseline_value, candidate_value in zip(baseline_gt, candidate_gt):
                if baseline_value is not None and candidate_value is not None:
                    common_improvements.append(float(baseline_value) - float(candidate_value))
        if common_improvements:
            chamfer_improvement.append(float(np.mean(common_improvements)))

    def interval(values: Sequence[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "mean": float(np.mean(values)),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
        }

    return {
        "resampling_unit": "split_group_id",
        "groups": len(names),
        "samples": samples,
        "candidate_minus_baseline_f1": interval(f1_delta),
        "baseline_minus_candidate_common_arm_chamfer": interval(chamfer_improvement),
    }



ABLATION_MODES = ("none", "local_only", "context_only")
STRATA = ("crossing", "long_arm", "support_gap", "nearby_tip")
SCALAR_METRICS = (
    "precision",
    "recall",
    "f1",
    "chamfer_mean",
    "chamfer_p90",
    "base_completion_rate",
    "base_err_mean",
    "base_err_p90",
    "tip_err_mean",
    "tip_err_p90",
    "first_divergence_px_mean",
    "first_divergence_px_p90",
    "mean_steps",
    "runtime_ms_per_image",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--allow-compatible-cache-composition",
        action="store_true",
        help=(
            "Allow a tracer trained in another study only when its dense-cache "
            "contract is composition-compatible."
        ),
    )
    parser.add_argument(
        "--multiscale-ablation",
        nargs="+",
        choices=ABLATION_MODES,
        default=["none"],
        help="Evaluate the intact model and/or masked multiscale branches.",
    )
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--force-lock", action="store_true")
    parser.add_argument(
        "--lock-manifest",
        type=Path,
        help="Lock output on validation; required input on test.",
    )
    parser.add_argument(
        "--test-lock-token",
        help="Exact lock_id from the explicitly supplied validation lock manifest.",
    )
    parser.add_argument("--force-test-rerun", action="store_true")
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args(argv)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _portable_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Remove identities and filesystem locations allowed to differ by study."""
    portable = json.loads(json.dumps(contract))
    for key in (
        "contract_id",
        "created_at",
        "frozen_checkpoint",
        "frozen_checkpoint_sha256",
    ):
        portable.pop(key, None)
    for split in portable.get("splits", {}).values():
        split.pop("materialized", None)
        split.pop("source", None)
    return portable


def verify_cache_composition_compatibility(
    training_contract: dict[str, Any],
    evaluation_contract: dict[str, Any],
) -> None:
    """Require identical cache semantics except frozen-dense model identity."""
    training_dense = training_contract.get("dense_output_contract")
    evaluation_dense = evaluation_contract.get("dense_output_contract")
    if not isinstance(training_dense, dict) or not isinstance(evaluation_dense, dict):
        raise ValueError("cache composition contracts require dense_output_contract")
    required_dense_fields = (
        "channels",
        "geometry_slice",
        "embedding_slice",
        "image_size",
        "cache_schema",
        "aligned_rgb",
    )
    for field in required_dense_fields:
        if training_dense.get(field) != evaluation_dense.get(field):
            raise ValueError(
                f"incompatible cache composition dense_output_contract.{field}"
            )
    training_slices = {
        key: value for key, value in training_dense.items() if key.endswith("_slice")
    }
    evaluation_slices = {
        key: value for key, value in evaluation_dense.items() if key.endswith("_slice")
    }
    if training_slices != evaluation_slices:
        raise ValueError("incompatible cache composition dense channel slices")

    training_splits = training_contract.get("splits")
    evaluation_splits = evaluation_contract.get("splits")
    if not isinstance(training_splits, dict) or not isinstance(evaluation_splits, dict):
        raise ValueError("cache composition contracts require split definitions")
    if set(training_splits) != set(evaluation_splits):
        raise ValueError("incompatible cache composition split names")
    for split in sorted(training_splits):
        for field in ("materialized_sha256", "records"):
            if training_splits[split].get(field) != evaluation_splits[split].get(field):
                raise ValueError(
                    f"incompatible cache composition splits.{split}.{field}"
                )
    if _portable_contract(training_contract) != _portable_contract(evaluation_contract):
        raise ValueError(
            "incompatible cache composition: only study paths and frozen dense "
            "checkpoint/contract identity may differ"
        )


def resolve_cache_composition(
    *,
    evaluation_study_root: Path,
    evaluation_contract_path: Path,
    evaluation_contract: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    allow_compatible_cache_composition: bool,
) -> dict[str, Any]:
    """Resolve and validate the tracer-training contract against evaluation cache."""
    evaluation_study_root = evaluation_study_root.expanduser().resolve()
    training_root_value = (
        checkpoint_metadata.get("training_signature") or {}
    ).get("study_root")
    training_study_root = (
        evaluation_study_root
        if training_root_value is None
        else Path(training_root_value).expanduser().resolve()
    )
    is_composition = training_study_root != evaluation_study_root
    if is_composition and not allow_compatible_cache_composition:
        raise ValueError(
            "Noodle Tracer checkpoint was trained against a different study root"
        )
    training_contract_path = training_study_root / "experiment_contract.json"
    if is_composition:
        if not training_contract_path.is_file():
            raise ValueError("tracer training study has no experiment contract")
        training_contract = json.loads(
            training_contract_path.read_text(encoding="utf-8")
        )
        verify_cache_composition_compatibility(
            training_contract,
            evaluation_contract,
        )
    else:
        training_contract_path = evaluation_contract_path
        training_contract = evaluation_contract
    checkpoint_contract = checkpoint_metadata.get("contract_id")
    if (
        checkpoint_contract is not None
        and checkpoint_contract != training_contract["contract_id"]
    ):
        raise ValueError(
            "Noodle Tracer checkpoint contract id differs from its training study"
        )
    return {
        "is_compatible_cache_composition": is_composition,
        "explicitly_allowed": bool(
            is_composition and allow_compatible_cache_composition
        ),
        "training_study_root": str(training_study_root),
        "training_contract": str(training_contract_path),
        "training_contract_id": training_contract["contract_id"],
        "training_contract_sha256": sha256_file(training_contract_path),
        "training_frozen_dense_checkpoint_sha256": training_contract[
            "frozen_checkpoint_sha256"
        ],
        "evaluation_cache_study_root": str(evaluation_study_root),
        "evaluation_cache_contract": str(evaluation_contract_path),
        "evaluation_cache_contract_id": evaluation_contract["contract_id"],
        "evaluation_cache_contract_sha256": sha256_file(evaluation_contract_path),
        "evaluation_cache_frozen_dense_checkpoint_sha256": evaluation_contract[
            "frozen_checkpoint_sha256"
        ],
    }


def _finite_values(rows: Sequence[dict[str, Any]], name: str) -> list[float]:
    return [
        float(value)
        for row in rows
        for value in row.get(name, [])
        if value is not None and np.isfinite(value)
    ]


def score_trace(
    decoded: Sequence[dict[str, Any]],
    gt: Sequence[np.ndarray],
    *,
    runtime_ms: float | None = None,
) -> dict[str, Any]:
    """Apply matching metrics and retain termination diagnostics."""
    scored = score_decoder(list(decoded), gt)
    reasons = Counter(
        str(arm["stop_reason"])
        for arm in decoded
        if arm.get("stop_reason") is not None
    )
    scored["termination_counts"] = dict(sorted(reasons.items()))
    scored["runtime_ms"] = runtime_ms
    return scored


def aggregate_scores(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate image-level scores, including endpoint tails and state counts."""
    base = aggregate(rows)
    base["base_completion_rate"] = base.pop("base_arrival_rate")
    for source, output in (
        ("base_errors", "base_err_p90"),
        ("tip_errors", "tip_err_p90"),
        ("first_divergence_px", "first_divergence_px_p90"),
    ):
        values = _finite_values(rows, source)
        base[output] = float(np.percentile(values, 90)) if values else None
    runtimes = [
        float(row["runtime_ms"])
        for row in rows
        if row.get("runtime_ms") is not None and np.isfinite(row["runtime_ms"])
    ]
    base["runtime_ms_per_image"] = float(np.mean(runtimes)) if runtimes else None
    counts: Counter[str] = Counter()
    available: set[str] = set()
    for row in rows:
        row_counts = row.get("termination_counts")
        if isinstance(row_counts, dict):
            available.update(str(key) for key in row_counts)
            counts.update({str(key): int(value) for key, value in row_counts.items()})
    normalized = {
        "duplicate": counts["duplicate"] + counts["duplicates"],
        "loop": counts["loop"],
        "max_steps": counts["max_steps"] + counts["max_step"],
        "lost": counts["lost"],
    }
    for reason, count in sorted(counts.items()):
        normalized.setdefault(reason, count)
    base["termination_counts"] = normalized
    base["observed_termination_reasons"] = sorted(available)
    base["termination_count_scope"] = "emitted traces only"
    return base


def report_method(
    per_image: Sequence[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    scores = [row[method] for row in per_image]
    aggregate = aggregate_scores(scores)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in per_image:
        groups.setdefault(str(row["split_group_id"]), []).append(row[method])
    group_reports = [aggregate_scores(groups[name]) for name in sorted(groups)]
    macro: dict[str, Any] = {"groups": len(group_reports)}
    for metric in SCALAR_METRICS:
        values = [
            float(report[metric])
            for report in group_reports
            if report.get(metric) is not None and np.isfinite(report[metric])
        ]
        macro[metric] = float(np.mean(values)) if values else None
    return {"aggregate": aggregate, "group_macro": macro}


def stratified_reports(
    per_image: Sequence[dict[str, Any]],
    methods: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for flag in STRATA:
        subset = [row for row in per_image if row["flags"].get(flag, False)]
        result[flag] = {
            "images": len(subset),
            "split_groups": len({str(row["split_group_id"]) for row in subset}),
            "reports": {method: report_method(subset, method) for method in methods},
        }
    return result


@contextmanager
def multiscale_branch_ablation(
    model: NoodleTracer,
    mode: str,
) -> Iterator[None]:
    """Exclude one attention-token branch without changing persistent model state."""
    if mode not in ABLATION_MODES:
        raise ValueError(f"unsupported multiscale ablation: {mode}")
    if mode == "none":
        yield
        return
    if model.context_mode != "multiscale_dilated":
        raise ValueError(f"{mode} requires a multiscale_dilated checkpoint")
    if model.training:
        raise RuntimeError("multiscale branch ablation is inference-only")
    branch = 0 if mode == "local_only" else 1
    missing = object()
    previous = getattr(model, _INFERENCE_ATTENTION_BRANCH_ATTRIBUTE, missing)
    setattr(model, _INFERENCE_ATTENTION_BRANCH_ATTRIBUTE, branch)
    try:
        yield
    finally:
        if previous is missing:
            delattr(model, _INFERENCE_ATTENTION_BRANCH_ATTRIBUTE)
        else:
            setattr(model, _INFERENCE_ATTENTION_BRANCH_ATTRIBUTE, previous)


def require_test_lock(
    *,
    split: str,
    lock_manifest: Path | None,
    supplied_token: str | None,
    expected_payload: dict[str, Any],
    access_path: Path,
    force_test_rerun: bool,
) -> dict[str, Any] | None:
    """Guard held-out cache access behind an explicit matching validation lock."""
    if split != "test":
        return None
    if lock_manifest is None or supplied_token is None:
        raise RuntimeError(
            "test requires explicit --lock-manifest and --test-lock-token from validation"
        )
    path = lock_manifest.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("validation lock manifest does not exist")
    locked = json.loads(path.read_text(encoding="utf-8"))
    if supplied_token != locked.get("lock_id"):
        raise ValueError("test lock token does not match the validation lock manifest")
    if locked.get("lock_id") != expected_payload.get("lock_id"):
        raise ValueError("test checkpoint/parameters differ from locked validation configuration")
    if access_path.is_file() and not force_test_rerun:
        raise RuntimeError("held-out test was already accessed; pass --force-test-rerun explicitly")
    return locked


def _locked_payload(
    *,
    checkpoint: Path,
    contract_id: str,
    cache_composition: dict[str, Any],
    model: NoodleTracer,
    trace_params: NoodleTraceParams,
    ablations: Sequence[str],
    validation_manifest: Path,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "evaluator": "noodle_tracer",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "contract_id": contract_id,
        "cache_composition": cache_composition,
        "model_config": model.checkpoint_config(),
        "trace_params": asdict(trace_params),
        "multiscale_ablation": list(ablations),
        "validation_manifest": str(validation_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "decoder_source_sha256": sha256_file(
            Path(__file__).with_name("noodle_tracer.py")
        ),
        "evaluator_source_sha256": sha256_file(Path(__file__)),
    }
    payload["lock_id"] = canonical_hash(payload)
    return payload


def _synchronize(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _timed(device: str, function: Any) -> tuple[Any, float]:
    _synchronize(device)
    started = time.perf_counter()
    result = function()
    _synchronize(device)
    return result, 1000.0 * (time.perf_counter() - started)


def _unique_modes(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study_root = args.study_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if args.lock and args.split != "validation":
        raise ValueError("--lock is only valid for validation")
    if args.lock and args.limit is not None:
        raise ValueError("cannot lock a limited validation run")
    ablations = _unique_modes(args.multiscale_ablation)
    contract_path = study_root / "experiment_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validation_manifest_path = study_root / "cache" / "validation_manifest.json"
    validation_manifest = validate_cache_manifest(
        validation_manifest_path,
        contract,
        split="validation",
    )
    model, trace_params, metadata, _ = load_noodle_tracer_checkpoint(
        str(checkpoint),
        device=args.device,
    )
    cache_composition = resolve_cache_composition(
        evaluation_study_root=study_root,
        evaluation_contract_path=contract_path,
        evaluation_contract=contract,
        checkpoint_metadata=metadata,
        allow_compatible_cache_composition=args.allow_compatible_cache_composition,
    )
    if model.context_mode != "multiscale_dilated" and any(
        mode != "none" for mode in ablations
    ):
        raise ValueError("branch ablations require a multiscale_dilated checkpoint")
    lock_payload = _locked_payload(
        checkpoint=checkpoint,
        contract_id=contract["contract_id"],
        cache_composition=cache_composition,
        model=model,
        trace_params=trace_params,
        ablations=ablations,
        validation_manifest=validation_manifest_path,
    )
    access_path = study_root / "test_access_noodle_tracer.json"
    require_test_lock(
        split=args.split,
        lock_manifest=args.lock_manifest,
        supplied_token=args.test_lock_token,
        expected_payload=lock_payload,
        access_path=access_path,
        force_test_rerun=args.force_test_rerun,
    )

    manifest_path = study_root / "cache" / f"{args.split}_manifest.json"
    manifest = (
        validation_manifest
        if args.split == "validation"
        else validate_cache_manifest(manifest_path, contract, split="test")
    )
    records = list(manifest["records"])
    if args.limit is not None:
        records = records[: args.limit]
    method_for_mode = {
        mode: "noodle_tracer" if mode == "none" else f"noodle_tracer_{mode}"
        for mode in ablations
    }
    methods = ["embedding_dijkstra", *method_for_mode.values()]
    per_image: list[dict[str, Any]] = []
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    for record in records:
        raw, image, gt = read_cache_record_full(
            manifest_path.parent / record["cache_file"],
            expected_identity=record.get("cache_identity"),
        )
        raw_np = raw.numpy()
        activated = activate_logits(raw[:10]).numpy()
        embedding, runtime_ms = _timed(
            args.device,
            lambda: decode_image_embed(activated, raw_np[10:18])["arms"],
        )
        row: dict[str, Any] = {
            "row_index": record["row_index"],
            "split_group_id": record["split_group_id"],
            "flags": _image_flags(gt, activated),
            "embedding_dijkstra": score_trace(
                embedding,
                gt,
                runtime_ms=runtime_ms,
            ),
        }
        for mode in ablations:
            with multiscale_branch_ablation(model, mode):
                decoded, runtime_ms = _timed(
                    args.device,
                    lambda: trace_arms_noodle(
                        raw,
                        image,
                        model,
                        trace_params,
                    ),
                )
            row[method_for_mode[mode]] = score_trace(
                decoded,
                gt,
                runtime_ms=runtime_ms,
            )
        per_image.append(row)

    reports = {method: report_method(per_image, method) for method in methods}
    bootstrap = {
        method: paired_bootstrap(
            per_image,
            baseline_name="embedding_dijkstra",
            candidate_name=method,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for method in method_for_mode.values()
    }
    decisions: dict[str, dict[str, bool]] = {}
    for method, comparison in bootstrap.items():
        f1_interval = comparison.get("candidate_minus_baseline_f1") or {}
        chamfer_interval = (
            comparison.get("baseline_minus_candidate_common_arm_chamfer") or {}
        )
        decision = {
            "path_quality_ci_excludes_zero": (
                chamfer_interval.get("ci95_low", -math.inf) > 0
            ),
            "no_material_f1_reduction": (
                f1_interval.get("ci95_low", -math.inf) >= -0.01
            ),
        }
        decision["supported"] = all(decision.values())
        decisions[method] = decision
    primary_method = method_for_mode.get("none", next(iter(method_for_mode.values())))
    result = {
        "schema_version": 1,
        "split": args.split,
        "images": len(records),
        "complete_split": args.limit is None,
        "contract_id": contract["contract_id"],
        "provenance": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "study_root": str(study_root),
            "experiment_contract": str(contract_path),
            "experiment_contract_sha256": sha256_file(contract_path),
            "cache_manifest": str(manifest_path),
            "cache_manifest_sha256": sha256_file(manifest_path),
            "cache_provenance": manifest["provenance"],
            "cache_composition": cache_composition,
            "checkpoint_metadata": metadata,
            "device": str(args.device),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "limit": args.limit,
            "multiscale_ablation": ablations,
            "termination_counts_note": (
                "Counts cover emitted traces; filtered duplicate/lost candidates are "
                "unavailable from the tracer API."
            ),
        },
        "locked_configuration": lock_payload,
        "reports": reports,
        "stratified": stratified_reports(per_image, methods),
        "paired_comparison_to_embedding_dijkstra": bootstrap,
        "decision_gates": decisions,
        "primary_decision_gate": decisions[primary_method],
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(args.device)))
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else None
        ),
        "per_image": per_image,
    }
    lock_written = False
    if args.split == "validation" and args.lock:
        if decisions[primary_method]["supported"] or args.force_lock:
            lock_path = (
                args.lock_manifest.expanduser().resolve()
                if args.lock_manifest is not None
                else study_root / "locked_noodle_tracer_configuration.json"
            )
            atomic_json(lock_path, lock_payload)
            result["validation_lock_manifest"] = str(lock_path)
            lock_written = True
    result["validation_lock_written"] = lock_written
    if args.split == "test":
        atomic_json(
            access_path,
            {
                "accessed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "lock_id": lock_payload["lock_id"],
                "checkpoint_sha256": lock_payload["checkpoint_sha256"],
                "cache_manifest_sha256": sha256_file(manifest_path),
                "images": len(records),
                "complete_test": args.limit is None,
            },
        )
    output = args.out_json or (
        study_root / "evaluation" / f"{args.split}_noodle_tracer_comparison.json"
    )
    atomic_json(output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_image"},
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
