#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Train the spatial, flexible-action Noodle Tracer."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / 'CORE'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from deepstarnoodles.evaluate_decode import match_arms
from deepstarnoodles.noodle_tracer_cache import (
    FrozenOutputDataset,
    OutputAugment,
    channel_statistics,
    collate_frozen_outputs,
    validate_cache_manifest,
)
from deepstarnoodles.noodle_tracer import (
    NoodleTraceParams,
    NoodleTracer,
    TraceState,
    extract_centered_patches,
    load_noodle_tracer_checkpoint,
    save_noodle_tracer_checkpoint,
    trace_arms_noodle,
)
from deepstarnoodles.noodle_tracer_data import (
    NoodleTeacherTrajectories,
    build_noodle_teacher_trajectories,
    projection_lookahead_targets,
)


DEFAULT_STUDY = REPO / "RUNS/noodle_tracer_sleepy18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=24,
        help="Fixed LR schedule horizon, independent of the run stopping epoch.",
    )
    parser.add_argument("--lookahead-px", type=float, default=8.0)
    parser.add_argument("--min-step-px", type=float, default=0.5)
    parser.add_argument("--max-step-px", type=float, default=6.0)
    parser.add_argument("--terminate-radius-px", type=float, default=4.0)
    parser.add_argument("--lost-radius-px", type=float, default=16.0)
    parser.add_argument("--tip-match-radius", type=float, default=12.0)
    parser.add_argument("--ambiguous-negative-radius", type=float, default=20.0)
    parser.add_argument("--position-noise", type=float, default=2.0)
    parser.add_argument("--lost-example-p", type=float, default=0.10)
    parser.add_argument("--rollout-probability", type=float, default=0.30)
    parser.add_argument("--rollout-ramp-epochs", type=int, default=12)
    parser.add_argument("--rollout-chunk-steps", type=int, default=16)
    parser.add_argument("--max-policy-steps", type=int, default=96)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--support-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-weight", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dense-patch-size", type=int, default=33)
    parser.add_argument(
        "--context-mode",
        choices=("single", "multiscale_dilated"),
        default="single",
    )
    parser.add_argument("--context-dilation", type=int, default=2)
    parser.add_argument("--pretrained-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rgb-unfreeze-scope",
        choices=("coarse", "all"),
        default="coarse",
    )
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Warm-start model weights without restoring optimizer/RNG state.",
    )
    parser.add_argument("--rgb-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-val-images", type=int)
    return parser.parse_args()


def training_signature(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"epochs", "out_dir", "resume", "device", "init_checkpoint"}
    return {
        key: value
        for key, value in vars(args).items()
        if key not in excluded
    }


def capture_rng_state(
    loader_generator: torch.Generator,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, Any]:
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": loader_generator.get_state(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }


def restore_rng_state(
    state: dict[str, Any],
    loader_generator: torch.Generator,
    scaler: torch.amp.GradScaler | None,
) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    loader_generator.set_state(state["loader_generator"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])


def rollout_probability(epoch: int, maximum: float, ramp_epochs: int) -> float:
    fraction = min(max(epoch - 1, 0) / max(ramp_epochs - 1, 1), 1.0)
    return float(maximum) * fraction


def learning_rate_scale(
    epoch: int,
    *,
    epochs: int,
    warmup_epochs: int,
    minimum_ratio: float,
) -> float:
    if epoch <= warmup_epochs:
        return max(epoch / max(warmup_epochs, 1), 1e-3)
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _move_teachers(
    teachers: NoodleTeacherTrajectories,
    device: torch.device,
) -> NoodleTeacherTrajectories:
    return NoodleTeacherTrajectories(
        image_indices=teachers.image_indices.to(device),
        seed_positions=teachers.seed_positions.to(device),
        seed_embeddings=teachers.seed_embeddings.to(device),
        paths=teachers.paths.to(device),
        path_valid=teachers.path_valid.to(device),
        valid_seed=teachers.valid_seed.to(device),
        matched_arms=teachers.matched_arms,
        false_tips=teachers.false_tips,
        ignored_tips=teachers.ignored_tips,
    )


def _initial_training_positions(
    teachers: NoodleTeacherTrajectories,
    *,
    position_noise: float,
    lost_example_p: float,
) -> torch.Tensor:
    positions = teachers.seed_positions.clone()
    if position_noise > 0:
        positions[teachers.valid_seed] += (
            torch.randn_like(positions[teachers.valid_seed]) * position_noise
        )
    if lost_example_p > 0 and bool(teachers.valid_seed.any()):
        selected = (
            torch.rand(len(positions), device=positions.device) < lost_example_p
        ) & teachers.valid_seed
        if bool(selected.any()):
            angle = torch.rand(int(selected.sum()), device=positions.device) * (2 * math.pi)
            radius = 20.0 + torch.rand_like(angle) * 8.0
            positions[selected] += torch.stack(
                (torch.cos(angle) * radius, torch.sin(angle) * radius),
                dim=1,
            )
    return positions


def train_or_score_batch(
    model: NoodleTracer,
    batch: dict[str, Any],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
    args: argparse.Namespace,
    rollout_p: float,
) -> dict[str, float]:
    training = optimizer is not None
    raw = batch["raw"][:, :model.dense_channels].to(device, non_blocking=True)
    image = batch["image"].to(device, non_blocking=True)
    teachers = build_noodle_teacher_trajectories(
        batch["raw"],
        batch["arms"],
        embedding_dim=model.embedding_dim,
        match_radius=args.tip_match_radius,
        ambiguous_negative_radius=args.ambiguous_negative_radius,
    )
    if len(teachers.seed_positions) == 0:
        return {
            "loss": 0.0,
            "action_loss": 0.0,
            "state_loss": 0.0,
            "support_loss": 0.0,
            "endpoint_loss": 0.0,
            "states": 0.0,
            "matched_arms": 0.0,
            "false_tips": 0.0,
            "ignored_tips": float(teachers.ignored_tips),
        }
    teachers = _move_teachers(teachers, device)
    count = len(teachers.seed_positions)
    positions = _initial_training_positions(
        teachers,
        position_noise=args.position_noise if training else 0.0,
        lost_example_p=args.lost_example_p if training else 0.0,
    )
    hidden = model.initial_state(count, device=device)
    previous = torch.zeros((count, 2), device=device)
    active = torch.ones(count, dtype=torch.bool, device=device)
    trajectory_indices = torch.arange(count, device=device)
    pyramid = model.encode_rgb(image)
    action_sum = raw.new_zeros(())
    state_sum = raw.new_zeros(())
    support_sum = raw.new_zeros(())
    endpoint_sum = raw.new_zeros(())
    action_count = 0
    state_count = 0
    support_count = 0
    endpoint_count = 0
    class_weight = raw.new_tensor([2.0, 1.0, 2.0, 2.0])
    rollout_mask = torch.zeros(count, dtype=torch.bool, device=device)
    amp_enabled = bool(amp and device.type == "cuda")

    for step in range(args.max_policy_steps):
        live = torch.nonzero(active, as_tuple=False).flatten()
        if not len(live):
            break
        if training and step % args.rollout_chunk_steps == 0:
            rollout_mask[live] = torch.rand(len(live), device=device) < rollout_p
        centres = positions[live]
        targets = projection_lookahead_targets(
            centres,
            teachers,
            trajectory_indices[live],
            lookahead_px=args.lookahead_px,
            max_step_px=args.max_step_px,
            terminate_radius_px=args.terminate_radius_px,
            lost_radius_px=args.lost_radius_px,
        )
        dense_patches = model.sample_dense(
            raw,
            centres,
            teachers.image_indices[live],
        )
        rgb_fine, rgb_coarse = model.sample_rgb(
            pyramid,
            centres,
            teachers.image_indices[live],
        )
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            displacement, state_logits, next_hidden = model.forward_step(
                dense_patches,
                rgb_fine,
                rgb_coarse,
                previous[live],
                teachers.seed_embeddings[live],
                hidden[live],
            )
            state_sum = state_sum + F.cross_entropy(
                state_logits,
                targets.state,
                weight=class_weight,
                reduction="sum",
            )
            state_count += len(live)
            continuing = targets.state == int(TraceState.CONTINUE)
            if bool(continuing.any()):
                predicted = displacement[continuing]
                expected = targets.displacement[continuing]
                action_sum = action_sum + F.smooth_l1_loss(
                    predicted,
                    expected,
                    reduction="sum",
                    beta=1.0,
                )
                predicted_direction = F.normalize(predicted, dim=1, eps=1e-6)
                expected_direction = F.normalize(expected, dim=1, eps=1e-6)
                action_sum = action_sum + (
                    1.0 - (predicted_direction * expected_direction).sum(dim=1)
                ).sum()
                action_count += int(continuing.sum())
                predicted_next = centres[continuing] + predicted
                support = torch.sigmoid(raw[:, 0])
                support_values = extract_centered_patches(
                    support[:, None],
                    predicted_next,
                    teachers.image_indices[live][continuing],
                    1,
                )[:, 0, 0, 0]
                support_sum = support_sum + (
                    -torch.log(support_values.clamp_min(1e-5))
                ).sum()
                support_count += int(continuing.sum())
            terminating_base = targets.state == int(TraceState.TERMINATE_BASE)
            if bool(terminating_base.any()):
                trajectory_ids = trajectory_indices[live][terminating_base]
                base_targets = torch.stack(
                    [
                        teachers.paths[index, teachers.path_valid[index]][-1]
                        for index in trajectory_ids
                    ]
                )
                endpoint_sum = endpoint_sum + F.smooth_l1_loss(
                    centres[terminating_base],
                    base_targets,
                    reduction="sum",
                    beta=1.0,
                )
                endpoint_count += int(terminating_base.sum())
        next_hidden_all = hidden.clone()
        next_hidden_all[live] = next_hidden.float()
        hidden = next_hidden_all
        next_positions = centres + targets.displacement
        if training:
            use_rollout = rollout_mask[live] & continuing
            next_positions[use_rollout] = (
                centres[use_rollout] + displacement[use_rollout].float()
            )
        positions_all = positions.clone()
        positions_all[live] = next_positions
        positions = positions_all
        previous_all = previous.clone()
        previous_all[live] = torch.where(
            (training and rollout_mask[live]).unsqueeze(1)
            if training
            else torch.zeros((len(live), 1), dtype=torch.bool, device=device),
            displacement.float(),
            targets.displacement,
        )
        previous = previous_all
        finished = targets.state != int(TraceState.CONTINUE)
        active[live[finished]] = False

    action_loss = action_sum / max(action_count * 2, 1)
    state_loss = state_sum / max(state_count, 1)
    support_loss = support_sum / max(support_count, 1)
    endpoint_loss = endpoint_sum / max(endpoint_count * 2, 1)
    loss = (
        args.action_weight * action_loss
        + args.state_weight * state_loss
        + args.support_weight * support_loss
        + args.endpoint_weight * endpoint_loss
    )
    if training:
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return {
        "loss": float(loss.detach()),
        "action_loss": float(action_loss.detach()),
        "state_loss": float(state_loss.detach()),
        "support_loss": float(support_loss.detach()),
        "endpoint_loss": float(endpoint_loss.detach()),
        "states": float(state_count),
        "matched_arms": float(teachers.matched_arms),
        "false_tips": float(teachers.false_tips),
        "ignored_tips": float(teachers.ignored_tips),
    }


def run_loader(
    model: NoodleTracer,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    args: argparse.Namespace,
    rollout_p: float,
) -> dict[str, float]:
    model.train(optimizer is not None)
    sums: dict[str, float] = {}
    count = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            result = train_or_score_batch(
                model,
                batch,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp=args.amp,
                args=args,
                rollout_p=rollout_p if optimizer is not None else 0.0,
            )
            for key, value in result.items():
                sums[key] = sums.get(key, 0.0) + value
            count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


@torch.no_grad()
def free_rollout_metrics(
    model: NoodleTracer,
    dataset: FrozenOutputDataset,
    *,
    params: NoodleTraceParams,
    limit: int | None,
) -> dict[str, float]:
    count = len(dataset) if limit is None else min(limit, len(dataset))
    totals = {"tp": 0, "fp": 0, "fn": 0}
    chamfers: list[float] = []
    base_completion: list[bool] = []
    groups: dict[str, dict[str, int]] = {}
    for index in range(count):
        item = dataset[index]
        arms = trace_arms_noodle(
            item["raw"],
            item["image"],
            model,
            params,
        )
        decoded = [arm["polyline"] for arm in arms]
        gt = [np.asarray(arm, dtype=np.float32) for arm in item["arms"]]
        matched = match_arms(decoded, gt, 8.0)
        for key in totals:
            totals[key] += int(matched[key])
        chamfers.extend(float(value[2]) for value in matched["matches"])
        base_completion.extend(arm.get("stop_reason") == "base" for arm in arms)
        group_id = str(item["record"].get("split_group_id", "unknown"))
        group = groups.setdefault(group_id, {"tp": 0, "fp": 0, "fn": 0})
        for key in group:
            group[key] += int(matched[key])
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    group_f1 = []
    for group in groups.values():
        gp = group["tp"] / max(group["tp"] + group["fp"], 1)
        gr = group["tp"] / max(group["tp"] + group["fn"], 1)
        group_f1.append(2 * gp * gr / max(gp + gr, 1e-9))
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "group_macro_f1": float(np.mean(group_f1)) if group_f1 else 0.0,
        "tp": float(totals["tp"]),
        "fp": float(totals["fp"]),
        "fn": float(totals["fn"]),
        "chamfer_mean": float(np.mean(chamfers)) if chamfers else math.nan,
        "base_completion_rate": (
            float(np.mean(base_completion)) if base_completion else 0.0
        ),
        "images": float(count),
    }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    args.study_root = args.study_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    contract = json.loads(
        (args.study_root / "experiment_contract.json").read_text(encoding="utf-8")
    )
    train_manifest = args.study_root / "cache/train_manifest.json"
    val_manifest = args.study_root / "cache/validation_manifest.json"
    validate_cache_manifest(train_manifest, contract, split="train")
    validate_cache_manifest(val_manifest, contract, split="validation")
    mean, std = channel_statistics(train_manifest)
    mean, std = mean[:18], std[:18]
    train_dataset = FrozenOutputDataset(
        train_manifest,
        augment=OutputAugment(
            enabled=args.augment,
            # v1 zero-logit dropout created artificial 0.5 tip plateaus.
            dropout_p=0.0,
        ),
    )
    validation_dataset = FrozenOutputDataset(
        val_manifest,
        augment=OutputAugment(enabled=False),
    )
    if args.max_train_images is not None:
        train_dataset.records = train_dataset.records[:args.max_train_images]
    if args.max_val_images is not None:
        validation_dataset.records = validation_dataset.records[:args.max_val_images]
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_frozen_outputs,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_frozen_outputs,
        pin_memory=device.type == "cuda",
    )
    trace_params = NoodleTraceParams(
        dense_patch_size=args.dense_patch_size,
        context_mode=args.context_mode,
        context_dilation=args.context_dilation,
        min_step_px=args.min_step_px,
        max_step_px=args.max_step_px,
    )
    start_epoch = 1
    best_smoothed_f1 = -math.inf
    recent_f1: list[float] = []
    restored_state = None
    if args.resume is not None:
        model, trace_params, metadata, restored_state = load_noodle_tracer_checkpoint(
            str(args.resume.expanduser().resolve()),
            device=device,
        )
        expected_signature = metadata.get("training_signature")
        if expected_signature != training_signature(args):
            raise ValueError("resume training signature differs from current arguments")
        start_epoch = int(metadata["epoch"]) + 1
        best_smoothed_f1 = float(metadata.get("best_smoothed_f1", -math.inf))
        recent_f1 = list(metadata.get("recent_group_macro_f1", []))
    elif args.init_checkpoint is not None:
        model, trace_params, _, _ = load_noodle_tracer_checkpoint(
            str(args.init_checkpoint.expanduser().resolve()),
            device=device,
        )
        requested_geometry = (
            args.dense_patch_size,
            args.context_mode,
            args.context_dilation,
        )
        loaded_geometry = (
            model.dense_patch_size,
            model.context_mode,
            model.context_dilation,
        )
        if requested_geometry != loaded_geometry:
            raise ValueError(
                "init checkpoint geometry differs from --dense-patch-size/"
                "--context-mode/--context-dilation"
            )
        model.rgb_encoder.frozen = bool(args.freeze_rgb)
        for parameter in model.rgb_encoder.parameters():
            parameter.requires_grad_(not args.freeze_rgb)
        if not args.freeze_rgb and args.rgb_unfreeze_scope == "coarse":
            for parameter in model.rgb_encoder.fine.parameters():
                parameter.requires_grad_(False)
    else:
        model = NoodleTracer(
            dense_channels=18,
            embedding_dim=8,
            dense_patch_size=args.dense_patch_size,
            context_mode=args.context_mode,
            context_dilation=args.context_dilation,
            hidden_size=args.hidden_size,
            pretrained_rgb=args.pretrained_rgb,
            freeze_rgb=args.freeze_rgb,
            dense_mean=mean,
            dense_std=std,
            min_step_px=args.min_step_px,
            max_step_px=args.max_step_px,
        ).to(device)
    if not args.freeze_rgb and args.rgb_unfreeze_scope == "coarse":
        model.rgb_encoder.frozen = False
        for parameter in model.rgb_encoder.fine.parameters():
            parameter.requires_grad_(False)
        for parameter in model.rgb_encoder.coarse.parameters():
            parameter.requires_grad_(True)
    rgb_parameters = [
        parameter
        for parameter in model.rgb_encoder.parameters()
        if parameter.requires_grad
    ]
    rgb_ids = {id(parameter) for parameter in rgb_parameters}
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in rgb_ids
    ]
    parameter_groups = [{"params": main_parameters, "lr_multiplier": 1.0}]
    if rgb_parameters:
        parameter_groups.append(
            {
                "params": rgb_parameters,
                "lr_multiplier": args.rgb_lr_multiplier,
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    if restored_state is not None:
        optimizer.load_state_dict(restored_state["optimizer"])
        restore_rng_state(restored_state["rng"], loader_generator, scaler)
    _atomic_json(
        args.out_dir / "config.json",
        {
            **vars(args),
            "trace_params": asdict(trace_params),
            "training_signature": training_signature(args),
        },
    )
    metric_path = args.out_dir / "metrics.csv"
    fieldnames = [
        "epoch",
        "lr",
        "rollout_probability",
        "train_loss",
        "train_action_loss",
        "train_state_loss",
        "train_support_loss",
        "train_endpoint_loss",
        "validation_teacher_loss",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_group_macro_f1",
        "validation_chamfer_mean",
        "validation_base_completion_rate",
    ]
    with metric_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
    for epoch in range(start_epoch, args.epochs + 1):
        scale = learning_rate_scale(
            epoch,
            epochs=args.lr_decay_epochs,
            warmup_epochs=args.warmup_epochs,
            minimum_ratio=args.min_lr / args.lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale * float(group.get("lr_multiplier", 1.0))
        current_rollout = rollout_probability(
            epoch,
            args.rollout_probability,
            args.rollout_ramp_epochs,
        )
        train_metrics = run_loader(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            rollout_p=current_rollout,
        )
        teacher_metrics = run_loader(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            scaler=None,
            args=args,
            rollout_p=0.0,
        )
        rollout_metrics = free_rollout_metrics(
            model,
            validation_dataset,
            params=trace_params,
            limit=args.max_val_images,
        )
        recent_f1.append(rollout_metrics["group_macro_f1"])
        recent_f1 = recent_f1[-3:]
        smoothed = float(np.mean(recent_f1))
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "rollout_probability": current_rollout,
            "train_loss": train_metrics["loss"],
            "train_action_loss": train_metrics["action_loss"],
            "train_state_loss": train_metrics["state_loss"],
            "train_support_loss": train_metrics["support_loss"],
            "train_endpoint_loss": train_metrics["endpoint_loss"],
            "validation_teacher_loss": teacher_metrics["loss"],
            "validation_precision": rollout_metrics["precision"],
            "validation_recall": rollout_metrics["recall"],
            "validation_f1": rollout_metrics["f1"],
            "validation_group_macro_f1": rollout_metrics["group_macro_f1"],
            "validation_chamfer_mean": rollout_metrics["chamfer_mean"],
            "validation_base_completion_rate": rollout_metrics[
                "base_completion_rate"
            ],
        }
        with metric_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fieldnames).writerow(row)
        metadata = {
            "epoch": epoch,
            "training_signature": training_signature(args),
            "best_smoothed_f1": max(best_smoothed_f1, smoothed),
            "recent_group_macro_f1": recent_f1,
            "rollout_metrics": rollout_metrics,
        }
        training_state = {
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng_state(loader_generator, scaler),
        }
        save_noodle_tracer_checkpoint(
            str(args.out_dir / "last.pt"),
            model,
            trace_params=trace_params,
            metadata=metadata,
            training_state=training_state,
        )
        if smoothed > best_smoothed_f1:
            best_smoothed_f1 = smoothed
            save_noodle_tracer_checkpoint(
                str(args.out_dir / "best.pt"),
                model,
                trace_params=trace_params,
                metadata=metadata,
                training_state=training_state,
            )
        print(json.dumps(row, sort_keys=True))
    _atomic_json(
        args.out_dir / "summary.json",
        {
            "best_smoothed_group_macro_f1": best_smoothed_f1,
            "best_checkpoint": str(args.out_dir / "best.pt"),
            "last_checkpoint": str(args.out_dir / "last.pt"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
