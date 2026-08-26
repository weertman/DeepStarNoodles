# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Projection-based supervision for the noodle tracer policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from deepstarnoodles.decode import activate_logits, find_peaks
from deepstarnoodles.noodle_tracer import TraceState


@dataclass
class NoodleTeacherTrajectories:
    image_indices: torch.Tensor
    seed_positions: torch.Tensor
    seed_embeddings: torch.Tensor
    paths: torch.Tensor
    path_valid: torch.Tensor
    valid_seed: torch.Tensor
    matched_arms: int
    false_tips: int
    ignored_tips: int


@dataclass
class NoodleStepTargets:
    displacement: torch.Tensor
    state: torch.Tensor
    projection: torch.Tensor
    distance_to_path: torch.Tensor
    remaining_length: torch.Tensor


def _radius_gated_assignments(
    peaks: np.ndarray,
    gt_tips: np.ndarray,
    radius: float,
) -> list[tuple[int, int]]:
    if not len(peaks) or not len(gt_tips):
        return []
    distances = np.linalg.norm(peaks[:, None, :2] - gt_tips[None], axis=2)
    forbidden = radius + 1e6
    cost = np.where(distances <= radius, distances, forbidden + distances)
    peak_indices, arm_indices = linear_sum_assignment(cost)
    return [
        (int(peak), int(arm))
        for peak, arm in zip(peak_indices, arm_indices)
        if float(distances[peak, arm]) <= radius
    ]


def _sample_seed_embedding(
    raw: torch.Tensor,
    xy: np.ndarray,
    embedding_dim: int,
) -> np.ndarray:
    x = int(np.clip(round(float(xy[0])), 0, raw.shape[-1] - 1))
    y = int(np.clip(round(float(xy[1])), 0, raw.shape[-2] - 1))
    return raw[10:10 + embedding_dim, y, x].detach().cpu().numpy().astype(np.float32)


def build_noodle_teacher_trajectories(
    raw_batch: torch.Tensor,
    arms_batch: Sequence[Sequence[np.ndarray]],
    *,
    embedding_dim: int = 8,
    tip_threshold: float = 0.2,
    tip_nms_radius: int = 4,
    match_radius: float = 12.0,
    ambiguous_negative_radius: float = 20.0,
    include_false_tips: bool = True,
) -> NoodleTeacherTrajectories:
    """Match seeds and retain GT paths without overwriting their true tips."""
    image_indices: list[int] = []
    seed_positions: list[np.ndarray] = []
    seed_embeddings: list[np.ndarray] = []
    paths: list[np.ndarray] = []
    valid_seeds: list[bool] = []
    matched_total = false_total = ignored_total = 0
    for image_index, (raw, arms) in enumerate(zip(raw_batch, arms_batch)):
        activated = activate_logits(raw[:10]).detach().cpu().numpy()
        peaks = find_peaks(activated[9], tip_threshold, tip_nms_radius)
        gt = [
            np.asarray(arm, dtype=np.float32)
            for arm in arms
            if len(arm) >= 2
        ]
        gt_tips = (
            np.stack([arm[-1] for arm in gt])
            if gt
            else np.zeros((0, 2), dtype=np.float32)
        )
        assignments = _radius_gated_assignments(
            peaks,
            gt_tips,
            match_radius,
        )
        used = {peak for peak, _ in assignments}
        for peak_index, arm_index in assignments:
            seed = peaks[peak_index, :2].astype(np.float32)
            image_indices.append(image_index)
            seed_positions.append(seed)
            seed_embeddings.append(
                _sample_seed_embedding(raw, seed, embedding_dim)
            )
            paths.append(gt[arm_index][::-1].copy())
            valid_seeds.append(True)
            matched_total += 1
        if not include_false_tips:
            continue
        for peak_index, peak in enumerate(peaks):
            if peak_index in used:
                continue
            distance = (
                float(np.linalg.norm(gt_tips - peak[:2], axis=1).min())
                if len(gt_tips)
                else float("inf")
            )
            if distance < ambiguous_negative_radius:
                ignored_total += 1
                continue
            seed = peak[:2].astype(np.float32)
            image_indices.append(image_index)
            seed_positions.append(seed)
            seed_embeddings.append(
                _sample_seed_embedding(raw, seed, embedding_dim)
            )
            paths.append(seed[None].copy())
            valid_seeds.append(False)
            false_total += 1
    if not paths:
        return NoodleTeacherTrajectories(
            image_indices=torch.zeros(0, dtype=torch.long),
            seed_positions=torch.zeros((0, 2), dtype=torch.float32),
            seed_embeddings=torch.zeros((0, embedding_dim), dtype=torch.float32),
            paths=torch.zeros((0, 1, 2), dtype=torch.float32),
            path_valid=torch.zeros((0, 1), dtype=torch.bool),
            valid_seed=torch.zeros(0, dtype=torch.bool),
            matched_arms=0,
            false_tips=0,
            ignored_tips=ignored_total,
        )
    max_length = max(len(path) for path in paths)
    padded = np.zeros((len(paths), max_length, 2), dtype=np.float32)
    path_valid = np.zeros((len(paths), max_length), dtype=np.bool_)
    for index, path in enumerate(paths):
        padded[index, :len(path)] = path
        path_valid[index, :len(path)] = True
    return NoodleTeacherTrajectories(
        image_indices=torch.as_tensor(image_indices, dtype=torch.long),
        seed_positions=torch.from_numpy(np.stack(seed_positions)),
        seed_embeddings=torch.from_numpy(np.stack(seed_embeddings)),
        paths=torch.from_numpy(padded),
        path_valid=torch.from_numpy(path_valid),
        valid_seed=torch.as_tensor(valid_seeds, dtype=torch.bool),
        matched_arms=matched_total,
        false_tips=false_total,
        ignored_tips=ignored_total,
    )


def _point_at_arc_length(
    path: torch.Tensor,
    cumulative: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    segment = int(
        torch.searchsorted(cumulative[1:], distance.detach(), right=False).item()
    )
    segment = min(segment, len(path) - 2)
    start_distance = cumulative[segment]
    segment_length = (cumulative[segment + 1] - start_distance).clamp_min(1e-6)
    fraction = ((distance - start_distance) / segment_length).clamp(0, 1)
    return path[segment] + fraction * (path[segment + 1] - path[segment])


def projection_lookahead_targets(
    positions: torch.Tensor,
    trajectories: NoodleTeacherTrajectories,
    trajectory_indices: torch.Tensor,
    *,
    lookahead_px: float = 8.0,
    max_step_px: float = 6.0,
    terminate_radius_px: float = 4.0,
    lost_radius_px: float = 16.0,
) -> NoodleStepTargets:
    """Project arbitrary rollout positions and derive displacement/state targets."""
    count = len(positions)
    displacement = torch.zeros_like(positions)
    projection = torch.zeros_like(positions)
    distance_to_path = torch.zeros(count, device=positions.device)
    remaining = torch.zeros(count, device=positions.device)
    state = torch.full(
        (count,),
        int(TraceState.INVALID_SEED),
        dtype=torch.long,
        device=positions.device,
    )
    paths = trajectories.paths.to(positions.device)
    path_valid = trajectories.path_valid.to(positions.device)
    valid_seed = trajectories.valid_seed.to(positions.device)
    for output_index, trajectory_tensor in enumerate(trajectory_indices):
        trajectory_index = int(trajectory_tensor)
        if not bool(valid_seed[trajectory_index]):
            projection[output_index] = positions[output_index]
            continue
        path = paths[trajectory_index, path_valid[trajectory_index]]
        if len(path) < 2:
            state[output_index] = int(TraceState.TERMINATE_LOST)
            projection[output_index] = positions[output_index]
            continue
        segments = path[1:] - path[:-1]
        squared_length = segments.square().sum(dim=1).clamp_min(1e-8)
        relative = positions[output_index][None] - path[:-1]
        fraction = (relative * segments).sum(dim=1) / squared_length
        fraction = fraction.clamp(0, 1)
        candidates = path[:-1] + fraction[:, None] * segments
        distances = torch.linalg.vector_norm(
            candidates - positions[output_index],
            dim=1,
        )
        nearest = int(distances.argmin())
        nearest_projection = candidates[nearest]
        lengths = torch.linalg.vector_norm(segments, dim=1)
        cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        arc = cumulative[nearest] + fraction[nearest] * lengths[nearest]
        remaining_length = cumulative[-1] - arc
        projection[output_index] = nearest_projection
        distance_to_path[output_index] = distances[nearest]
        remaining[output_index] = remaining_length
        if float(distances[nearest]) > lost_radius_px:
            state[output_index] = int(TraceState.TERMINATE_LOST)
            continue
        if float(remaining_length) <= terminate_radius_px:
            state[output_index] = int(TraceState.TERMINATE_BASE)
            continue
        target_arc = torch.minimum(
            arc + float(lookahead_px),
            cumulative[-1],
        )
        target = _point_at_arc_length(path, cumulative, target_arc)
        delta = target - positions[output_index]
        norm = torch.linalg.vector_norm(delta).clamp_min(1e-6)
        displacement[output_index] = delta * min(1.0, max_step_px / float(norm))
        state[output_index] = int(TraceState.CONTINUE)
    return NoodleStepTargets(
        displacement=displacement,
        state=state,
        projection=projection,
        distance_to_path=distance_to_path,
        remaining_length=remaining,
    )
