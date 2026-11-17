# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Experimental fixed-slot GRU rollout; dynamic execution remains the default."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from .noodle_tracer import (
    NoodleTracer, NoodleTraceParams, RGBPyramid, TraceState,
    select_trace_seeds, find_peaks, _seed_embedding, _smooth_resample,
    _suppress_duplicates_and_fragments, EmbedDecodeParams,
)


def _sample_fixed_patches(model, fields, pyramid, positions, image_indices, ranges):
    if fields.shape[0] == 1:
        dense = model.sample_dense(fields, positions, None)
        fine, coarse = model.sample_rgb(pyramid, positions, image_indices)
        return dense, fine, coarse
    dense_parts, fine_parts, coarse_parts = [], [], []
    for image_index, start, end in ranges:
        local_positions = positions[start:end]
        dense_parts.append(model.sample_dense(fields[image_index:image_index + 1], local_positions, None))
        local_pyramid = RGBPyramid(pyramid.fine[image_index:image_index + 1],
                                   pyramid.coarse[image_index:image_index + 1], pyramid.image_size)
        fine, coarse = model.sample_rgb(local_pyramid, local_positions, image_indices.new_zeros(end - start))
        fine_parts.append(fine)
        coarse_parts.append(coarse)
    return torch.cat(dense_parts), torch.cat(fine_parts), torch.cat(coarse_parts)


@dataclass
class FixedSlotJob:
    """Seeds and fields for one fixed-slot rollout, before any recurrent step."""

    model: NoodleTracer
    params: NoodleTraceParams
    raw_t: torch.Tensor
    image_t: torch.Tensor
    batch_size: int
    outputs: list
    device_peaks: bool
    host_fields: np.ndarray | None
    peak_support_by_image: list
    device_base_peaks: list | None
    tips: np.ndarray | None = None
    positions: torch.Tensor | None = None
    image_indices: torch.Tensor | None = None
    seeds: torch.Tensor | None = None
    ranges: list = field(default_factory=list)
    count: int = 0


@torch.no_grad()
def prepare_fixed_slots(raw, images, model, params=None, *, device_peaks=False) -> FixedSlotJob:
    """Validate inputs and select seeds exactly as the fixed-slot decoder does."""
    params = params or NoodleTraceParams(
        dense_patch_size=model.dense_patch_size,
        rgb_feature_patch_size=model.rgb_feature_patch_size,
        context_mode=model.context_mode,
        context_dilation=model.context_dilation,
        min_step_px=model.min_step_px,
        max_step_px=model.max_step_px,
    )
    if (
        params.dense_patch_size != model.dense_patch_size
        or params.rgb_feature_patch_size != model.rgb_feature_patch_size
        or params.context_mode != model.context_mode
        or params.context_dilation != model.context_dilation
    ):
        raise ValueError("trace patch geometry must match the model")
    device = next(model.parameters()).device
    raw_t = torch.as_tensor(raw, dtype=torch.float32, device=device)
    image_t = torch.as_tensor(images, device=device)
    if raw_t.ndim == 3:
        raw_t = raw_t[None]
    if image_t.ndim == 3:
        image_t = image_t[None]
    if image_t.ndim == 4 and image_t.shape[-1] == 3 and image_t.shape[1] != 3:
        image_t = image_t.permute(0, 3, 1, 2)
    if image_t.ndim != 4 or image_t.shape[1] != 3:
        raise ValueError("images must be aligned RGB as (B,3,H,W) or (B,H,W,3)")
    if raw_t.ndim != 4 or raw_t.shape[1] < model.dense_channels:
        raise ValueError("raw must be (B,C,H,W) with enough dense channels")
    if raw_t.shape[0] != image_t.shape[0]:
        raise ValueError("raw and images must have the same batch size")
    if raw_t.shape[-2:] != image_t.shape[-2:]:
        raise ValueError("raw and images must be spatially aligned")
    raw_t = raw_t[:, :model.dense_channels]
    batch_size = int(raw_t.shape[0])
    outputs: list[list[dict]] = [[] for _ in range(batch_size)]

    tips_by_image: list[np.ndarray] = []
    peak_support_by_image: list[np.ndarray | None] = []
    seed_arrays: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    device_base_peaks: list[np.ndarray] | None = None
    if device_peaks:
        from .device_peaks import device_seed_inputs

        host_fields = None
        device_base_peaks = []
        for image_index in range(batch_size):
            tips, image_seeds, base_peaks = device_seed_inputs(raw_t, params, image_index)
            tips_by_image.append(tips)
            device_base_peaks.append(base_peaks)
            if len(tips):
                seed_arrays.append(image_seeds)
                source_indices.append(np.full(len(tips), image_index, dtype=np.int64))
    else:
        # Preserve exact SciPy peak extraction while moving only fields used by
        # host-side seed/base postprocessing: raw support, activated support/base/
        # tip, and the identity embedding.  The full 18-channel tensor stays on GPU.
        host_fields = torch.cat(
            (
                raw_t[:, 0:1],
                torch.sigmoid(raw_t[:, 0:1]),
                torch.sigmoid(raw_t[:, 8:10]),
                raw_t[:, 10 : 10 + params.embedding_dim],
            ),
            dim=1,
        ).detach().cpu().numpy()
    for image_index in range(0 if device_peaks else batch_size):
        fields = host_fields[image_index]
        raw_support = fields[0]
        activated_support = fields[1]
        activated_tip = fields[3]
        embedding = fields[4:]
        peak_support = (
            raw_support > 0 if params.support_gated_peaks else None
        )
        tip_heat = (
            np.where(peak_support, activated_tip, 0.0)
            if peak_support is not None
            else activated_tip
        )
        tips = select_trace_seeds(
            find_peaks(tip_heat, params.tip_threshold, params.tip_nms_radius),
            max_tips=params.max_tips,
            min_separation=params.tip_seed_separation,
        )
        tips_by_image.append(tips)
        peak_support_by_image.append(peak_support)
        if not len(tips):
            continue
        seed_support = (
            peak_support
            if peak_support is not None
            else activated_support > 0.2
        )
        seed_arrays.append(
            np.stack(
                [
                    _seed_embedding(embedding, tip[:2], params, seed_support)
                    for tip in tips
                ]
            )
        )
        source_indices.append(
            np.full(len(tips), image_index, dtype=np.int64)
        )
    count = sum(len(tips) for tips in tips_by_image)
    job = FixedSlotJob(model=model, params=params, raw_t=raw_t, image_t=image_t, batch_size=batch_size,
                       outputs=outputs, device_peaks=device_peaks, host_fields=host_fields,
                       peak_support_by_image=peak_support_by_image, device_base_peaks=device_base_peaks)
    if count == 0:
        return job

    tips = np.concatenate(
        [tips for tips in tips_by_image if len(tips)],
        axis=0,
    )
    positions = torch.as_tensor(tips[:, :2], dtype=torch.float32, device=device)
    image_indices = torch.as_tensor(
        np.concatenate(source_indices),
        dtype=torch.long,
        device=device,
    )
    seeds = torch.as_tensor(
        np.concatenate(seed_arrays, axis=0),
        dtype=torch.float32,
        device=device,
    )
    # CPU peak extraction already knows these ranges; never re-group CUDA
    # image indices with nonzero during the rollout.
    ranges = []
    offset = 0
    for image_index, image_tips in enumerate(tips_by_image):
        end = offset + len(image_tips)
        if end > offset:
            ranges.append((image_index, offset, end))
        offset = end
    job.tips, job.positions, job.image_indices, job.seeds = tips, positions, image_indices, seeds
    job.ranges, job.count = ranges, count
    return job


@torch.no_grad()
def trace_arms_noodle_batch_fixed_slots(
    raw: np.ndarray | torch.Tensor,
    images: np.ndarray | torch.Tensor,
    model: NoodleTracer,
    params: NoodleTraceParams | None = None,
    *,
    check_interval: int = 8,
    diagnostics: dict | None = None,
    graph_rollout=None,
    device_peaks: bool = False,
) -> list[list[dict]]:
    """Experimental opt-in rollout with fixed walker slots and masked state.

    Initial CPU seed counts define permanent, contiguous image ranges. Dead
    walkers keep their state but still execute until a periodic aggregate check.
    No seed limits, model precision, decoder thresholds or output rules change.
    This prototype must pass parity and measured GPU performance gates before
    promotion. The unchanged noodle_tracer.trace_arms_noodle_batch is reference.

    ``graph_rollout`` (a ``FixedSlotGraphRollout``) replays the same loop body
    from cached CUDA graphs; seed selection and output assembly are shared.
    ``device_peaks`` extracts peak candidates and seed windows on the field's
    device and applies the unchanged host rules to them (see device_peaks).
    """
    if isinstance(check_interval, bool) or not isinstance(check_interval, int) or not 1 <= check_interval <= 128:
        raise ValueError("check_interval must be a positive integer between 1 and 128")
    if diagnostics is not None:
        diagnostics.update(execution="fixed_slots", check_interval=check_interval,
                           seeds=0, rollout_steps=0, aggregate_checks=0, scheduled_slot_steps=0)

    job = prepare_fixed_slots(raw, images, model, params, device_peaks=device_peaks)
    if job.count == 0:
        return job.outputs
    params, raw_t, image_t, count = job.params, job.raw_t, job.image_t, job.count
    positions, image_indices, seeds, ranges = job.positions, job.image_indices, job.seeds, job.ranges
    device = raw_t.device
    hidden = model.initial_state(count, device=device)
    previous = torch.zeros((count, 2), device=device)
    active = torch.ones(count, dtype=torch.bool, device=device)
    accepted = torch.zeros(count, dtype=torch.bool, device=device)
    path_history = torch.zeros(
        (count, params.max_steps + 1, 2),
        dtype=torch.float32,
        device=device,
    )
    path_history[:, 0] = positions
    path_counts = torch.ones(count, dtype=torch.long, device=device)
    path_lengths = torch.zeros(count, dtype=torch.float32, device=device)
    state_confidence = torch.zeros(count, dtype=torch.float32, device=device)

    stop_invalid = int(TraceState.INVALID_SEED)
    stop_base = int(TraceState.TERMINATE_BASE)
    stop_lost = int(TraceState.TERMINATE_LOST)
    stop_bounds = len(TraceState)
    stop_max_length = stop_bounds + 1
    stop_loop = stop_bounds + 2
    stop_max_steps = stop_bounds + 3
    stop_codes = torch.full(
        (count,),
        stop_max_steps,
        dtype=torch.int8,
        device=device,
    )
    if diagnostics is not None:
        diagnostics["seeds"] = count
    pyramid = model.encode_rgb(image_t)
    height, width = raw_t.shape[-2:]
    graph_state = (None if graph_rollout is None else graph_rollout.run(
        raw_t, pyramid, positions, image_indices, seeds, ranges, diagnostics))
    if graph_state is not None:
        if diagnostics is not None:
            diagnostics["execution"] = "fixed_slots_cuda_graph"
        path_history, path_counts = graph_state.path_history, graph_state.path_counts
        accepted, state_confidence = graph_state.accepted, graph_state.state_confidence
        stop_codes = graph_state.stop_codes
    for step_index in range(0 if graph_state is not None else params.max_steps):
        if step_index and step_index % check_interval == 0:
            if diagnostics is not None:
                diagnostics["aggregate_checks"] += 1
            # The only data-dependent device-to-host synchronization in this
            # loop. Every other control-flow choice is static Python metadata.
            if not bool(active.any()):
                break
        if diagnostics is not None:
            diagnostics["rollout_steps"] += 1
            diagnostics["scheduled_slot_steps"] += count
        dense_patches, rgb_fine, rgb_coarse = _sample_fixed_patches(
            model, raw_t, pyramid, positions, image_indices, ranges,
        )
        displacement, state_logits, next_hidden = model.forward_step(
            dense_patches, rgb_fine, rgb_coarse, previous, seeds, hidden,
            identity_scale=params.identity_scale,
        )
        probabilities = torch.softmax(state_logits, dim=1)
        decisions = probabilities.argmax(dim=1)
        if params.lost_state_threshold is not None:
            decisions = torch.where(
                probabilities[:, int(TraceState.TERMINATE_LOST)] >= params.lost_state_threshold,
                decisions.new_full((), int(TraceState.TERMINATE_LOST)), decisions,
            )
        if params.base_state_threshold is not None:
            decisions = torch.where(
                probabilities[:, int(TraceState.TERMINATE_BASE)] >= params.base_state_threshold,
                decisions.new_full((), int(TraceState.TERMINATE_BASE)), decisions,
            )
        if step_index == 0 and params.invalid_seed_threshold is not None:
            reject = probabilities[:, int(TraceState.INVALID_SEED)] >= params.invalid_seed_threshold
            alternatives = probabilities.clone()
            alternatives[:, int(TraceState.INVALID_SEED)] = -1
            non_invalid_decision = alternatives.argmax(dim=1)
            decisions = torch.where(
                reject, decisions.new_full((), int(TraceState.INVALID_SEED)),
                torch.where(decisions == int(TraceState.INVALID_SEED), non_invalid_decision, decisions),
            )
        hidden = torch.where(active[:, None], next_hidden, hidden)
        confidence = probabilities.gather(1, decisions[:, None]).squeeze(1)
        state_confidence = torch.where(active, confidence, state_confidence)

        terminal = active & (decisions != int(TraceState.CONTINUE))
        terminal_decisions = decisions.to(torch.int8)
        terminal_decisions = torch.where(
            (terminal_decisions == stop_invalid) & accepted,
            terminal_decisions.new_full((), stop_lost), terminal_decisions,
        )
        stop_codes = torch.where(terminal, terminal_decisions, stop_codes)
        continuing = active & ~terminal
        accepted = accepted | continuing
        candidates = positions + displacement
        in_bounds = ((candidates[:, 0] >= 0) & (candidates[:, 0] <= width - 1)
                     & (candidates[:, 1] >= 0) & (candidates[:, 1] <= height - 1))
        out_of_bounds = continuing & ~in_bounds
        stop_codes = torch.where(out_of_bounds, stop_codes.new_full((), stop_bounds), stop_codes)
        valid = continuing & in_bounds
        positions = torch.where(valid[:, None], candidates, positions)
        previous = torch.where(valid[:, None], displacement, previous)
        # Every still-active walker has appended on every prior step. Its next
        # slot is therefore step_index + 1, identical to path_counts[valid].
        path_history[:, step_index + 1] = torch.where(
            valid[:, None], candidates, path_history[:, step_index + 1],
        )
        path_counts = path_counts + valid.to(torch.long)
        step_lengths = torch.linalg.vector_norm(displacement, dim=1)
        path_lengths = path_lengths + torch.where(valid, step_lengths, torch.zeros_like(step_lengths))
        max_length = valid & (path_lengths >= params.max_path_px)
        stop_codes = torch.where(max_length, stop_codes.new_full((), stop_max_length), stop_codes)
        next_active = valid & ~max_length
        history_stop = step_index + 2 - params.loop_history
        if history_stop > 0:
            distances = torch.linalg.vector_norm(
                path_history[:, :history_stop] - positions[:, None], dim=2,
            )
            looped = next_active & (distances.amin(dim=1) <= params.loop_radius)
            stop_codes = torch.where(looped, stop_codes.new_full((), stop_loop), stop_codes)
            next_active = next_active & ~looped
        active = next_active

    path_history_np = path_history.detach().cpu().numpy()
    path_counts_np = path_counts.detach().cpu().numpy()
    accepted_np = accepted.detach().cpu().numpy()
    state_confidence_np = state_confidence.detach().cpu().numpy()
    stop_codes_np = stop_codes.detach().cpu().numpy()
    return finish_fixed_slots(job, path_history_np, path_counts_np, accepted_np,
                              state_confidence_np, stop_codes_np)


def finish_fixed_slots(job: FixedSlotJob, path_history_np, path_counts_np, accepted_np,
                       state_confidence_np, stop_codes_np) -> list[list[dict]]:
    """Assemble arms from final host rollout arrays exactly as the decoder does."""
    model, params, outputs, tips, count = job.model, job.params, job.outputs, job.tips, job.count
    seeds, image_indices, batch_size = job.seeds, job.image_indices, job.batch_size
    host_fields, device_peaks = job.host_fields, job.device_peaks
    peak_support_by_image, device_base_peaks = job.peak_support_by_image, job.device_base_peaks
    stop_invalid = int(TraceState.INVALID_SEED)
    stop_base = int(TraceState.TERMINATE_BASE)
    stop_lost = int(TraceState.TERMINATE_LOST)
    stop_bounds = len(TraceState)
    stop_max_length = stop_bounds + 1
    stop_loop = stop_bounds + 2
    stop_max_steps = stop_bounds + 3
    seeds_np = seeds.detach().cpu().numpy()
    image_indices_np = image_indices.detach().cpu().numpy()
    stop_names = {
        stop_invalid: "invalid_seed",
        stop_base: "base",
        stop_lost: "lost",
        stop_bounds: "bounds",
        stop_max_length: "max_length",
        stop_loop: "loop",
        stop_max_steps: "max_steps",
    }
    base_peaks_by_image: list[np.ndarray] = device_base_peaks or []
    for image_index in range(0 if device_peaks else batch_size):
        fields = host_fields[image_index]
        peak_support = peak_support_by_image[image_index]
        base_heat = (
            np.where(peak_support, fields[2], 0.0)
            if peak_support is not None
            else fields[2]
        )
        base_peaks_by_image.append(
            find_peaks(base_heat, params.base_threshold, 3)
        )

    for index in range(count):
        path_count = int(path_counts_np[index])
        if not accepted_np[index] or path_count < 2:
            continue
        stop_reason = stop_names[int(stop_codes_np[index])]
        if stop_reason == "lost" and not params.emit_lost:
            continue
        tip_to_base = path_history_np[index, :path_count].copy()
        length = float(np.linalg.norm(np.diff(tip_to_base, axis=0), axis=1).sum())
        if length < params.min_path_px:
            continue
        base_to_tip = tip_to_base[::-1].copy()
        image_index = int(image_indices_np[index])
        base_peaks = base_peaks_by_image[image_index]
        base_score = 0.0
        if len(base_peaks):
            distances = np.linalg.norm(base_peaks[:, :2] - base_to_tip[0], axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= params.base_hit_radius:
                base_score = float(base_peaks[nearest, 2])
        arm = {
            "polyline": _smooth_resample(
                base_to_tip,
                params.smooth_window,
                params.resample_spacing,
            ),
            "tip_score": float(tips[index, 2]),
            "base_score": base_score,
            "mean_step_cost": float(1.0 - state_confidence_np[index]),
            "mean_identity_dist": 0.0,
            "length_px": length,
            "island_frac": 1.0,
            "identity": (seeds_np[index].copy() if model.embedding_features
                         else np.zeros_like(seeds_np[index])),
            "state_confidence": float(state_confidence_np[index]),
            "stop_reason": stop_reason,
            "steps": path_count - 1,
        }
        outputs[image_index].append(arm)
    if params.suppress_duplicates and params.embedding_dim and model.embedding_features:
        outputs = [
            _suppress_duplicates_and_fragments(image_arms, EmbedDecodeParams())
            for image_arms in outputs
        ]
    return outputs
