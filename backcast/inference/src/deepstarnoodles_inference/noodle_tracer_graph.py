# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""CUDA-graph replay of the experimental fixed-slot GRU rollout.

The fixed-slot loop is dominated by host launch overhead: each recurrent step
issues roughly two hundred small kernels for a few dozen walkers. This module
records those same eager kernels once per walker count and replays them. It
changes no model, precision, threshold, seed, patch or termination rule:

* walker counts are never padded, so every kernel sees the fixed-slot shapes;
* the step-dependent path write and loop-history window use a device-side step
  counter instead of Python slicing; masked history slots are excluded with
  +inf before an exact minimum, matching the populated-prefix comparison;
* the first step, and any remainder too short for one replay, run eagerly
  through the same step code; and
* completed walkers stay masked, so steps after every walker stopped do not
  alter any output, exactly as for fixed-slot aggregate checks.

Parity with ``--gru-execution fixed`` must still be measured on the target GPU.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import cuda_sched
from .capture_lock import CAPTURE_LOCK, GRAPH_GATE
from .noodle_tracer import NoodleTraceParams, NoodleTracer, RGBPyramid, TraceState


_INVALID = int(TraceState.INVALID_SEED)
_CONTINUE = int(TraceState.CONTINUE)
_BASE = int(TraceState.TERMINATE_BASE)
_LOST = int(TraceState.TERMINATE_LOST)
STOP_BOUNDS = len(TraceState)
STOP_MAX_LENGTH = STOP_BOUNDS + 1
STOP_LOOP = STOP_BOUNDS + 2
STOP_MAX_STEPS = STOP_BOUNDS + 3


@dataclass
class RolloutState:
    """Preallocated per-walker state; graphs read and write these tensors."""

    positions: torch.Tensor
    image_indices: torch.Tensor
    seeds: torch.Tensor
    hidden: torch.Tensor
    previous: torch.Tensor
    active: torch.Tensor
    accepted: torch.Tensor
    path_history: torch.Tensor
    path_counts: torch.Tensor
    path_lengths: torch.Tensor
    state_confidence: torch.Tensor
    stop_codes: torch.Tensor
    step: torch.Tensor
    slots: torch.Tensor


@dataclass
class _Inputs:
    """Preallocated dense fields and RGB pyramid shared by one geometry."""

    raw: torch.Tensor
    pyramid: RGBPyramid


class _Entry:
    def __init__(self, state: RolloutState, inputs: _Inputs, ranges: tuple, size: tuple[int, int]):
        self.state = state
        self.inputs = inputs
        self.ranges = ranges
        self.size = size
        self.graph: torch.cuda.CUDAGraph | None = None
        self.first_graph: torch.cuda.CUDAGraph | None = None


def dense_patches_one_image(fields: torch.Tensor, positions: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``extract_centered_patches`` for one image without per-walker field copies.

    The reference expands one ``(1,C,H,W)`` field to N walkers. With zero
    padding and aligned corners, ``grid_sample`` then dispatches to cuDNN, which
    first materializes N contiguous copies of the whole field. Folding the N
    patch grids into one grid over the single image uses the same backend and
    identical per-sample coordinates, without those copies.
    """
    if patch_size % 2 != 1:
        raise ValueError("patch_size must be odd")
    if fields.ndim != 4 or fields.shape[0] != 1 or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("invalid field or position shape")
    _, channels, height, width = fields.shape
    radius = patch_size // 2
    offsets = torch.arange(-radius, radius + 1, device=positions.device, dtype=positions.dtype)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    x = positions[:, 0, None, None] + xx
    y = positions[:, 1, None, None] + yy
    gx = 2.0 * x / max(width - 1, 1) - 1.0
    gy = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    count = grid.shape[0]
    sampled = F.grid_sample(
        fields,
        grid.reshape(1, count * patch_size, patch_size, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(channels, count, patch_size, patch_size).transpose(0, 1).clone(
        memory_format=torch.contiguous_format,
    )


def _sample_patches(model, fields, pyramid, positions, image_indices, ranges):
    # Imported lazily: noodle_tracer_fixed imports this module for dispatch.
    from .noodle_tracer_fixed import _sample_fixed_patches

    if model.context_mode != "single" or type(model).sample_dense is not NoodleTracer.sample_dense:
        return _sample_fixed_patches(model, fields, pyramid, positions, image_indices, ranges)
    if fields.shape[0] == 1:
        dense = dense_patches_one_image(fields, positions, model.dense_patch_size)
        fine, coarse = model.sample_rgb(pyramid, positions, image_indices)
        return dense, fine, coarse
    dense_parts, fine_parts, coarse_parts = [], [], []
    for image_index, start, end in ranges:
        local_positions = positions[start:end]
        dense_parts.append(dense_patches_one_image(
            fields[image_index:image_index + 1], local_positions, model.dense_patch_size,
        ))
        local_pyramid = RGBPyramid(pyramid.fine[image_index:image_index + 1],
                                   pyramid.coarse[image_index:image_index + 1], pyramid.image_size)
        fine, coarse = model.sample_rgb(local_pyramid, local_positions, image_indices.new_zeros(end - start))
        fine_parts.append(fine)
        coarse_parts.append(coarse)
    return torch.cat(dense_parts), torch.cat(fine_parts), torch.cat(coarse_parts)


def rollout_step(
    model: NoodleTracer,
    params: NoodleTraceParams,
    state: RolloutState,
    inputs: _Inputs,
    ranges: tuple,
    size: tuple[int, int],
    *,
    first: bool,
) -> None:
    """One fixed-slot recurrent step, committed into ``state`` in place.

    Arithmetic mirrors one iteration of ``trace_arms_noodle_batch_fixed_slots``.
    Only the step index lives on the device, so the same code can be captured.
    """
    height, width = size
    dense_patches, rgb_fine, rgb_coarse = _sample_patches(
        model, inputs.raw, inputs.pyramid, state.positions, state.image_indices, ranges,
    )
    displacement, state_logits, next_hidden = model.forward_step(
        dense_patches, rgb_fine, rgb_coarse, state.previous, state.seeds, state.hidden,
        identity_scale=params.identity_scale,
    )
    probabilities = torch.softmax(state_logits, dim=1)
    decisions = probabilities.argmax(dim=1)
    if params.lost_state_threshold is not None:
        decisions = torch.where(
            probabilities[:, _LOST] >= params.lost_state_threshold,
            decisions.new_full((), _LOST), decisions,
        )
    if params.base_state_threshold is not None:
        decisions = torch.where(
            probabilities[:, _BASE] >= params.base_state_threshold,
            decisions.new_full((), _BASE), decisions,
        )
    if first and params.invalid_seed_threshold is not None:
        reject = probabilities[:, _INVALID] >= params.invalid_seed_threshold
        alternatives = probabilities.clone()
        alternatives[:, _INVALID] = -1
        non_invalid_decision = alternatives.argmax(dim=1)
        decisions = torch.where(
            reject, decisions.new_full((), _INVALID),
            torch.where(decisions == _INVALID, non_invalid_decision, decisions),
        )
    active = state.active
    hidden = torch.where(active[:, None], next_hidden, state.hidden)
    confidence = probabilities.gather(1, decisions[:, None]).squeeze(1)
    state_confidence = torch.where(active, confidence, state.state_confidence)

    terminal = active & (decisions != _CONTINUE)
    terminal_decisions = decisions.to(torch.int8)
    terminal_decisions = torch.where(
        (terminal_decisions == _INVALID) & state.accepted,
        terminal_decisions.new_full((), _LOST), terminal_decisions,
    )
    stop_codes = torch.where(terminal, terminal_decisions, state.stop_codes)
    continuing = active & ~terminal
    accepted = state.accepted | continuing
    candidates = state.positions + displacement
    in_bounds = ((candidates[:, 0] >= 0) & (candidates[:, 0] <= width - 1)
                 & (candidates[:, 1] >= 0) & (candidates[:, 1] <= height - 1))
    out_of_bounds = continuing & ~in_bounds
    stop_codes = torch.where(out_of_bounds, stop_codes.new_full((), STOP_BOUNDS), stop_codes)
    valid = continuing & in_bounds
    positions = torch.where(valid[:, None], candidates, state.positions)
    previous = torch.where(valid[:, None], displacement, state.previous)
    # Every still-active walker has appended on every prior step, so its next
    # slot is step + 1 for all walkers, as in the fixed-slot loop.
    slot = state.step + 1
    written = torch.where(
        valid[:, None], candidates, state.path_history.index_select(1, slot).squeeze(1),
    )
    state.path_history.index_copy_(1, slot, written[:, None])
    path_counts = state.path_counts + valid.to(torch.long)
    step_lengths = torch.linalg.vector_norm(displacement, dim=1)
    path_lengths = state.path_lengths + torch.where(valid, step_lengths, torch.zeros_like(step_lengths))
    max_length = valid & (path_lengths >= params.max_path_px)
    stop_codes = torch.where(max_length, stop_codes.new_full((), STOP_MAX_LENGTH), stop_codes)
    next_active = valid & ~max_length
    # Loop history is the populated prefix [0, step + 2 - loop_history). Slots
    # outside it become +inf; the minimum over the prefix is unchanged, and an
    # empty prefix cannot terminate a walker.
    history_stop = state.step + (2 - params.loop_history)
    distances = torch.linalg.vector_norm(state.path_history - positions[:, None], dim=2)
    distances = distances.masked_fill(state.slots[None, :] >= history_stop, float("inf"))
    looped = next_active & (distances.amin(dim=1) <= params.loop_radius)
    stop_codes = torch.where(looped, stop_codes.new_full((), STOP_LOOP), stop_codes)
    next_active = next_active & ~looped

    state.hidden.copy_(hidden)
    state.state_confidence.copy_(state_confidence)
    state.stop_codes.copy_(stop_codes)
    state.accepted.copy_(accepted)
    state.positions.copy_(positions)
    state.previous.copy_(previous)
    state.path_counts.copy_(path_counts)
    state.path_lengths.copy_(path_lengths)
    state.active.copy_(next_active)
    state.step.add_(1)


class FixedSlotGraphRollout:
    """Cache one CUDA graph of ``steps_per_replay`` steps per walker layout.

    A layout is the dense/RGB geometry plus the contiguous per-image seed
    ranges; crop batch 1 therefore keys on the walker count alone. Graphs share
    one memory pool and replay sequentially on the caller's stream. At most
    ``max_graphs`` layouts are retained (least recently used are released).
    """

    def __init__(
        self,
        model: NoodleTracer,
        params: NoodleTraceParams,
        *,
        steps_per_replay: int = 1,
        max_graphs: int = 96,
        max_walkers: int = 1024,
        compile_step: bool = False,
        graph_first_step: bool = False,
    ) -> None:
        if isinstance(steps_per_replay, bool) or not isinstance(steps_per_replay, int) or not 1 <= steps_per_replay <= 128:
            raise ValueError("steps_per_replay must be an integer between 1 and 128")
        if max_graphs < 1 or max_walkers < 1:
            raise ValueError("graph cache limits must be positive")
        self.model = model
        self.params = params
        self.steps_per_replay = int(steps_per_replay)
        self.max_graphs = int(max_graphs)
        self.max_walkers = int(max_walkers)
        # torch.compile'd step (see gru_compile); not byte-identical to the eager step.
        self.compile_step = bool(compile_step)
        # Replays each rollout's first step (seed rejection included) from its own graph.
        self.graph_first_step = bool(graph_first_step)
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("CUDA-graph GRU execution requires a CUDA tracer")
        self._entries: OrderedDict = OrderedDict()
        self._inputs: dict = {}
        self._pool = None
        self._stream = None
        self.captures = 0
        self.evictions = 0

    def summary(self) -> dict:
        return {"steps_per_replay": self.steps_per_replay, "captures": self.captures,
                "layouts": len(self._entries), "evictions": self.evictions,
                "max_graphs": self.max_graphs, "max_walkers": self.max_walkers,
                "compiled_step": self.compile_step, "graph_first_step": self.graph_first_step}

    def call_step(self, state, inputs, ranges, size, *, first: bool) -> None:
        """One step outside replay: eager, or compiled when ``compile_step`` is set.

        A compiled call may compile, and compilation synchronizes the device; it
        therefore runs under CAPTURE_LOCK so no other thread is capturing.
        """
        if not self.compile_step:
            rollout_step(self.model, self.params, state, inputs, ranges, size, first=first)
            return
        from .gru_compile import compiled_rollout_step
        with CAPTURE_LOCK:
            compiled_rollout_step()(self.model, self.params, state, inputs, ranges, size, first=first)

    def first_step(self, entry: _Entry) -> None:
        """The rollout's first step on the current stream: replayed, or a direct call."""
        if entry.first_graph is None:
            self.call_step(entry.state, entry.inputs, entry.ranges, entry.size, first=True)
            return
        with GRAPH_GATE.replay():
            entry.first_graph.replay()

    def _inputs_for(self, raw: torch.Tensor, pyramid: RGBPyramid) -> tuple[tuple, _Inputs]:
        key = (tuple(raw.shape), raw.dtype, tuple(pyramid.fine.shape), pyramid.fine.dtype,
               tuple(pyramid.coarse.shape), pyramid.coarse.dtype, tuple(pyramid.image_size))
        inputs = self._inputs.get(key)
        if inputs is None:
            inputs = _Inputs(
                raw=torch.empty_like(raw, memory_format=torch.contiguous_format),
                pyramid=RGBPyramid(
                    fine=torch.empty_like(pyramid.fine, memory_format=torch.contiguous_format),
                    coarse=torch.empty_like(pyramid.coarse, memory_format=torch.contiguous_format),
                    image_size=tuple(pyramid.image_size),
                ),
            )
            self._inputs[key] = inputs
        return key, inputs

    def _new_state(self, count: int, seed_dim: int) -> RolloutState:
        device, steps = self.device, self.params.max_steps
        return RolloutState(
            positions=torch.zeros((count, 2), dtype=torch.float32, device=device),
            image_indices=torch.zeros(count, dtype=torch.long, device=device),
            seeds=torch.zeros((count, seed_dim), dtype=torch.float32, device=device),
            hidden=self.model.initial_state(count, device=device),
            previous=torch.zeros((count, 2), dtype=torch.float32, device=device),
            active=torch.ones(count, dtype=torch.bool, device=device),
            accepted=torch.zeros(count, dtype=torch.bool, device=device),
            path_history=torch.zeros((count, steps + 1, 2), dtype=torch.float32, device=device),
            path_counts=torch.ones(count, dtype=torch.long, device=device),
            path_lengths=torch.zeros(count, dtype=torch.float32, device=device),
            state_confidence=torch.zeros(count, dtype=torch.float32, device=device),
            stop_codes=torch.full((count,), STOP_MAX_STEPS, dtype=torch.int8, device=device),
            step=torch.zeros(1, dtype=torch.long, device=device),
            slots=torch.arange(steps + 1, dtype=torch.long, device=device),
        )

    @staticmethod
    def _reset(state: RolloutState, positions, image_indices, seeds) -> None:
        state.positions.copy_(positions)
        state.image_indices.copy_(image_indices)
        state.seeds.copy_(seeds)
        state.hidden.zero_()
        state.previous.zero_()
        state.active.fill_(True)
        state.accepted.fill_(False)
        state.path_history.zero_()
        state.path_history[:, 0].copy_(positions)
        state.path_counts.fill_(1)
        state.path_lengths.zero_()
        state.state_confidence.zero_()
        state.stop_codes.fill_(STOP_MAX_STEPS)
        state.step.zero_()

    def _steps(self, entry: _Entry, count: int) -> None:
        for _ in range(count):
            self.call_step(entry.state, entry.inputs, entry.ranges, entry.size, first=False)

    def _capture(self, entry: _Entry) -> None:
        with CAPTURE_LOCK, GRAPH_GATE.capture():
            self._capture_locked(entry)

    def _capture_locked(self, entry: _Entry) -> None:
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            self._stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)
        # Warm up on the capture stream so library handles and plans exist.
        # State is reset with real walkers after capture.
        with torch.cuda.stream(self._stream):
            entry.state.step.zero_()
            self._steps(entry, self.steps_per_replay)
            entry.state.step.zero_()
            if self.graph_first_step:
                self.call_step(entry.state, entry.inputs, entry.ranges, entry.size, first=True)
                entry.state.step.zero_()
        self._stream.synchronize()
        if self.graph_first_step:
            # Captured before the step graph: rollouts replay the two in this order.
            first = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self._stream):
                first.capture_begin(pool=self._pool, capture_error_mode="thread_local")
                try:
                    self.call_step(entry.state, entry.inputs, entry.ranges, entry.size, first=True)
                finally:
                    first.capture_end()
            entry.first_graph = first
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self._stream):
            graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")
            try:
                self._steps(entry, self.steps_per_replay)
            finally:
                graph.capture_end()
        current.wait_stream(self._stream)
        entry.graph = graph
        self.captures += 1

    def _entry(self, inputs_key, inputs: _Inputs, ranges: tuple, size, count: int, seed_dim: int) -> _Entry:
        key = (inputs_key, ranges, seed_dim)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return entry
        while len(self._entries) >= self.max_graphs:
            _, old = self._entries.popitem(last=False)
            for captured in (old.graph, old.first_graph):
                if captured is not None:
                    captured.reset()
            self.evictions += 1
        entry = _Entry(self._new_state(count, seed_dim), inputs, ranges, size)
        self._capture(entry)
        self._entries[key] = entry
        return entry

    def run(
        self,
        raw: torch.Tensor,
        pyramid: RGBPyramid,
        positions: torch.Tensor,
        image_indices: torch.Tensor,
        seeds: torch.Tensor,
        ranges: list[tuple[int, int, int]],
        diagnostics: dict | None = None,
    ) -> RolloutState | None:
        """Roll out all walkers; returns live state buffers, or None if too large.

        The returned tensors are reused by the next call with the same layout;
        copy them before the next rollout.
        """
        count = int(positions.shape[0])
        if count > self.max_walkers:
            return None
        # Graph state is created, updated and captured only in inference mode,
        # so callers may hold or read it outside that mode.
        with torch.inference_mode():
            return self._run(raw, pyramid, positions, image_indices, seeds, ranges, count, diagnostics)

    def _run(self, raw, pyramid, positions, image_indices, seeds, ranges, count, diagnostics):
        inputs_key, inputs = self._inputs_for(raw, pyramid)
        size = (int(raw.shape[-2]), int(raw.shape[-1]))
        entry = self._entry(inputs_key, inputs, tuple(ranges), size, count, int(seeds.shape[1]))
        inputs.raw.copy_(raw)
        inputs.pyramid.fine.copy_(pyramid.fine)
        inputs.pyramid.coarse.copy_(pyramid.coarse)
        state = entry.state
        self._reset(state, positions, image_indices, seeds)
        max_steps = self.params.max_steps
        self.first_step(entry)
        steps, replays, checks = 1, 0, 0
        while steps < max_steps:
            checks += 1
            # The only data-dependent host synchronization, as in fixed slots.
            if not bool(state.active.any()):
                break
            if steps + self.steps_per_replay <= max_steps:
                with GRAPH_GATE.replay():
                    entry.graph.replay()
                replays += 1
                steps += self.steps_per_replay
            else:
                self._steps(entry, 1)
                steps += 1
        if diagnostics is not None:
            diagnostics.update(rollout_steps=steps, aggregate_checks=checks,
                               scheduled_slot_steps=steps * count, graph_replays=replays,
                               graph_captures=self.captures, graph_layouts=len(self._entries),
                               graph_evictions=self.evictions)
        return state


class _Slot:
    def __init__(self, runner: FixedSlotGraphRollout, device) -> None:
        self.runner = runner
        self.stream = torch.cuda.Stream(device=device)
        # Two pinned flags/events: one is being written while the other is read.
        self.flags = [torch.zeros(1, dtype=torch.bool, pin_memory=True) for _ in range(2)]
        self.events = [cuda_sched.event(), cuda_sched.event()]


class _Running:
    def __init__(self, index: int, entry: _Entry) -> None:
        self.index = index
        self.entry = entry
        self.steps = 1


class ConcurrentGraphRollouts:
    """Roll out several independent crops at once, one CUDA stream per slot.

    Each slot owns a ``FixedSlotGraphRollout`` (its own static buffers, graphs
    and memory pool), so concurrent crops never share buffers. One host thread
    replays every running slot's graph in turn and reads each slot's
    any-walker-active flag one step late through a pinned copy and an event, so
    streams never wait for the host between steps. Each crop still runs the
    exact fixed-slot step sequence; steps after all of its walkers have stopped
    are no-ops, as for fixed-slot aggregate checks. A finished slot is refilled
    from the remaining jobs.
    """

    def __init__(self, model: NoodleTracer, params: NoodleTraceParams, *, slots: int = 4,
                 max_graphs: int = 96, max_walkers: int = 1024, compile_step: bool = False,
                 graph_first_step: bool = False) -> None:
        if isinstance(slots, bool) or not isinstance(slots, int) or not 1 <= slots <= 8:
            raise ValueError("slots must be an integer between 1 and 8")
        self.model = model
        self.params = params
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("concurrent GRU rollouts require a CUDA tracer")
        self.max_walkers = int(max_walkers)
        self._slots = [_Slot(FixedSlotGraphRollout(model, params, steps_per_replay=1, max_graphs=max_graphs,
                                                   max_walkers=max_walkers, compile_step=compile_step,
                                                   graph_first_step=graph_first_step), self.device)
                       for _ in range(slots)]
        self.jobs = 0
        self.rounds = 0
        self.steps = 0

    def summary(self) -> dict:
        return {"slots": len(self._slots), "jobs": self.jobs, "rounds": self.rounds, "steps": self.steps,
                "captures": sum(slot.runner.captures for slot in self._slots),
                "layouts": sum(len(slot.runner._entries) for slot in self._slots)}

    def _flag(self, slot: _Slot, entry: _Entry, step: int) -> None:
        parity = step % 2
        slot.flags[parity].copy_(entry.state.active.any().reshape(1), non_blocking=True)
        slot.events[parity].record(slot.stream)

    def _start(self, slot: _Slot, index: int, job) -> _Running:
        runner, model, params = slot.runner, self.model, self.params
        slot.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(slot.stream):
            pyramid = model.encode_rgb(job.image_t)
            inputs_key, inputs = runner._inputs_for(job.raw_t, pyramid)
            size = (int(job.raw_t.shape[-2]), int(job.raw_t.shape[-1]))
            entry = runner._entry(inputs_key, inputs, tuple(job.ranges), size, job.count, int(job.seeds.shape[1]))
            inputs.raw.copy_(job.raw_t)
            inputs.pyramid.fine.copy_(pyramid.fine)
            inputs.pyramid.coarse.copy_(pyramid.coarse)
            runner._reset(entry.state, job.positions, job.image_indices, job.seeds)
            runner.first_step(entry)
            self._flag(slot, entry, 1)
        self.steps += 1
        return _Running(index, entry)

    def _finish(self, slot: _Slot, run: _Running):
        slot.stream.synchronize()
        state = run.entry.state
        return (state.path_history.cpu().numpy(), state.path_counts.cpu().numpy(), state.accepted.cpu().numpy(),
                state.state_confidence.cpu().numpy(), state.stop_codes.cpu().numpy())

    def run_many(self, jobs) -> list:
        """Final host rollout arrays for each prepared ``FixedSlotJob``, in order."""
        results = [None] * len(jobs)

        def done(index, arrays):
            results[index] = arrays

        with torch.inference_mode():
            session = _Session(self, done)
            for index, job in enumerate(jobs):
                session.submit(index, job)
            session.drain()
        return results

    def session(self, on_done):
        """Incremental scheduling: submit prepared jobs while others roll out."""
        return _Session(self, on_done)


class _Session:
    """Keeps every free slot busy; ``on_done(index, arrays)`` runs as crops finish.

    Host work between calls (preparing the next crop, assembling a finished
    one) overlaps the queued replays on the slot streams.
    """

    def __init__(self, owner: ConcurrentGraphRollouts, on_done) -> None:
        self.owner = owner
        self.on_done = on_done
        self.pending: list = []
        self.free = list(range(len(owner._slots)))[::-1]
        self.running: dict[int, _Running] = {}
        self.jobs: dict[int, object] = {}

    def submit(self, index: int, job) -> None:
        if job.count > self.owner.max_walkers:
            raise ValueError("walker count exceeds the concurrent rollout limit")
        self.jobs[index] = job
        self.pending.append(index)
        self._fill()
        self._round(block=False)

    def _fill(self) -> None:
        owner = self.owner
        while self.free and self.pending:
            index = self.pending.pop(0)
            slot_index = self.free.pop()
            self.running[slot_index] = owner._start(owner._slots[slot_index], index, self.jobs[index])
            owner.jobs += 1

    def _round(self, *, block: bool) -> None:
        owner, max_steps = self.owner, self.owner.params.max_steps
        owner.rounds += 1
        for slot_index, run in self.running.items():
            if run.steps < max_steps:
                slot = owner._slots[slot_index]
                with torch.cuda.stream(slot.stream):
                    with GRAPH_GATE.replay():
                        run.entry.graph.replay()
                    run.steps += 1
                    owner.steps += 1
                    owner._flag(slot, run.entry, run.steps)
        for slot_index in list(self.running):
            run = self.running[slot_index]
            slot = owner._slots[slot_index]
            if run.steps >= max_steps:
                done = True
            else:
                # Flag after the previous step; the step just enqueued is a
                # no-op when no walker was active before it.
                parity = (run.steps - 1) % 2
                event = slot.events[parity]
                if not block and not event.query():
                    continue
                event.synchronize()
                done = not bool(slot.flags[parity].item())
            if done:
                arrays = owner._finish(slot, run)
                del self.running[slot_index]
                self.free.append(slot_index)
                self.jobs.pop(run.index, None)
                self.on_done(run.index, arrays)
        self._fill()

    def drain(self) -> None:
        while self.running or self.pending:
            self._round(block=True)


@torch.no_grad()
def trace_arms_fixed_slots_concurrent(groups, model: NoodleTracer, params: NoodleTraceParams,
                                      scheduler: ConcurrentGraphRollouts, *, device_peaks: bool = True):
    """Decode several independent ``(raw, images)`` groups with concurrent rollouts.

    Seed selection and output assembly are those of the fixed-slot decoder;
    only the recurrent rollouts of different groups overlap on separate
    streams. Each group starts as soon as its seeds are ready, and finished
    groups are assembled while later rollouts continue.
    """
    from .noodle_tracer_fixed import (
        finish_fixed_slots, prepare_fixed_slots, trace_arms_noodle_batch_fixed_slots,
    )

    groups = list(groups)
    outputs = [None] * len(groups)
    jobs = {}

    def done(index, arrays):
        outputs[index] = finish_fixed_slots(jobs.pop(index), *arrays)

    with torch.inference_mode():
        session = scheduler.session(done)
        for index, (raw, images) in enumerate(groups):
            job = prepare_fixed_slots(raw, images, model, params, device_peaks=device_peaks)
            if job.count == 0:
                outputs[index] = job.outputs
            elif job.count > scheduler.max_walkers:
                outputs[index] = trace_arms_noodle_batch_fixed_slots(
                    raw, images, model, params, check_interval=1, device_peaks=device_peaks)
            else:
                jobs[index] = job
                session.submit(index, job)
        session.drain()
    return outputs
