# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Experimental concurrent crop groups for one frame on separate CUDA streams.

Each worker owns a stream plus its own Hydra and GRU graph state, so graphs
never share static buffers across concurrent groups. Every group still runs
the unchanged singleton Hydra/GRU computation; only independent groups overlap.
The caller consumes results in submission order, so arm tracking and output
records keep the original crop order.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import queue

import torch


class CropWorkspace:
    """Stream and graph state used by one crop-group job at a time."""

    def __init__(self, cascade) -> None:
        self.cascade = cascade
        self.stream = torch.cuda.Stream(device=cascade.device)
        self.hydra_graph = None
        if getattr(cascade.args, "hydra_execution", "eager") == "graph":
            from .hydra_graph import HydraGraph
            self.hydra_graph = HydraGraph(cascade.hydra)
        self._rollout = None

    def rollout(self):
        if self._rollout is None:
            from .noodle_tracer_graph import FixedSlotGraphRollout
            self._rollout = FixedSlotGraphRollout(
                self.cascade.tracer, self.cascade.trace_params,
                steps_per_replay=getattr(self.cascade.args, "gru_check_interval", 8),
                compile_step=getattr(self.cascade.args, "gru_compile", False),
                graph_first_step=getattr(self.cascade.args, "gru_graph_first_step", False))
        return self._rollout


class CropStreams:
    """Bounded thread/stream pool for crop-group jobs."""

    def __init__(self, cascade, count: int) -> None:
        if cascade.device.type != "cuda":
            raise ValueError("crop streams require CUDA")
        if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 4:
            raise ValueError("crop stream count must be between 2 and 4")
        self.cascade = cascade
        self.workspaces = [CropWorkspace(cascade) for _ in range(count)]
        self._free: queue.SimpleQueue = queue.SimpleQueue()
        for workspace in self.workspaces:
            self._free.put(workspace)
        self._pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="crop-stream")

    def _run(self, job):
        from .benchmark import Timer

        workspace = self._free.get()
        try:
            device = self.cascade.device
            with torch.inference_mode(), torch.cuda.device(device), torch.cuda.stream(workspace.stream):
                crop_group, images = job(workspace)
                activated, decoded, bodies = self.cascade.compute_crops(
                    crop_group, images, Timer(device, False), workspace)
                # Results (including live tracking maps) are complete before the
                # consumer reads them on another stream.
                workspace.stream.synchronize()
            return crop_group, activated, decoded, bodies
        finally:
            self._free.put(workspace)

    def map(self, jobs):
        """Yield ``(crop_group, activated, decoded, bodies)`` in job order."""
        futures = [self._pool.submit(self._run, job) for job in jobs]
        try:
            for future in futures:
                yield future.result()
        finally:
            for future in futures:
                future.cancel()
            wait(futures)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
