# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""CUDA-graph replay of Hydra raw-field forwards at fixed input shapes.

Hydra preprocessing resizes every crop to one square size, so archive and live
forwards repeat a few static shapes. Replaying a captured forward launches the
same autocast kernels without per-operator host overhead. Weights, precision,
memory format and output contract are unchanged; parity must still be measured
on the target GPU.
"""
from __future__ import annotations

import torch

from .capture_lock import CAPTURE_LOCK, GRAPH_GATE


class HydraGraph:
    """Replay ``HydraPredictor.predict_tensor_batch_raw`` for repeated shapes."""

    def __init__(self, predictor, *, max_graphs: int = 4) -> None:
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        if predictor.device.type != "cuda":
            raise ValueError("Hydra graph execution requires CUDA")
        self.predictor = predictor
        self.max_graphs = int(max_graphs)
        self._graphs: dict = {}
        self._pool = None
        self._stream = None
        self.captures = 0
        self.replays = 0
        self.eager_calls = 0

    @staticmethod
    def _key(images: torch.Tensor):
        layout = ("channels_last" if images.is_contiguous(memory_format=torch.channels_last)
                  and not images.is_contiguous() else "contiguous" if images.is_contiguous() else None)
        return None if layout is None else (tuple(images.shape), images.dtype, str(images.device), layout)

    def _capture(self, images: torch.Tensor, key):
        with CAPTURE_LOCK, GRAPH_GATE.capture():
            return self._capture_locked(images, key)

    def _capture_locked(self, images: torch.Tensor, key):
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            self._stream = torch.cuda.Stream(device=images.device)
        static_input = images.clone(memory_format=torch.preserve_format)
        current = torch.cuda.current_stream(images.device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            self.predictor.predict_tensor_batch_raw(static_input)
        self._stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self._stream):
            graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")
            try:
                static_output = self.predictor.predict_tensor_batch_raw(static_input)
            finally:
                graph.capture_end()
        current.wait_stream(self._stream)
        self.captures += 1
        entry = (graph, static_input, static_output)
        self._graphs[key] = entry
        return entry

    def raw(self, images: torch.Tensor) -> torch.Tensor:
        """Return float32 raw fields; the result is a fresh tensor."""
        # Static buffers are created and updated only in inference mode.
        with torch.inference_mode():
            return self._raw(images)

    def _raw(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.predictor.device)
        key = self._key(images)
        if key is None:
            self.eager_calls += 1
            return self.predictor.predict_tensor_batch_raw(images)
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= self.max_graphs:
                self.eager_calls += 1
                return self.predictor.predict_tensor_batch_raw(images)
            entry = self._capture(images, key)
        graph, static_input, static_output = entry
        static_input.copy_(images)
        with GRAPH_GATE.replay():
            graph.replay()
        self.replays += 1
        return static_output.clone(memory_format=torch.preserve_format)

    def summary(self) -> dict:
        return {"captures": self.captures, "replays": self.replays,
                "eager_calls": self.eager_calls, "max_graphs": self.max_graphs}
