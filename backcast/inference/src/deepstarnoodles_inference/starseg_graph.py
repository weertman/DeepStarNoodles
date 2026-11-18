# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""CUDA-graph replay of StarSeg's backbone and neck at fixed input shapes.

Ultralytics runs the segmentation model layer by layer
(``BaseModel._predict_once``), launching about 800 kernels for one 1024x1024
image; on a fast GPU the host spends longer launching them than the GPU spends
running them. Every layer before the head is captured once per input shape and
replayed; the head runs eagerly on the replayed features, because the YOLO26
end-to-end selection indexes with a host tensor, which a graph cannot capture.
Replays launch the captured kernels, so features equal eager execution.
Ultralytics' preprocessing, head and postprocessing are unchanged.
"""
from __future__ import annotations

import torch

from .capture_lock import CAPTURE_LOCK, GRAPH_GATE


class StarSegGraph:
    """Replay ``model._predict_once`` up to the head for repeated input shapes."""

    def __init__(self, model, *, max_graphs: int = 2) -> None:
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        layers = list(model.model)
        if not layers or getattr(layers[-1], "f", -1) == -1:
            raise ValueError("expected an Ultralytics model whose head reads saved layers")
        self.model = model
        self.layers = layers
        self.max_graphs = int(max_graphs)
        self._eager = model._predict_once
        self._graphs: dict = {}
        self._pool = None
        self._stream = None
        self.captures = 0
        self.replays = 0
        self.eager_calls = 0

    def install(self) -> "StarSegGraph":
        """Route the model's plain inference forwards through ``__call__``."""
        def predict_once(x, profile=False, visualize=False, embed=None):
            if profile or visualize or embed is not None or self._key(x) is None:
                self.eager_calls += 1
                return self._eager(x, profile, visualize, embed)
            return self(x)
        self.model._predict_once = predict_once
        return self

    @staticmethod
    def _key(x):
        if (not isinstance(x, torch.Tensor) or x.device.type != "cuda" or not x.is_contiguous()
                or not torch.is_inference_mode_enabled()):
            return None
        return tuple(x.shape), x.dtype, str(x.device)

    def _features(self, x):
        """``_predict_once`` for every layer before the head; returns the head's input."""
        saved = []
        for layer in self.layers[:-1]:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.model.save else None)
        head = self.layers[-1]
        return saved[head.f] if isinstance(head.f, int) else [x if j == -1 else saved[j] for j in head.f]

    def _capture(self, x, key):
        with CAPTURE_LOCK, GRAPH_GATE.capture():
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
                self._stream = torch.cuda.Stream(device=x.device)
            static_input = x.clone()
            current = torch.cuda.current_stream(x.device)
            self._stream.wait_stream(current)
            with torch.cuda.stream(self._stream):
                self._features(static_input)
            self._stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self._stream):
                graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")
                try:
                    static_features = self._features(static_input)
                finally:
                    graph.capture_end()
            current.wait_stream(self._stream)
        self.captures += 1
        entry = (graph, static_input, static_features)
        self._graphs[key] = entry
        return entry

    def __call__(self, x):
        key = self._key(x)
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= self.max_graphs:
                self.eager_calls += 1
                return self._eager(x)
            entry = self._capture(x, key)
        graph, static_input, static_features = entry
        static_input.copy_(x)
        with GRAPH_GATE.replay():
            graph.replay()
        self.replays += 1
        # The head reads the static features on this stream before the next replay overwrites them.
        return self.layers[-1](static_features)

    def summary(self) -> dict:
        return {"captures": self.captures, "replays": self.replays,
                "eager_calls": self.eager_calls, "max_graphs": self.max_graphs}
