# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepstarnoodles_inference import benchmark

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def detections(model, frames):
    out = []
    for frame in frames:
        result = model.predict([frame], imgsz=256, conf=0.0001, iou=0.7, device="cuda:0", verbose=False,
                               half=False, rect=False)[0]
        out.append((result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(), result.masks.data.cpu().numpy()))
    return out


@cuda
def test_starseg_graph_reproduces_eager_detections_and_masks():
    from ultralytics import YOLO
    from deepstarnoodles_inference.starseg_graph import StarSegGraph

    torch.manual_seed(0)
    model = YOLO("yolo26n-seg.yaml")
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, shape, dtype=np.uint8) for shape in ((240, 320, 3), (300, 200, 3), (256, 256, 3))]
    expected = detections(model, frames)
    graph = StarSegGraph(model.model).install()
    observed = detections(model, frames)
    assert sum(len(item[1]) for item in expected) > 0
    for want, got in zip(expected, observed):
        for a, b in zip(want, got):
            np.testing.assert_array_equal(a, b)
    assert graph.summary() == {"captures": 1, "replays": len(frames), "eager_calls": 0, "max_graphs": 2}
    # Calls outside inference mode run the original eager forward.
    with torch.no_grad():
        model.model(torch.zeros(1, 3, 256, 256, device="cuda:0"))
    assert graph.summary()["eager_calls"] == 1


def test_starseg_graph_validation():
    from deepstarnoodles_inference.starseg_graph import StarSegGraph

    class Layer(torch.nn.Module):
        f = -1

    model = torch.nn.Module()
    model.model = torch.nn.ModuleList([Layer()])
    with pytest.raises(ValueError):
        StarSegGraph(model)
    base = ["--models", "m", "--index", "i.csv", "--output", "o", "--mode", "archive"]
    assert benchmark.parse_args(base).starseg_execution == "eager"
    assert benchmark.parse_args(base + ["--starseg-execution", "graph"]).starseg_execution == "graph"
    with pytest.raises(SystemExit):
        benchmark.parse_args(base + ["--starseg-execution", "graph", "--device", "cpu"])
    with pytest.raises(SystemExit):
        benchmark.parse_args(base + ["--gru-graph-first-step"])
    assert benchmark.parse_args(base + ["--gru-execution", "graph", "--gru-graph-first-step"]).gru_graph_first_step
