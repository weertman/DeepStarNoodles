# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import torch

from deepstarnoodles.hydra import HydraInferConfig, HydraPredictor
from deepstarnoodles.noodle_tracer import (
    NoodleTraceParams,
    NoodleTracer,
    TraceState,
    trace_arms_noodle,
    trace_arms_noodle_batch,
)


def _straight_raw(size: int = 33) -> torch.Tensor:
    raw = torch.full((18, size, size), -10.0)
    y = size // 2
    raw[0, y - 2 : y + 3, 4 : size - 4] = 10.0
    raw[1, y, 4 : size - 4] = 10.0
    raw[2] = 1.0
    raw[8, y, 5] = 10.0
    raw[9, y, size - 6] = 10.0
    raw[10:] = 0.1
    return raw


def _walking_model() -> NoodleTracer:
    model = NoodleTracer(
        dense_channels=18,
        embedding_dim=8,
        hidden_size=24,
        pretrained_rgb=False,
        freeze_rgb=True,
        dense_mean=[0.0] * 18,
        dense_std=[1.0] * 18,
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.direction_head.bias.copy_(torch.tensor([-1.0, 0.0]))
        model.step_head.bias.fill_(0.0)
        model.state_head.bias[int(TraceState.CONTINUE)] = 10.0
    return model


def _assert_same_arms(expected: list[dict], observed: list[dict]) -> None:
    assert len(expected) == len(observed)
    for left, right in zip(expected, observed):
        assert left["stop_reason"] == right["stop_reason"]
        assert left["steps"] == right["steps"]
        assert left["tip_score"] == right["tip_score"]
        assert left["base_score"] == right["base_score"]
        assert left["state_confidence"] == right["state_confidence"]
        assert np.array_equal(left["identity"], right["identity"])
        assert np.array_equal(left["polyline"], right["polyline"])


def test_recurrent_batch_matches_independent_single_calls() -> None:
    model = _walking_model()
    first = _straight_raw()
    second = _straight_raw()
    second[9, second.shape[-1] // 2 + 5, second.shape[-1] - 8] = 9.0
    empty = torch.full_like(first, -10.0)
    raws = torch.stack((first, second, empty))
    images = torch.zeros((3, 3, first.shape[-2], first.shape[-1]))
    params = NoodleTraceParams(
        max_steps=16,
        suppress_duplicates=False,
    )

    expected = [
        trace_arms_noodle(raws[index], images[index], model, params)
        for index in range(len(raws))
    ]
    observed = trace_arms_noodle_batch(raws, images, model, params)

    assert len(observed) == 3
    for left, right in zip(expected, observed):
        _assert_same_arms(left, right)
    assert observed[2] == []


def test_recurrent_batch_accepts_channel_last_rgb() -> None:
    raw = _straight_raw()
    model = _walking_model()
    params = NoodleTraceParams(max_steps=4, suppress_duplicates=False)
    channel_first = torch.zeros((1, 3, 33, 33))
    channel_last = channel_first.permute(0, 2, 3, 1)
    expected = trace_arms_noodle_batch(raw[None], channel_first, model, params)
    observed = trace_arms_noodle_batch(raw[None], channel_last, model, params)
    _assert_same_arms(expected[0], observed[0])


class _FakeDenseModel(torch.nn.Module):
    output_spec = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        base = images.mean(dim=1, keepdim=True)
        return torch.cat([base + float(index) for index in range(18)], dim=1)


def _fake_predictor(size: int = 32) -> HydraPredictor:
    predictor = object.__new__(HydraPredictor)
    predictor.config = HydraInferConfig(
        image_size=size,
        device="cpu",
        amp=False,
    )
    predictor.device = torch.device("cpu")
    predictor.output_spec = {}
    predictor.model = _FakeDenseModel().eval()
    predictor.checkpoint_image_size = size
    return predictor


def test_hydra_batch_matches_single_calls() -> None:
    predictor = _fake_predictor()
    crops = [
        np.full((20, 30, 3), 20, dtype=np.uint8),
        np.full((40, 25, 3), 100, dtype=np.uint8),
        np.full((32, 32, 3), 220, dtype=np.uint8),
    ]
    raw_batch, activated_batch = predictor.predict_raw_batch(crops)
    assert raw_batch.shape == (3, 18, 32, 32)
    assert activated_batch.shape == (3, 18, 32, 32)
    for index, crop in enumerate(crops):
        raw, activated = predictor.predict_raw(crop)
        assert torch.equal(raw, raw_batch[index])
        assert torch.equal(activated, activated_batch[index])
