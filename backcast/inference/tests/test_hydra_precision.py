# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch
from torch import nn

from deepstarnoodles_inference import benchmark
from deepstarnoodles_inference.gru_compile import compile_hydra
from deepstarnoodles_inference.hydra import store_autocast_weights

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autocast")


class Block(nn.Module):
    """A ConvNeXt-shaped block: float16 convolution and linear, float32 norm and layer scale."""

    def __init__(self, channels=32):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.fc = nn.Linear(channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-2))
        self.group = nn.GroupNorm(8, channels)
        self.out = nn.Conv2d(channels, 4, 1)

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.gamma * self.fc(self.norm(y))
        return self.out(self.group(x + y.permute(0, 3, 1, 2)))


def test_store_autocast_weights_changes_only_conv_and_linear():
    model = Block()
    assert store_autocast_weights(model) == 3
    float16 = {name for name, value in model.named_parameters() if value.dtype == torch.float16}
    assert float16 == {"dwconv.weight", "dwconv.bias", "fc.weight", "fc.bias", "out.weight", "out.bias"}
    assert {value.dtype for name, value in model.named_parameters() if name not in float16} == {torch.float32}


@needs_cuda
def test_stored_weights_keep_autocast_outputs_identical():
    torch.manual_seed(0)
    model = Block().cuda().eval()
    images = torch.rand(1, 32, 64, 64, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda"):
        reference = model(images).clone()
        store_autocast_weights(model)
        candidate = model(images)
    assert candidate.dtype == reference.dtype and torch.equal(candidate, reference)


@needs_cuda
def test_compile_hydra_replaces_the_model_and_stays_close():
    torch.manual_seed(0)
    model = Block().cuda().eval()
    predictor = type("Predictor", (), {"model": model})()
    images = torch.rand(1, 32, 64, 64, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda"):
        reference = model(images).float().clone()
        compile_hydra(predictor)
        candidate = predictor.model(images).float()
    assert predictor.model is not model and predictor.model._orig_mod is model
    assert (candidate - reference).abs().max() < 1e-2


@pytest.mark.parametrize("extra", [["--hydra-fp16-weights", "--hydra-fp32"], ["--hydra-compile"],
                                   ["--hydra-compile", "--hydra-execution", "eager"]])
def test_flag_requirements(tmp_path, extra):
    base = ["--models", "m", "--index", "i", "--output", str(tmp_path), "--mode", "archive", "--device", "cuda:0",
            "--images", "1", "--detector-batch-size", "1", "--crop-batch-size", "1"]
    with pytest.raises(SystemExit):
        benchmark.parse_args(base + extra)
