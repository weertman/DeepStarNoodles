# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from deepstarnoodles_inference import benchmark
from deepstarnoodles_inference.crop import CropConfig, _crop_geometry, _crop_result
from deepstarnoodles_inference.crop_gpu import PRECISION_BITS, bilinear_coefficients
from deepstarnoodles_inference.hydra import HydraPredictor

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def numpy_resample(image, size):
    """Both fixed-point passes of crop_gpu in numpy, for CPU-only checks."""
    first, _counts, fixed = bilinear_coefficients(image.shape[0], size)
    taps = first[:, None] + np.arange(fixed.shape[1])[None, :]
    weights = np.where(taps < image.shape[0], fixed, 0).astype(np.int64)
    taps = np.clip(taps, 0, image.shape[0] - 1)
    horizontal = (np.take(image.astype(np.int64), taps, axis=1) * weights[None, :, :, None]).sum(axis=2)
    horizontal = np.clip((horizontal + (1 << (PRECISION_BITS - 1))) >> PRECISION_BITS, 0, 255)
    vertical = (np.take(horizontal, taps, axis=0) * weights[:, :, None, None]).sum(axis=1)
    return np.clip((vertical + (1 << (PRECISION_BITS - 1))) >> PRECISION_BITS, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("side", [16, 17, 100, 255, 511, 513, 1000, 1537])
def test_fixed_point_resample_matches_pillow_bilinear(side):
    rng = np.random.default_rng(side)
    image = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
    image[: side // 3, side // 2:] = (255, 0, 7)
    expected = np.asarray(Image.fromarray(image, mode="RGB").resize((64, 64), Image.Resampling.BILINEAR))
    np.testing.assert_array_equal(numpy_resample(image, 64), expected)


def test_coefficients_are_normalized_fixed_point():
    first, counts, fixed = bilinear_coefficients(1537, 512)
    assert first.min() == 0 and (first + counts).max() == 1537
    sums = fixed.sum(axis=1)
    assert np.all(np.abs(sums - (1 << PRECISION_BITS)) <= fixed.shape[1])
    with pytest.raises(ValueError):
        bilinear_coefficients(0, 512)


def test_cli_validates_gpu_crops_and_blocking_sync():
    base = ["--models", "m", "--index", "i.csv", "--output", "o", "--mode", "archive"]
    parsed = benchmark.parse_args(base + ["--gpu-crops", "--cuda-blocking-sync"])
    assert parsed.gpu_crops is True and parsed.cuda_blocking_sync is True
    assert benchmark.parse_args(base).gpu_crops is False
    assert benchmark.parse_args(base).cuda_blocking_sync is False
    for extra in (["--gpu-crops", "--device", "cpu"], ["--gpu-crops", "--archive-frame-window", "4"],
                  ["--gpu-crops", "--crop-streams", "2"], ["--cuda-blocking-sync", "--device", "cpu"]):
        with pytest.raises(SystemExit):
            benchmark.parse_args(base + extra)
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--models", "m", "--index", "i.csv", "--output", "o", "--mode", "live-replay",
                              "--gpu-crops"])


def detection(x1, y1, x2, y2):
    return {"xyxy": (x1, y1, x2, y2), "center": ((x1 + x2) / 2, (y1 + y2) / 2), "confidence": 0.9,
            "polygon": None, "star_track_id": None}


def test_crop_geometry_matches_crop_result_without_pixels():
    frame = np.zeros((40, 60, 3), np.uint8)
    item = detection(-3.2, 5.5, 20.7, 31.0)
    full, bare = _crop_result(frame, item, context_scale=2.0), _crop_geometry(frame, item, context_scale=2.0)
    assert bare.crop_rgb is None
    assert (bare.crop_left, bare.crop_top, bare.crop_side) == (full.crop_left, full.crop_top, full.crop_side)
    assert (bare.bbox_xyxy, bare.center, bare.confidence) == (full.bbox_xyxy, full.center, full.confidence)


@cuda
@pytest.mark.parametrize("size", [64, 512])
def test_gpu_crops_equal_hydra_cpu_preprocessing(size):
    rng = np.random.default_rng(size)
    frame = rng.integers(0, 256, (700, 900, 3), dtype=np.uint8)
    frame[100:300, 200:260] = (3, 250, 128)
    # Interior, edge-padded, corner-padded, large padded, exact-size and upscaled crops.
    items = [detection(300, 200, 500, 380), detection(-40, 600, 120, 690), detection(820, -30, 905, 60),
             detection(50, 50, 450, 400), detection(100, 100, 100 + size / 2, 100 + size / 2),
             detection(450, 450, 470, 465)]
    pipeline = object.__new__(benchmark.Cascade)
    pipeline.args = SimpleNamespace(channels_last=False)
    pipeline.device = torch.device("cuda:0")
    pipeline.crop_config = CropConfig(device="cuda:0")
    pipeline.live_cropper = None
    hydra = SimpleNamespace(image_size=size)
    pipeline.hydra = SimpleNamespace(image_size=size,
                                     preprocess_batch=lambda crops: HydraPredictor.preprocess_batch(hydra, crops))
    group = [(0, j, item) for j, item in enumerate(items)]
    expected = pipeline.preprocess_crops(pipeline.crop_group([frame], group))
    from deepstarnoodles_inference.crop_gpu import GpuCropPreparer
    pipeline.gpu_crops = GpuCropPreparer(size, "cuda:0")
    crops = pipeline.crop_group([frame], group)
    assert all(crop.crop_rgb is None for _i, _j, crop in crops)
    sides = {int(crop.crop_side) for _i, _j, crop in crops}
    assert size in sides and min(sides) < size < max(sides)
    actual = pipeline.preprocess_crops(crops, pipeline.upload_frames([frame], group))
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.cpu(), expected.cpu())
    assert pipeline.gpu_crops.summary()["crops"] == len(items)


@cuda
def test_blocking_sync_sleeps_through_waits():
    # A fresh process: the flag stays on the primary context for its lifetime.
    script = """
import resource, time, torch
from deepstarnoodles_inference import cuda_sched
state = cuda_sched.request_blocking_sync("cuda:0")
assert state["primary_context_flags"] & 7 == 4, state
def cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime
for wait in ("synchronize", "event"):
    torch.cuda.synchronize()
    started, used = time.perf_counter(), cpu()
    torch.cuda._sleep(int(5e8))
    if wait == "event":
        event = cuda_sched.event()
        event.record()
        event.synchronize()
    else:
        torch.cuda.synchronize()
    wall, spent = time.perf_counter() - started, cpu() - used
    assert wall > 0.05 and spent < 0.5 * wall, (wait, wall, spent)
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stderr[-2000:]


def test_blocking_sync_requires_cuda_device():
    from deepstarnoodles_inference import cuda_sched
    with pytest.raises(ValueError):
        cuda_sched.request_blocking_sync("cpu")
