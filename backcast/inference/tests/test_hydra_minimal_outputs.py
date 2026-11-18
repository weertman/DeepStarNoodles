"""Numerical equivalence of raw-only and support/mouth inference outputs."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepstarnoodles_inference import hydra
from deepstarnoodles_inference.decode import activate_logits
from deepstarnoodles_inference.model import CHANNEL_NAMES
from deepstarnoodles_inference.runtime import (
    StarArmTrackerRegistry, body_center, body_center_from_maps,
)


@pytest.fixture
def predictor():
    fields = torch.randn(3, 18, 32, 40, generator=torch.Generator().manual_seed(37))
    fields[:, (0, 5)] = -20
    fields[0, 5, 12, 17] = 8  # Mouth wins over support centroid.
    fields[0, 0, 2:5, 3:7] = 3
    fields[1, 0, 4:8, 10:16] = 3  # No mouth: support centroid.
    # Crop 2 has no support or mouth: image center.
    spec = {"geometry": {"slice": [0, 10], "names": list(CHANNEL_NAMES)},
            "embedding": {"slice": [10, 18], "dim": 8}}

    class Model(torch.nn.Module):
        output_spec = spec

        def forward(self, images):
            assert len(images) == len(fields)
            return fields.to(device=images.device).clone()

    result = object.__new__(hydra.HydraPredictor)
    result.device = torch.device("cpu")
    result.config = hydra.HydraInferConfig(device="cpu", amp=False)
    result.output_spec = spec
    result.model = Model()
    return result


def test_raw_and_two_channel_maps_match_compatibility_outputs(predictor):
    images = torch.zeros(3, 3, 32, 40)
    raw, activated = predictor.predict_tensor_batch(images)
    raw_only = predictor.predict_tensor_batch_raw(images)
    original = torch.stack([torch.cat((activate_logits(sample[:10]), sample[10:]))
                            for sample in raw])
    torch.testing.assert_close(raw_only, raw, rtol=0, atol=0)
    torch.testing.assert_close(activated, original, rtol=0, atol=0)
    snapshot = raw_only.clone()
    maps = predictor.tracking_maps(raw_only)
    torch.testing.assert_close(maps, activated[:, (0, 5)], rtol=0, atol=0)
    torch.testing.assert_close(raw_only, snapshot, rtol=0, atol=0)
    assert maps.shape == (3, 2, 32, 40)
    assert maps.dtype == raw_only.dtype == torch.float32
    assert maps.untyped_storage().nbytes() == 3 * 2 * 32 * 40 * 4
    assert maps.untyped_storage().data_ptr() != raw_only.untyped_storage().data_ptr()


def test_raw_only_and_tracking_maps_do_not_call_full_activation(predictor, monkeypatch):
    def forbidden(_sample):
        raise AssertionError("full activations must not be constructed")

    monkeypatch.setattr(hydra, "activate_logits", forbidden)
    raw = predictor.predict_tensor_batch_raw(torch.zeros(3, 3, 32, 40))
    assert predictor.tracking_maps(raw).shape == (3, 2, 32, 40)


def test_body_center_and_tracker_inputs_match_for_multiple_crops(predictor):
    raw, activated = predictor.predict_tensor_batch(torch.zeros(3, 3, 32, 40))
    full = activated.numpy()
    maps = predictor.tracking_maps(raw).numpy()
    expected_centers = [[17, 12], [12.5, 5.5], [20, 16]]
    calls_full, calls_minimal = [], []

    class Tracker:
        def __init__(self, calls):
            self.calls = calls

        def step(self, frame, points, lengths, identities):
            self.calls.append((frame, points.copy(), lengths.copy(), identities.copy()))
            return list(range(len(points)))

    full_registry = StarArmTrackerRegistry(tracker_factory=lambda: Tracker(calls_full))
    minimal_registry = StarArmTrackerRegistry(tracker_factory=lambda: Tracker(calls_minimal))
    arms = [{"polyline": [[1, 2], [8, 10], [22, 26]], "identity": [0.1, 0.2]},
            {"polyline": [[18, 12], [9, 8], [3, 4]], "identity": [0.3, 0.4]}]
    for crop_index, (dense, compact, expected) in enumerate(zip(full, maps, expected_centers)):
        np.testing.assert_array_equal(body_center(dense), expected)
        np.testing.assert_array_equal(body_center_from_maps(*compact), expected)
        for frame in (0, 1):
            labels = full_registry.track(crop_index, frame, dense, arms)
            compact_labels = minimal_registry.track(crop_index, frame, compact, arms,
                                                     support_mouth_only=True)
            assert labels == compact_labels
        assert full_registry.track(crop_index, 2, dense, []) == []
        assert minimal_registry.track(crop_index, 2, compact, [], support_mouth_only=True) == []
    assert full_registry.star_track_ids == minimal_registry.star_track_ids == (0, 1, 2)
    assert len(calls_full) == len(calls_minimal) == 6
    for left, right in zip(calls_full, calls_minimal):
        assert left[0] == right[0]
        for a, b in zip(left[1:], right[1:]):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="compact tracking maps"):
        minimal_registry.track(0, 3, full[0], arms, support_mouth_only=True)


def test_raw_only_preserves_input_and_output_contract_validation(predictor):
    with pytest.raises(ValueError, match="images must be"):
        predictor.predict_tensor_batch_raw(torch.zeros(3, 32, 40))
    predictor.output_spec = {"geometry": {"slice": [0, 10], "names": ["unknown"]},
                             "embedding": {"slice": [10, 18], "dim": 8}}
    with pytest.raises(ValueError, match="channel contract"):
        predictor.predict_tensor_batch_raw(torch.zeros(3, 3, 32, 40))


def test_benchmark_and_staged_modes_are_explicit_opt_in():
    from deepstarnoodles_inference import benchmark, staged

    argv = ["--models", "models.json", "--index", "images.csv", "--output", "output",
            "--mode", "archive", "--device", "cpu", "--warmup", "0"]
    assert benchmark.parse_args(argv).hydra_output_mode == "full"
    assert staged.parse_args(argv).hydra_output_mode == "full"
    assert benchmark.parse_args(argv + ["--hydra-output-mode", "minimal"]).hydra_output_mode == "minimal"
    assert staged.parse_args(argv + ["--hydra-output-mode", "minimal"]).hydra_output_mode == "minimal"
