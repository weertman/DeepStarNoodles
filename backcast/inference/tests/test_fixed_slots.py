# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Synthetic termination and numerical parity for experimental fixed-slot GRU."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from deepstarnoodles_inference.noodle_tracer import (
    NoodleTraceParams, NoodleTracer, RGBPyramid, TraceState, trace_arms_noodle_batch,
)
from deepstarnoodles_inference.noodle_tracer_fixed import trace_arms_noodle_batch_fixed_slots


def assert_outputs_close(reference, candidate, atol=0.0):
    assert len(reference) == len(candidate)
    for ref_arms, new_arms in zip(reference, candidate):
        assert len(ref_arms) == len(new_arms)
        for old, new in zip(ref_arms, new_arms):
            assert old.keys() == new.keys()
            for key in old:
                if isinstance(old[key], np.ndarray):
                    np.testing.assert_allclose(new[key], old[key], rtol=0, atol=atol, err_msg=key)
                elif isinstance(old[key], float):
                    assert abs(new[key] - old[key]) <= atol, (key, new[key], old[key])
                else:
                    assert new[key] == old[key], key


class ScriptedTracer(torch.nn.Module):
    """Use seed identities and per-walker hidden state to control termination."""
    dense_channels = 18
    embedding_dim = 8
    dense_patch_size = 33
    rgb_feature_patch_size = 9
    context_mode = 'single'
    context_dilation = 2
    hidden_size = 128
    min_step_px = 0.5
    max_step_px = 6.0
    embedding_features = True

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.batch_sizes = []

    def initial_state(self, count, device=None):
        return torch.zeros(count, self.hidden_size, device=device)

    def encode_rgb(self, image):
        return RGBPyramid(image.new_zeros((len(image), 16, 1, 1)),
                          image.new_zeros((len(image), 24, 1, 1)), tuple(image.shape[-2:]))

    def sample_dense(self, fields, positions, image_indices):
        # Also encode source-image identity in sampled patches to expose wrong
        # fixed ranges, source mapping, or output order.
        return fields[:, :18, :33, :33].expand(len(positions), -1, -1, -1)

    def sample_rgb(self, pyramid, positions, image_indices):
        return positions.new_zeros((len(positions), 16, 9, 9)), positions.new_zeros((len(positions), 24, 9, 9))

    def forward_step(self, dense, fine, coarse, previous, seeds, hidden, *, identity_scale):
        self.batch_sizes.append(len(seeds))
        code = seeds[:, 0].round().long()
        step = hidden[:, 0].long()
        dx = torch.tensor([1., 1., .75, 10., 3., 1., .25, 1.5], device=seeds.device)[code]
        dx = torch.where((code == 5) & (step % 2 == 1), -dx, dx)
        states = torch.full_like(code, int(TraceState.CONTINUE))
        states = torch.where((code == 0) & (step >= 2), int(TraceState.TERMINATE_BASE), states)
        states = torch.where(code == 1, int(TraceState.INVALID_SEED), states)
        states = torch.where((code == 2) & (step >= 1), int(TraceState.TERMINATE_LOST), states)
        states = torch.where((code == 7) & (step >= 1), int(TraceState.INVALID_SEED), states)
        logits = seeds.new_full((len(seeds), 4), -5.)
        logits.scatter_(1, states[:, None], (5. + step.float() / 4)[:, None])
        new_hidden = hidden.clone()
        new_hidden[:, 0] += 1
        # Batch source marker is carried in an otherwise unused recurrent slot.
        new_hidden[:, 1] = dense[:, 1, 0, 0]
        return torch.stack((dx, torch.zeros_like(dx)), dim=1), logits, new_hidden


def fields_with_seeds():
    raw = torch.full((3, 18, 65, 65), -10.)
    raw[:, 0] = 10.
    raw[:, 10:] = 0.
    locations = [(0, 0, 8, 8), (0, 1, 24, 8), (0, 2, 40, 8),
                 (2, 3, 40, 24), (2, 4, 8, 24), (2, 5, 24, 24),
                 (2, 6, 8, 40), (2, 7, 24, 40)]
    for image, code, x, y in locations:
        raw[image, 9, y, x] = 9. - code / 10
        raw[image, 10:, y-2:y+3, x-2:x+3] = float(code)
        raw[image, 8, y, max(0, x + 2)] = 8.
    raw[0, 1] = 3.
    raw[2, 1] = 7.
    return raw, torch.zeros(3, 3, 65, 65)


def params(**kwargs):
    return NoodleTraceParams(max_steps=6, max_path_px=200, min_path_px=0,
                            smooth_window=1, resample_spacing=.5, loop_history=2,
                            loop_radius=.1, suppress_duplicates=False, emit_lost=True, **kwargs)


@pytest.mark.parametrize('interval', [1, 4, 8, 128])
def test_mixed_termination_and_empty_image_exact_parity(interval):
    raw, images = fields_with_seeds()
    model = ScriptedTracer()
    reference = trace_arms_noodle_batch(raw, images, model, params())
    reference_batches = list(model.batch_sizes)
    model.batch_sizes.clear()
    diagnostics = {}
    result = trace_arms_noodle_batch_fixed_slots(raw, images, model, params(),
                                                check_interval=interval, diagnostics=diagnostics)
    assert_outputs_close(reference, result)
    assert result[1] == []
    assert {arm['stop_reason'] for arms in result for arm in arms} == {'base', 'lost', 'bounds', 'loop', 'max_steps'}
    assert len(set(reference_batches)) > 1
    assert set(model.batch_sizes) == {8}
    assert diagnostics['seeds'] == 8


@pytest.mark.parametrize('override', [
    dict(max_path_px=5), dict(emit_lost=False), dict(invalid_seed_threshold=.99999),
    dict(base_state_threshold=0.), dict(lost_state_threshold=0.),
    dict(base_state_threshold=0., lost_state_threshold=0.),
    dict(base_state_threshold=0., lost_state_threshold=0., invalid_seed_threshold=0.),
    dict(support_gated_peaks=True), dict(suppress_duplicates=True), dict(max_tips=2),
])
def test_thresholds_filters_and_stop_precedence(override):
    raw, images = fields_with_seeds()
    model = ScriptedTracer()
    config = replace(params(), **override)
    expected = trace_arms_noodle_batch(raw, images, model, config)
    observed = trace_arms_noodle_batch_fixed_slots(raw, images, model, config, check_interval=4)
    assert_outputs_close(expected, observed)
    if override == dict(max_path_px=5):
        assert any(arm['stop_reason'] == 'max_length' for arms in observed for arm in arms)


def test_empty_seeds_skip_forward_and_encoder(monkeypatch):
    raw = torch.full((2, 18, 33, 33), -20.)
    images = torch.zeros(2, 3, 33, 33)
    model = ScriptedTracer()
    monkeypatch.setattr(model, 'encode_rgb', lambda image: pytest.fail('encoder should be skipped'))
    assert trace_arms_noodle_batch_fixed_slots(raw, images, model, params()) == [[], []]
    assert model.batch_sizes == []


def test_fixed_slots_never_call_torch_nonzero(monkeypatch):
    raw, images = fields_with_seeds()
    monkeypatch.setattr(torch, 'nonzero', lambda *args, **kwargs: pytest.fail('dynamic compaction'))
    assert len(trace_arms_noodle_batch_fixed_slots(raw, images, ScriptedTracer(), params())) == 3


@pytest.mark.parametrize('invalid', [0, -1, True, 1.5, 129])
def test_invalid_check_interval_rejected(invalid):
    raw, images = fields_with_seeds()
    with pytest.raises(ValueError, match='positive integer'):
        trace_arms_noodle_batch_fixed_slots(raw, images, ScriptedTracer(), params(), check_interval=invalid)


@pytest.mark.parametrize('context_mode', ['single', 'multiscale_dilated'])
def test_actual_model_patch_sampling_and_numerical_parity(context_mode):
    torch.manual_seed(5)
    torch.set_num_threads(2)
    model = NoodleTracer(pretrained_rgb=False, hidden_size=128, context_mode=context_mode).eval()
    # Retain real RGB/dense stems, attention and recurrent weights, but guarantee
    # some continuing walkers so this test exercises multiple recurrent steps.
    with torch.no_grad():
        model.state_head.weight.zero_()
        model.state_head.bias.zero_()
        model.state_head.bias[int(TraceState.CONTINUE)] = 10.
    raw, images = fields_with_seeds()
    images = torch.rand_like(images)
    config = replace(params(), context_mode=context_mode, max_steps=4)
    expected = trace_arms_noodle_batch(raw, images, model, config)
    observed = trace_arms_noodle_batch_fixed_slots(raw, images, model, config, check_interval=4)
    assert sum(map(len, expected)) > 0
    assert_outputs_close(expected, observed, atol=2e-5)


@pytest.mark.parametrize('image_index', [0, 1, 2])
@pytest.mark.parametrize('numpy_hwc', [False, True])
def test_single_image_shape_and_numpy_input_parity(image_index, numpy_hwc):
    raw, images = fields_with_seeds()
    raw, images = raw[image_index], images[image_index]
    if numpy_hwc:
        raw = raw.numpy()
        images = images.permute(1, 2, 0).numpy()
    model = ScriptedTracer()
    expected = trace_arms_noodle_batch(raw, images, model, params())
    observed = trace_arms_noodle_batch_fixed_slots(raw, images, model, params(), check_interval=4)
    assert_outputs_close(expected, observed)
