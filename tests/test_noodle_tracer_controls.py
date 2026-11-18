"""Mechanistic controls must remain effective in training, reload and decoding."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import deepstarnoodles.noodle_tracer as tracer
from train_noodle_tracer import validate_normalization_source


def model_and_inputs(context_mode="single", **controls):
    torch.manual_seed(19)
    model = tracer.NoodleTracer(
        hidden_size=16, pretrained_rgb=False, context_mode=context_mode,
        dense_mean=[0.3] * 18, dense_std=[1.7] * 18, **controls,
    ).eval()
    leading = (2,) if context_mode == "single" else (2, 2)
    inputs = [
        torch.randn((*leading, 18, 33, 33)),
        torch.randn((*leading, 16, 9, 9)),
        torch.randn((*leading, 24, 9, 9)),
        torch.randn((2, 2)), torch.randn((2, 8)), torch.randn((2, 16)),
    ]
    return model, inputs


@pytest.mark.parametrize("context_mode", ["single", "multiscale_dilated"])
def test_no_rgb_ignores_pixels_features_and_projection_bias(context_mode, monkeypatch):
    model, inputs = model_and_inputs(context_mode, rgb_features=False)
    expected = model.forward_step(*inputs)
    inputs[1] = torch.randn_like(inputs[1]) * 100
    inputs[2] = torch.randn_like(inputs[2]) * 100
    with torch.no_grad():
        model.rgb_fine_projection.bias.fill_(1000)
        model.rgb_coarse_projection.bias.fill_(-1000)
    observed = model.forward_step(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(expected, observed))

    def forbidden(*args, **kwargs):
        raise AssertionError("RGB ablation must not execute its encoder")

    monkeypatch.setattr(model.rgb_encoder, "forward", forbidden)
    pyramid = model.encode_rgb(torch.rand((2, 3, 40, 40)))
    patches = model.sample_rgb(pyramid, torch.rand((2, 2)), torch.tensor([0, 1]))
    assert all(torch.count_nonzero(patch) == 0 for patch in patches)


@pytest.mark.parametrize("context_mode", ["single", "multiscale_dilated"])
def test_no_embedding_blocks_raw_channels_similarity_and_conditioning(context_mode):
    model, inputs = model_and_inputs(context_mode, embedding_features=False)
    inputs[0].requires_grad_()
    inputs[4].requires_grad_()
    expected = model.forward_step(*inputs)
    changed = [value.detach().clone() for value in inputs]
    changed[0][..., 10:18, :, :] = torch.randn_like(changed[0][..., 10:18, :, :]) * 100
    changed[4] = torch.randn_like(changed[4]) * 100
    observed = model.forward_step(*changed)
    assert all(torch.equal(a, b) for a, b in zip(expected, observed))
    expected[1].sum().backward()
    assert torch.count_nonzero(inputs[0].grad[..., 10:18, :, :]) == 0
    assert torch.count_nonzero(inputs[0].grad[..., :10, :, :]) > 0
    assert inputs[4].grad is None or torch.count_nonzero(inputs[4].grad) == 0


@pytest.mark.parametrize("context_mode", ["single", "multiscale_dilated"])
def test_state_reset_removes_attention_history_as_well_as_gru_history(context_mode):
    model, inputs = model_and_inputs(context_mode, reset_recurrent_state=True)
    inputs[5].requires_grad_()
    expected = model.forward_step(*inputs)
    changed = inputs[:-1] + [torch.randn_like(inputs[5]) * 100]
    observed = model.forward_step(*changed)
    assert all(torch.equal(a, b) for a, b in zip(expected, observed))
    expected[1].sum().backward()
    assert inputs[5].grad is None or torch.count_nonzero(inputs[5].grad) == 0


@pytest.mark.parametrize("width", [64, 128])
def test_controls_and_requested_width_survive_checkpoint_reload(width, tmp_path):
    model = tracer.NoodleTracer(hidden_size=width, pretrained_rgb=False,
                                rgb_features=False, embedding_features=False,
                                reset_recurrent_state=True)
    path = tmp_path / "control.pt"
    tracer.save_noodle_tracer_checkpoint(str(path), model,
                                         trace_params=tracer.NoodleTraceParams(), metadata={})
    loaded, _, _, _ = tracer.load_noodle_tracer_checkpoint(str(path))
    assert loaded.hidden_size == width
    assert not loaded.rgb_features and not loaded.embedding_features
    assert loaded.reset_recurrent_state
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())


def test_no_embedding_disables_identity_postprocessing_in_real_decode(monkeypatch):
    model = tracer.NoodleTracer(hidden_size=16, pretrained_rgb=False,
                                rgb_features=False, embedding_features=False).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.direction_head.bias[0] = -1
        model.state_head.bias[int(tracer.TraceState.CONTINUE)] = 10
    raw = torch.full((18, 33, 33), -10.0)
    raw[0, 14:19, 4:29] = 10
    raw[8, 16, 5] = 10
    raw[9, 16, 27] = 10
    raw[10:] = 3

    def forbidden(*args, **kwargs):
        raise AssertionError("identity postprocessing leaked into the no-embedding control")

    monkeypatch.setattr(tracer, "_suppress_duplicates_and_fragments", forbidden)
    arms = tracer.trace_arms_noodle(raw, torch.zeros((3, 33, 33)), model)
    assert arms
    assert all(np.count_nonzero(arm["identity"]) == 0 for arm in arms)


def test_normalization_rejects_another_hydra_or_nontraining_statistics():
    validate_normalization_source({"frozen_checkpoint_sha256": "parent", "source_split": "train"}, "parent")
    with pytest.raises(ValueError, match="different frozen Hydra"):
        validate_normalization_source({"frozen_checkpoint_sha256": "released"}, "new-parent")
    with pytest.raises(ValueError, match="training pixels"):
        validate_normalization_source({"source_split": "validation"}, "parent")
