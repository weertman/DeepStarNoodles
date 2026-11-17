# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Explicit decoder selection through the real resident frame orchestration."""
from dataclasses import replace

import pytest
import torch

from deepstarnoodles_inference import benchmark, live, staged
from deepstarnoodles_inference.hydra import HydraInferConfig, HydraPredictor
from deepstarnoodles_inference.model import CHANNEL_NAMES
from deepstarnoodles_inference.noodle_tracer import trace_arms_noodle_batch
from deepstarnoodles_inference.runtime import StarArmTrackerRegistry
from test_fixed_slots import ScriptedTracer, fields_with_seeds, params
from test_live import live_pipeline, pixels


@pytest.mark.parametrize("mode", ["full", "minimal"])
def test_live_fixed_and_dynamic_decode_match_through_crop_and_tracking(monkeypatch, mode):
    raw, _ = fields_with_seeds()
    spec = {"geometry": {"slice": [0, 10], "names": list(CHANNEL_NAMES)},
            "embedding": {"slice": [10, 18], "dim": 8}}

    class Model(torch.nn.Module):
        output_spec = spec

        def forward(self, images):
            return raw[:1].repeat(len(images), 1, 1, 1)

    outputs, batch_counts = [], []
    for execution in ("dynamic", "fixed"):
        pipeline, _ = live_pipeline(monkeypatch)
        # Both paths now use the original neural rollout, not the helper's mock.
        monkeypatch.setattr(benchmark, "trace_arms_noodle_batch", trace_arms_noodle_batch)
        pipeline.config = replace(pipeline.config, gru_execution=execution,
                                  gru_check_interval=4, hydra_output_mode=mode)
        pipeline.cascade.args.gru_execution = execution
        pipeline.cascade.args.gru_check_interval = 4
        pipeline.cascade.args.hydra_output_mode = mode
        predictor = object.__new__(HydraPredictor)
        predictor.config = HydraInferConfig(device="cpu", amp=False, image_size=65)
        predictor.device = torch.device("cpu")
        predictor.model, predictor.output_spec = Model(), spec
        pipeline.cascade.hydra = predictor
        pipeline.cascade.tracer = ScriptedTracer()
        pipeline.cascade.trace_params = params()
        pipeline.cascade.registry = StarArmTrackerRegistry()
        records = [pipeline.predict(pixels(count), source_index=index)
                   for count, index in ((3, 0), (0, 5), (1, 6))]
        outputs.append(records)
        batch_counts.append(pipeline.cascade.tracer.batch_sizes)
    for dynamic, fixed in zip(*outputs):
        assert dynamic["stars"] == fixed["stars"]
        assert dynamic["processed_index"] == fixed["processed_index"]
    assert [len(row["stars"]) for row in outputs[1]] == [3, 0, 1]
    assert all(len(star["arms"]) == 2 for row in outputs[1] for star in row["stars"])
    # Fixed executes full seed-slot batches even after individual terminations.
    assert 1 in batch_counts[0] and set(batch_counts[1]) == {3, 6}


def test_all_cli_parsers_select_dynamic_by_default_and_forward_fixed():
    shared = ["--models", "models", "--output", "out", "--device", "cpu"]
    indexed = shared + ["--index", "images.csv", "--mode", "archive", "--warmup", "0"]
    for parse, argv in ((benchmark.parse_args, indexed), (staged.parse_args, indexed),
                        (lambda args: live.parse_args(args)[1], shared + ["--camera", "0"])):
        default = parse(argv)
        assert default.gru_execution == "dynamic" and default.gru_check_interval == 8
        fixed = parse(argv + ["--gru-execution", "fixed", "--gru-check-interval", "128"])
        assert fixed.gru_execution == "fixed" and fixed.gru_check_interval == 128


@pytest.mark.parametrize("interval", ["0", "-1", "129", "1.5"])
def test_cli_rejects_unbounded_or_noninteger_check_interval(interval):
    args = ["--models", "models", "--output", "out", "--device", "cpu"]
    for parse, argv in ((benchmark.parse_args, args + ["--index", "index", "--mode", "archive"]),
                        (staged.parse_args, args + ["--index", "index", "--mode", "archive", "--warmup", "0"]),
                        (live.parse_args, args + ["--camera", "0"])):
        with pytest.raises(SystemExit):
            parse(argv + ["--gru-check-interval", interval])


@pytest.mark.parametrize("values", [{"gru_execution": "unknown"}, {"gru_check_interval": True},
                                    {"gru_check_interval": 0}, {"gru_check_interval": 129},
                                    {"gru_check_interval": 1.5}])
def test_live_config_rejects_invalid_decoder_settings(values):
    with pytest.raises(ValueError, match="gru_"):
        live.LiveConfig(**values)
