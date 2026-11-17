"""Hydra-only microbatching preserves ordered fields and GRU crop groups."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepstarnoodles_inference import benchmark, hydra, live, staged
from deepstarnoodles_inference.model import CHANNEL_NAMES
from test_live import live_pipeline, pixels
from test_staged import arguments, fake_models, frames, model_paths, spool_state


def predictor():
    spec = {"geometry": {"slice": [0, 10], "names": list(CHANNEL_NAMES)},
            "embedding": {"slice": [10, 18], "dim": 8}}

    class Model(torch.nn.Module):
        output_spec = spec

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, images):
            self.calls.append(len(images))
            # Distinct per-crop/channel values expose permutations or loss of a
            # remainder without asserting GPU batch-dependent arithmetic parity.
            return torch.cat([images[:, :1] + channel / 8 for channel in range(18)], dim=1)

    result = object.__new__(hydra.HydraPredictor)
    result.device = torch.device("cpu")
    result.config = hydra.HydraInferConfig(device="cpu", amp=False)
    result.output_spec, result.model = spec, Model()
    return result


@pytest.mark.parametrize("batch_size,expected_calls", [(None, [5]), (1, [1]*5), (2, [2, 2, 1]), (99, [5])])
@pytest.mark.parametrize("minimal,tracking", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("channels_last", [False, True])
def test_raw_and_activation_order_remainders_and_layout(batch_size, expected_calls, minimal, tracking, channels_last):
    model = predictor()
    images = torch.arange(5 * 3 * 7 * 9, dtype=torch.float32).reshape(5, 3, 7, 9) / 1000
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    raw_reference, activated_reference = model.predict_tensor_batch(images)
    model.model.calls.clear()
    raw, activated = hydra.predict_hydra_crops(model, images, batch_size=batch_size,
                                             minimal=minimal, tracking=tracking)
    assert model.model.calls == expected_calls
    torch.testing.assert_close(raw, raw_reference, rtol=0, atol=0)
    if minimal and not tracking:
        assert activated is None
    else:
        # Compare each mode against its existing API, isolating batching from
        # CPU sigmoid vectorization differences between full and compact maps.
        expected = model.tracking_maps(raw_reference) if minimal else activated_reference
        torch.testing.assert_close(activated, expected, rtol=0, atol=0)
    assert raw.dtype == torch.float32 and not raw.requires_grad


@pytest.mark.parametrize("raw_layout", [torch.contiguous_format, torch.channels_last])
@pytest.mark.parametrize("activated_layout", [torch.contiguous_format, torch.channels_last])
def test_chunk_buffers_preserve_each_output_memory_format(raw_layout, activated_layout):
    images = torch.arange(5 * 3 * 7 * 9, dtype=torch.float32).reshape(5, 3, 7, 9)

    def forward(part):
        raw = part[:, :1].expand(-1, 18, -1, -1).contiguous(memory_format=raw_layout)
        activated = part[:, :1].expand(-1, 2, -1, -1).contiguous(memory_format=activated_layout)
        return raw, activated

    expected = forward(images)
    result = hydra.predict_hydra_crops(SimpleNamespace(predict_tensor_batch=forward), images, batch_size=2)
    for actual, reference, layout in zip(result, expected, (raw_layout, activated_layout)):
        assert actual.is_contiguous(memory_format=layout)
        assert actual.stride() == reference.stride()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_default_and_nonbinding_cap_preserve_single_call_and_output_storage():
    raw, activated = torch.rand(3, 18, 2, 2), torch.rand(3, 18, 2, 2)
    images = torch.rand(3, 3, 2, 2)
    calls = []
    def forward(part):
        assert part is images
        calls.append(len(part))
        return raw, activated
    model = SimpleNamespace(predict_tensor_batch=forward)
    for cap in (None, 3, 4):
        result = hydra.predict_hydra_crops(model, images, batch_size=cap)
        assert result[0] is raw and result[1] is activated
    assert calls == [3, 3, 3]


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1"])
def test_invalid_caps_fail_before_inference(value):
    with pytest.raises(ValueError, match="hydra_batch_size"):
        hydra.predict_hydra_crops(object(), torch.zeros(3, 3, 2, 2), batch_size=value)
    with pytest.raises(ValueError, match="hydra_batch_size"):
        live.LiveConfig(hydra_batch_size=value)


@pytest.mark.parametrize("shape", [(0, 3, 2, 2), (3, 2, 2), (2, 4, 2, 2)])
def test_helper_rejects_empty_or_malformed_input(shape):
    with pytest.raises(ValueError, match="nonempty"):
        hydra.predict_hydra_crops(object(), torch.zeros(shape), batch_size=1)


def cli_cases():
    shared = ["--models", "models", "--output", "out", "--device", "cpu"]
    indexed = shared + ["--index", "images.csv", "--mode", "archive", "--warmup", "0"]
    return ((benchmark.parse_args, indexed), (staged.parse_args, indexed),
            (lambda argv: live.parse_args(argv)[1], shared + ["--camera", "0"]))


def test_all_parsers_default_to_coupled_batches_and_expose_independent_cap():
    for parse, argv in cli_cases():
        assert parse(argv).hydra_batch_size is None
        selected = parse(argv + ["--crop-batch-size", "4", "--hydra-batch-size", "1"])
        assert selected.crop_batch_size == 4 and selected.hydra_batch_size == 1
    assert asdict(live.LiveConfig(hydra_batch_size=1))["hydra_batch_size"] == 1


def test_live_factory_forwards_cap_and_reports_it(tmp_path, monkeypatch):
    pipeline, _ = live_pipeline(monkeypatch)
    calls = []
    def construct(args, _paths):
        calls.append(args)
        return pipeline.cascade
    monkeypatch.setattr(live, "verify_models", lambda _path: ({"models": {}}, {}))
    monkeypatch.setattr(live, "sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(live, "Cascade", construct)
    config = live.LiveConfig(device="cpu", hydra_batch_size=1, crop_batch_size=4)
    result = live.LivePipeline.from_manifest(tmp_path / "manifest.json", config)
    assert calls[0].hydra_batch_size == 1 and calls[0].crop_batch_size == 4
    assert result.report()["configuration"]["hydra_batch_size"] == 1


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true"])
def test_cli_rejects_invalid_caps(value):
    for parse, argv in cli_cases():
        with pytest.raises(SystemExit):
            parse(argv + ["--hydra-batch-size", value])


@pytest.mark.parametrize("mode", ["full", "minimal"])
@pytest.mark.parametrize("batch_size", [None, 1])
def test_live_hydra_singletons_keep_whole_gru_groups_and_tracking_order(monkeypatch, mode, batch_size):
    pipeline, _ = live_pipeline(monkeypatch)
    pipeline.config = replace(pipeline.config, hydra_batch_size=batch_size, hydra_output_mode=mode)
    cascade = pipeline.cascade
    cascade.args.hydra_batch_size, cascade.args.hydra_output_mode = batch_size, mode
    hydra_calls, gru_calls, tracked = [], [], []
    original_hydra = cascade.hydra.predict_tensor_batch_raw
    original_trace = benchmark.trace_arms_noodle_batch
    def forward(images):
        hydra_calls.append(len(images))
        return original_hydra(images)
    def trace(raw, images, *args):
        gru_calls.append((len(raw), len(images)))
        return original_trace(raw, images, *args)
    class Registry:
        def track(self, star, frame, fields, arms, **options):
            assert fields.shape[0] == (2 if mode == "minimal" else 18)
            assert options == ({"support_mouth_only": True} if mode == "minimal" else {})
            tracked.append((frame, star))
            return [42] * len(arms)
        def prune(self, _frame):
            pass
    cascade.hydra.predict_tensor_batch_raw = forward
    cascade.registry = Registry()
    monkeypatch.setattr(benchmark, "trace_arms_noodle_batch", trace)
    records = [pipeline.predict(pixels(count)) for count in (3, 0, 1)]
    assert hydra_calls == ([2, 1, 1] if batch_size is None else [1, 1, 1, 1])
    assert gru_calls == [(2, 2), (1, 1), (1, 1)]
    assert tracked == [(0, 7), (0, 8), (0, 9), (2, 7)]
    assert [len(record["stars"]) for record in records] == [3, 0, 1]
    assert all(star["arms"][0]["track_id"] == 42 for row in records for star in row["stars"])


@pytest.mark.parametrize("mode", ["full", "minimal"])
@pytest.mark.parametrize("execution", ["dynamic", "fixed"])
def test_staged_and_resident_singleton_hydra_preserve_batched_gru_and_spool(
    tmp_path, frames, fake_models, model_paths, mode, execution
):
    args = arguments(crop_batch_size=4, hydra_batch_size=1, hydra_output_mode=mode,
                     gru_execution=execution, gru_check_interval=4)
    fake_models.forbid_full_activations = mode == "minimal"
    state = spool_state()
    timer = benchmark.Timer(torch.device("cpu"), False)
    actual, crops, arms = staged.run_shard(args, model_paths, frames, 0, tmp_path, timer, state)
    assert (crops, arms) == (3, 3)
    assert [event for event in fake_models.events if isinstance(event, tuple) and event[0] == "hydra_forward"] == [("hydra_forward", 1)] * 3
    assert [event for event in fake_models.events if isinstance(event, tuple) and event[0] == "trace"] == [("trace", 3)]
    assert state["live_bytes"] == 0
    assert all(not list(tmp_path.glob(f"*_{suffix}.npy")) for suffix in ("crop", "raw", "image"))
    staged_centers = list(fake_models.traced_centers)
    fake_models.events.clear()
    fake_models.enforce_exclusive = False
    args.detector_batch_size = 3
    cascade = benchmark.Cascade(args, model_paths)
    expected, expected_crops, expected_arms = cascade.run_group(frames, 0, timer)
    assert (actual, crops, arms) == (expected, expected_crops, expected_arms)
    assert fake_models.traced_centers[3:] == staged_centers
    assert [event for event in fake_models.events if isinstance(event, tuple) and event[0] == "hydra_forward"] == [("hydra_forward", 1)] * 3
    assert [event for event in fake_models.events if isinstance(event, tuple) and event[0] == "trace"] == [("trace", 3)]
