# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""The device body summary equals the host rules bit for bit (runtime body origin and re-trace area)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepstarnoodles_inference import body_fields as BF
from deepstarnoodles_inference.decode import activate_logits, find_peaks
from deepstarnoodles_inference.runtime import body_center_from_maps

CUDA = torch.cuda.is_available()


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def retrace_fields(support, mouth):
    """Inline copy of the paper re-trace's frame landmarks plus the runtime body origin."""
    origin = body_center_from_maps(support, mouth)
    ys, xs = np.nonzero(support > 0.5)
    centroid = [float(xs.mean()), float(ys.mean())] if len(xs) else None
    return [float(origin[0]), float(origin[1])], int(len(xs)), centroid


def raw_cases(height=48, width=40, seed=0):
    """(name, raw (18,H,W) float32) cases covering every branch and the threshold edges."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    cases = []

    def put(name, support, mouth):
        raw = rng.normal(0, 3, (18, height, width)).astype(np.float32)
        raw[0], raw[5] = support, mouth
        cases.append((name, raw))

    for k in range(4):
        put(f"random{k}", rng.normal(-1, 3, (height, width)), rng.normal(-3, 2, (height, width)))
    blob = lambda cy, cx, s, a: logit(a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s)))
    put("blobs", blob(20, 15, 6, 0.95), blob(22, 17, 3, 0.7))
    put("mouth_below_threshold", blob(20, 15, 6, 0.95), blob(22, 17, 3, 0.1))
    put("support_only_low", blob(10, 30, 4, 0.4), np.full((height, width), -30.0))
    put("empty", np.full((height, width), -30.0), np.full((height, width), -30.0))
    sat = np.full((height, width), -30.0)
    sat[5:9, 5:9] = 40.0                   # sigmoid saturates to exactly 1.0: a 16-way tie
    put("saturated_mouth_tie", blob(20, 15, 6, 0.95), sat)
    tie = np.full((height, width), -30.0)
    tie[3, 4] = tie[30, 30] = 2.0          # two separated equal peaks
    put("two_equal_peaks", blob(20, 15, 6, 0.95), tie)
    flat = np.full((height, width), 0.0)   # every pixel 0.5: tie at the maximum
    put("flat_mouth", flat, flat)
    # values on the float32 thresholds and one ulp either side, written as logits that map back to them
    for name, threshold in (("area", 0.5), ("origin", 0.3), ("mouth", 0.15)):
        value = np.float32(threshold)
        support = np.full((height, width), -30.0, np.float32)
        mouth = np.full((height, width), -30.0, np.float32)
        target = support if name != "mouth" else mouth
        for i, v in enumerate((np.nextafter(value, np.float32(0)), value, np.nextafter(value, np.float32(1)))):
            target[10 + 3 * i, 5 + 4 * i] = np.float32(logit(v))
        put(f"edge_{name}", support, mouth)
    nan = np.full((height, width), -2.0)
    nan[4, 4] = np.nan
    put("nan_mouth", blob(20, 15, 6, 0.95), nan)
    put("nan_support", np.where(xx > 30, np.nan, blob(20, 15, 6, 0.95)), blob(22, 17, 3, 0.7))
    return cases


def summarize_batch(raw, mode):
    if mode == "minimal":
        activated = None
    elif mode == "compact":
        activated = torch.stack([torch.sigmoid(raw[:, 0]), torch.sigmoid(raw[:, 5])], dim=1)
    else:
        activated = torch.stack([torch.cat([activate_logits(s[:10]), s[10:]]) for s in raw])
    host = activated if activated is not None else torch.sigmoid(raw)
    mouth_channel = 1 if mode == "compact" else 5
    expected = [retrace_fields(host[b, 0].cpu().numpy(), host[b, mouth_channel].cpu().numpy())
                for b in range(len(raw))]
    return BF.summarize(raw, activated).finish(), expected, host, mouth_channel


@pytest.mark.parametrize("mode", ["minimal", "compact", "full"])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("batch", [1, 5])
def test_device_summary_equals_host_rules(mode, channels_last, batch):
    cases = raw_cases()
    for start in range(0, len(cases), batch):
        chunk = cases[start:start + batch]
        raw = torch.from_numpy(np.stack([r for _, r in chunk]))
        if channels_last:
            raw = raw.contiguous(memory_format=torch.channels_last)
        got, expected, host, mouth_channel = summarize_batch(raw, mode)
        for (name, _), fields, (origin, area, centroid), b in zip(chunk, got, expected, range(len(chunk))):
            assert fields["body_origin_model"] == origin, name
            assert fields["support_area_model_px"] == area, name
            assert fields["support_centroid_model"] == centroid, name
            assert body(fields) == BF.host_body_fields(host[b, 0].numpy(), host[b, mouth_channel].numpy()), name
            assert list(fields) == [*BF.host_body_fields(host[b, 0].numpy(), host[b, 0].numpy()), "landmarks_model"]
            assert fields["landmarks_model"] == BF.host_landmarks(torch.sigmoid(raw[b, 5:8]).numpy()), name


def body(fields):
    """The schema-3 fields of a summary, without the landmarks."""
    return {key: value for key, value in fields.items() if key != "landmarks_model"}


def test_source_codes_and_tie_fallback(monkeypatch):
    cases = dict(raw_cases())
    raw = torch.from_numpy(np.stack([cases[name] for name in (
        "blobs", "mouth_below_threshold", "empty", "saturated_mouth_tie", "flat_mouth", "nan_mouth")]))
    calls = []
    real = BF.origin_from_maps
    monkeypatch.setattr(BF, "origin_from_maps", lambda s, m: calls.append(1) or real(s, m))
    fields = BF.summarize(raw, None).finish()
    assert [f["body_origin_code"] for f in fields] == [0, 1, 2, 0, 0, 1]
    assert len(calls) == 3          # tie, flat tie and NaN crops use the host rule
    assert body(fields[2]) == {"body_origin_model": [20.0, 24.0], "body_origin_code": 2,
                               "support_area_model_px": 0, "support_centroid_model": None}
    assert fields[3]["body_origin_model"] == list(map(float, body_center_from_maps(
        torch.sigmoid(raw[3, 0]).numpy(), torch.sigmoid(raw[3, 5]).numpy())))


def test_maps_for_body_sources_agree():
    raw = torch.from_numpy(np.stack([r for _, r in raw_cases()[:6]]))
    results = [summarize_batch(raw, mode)[0] for mode in ("minimal", "compact", "full")]
    assert results[0] == results[1] == results[2]


def test_summary_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="matching"):
        BF.body_summary(torch.zeros(1, 4, 4), torch.zeros(1, 4, 5), None)
    with pytest.raises(ValueError, match="float32"):
        BF.body_summary(torch.zeros(1, 4, 4, dtype=torch.float64), torch.zeros(1, 4, 4, dtype=torch.float64), None)


@pytest.mark.skipif(not CUDA, reason="device summary on CUDA")
@pytest.mark.parametrize("channels_last", [False, True])
def test_cuda_summary_equals_host_rules(channels_last):
    raw = torch.from_numpy(np.stack([r for _, r in raw_cases(256, 320)])).cuda()
    if channels_last:
        raw = raw.contiguous(memory_format=torch.channels_last)
    for mode in ("minimal", "compact", "full"):
        got, expected, host, mouth_channel = summarize_batch(raw, mode)
        for fields, (origin, area, centroid), b in zip(got, expected, range(len(raw))):
            assert (fields["body_origin_model"], fields["support_area_model_px"],
                    fields["support_centroid_model"]) == (origin, area, centroid)
            assert body(fields) == BF.host_body_fields(host[b, 0].cpu().numpy(), host[b, mouth_channel].cpu().numpy())
            assert fields["landmarks_model"] == BF.host_landmarks(torch.sigmoid(raw[b, 5:8]).cpu().numpy())


def test_exact_threshold_values_on_activated_maps():
    # Activated maps holding f32(t) and its neighbours exactly: strict ">" for support, ">=" for the mouth.
    height, width = 32, 36
    maps = []
    for t_support, t_mouth in ((0.5, 0.0), (0.3, 0.0), (0.0, 0.15), (0.5, 0.15)):
        support = np.zeros((height, width), np.float32)
        mouth = np.zeros((height, width), np.float32)
        for target, t in ((support, t_support), (mouth, t_mouth)):
            if t:
                v = np.float32(t)
                for i, value in enumerate((np.nextafter(v, np.float32(0)), v, np.nextafter(v, np.float32(1)))):
                    target[4 + 5 * i, 3 + 7 * i] = value
        if t_mouth:
            mouth[25, 30] = np.float32(t_mouth)   # equal to the threshold, below the ulp-above peak
        maps.append((support, mouth))
    support = torch.from_numpy(np.stack([s for s, _ in maps]))
    mouth = torch.from_numpy(np.stack([m for _, m in maps]))
    fields = BF.body_summary(support, mouth, lambda b: maps[b]).finish()
    for (s, m), got in zip(maps, fields):
        assert got == BF.host_body_fields(s, m)
    assert [f["support_area_model_px"] for f in fields] == [1, 0, 0, 1]
    assert [f["body_origin_code"] for f in fields] == [1, 1, 0, 0]


def test_landmarks_are_each_maps_maximum_with_its_probability():
    raw = torch.full((2, 18, 20, 24), -6.0)
    for channel, (y, x, value) in zip((5, 6, 7), ((3, 7, 2.0), (15, 1, -1.0), (0, 23, 0.5))):
        raw[0, channel, y, x] = value
    raw[1, 7, 4, 4] = float("nan")
    first, second = (fields["landmarks_model"] for fields in BF.summarize(raw, None).finish())
    p = lambda v: float(torch.sigmoid(torch.tensor(v, dtype=torch.float32)))
    assert first == {"mouth": [7.0, 3.0, p(2.0)], "madreporite": [1.0, 15.0, p(-1.0)], "butthole": [23.0, 0.0, p(0.5)]}
    # Flat maps tie everywhere: the first pixel. A non-finite map gives None.
    assert second == {"mouth": [0.0, 0.0, p(-6.0)], "madreporite": [0.0, 0.0, p(-6.0)], "butthole": None}


def test_first_peak_equals_find_peaks_first_row():
    rng = np.random.default_rng(3)
    maps = [rng.random((40, 44), dtype=np.float32) for _ in range(5)]
    plateau = np.full((40, 44), 0.2, np.float32)
    plateau[10:30, 5:40] = 1.0               # saturated plateau: hundreds of tied candidates
    maps += [plateau, np.full((40, 44), 0.5, np.float32), np.zeros((40, 44), np.float32)]
    for heat in maps:
        expected = find_peaks(heat, 0.15, 8)
        got = BF.first_peak(heat, 0.15, 8)
        if len(expected):
            assert got.dtype == np.float32 and np.array_equal(got, expected[0, :2])
        else:
            assert got is None
