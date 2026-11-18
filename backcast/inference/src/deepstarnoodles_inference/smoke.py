# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Offline synthetic inference smoke; no pretrained weights or downloads."""
from __future__ import annotations

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    import torch
    from .model import HydraModel
    from .noodle_tracer import NoodleTracer, NoodleTraceParams, trace_arms_noodle_batch
    from .paths import DEFAULT_BOTSORT_CONFIG

    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = HydraModel(embedding_dim=8, pretrained_backbone=False).to(args.device).eval()
    tracer = NoodleTracer(pretrained_rgb=False, hidden_size=128).to(args.device).eval()
    with torch.inference_mode():
        images = torch.zeros(1, 3, 32, 32, device=args.device)
        raw = model(images).float()
        assert raw.shape == (1, 18, 32, 32)
        assert torch.isfinite(raw).all()
        arms = trace_arms_noodle_batch(raw, images, tracer, NoodleTraceParams(max_steps=2, max_tips=2))
        assert len(arms) == 1
    assert DEFAULT_BOTSORT_CONFIG.is_file()
    print(json.dumps({"status": "ok", "device": args.device, "torch": torch.__version__,
                      "hydra_output_shape": list(raw.shape), "gru_hidden_size": tracer.hidden_size,
                      "scope": "synthetic random weights; no pretrained accuracy or throughput claim"}))
    return 0
