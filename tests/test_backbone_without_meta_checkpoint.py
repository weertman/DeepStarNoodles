# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
# pyright: reportPrivateImportUsage=false, reportMissingImports=false
"""Hydra inference must not require Meta's DINOv3 file (GitHub issue #1)."""
from pathlib import Path

import pytest
import torch

from deepstarnoodles.model import DINOv3ConvNeXtBackbone, HydraModel
from deepstarnoodles.paths import DEFAULT_DINOV3_REPO

MISSING = Path("/nonexistent/dinov3_convnext_base_pretrain.pth")

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_DINOV3_REPO).exists(), reason="vendored external/dinov3 hub missing"
)


def test_pretrained_backbone_still_requires_meta_file():
    with pytest.raises(FileNotFoundError):
        DINOv3ConvNeXtBackbone(checkpoint=MISSING, pretrained=True)


def test_backbone_builds_without_meta_file():
    backbone = DINOv3ConvNeXtBackbone(checkpoint=MISSING, pretrained=False)
    stages = backbone(torch.zeros(1, 3, 64, 64))
    assert tuple(s.shape[1] for s in stages) == backbone.stage_channels


def test_hydra_model_state_dict_matches_between_construction_paths():
    """A full Hydra state dict loads identically regardless of how the
    backbone was initialized, so the Meta weights are provably discarded."""
    torch.manual_seed(0)
    model_a = HydraModel(embedding_dim=8, pretrained_backbone=False)
    torch.manual_seed(1)
    model_b = HydraModel(embedding_dim=8, pretrained_backbone=False)
    state = model_a.state_dict()
    assert any(k.startswith("fpn.backbone.") for k in state), "backbone must live in the state dict"
    missing, unexpected = model_b.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    model_a.eval()
    model_b.eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model_a(x), model_b(x))
