# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Load an 18-channel Hydra checkpoint and run training-time preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from deepstarnoodles.decode import activate_logits
from deepstarnoodles.evaluate_decode import (
    checkpoint_output_spec,
    infer_count_classes,
    infer_embedding_dim,
    output_slices,
)
from deepstarnoodles.model import HydraModel
from deepstarnoodles.paths import DEFAULT_HYDRA_CHECKPOINT


def _load_state_dict(ckpt: dict, prefer_ema: bool) -> dict:
    if prefer_ema:
        for key in ("model_ema", "ema", "ema_model"):
            sd = ckpt.get(key)
            if isinstance(sd, dict) and sd:
                return sd
    return ckpt["model"]


@dataclass(frozen=True)
class HydraInferConfig:
    image_size: int = 512
    prefer_ema: bool = True
    device: str = "cuda:0"
    amp: bool = True


class HydraPredictor:
    """Single-checkpoint dense predictor that preserves embedding channels.

    Preprocessing matches training: PIL bilinear resize, float32/255. ImageNet
    normalization happens inside the DINOv3 backbone.
    """

    def __init__(
        self,
        checkpoint: Path = DEFAULT_HYDRA_CHECKPOINT,
        config: HydraInferConfig = HydraInferConfig(),
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.output_spec = checkpoint_output_spec(ckpt) or {}
        # The Hydra checkpoint carries the backbone weights, so Meta's separate
        # DINOv3 file is not needed for inference (see GitHub issue #1).
        model = HydraModel(
            embedding_dim=infer_embedding_dim(ckpt),
            count_classes=infer_count_classes(ckpt),
            pretrained_backbone=False,
        ).to(self.device)
        model.load_state_dict(_load_state_dict(ckpt, config.prefer_ema))
        model.eval()
        self.model = model
        cfg = ckpt.get("config") or {}
        if isinstance(cfg, dict) and cfg.get("image_size"):
            self.checkpoint_image_size = int(cfg["image_size"])
        else:
            self.checkpoint_image_size = config.image_size

    @property
    def image_size(self) -> int:
        return self.config.image_size or self.checkpoint_image_size

    def preprocess_batch(
        self,
        crops_rgb: Sequence[np.ndarray] | np.ndarray,
    ) -> torch.Tensor:
        """Convert equally-sized RGB crops to a contiguous ``(B,3,S,S)`` tensor."""
        if isinstance(crops_rgb, np.ndarray) and crops_rgb.ndim == 3:
            crops = [crops_rgb]
        else:
            crops = list(crops_rgb)
        if not crops:
            raise ValueError("at least one crop is required")
        size = self.image_size
        tensors: list[torch.Tensor] = []
        for crop_rgb in crops:
            if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
                raise ValueError("each crop must be HxWx3 RGB")
            img = Image.fromarray(crop_rgb, mode="RGB")
            if img.size != (size, size):
                img = img.resize((size, size), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        return torch.stack(tensors).contiguous()

    def preprocess(self, crop_rgb: np.ndarray) -> torch.Tensor:
        """Compatibility wrapper returning one ``(1,3,S,S)`` tensor."""
        return self.preprocess_batch([crop_rgb])

    @torch.no_grad()
    def predict_tensor_batch(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run an already-preprocessed ``(B,3,S,S)`` tensor."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be (B,3,H,W)")
        image = images.to(self.device)
        use_amp = self.config.amp and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            raw = self.model(image).float()
        spec = self.output_spec or self.model.output_spec
        geometry_slice, embedding_slice = output_slices(spec, raw.shape[1])
        activated = torch.stack(
            [
                torch.cat(
                    [
                        activate_logits(sample[slice(*geometry_slice)]),
                        sample[slice(*embedding_slice)],
                    ],
                    dim=0,
                )
                for sample in raw
            ]
        )
        return raw, activated

    @torch.no_grad()
    def predict_raw_batch(
        self,
        crops_rgb: Sequence[np.ndarray] | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess crops, then return raw and activated ``(B,C,H,W)`` tensors."""
        return self.predict_tensor_batch(self.preprocess_batch(crops_rgb))

    @torch.no_grad()
    def predict_raw(self, crop_rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one image as raw and activated ``(C,H,W)`` tensors."""
        raw, activated = self.predict_raw_batch([crop_rgb])
        return raw[0], activated[0]
