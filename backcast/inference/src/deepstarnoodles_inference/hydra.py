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

from deepstarnoodles_inference.decode import activate_logits
from deepstarnoodles_inference.checkpoint import (
    checkpoint_output_spec,
    infer_count_classes,
    infer_embedding_dim,
    output_slices,
)
from deepstarnoodles_inference.model import HydraModel
from deepstarnoodles_inference.paths import DEFAULT_HYDRA_CHECKPOINT


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


def store_autocast_weights(model: torch.nn.Module) -> int:
    """Hold Conv2d and Linear parameters in float16; returns how many modules changed.

    CUDA autocast casts these parameters to float16 on every forward, hundreds
    of small copies per crop. Casting once stores the same values. Everything
    else keeps float32 because it does not run through autocast's float16 cast:
    the normalization layers and ConvNeXt layer scale. The model must then
    always run under autocast.
    """
    changed = 0
    for module in model.modules():
        if type(module) in (torch.nn.Conv2d, torch.nn.Linear):
            module.half()
            changed += 1
    return changed


def validate_hydra_batch_size(value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError("hydra_batch_size must be a positive integer or None")


@torch.no_grad()
def predict_hydra_crops(
    predictor,
    images: torch.Tensor,
    *,
    batch_size: int | None = None,
    minimal: bool = False,
    tracking: bool = False,
    graph=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run bounded Hydra forwards, retaining the full ordered GRU crop group.

    None preserves one Hydra forward for the entire input. Explicit chunking
    changes only Hydra's forward batch size, never the subsequent GRU batch.
    Singleton forwards can help localize batch-dependent numerical drift; this
    does not guarantee GPU parity or lower peak memory. Assemble directly into
    output buffers to avoid retaining all chunks plus a second concatenation.
    The public predictor methods retain their existing behavior. A
    ``HydraGraph`` replays minimal-mode raw forwards; full mode stays eager.
    """
    validate_hydra_batch_size(batch_size)
    if images.ndim != 4 or images.shape[1] != 3 or len(images) < 1:
        raise ValueError("images must be nonempty (B,3,H,W)")

    def forward(part):
        if minimal:
            raw = graph.raw(part) if graph is not None else predictor.predict_tensor_batch_raw(part)
            return raw, predictor.tracking_maps(raw) if tracking else None
        return predictor.predict_tensor_batch(part)

    if batch_size is None or batch_size >= len(images):
        return forward(images)

    def allocate_output(part):
        # Preserve channels-last outputs so a Hydra batch cap does not also
        # change the GRU's dense-input layout. Raw and activated maps may use
        # different layouts (especially compact tracking maps).
        layout = (torch.channels_last if part.is_contiguous(memory_format=torch.channels_last)
                  else torch.contiguous_format)
        return torch.empty((len(images), *part.shape[1:]), dtype=part.dtype,
                           device=part.device, memory_format=layout)

    raw_out = activated_out = None
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        raw, activated = forward(images[start:stop])
        if raw_out is None:
            raw_out = allocate_output(raw)
            if activated is not None:
                activated_out = allocate_output(activated)
        raw_out[start:stop].copy_(raw)
        if activated is not None:
            activated_out[start:stop].copy_(activated)
        del raw, activated
    return raw_out, activated_out


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
        self.selected_state_dict_key = "model"
        if config.prefer_ema:
            for key in ("model_ema", "ema", "ema_model"):
                if isinstance(ckpt.get(key), dict) and ckpt[key]:
                    self.selected_state_dict_key = key
                    break
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
    def predict_tensor_batch_raw(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Return float32 raw fields without allocating activated fields.

        The GRU consumes these raw fields directly. Archive inference does not
        need the second dense tensor returned by :meth:`predict_tensor_batch`.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be (B,3,H,W)")
        image = images.to(self.device)
        use_amp = self.config.amp and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            raw = self.model(image).float()
        # Retain the output-contract validation of the compatibility path.
        output_slices(self.output_spec or self.model.output_spec, raw.shape[1])
        return raw

    @torch.no_grad()
    def tracking_maps(self, raw: torch.Tensor) -> torch.Tensor:
        """Activate only support and mouth, in that order, as ``(B,2,H,W)``.

        Live arm association uses only these maps. Keep raw fields unchanged for
        the GRU and avoid transferring embeddings or unused geometry to CPU.
        """
        if raw.ndim != 4:
            raise ValueError("raw fields must be (B,C,H,W)")
        geometry_slice, _ = output_slices(
            self.output_spec or self.model.output_spec, raw.shape[1]
        )
        if geometry_slice != (0, 10):
            raise ValueError("tracking maps require ten-channel Hydra geometry")
        # Validated ten-channel geometry fixes arm_support=0 and mouth=5.
        maps = torch.empty(
            (raw.shape[0], 2, *raw.shape[-2:]), dtype=torch.float32, device=raw.device
        )
        torch.sigmoid(raw[:, 0], out=maps[:, 0])
        torch.sigmoid(raw[:, 5], out=maps[:, 1])
        return maps

    @torch.no_grad()
    def predict_tensor_batch(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``(B,3,S,S)`` input and return raw plus full activated fields.

        This compatibility API retains its original shapes and return value.
        Use :meth:`predict_tensor_batch_raw` when activations are unnecessary.
        """
        raw = self.predict_tensor_batch_raw(images)
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
