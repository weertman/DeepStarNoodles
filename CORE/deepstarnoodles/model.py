# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""DINOv3 ConvNeXt hydra model: frozen-or-finetuned backbone + FPN neck + four dense heads."""
from __future__ import annotations
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false, reportCallIssue=false

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from deepstarnoodles.paths import (
    DEFAULT_DINOV3_CONVNEXT_BASE_CHECKPOINT,
    DEFAULT_DINOV3_REPO,
)

CHANNEL_NAMES = (
    'arm_support',
    'arm_centerline',
    'arm_dx',
    'arm_dy',
    'arm_progress',
    'mouth',
    'madreporite',
    'butthole',
    'arm_base',
    'arm_tip',
)
CHANNEL_NAMES_WITHOUT_CENTERLINE = tuple(
    name for name in CHANNEL_NAMES if name != 'arm_centerline'
)


def _group_count(channels: int, preferred: int = 32) -> int:
    for groups in (preferred, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class GroupNormResidualConvHead(nn.Module):
    """Tiny-batch-stable residual dense head (GroupNorm, no BatchNorm)."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
        )
        self.activation = nn.SiLU(inplace=True)
        self.out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        y = self.activation(y + self.block(y))
        return self.out(y)


class DINOv3ConvNeXtBackbone(nn.Module):
    """Local DINOv3 ConvNeXt wrapper exposing dense intermediate stage maps.

    Inputs are 0..1 RGB tensors; ImageNet normalization is applied internally.
    """

    stage_channels = (128, 256, 512, 1024)

    def __init__(
        self,
        *,
        repo_dir: str | Path = DEFAULT_DINOV3_REPO,
        checkpoint: str | Path = DEFAULT_DINOV3_CONVNEXT_BASE_CHECKPOINT,
        arch: str = 'dinov3_convnext_base',
        freeze: bool = True,
        activation_checkpointing: bool = False,
        pretrained: bool = True,
    ) -> None:
        """Build the ConvNeXt backbone from the vendored DINOv3 hub.

        ``pretrained=True`` (training / from-scratch use) requires Meta's
        DINOv3 checkpoint on disk. ``pretrained=False`` builds the architecture
        with random weights and is the right choice whenever a full Hydra
        ``state_dict`` is about to be loaded on top, because that state dict
        already contains the backbone and would overwrite Meta's weights anyway.
        """
        super().__init__()
        repo_dir = Path(repo_dir)
        if not repo_dir.exists():
            raise FileNotFoundError(f'DINOv3 repo not found: {repo_dir}')
        if pretrained:
            checkpoint = Path(checkpoint)
            if not checkpoint.exists():
                raise FileNotFoundError(f'DINOv3 checkpoint not found: {checkpoint}')
            hub_kwargs = {'weights': str(checkpoint)}
        else:
            hub_kwargs = {'pretrained': False}
        self.backbone = cast(
            nn.Module,
            torch.hub.load(str(repo_dir), arch, source='local', verbose=False, **hub_kwargs),
        )
        self.activation_checkpointing = bool(activation_checkpointing)
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep a fully frozen foundation backbone deterministic in train mode.
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = (x - self.mean.to(dtype=x.dtype, device=x.device)) / self.std.to(dtype=x.dtype, device=x.device)
        backbone_grad = self.training and any(p.requires_grad for p in self.backbone.parameters())
        with torch.set_grad_enabled(backbone_grad):
            if self.activation_checkpointing and backbone_grad:
                stages = checkpoint(
                    self._intermediate_layers,
                    x_norm,
                    use_reentrant=False,
                )
            else:
                stages = self._intermediate_layers(x_norm)
        return tuple(stages)  # type: ignore[return-value]

    def _intermediate_layers(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stages = self.backbone.get_intermediate_layers(
            x, n=[0, 1, 2, 3], reshape=True, norm=False
        )
        return tuple(stages)  # type: ignore[return-value]


class DINOv3ConvNeXtFPN(nn.Module):
    """FPN-style dense neck over DINOv3 ConvNeXt intermediate maps."""

    def __init__(
        self,
        *,
        fpn_channels: int = 256,
        backbone: DINOv3ConvNeXtBackbone | None = None,
        activation_checkpointing: bool = False,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.activation_checkpointing = bool(activation_checkpointing)
        self.backbone = (
            backbone
            if backbone is not None
            else DINOv3ConvNeXtBackbone(
                freeze=True,
                activation_checkpointing=self.activation_checkpointing,
                pretrained=pretrained_backbone,
            )
        )
        if hasattr(self.backbone, 'activation_checkpointing'):
            self.backbone.activation_checkpointing = self.activation_checkpointing
        c1, c2, c3, c4 = self.backbone.stage_channels
        self.lateral1 = nn.Conv2d(c1, fpn_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(c2, fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3, fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4, fpn_channels, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(fpn_channels), fpn_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(fpn_channels), fpn_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1, c2, c3, c4 = self.backbone(x)
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                self._forward_pyramid,
                c1,
                c2,
                c3,
                c4,
                use_reentrant=False,
            )
        return self._forward_pyramid(c1, c2, c3, c4)

    def _forward_pyramid(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
    ) -> torch.Tensor:
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(p1)


class HydraModel(nn.Module):
    """Full 10-channel DeepStarNoodles hydra model on the DINOv3 FPN neck."""

    channel_names = CHANNEL_NAMES

    def __init__(
        self,
        *,
        fpn_channels: int = 256,
        head_hidden_channels: int = 128,
        embedding_dim: int = 0,
        embedding_bandwidth: bool = False,
        count_classes: int = 0,
        include_centerline: bool = True,
        activation_checkpointing: bool = False,
        fpn: nn.Module | None = None,
        pretrained_backbone: bool = True,
    ) -> None:
        """``pretrained_backbone=False`` skips Meta's DINOv3 file; use it when a
        full Hydra ``state_dict`` (which includes the backbone) is loaded next."""
        super().__init__()
        self.activation_checkpointing = bool(activation_checkpointing)
        self.embedding_dim = int(embedding_dim)
        self.embedding_bandwidth = bool(embedding_bandwidth)
        if self.embedding_bandwidth and self.embedding_dim <= 0:
            raise ValueError('embedding_bandwidth requires embedding_dim > 0')
        self.count_classes = int(count_classes)
        self.include_centerline = bool(include_centerline)
        self.channel_names = (
            CHANNEL_NAMES if self.include_centerline else CHANNEL_NAMES_WITHOUT_CENTERLINE
        )
        self.fpn = (
            fpn
            if fpn is not None
            else DINOv3ConvNeXtFPN(
                fpn_channels=fpn_channels,
                activation_checkpointing=self.activation_checkpointing,
                pretrained_backbone=pretrained_backbone,
            )
        )
        self.support_centerline_head = GroupNormResidualConvHead(
            fpn_channels, 2 if self.include_centerline else 1, head_hidden_channels
        )
        self.geometry_head = GroupNormResidualConvHead(fpn_channels, 3, head_hidden_channels)
        self.landmark_head = GroupNormResidualConvHead(fpn_channels, 3, head_hidden_channels)
        self.endpoint_head = GroupNormResidualConvHead(fpn_channels, 2, head_hidden_channels)
        # Optional per-pixel arm-identity embedding appended at channels 10:10+D.
        # Linear output (no activation); the discriminative loss reg term bounds it.
        self.embedding_head = (
            GroupNormResidualConvHead(fpn_channels, self.embedding_dim, head_hidden_channels)
            if self.embedding_dim > 0
            else None
        )
        self.embedding_bandwidth_head = (
            GroupNormResidualConvHead(fpn_channels, 1, head_hidden_channels)
            if self.embedding_bandwidth
            else None
        )
        self.count_head = (
            nn.Linear(fpn_channels, self.count_classes)
            if self.count_classes > 0
            else None
        )

    @property
    def output_spec(self) -> dict[str, Any]:
        """Explicit JSON-safe contract for dense and auxiliary outputs."""
        geometry_stop = len(self.channel_names)
        embedding_stop = geometry_stop + self.embedding_dim
        return {
            'version': 1,
            'representation_id': (
                'legacy10' if self.include_centerline else 'no_centerline9'
            ),
            'geometry': {'names': list(self.channel_names), 'slice': [0, geometry_stop]},
            'embedding': {
                'slice': [geometry_stop, embedding_stop],
                'dim': self.embedding_dim,
                'normalized': False,
            },
            'embedding_bandwidth': {
                'slice': [embedding_stop, embedding_stop + 1],
                'dim': 1,
                'parameterization': 'raw_log_sigma',
                'activation': 'softplus_plus_floor',
            } if self.embedding_bandwidth_head is not None else None,
            'auxiliary': {
                'count_logits': {
                    'classes': self.count_classes,
                    'class_values': list(range(self.count_classes)),
                } if self.count_head is not None else None,
            },
        }

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.fpn(x)
        checkpoint_heads = (
            self.activation_checkpointing and self.training and torch.is_grad_enabled()
        )

        def run_head(head: nn.Module) -> torch.Tensor:
            if checkpoint_heads:
                return checkpoint(head, features, use_reentrant=False)
            return head(features)

        heads = [
            run_head(self.support_centerline_head),
            run_head(self.geometry_head),
            run_head(self.landmark_head),
            run_head(self.endpoint_head),
        ]
        if self.embedding_head is not None:
            heads.append(run_head(self.embedding_head))
        if self.embedding_bandwidth_head is not None:
            heads.append(run_head(self.embedding_bandwidth_head))
        dense = torch.cat(heads, dim=1)
        if dense.shape[-2:] != x.shape[-2:]:
            dense = F.interpolate(dense, size=x.shape[-2:], mode='bilinear', align_corners=False)
        aux: dict[str, torch.Tensor] = {}
        if self.count_head is not None:
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
            aux['count_logits'] = (
                checkpoint(self.count_head, pooled, use_reentrant=False)
                if checkpoint_heads
                else self.count_head(pooled)
            )
        return dense, aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense, _ = self.forward_with_aux(x)
        return dense

    def freeze_all_but_embedding(self) -> None:
        """Freeze the FPN neck/backbone and the four 10-channel heads, leaving
        only the embedding head trainable (warm-start embedding-only training)."""
        if self.embedding_head is None:
            raise RuntimeError('freeze_all_but_embedding requires embedding_dim > 0')
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.embedding_head.parameters():
            p.requires_grad_(True)
        if self.embedding_bandwidth_head is not None:
            for p in self.embedding_bandwidth_head.parameters():
                p.requires_grad_(True)
        self.fpn.backbone.freeze()

    def unfreeze_full_backbone(self) -> None:
        """Unfreeze the complete DINOv3 ConvNeXt backbone."""
        self.fpn.backbone.backbone.train()
        for p in self.fpn.backbone.parameters():
            p.requires_grad_(True)

    def initialize_heatmap_output_biases(self, value: float) -> None:
        """Start sparse sigmoid-heatmap channels near background.

        Applies to support/centerline, landmark, and endpoint heads. The
        geometry head (tanh vectors + progress) is left at default init.
        """
        with torch.no_grad():
            for head in (self.support_centerline_head, self.landmark_head, self.endpoint_head):
                if head.out.bias is None:
                    raise RuntimeError('output head unexpectedly has no bias')
                head.out.bias.fill_(float(value))

    def trainable_parameter_groups(self) -> dict[str, int]:
        wrapped_backbone = getattr(getattr(self.fpn, 'backbone', None), 'backbone', None)
        return {
            'fpn_neck': sum(p.numel() for name, p in self.fpn.named_parameters() if p.requires_grad and not name.startswith('backbone.')),
            'support_centerline_head': sum(p.numel() for p in self.support_centerline_head.parameters() if p.requires_grad),
            'geometry_head': sum(p.numel() for p in self.geometry_head.parameters() if p.requires_grad),
            'landmark_head': sum(p.numel() for p in self.landmark_head.parameters() if p.requires_grad),
            'endpoint_head': sum(p.numel() for p in self.endpoint_head.parameters() if p.requires_grad),
            'embedding_head': sum(p.numel() for p in self.embedding_head.parameters() if p.requires_grad) if self.embedding_head is not None else 0,
            'embedding_bandwidth_head': (
                sum(p.numel() for p in self.embedding_bandwidth_head.parameters() if p.requires_grad)
                if self.embedding_bandwidth_head is not None else 0
            ),
            'count_head': sum(p.numel() for p in self.count_head.parameters() if p.requires_grad) if self.count_head is not None else 0,
            'dinov3_backbone': (
                sum(p.numel() for p in wrapped_backbone.parameters() if p.requires_grad)
                if wrapped_backbone is not None else 0
            ),
        }
