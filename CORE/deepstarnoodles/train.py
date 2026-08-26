# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Train the 18-channel DINOv3 hydra model jointly on DeepStarNoodles dense targets.

All four head families are optimized in one run:

- support (ch 0):      balanced BCE + Dice + area penalty
- centerline (ch 1):   focal BCE on the compact-poly target, masked to target support
- geometry (ch 2-4):   support-masked L1 on tanh(dx,dy) and sigmoid(progress)
- landmarks (ch 5-7):  MSE on sigmoid heatmaps
- endpoints (ch 8-9):  MSE on sigmoid heatmaps

Run from the repo root, e.g.:

    python -m deepstarnoodles.train --out-dir RUNS/hydra_18ch
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from deepstarnoodles.dataset import (
    CHANNEL_NAMES,
    CHANNEL_NAMES_WITHOUT_CENTERLINE,
    HydraDenseDataset,
    moderate_augment_kwargs,
)
from deepstarnoodles.embedding_loss import (
    crossing_continuity_loss,
    discriminative_embedding_loss,
    embedding_bandwidth_loss,
    local_affinity_loss,
    prototypical_embedding_loss,
)
from deepstarnoodles.metrics import centerline_path_metrics
from deepstarnoodles.model import HydraModel
from deepstarnoodles.paths import (
    DEFAULT_DENSE_INDEX,
    DEFAULT_IGNORE_SHAPES_DIR,
    DEFAULT_TRAIN_INDEX,
    DEFAULT_VALIDATION_INDEX,
)

DEFAULT_INDEX = DEFAULT_DENSE_INDEX


def parse_save_at_epochs(value: str) -> tuple[int, ...]:
    """Parse a comma-separated epoch list into sorted, unique positive values."""
    if not value.strip():
        return ()
    epochs: list[int] = []
    for part in value.split(','):
        token = part.strip()
        if not token:
            raise argparse.ArgumentTypeError('--save-at-epochs must be comma-separated positive integers')
        try:
            epoch = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f'--save-at-epochs contains a non-integer value: {token!r}'
            ) from exc
        if epoch <= 0:
            raise argparse.ArgumentTypeError(
                f'--save-at-epochs values must be positive; got {epoch}'
            )
        epochs.append(epoch)
    return tuple(sorted(set(epochs)))


def validate_snapshot_epochs(
    requested_epochs: Collection[int],
    *,
    planned_epochs: int,
) -> tuple[int, ...]:
    """Return normalized snapshot epochs, rejecting depths outside the run plan."""
    normalized = tuple(sorted(set(requested_epochs)))
    impossible = [epoch for epoch in normalized if epoch <= 0 or epoch > planned_epochs]
    if impossible:
        raise ValueError(
            '--save-at-epochs contains impossible epoch(s) '
            f'{impossible}; this run will enter epochs 1..{planned_epochs}'
        )
    return normalized


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


_CONFIG_IDENTITY_EXCLUDED_KEYS = frozenset({
    'actual_optimizer_steps',
    'out_dir',
    'training_config_identity',
    'validations_completed',
    'written_depth_snapshots',
})


def _canonical_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical_config_value(item) for item in value), key=repr)
    return value


def training_config_identity(config: Mapping[str, Any]) -> str:
    """Hash the deterministic, training-relevant portion of a run configuration."""
    identity_config = {
        key: value
        for key, value in config.items()
        if key not in _CONFIG_IDENTITY_EXCLUDED_KEYS
    }
    canonical = json.dumps(
        _canonical_config_value(identity_config),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Joint 18-channel DINOv3 hydra training.')
    p.add_argument(
        '--index',
        type=Path,
        default=DEFAULT_INDEX,
        help='Canonical index used for the legacy seeded 80/20 row split.',
    )
    p.add_argument(
        '--train-index',
        type=Path,
        default=DEFAULT_TRAIN_INDEX,
        help='Explicit leakage-safe training index; requires --val-index.',
    )
    p.add_argument(
        '--val-index',
        type=Path,
        default=DEFAULT_VALIDATION_INDEX,
        help='Explicit leakage-safe validation index; requires --train-index.',
    )
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260606)
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--max-steps', type=int, default=None,
                   help='Optional exact optimizer-step budget. The final epoch may be partial; '
                        'the LR schedule and final checkpoint end exactly at this step.')
    p.add_argument('--validate-every-epochs', type=int, default=1,
                   help='Run validation/checkpointing every N training epochs (default: 1). '
                        'The final full or partial epoch is always validated.')
    p.add_argument('--train-subset', type=int, default=None, help='Optional cap on train images (default: full 80% split).')
    p.add_argument('--val-subset', type=int, default=None, help='Optional cap on val images (default: full 20% split).')
    p.add_argument('--augment', choices=['none', 'moderate'], default='moderate')
    p.add_argument('--degrade-p', type=float, default=0.0,
                   help='Probability of USB-quality degradation augment, applied only to images '
                        'with native min side >= --degrade-min-native.')
    p.add_argument('--degrade-min-native', type=int, default=384)
    p.add_argument('--ignore-shapes-dir', type=Path, default=DEFAULT_IGNORE_SHAPES_DIR,
                   help='Directory of star_ignore polygon JSONs (from deepstarnoodles.build_ignore_masks); '
                        'those pixels are excluded from support/heatmap losses.')
    p.add_argument('--target-cache-dir', type=Path, default=None, help='On-disk cache for rasterized (non-augmented) targets.')
    p.add_argument('--image-cache-dir', type=Path, default=None,
                   help='On-disk cache of source images pre-resized to --image-size, to avoid re-decoding '
                        'full-resolution originals every epoch (safe with augmentation).')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        '--activation-checkpointing',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Recompute ConvNeXt/FPN/head activations during backward to reduce training memory.',
    )
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--backbone-llrd', type=float, default=1.0,
                   help='Layer-wise LR decay per backbone level: level k of 0..3 trains at '
                        'backbone_lr * llrd^(3-k). 1.0 = uniform (off); try 0.7.')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--ema-decay', type=float, default=0.0,
                   help='If > 0, keep an EMA of weights; validation and checkpoints use the '
                        'EMA weights (raw weights saved under model_raw). Try 0.999.')
    p.add_argument('--lr-warmup-steps', type=int, default=800)
    p.add_argument('--lr-warmup-start-factor', type=float, default=0.02)
    p.add_argument('--lr-final-factor', type=float, default=0.05, help='Cosine-decay LR to this fraction of base by the last step.')
    p.add_argument('--freeze-backbone', action='store_true', help='Keep DINOv3 frozen (default: full backbone finetune).')
    p.add_argument('--output-bias-init', type=float, default=-4.0, help='Initial logit bias for sigmoid heatmap channels.')
    # Warm-start: load an existing checkpoint (strict=False) before training. With
    # an --init-checkpoint, the heatmap output-bias init is skipped so the loaded
    # biases are preserved.
    p.add_argument('--init-checkpoint', type=Path, default=None,
                   help='Checkpoint to warm-start from (strict=False); the embedding head stays fresh.')
    # Arm-identity embedding head (channels 10:10+D) + discriminative loss.
    p.add_argument('--embedding-dim', type=int, default=8, help='Arm-identity embedding channels. 0 disables the head. Locked recipe uses 8.')
    p.add_argument(
        '--embedding-bandwidth',
        action='store_true',
        help='Append a learned scalar log-bandwidth for heteroscedastic arm clustering.',
    )
    p.add_argument('--bandwidth-loss-weight', type=float, default=0.0)
    p.add_argument('--bandwidth-consistency-weight', type=float, default=0.05)
    p.add_argument('--bandwidth-sigma-floor', type=float, default=0.05)
    p.add_argument('--bandwidth-max-pixels-per-arm', type=int, default=1024)
    p.add_argument(
        '--reset-embedding-head',
        action='store_true',
        help='Warm-start dense/backbone weights but initialize the embedding head fresh. '
             'Embedding shape mismatches are always reset.',
    )
    p.add_argument('--embedding-loss-weight', type=float, default=1.0, help='Steady-state weight on the embedding loss.')
    p.add_argument('--embedding-warmup-epochs', type=int, default=0,
                   help='Linearly ramp the embedding-loss weight from 0 to --embedding-loss-weight over this many '
                        'epochs (0 = no warmup). Protects warm-started channels from the fresh head\'s initial push spike.')
    p.add_argument('--embed-delta-v', type=float, default=0.5)
    p.add_argument('--embed-delta-d', type=float, default=1.5)
    p.add_argument('--embed-w-pull', type=float, default=1.0)
    p.add_argument('--embed-w-push', type=float, default=1.0)
    p.add_argument('--embed-w-reg', type=float, default=1e-3)
    p.add_argument('--prototype-loss-weight', type=float, default=0.0)
    p.add_argument('--prototype-temperature', type=float, default=0.1)
    p.add_argument('--prototype-max-pixels-per-arm', type=int, default=256)
    p.add_argument('--affinity-loss-weight', type=float, default=0.0)
    p.add_argument('--affinity-offsets', type=str, default='0:1,1:0,0:2,2:0',
                   help='Comma-separated dy:dx local affinity offsets.')
    p.add_argument('--affinity-margin', type=float, default=0.25)
    p.add_argument('--continuity-loss-weight', type=float, default=0.0)
    p.add_argument('--max-bridge-pairs', type=int, default=0)
    p.add_argument('--bridge-offset', type=float, default=6.0)
    p.add_argument('--count-classes', type=int, default=0,
                   help='Number of arm-count classes (class index equals count); 0 disables.')
    p.add_argument('--count-loss-weight', type=float, default=0.0)
    # Loss weights and shape.
    p.add_argument('--support-loss-weight', type=float, default=1.0)
    p.add_argument('--support-pos-weight-cap', type=float, default=100.0, help='Upper clamp on the balanced-BCE positive weight for support.')
    p.add_argument('--support-area-penalty-weight', type=float, default=2.0)
    p.add_argument('--support-area-margin', type=float, default=1.15)
    p.add_argument('--support-boundary-weight', type=float, default=0.0, help='Extra BCE weight on background pixels near the support tube (the over-paint halo).')
    p.add_argument('--support-boundary-radius', type=int, default=4)
    # Plateau-triggered support tightening (phase 2): once val support dice stalls,
    # switch to a precision-oriented support loss.
    p.add_argument('--support-tighten-on-plateau', action='store_true')
    p.add_argument('--support-plateau-min-epochs', type=int, default=10)
    p.add_argument('--support-plateau-patience', type=int, default=4)
    p.add_argument('--support-plateau-min-delta', type=float, default=0.002)
    p.add_argument('--phase2-support-pos-weight-cap', type=float, default=8.0)
    p.add_argument('--phase2-support-area-margin', type=float, default=1.0)
    p.add_argument('--phase2-support-area-penalty-weight', type=float, default=4.0)
    p.add_argument('--phase2-support-boundary-weight', type=float, default=3.0)
    p.add_argument('--centerline-loss-weight', type=float, default=3.0)
    p.add_argument('--no-centerline', action='store_true',
                   help='Remove the centerline target and model output channel entirely.')
    p.add_argument('--centerline-focal-alpha', type=float, default=0.75)
    p.add_argument('--centerline-focal-gamma', type=float, default=2.0)
    p.add_argument('--geometry-loss-weight', type=float, default=1.0)
    p.add_argument('--landmark-loss-weight', type=float, default=10.0)
    p.add_argument('--endpoint-loss-weight', type=float, default=10.0)
    p.add_argument(
        '--endpoint-heatmap-loss',
        choices=['peak_weighted_mse', 'modified_focal'],
        default='peak_weighted_mse',
        help='Endpoint heatmap objective (default preserves the current peak-weighted MSE).',
    )
    p.add_argument('--endpoint-focal-alpha', type=float, default=2.0)
    p.add_argument('--endpoint-focal-beta', type=float, default=4.0)
    p.add_argument('--arm-base-loss-multiplier', type=float, default=1.0)
    p.add_argument('--arm-tip-loss-multiplier', type=float, default=1.0)
    p.add_argument(
        '--endpoint-only',
        action='store_true',
        help='Freeze every model parameter except endpoint_head for endpoint fine-tuning.',
    )
    p.add_argument('--heatmap-peak-weight', type=float, default=20.0, help='Extra per-pixel MSE weight on landmark/endpoint peaks: w = 1 + peak_weight * target.')
    # Targets.
    p.add_argument('--support-arm-width', type=float, default=8.0)
    p.add_argument('--centerline-radius', type=float, default=1.5)
    p.add_argument('--arm-base-sigma', type=float, default=1.5)
    p.add_argument('--arm-tip-sigma', type=float, default=2.0)
    p.add_argument('--point-sigma', type=float, default=4.0)
    p.add_argument('--trim-overlapping-arm-bases', action='store_true',
                   help='Shorten smooth arm polylines before transforms until base kernels separate.')
    p.add_argument('--base-trim-kernel-radius-sigmas', type=float, default=1.5)
    p.add_argument('--base-trim-max-fraction', type=float, default=0.5)
    p.add_argument('--base-trim-min-remaining-length', type=float, default=8.0)
    # Reporting.
    p.add_argument('--path-metric-threshold', type=float, default=0.5)
    p.add_argument('--path-metric-tolerances', type=str, default='2,4,8')
    p.add_argument('--train-path-metrics', action=argparse.BooleanOptionalAction, default=False,
                   help='Compute the expensive centerline path metric on training batches too. '
                        'Off by default (computed only on validation) since it idles the GPU.')
    p.add_argument('--save-visuals-every', type=int, default=5)
    p.add_argument('--num-visuals', type=int, default=4)
    p.add_argument('--save-checkpoints', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--save-every', type=int, default=10,
                   help='Save numbered checkpoints every N epochs (<=0 disables).')
    p.add_argument(
        '--save-at-epochs',
        type=parse_save_at_epochs,
        default=(),
        help='Comma-separated positive epochs for immutable weights-only transfer snapshots.',
    )
    p.add_argument('--best-selector', choices=['val_loss', 'path_f1'], default='path_f1', help='Metric for saving best.pt.')
    args = p.parse_args()
    if os.environ.get('HYDRA_METRICS_ONLY') == '1':
        args.save_checkpoints = False
        args.save_at_epochs = ()
    if (args.train_index is None) != (args.val_index is None):
        p.error('--train-index and --val-index must be provided together')
    if args.max_steps is not None and args.max_steps <= 0:
        p.error('--max-steps must be positive')
    if args.validate_every_epochs <= 0:
        p.error('--validate-every-epochs must be positive')
    if args.embedding_bandwidth and args.embedding_dim <= 0:
        p.error('--embedding-bandwidth requires --embedding-dim')
    if args.bandwidth_loss_weight != 0 and not args.embedding_bandwidth:
        p.error('--bandwidth-loss-weight requires --embedding-bandwidth')
    if args.bandwidth_sigma_floor <= 0:
        p.error('--bandwidth-sigma-floor must be positive')
    if args.bandwidth_max_pixels_per_arm <= 0:
        p.error('--bandwidth-max-pixels-per-arm must be positive')
    return args


def weighted_heatmap_mse(
    logits: torch.Tensor,
    true: torch.Tensor,
    peak_weight: float,
    keep: torch.Tensor | None = None,
    channel_multipliers: Sequence[float] | None = None,
) -> torch.Tensor:
    """MSE on sigmoid heatmaps with extra weight on peak pixels.

    Plain MSE is dominated by the >99.9% background pixels; w = 1 + peak_weight * target
    keeps background pressure while making the gaussian peaks matter.
    `keep` (B, 1, H, W) zeroes out ignore regions (unannotated neighbor stars).
    """
    w = 1.0 + float(peak_weight) * true
    if channel_multipliers is not None:
        multipliers = logits.new_tensor(channel_multipliers).view(1, -1, 1, 1)
        if multipliers.shape[1] != logits.shape[1]:
            raise ValueError('channel_multipliers must match the heatmap channel count')
        w = w * multipliers
    if keep is not None:
        w = w * keep
    return ((torch.sigmoid(logits) - true).pow(2) * w).sum() / w.sum().clamp_min(1.0)


def modified_focal_heatmap_loss(
    logits: torch.Tensor,
    true: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    keep: torch.Tensor | None = None,
    channel_multipliers: Sequence[float] | None = None,
) -> torch.Tensor:
    """Numerically stable CornerNet-style focal loss for Gaussian heatmaps."""
    positive = true.eq(1.0).to(logits.dtype)
    negative = true.lt(1.0).to(logits.dtype)
    if keep is not None:
        positive = positive * keep
        negative = negative * keep
    if channel_multipliers is not None:
        multipliers = logits.new_tensor(channel_multipliers).view(1, -1, 1, 1)
        if multipliers.shape[1] != logits.shape[1]:
            raise ValueError('channel_multipliers must match the heatmap channel count')
        positive = positive * multipliers
        negative = negative * multipliers
    prob = torch.sigmoid(logits)
    positive_loss = -F.logsigmoid(logits) * (1.0 - prob).pow(float(alpha)) * positive
    negative_loss = (
        -F.logsigmoid(-logits)
        * prob.pow(float(alpha))
        * (1.0 - true).pow(float(beta))
        * negative
    )
    positive_count = positive.sum()
    total = positive_loss.sum() + negative_loss.sum()
    return total / positive_count.clamp_min(1.0)


def support_boundary_ring(support_true: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Background pixels within radius_px of the target support tube.

    This is exactly the halo a recall-heavy support head over-paints; phase-2
    tightening upweights BCE there.
    """
    r = max(1, int(radius_px))
    dilated = F.max_pool2d(support_true, kernel_size=2 * r + 1, stride=1, padding=r)
    return (dilated - support_true).clamp(0.0, 1.0)


def hydra_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    support_loss_weight: float = 1.0,
    support_pos_weight_cap: float = 100.0,
    support_area_penalty_weight: float = 2.0,
    support_area_margin: float = 1.15,
    support_boundary_weight: float = 0.0,
    support_boundary_radius: int = 4,
    centerline_loss_weight: float = 3.0,
    centerline_focal_alpha: float = 0.75,
    centerline_focal_gamma: float = 2.0,
    geometry_loss_weight: float = 1.0,
    landmark_loss_weight: float = 10.0,
    endpoint_loss_weight: float = 10.0,
    endpoint_heatmap_loss: str = 'peak_weighted_mse',
    endpoint_focal_alpha: float = 2.0,
    endpoint_focal_beta: float = 4.0,
    arm_base_loss_multiplier: float = 1.0,
    arm_tip_loss_multiplier: float = 1.0,
    heatmap_peak_weight: float = 20.0,
    ignore_mask: torch.Tensor | None = None,
    channel_names: Sequence[str] = CHANNEL_NAMES,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint loss over configured dense channels; returns loss and per-part floats.

    ignore_mask (B, 1, H, W): pixels of unannotated neighbor stars (StarSeg,
    containment-gated). Excluded from the loss — neither positive nor negative.
    """
    keep = None if ignore_mask is None else (1.0 - ignore_mask.clamp(0.0, 1.0))
    # Support: balanced BCE + Dice + area penalty (+ optional boundary-ring weighting).
    indices = {name: channel_names.index(name) for name in channel_names}
    support_index = indices['arm_support']
    support_logits = pred[:, support_index:support_index + 1]
    support_true = target[:, support_index:support_index + 1]
    pos = support_true.sum()
    neg = (1.0 - support_true).sum() if keep is None else (keep * (1.0 - support_true)).sum()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, float(support_pos_weight_cap))
    bce_w = torch.ones_like(support_true)
    if support_boundary_weight > 0.0:
        ring = support_boundary_ring(support_true, support_boundary_radius)
        bce_w = bce_w + float(support_boundary_weight) * ring
    if keep is not None:
        bce_w = bce_w * keep
    bce_map = F.binary_cross_entropy_with_logits(support_logits, support_true, pos_weight=pos_weight, reduction='none')
    support_bce = (bce_map * bce_w).sum() / bce_w.sum().clamp_min(1.0)
    support_prob = torch.sigmoid(support_logits) if keep is None else torch.sigmoid(support_logits) * keep
    eps = 1e-6
    inter = (support_prob * support_true).sum()
    support_dice = 1.0 - (2.0 * inter + eps) / (support_prob.sum() + support_true.sum() + eps)
    support_target_frac = support_true.mean()
    support_pred_frac = support_prob.mean()
    support_area_penalty = torch.relu(support_pred_frac - support_target_frac * support_area_margin).pow(2) / support_target_frac.clamp_min(1e-6)
    support_loss = support_bce + support_dice + support_area_penalty_weight * support_area_penalty

    # Centerline: focal BCE, loss domain restricted to target support.
    support_mask = (support_true > 0.05).float()
    if 'arm_centerline' in indices:
        center_index = indices['arm_centerline']
        center_logits = pred[:, center_index:center_index + 1]
        center_true = target[:, center_index:center_index + 1]
        center_prob = torch.sigmoid(center_logits)
        bce_map = F.binary_cross_entropy_with_logits(center_logits, center_true, reduction='none')
        p_t = center_prob * center_true + (1.0 - center_prob) * (1.0 - center_true)
        alpha_t = centerline_focal_alpha * center_true + (1.0 - centerline_focal_alpha) * (1.0 - center_true)
        center_loss = (alpha_t * torch.pow(1.0 - p_t, centerline_focal_gamma) * bce_map * support_mask).sum() / support_mask.sum().clamp_min(1.0)
    else:
        center_loss = pred.sum() * 0.0

    # Geometry: support-masked L1 on tanh vectors and sigmoid progress.
    denom = support_mask.sum().clamp_min(1.0)
    vector_indices = [indices['arm_dx'], indices['arm_dy']]
    progress_index = indices['arm_progress']
    vector_loss = ((torch.tanh(pred[:, vector_indices]) - target[:, vector_indices]).abs() * support_mask).sum() / (denom * 2.0)
    progress_loss = ((torch.sigmoid(pred[:, progress_index:progress_index + 1]) - target[:, progress_index:progress_index + 1]).abs() * support_mask).sum() / denom
    geometry_loss = vector_loss + progress_loss

    # Landmarks / endpoints: peak-weighted MSE on sigmoid heatmaps.
    landmark_indices = [indices[name] for name in ('mouth', 'madreporite', 'butthole')]
    endpoint_indices = [indices[name] for name in ('arm_base', 'arm_tip')]
    landmark_loss = weighted_heatmap_mse(pred[:, landmark_indices], target[:, landmark_indices], heatmap_peak_weight, keep)
    endpoint_logits = pred[:, endpoint_indices]
    endpoint_true = target[:, endpoint_indices]
    endpoint_multipliers = (arm_base_loss_multiplier, arm_tip_loss_multiplier)
    if endpoint_heatmap_loss == 'peak_weighted_mse':
        endpoint_loss = weighted_heatmap_mse(
            endpoint_logits,
            endpoint_true,
            heatmap_peak_weight,
            keep,
            endpoint_multipliers,
        )
    elif endpoint_heatmap_loss == 'modified_focal':
        endpoint_loss = modified_focal_heatmap_loss(
            endpoint_logits,
            endpoint_true,
            alpha=endpoint_focal_alpha,
            beta=endpoint_focal_beta,
            keep=keep,
            channel_multipliers=endpoint_multipliers,
        )
    else:
        raise ValueError(f'unknown endpoint heatmap loss: {endpoint_heatmap_loss!r}')

    total = (
        support_loss_weight * support_loss
        + centerline_loss_weight * center_loss
        + geometry_loss_weight * geometry_loss
        + landmark_loss_weight * landmark_loss
        + endpoint_loss_weight * endpoint_loss
    )
    parts = {
        'support_loss': float(support_loss.detach().item()),
        'support_bce': float(support_bce.detach().item()),
        'support_dice_loss': float(support_dice.detach().item()),
        'support_area_penalty': float(support_area_penalty.detach().item()),
        'centerline_loss': float(center_loss.detach().item()),
        'geometry_loss': float(geometry_loss.detach().item()),
        'geometry_vector_l1': float(vector_loss.detach().item()),
        'geometry_progress_l1': float(progress_loss.detach().item()),
        'landmark_loss': float(landmark_loss.detach().item()),
        'endpoint_loss': float(endpoint_loss.detach().item()),
    }
    return total, parts


def hydra_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    path_metric_threshold: float = 0.5,
    path_metric_tolerances: tuple[int, ...] = (2, 4, 8),
    compute_path_metrics: bool = True,
    channel_names: Sequence[str] = CHANNEL_NAMES,
) -> dict[str, float]:
    """Per-family validation metrics over configured dense channels.

    `compute_path_metrics` gates the centerline path metric, which runs on CPU
    (skeletonization + distance transforms) and is the dominant per-batch cost.
    It is skipped on training batches and computed only for validation.
    """
    out: dict[str, float] = {}
    indices = {name: channel_names.index(name) for name in channel_names}
    with torch.no_grad():
        support_prob = torch.sigmoid(pred[:, indices['arm_support']])
        support_true = target[:, indices['arm_support']]
        p = support_prob > 0.5
        t = support_true > 0.5
        inter = (p & t).float().sum()
        union = (p | t).float().sum().clamp_min(1.0)
        dice_den = (p.float().sum() + t.float().sum()).clamp_min(1.0)
        out['support_pred_frac_t0p5'] = float(p.float().mean().item())
        out['support_target_frac_t0p5'] = float(t.float().mean().item())
        out['support_precision_t0p5'] = float((inter / p.float().sum().clamp_min(1.0)).item())
        out['support_recall_t0p5'] = float((inter / t.float().sum().clamp_min(1.0)).item())
        out['support_iou'] = float((inter / union).item())
        out['support_dice'] = float((2 * inter / dice_den).item())

        if compute_path_metrics and 'arm_centerline' in indices:
            center_index = indices['arm_centerline']
            center_prob = torch.sigmoid(pred[:, center_index])
            path_out = centerline_path_metrics(
                center_prob.detach().cpu().numpy(),
                target[:, center_index].detach().cpu().numpy(),
                threshold=path_metric_threshold,
                tolerances=path_metric_tolerances,
            )
            out.update(path_out)

        support_mask = target[:, indices['arm_support']:indices['arm_support'] + 1] > 0.05
        if bool(support_mask.any()):
            vector_indices = [indices['arm_dx'], indices['arm_dy']]
            pred_vec = torch.tanh(pred[:, vector_indices]).permute(0, 2, 3, 1)[support_mask[:, 0]]
            true_vec = target[:, vector_indices].permute(0, 2, 3, 1)[support_mask[:, 0]]
            out['geometry_vector_l1'] = float((pred_vec - true_vec).abs().mean().item())
            progress_index = indices['arm_progress']
            pred_prog = torch.sigmoid(pred[:, progress_index:progress_index + 1])[support_mask]
            true_prog = target[:, progress_index:progress_index + 1][support_mask]
            out['geometry_progress_l1'] = float((pred_prog - true_prog).abs().mean().item())

        landmark_indices = [indices[name] for name in ('mouth', 'madreporite', 'butthole')]
        endpoint_indices = [indices[name] for name in ('arm_base', 'arm_tip')]
        out['landmarks_mse'] = float(F.mse_loss(torch.sigmoid(pred[:, landmark_indices]), target[:, landmark_indices]).item())
        out['endpoints_mse'] = float(F.mse_loss(torch.sigmoid(pred[:, endpoint_indices]), target[:, endpoint_indices]).item())
    return out


def activated_pred(
    pred: torch.Tensor,
    channel_names: Sequence[str] = CHANNEL_NAMES,
) -> torch.Tensor:
    """Map raw configured dense logits to display/decode space."""
    channels = []
    for index, name in enumerate(channel_names):
        raw = pred[:, index:index + 1]
        channels.append(torch.tanh(raw) if name in {'arm_dx', 'arm_dy'} else torch.sigmoid(raw))
    return torch.cat(channels, dim=1)


def should_tighten_support(*, epochs_seen: int, bad_epochs: int, min_epochs: int, patience: int) -> bool:
    """Phase-2 trigger: support metric has stalled for `patience` epochs after a minimum run-in."""
    return epochs_seen >= min_epochs and bad_epochs >= patience


def active_support_params(args: argparse.Namespace, phase2: bool) -> dict[str, float | int]:
    if not phase2:
        return {
            'support_pos_weight_cap': args.support_pos_weight_cap,
            'support_area_margin': args.support_area_margin,
            'support_area_penalty_weight': args.support_area_penalty_weight,
            'support_boundary_weight': args.support_boundary_weight,
            'support_boundary_radius': args.support_boundary_radius,
        }
    return {
        'support_pos_weight_cap': args.phase2_support_pos_weight_cap,
        'support_area_margin': args.phase2_support_area_margin,
        'support_area_penalty_weight': args.phase2_support_area_penalty_weight,
        'support_boundary_weight': args.phase2_support_boundary_weight,
        'support_boundary_radius': args.support_boundary_radius,
    }


def split_indices(n: int, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_n = max(1, int(round(0.20 * n)))
    return idx[val_n:], idx[:val_n]


def collate(batch: list[dict]) -> dict[str, object]:
    out = {
        'image': torch.stack([b['image'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'meta': [b['meta'] for b in batch],
    }
    if 'ignore' in batch[0]:
        out['ignore'] = torch.stack([b['ignore'] for b in batch])
    if 'arm_instance' in batch[0]:
        out['arm_instance'] = torch.stack([b['arm_instance'] for b in batch])
    for key in ('arm_count', 'bridge_pairs', 'bridge_valid', 'bridge_count'):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    return out


def parse_affinity_offsets(value: str) -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    for token in value.split(','):
        if not token.strip():
            continue
        try:
            dy, dx = (int(part) for part in token.split(':', 1))
        except ValueError as exc:
            raise ValueError(f'invalid affinity offset {token!r}; expected dy:dx') from exc
        if dy == 0 and dx == 0:
            raise ValueError('affinity offset 0:0 is not useful')
        offsets.append((dy, dx))
    if not offsets:
        raise ValueError('at least one affinity offset is required')
    return tuple(offsets)


def warmstart_allowed_missing_keys(model: HydraModel) -> set[str]:
    """Keys permitted to be fresh when adding optional study heads."""
    prefixes = []
    if model.embedding_head is not None:
        prefixes.append('embedding_head.')
    if model.embedding_bandwidth_head is not None:
        prefixes.append('embedding_bandwidth_head.')
    if model.count_head is not None:
        prefixes.append('count_head.')
    return {
        key for key in model.state_dict()
        if any(key.startswith(prefix) for prefix in prefixes)
    }


def unexpected_warmstart_missing_keys(model: HydraModel, missing: Collection[str]) -> list[str]:
    allowed = warmstart_allowed_missing_keys(model)
    return [key for key in missing if key not in allowed]


def prepare_warmstart_state(
    model: HydraModel,
    state: Mapping[str, torch.Tensor],
    *,
    reset_embedding_head: bool = False,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop incompatible or explicitly reset embedding tensors before loading."""
    expected = model.state_dict()
    prepared = dict(state)
    reset: list[str] = []
    for key, value in list(prepared.items()):
        target = expected.get(key)
        if (
            key in {'support_centerline_head.out.weight', 'support_centerline_head.out.bias'}
            and target is not None
            and value.shape[0] == 2
            and target.shape[0] == 1
            and tuple(value.shape[1:]) == tuple(target.shape[1:])
        ):
            prepared[key] = value[:1].clone()
            continue
        embedding_key = key.startswith('embedding_head.')
        shape_mismatch = target is not None and tuple(value.shape) != tuple(target.shape)
        if embedding_key and (reset_embedding_head or shape_mismatch):
            prepared.pop(key)
            reset.append(key)
    return prepared, sorted(reset)


def initialize_optimizer_base_lrs(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group['base_lr'] = group.get('lr', 0.0)


def lr_schedule_scale(
    next_step: int,
    *,
    warmup_steps: int,
    start_factor: float,
    total_steps: int | None = None,
    final_factor: float = 1.0,
) -> float:
    """Linear warmup to 1.0, then cosine decay to final_factor by total_steps."""
    clamped_start = max(0.0, min(float(start_factor), 1.0))
    if warmup_steps > 0 and next_step <= warmup_steps:
        progress = min(max(float(next_step), 1.0) / float(max(1, warmup_steps)), 1.0)
        return clamped_start + (1.0 - clamped_start) * progress
    if total_steps is not None and total_steps > warmup_steps:
        t = min(max(float(next_step - warmup_steps), 0.0) / float(total_steps - warmup_steps), 1.0)
        clamped_final = max(0.0, min(float(final_factor), 1.0))
        return clamped_final + (1.0 - clamped_final) * 0.5 * (1.0 + math.cos(math.pi * t))
    return 1.0


def apply_lr_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    next_step: int,
    warmup_steps: int,
    start_factor: float,
    total_steps: int | None = None,
    final_factor: float = 1.0,
) -> None:
    scale = lr_schedule_scale(
        next_step,
        warmup_steps=warmup_steps,
        start_factor=start_factor,
        total_steps=total_steps,
        final_factor=final_factor,
    )
    for group in optimizer.param_groups:
        base_lr = float(group.get('base_lr', group.get('lr', 0.0)))
        group['lr'] = base_lr * scale


def exact_step_plan(*, steps_per_epoch: int, epochs: int, max_steps: int | None) -> tuple[int, int]:
    """Return (epochs_to_enter, optimizer_steps) for epoch or exact-step training."""
    if steps_per_epoch <= 0:
        raise ValueError('steps_per_epoch must be positive')
    if epochs <= 0:
        raise ValueError('epochs must be positive')
    if max_steps is None:
        return epochs, epochs * steps_per_epoch
    if max_steps <= 0:
        raise ValueError('max_steps must be positive')
    return math.ceil(max_steps / steps_per_epoch), max_steps


def should_validate_epoch(
    *,
    epoch: int,
    final_epoch: bool,
    cadence: int,
    snapshot_epochs: Collection[int] = (),
) -> bool:
    """Validation cadence with an unconditional final training boundary."""
    if epoch <= 0 or cadence <= 0:
        raise ValueError('epoch and cadence must be positive')
    return final_epoch or epoch % cadence == 0 or epoch in snapshot_epochs


def _backbone_depth_level(name: str) -> int:
    """ConvNeXt depth level (0=stem .. 3=deepest) for an unwrapped backbone param name."""
    for part in ('downsample_layers.', 'stages.'):
        if name.startswith(part):
            return int(name[len(part)])
    return 3  # final norm rides with the deepest stage


def optimizer_param_groups(
    model: torch.nn.Module,
    *,
    lr: float,
    backbone_lr: float,
    backbone_llrd: float = 1.0,
) -> list[dict[str, object]]:
    """Head/neck params at lr; backbone at backbone_lr with optional layer-wise
    LR decay: level k (of 0..3) trains at backbone_lr * backbone_llrd^(3 - k),
    so the deepest stage adapts fastest and the stem barely moves."""
    prefix = 'fpn.backbone.backbone.'
    by_level: dict[int, list[torch.nn.Parameter]] = {}
    other_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(prefix):
            by_level.setdefault(_backbone_depth_level(name[len(prefix):]), []).append(param)
        else:
            other_params.append(param)
    groups: list[dict[str, object]] = []
    if other_params:
        groups.append({'params': other_params, 'lr': lr})
    for level in sorted(by_level, reverse=True):
        groups.append({
            'params': by_level[level],
            'lr': backbone_lr * (backbone_llrd ** (3 - level)),
        })
    return groups


def freeze_all_but_endpoint_head(model: torch.nn.Module) -> None:
    """Leave only endpoint_head trainable for endpoint-only fine-tuning."""
    endpoint_head = getattr(model, 'endpoint_head', None)
    if endpoint_head is None:
        raise RuntimeError('endpoint-only fine-tuning requires model.endpoint_head')
    for param in model.parameters():
        param.requires_grad_(False)
    for param in endpoint_head.parameters():
        param.requires_grad_(True)


def verify_optimizer_trainable_parameters(
    model: torch.nn.Module,
    param_groups: Sequence[Mapping[str, object]],
) -> None:
    """Require optimizer groups to contain every and only trainable parameter."""
    trainable = {id(param) for param in model.parameters() if param.requires_grad}
    grouped: list[torch.nn.Parameter] = []
    for group in param_groups:
        grouped.extend(group.get('params', []))  # type: ignore[arg-type]
    grouped_ids = [id(param) for param in grouped]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError('optimizer parameter groups contain duplicate parameters')
    frozen = [param for param in grouped if not param.requires_grad]
    if frozen:
        raise RuntimeError('optimizer parameter groups contain frozen parameters')
    if set(grouped_ids) != trainable:
        raise RuntimeError('optimizer parameter groups do not exactly match trainable parameters')


def _gray(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size and (float(arr.max()) > 1.0 or float(arr.min()) < 0.0):
        arr = (arr + 1.0) / 2.0
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert('RGB')


def _label(img: Image.Image, text: str) -> Image.Image:
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, min(img.width, 260), 22), fill=(255, 255, 255))
    draw.text((4, 4), text, fill=(0, 0, 0))
    return img


# Support-conditioned channels; their pred panels are shown only where the
# model's own predicted support exceeds the decode threshold.
SUPPORT_MASKED_CHANNEL_NAMES = frozenset(
    {'arm_centerline', 'arm_dx', 'arm_dy', 'arm_progress'}
)
# Backward-compatible canonical indices for callers using the 10-channel contract.
SUPPORT_MASKED_CHANNELS = frozenset(
    index for index, name in enumerate(CHANNEL_NAMES)
    if name in SUPPORT_MASKED_CHANNEL_NAMES
)
PRED_SUPPORT_THRESHOLD = 0.5


def pred_support_masked_pred(
    pred_a: torch.Tensor,
    channel_names: Sequence[str] = CHANNEL_NAMES,
    threshold: float = PRED_SUPPORT_THRESHOLD,
) -> torch.Tensor:
    """Zero support-conditioned channels outside the model's own predicted support."""
    out = pred_a.clone()
    support_mask = (pred_a[:, 0:1] > threshold).to(pred_a.dtype)
    for ch, name in enumerate(channel_names):
        if name in SUPPORT_MASKED_CHANNEL_NAMES:
            out[:, ch:ch + 1] = out[:, ch:ch + 1] * support_mask
    return out


def save_visuals(
    path: Path,
    image: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    limit: int = 4,
    channel_names: Sequence[str] = CHANNEL_NAMES,
) -> None:
    """One row per sample: image, then true/pred pairs for dense channels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target_f = target.detach().cpu()
    pred_a = pred_support_masked_pred(
        activated_pred(pred, channel_names).detach().cpu(),
        channel_names,
    )
    image = image.detach().cpu()
    panels: list[Image.Image] = []
    n = min(limit, image.shape[0])
    for b in range(n):
        rgb = (image[b].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        panels.append(_label(Image.fromarray(rgb), f'image {b}'))
        for ch, name in enumerate(channel_names):
            tag = ' (pred support)' if name in SUPPORT_MASKED_CHANNEL_NAMES else ''
            panels.append(_label(_gray(target_f[b, ch].numpy()), f'true {name}'))
            panels.append(_label(_gray(pred_a[b, ch].numpy()), f'pred {name}{tag}'))
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    cols = 1 + 2 * len(channel_names)
    rows = int(np.ceil(len(panels) / cols))
    canvas = Image.new('RGB', (cols * w, rows * h), 'white')
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


class ModelEma:
    """Exponential moving average of model weights, validated/saved in place of
    the raw weights when enabled. Non-float buffers are copied verbatim."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Load EMA weights into the model; returns the raw weights for swap_out."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup

    @staticmethod
    @torch.no_grad()
    def swap_out(model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    args: argparse.Namespace,
    path_tolerances: tuple[int, ...],
    *,
    global_step_start: int = 0,
    total_steps: int | None = None,
    support_phase2: bool = False,
    ema: 'ModelEma | None' = None,
    embedding_loss_weight: float | None = None,
    identity_weight_scale: float = 1.0,
    compute_path_metrics: bool = True,
    max_optimizer_steps: int | None = None,
) -> tuple[float, dict[str, float], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    metric_sums: dict[str, float] = {}
    saved: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] = (None, None, None)
    steps_taken = 0
    amp_enabled = bool(args.amp and device.type == 'cuda')
    channel_names = tuple(getattr(model, 'channel_names', CHANNEL_NAMES))
    dense_channel_count = len(channel_names)
    for batch in loader:
        if training and max_optimizer_steps is not None and steps_taken >= max_optimizer_steps:
            break
        image = batch['image'].to(device, non_blocking=True)  # type: ignore[index]
        target = batch['target'].to(device, non_blocking=True)  # type: ignore[index]
        ignore = batch.get('ignore')
        if ignore is not None:
            ignore = ignore.to(device, non_blocking=True)
        arm_instance = batch.get('arm_instance')
        if arm_instance is not None:
            arm_instance = arm_instance.to(device, non_blocking=True)
        arm_count = batch.get('arm_count')
        if arm_count is not None:
            arm_count = arm_count.to(device, non_blocking=True)
        bridge_pairs = batch.get('bridge_pairs')
        bridge_valid = batch.get('bridge_valid')
        if bridge_pairs is not None:
            bridge_pairs = bridge_pairs.to(device, non_blocking=True)
        if bridge_valid is not None:
            bridge_valid = bridge_valid.to(device, non_blocking=True)
        if training:
            apply_lr_schedule(
                optimizer,
                next_step=global_step_start + steps_taken + 1,
                warmup_steps=args.lr_warmup_steps,
                start_factor=args.lr_warmup_start_factor,
                total_steps=total_steps,
                final_factor=args.lr_final_factor,
            )
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            if hasattr(model, 'forward_with_aux'):
                pred, aux = model.forward_with_aux(image)  # type: ignore[attr-defined]
            else:
                pred, aux = model(image), {}
            loss, parts = hydra_loss(
                pred,
                target,
                support_loss_weight=args.support_loss_weight,
                **active_support_params(args, support_phase2),
                centerline_loss_weight=args.centerline_loss_weight,
                centerline_focal_alpha=args.centerline_focal_alpha,
                centerline_focal_gamma=args.centerline_focal_gamma,
                geometry_loss_weight=args.geometry_loss_weight,
                landmark_loss_weight=args.landmark_loss_weight,
                endpoint_loss_weight=args.endpoint_loss_weight,
                endpoint_heatmap_loss=getattr(args, 'endpoint_heatmap_loss', 'peak_weighted_mse'),
                endpoint_focal_alpha=getattr(args, 'endpoint_focal_alpha', 2.0),
                endpoint_focal_beta=getattr(args, 'endpoint_focal_beta', 4.0),
                arm_base_loss_multiplier=getattr(args, 'arm_base_loss_multiplier', 1.0),
                arm_tip_loss_multiplier=getattr(args, 'arm_tip_loss_multiplier', 1.0),
                heatmap_peak_weight=args.heatmap_peak_weight,
                ignore_mask=ignore,
                channel_names=channel_names,
            )
        # Arm-identity embedding loss (fp32, outside autocast for numerical
        # stability). Added to the joint total so the backbone adapts to arm
        # separation while the 10-channel heads keep their supervision.
        embed = None
        if args.embedding_dim > 0:
            embed = pred[:, dense_channel_count:dense_channel_count + args.embedding_dim].float()
        raw_bandwidth = (
            pred[:, dense_channel_count + args.embedding_dim:
                 dense_channel_count + args.embedding_dim + 1].float()
            if bool(getattr(args, 'embedding_bandwidth', False)) else None
        )
        ew = args.embedding_loss_weight if embedding_loss_weight is None else embedding_loss_weight
        if embed is not None and arm_instance is not None and ew != 0:
            embed_loss, embed_parts = discriminative_embedding_loss(
                embed,
                arm_instance,
                delta_v=args.embed_delta_v,
                delta_d=args.embed_delta_d,
                w_pull=args.embed_w_pull,
                w_push=args.embed_w_push,
                w_reg=args.embed_w_reg,
            )
            loss = loss + ew * embed_loss
            parts['embedding_loss_weight'] = float(ew)
            parts.update(embed_parts)
        if (
            embed is not None
            and raw_bandwidth is not None
            and arm_instance is not None
            and args.bandwidth_loss_weight != 0
        ):
            bandwidth_loss, bandwidth_parts = embedding_bandwidth_loss(
                embed,
                raw_bandwidth,
                arm_instance,
                sigma_floor=args.bandwidth_sigma_floor,
                consistency_weight=args.bandwidth_consistency_weight,
                max_pixels_per_arm=args.bandwidth_max_pixels_per_arm,
            )
            weight = args.bandwidth_loss_weight * identity_weight_scale
            loss = loss + weight * bandwidth_loss
            parts['bandwidth_loss_weight'] = float(weight)
            parts.update(bandwidth_parts)
        if embed is not None and arm_instance is not None and args.prototype_loss_weight != 0:
            prototype_loss, prototype_parts = prototypical_embedding_loss(
                embed, arm_instance,
                temperature=args.prototype_temperature,
                max_pixels_per_arm=args.prototype_max_pixels_per_arm,
            )
            weight = args.prototype_loss_weight * identity_weight_scale
            loss = loss + weight * prototype_loss
            parts['prototype_loss_weight'] = float(weight)
            parts.update(prototype_parts)
        if embed is not None and arm_instance is not None and args.affinity_loss_weight != 0:
            affinity_loss, affinity_parts = local_affinity_loss(
                embed, arm_instance,
                offsets=parse_affinity_offsets(args.affinity_offsets),
                margin=args.affinity_margin,
            )
            weight = args.affinity_loss_weight * identity_weight_scale
            loss = loss + weight * affinity_loss
            parts['affinity_loss_weight'] = float(weight)
            parts.update(affinity_parts)
        if (embed is not None and bridge_pairs is not None and bridge_valid is not None
                and args.continuity_loss_weight != 0):
            continuity_loss, continuity_parts = crossing_continuity_loss(
                embed, bridge_pairs, bridge_valid,
            )
            weight = args.continuity_loss_weight * identity_weight_scale
            loss = loss + weight * continuity_loss
            parts['continuity_loss_weight'] = float(weight)
            parts.update(continuity_parts)
        count_logits = aux.get('count_logits')
        if count_logits is not None and arm_count is not None:
            if bool((arm_count >= count_logits.shape[1]).any()):
                raise ValueError('arm_count exceeds configured count classes')
            parts['count_accuracy'] = float(
                (count_logits.argmax(1) == arm_count).float().mean().detach().item()
            )
            if args.count_loss_weight != 0:
                count_loss = F.cross_entropy(count_logits.float(), arm_count)
                loss = loss + args.count_loss_weight * identity_weight_scale * count_loss
                parts['count_loss'] = float(count_loss.detach().item())
                parts['count_loss_weight'] = float(args.count_loss_weight * identity_weight_scale)
        if training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if ema is not None:
                ema.update(model)
            steps_taken += 1
        bs = int(image.shape[0])
        total_loss += float(loss.detach().item()) * bs
        total_n += bs
        metrics = hydra_metrics(
            pred.float(),
            target,
            path_metric_threshold=args.path_metric_threshold,
            path_metric_tolerances=path_tolerances,
            compute_path_metrics=compute_path_metrics,
            channel_names=channel_names,
        )
        metrics.update(parts)
        for k, v in metrics.items():
            if not np.isnan(v):
                metric_sums[k] = metric_sums.get(k, 0.0) + v * bs
        if saved[0] is None:
            saved = (image.detach().cpu(), pred.detach().float().cpu(), target.detach().cpu())
    mean_metrics = {k: v / max(1, total_n) for k, v in metric_sums.items()}
    return total_loss / max(1, total_n), mean_metrics, saved[0], saved[1], saved[2], steps_taken


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def split_indices(size: int, seed: int) -> tuple[list[int], list[int]]:
    """Return the legacy deterministic 80/20 row split."""
    if size < 2:
        raise ValueError(f'At least two rows are required for a train/validation split, got {size}')
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    val_size = max(1, round(size * 0.2))
    return indices[val_size:], indices[:val_size]


def main() -> int:
    args = parse_args()
    if args.no_centerline:
        args.centerline_loss_weight = 0.0
        args.train_path_metrics = False
        if args.best_selector == 'path_f1':
            raise ValueError('--best-selector=path_f1 requires the centerline channel')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    path_tolerances = tuple(int(part.strip()) for part in args.path_metric_tolerances.split(',') if part.strip())
    dense_contract = 'full9ch' if args.no_centerline else 'full10ch'
    out_dir = args.out_dir / f'{dense_contract}_{time.strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=False)

    common_ds = dict(
        image_size=(args.image_size, args.image_size),
        support_arm_width=args.support_arm_width,
        centerline_radius=args.centerline_radius,
        arm_base_sigma=args.arm_base_sigma,
        arm_tip_sigma=args.arm_tip_sigma,
        point_sigma=args.point_sigma,
        ignore_shapes_dir=args.ignore_shapes_dir,
        target_cache_dir=args.target_cache_dir,
        image_cache_dir=args.image_cache_dir,
        max_bridge_pairs=args.max_bridge_pairs,
        bridge_offset=args.bridge_offset,
        include_centerline=not args.no_centerline,
    )
    train_target_kwargs = {
        'trim_overlapping_arm_bases': args.trim_overlapping_arm_bases,
        'base_trim_kernel_radius_sigmas': args.base_trim_kernel_radius_sigmas,
        'base_trim_max_fraction': args.base_trim_max_fraction,
        'base_trim_min_remaining_length': args.base_trim_min_remaining_length,
    }
    augment_kwargs = moderate_augment_kwargs() if args.augment == 'moderate' else {'augment': False}
    if args.degrade_p > 0 and augment_kwargs.get('augment'):
        augment_kwargs.update({'degrade_p': args.degrade_p, 'degrade_min_native': args.degrade_min_native})
    if args.train_index is not None and args.val_index is not None:
        split_mode = 'explicit_indexes'
        resolved_train_index = args.train_index.expanduser().resolve()
        resolved_val_index = args.val_index.expanduser().resolve()
        train_ds_full = HydraDenseDataset(
            resolved_train_index, **common_ds, **train_target_kwargs, **augment_kwargs
        )
        val_ds_full = HydraDenseDataset(resolved_val_index, **common_ds, augment=False)
        train_idx = list(range(len(train_ds_full)))
        val_idx = list(range(len(val_ds_full)))
    else:
        split_mode = 'legacy_seeded_80_20'
        resolved_train_index = args.index.expanduser().resolve()
        resolved_val_index = resolved_train_index
        train_ds_full = HydraDenseDataset(
            resolved_train_index, **common_ds, **train_target_kwargs, **augment_kwargs
        )
        val_ds_full = HydraDenseDataset(resolved_val_index, **common_ds, augment=False)
        train_idx, val_idx = split_indices(len(train_ds_full), args.seed)
    if args.train_subset is not None:
        train_idx = train_idx[: args.train_subset]
    if args.val_subset is not None:
        val_idx = val_idx[: args.val_subset]
    train_loader = DataLoader(
        Subset(train_ds_full, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device.startswith('cuda'),
    )
    val_loader = DataLoader(
        Subset(val_ds_full, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device.startswith('cuda'),
    )
    planned_epochs, total_steps = exact_step_plan(
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    snapshot_epochs = validate_snapshot_epochs(
        args.save_at_epochs,
        planned_epochs=planned_epochs,
    )
    train_index_sha256 = file_sha256(resolved_train_index)
    val_index_sha256 = (
        train_index_sha256
        if resolved_val_index == resolved_train_index
        else file_sha256(resolved_val_index)
    )

    device = torch.device(args.device)
    model = HydraModel(
        embedding_dim=args.embedding_dim,
        embedding_bandwidth=args.embedding_bandwidth,
        count_classes=args.count_classes,
        include_centerline=not args.no_centerline,
        activation_checkpointing=args.activation_checkpointing,
    ).to(device)
    warmstart_info: dict[str, list[str]] | None = None
    if args.init_checkpoint is not None:
        ckpt = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        state, reset_keys = prepare_warmstart_state(
            model,
            state,
            reset_embedding_head=args.reset_embedding_head,
        )
        incompatible = model.load_state_dict(state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        disallowed_missing = unexpected_warmstart_missing_keys(model, missing)
        if disallowed_missing:
            raise RuntimeError(f'Warm-start checkpoint missing required weights: {disallowed_missing[:8]}')
        warmstart_info = {
            'missing': missing,
            'unexpected': unexpected,
            'reset_embedding_keys': reset_keys,
        }
        print(json.dumps({'warmstart_from': str(args.init_checkpoint),
                          'missing_keys': len(missing), 'unexpected_keys': len(unexpected),
                          'reset_embedding_keys': len(reset_keys)}))
    if not args.freeze_backbone:
        model.unfreeze_full_backbone()
    if args.endpoint_only:
        freeze_all_but_endpoint_head(model)
    # Preserve warm-started output biases; only initialize them for a fresh run.
    if args.output_bias_init is not None and args.init_checkpoint is None:
        model.initialize_heatmap_output_biases(args.output_bias_init)
    param_groups = optimizer_param_groups(model, lr=args.lr, backbone_lr=args.backbone_lr, backbone_llrd=args.backbone_llrd)
    verify_optimizer_trainable_parameters(model, param_groups)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    initialize_optimizer_base_lrs(optimizer)
    scaler = torch.amp.GradScaler(device.type, enabled=bool(args.amp and device.type == 'cuda'))
    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None

    config = vars(args).copy()
    config.update({
        'out_dir': str(out_dir),
        'split_mode': split_mode,
        'resolved_train_index': str(resolved_train_index),
        'resolved_val_index': str(resolved_val_index),
        'train_index_sha256': train_index_sha256,
        'val_index_sha256': val_index_sha256,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'channel_names': list(model.channel_names),
        'output_spec': model.output_spec,
        'trainable_parameter_groups': model.trainable_parameter_groups(),
        'optimizer_group_base_lrs': [group.get('base_lr') for group in optimizer.param_groups],
        'warmstart': None if warmstart_info is None else {k: v[:8] for k, v in warmstart_info.items()},
        'requested_max_steps': args.max_steps,
        'planned_optimizer_steps': total_steps,
        'actual_optimizer_steps': None,
        'planned_epochs': planned_epochs,
        'validation_cadence_epochs': args.validate_every_epochs,
        'requested_depth_snapshot_epochs': list(snapshot_epochs),
        'written_depth_snapshots': [],
    })
    config_identity = training_config_identity(config)
    config['training_config_identity'] = config_identity
    write_json(out_dir / 'config.json', config)
    write_json(out_dir / 'status.json', {
        'state': 'running',
        'requested_max_steps': args.max_steps,
        'planned_optimizer_steps': total_steps,
        'actual_optimizer_steps': 0,
        'validation_cadence_epochs': args.validate_every_epochs,
        'validations_completed': 0,
        'requested_depth_snapshot_epochs': list(snapshot_epochs),
        'written_depth_snapshots': [],
        'training_config_identity': config_identity,
    })
    print(json.dumps({'train_size': len(train_idx), 'val_size': len(val_idx), 'steps_per_epoch': len(train_loader), 'out_dir': str(out_dir)}))

    metrics_path = out_dir / 'metrics.csv'
    best_val = float('inf')
    best_selector_value = -float('inf')
    best_selector_epoch = 0
    best_embedding_val = float('inf')
    best_embedding_epoch = 0
    global_steps = 0
    validations_completed = 0
    written_depth_snapshots: list[dict[str, object]] = []
    support_phase2 = False
    support_phase2_start_epoch: int | None = None
    plateau_best = -float('inf')
    plateau_bad_epochs = 0
    with metrics_path.open('w', newline='', encoding='utf-8') as f:
        writer: csv.DictWriter | None = None
        for epoch in range(1, planned_epochs + 1):
            if args.embedding_dim > 0 and args.embedding_warmup_epochs > 0:
                identity_scale = min(1.0, epoch / float(args.embedding_warmup_epochs))
            else:
                identity_scale = 1.0
            epoch_embed_weight = args.embedding_loss_weight * identity_scale
            train_loss, train_metrics, _, _, _, steps_taken = run_epoch(
                model, train_loader, device, optimizer, scaler, args, path_tolerances,
                global_step_start=global_steps,
                total_steps=total_steps,
                support_phase2=support_phase2,
                ema=ema,
                embedding_loss_weight=epoch_embed_weight,
                identity_weight_scale=identity_scale,
                compute_path_metrics=args.train_path_metrics,
                max_optimizer_steps=total_steps - global_steps,
            )
            global_steps += steps_taken
            final_epoch = global_steps >= total_steps
            if not should_validate_epoch(
                epoch=epoch,
                final_epoch=final_epoch,
                cadence=args.validate_every_epochs,
                snapshot_epochs=snapshot_epochs,
            ):
                continue
            validations_completed += 1
            raw_backup = ema.swap_in(model) if ema is not None else None
            with torch.no_grad():
                val_loss, val_metrics, val_img, val_pred, val_target, _ = run_epoch(
                    model, val_loader, device, None, None, args, path_tolerances,
                    support_phase2=support_phase2,
                    embedding_loss_weight=epoch_embed_weight,
                    identity_weight_scale=identity_scale,
                )
            if args.support_tighten_on_plateau and not support_phase2:
                plateau_metric = float(val_metrics['support_dice'])
                if plateau_metric > plateau_best + args.support_plateau_min_delta:
                    plateau_best = plateau_metric
                    plateau_bad_epochs = 0
                else:
                    plateau_bad_epochs += 1
                if should_tighten_support(
                    epochs_seen=epoch,
                    bad_epochs=plateau_bad_epochs,
                    min_epochs=args.support_plateau_min_epochs,
                    patience=args.support_plateau_patience,
                ):
                    support_phase2 = True
                    support_phase2_start_epoch = epoch + 1
            row: dict[str, object] = {
                'epoch': epoch,
                'global_steps': global_steps,
                'validation_number': validations_completed,
                'steps_per_epoch': len(train_loader),
                'optimizer_group_lrs': [float(group.get('lr', 0.0)) for group in optimizer.param_groups],
                'support_phase2_active': float(support_phase2),
                'support_phase2_start_epoch': support_phase2_start_epoch if support_phase2_start_epoch is not None else '',
                'support_plateau_bad_epochs': plateau_bad_epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            row.update({f'train_{k}': v for k, v in train_metrics.items()})
            row.update({f'val_{k}': v for k, v in val_metrics.items()})
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(json.dumps(row, sort_keys=True))
            if args.save_visuals_every and validations_completed % args.save_visuals_every == 0 and val_img is not None and val_pred is not None and val_target is not None:
                save_visuals(
                    out_dir / 'visuals' / f'epoch_{epoch:03d}.png',
                    val_img,
                    val_pred,
                    val_target,
                    args.num_visuals,
                    model.channel_names,
                )
            best_val = min(best_val, val_loss)
            if args.best_selector == 'path_f1':
                best_score = float(val_metrics.get('centerline_path_f1_tol4', 0.0))
                is_best = best_score > best_selector_value
            else:
                best_score = -val_loss
                is_best = best_score > best_selector_value
            # With EMA enabled the EMA weights are currently swapped in, so the
            # 'model' key in checkpoints is the EMA model (what eval should use);
            # the raw weights ride along for resuming.
            ckpt_payload = {
                'model': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'optimizer_steps': global_steps,
                'val_loss': val_loss,
                'output_spec': model.output_spec,
                'best_selector': args.best_selector,
                'selector_value': best_score,
                'pixel_instance_selector': None,
            }
            if raw_backup is not None:
                ckpt_payload['model_raw'] = raw_backup
            if epoch in snapshot_epochs:
                snapshot_name = f'depth_epoch_{epoch:03d}.pt'
                snapshot_path = out_dir / snapshot_name
                if snapshot_path.exists():
                    raise FileExistsError(f'Refusing to overwrite immutable snapshot: {snapshot_path}')
                snapshot_record: dict[str, object] = {
                    'epoch': epoch,
                    'optimizer_steps': global_steps,
                    'path': snapshot_name,
                }
                snapshot_payload = {
                    **ckpt_payload,
                    'config': {
                        **config,
                        'written_depth_snapshots': [
                            *written_depth_snapshots,
                            snapshot_record,
                        ],
                    },
                    'snapshot_role': 'depth_transfer',
                    'source_train_index': str(resolved_train_index),
                    'source_val_index': str(resolved_val_index),
                    'source_train_index_sha256': train_index_sha256,
                    'source_val_index_sha256': val_index_sha256,
                    'training_config_identity': config_identity,
                }
                torch.save(snapshot_payload, snapshot_path)
                written_depth_snapshots.append(snapshot_record)
                config['written_depth_snapshots'] = list(written_depth_snapshots)
            if is_best:
                best_selector_value = best_score
                best_selector_epoch = epoch
                if args.save_checkpoints:
                    torch.save(ckpt_payload, out_dir / 'best.pt')
                    selector_path = {
                        'path_f1': 'best_path.pt',
                        'val_loss': 'best_val_loss.pt',
                    }[args.best_selector]
                    torch.save(ckpt_payload, out_dir / selector_path)
            embedding_val = float(val_metrics.get('embedding_loss', float('inf')))
            if args.embedding_dim > 0 and embedding_val < best_embedding_val:
                best_embedding_val = embedding_val
                best_embedding_epoch = epoch
                if args.save_checkpoints:
                    torch.save(ckpt_payload, out_dir / 'best_embedding.pt')
            if args.save_checkpoints or (final_epoch and args.max_steps is not None):
                torch.save(ckpt_payload, out_dir / 'last.pt')
            if args.save_checkpoints:
                if args.save_every > 0 and validations_completed % args.save_every == 0:
                    torch.save(ckpt_payload, out_dir / f'epoch_{epoch:03d}.pt')
            if raw_backup is not None:
                ModelEma.swap_out(model, raw_backup)
            write_json(out_dir / 'status.json', {
                'state': 'running' if not final_epoch else 'completed',
                'requested_max_steps': args.max_steps,
                'planned_optimizer_steps': total_steps,
                'actual_optimizer_steps': global_steps,
                'validation_cadence_epochs': args.validate_every_epochs,
                'validations_completed': validations_completed,
                'last_validated_epoch': epoch,
                'requested_depth_snapshot_epochs': list(snapshot_epochs),
                'written_depth_snapshots': written_depth_snapshots,
                'training_config_identity': config_identity,
            })
            if final_epoch:
                break
    write_json(out_dir / 'summary.json', {
        'best_val_loss': best_val,
        'best_selector': args.best_selector,
        'best_selector_value': best_selector_value,
        'best_selector_epoch': best_selector_epoch,
        'pixel_instance_selector': None,
        'best_embedding_val_loss': best_embedding_val if args.embedding_dim > 0 else None,
        'best_embedding_epoch': best_embedding_epoch if args.embedding_dim > 0 else None,
        'support_phase2_start_epoch': support_phase2_start_epoch,
        'requested_max_steps': args.max_steps,
        'planned_optimizer_steps': total_steps,
        'actual_optimizer_steps': global_steps,
        'validation_cadence_epochs': args.validate_every_epochs,
        'validations_completed': validations_completed,
        'requested_depth_snapshot_epochs': list(snapshot_epochs),
        'written_depth_snapshots': written_depth_snapshots,
        'training_config_identity': config_identity,
        'out_dir': str(out_dir),
    })
    config['actual_optimizer_steps'] = global_steps
    config['validations_completed'] = validations_completed
    write_json(out_dir / 'config.json', config)
    print('WROTE', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
