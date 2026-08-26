# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Discriminative (associative) embedding loss for per-pixel arm identity.

The model emits a D-dim embedding per pixel (channels 10:10+D of the hydra
output). The ground-truth is an arbitrary per-image instance index map
(``arm_instance``): ``0`` = background, ``1..n`` = arms, ``-1`` = ignore
(crossings / unannotated neighbor stars). There is no canonical numbering, so
the loss only asks that pixels of the same arm cluster together and different
arms repel (De Brabandere et al., "Semantic Instance Segmentation with a
Discriminative Loss Function"). Canonical numbering is recovered downstream
from geometry, not here.

The loss is computed per image (an arm's embedding has no meaning across
images) and only over instance ids ``> 0``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _deterministic_indices(indices: torch.Tensor, limit: int) -> torch.Tensor:
    """Evenly sample sorted flat indices without using process RNG state."""
    if limit <= 0 or indices.numel() <= limit:
        return indices
    positions = torch.linspace(
        0, indices.numel() - 1, steps=limit, device=indices.device
    ).round().long()
    return indices[positions]


def prototypical_embedding_loss(
    embed: torch.Tensor,
    arm_instance: torch.Tensor,
    *,
    temperature: float = 0.1,
    max_pixels_per_arm: int = 256,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Balanced per-image prototype/query classification on unit embeddings.

    Each arm is weighted equally regardless of area. Sorted raster indices are
    deterministically subsampled and alternated between prototype and query
    sets, so repeated calls produce exactly the same loss.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    unit = F.normalize(embed.float(), dim=1)
    image_losses: list[torch.Tensor] = []
    arm_losses: list[torch.Tensor] = []
    correct = queries = used_arms = used_pixels = 0
    for features, labels_2d in zip(unit, arm_instance):
        labels = labels_2d.reshape(-1)
        flat = features.reshape(features.shape[0], -1).t()
        ids = torch.unique(labels[labels > 0], sorted=True)
        prototypes: list[torch.Tensor] = []
        query_sets: list[torch.Tensor] = []
        kept_ids: list[torch.Tensor] = []
        for arm_id in ids:
            idx = torch.nonzero(labels == arm_id, as_tuple=False).flatten()
            idx = _deterministic_indices(idx, max_pixels_per_arm)
            if idx.numel() < 2:
                continue
            proto_idx, query_idx = idx[::2], idx[1::2]
            prototypes.append(F.normalize(flat[proto_idx].mean(0), dim=0))
            query_sets.append(flat[query_idx])
            kept_ids.append(arm_id)
            used_pixels += int(idx.numel())
        if not prototypes:
            continue
        proto = torch.stack(prototypes)
        per_arm: list[torch.Tensor] = []
        for class_index, query in enumerate(query_sets):
            logits = query @ proto.t() / float(temperature)
            target = torch.full(
                (query.shape[0],), class_index, dtype=torch.long, device=query.device
            )
            value = F.cross_entropy(logits, target)
            per_arm.append(value)
            arm_losses.append(value)
            correct += int((logits.argmax(1) == target).sum().item())
            queries += int(query.shape[0])
        image_losses.append(torch.stack(per_arm).mean())
        used_arms += len(kept_ids)
    loss = torch.stack(image_losses).mean() if image_losses else embed.sum() * 0.0
    return loss, {
        'prototype_loss': float(loss.detach().item()),
        'prototype_accuracy': correct / max(queries, 1),
        'prototype_queries': float(queries),
        'prototype_arms': float(used_arms),
        'prototype_images': float(len(image_losses)),
        'prototype_sampled_pixels': float(used_pixels),
        'prototype_temperature': float(temperature),
        'prototype_mean_arm_loss': (
            float(torch.stack(arm_losses).mean().detach().item()) if arm_losses else 0.0
        ),
    }


def local_affinity_loss(
    embed: torch.Tensor,
    arm_instance: torch.Tensor,
    *,
    offsets: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (0, 2), (2, 0)),
    margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Balanced local cosine affinity for same-arm and hard-negative pairs."""
    unit = F.normalize(embed.float(), dim=1)
    positive_terms: list[torch.Tensor] = []
    negative_terms: list[torch.Tensor] = []
    positive_pairs = negative_pairs = 0
    _, _, h, w = unit.shape
    for dy, dx in offsets:
        if dy == 0 and dx == 0:
            continue
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        if y0 >= y1 or x0 >= x1:
            continue
        a = unit[:, :, y0:y1, x0:x1]
        b = unit[:, :, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        la = arm_instance[:, y0:y1, x0:x1]
        lb = arm_instance[:, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        valid = (la > 0) & (lb > 0)
        same = valid & (la == lb)
        different = valid & (la != lb)
        similarity = (a * b).sum(1)
        if bool(same.any()):
            positive_terms.append((1.0 - similarity[same]).mean())
            positive_pairs += int(same.sum().item())
        if bool(different.any()):
            negative_terms.append(F.relu(similarity[different] - float(margin)).mean())
            negative_pairs += int(different.sum().item())
    zero = embed.sum() * 0.0
    positive = torch.stack(positive_terms).mean() if positive_terms else zero
    negative = torch.stack(negative_terms).mean() if negative_terms else zero
    loss = positive + negative
    return loss, {
        'affinity_loss': float(loss.detach().item()),
        'affinity_positive_loss': float(positive.detach().item()),
        'affinity_negative_loss': float(negative.detach().item()),
        'affinity_positive_pairs': float(positive_pairs),
        'affinity_negative_pairs': float(negative_pairs),
        'affinity_offsets': float(len(offsets)),
    }


def embedding_bandwidth_loss(
    embed: torch.Tensor,
    raw_bandwidth: torch.Tensor,
    arm_instance: torch.Tensor,
    *,
    sigma_floor: float = 0.05,
    consistency_weight: float = 0.05,
    max_pixels_per_arm: int = 1024,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Learn an instance bandwidth through balanced Gaussian membership.

    Each ground-truth arm supplies an embedding centroid and the mean predicted
    scalar sigma of its pixels. Labeled pixels are classified with normalized
    heteroscedastic Gaussian scores. The log-sigma normalization prevents a
    cluster from winning merely by predicting an arbitrarily broad bandwidth.
    """
    if raw_bandwidth.ndim != 4 or raw_bandwidth.shape[1] != 1:
        raise ValueError('raw_bandwidth must have shape (B, 1, H, W)')
    if embed.shape[0] != raw_bandwidth.shape[0] or embed.shape[-2:] != raw_bandwidth.shape[-2:]:
        raise ValueError('embedding and bandwidth spatial shapes must match')
    if arm_instance.shape != (embed.shape[0], *embed.shape[-2:]):
        raise ValueError('arm_instance shape must match embedding spatial shape')
    if sigma_floor <= 0:
        raise ValueError('sigma_floor must be positive')
    if consistency_weight < 0:
        raise ValueError('consistency_weight must be non-negative')

    sigma_map = F.softplus(raw_bandwidth.float()) + float(sigma_floor)
    image_losses: list[torch.Tensor] = []
    membership_losses: list[torch.Tensor] = []
    consistency_losses: list[torch.Tensor] = []
    accuracies: list[torch.Tensor] = []
    all_sigmas: list[torch.Tensor] = []
    used_arms = used_pixels = 0
    embedding_dim = embed.shape[1]

    for features_2d, sigma_2d, labels_2d in zip(embed.float(), sigma_map, arm_instance):
        labels = labels_2d.reshape(-1)
        features = features_2d.reshape(embedding_dim, -1).t()
        pixel_sigmas = sigma_2d.reshape(-1)
        ids = torch.unique(labels[labels > 0], sorted=True)
        if ids.numel() < 2:
            continue

        centroids = torch.stack([features[labels == arm_id].mean(0) for arm_id in ids])
        arm_sigmas = torch.stack([pixel_sigmas[labels == arm_id].mean() for arm_id in ids])
        all_sigmas.append(arm_sigmas)

        per_arm_membership: list[torch.Tensor] = []
        per_arm_consistency: list[torch.Tensor] = []
        per_arm_accuracy: list[torch.Tensor] = []
        sigma_sq = arm_sigmas.square().clamp_min(float(sigma_floor) ** 2)
        normalizer = float(embedding_dim) * torch.log(arm_sigmas.clamp_min(float(sigma_floor)))
        for class_index, arm_id in enumerate(ids):
            indices = torch.nonzero(labels == arm_id, as_tuple=False).flatten()
            indices = _deterministic_indices(indices, max_pixels_per_arm)
            query = features[indices]
            distance_sq = (query[:, None, :] - centroids[None, :, :]).square().sum(-1)
            logits = -distance_sq / (2.0 * sigma_sq[None, :]) - normalizer[None, :]
            target = torch.full(
                (len(indices),), class_index, dtype=torch.long, device=embed.device
            )
            per_arm_membership.append(F.cross_entropy(logits, target))
            per_arm_accuracy.append((logits.argmax(1) == target).float().mean())
            query_log_sigma = torch.log(pixel_sigmas[indices].clamp_min(float(sigma_floor)))
            target_log_sigma = torch.log(arm_sigmas[class_index].clamp_min(float(sigma_floor)))
            per_arm_consistency.append(
                F.smooth_l1_loss(query_log_sigma, target_log_sigma.expand_as(query_log_sigma))
            )
            used_pixels += int(indices.numel())

        membership = torch.stack(per_arm_membership).mean()
        consistency = torch.stack(per_arm_consistency).mean()
        image_losses.append(membership + float(consistency_weight) * consistency)
        membership_losses.append(membership.detach())
        consistency_losses.append(consistency.detach())
        accuracies.append(torch.stack(per_arm_accuracy).mean().detach())
        used_arms += int(ids.numel())

    zero = embed.sum() * 0.0 + raw_bandwidth.sum() * 0.0
    loss = torch.stack(image_losses).mean() if image_losses else zero
    sigma_values = torch.cat(all_sigmas) if all_sigmas else sigma_map.new_empty(0)
    return loss, {
        'bandwidth_loss': float(loss.detach().item()),
        'bandwidth_membership_loss': (
            float(torch.stack(membership_losses).mean().item()) if membership_losses else 0.0
        ),
        'bandwidth_consistency_loss': (
            float(torch.stack(consistency_losses).mean().item()) if consistency_losses else 0.0
        ),
        'bandwidth_membership_accuracy': (
            float(torch.stack(accuracies).mean().item()) if accuracies else 0.0
        ),
        'bandwidth_mean_sigma': (
            float(sigma_values.mean().detach().item()) if sigma_values.numel() else 0.0
        ),
        'bandwidth_min_sigma': (
            float(sigma_values.min().detach().item()) if sigma_values.numel() else 0.0
        ),
        'bandwidth_max_sigma': (
            float(sigma_values.max().detach().item()) if sigma_values.numel() else 0.0
        ),
        'bandwidth_arms': float(used_arms),
        'bandwidth_sampled_pixels': float(used_pixels),
    }


def crossing_continuity_loss(
    embed: torch.Tensor,
    bridge_pairs: torch.Tensor,
    bridge_valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pull normalized embeddings together at same-arm crossing bridge endpoints.

    ``bridge_pairs`` is ``(B, M, 2, 2)`` in pixel ``(x, y)`` coordinates.
    """
    if bridge_pairs.ndim != 4 or bridge_pairs.shape[-2:] != (2, 2):
        raise ValueError('bridge_pairs must have shape (B, M, 2, 2)')
    b, _, h, w = embed.shape
    coords = bridge_pairs.to(device=embed.device, dtype=embed.dtype)
    gx = 2.0 * coords[..., 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * coords[..., 1] / max(h - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1).reshape(b, -1, 1, 2)
    sampled = F.grid_sample(
        F.normalize(embed.float(), dim=1), grid.float(),
        mode='bilinear', padding_mode='border', align_corners=True,
    )
    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, -1, 2, embed.shape[1])
    cosine = (F.normalize(sampled[:, :, 0], dim=-1) *
              F.normalize(sampled[:, :, 1], dim=-1)).sum(-1)
    valid = bridge_valid.to(device=embed.device, dtype=torch.bool)
    loss = (1.0 - cosine[valid]).mean() if bool(valid.any()) else embed.sum() * 0.0
    return loss, {
        'continuity_loss': float(loss.detach().item()),
        'continuity_pairs': float(valid.sum().item()),
        'continuity_mean_cosine': (
            float(cosine[valid].mean().detach().item()) if bool(valid.any()) else 0.0
        ),
    }


@torch.no_grad()
def embedding_centroid_accuracy(
    embed: torch.Tensor,
    arm_instance: torch.Tensor,
) -> float:
    """Mean per-image labeled-pixel accuracy under nearest arm centroid."""
    accuracies: list[torch.Tensor] = []
    for image_embed, image_instance in zip(embed, arm_instance):
        labels = image_instance.reshape(-1)
        ids = torch.unique(labels)
        ids = ids[ids > 0]
        if ids.numel() < 2:
            continue
        flat = image_embed.reshape(image_embed.shape[0], -1).t()
        centroids = torch.stack([flat[labels == arm_id].mean(dim=0) for arm_id in ids])
        labeled = labels > 0
        distances = torch.cdist(flat[labeled], centroids)
        predictions = ids[distances.argmin(dim=1)]
        accuracies.append((predictions == labels[labeled]).float().mean())
    if not accuracies:
        return 0.0
    return float(torch.stack(accuracies).mean().item())


def _single_image_loss(
    embed: torch.Tensor,          # (D, H, W)
    instance: torch.Tensor,       # (H, W) long
    *,
    delta_v: float,
    delta_d: float,
    w_pull: float,
    w_push: float,
    w_reg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = embed.shape[0]
    flat = embed.reshape(d, -1)                     # (D, H*W)
    labels = instance.reshape(-1)                   # (H*W,)
    ids = torch.unique(labels)
    ids = ids[ids > 0]
    zero = embed.new_tensor(0.0)
    if ids.numel() == 0:
        return zero, zero, zero

    means = []
    pull_terms = []
    for arm_id in ids:
        sel = labels == arm_id
        pts = flat[:, sel]                          # (D, N_c)
        mu = pts.mean(dim=1)                        # (D,)
        means.append(mu)
        dist = torch.linalg.norm(pts - mu[:, None], dim=0)   # (N_c,)
        hinge = torch.clamp(dist - delta_v, min=0.0) ** 2
        pull_terms.append(hinge.mean())
    pull = torch.stack(pull_terms).mean()

    mu_mat = torch.stack(means, dim=0)              # (C, D)
    reg = torch.linalg.norm(mu_mat, dim=1).mean()

    c = mu_mat.shape[0]
    if c > 1:
        diff = mu_mat[:, None, :] - mu_mat[None, :, :]        # (C, C, D)
        cdist = torch.linalg.norm(diff, dim=2)               # (C, C)
        margin = torch.clamp(2.0 * delta_d - cdist, min=0.0) ** 2
        eye = torch.eye(c, device=embed.device, dtype=torch.bool)
        push = margin[~eye].sum() / (c * (c - 1))
    else:
        push = zero

    total = w_pull * pull + w_push * push + w_reg * reg
    return total, pull.detach(), push.detach()


def discriminative_embedding_loss(
    embed: torch.Tensor,            # (B, D, H, W) raw embedding logits
    arm_instance: torch.Tensor,     # (B, H, W) long instance ids
    *,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    w_pull: float = 1.0,
    w_push: float = 1.0,
    w_reg: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-image discriminative pull/push/reg loss; returns (loss, parts)."""
    b = embed.shape[0]
    total = embed.new_tensor(0.0)
    pull_sum = embed.new_tensor(0.0)
    push_sum = embed.new_tensor(0.0)
    n_valid = 0
    for i in range(b):
        loss_i, pull_i, push_i = _single_image_loss(
            embed[i], arm_instance[i],
            delta_v=delta_v, delta_d=delta_d,
            w_pull=w_pull, w_push=w_push, w_reg=w_reg,
        )
        total = total + loss_i
        pull_sum = pull_sum + pull_i
        push_sum = push_sum + push_i
        if torch.unique(arm_instance[i])[torch.unique(arm_instance[i]) > 0].numel() > 0:
            n_valid += 1
    denom = max(1, b)
    loss = total / denom
    parts = {
        'embedding_loss': float(loss.detach().item()),
        'embedding_pull': float((pull_sum / denom).item()),
        'embedding_push': float((push_sum / denom).item()),
        'embedding_images_with_arms': float(n_valid),
        'embedding_centroid_accuracy': embedding_centroid_accuracy(
            embed.detach(), arm_instance
        ),
    }
    return loss, parts
