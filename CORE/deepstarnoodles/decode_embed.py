# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Identity-embedding-aware polyline tracer for the hydra representation.

This is a second, from-scratch decoder that traces per-arm polylines using
*all* model channels **plus** the per-pixel arm-identity embedding (channels
10:10+D). The plain geometric decoder in :mod:`deepstarnoodles.decode` resolves
crossings only with a turn-momentum penalty, which fails when two arms overlap
and the averaged tangent/progress fields become uninformative. The embedding
gives each arm a distinct per-pixel signature, so a walk that drifts onto a
neighbouring arm can be charged directly for the identity change.

Algorithm (per image):

1. Peaks: arm-tip and arm-base heatmap maxima (tips seed the walks, bases
   terminate them), same as the geometric decoder.
2. Identity seeds: for every tip/base peak, sample a reference embedding vector
   by averaging the embedding over support pixels in a small disk around it.
3. Geometric edges: build the (pixel, incoming-direction) state graph once with
   the geometry-only edge cost (ridge + against-tangent + progress descent +
   turn penalty). These costs are identity-independent.
4. Identity-gated geodesics: for each tip, ADD an identity cost to every edge
   equal to ``embed_weight * step * ||emb(dst) - e_tip||`` and run Dijkstra from
   that tip. Stepping onto another arm's pixels (far embedding) is expensive, so
   the cheapest path naturally stays on the tip's own arm through a crossing.
5. Pairing: a global Hungarian tip->base assignment whose cost is the geodesic
   distance plus an identity-disagreement term ``embed_pairing_weight *
   ||e_tip - e_base||`` (a tip should reach a base of the same identity).
6. Acceptance: reject paths whose mean per-pixel geometric cost is too high or
   whose mean identity distance to the tip seed is too high (i.e. the walk could
   only reach a base by leaving its own arm).

The decoder consumes the activated 10 channels and the raw D-dim embedding; it
runs identically on rasterized-target-derived inputs (with a synthesized
embedding) and on model outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from deepstarnoodles.decode import (
    _OFFSETS,
    _extract_path,
    _smooth_resample,
    _snap_to_mask,
    activate_logits,
    find_peaks,
)
from deepstarnoodles.temporal_quality import compute_decode_quality


@dataclass(frozen=True)
class EmbedDecodeParams:
    support_threshold: float = 0.5
    # Peak extraction. Tip/base thresholds are kept low for recall (catch faint
    # arms the head is unsure about); the identity-aware suppression below cleans
    # up the resulting duplicates/fragments, so low thresholds are safe.
    tip_threshold: float = 0.2
    base_threshold: float = 0.25
    point_threshold: float = 0.3
    tip_nms_radius: int = 4
    base_nms_radius: int = 3
    point_nms_radius: int = 8
    base_hit_radius: float = 3.0
    peak_snap_radius: float = 8.0
    # Geometric edge costs (identity-independent).
    ridge_weight: float = 2.0
    align_weight: float = 1.0
    progress_weight: float = 2.0
    progress_drop_tolerance: float = 0.05
    progress_drop_weight: float = 20.0
    turn_weight: float = 1.0
    cost_floor: float = 0.01
    # Identity-embedding terms. Scale is set by the discriminative-loss margins
    # the model was trained with (delta_v ~ intra-arm spread, delta_d ~ half the
    # inter-arm mean gap), so identity distances run ~0..2*delta_d.
    embed_weight: float = 2.0          # cost per unit embedding distance from the tip identity, per px
    embed_radius: int = 2              # disk radius to sample reference embeddings at peaks
    embed_pairing_weight: float = 2.0  # weight on tip/base identity disagreement in assignment
    # Path acceptance. mean_step_cost is the GEOMETRY-only mean cost along the
    # chosen path (independent of embed_weight); mean_identity_dist is the mean
    # embedding distance to the tip seed (rejects paths that only reach a base by
    # leaving their own arm).
    max_mean_step_cost: float = 1.5
    max_mean_identity_dist: float = 1.5
    min_path_px: float = 10.0
    # Identity-aware suppression of duplicate / fragment / island arms.
    # Two arms are the SAME physical arm only if their identity embeddings agree
    # (distance <= dup_identity_dist, well below the inter-arm gap ~2*delta_d), so
    # genuine crossings (which overlap spatially but differ in identity) survive.
    dup_identity_dist: float = 1.2   # heavy overlap + identity within this -> duplicate (crossings ~2 survive)
    dup_overlap: float = 0.55        # frac of the shorter arm lying on the other -> duplicate
    fragment_identity_dist: float = 0.7  # stricter gate for the proximity-based fragment rule
    fragment_len_ratio: float = 0.6  # a same-identity arm this much shorter and touching is a fragment
    fragment_gap_px: float = 12.0    # how close a fragment must lie to its parent arm
    min_island_frac: float = 0.04    # drop arms whose support component is < this of the largest
    # Output polyline shape.
    smooth_window: int = 7
    resample_spacing: float = 4.0


def activated_and_embedding(pred: torch.Tensor, embedding_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a raw model output (10+D, H, W) into activated 10 channels + raw embedding.

    The embedding head is linear (no activation); only the 10 hydra channels are
    passed through their sigmoid/tanh activations.
    """
    act = activate_logits(pred[:10]).detach().cpu().numpy()
    emb = pred[10:10 + embedding_dim].detach().float().cpu().numpy()
    return act, emb


def _sample_reference_embedding(emb: np.ndarray, x: float, y: float, radius: int, mask: np.ndarray) -> np.ndarray:
    """Mean embedding over support pixels in a disk around (x, y).

    Falls back to the nearest in-bounds pixel's embedding when the disk contains
    no support (peaks can land just off the mask).
    """
    d, h, w = emb.shape
    iy, ix = int(round(y)), int(round(x))
    y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
    if y0 < y1 and x0 < x1:
        sub_mask = mask[y0:y1, x0:x1]
        if bool(sub_mask.any()):
            patch = emb[:, y0:y1, x0:x1][:, sub_mask]
            return patch.mean(axis=1).astype(np.float32)
    cy = min(max(iy, 0), h - 1)
    cx = min(max(ix, 0), w - 1)
    return emb[:, cy, cx].astype(np.float32)


def _build_geometric_coo(
    node_id: np.ndarray,
    centerline: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    progress: np.ndarray,
    params: EmbedDecodeParams,
) -> dict:
    """COO components of the geometry-only (pixel, direction) state graph.

    Returns arrays (rows, cols, w_geo, dst_pix, step) plus n_states/n_pix. The
    identity term is added per tip on top of ``w_geo`` without rebuilding the
    geometry, since geometry is identity-independent.
    """
    h, w = node_id.shape
    n_pix = int(node_id.max()) + 1
    n_states = n_pix * 8

    unit = np.array([(ox / np.hypot(ox, oy), oy / np.hypot(ox, oy)) for oy, ox in _OFFSETS])
    turn_cost = params.turn_weight * 0.5 * (1.0 - unit @ unit.T)  # [d_in, d_out]

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    w_geo: list[np.ndarray] = []
    dst_pix_all: list[np.ndarray] = []
    step_all: list[np.ndarray] = []
    for d_out, (oy, ox) in enumerate(_OFFSETS):
        step = float(np.hypot(ox, oy))
        sy0, sy1 = max(0, -oy), min(h, h - oy)
        sx0, sx1 = max(0, -ox), min(w, w - ox)
        src = node_id[sy0:sy1, sx0:sx1]
        dst = node_id[sy0 + oy:sy1 + oy, sx0 + ox:sx1 + ox]
        valid = (src >= 0) & (dst >= 0)
        if not valid.any():
            continue
        sy, sx = np.nonzero(valid)
        sy += sy0
        sx += sx0
        ty, tx = sy + oy, sx + ox

        ridge = 1.0 - 0.5 * (centerline[sy, sx] + centerline[ty, tx])
        ux, uy = ox / step, oy / step
        align = ux * dx[sy, sx] + uy * dy[sy, sx]
        align_cost = 0.5 * (1.0 + align)
        d_prog = (progress[ty, tx] - progress[sy, sx]) / step
        prog_rise = np.maximum(d_prog, 0.0)
        prog_drop = np.maximum(-d_prog - params.progress_drop_tolerance, 0.0)
        base_cost = step * (
            params.cost_floor
            + params.ridge_weight * ridge
            + params.align_weight * align_cost
            + params.progress_weight * prog_rise
            + params.progress_drop_weight * prog_drop
        )
        src_pix = node_id[sy, sx]
        dst_pix = node_id[ty, tx]
        dst_state = dst_pix * 8 + d_out
        for d_in in range(8):
            rows.append(src_pix * 8 + d_in)
            cols.append(dst_state)
            w_geo.append((base_cost + turn_cost[d_in, d_out]).astype(np.float32))
            dst_pix_all.append(dst_pix)
            step_all.append(np.full(len(dst_pix), step, dtype=np.float32))

    return {
        'rows': np.concatenate(rows).astype(np.int64),
        'cols': np.concatenate(cols).astype(np.int64),
        'w_geo': np.concatenate(w_geo).astype(np.float32),
        'dst_pix': np.concatenate(dst_pix_all).astype(np.int64),
        'step': np.concatenate(step_all).astype(np.float32),
        'n_states': n_states,
        'n_pix': n_pix,
    }


def _tip_geodesic(
    coo: dict,
    emb_nodes: np.ndarray,
    tip_ref: np.ndarray,
    tip_node: int,
    params: EmbedDecodeParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dijkstra from one tip with identity cost gated to that tip's embedding.

    Returns (pix_min_dist, best_state_per_pix, predecessors, node_identity_dist).
    """
    n_states = coo['n_states']
    ndist = np.linalg.norm(emb_nodes - tip_ref[None, :], axis=1).astype(np.float32)  # (n_pix,)
    w_emb = coo['step'] * float(params.embed_weight) * ndist[coo['dst_pix']]
    w_all = coo['w_geo'] + w_emb

    # Virtual source -> all 8 direction states of the tip pixel (no first-step turn penalty).
    src_rows = np.full(8, n_states, dtype=np.int64)
    src_cols = tip_node * 8 + np.arange(8, dtype=np.int64)
    all_rows = np.concatenate([coo['rows'], src_rows])
    all_cols = np.concatenate([coo['cols'], src_cols])
    all_w = np.concatenate([w_all, np.full(8, 1e-6, dtype=np.float32)])
    n_total = n_states + 1
    graph = csr_matrix((all_w, (all_rows, all_cols)), shape=(n_total, n_total))

    dist, pred = dijkstra(graph, directed=True, indices=n_states, return_predecessors=True)
    pix_dist = dist[:n_states].reshape(-1, 8)
    best_state = np.argmin(pix_dist, axis=1)
    pix_min = np.take_along_axis(pix_dist, best_state[:, None], axis=1)[:, 0]
    return pix_min, best_state, pred, ndist


def _path_mean_geo_cost(
    yx_tip_to_base: np.ndarray,
    centerline: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    progress: np.ndarray,
    params: EmbedDecodeParams,
) -> tuple[float, float]:
    """Mean per-px geometry-only edge cost along the path (ordered tip->base).

    Mirrors the edge cost in :func:`_build_geometric_coo` (ridge + against-tangent
    + progress descent + turn) but excludes the identity term, so acceptance is
    independent of ``embed_weight``. Returns (mean_cost, length_px).
    """
    pts = np.asarray(yx_tip_to_base, dtype=np.int64)
    if len(pts) < 2:
        return 0.0, 0.0
    total = 0.0
    length = 0.0
    prev_u: tuple[float, float] | None = None
    for k in range(len(pts) - 1):
        cy, cx = int(pts[k, 0]), int(pts[k, 1])
        ny, nx = int(pts[k + 1, 0]), int(pts[k + 1, 1])
        oy, ox = ny - cy, nx - cx
        step = float(np.hypot(ox, oy))
        if step < 1e-6:
            continue
        ux, uy = ox / step, oy / step
        ridge = 1.0 - 0.5 * (float(centerline[cy, cx]) + float(centerline[ny, nx]))
        align = ux * float(dx[cy, cx]) + uy * float(dy[cy, cx])
        align_cost = 0.5 * (1.0 + align)
        d_prog = (float(progress[ny, nx]) - float(progress[cy, cx])) / step
        prog_rise = max(d_prog, 0.0)
        prog_drop = max(-d_prog - params.progress_drop_tolerance, 0.0)
        base = step * (
            params.cost_floor
            + params.ridge_weight * ridge
            + params.align_weight * align_cost
            + params.progress_weight * prog_rise
            + params.progress_drop_weight * prog_drop
        )
        turn = 0.0
        if prev_u is not None:
            cosang = prev_u[0] * ux + prev_u[1] * uy
            turn = params.turn_weight * 0.5 * (1.0 - cosang)
        total += base + turn
        length += step
        prev_u = (ux, uy)
    return total / max(length, 1e-6), length


def _arm_min_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(overlap_fraction, min_separation) between two polylines.

    overlap_fraction = fraction of the shorter polyline's vertices within 4px of
    the other; min_separation = closest distance between any two vertices.
    """
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    d = np.sqrt(((short[:, None, :] - long[None, :, :]) ** 2).sum(-1))
    nearest = d.min(axis=1)
    return float((nearest <= 4.0).mean()), float(nearest.min())


def _suppress_duplicates_and_fragments(arms: list[dict], params: EmbedDecodeParams) -> list[dict]:
    """Identity-aware NMS: collapse duplicate traces and split-tip fragments.

    Greedy by quality (prefer arms reaching a real base, then longer, then higher
    tip score). A later arm is dropped only when it shares identity with a kept
    arm (so real crossings are never merged) AND either overlaps it heavily
    (duplicate) or is a much shorter piece lying right next to it (fragment).
    """
    if len(arms) <= 1:
        return arms
    order = sorted(
        range(len(arms)),
        key=lambda i: (arms[i]['base_score'] > 0, arms[i]['length_px'], arms[i]['tip_score']),
        reverse=True,
    )
    kept: list[int] = []
    for i in order:
        a = arms[i]
        drop = False
        for j in kept:
            b = arms[j]
            idd = float(np.linalg.norm(a['identity'] - b['identity']))
            if idd > params.dup_identity_dist:
                continue  # different physical arm (e.g. a crossing) -> keep both
            overlap, sep = _arm_min_distance(a['polyline'], b['polyline'])
            if overlap >= params.dup_overlap:
                drop = True  # heavy overlap + same identity -> duplicate trace
                break
            if (idd <= params.fragment_identity_dist
                    and a['length_px'] <= params.fragment_len_ratio * b['length_px']
                    and sep <= params.fragment_gap_px):
                drop = True  # short same-identity piece touching its parent -> split-tip fragment
                break
        if not drop:
            kept.append(i)
    return [arms[i] for i in kept]


def decode_arms_embed(
    act: np.ndarray,
    emb: np.ndarray,
    params: EmbedDecodeParams = EmbedDecodeParams(),
    *,
    detector_confidence: float | None = None,
    body_radius_px: float | None = None,
) -> list[dict]:
    """Trace per-arm polylines (base->tip) from activated channels + embedding.

    act: (10, H, W) activated hydra channels. emb: (D, H, W) raw embedding.
    Returns dicts with keys: polyline (N,2 xy base->tip), tip_score, base_score,
    mean_step_cost, mean_identity_dist, length_px, identity (D-vector seed),
    decode_quality, and quality_components.  Optional detector confidence and
    body radius enrich quality provenance only; they do not affect filtering.
    """
    support, centerline = act[0], act[1]
    dx, dy, progress = act[2], act[3], act[4]
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    mask = support > params.support_threshold
    if not mask.any():
        return []

    tips = find_peaks(act[9], params.tip_threshold, params.tip_nms_radius)
    bases = find_peaks(act[8], params.base_threshold, params.base_nms_radius)
    if len(tips) == 0:
        return []

    h, w = mask.shape
    node_id = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.nonzero(mask)
    node_id[ys, xs] = np.arange(len(ys))
    node_yx = np.stack([ys, xs], axis=1)
    emb_nodes = emb[:, ys, xs].T.astype(np.float32)  # (n_pix, D)

    # Connected components of support: arms whose pixels live on a tiny detached
    # island (e.g. a tip split off in the segmentation) are dropped later.
    lab, n_cc = ndimage.label(mask)
    cc_area = np.bincount(lab.ravel())
    max_cc = float(cc_area[1:].max()) if n_cc >= 1 else 1.0

    tip_nodes = _snap_to_mask(tips, node_id, node_yx, params.peak_snap_radius)
    base_nodes = _snap_to_mask(bases, node_id, node_yx, params.peak_snap_radius) if len(bases) else []
    live_tips = [(i, n) for i, n in enumerate(tip_nodes) if n >= 0]
    if not live_tips:
        return []
    live_bases = [(j, n) for j, n in enumerate(base_nodes) if n >= 0]

    tip_ref = {i: _sample_reference_embedding(emb, tips[i, 0], tips[i, 1], params.embed_radius, mask)
               for i, _ in live_tips}
    base_ref = {j: _sample_reference_embedding(emb, bases[j, 0], bases[j, 1], params.embed_radius, mask)
                for j, _ in live_bases}

    coo = _build_geometric_coo(node_id, centerline, dx, dy, progress, params)

    geo = {}
    for ti, tn in live_tips:
        geo[ti] = _tip_geodesic(coo, emb_nodes, tip_ref[ti], tn, params)

    # Global tip->base assignment: geodesic distance + identity disagreement.
    assign: dict[int, int] = {}  # live_tips row r -> live_bases index k
    if live_bases:
        big = 1e9
        cost_m = np.full((len(live_tips), len(live_bases)), big, dtype=np.float64)
        for r, (ti, tn) in enumerate(live_tips):
            pix_min = geo[ti][0]
            for k, (j, bn) in enumerate(live_bases):
                c = float(pix_min[bn])
                if np.isfinite(c):
                    ident = float(np.linalg.norm(tip_ref[ti] - base_ref[j]))
                    cost_m[r, k] = c + params.embed_pairing_weight * ident
        rr, kk = linear_sum_assignment(cost_m)
        for r, k in zip(rr, kk):
            if cost_m[r, k] < big:
                assign[r] = int(k)

    arms: list[dict] = []
    for r, (ti, tn) in enumerate(live_tips):
        pix_min, best_state, pred, ndist = geo[ti]
        if r in assign:
            j, bn = live_bases[assign[r]]
            target_pix = bn
            cost = float(pix_min[bn])
            base_score = float(bases[j, 2])
            base_xy = (float(bases[j, 0]), float(bases[j, 1]))
        else:
            reach = np.isfinite(pix_min)
            if not reach.any():
                continue
            ridx = np.nonzero(reach)[0]
            prog_vals = progress[node_yx[ridx, 0], node_yx[ridx, 1]]
            target_pix = int(ridx[int(np.argmin(prog_vals))])
            cost = float(pix_min[target_pix])
            base_score = 0.0
            base_xy = None
        if not np.isfinite(cost):
            continue

        target_state = target_pix * 8 + int(best_state[target_pix])
        state_path = _extract_path(pred, target_state)  # base ... tip(, virtual source)
        pix_path = np.asarray([s // 8 for s in state_path if s < coo['n_states']], dtype=np.int64)
        if len(pix_path) < 2:
            continue
        yx = node_yx[pix_path]
        if base_xy is not None and params.base_hit_radius > 0:
            bx, by = base_xy
            d2 = (yx[:, 1] - bx) ** 2 + (yx[:, 0] - by) ** 2
            hits = np.nonzero(d2 <= params.base_hit_radius ** 2)[0]
            if len(hits):
                cut = int(hits[-1])  # path is base...tip; last hit is the first from the tip side
                pix_path = pix_path[cut:]
                yx = yx[cut:]
        if len(pix_path) < 2:
            continue

        mean_identity = float(ndist[pix_path].mean())
        # Geometry-only mean cost (independent of embed_weight) for acceptance.
        mean_cost, _ = _path_mean_geo_cost(yx[::-1], centerline, dx, dy, progress, params)
        xy = yx[:, ::-1].astype(np.float32)  # base->tip in (x, y)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        length = float(seg.sum())
        if length < params.min_path_px:
            continue
        if mean_cost > params.max_mean_step_cost:
            continue
        if mean_identity > params.max_mean_identity_dist:
            continue
        path_labels = lab[node_yx[pix_path, 0], node_yx[pix_path, 1]]
        nz = path_labels[path_labels > 0]
        island_frac = float(cc_area[np.bincount(nz).argmax()]) / max(max_cc, 1.0) if len(nz) else 1.0
        polyline = _smooth_resample(xy, params.smooth_window, params.resample_spacing)
        arms.append({
            'polyline': polyline,
            'tip_score': float(tips[ti, 2]),
            'base_score': base_score,
            'mean_step_cost': mean_cost,
            'mean_identity_dist': mean_identity,
            'length_px': length,
            'island_frac': island_frac,
            'identity': tip_ref[ti].astype(np.float32),
        })

    arms = [a for a in arms if a['island_frac'] >= params.min_island_frac]
    arms = _suppress_duplicates_and_fragments(arms, params)
    for arm in arms:
        quality = compute_decode_quality(
            arm,
            detector_confidence=detector_confidence,
            body_radius_px=body_radius_px,
        )
        arm['decode_quality'] = quality.decode_quality
        arm['quality_components'] = quality.quality_components
    return arms


def decode_image_embed(act: np.ndarray, emb: np.ndarray, params: EmbedDecodeParams = EmbedDecodeParams()) -> dict:
    """Full decode: identity-aware arms + point landmarks."""
    return {
        'arms': decode_arms_embed(act, emb, params),
        'mouth': find_peaks(act[5], params.point_threshold, params.point_nms_radius),
        'madreporite': find_peaks(act[6], params.point_threshold, params.point_nms_radius),
        'butthole': find_peaks(act[7], params.point_threshold, params.point_nms_radius),
    }


def embed_decode_params_with(**overrides) -> EmbedDecodeParams:
    return replace(EmbedDecodeParams(), **overrides)
