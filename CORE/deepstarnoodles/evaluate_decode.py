# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Evaluate the decoder against ground-truth CVAT polylines/points.

Two input modes share one metric stack:
- target: decode rasterized target channels (perfect inputs; validates the
  algorithm itself),
- model: decode activated model outputs from a checkpoint.

Arm matching: decoded and GT polylines are resampled densely, paired greedily
by symmetric Chamfer distance, and a pair counts as a match when its Chamfer
distance is under --match-tolerance px.

Usage:
  python -m deepstarnoodles.evaluate_decode --mode target --limit 40
  python -m deepstarnoodles.evaluate_decode --mode model --checkpoint MODELS/hydra/endpoint_mse_best.pt
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from deepstarnoodles.annotation_strata import (
    annotation_strata_from_index_row,
    native_tolerance_metadata,
)
from deepstarnoodles.dataset import (
    CHANNEL_NAMES,
    CHANNEL_NAMES_WITHOUT_CENTERLINE,
    HydraDenseDataset,
    _chaikin_smooth,
    _resample_polyline,
    _shape_points,
)
from deepstarnoodles.decode import DecodeParams, activate_logits, decode_image
from deepstarnoodles.decode_embed import EmbedDecodeParams, decode_image_embed
from deepstarnoodles.metrics import centerline_path_metrics
from deepstarnoodles.paths import DEFAULT_TEST_INDEX, resolve_repo_path
from deepstarnoodles.train import split_indices
from deepstarnoodles.metrics import bootstrap_mean_ci

DEFAULT_INDEX = DEFAULT_TEST_INDEX
POINT_LABELS = ('mouth', 'madreporite', 'butthole')


# (flip_x, flip_y); identity first so --no-tta is just the first op.
FLIP_OPS = ((False, False), (True, False), (False, True), (True, True))


def predict_activated_ensemble(
    models: list[torch.nn.Module],
    image: torch.Tensor,
    tta: bool,
    output_specs: list[dict] | None = None,
) -> torch.Tensor:
    """Mean activated channels over models x flip ops, with vector channels
    (dx=2, dy=3) sign-corrected when un-flipping."""
    ops = FLIP_OPS if tta else FLIP_OPS[:1]
    accum: torch.Tensor | None = None
    for model_index, m in enumerate(models):
        for fx, fy in ops:
            img_t = image
            if fx:
                img_t = img_t.flip(-1)
            if fy:
                img_t = img_t.flip(-2)
            raw = m(img_t).float()[0]
            spec = output_specs[model_index] if output_specs is not None else None
            geometry_slice, embedding_slice = output_slices(spec, raw.shape[0])
            # Only the canonical geometry channels are nonlinear outputs.
            # Embeddings remain raw linear features throughout evaluation.
            act = torch.cat([
                activate_logits(raw[slice(*geometry_slice)]),
                raw[slice(*embedding_slice)],
            ], dim=0)
            if fx:
                act = act.flip(-1)
                act[2] = -act[2]
            if fy:
                act = act.flip(-2)
                act[3] = -act[3]
            accum = act if accum is None else accum + act
    assert accum is not None
    return accum / (len(models) * len(ops))


def infer_embedding_dim(checkpoint: dict | object) -> int:
    """Infer appended embedding channels from config, then embedding-head state."""
    if not isinstance(checkpoint, dict):
        return 0
    spec = checkpoint.get('output_spec')
    if not isinstance(spec, dict):
        config = checkpoint.get('config')
        spec = config.get('output_spec') if isinstance(config, dict) else None
    if isinstance(spec, dict):
        embedding = spec.get('embedding')
        if isinstance(embedding, dict) and embedding.get('dim') is not None:
            return max(0, int(embedding['dim']))
    config = checkpoint.get('config')
    if isinstance(config, dict) and config.get('embedding_dim') is not None:
        dim = int(config['embedding_dim'])
        if dim >= 0:
            return dim
    state = checkpoint.get('model', checkpoint)
    if isinstance(state, dict):
        for key in ('embedding_head.out.weight', 'module.embedding_head.out.weight'):
            weight = state.get(key)
            if isinstance(weight, torch.Tensor):
                return int(weight.shape[0])
    return 0


def infer_embedding_bandwidth(checkpoint: dict | object) -> bool:
    """Resolve whether a checkpoint carries a dense embedding-bandwidth channel."""
    if not isinstance(checkpoint, dict):
        return False
    spec = checkpoint_output_spec(checkpoint)
    if isinstance(spec, dict):
        section = spec.get('embedding_bandwidth')
        if isinstance(section, dict):
            return int(section.get('dim', 1)) == 1
        if section is not None:
            raise ValueError('output_spec.embedding_bandwidth must be a dictionary or null')
    config = checkpoint.get('config')
    if isinstance(config, dict) and config.get('embedding_bandwidth') is not None:
        return bool(config['embedding_bandwidth'])
    state = checkpoint.get('model', checkpoint)
    if isinstance(state, dict):
        return any(
            key in state
            for key in (
                'embedding_bandwidth_head.out.weight',
                'module.embedding_bandwidth_head.out.weight',
            )
        )
    return False


def output_slices(spec: dict | None, dense_channels: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Resolve dense geometry/embedding slices from an explicit output contract."""
    if spec is None:
        if dense_channels < 10:
            raise ValueError(f'dense output has {dense_channels} channels; expected at least 10')
        return (0, 10), (10, dense_channels)
    if not isinstance(spec, dict):
        raise ValueError('output_spec must be a dictionary')

    def validated(name: str) -> tuple[int, int]:
        section = spec.get(name)
        if not isinstance(section, dict):
            raise ValueError(f'output_spec.{name} must be a dictionary')
        value = section.get('slice')
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError(f'output_spec.{name}.slice must contain two integers')
        start, stop = value
        if start < 0 or stop < start or stop > dense_channels:
            raise ValueError(
                f'output_spec.{name}.slice {value} is out of bounds for {dense_channels} channels'
            )
        return start, stop

    geometry_slice = validated('geometry')
    embedding_slice = validated('embedding')
    geometry = spec['geometry']
    names = geometry.get('names')
    allowed_names = {tuple(CHANNEL_NAMES), tuple(CHANNEL_NAMES_WITHOUT_CENTERLINE)}
    if names is None and geometry_slice == (0, 10):
        names = CHANNEL_NAMES
    if not isinstance(names, (list, tuple)) or tuple(names) not in allowed_names:
        raise ValueError('output_spec.geometry.names is not a supported dense channel contract')
    if geometry_slice != (0, len(names)):
        raise ValueError('output_spec.geometry.slice does not match geometry.names')
    if embedding_slice[0] < geometry_slice[1]:
        raise ValueError('output_spec embedding slice overlaps geometry')
    embedding = spec['embedding']
    if embedding.get('dim') is not None:
        dim = embedding['dim']
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError('output_spec.embedding.dim must be a non-negative integer')
        if embedding_slice[1] - embedding_slice[0] != dim:
            raise ValueError('output_spec embedding slice and dim disagree')
    return geometry_slice, embedding_slice


def output_bandwidth_slice(
    spec: dict | None,
    dense_channels: int,
) -> tuple[int, int] | None:
    """Resolve an optional one-channel embedding-bandwidth slice."""
    if spec is None:
        return None
    section = spec.get('embedding_bandwidth')
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError('output_spec.embedding_bandwidth must be a dictionary or null')
    value = section.get('slice')
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError('output_spec.embedding_bandwidth.slice must contain two integers')
    start, stop = value
    if start < 0 or stop != start + 1 or stop > dense_channels:
        raise ValueError(
            f'output_spec.embedding_bandwidth.slice {value} is invalid for '
            f'{dense_channels} channels'
        )
    _, embedding_slice = output_slices(spec, dense_channels)
    if start < embedding_slice[1]:
        raise ValueError('output_spec embedding-bandwidth slice overlaps embedding')
    if section.get('dim') not in (None, 1):
        raise ValueError('output_spec.embedding_bandwidth.dim must equal 1')
    return start, stop


def checkpoint_output_spec(checkpoint: dict | object) -> dict | None:
    if not isinstance(checkpoint, dict):
        return None
    if isinstance(checkpoint.get('output_spec'), dict):
        return checkpoint['output_spec']
    config = checkpoint.get('config')
    return config.get('output_spec') if isinstance(config, dict) and isinstance(config.get('output_spec'), dict) else None


def infer_count_classes(checkpoint: dict | object) -> int:
    """Resolve optional image-level count classes without inspecting dense channels."""
    spec = checkpoint_output_spec(checkpoint)
    if isinstance(spec, dict):
        count_spec = spec.get('auxiliary', {}).get('count_logits')
        if isinstance(count_spec, dict) and count_spec.get('classes') is not None:
            return max(0, int(count_spec['classes']))
    if isinstance(checkpoint, dict):
        config = checkpoint.get('config')
        if isinstance(config, dict) and config.get('count_classes') is not None:
            return max(0, int(config['count_classes']))
        state = checkpoint.get('model', checkpoint)
        if isinstance(state, dict):
            weight = state.get('count_head.weight')
            if isinstance(weight, torch.Tensor):
                return int(weight.shape[0])
    return 0


def embedding_centroid_accuracy(embedding: np.ndarray, labels: np.ndarray) -> dict:
    """Nearest within-image GT centroid accuracy on valid arm-instance pixels."""
    valid = labels > 0
    ids = np.unique(labels[valid])
    if not len(ids):
        return {'correct': 0, 'total': 0, 'accuracy': None, 'n_instances': 0}
    features = np.moveaxis(embedding, 0, -1)
    centroids = np.stack([features[labels == arm_id].mean(axis=0) for arm_id in ids])
    x = features[valid]
    true = labels[valid]
    distances = (
        (x * x).sum(axis=1, keepdims=True)
        + (centroids * centroids).sum(axis=1)[None]
        - 2.0 * x @ centroids.T
    )
    pred = ids[np.argmin(distances, axis=1)]
    correct = int((pred == true).sum())
    return {
        'correct': correct,
        'total': int(len(true)),
        'accuracy': correct / max(len(true), 1),
        'n_instances': int(len(ids)),
    }


def baseline_embedding_decode(
    geometry: np.ndarray,
    embedding: np.ndarray,
    params: EmbedDecodeParams | None = None,
) -> dict:
    """Run embedding-Dijkstra, the geometric baseline (not the released tracer)."""
    return decode_image_embed(
        geometry,
        embedding,
        EmbedDecodeParams() if params is None else params,
    )


def summarize_dense_pose_metrics(
    *,
    support_intersection: int,
    support_pred_count: int,
    support_target_count: int,
    centerline_pred: list[np.ndarray],
    centerline_target: list[np.ndarray],
) -> dict[str, float | str]:
    """Global pixel/skeleton-count dense metrics across an evaluation set."""
    return {
        'aggregation_method': 'global_pixel_and_skeleton_counts_across_images',
        'support_threshold': 0.5,
        'support_dice': (
            2.0 * support_intersection / max(support_pred_count + support_target_count, 1)
        ),
        **centerline_path_metrics(
            np.stack(centerline_pred),
            np.stack(centerline_target),
            threshold=0.5,
            tolerances=(2, 4, 8),
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate polyline/point decoding.')
    p.add_argument('--mode', choices=['target', 'model'], required=True)
    p.add_argument('--checkpoint', type=Path, nargs='+', default=None,
                   help='Channel-model checkpoint(s); multiple paths are ensembled by '
                        'averaging activated channels. Required for --mode model.')
    p.add_argument('--tta', action='store_true',
                   help='4-way flip test-time augmentation (activated channels averaged, '
                        'dx/dy sign-corrected).')
    p.add_argument('--index', type=Path, default=DEFAULT_INDEX,
                   help='Defaults to the grouped test index.')
    p.add_argument('--target-cache-dir', type=Path, default=None,
                   help='Optional rasterized-target cache (disabled by default).')
    p.add_argument(
        '--split',
        choices=['val', 'train', 'all'],
        default='all',
        help='all rows of --index (default), or a legacy seeded subset of a combined index.',
    )
    p.add_argument('--limit', type=int, default=None, help='Cap on evaluated images.')
    p.add_argument('--offset', type=int, default=0, help='Skip the first N split images.')
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=20260606)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--match-tolerance', type=float, default=8.0)
    p.add_argument('--point-tolerance', type=float, default=12.0)
    p.add_argument('--params-json', type=str, default=None, help='JSON dict of DecodeParams overrides.')
    p.add_argument(
        '--embed-params-json',
        type=str,
        default=None,
        help='JSON dict of EmbedDecodeParams overrides.',
    )
    p.add_argument('--out-json', type=Path, default=None)
    return p.parse_args()


def gt_from_shapes(shapes: list[dict]) -> dict:
    """Ground-truth polylines (smoothed like the rasterizer) and points."""
    arms: list[np.ndarray] = []
    points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in shapes:
        if s.get('outside'):
            continue
        pts = _shape_points(s)
        if s.get('label') == 'arm' and s.get('type') == 'polyline' and len(pts) >= 2:
            arms.append(_resample_polyline(_chaikin_smooth(pts, iterations=2), spacing=2.0))
        elif s.get('label') in POINT_LABELS and s.get('type') == 'points':
            for x, y in pts:
                points[str(s['label'])].append((float(x), float(y)))
    return {'arms': arms, 'points': points}


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbor distance between two point sets."""
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return 0.5 * (float(np.sqrt(d2.min(axis=1)).mean()) + float(np.sqrt(d2.min(axis=0)).mean()))


def match_arms(decoded: list[np.ndarray], gt: list[np.ndarray], tolerance: float) -> dict:
    pairs: list[tuple[float, int, int]] = []
    for i, d in enumerate(decoded):
        dd = _resample_polyline(d, spacing=2.0)
        for j, g in enumerate(gt):
            pairs.append((chamfer(dd, g), i, j))
    pairs.sort()
    used_d: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for c, i, j in pairs:
        if c > tolerance or i in used_d or j in used_g:
            continue
        matches.append((i, j, c))
        used_d.add(i)
        used_g.add(j)
    return {
        'matches': matches,
        'n_decoded': len(decoded),
        'n_gt': len(gt),
        'tp': len(matches),
        'fp': len(decoded) - len(matches),
        'fn': len(gt) - len(matches),
    }


def endpoint_errors(decoded: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """(base_error, tip_error); GT polylines run base -> tip by convention."""
    direct = np.linalg.norm(decoded[0] - gt[0]) + np.linalg.norm(decoded[-1] - gt[-1])
    flipped = np.linalg.norm(decoded[0] - gt[-1]) + np.linalg.norm(decoded[-1] - gt[0])
    if flipped < direct:
        return float(np.linalg.norm(decoded[0] - gt[-1])), float(np.linalg.norm(decoded[-1] - gt[0]))
    return float(np.linalg.norm(decoded[0] - gt[0])), float(np.linalg.norm(decoded[-1] - gt[-1]))


def orientation_correct(decoded: np.ndarray, gt: np.ndarray) -> bool:
    direct = np.linalg.norm(decoded[0] - gt[0]) + np.linalg.norm(decoded[-1] - gt[-1])
    flipped = np.linalg.norm(decoded[0] - gt[-1]) + np.linalg.norm(decoded[-1] - gt[0])
    return direct <= flipped


def match_points(decoded: np.ndarray, gt: list[tuple[float, float]], tolerance: float) -> dict:
    gt_a = np.asarray(gt, dtype=np.float32).reshape(-1, 2)
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(decoded)):
        for j in range(len(gt_a)):
            d = float(np.hypot(decoded[i, 0] - gt_a[j, 0], decoded[i, 1] - gt_a[j, 1]))
            pairs.append((d, i, j))
    pairs.sort()
    used_d: set[int] = set()
    used_g: set[int] = set()
    errs: list[float] = []
    for d, i, j in pairs:
        if d > tolerance or i in used_d or j in used_g:
            continue
        errs.append(d)
        used_d.add(i)
        used_g.add(j)
    return {'tp': len(errs), 'fp': len(decoded) - len(errs), 'fn': len(gt_a) - len(errs), 'errors': errs}


def stratified_decode_reports(per_image: list[dict]) -> dict:
    """Equal-image arm summaries for canonical annotation strata."""
    selectors = {
        'all': lambda row: True,
        'close_arm': lambda row: bool(row.get('flags', {}).get('close_arm')),
        'not_close_arm': lambda row: not bool(row.get('flags', {}).get('close_arm')),
        'nearby_tip': lambda row: bool(row.get('flags', {}).get('nearby_tip')),
        'not_nearby_tip': lambda row: not bool(row.get('flags', {}).get('nearby_tip')),
    }

    def method_report(rows: list[dict], field: str) -> dict | None:
        scores = [row.get(field) for row in rows if row.get(field) is not None]
        if not scores and field == 'embedding_aware_geometry':
            return None
        image_f1 = []
        image_chamfer = []
        tp = fp = fn = 0
        all_chamfers: list[float] = []
        for score in scores:
            score_tp = int(score['arm_tp'])
            score_fp = int(score['arm_fp'])
            score_fn = int(score['arm_fn'])
            tp += score_tp
            fp += score_fp
            fn += score_fn
            precision = score_tp / max(score_tp + score_fp, 1)
            recall = score_tp / max(score_tp + score_fn, 1)
            image_f1.append(2 * precision * recall / max(precision + recall, 1e-9))
            chamfers = [float(value) for value in score.get('chamfers', [])]
            if chamfers:
                image_chamfer.append(float(np.mean(chamfers)))
                all_chamfers.extend(chamfers)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {
            'n_images': len(scores),
            'arm_f1_equal_image': bootstrap_mean_ci(image_f1),
            'chamfer_mean_equal_image': bootstrap_mean_ci(image_chamfer),
            'pooled': {
                'arm_f1': 2 * precision * recall / max(precision + recall, 1e-9),
                'arm_precision': precision,
                'arm_recall': recall,
                'arm_tp': tp,
                'arm_fp': fp,
                'arm_fn': fn,
                'chamfer_mean': (
                    float(np.mean(all_chamfers)) if all_chamfers else None
                ),
            },
        }

    result = {}
    for name, select in selectors.items():
        rows = [row for row in per_image if select(row)]
        result[name] = {
            'n_images': len(rows),
            'standard_geometry_decode': method_report(rows, 'standard_geometry'),
            'embedding_aware_geometry_decode': method_report(
                rows, 'embedding_aware_geometry'
            ),
        }
    return result


def main() -> int:
    args = parse_args()
    overrides = json.loads(args.params_json) if args.params_json else {}
    embed_overrides = json.loads(args.embed_params_json) if args.embed_params_json else {}
    params = DecodeParams(**overrides)
    embed_params = EmbedDecodeParams(**embed_overrides)
    tolerance_metadata = native_tolerance_metadata(
        args.image_size,
        match_tolerance_canonical_px=args.match_tolerance,
        point_tolerance_canonical_px=args.point_tolerance,
    )
    native_match_tolerance = tolerance_metadata['match_tolerance_native_px']
    native_point_tolerance = tolerance_metadata['point_tolerance_native_px']

    ds = HydraDenseDataset(
        args.index,
        image_size=(args.image_size, args.image_size),
        target_cache_dir=args.target_cache_dir,
    )
    if args.split == 'all':
        indices = list(range(len(ds)))
    else:
        train_idx, val_idx = split_indices(len(ds), args.seed)
        indices = list(val_idx if args.split == 'val' else train_idx)
    indices = indices[args.offset:]
    if args.limit:
        indices = indices[: args.limit]

    models: list[torch.nn.Module] = []
    output_specs: list[dict] = []
    embedding_dim = 0
    if args.mode == 'model':
        if not args.checkpoint:
            raise SystemExit('--checkpoint is required for --mode model')
        from deepstarnoodles.model import HydraModel

        for ckpt_path in args.checkpoint:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            checkpoint_embedding_dim = infer_embedding_dim(ckpt)
            spec = checkpoint_output_spec(ckpt)
            geometry = spec.get('geometry') if isinstance(spec, dict) else None
            geometry_names = geometry.get('names') if isinstance(geometry, dict) else None
            if (
                isinstance(geometry_names, (list, tuple))
                and 'arm_centerline' not in geometry_names
            ):
                raise ValueError(
                    'Polyline evaluation requires the legacy10 centerline contract; '
                    'use deepstarnoodles.evaluate_pixel_instances for no_centerline9 checkpoints.'
                )
            if models and checkpoint_embedding_dim != embedding_dim:
                raise ValueError('All ensemble checkpoints must have the same embedding_dim')
            embedding_dim = checkpoint_embedding_dim
            model = HydraModel(
                embedding_dim=embedding_dim,
                count_classes=infer_count_classes(ckpt),
            ).to(args.device)
            model.load_state_dict(ckpt['model'])
            model.eval()
            models.append(model)
            output_specs.append(spec or model.output_spec)

    def new_agg() -> dict:
        return {
            'arm_tp': 0, 'arm_fp': 0, 'arm_fn': 0,
            'chamfer': [], 'base_err': [], 'tip_err': [], 'orient_ok': 0, 'orient_total': 0,
            'points': {k: {'tp': 0, 'fp': 0, 'fn': 0, 'errors': []} for k in POINT_LABELS},
        }

    standard_agg = new_agg()
    embedding_agg = new_agg() if embedding_dim > 0 else None
    centroid_correct = 0
    centroid_total = 0
    centroid_instances = 0
    support_intersection = 0
    support_pred_count = 0
    support_target_count = 0
    centerline_pred: list[np.ndarray] = []
    centerline_target: list[np.ndarray] = []
    per_image: list[dict] = []

    def accumulate(dec: dict, gt: dict, agg: dict) -> dict:
        decoded_arms = [a['polyline'] for a in dec['arms']]
        am = match_arms(decoded_arms, gt['arms'], native_match_tolerance)
        agg['arm_tp'] += am['tp']
        agg['arm_fp'] += am['fp']
        agg['arm_fn'] += am['fn']
        for i, j, c in am['matches']:
            agg['chamfer'].append(c)
            be, te = endpoint_errors(decoded_arms[i], gt['arms'][j])
            agg['base_err'].append(be)
            agg['tip_err'].append(te)
            agg['orient_ok'] += int(orientation_correct(decoded_arms[i], gt['arms'][j]))
            agg['orient_total'] += 1
        for label in POINT_LABELS:
            pm = match_points(dec[label], gt['points'].get(label, []), native_point_tolerance)
            agg['points'][label]['tp'] += pm['tp']
            agg['points'][label]['fp'] += pm['fp']
            agg['points'][label]['fn'] += pm['fn']
            agg['points'][label]['errors'].extend(pm['errors'])
        return am

    for idx in indices:
        sample = ds[idx]
        strata = annotation_strata_from_index_row(ds.rows[idx])
        shapes = json.loads(resolve_repo_path(sample['meta']['row']['shapes_path']).read_text())
        sx = args.image_size / float(sample['meta']['row']['width'])
        sy = args.image_size / float(sample['meta']['row']['height'])
        for s in shapes:
            raw = list(s.get('points') or [])
            s['points'] = [v * (sx if i % 2 == 0 else sy) for i, v in enumerate(raw)]
        gt = gt_from_shapes(shapes)

        if args.mode == 'target':
            act = sample['target'].numpy()
        else:
            with torch.no_grad():
                image = sample['image'][None].to(args.device)
                act_t = predict_activated_ensemble(models, image, args.tta, output_specs)[None]
            act = act_t[0].cpu().numpy()

        geometry = act[:10]
        target = sample['target'].numpy()
        support_pred = geometry[0] >= 0.5
        support_true = target[0] >= 0.5
        support_intersection += int((support_pred & support_true).sum())
        support_pred_count += int(support_pred.sum())
        support_target_count += int(support_true.sum())
        centerline_pred.append(geometry[1])
        centerline_target.append(target[1])
        dec = decode_image(geometry, params)
        am = accumulate(dec, gt, standard_agg)
        aware_am = None
        image_centroid = None
        if embedding_dim > 0:
            assert embedding_agg is not None
            aware = baseline_embedding_decode(geometry, act[10:], embed_params)
            aware_am = accumulate(aware, gt, embedding_agg)
            image_centroid = embedding_centroid_accuracy(act[10:], sample['arm_instance'].numpy())
            centroid_correct += image_centroid['correct']
            centroid_total += image_centroid['total']
            centroid_instances += image_centroid['n_instances']
        per_image.append({
            'idx': int(idx),
            'flags': strata['flags'],
            'stratum_measurements': strata['measurements'],
            'stratum_metadata': {
                key: strata[key]
                for key in (
                    'canonical_size',
                    'source_size',
                    'thresholds_canonical_px',
                    'arm_count',
                )
            },
            'standard_geometry': {
                'arm_tp': am['tp'],
                'arm_fp': am['fp'],
                'arm_fn': am['fn'],
                'chamfers': [float(match[2]) for match in am['matches']],
            },
            'embedding_aware_geometry': None if aware_am is None else {
                'arm_tp': aware_am['tp'], 'arm_fp': aware_am['fp'], 'arm_fn': aware_am['fn'],
                'chamfers': [float(match[2]) for match in aware_am['matches']],
            },
            'embedding_centroid_accuracy': image_centroid,
        })

    def prf(tp: int, fp: int, fn: int) -> dict:
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return {'precision': p, 'recall': r, 'f1': 2 * p * r / max(p + r, 1e-9), 'tp': tp, 'fp': fp, 'fn': fn}

    def geometry_report(agg: dict) -> dict:
        return {
            'arms': {
                **prf(agg['arm_tp'], agg['arm_fp'], agg['arm_fn']),
                'chamfer_mean': float(np.mean(agg['chamfer'])) if agg['chamfer'] else None,
                'chamfer_p90': float(np.percentile(agg['chamfer'], 90)) if agg['chamfer'] else None,
                'base_err_mean': float(np.mean(agg['base_err'])) if agg['base_err'] else None,
                'tip_err_mean': float(np.mean(agg['tip_err'])) if agg['tip_err'] else None,
                'orientation_acc': agg['orient_ok'] / max(agg['orient_total'], 1),
            },
            'points': {
                label: {
                    **prf(d['tp'], d['fp'], d['fn']),
                    'err_mean': float(np.mean(d['errors'])) if d['errors'] else None,
                }
                for label, d in agg['points'].items()
            },
        }

    standard_report = geometry_report(standard_agg)
    dense_pose = summarize_dense_pose_metrics(
        support_intersection=support_intersection,
        support_pred_count=support_pred_count,
        support_target_count=support_target_count,
        centerline_pred=centerline_pred,
        centerline_target=centerline_target,
    )
    report = {
        'mode': args.mode,
        'split': args.split,
        'n_images': len(indices),
        'match_tolerance': args.match_tolerance,
        'tolerance_metadata': tolerance_metadata,
        'params': overrides,
        'standard_decode_method': 'deepstarnoodles.decode.decode_image',
        'embedding_decode_method': (
            'deepstarnoodles.decode_embed.decode_image_embed' if embedding_dim > 0 else None
        ),
        'embedding_decode_params': asdict(embed_params) if embedding_dim > 0 else None,
        'embedding_dim': embedding_dim,
        'dense_pose': dense_pose,
        'standard_geometry_decode': standard_report,
        'embedding_aware_geometry_decode': (
            geometry_report(embedding_agg) if embedding_agg is not None else None
        ),
        'embedding_centroid_accuracy': (
            {
                'correct': centroid_correct,
                'total': centroid_total,
                'accuracy': centroid_correct / max(centroid_total, 1),
                'n_instances': centroid_instances,
            }
            if embedding_dim > 0 else None
        ),
        'stratified_geometry_decode': stratified_decode_reports(per_image),
        # Backwards-compatible aliases for consumers of the 10-channel report.
        'arms': standard_report['arms'],
        'points': standard_report['points'],
    }
    print(json.dumps(report, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({'report': report, 'per_image': per_image}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
