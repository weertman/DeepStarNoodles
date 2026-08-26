# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter


def _zhang_suen_thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning fallback for path metrics.

    The project runtime does not always include scikit-image, so keep this
    metric-side skeletonization dependency-free.
    """
    img = np.asarray(mask, dtype=bool).copy()
    if img.ndim != 2 or not img.any():
        return img
    img = np.pad(img, 1, mode='constant', constant_values=False)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            remove: list[tuple[int, int]] = []
            h, w = img.shape
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if not img[y, x]:
                        continue
                    p2 = img[y - 1, x]
                    p3 = img[y - 1, x + 1]
                    p4 = img[y, x + 1]
                    p5 = img[y + 1, x + 1]
                    p6 = img[y + 1, x]
                    p7 = img[y + 1, x - 1]
                    p8 = img[y, x - 1]
                    p9 = img[y - 1, x - 1]
                    neigh = [p2, p3, p4, p5, p6, p7, p8, p9]
                    count = int(sum(bool(v) for v in neigh))
                    if count < 2 or count > 6:
                        continue
                    transitions = sum((not bool(a)) and bool(b) for a, b in zip(neigh, neigh[1:] + neigh[:1]))
                    if transitions != 1:
                        continue
                    if step == 0:
                        if p2 and p4 and p6:
                            continue
                        if p4 and p6 and p8:
                            continue
                    else:
                        if p2 and p4 and p8:
                            continue
                        if p2 and p6 and p8:
                            continue
                    remove.append((y, x))
            if remove:
                changed = True
                for y, x in remove:
                    img[y, x] = False
    return img[1:-1, 1:-1]


def _thin(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    if int(mask.sum()) <= 4096:
        return _zhang_suen_thin(mask)
    dist = distance_transform_edt(mask).astype(np.float32)
    return mask & (dist > 0.0) & (dist >= maximum_filter(dist, size=3))


def _as_bhw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        return arr
    raise ValueError(f'Expected centerline map with shape HxW or BxHxW, got {arr.shape}')


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0.0 else 0.0


def _distance_to(mask: np.ndarray) -> np.ndarray:
    if np.asarray(mask, dtype=bool).any():
        return distance_transform_edt(~np.asarray(mask, dtype=bool)).astype(np.float32)
    return np.full(mask.shape, float(max(mask.shape) if mask.size else 0), dtype=np.float32)


def centerline_path_metrics(
    pred_score: np.ndarray,
    target_score: np.ndarray,
    *,
    threshold: float = 0.5,
    tolerances: tuple[int, ...] = (2, 4, 8),
) -> dict[str, float]:
    """Compare predicted and target centerline path-score maps.

    Inputs are high-is-path scores in [0, 1]. Distance-representation callers
    should pass ``1 - distance`` for both prediction and target.
    """
    pred_b = _as_bhw(pred_score)
    target_b = _as_bhw(target_score)
    if pred_b.shape != target_b.shape:
        raise ValueError(f'Prediction and target shapes differ: {pred_b.shape} vs {target_b.shape}')

    pred_skel_count = 0.0
    target_skel_count = 0.0
    exact_inter = 0.0
    pred_to_target_sum = 0.0
    target_to_pred_sum = 0.0
    pred_to_target_n = 0.0
    target_to_pred_n = 0.0
    tol_hits = {tol: {'pred': 0.0, 'target': 0.0} for tol in tolerances}

    for pred, target in zip(pred_b, target_b):
        pred_skel = _thin(pred >= threshold)
        target_skel = _thin(target >= threshold)
        pred_n = float(pred_skel.sum())
        target_n = float(target_skel.sum())
        pred_skel_count += pred_n
        target_skel_count += target_n
        exact_inter += float((pred_skel & target_skel).sum())

        target_dist = _distance_to(target_skel)
        pred_dist = _distance_to(pred_skel)
        if pred_n > 0:
            vals = target_dist[pred_skel]
            pred_to_target_sum += float(vals.sum())
            pred_to_target_n += pred_n
            for tol in tolerances:
                tol_hits[tol]['pred'] += float((vals <= tol).sum())
        if target_n > 0:
            vals = pred_dist[target_skel]
            target_to_pred_sum += float(vals.sum())
            target_to_pred_n += target_n
            for tol in tolerances:
                tol_hits[tol]['target'] += float((vals <= tol).sum())

    precision = _safe_div(exact_inter, pred_skel_count)
    recall = _safe_div(exact_inter, target_skel_count)
    out = {
        'centerline_skel_pred_frac': float(pred_skel_count / max(1, pred_b.size)),
        'centerline_skel_target_frac': float(target_skel_count / max(1, target_b.size)),
        'centerline_skel_area_ratio': _safe_div(pred_skel_count, target_skel_count),
        'centerline_skel_precision': precision,
        'centerline_skel_recall': recall,
        'centerline_skel_f1': _safe_div(2.0 * precision * recall, precision + recall),
        'centerline_pred_to_target_mean_dist': _safe_div(pred_to_target_sum, pred_to_target_n),
        'centerline_target_to_pred_mean_dist': _safe_div(target_to_pred_sum, target_to_pred_n),
    }
    for tol in tolerances:
        p = _safe_div(tol_hits[tol]['pred'], pred_skel_count)
        r = _safe_div(tol_hits[tol]['target'], target_skel_count)
        out[f'centerline_path_precision_tol{tol}'] = p
        out[f'centerline_path_recall_tol{tol}'] = r
        out[f'centerline_path_f1_tol{tol}'] = _safe_div(2.0 * p * r, p + r)
    return out

def bootstrap_mean_ci(
    values,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 20260718,
) -> dict[str, float | int | None]:
    """Return a deterministic image-resampled percentile confidence interval."""
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_resamples, len(array)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(samples, alpha)),
        "high": float(np.quantile(samples, 1.0 - alpha)),
        "n": int(len(array)),
    }

