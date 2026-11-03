# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""T0.6 and T2: untracked per-frame aggregates over a :class:`PolylineContainer`.

Every function returns an (F,) array aligned to the container's frames (NaN where undefined).
``r0`` is a scalar or an (F,) array in source px (see :mod:`noodle_polylines.scale`).

The figures differ in the last bits of how per-arm values are scaled before they are aggregated, so
each aggregate takes ``variant``:

* ``'fig4e'`` - prepare_figure_04_kinematics.py (export columns: ``tip_source``, ``arm_length_source_px =
  L * side / 512``, ``kappa * 512 / side``, ``np.median``); frames need a star and >= 1 arm.
* ``'fig4d'`` - prepare_figure_04_tree_features.py (as 4E, but tips are ``poly[-1] * (side/512) + offset``).
* ``'fig5'`` - prepare_figure_05_measures.py (float32 tip mean, ``L * (side/512) / R0``,
  ``kappa / (side/512) * R0``, ``np.nanmedian``); frames need >= 1 arm and a finite body origin.
"""
from __future__ import annotations

import math

import numpy as np

from . import _csr
from . import _segments as SG
from . import geometry as G
from .container import PolylineContainer, _ranges

MODEL_SIZE = 512
VARIANTS = ('fig4e', 'fig4d', 'fig5')
MIN_ARMS_FOR_SPREAD = 4


def _r0_per_frame(r0, F: int) -> np.ndarray:
    r = np.asarray(r0, np.float64)
    return np.full(F, float(r)) if r.ndim == 0 else r


def _check(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}')


def valid_frames(c: PolylineContainer, variant: str = 'fig5') -> np.ndarray:
    """(F,) bool: frames on which the per-frame arm aggregates are defined for this variant."""
    _check(variant)
    n = c.arms_per_frame()
    if variant == 'fig5':
        c.require('body_origin_model')
        return (n > 0) & np.all(np.isfinite(c.body_origin_model), axis=1)
    return (n > 0) & c.star_mask()


# ------------------------------------------------------------------------------------ T2.1 / T0.6
def n_arms(c: PolylineContainer) -> np.ndarray:
    """T2.1 (F,) int64 number of traced arms per frame."""
    return c.arms_per_frame()


def body_radius(c: PolylineContainer, empty: str = 'nan') -> tuple[np.ndarray, np.ndarray]:
    """T0.6 per-frame body radius ``sqrt(area / pi)``: (model px, source px) float64.

    ``empty='nan'``: an empty support gives NaN (the export, retrace_released.py:666-673); ``'zero'`` gives
    0.0 (the Fig 5 vector expression, prepare_figure_05_measures.py:147). Frames without a star are NaN
    with ``'nan'``.
    """
    c.require('support_area_model_px', 'crop')
    area = c.support_area_model_px.astype(np.float64)
    side = c.crop[:, 2]
    r_model = np.sqrt(area / np.pi)
    if empty == 'nan':
        r_model = np.where((area != 0) & c.star_mask(), r_model, np.nan)
    elif empty != 'zero':
        raise ValueError("empty must be 'nan' or 'zero'")
    return r_model, r_model * side / MODEL_SIZE


def support_area_source(c: PolylineContainer) -> np.ndarray:
    """(F,) support area in source px^2: ``area * (side / 512) ** 2``."""
    c.require('support_area_model_px', 'crop')
    return c.support_area_model_px * (c.crop[:, 2] / MODEL_SIZE) ** 2


# ------------------------------------------------------------------------------------ T2.2 - T2.4
# The aggregates below are batched over frames with the exact segment reductions of ``_segments``; each
# per-frame value is the same IEEE expression the figure scripts evaluate on that frame's arms.
def _frame_layout(c: PolylineContainer) -> tuple[np.ndarray, np.ndarray]:
    fo = c.frame_arm_offsets
    return fo[:-1], np.diff(fo)


def _tip_index(c: PolylineContainer) -> np.ndarray:
    return c.arm_point_offsets[1:] - 1


def fold_asymmetry(c: PolylineContainer, r0, variant: str = 'fig5') -> np.ndarray:
    """T2.2 (F,) distance from the body origin to the mean arm tip, / R0 (source px / source px)."""
    _check(variant)
    c.require('crop', 'body_origin_model')
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    ok = valid_frames(c, variant)
    out = np.full(F, np.nan)
    if not ok.any():
        return out
    st, n = _frame_layout(c)
    crop = c.crop
    if variant == 'fig4e':
        tips = G.tips_source(c.points, c.arm_point_offsets, c.frame_arm_offsets, crop)
    elif variant == 'fig4d':
        # np.asarray(tip, float) * (side / 512) + [left, top], per arm
        af = c.arm_frame()
        tips = np.asarray(c.points[_tip_index(c)], np.float64)
        tips *= (crop[af, 2] / MODEL_SIZE)[:, None]
        tips += crop[af, :2]
    else:
        tips = c.points[_tip_index(c)]                      # float32, as the script's t array
    f = np.flatnonzero(ok)
    sums = SG.seg_rowsum(tips, st[f], n[f])                 # t.mean(0): row-by-row sum, then / n
    mean = sums / n[f][:, None] if sums.dtype == np.float64 else \
        (sums.astype(np.float64) / n[f][:, None]).astype(sums.dtype)
    if variant == 'fig5':
        d = SG.row_norm(mean.astype(np.float64) - c.body_origin_model[f])
        with np.errstate(divide='ignore', invalid='ignore'):
            out[f] = d * (crop[f, 2] / MODEL_SIZE) / r0[f]
    else:
        d = SG.row_norm(mean - G.origin_source(c.body_origin_model[f], crop[f]))
        with np.errstate(divide='ignore', invalid='ignore'):
            out[f] = d / r0[f]
    return out


def _per_arm_lengths(c, r0, variant, arc_length):
    if arc_length is None:
        arc_length, _ = G.arm_metrics(c, two_point='nan' if variant == 'fig5' else 'zero')
    return arc_length


def arm_length_spread(c: PolylineContainer, r0, variant: str = 'fig5', arc_length: np.ndarray | None = None,
                      min_arms: int = MIN_ARMS_FOR_SPREAD) -> np.ndarray:
    """T2.3 (F,) interquartile range (``np.percentile`` 25/75, linear) of arm lengths in source px / R0.

    NaN with fewer than ``min_arms`` arms. ``arc_length`` (A,) defaults to T0.3.
    """
    _check(variant)
    c.require('crop')
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    L = _per_arm_lengths(c, r0, variant, arc_length)
    st, n = _frame_layout(c)
    ok = valid_frames(c, variant) & (n >= min_arms)
    out = np.full(F, np.nan)
    if not ok.any():
        return out
    af = c.arm_frame()
    L = np.asarray(L, np.float64)                             # float(x)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if variant == 'fig5':
            vals = L * (c.crop[af, 2] / MODEL_SIZE) / r0[af]  # float(x) * (side / 512) / R0
        else:
            vals = L * c.crop[af, 2] / MODEL_SIZE / r0[af]    # (float(x) * side / 512) / R0
        f = np.flatnonzero(ok)
        sv = SG.seg_sort(vals, st[f], n[f])
        q1 = SG.seg_percentile(sv, st[f], n[f], 25)
        q3 = SG.seg_percentile(sv, st[f], n[f], 75)
        out[f] = q3 - q1
    return out


def arm_curvature(c: PolylineContainer, r0, variant: str = 'fig5', curvature: np.ndarray | None = None,
                  reducer: str | None = None) -> np.ndarray:
    """T2.4 (F,) median arm curvature in rad per R0 (``kappa`` converted to rad / source px, x R0).

    ``reducer``: 'median' (Fig 4, NaN-propagating) or 'nanmedian' (Fig 5); default per variant.
    ``curvature`` (A,) defaults to T1.1 with the variant's 2-point rule.
    """
    _check(variant)
    c.require('crop')
    reducer = reducer or ('nanmedian' if variant == 'fig5' else 'median')
    if reducer not in ('median', 'nanmedian'):
        raise ValueError("reducer must be 'median' or 'nanmedian'")
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    if curvature is None:
        _, curvature = G.arm_metrics(c, two_point='nan' if variant == 'fig5' else 'zero')
    ok = valid_frames(c, variant)
    out = np.full(F, np.nan)
    if not ok.any():
        return out
    st, n = _frame_layout(c)
    af = c.arm_frame()
    k = np.asarray(curvature, np.float64)                     # float(k)
    f = np.flatnonzero(ok)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if variant == 'fig5':
            vals = k / (c.crop[af, 2] / MODEL_SIZE) * r0[af]  # float(k) / (side / 512) * R0
        else:
            vals = k * MODEL_SIZE / c.crop[af, 2]             # float(k) * 512 / side
        sv = SG.seg_sort(vals, st[f], n[f])
        med = SG.seg_median(sv, st[f], n[f], skipna=reducer == 'nanmedian')
        out[f] = med if variant == 'fig5' else med * r0[f]
    return out


# ------------------------------------------------------------------------------------ T2.5 / T2.6
def frame_lists_4b(c: PolylineContainer, arc_length: np.ndarray | None = None,
                   curvature: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """T2.5 Fig 4B right panel: per arm normalised length (T1.2) and curvature x per-frame model radius;
    per frame the mean normalised length (NaN without arms). Source: prepare_figure_04.py:128-143."""
    if arc_length is None or curvature is None:
        arc_length, curvature = G.arm_metrics(c, two_point='zero')
    r_model, _ = body_radius(c, empty='nan')
    nl = G.norm_length(arc_length, c.frame_arm_offsets, r_model)
    kr = curvature * r_model[c.arm_frame()]
    mean = np.full(c.n_frames, np.nan)
    st, n = _frame_layout(c)
    f = np.flatnonzero(n > 0)
    # np.mean of the frame's float64 values: numpy's pairwise sum, then / n
    mean[f] = _csr.seg_sum(np.asarray(nl, np.float64), st[f], n[f]) / n[f]
    return {'arm_norm_length': nl, 'arm_curvature_body_radius': kr, 'mean_norm_length': mean}


_ISOTROPY_MIN_TIPS = 4          # R.isotropy_frame: NaN with fewer tips
_ISOTROPY_BLOCK_FRAMES = 1 << 15


def tip_isotropy(c: PolylineContainer, centre: str = 'support_centroid_model', min_tips: int = 4) -> np.ndarray:
    """T2.6 (F,) tip isotropy ``sqrt(max(l_min, 0) / max(l_max, 1e-9))`` of the covariance of the arm tips.

    Frames with fewer than ``min_tips`` (and always fewer than 4) tips or a non-finite centre are NaN. The
    value is translation invariant; ``centre`` only decides which frames count (the scripts skip frames
    with a NaN support centroid). Source: prepare_figure_05_summary.py:96-104.

    Batched over frames with the scripts' arithmetic: ``np.cov`` (row-by-row mean, centring,
    ``X @ X.T``, ``* 1 / (n - 1)``) becomes a stacked ``matmul`` of ``X`` with its own transposed view over
    frames with the same number of tips, and LAPACK ``eigvalsh`` runs on the stacked 2x2 matrices. A
    closed-form 2x2 eigen solve (or a pairwise-summed covariance) would differ from the BLAS / LAPACK
    result in the last bits, so neither is used.
    """
    cen = getattr(c, centre)
    if cen is None:
        raise ValueError(f'container lacks {centre}')
    out = np.full(c.n_frames, np.nan)
    st, n = _frame_layout(c)
    sel = np.flatnonzero(np.all(np.isfinite(cen), axis=1) & (n >= max(min_tips, _ISOTROPY_MIN_TIPS)))
    tip_of_arm = c.arm_point_offsets[1:] - 1
    for b in range(0, len(sel), _ISOTROPY_BLOCK_FRAMES):
        f = sel[b:b + _ISOTROPY_BLOCK_FRAMES]
        nf = n[f]
        owner = np.repeat(np.arange(len(f)), nf)
        # tips - centre (float64), the script's (n, 2) array per frame, laid out frame after frame
        rel = c.points[tip_of_arm[_ranges(st[f], nf)]] - cen[f][owner]
        seg = SG.seg_starts(nf)
        # np.cov: X.mean(axis=1) of the (2, n) view, which numpy accumulates tip by tip
        rel -= (SG.seg_rowsum(rel, seg, nf) / nf[:, None])[owner]
        cov = np.empty((len(f), 2, 2))
        for m in np.unique(nf):
            k = np.flatnonzero(nf == m)
            x = rel[seg[k][:, None] + np.arange(m)]                  # (G, m, 2) centred tips
            xt = np.ascontiguousarray(np.swapaxes(x, 1, 2))          # (G, 2, m) = np.cov's X
            # dot(X, X.T): the second operand must be a transposed view of the same buffer, as in
            # np.cov, so that BLAS takes the same (symmetric product) path; a separate copy can differ
            cv = np.matmul(xt, np.swapaxes(xt, 1, 2))
            cv *= np.true_divide(1, m - 1)
            cov[k] = cv
        ev = np.linalg.eigvalsh(cov)
        out[f] = np.sqrt(np.maximum(ev[:, 0], 0.0) / np.maximum(ev[:, 1], 1e-9))
    return out


def window_median(values: np.ndarray, t_rel: np.ndarray, window: tuple[float, float],
                  fallback: float = math.nan) -> float:
    """Median of the finite per-frame ``values`` with ``window[0] <= t_rel <= window[1]``; ``fallback`` if none.

    T2.6 per boundary: window (-300, -60) s; fallback NaN (prepare_figure_05_summary.py:105) or 0.0
    (prepare_figure_05_example.py:92); the Fig 5A/B example needs >= 0.6.
    """
    m = (t_rel >= window[0]) & (t_rel <= window[1]) & np.isfinite(values)
    return float(np.median(values[m])) if m.any() else fallback
