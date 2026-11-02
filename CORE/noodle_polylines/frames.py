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

from . import _reference as R
from . import geometry as G
from .container import PolylineContainer

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
def fold_asymmetry(c: PolylineContainer, r0, variant: str = 'fig5') -> np.ndarray:
    """T2.2 (F,) distance from the body origin to the mean arm tip, / R0 (source px / source px)."""
    _check(variant)
    c.require('crop', 'body_origin_model')
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    ok = valid_frames(c, variant)
    out = np.full(F, np.nan)
    fo, ao, pts = c.frame_arm_offsets, c.arm_point_offsets, c.points
    if variant == 'fig4e':
        tips = G.tips_source(pts, ao, fo, c.crop)
        origin = G.origin_source(c.body_origin_model, c.crop)
    elif variant == 'fig4d':
        origin = G.origin_source(c.body_origin_model, c.crop)
    for f in np.flatnonzero(ok):
        a0, a1 = fo[f], fo[f + 1]
        if variant == 'fig4e':
            t = tips[a0:a1]
            out[f] = float(np.linalg.norm(t.mean(0) - origin[f]) / r0[f])
        elif variant == 'fig4d':
            scale = c.crop[f, 2] / MODEL_SIZE
            offset = np.array([c.crop[f, 0], c.crop[f, 1]])
            t = np.array([np.asarray(pts[ao[a + 1] - 1], float) * scale + offset for a in range(a0, a1)])
            out[f] = float(np.linalg.norm(t.mean(0) - origin[f]) / r0[f])
        else:
            scale = c.crop[f, 2] / MODEL_SIZE
            centre = np.asarray(c.body_origin_model[f], float)
            t = np.asarray([pts[ao[a + 1] - 1] for a in range(a0, a1)])
            out[f] = float(np.linalg.norm(t.mean(0) - centre) * scale / r0[f])
    return out


def _per_arm_lengths(c, r0, variant, arc_length):
    if arc_length is None:
        arc_length, _ = G.arc_length_curvature(c.points, c.arm_point_offsets,
                                               two_point='nan' if variant == 'fig5' else 'zero')
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
    ok = valid_frames(c, variant)
    out = np.full(F, np.nan)
    fo = c.frame_arm_offsets
    for f in np.flatnonzero(ok):
        side = float(c.crop[f, 2])
        if variant == 'fig5':
            scale = c.crop[f, 2] / MODEL_SIZE
            lengths = [float(x) * scale / r0[f] for x in L[fo[f]:fo[f + 1]]]
        else:
            lengths = np.array([float(x) * side / MODEL_SIZE for x in L[fo[f]:fo[f + 1]]]) / r0[f]
        if len(lengths) >= min_arms:
            q1, q3 = np.percentile(lengths, [25, 75])
            out[f] = float(q3 - q1)
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
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    if curvature is None:
        _, curvature = G.arc_length_curvature(c.points, c.arm_point_offsets,
                                              two_point='nan' if variant == 'fig5' else 'zero')
    ok = valid_frames(c, variant)
    out = np.full(F, np.nan)
    fo = c.frame_arm_offsets
    red = np.nanmedian if reducer == 'nanmedian' else np.median
    for f in np.flatnonzero(ok):
        if variant == 'fig5':
            scale = c.crop[f, 2] / MODEL_SIZE
            vals = [float(k) / scale * r0[f] for k in curvature[fo[f]:fo[f + 1]]]
            out[f] = float(red(vals))
        else:
            side = float(c.crop[f, 2])
            vals = [float(k) * MODEL_SIZE / side for k in curvature[fo[f]:fo[f + 1]]]
            out[f] = float(red(vals) * r0[f])
    return out


# ------------------------------------------------------------------------------------ T2.5 / T2.6
def frame_lists_4b(c: PolylineContainer, arc_length: np.ndarray | None = None,
                   curvature: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """T2.5 Fig 4B right panel: per arm normalised length (T1.2) and curvature x per-frame model radius;
    per frame the mean normalised length (NaN without arms). Source: prepare_figure_04.py:128-143."""
    if arc_length is None or curvature is None:
        arc_length, curvature = G.arc_length_curvature(c.points, c.arm_point_offsets, two_point='zero')
    r_model, _ = body_radius(c, empty='nan')
    nl = G.norm_length(arc_length, c.frame_arm_offsets, r_model)
    kr = curvature * r_model[c.arm_frame()]
    mean = np.full(c.n_frames, np.nan)
    fo = c.frame_arm_offsets
    for f in range(c.n_frames):
        if fo[f + 1] > fo[f]:
            mean[f] = float(np.mean([float(x) for x in nl[fo[f]:fo[f + 1]]]))
    return {'arm_norm_length': nl, 'arm_curvature_body_radius': kr, 'mean_norm_length': mean}


def tip_isotropy(c: PolylineContainer, centre: str = 'support_centroid_model', min_tips: int = 4) -> np.ndarray:
    """T2.6 (F,) tip isotropy ``sqrt(max(l_min, 0) / max(l_max, 1e-9))`` of the covariance of the arm tips.

    Frames with fewer than ``min_tips`` tips or a non-finite centre are NaN. The value is translation
    invariant; ``centre`` only decides which frames count (the scripts skip frames with a NaN support
    centroid). Source: prepare_figure_05_summary.py:96-104.
    """
    cen = getattr(c, centre)
    if cen is None:
        raise ValueError(f'container lacks {centre}')
    out = np.full(c.n_frames, np.nan)
    fo, ao, pts = c.frame_arm_offsets, c.arm_point_offsets, c.points
    for f in range(c.n_frames):
        if not np.all(np.isfinite(cen[f])):
            continue
        n = fo[f + 1] - fo[f]
        if n < min_tips:
            continue
        tips = np.array([pts[ao[a + 1] - 1] for a in range(fo[f], fo[f + 1])]) - cen[f]
        out[f] = R.isotropy_frame(tips)
    return out


def window_median(values: np.ndarray, t_rel: np.ndarray, window: tuple[float, float],
                  fallback: float = math.nan) -> float:
    """Median of the finite per-frame ``values`` with ``window[0] <= t_rel <= window[1]``; ``fallback`` if none.

    T2.6 per boundary: window (-300, -60) s; fallback NaN (prepare_figure_05_summary.py:105) or 0.0
    (prepare_figure_05_example.py:92); the Fig 5A/B example needs >= 0.6.
    """
    m = (t_rel >= window[0]) & (t_rel <= window[1]) & np.isfinite(values)
    return float(np.median(values[m])) if m.any() else fallback
