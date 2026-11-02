# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""T3: tracked kinematics. Every function needs a per-arm ``track_id`` (A,) (stored or from
:func:`noodle_polylines.tracking.relink`); arms with ``track_id < 0`` are ignored.

Deformation lag variants (T3.1):

* ``'last_arm_frame'`` (default, Fig 5, prepare_figure_05_measures.py:199-202): compare with the last frame
  that had arms and a finite body origin; ``dt`` spans any gap.
* ``'prev_frame_reset'`` (Fig 4D/4E, prepare_figure_04_kinematics.py:141-156): compare with the previous
  recorded frame only; a frame without a star or arms resets the reference, so the next frame is NaN.
"""
from __future__ import annotations

import numpy as np

from . import _reference as R
from . import geometry as G
from .container import PolylineContainer
from .frames import _r0_per_frame

MODEL_SIZE = 512
LAGS = ('last_arm_frame', 'prev_frame_reset')


def rigid_residual(a: np.ndarray, b: np.ndarray) -> float:
    """RMS distance between two (k,2) shapes after the best rotation + translation (Kabsch, SVD)."""
    return R.rigid_residual(a, b)


def _shapes(c: PolylineContainer, r0: np.ndarray, order: str, base: np.ndarray | None) -> np.ndarray:
    if base is None:
        base = G.resample_base(c.points, c.arm_point_offsets)
    af = c.arm_frame()
    return G.shape12(base, c.crop[af, 2], r0[af], order=order)


def _track(c: PolylineContainer, track_id) -> np.ndarray:
    if track_id is None:
        if c.track_id is None:
            raise ValueError('no track_id: pass one (tracking.relink) or use a container with stored IDs')
        return c.track_id
    t = np.asarray(track_id, np.int64)
    if t.shape != (c.n_arms,):
        raise ValueError('track_id must be (A,)')
    return t


def arm_deformation(c: PolylineContainer, r0, track_id: np.ndarray | None = None, lag: str = 'last_arm_frame',
                    shape_order: str | None = None, shape_mask: np.ndarray | None = None,
                    base: np.ndarray | None = None) -> np.ndarray:
    """T3.1 (F,) median over linked arms of the rigid-fit RMS shape residual / R0 / dt (R0 / s).

    Shapes are T1.6 (12-point base-first resample x side/512 / R0). ``shape_order`` defaults to 'fig4' for
    ``prev_frame_reset`` and 'fig5' for ``last_arm_frame`` (last-bit differences only). ``shape_mask`` (F,)
    restricts the frames that contribute shapes (Fig 4D uses star frames within [start - 1, stop]); the
    default is star frames for ``prev_frame_reset``. Raw values; Fig 4D plots ``log10(max(x, 1e-3))``.
    """
    if lag not in LAGS:
        raise ValueError(f'lag must be one of {LAGS}')
    c.require('crop')
    tid = _track(c, track_id)
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    order = shape_order or ('fig4' if lag == 'prev_frame_reset' else 'fig5')
    shapes = _shapes(c, r0, order, base)
    t = c.times()
    fo = c.frame_arm_offsets
    out = np.full(F, np.nan)
    if lag == 'prev_frame_reset':
        mask = c.star_mask() if shape_mask is None else (np.asarray(shape_mask, bool) & c.star_mask())
        prev_t, prev = None, {}
        for f in range(F):
            cur = {int(tid[a]): shapes[a] for a in range(fo[f], fo[f + 1]) if tid[a] >= 0} if mask[f] else {}
            if prev_t is not None and cur and prev:
                bends = [R.rigid_residual(prev[k], v) / (t[f] - prev_t) for k, v in cur.items() if k in prev]
                if bends:
                    out[f] = float(np.median(bends))
            prev_t, prev = t[f], cur
        return out
    c.require('body_origin_model')
    ok = (np.diff(fo) > 0) & np.all(np.isfinite(c.body_origin_model), axis=1)
    if shape_mask is not None:
        ok &= np.asarray(shape_mask, bool)
    prev_t, prev = None, {}
    for f in np.flatnonzero(ok):
        arms = [a for a in range(fo[f], fo[f + 1]) if tid[a] >= 0]
        if prev_t is not None:
            dt = t[f] - prev_t
            bends = [R.rigid_residual(prev[int(tid[a])], shapes[a]) / dt for a in arms if int(tid[a]) in prev]
            if bends:
                out[f] = float(np.median(bends))
        prev = {int(tid[a]): shapes[a] for a in arms}
        prev_t = t[f]
    return out


def tip_motion(c: PolylineContainer, r0, track_id: np.ndarray | None = None) -> np.ndarray:
    """T3.2 (F,) median over linked arms of the body-relative tip displacement / R0 / dt (R0 / s).

    ``rel_tip = (tip - body_origin) * (side/512) / R0``; an arm counts when its track had a tip in the last
    frame with arms and a finite origin (prepare_figure_05_measures.py:192-198,203-205).
    """
    c.require('crop', 'body_origin_model')
    tid = _track(c, track_id)
    F = c.n_frames
    r0 = _r0_per_frame(r0, F)
    t = c.times()
    fo, ao, pts = c.frame_arm_offsets, c.arm_point_offsets, c.points
    ok = (np.diff(fo) > 0) & np.all(np.isfinite(c.body_origin_model), axis=1)
    out = np.full(F, np.nan)
    prev_t, prev = None, {}
    for f in np.flatnonzero(ok):
        scale = c.crop[f, 2] / MODEL_SIZE
        centre = np.asarray(c.body_origin_model[f], float)
        arms = list(range(fo[f], fo[f + 1]))
        tips = np.asarray([pts[ao[a + 1] - 1] for a in arms])
        rel = (tips - centre) * scale / r0[f]
        if prev_t is not None:
            dt = t[f] - prev_t
            moves = [np.linalg.norm(rel[i] - prev[int(tid[a])]) / dt for i, a in enumerate(arms)
                     if tid[a] >= 0 and int(tid[a]) in prev]
            if moves:
                out[f] = float(np.median(moves))
        prev = {int(tid[a]): rel[i] for i, a in enumerate(arms) if tid[a] >= 0}
        prev_t = t[f]
    return out


def tip_angle_track(c: PolylineContainer, track_id: np.ndarray | None = None, origin_dtype: str = 'auto',
                    angle_dtype: str = 'auto') -> dict[str, np.ndarray]:
    """T3.3 (A,) tip angle (deg) and radius (crop px) with each arm's frame time and track ID (Fig 6B/C).

    See :func:`noodle_polylines.geometry.tip_polar` for the dtype options. ``'auto'`` follows the panel of
    the container's ``meta['source_format']``: flipping -> the Fig 6C kymograph (float32 origin, float32
    angles, prepare_figure_06_flip.py:117-118); anything else -> Fig 6B (float64 origin and angles)."""
    c.require('body_origin_model')
    tid = _track(c, track_id)
    flipping = c.meta.get('source_format') == 'flipping'
    if origin_dtype == 'auto':
        origin_dtype = 'float32' if flipping else 'float64'
    if angle_dtype == 'auto':
        angle_dtype = 'float32' if flipping and origin_dtype == 'float32' else 'float64'
    angle, radius = G.tip_polar(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.body_origin_model,
                                origin_dtype=origin_dtype, angle_dtype=angle_dtype)
    af = c.arm_frame()
    return {'frame': af, 'frame_index': c.frame_index[af], 'time_s': c.times()[af], 'track_id': tid,
            'tip_angle_deg': angle, 'tip_radius_px': radius}
