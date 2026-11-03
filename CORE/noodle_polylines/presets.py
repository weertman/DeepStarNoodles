# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Named parameter sets reproducing each figure panel's exact choices, and :func:`compute_all`.

A preset fixes every definition choice that differs between panels (R0 policy, per-frame variant,
deformation lag, curvature 2-point rule and reducer, isotropy fallback, origin dtypes). Nothing is
unified silently: pick the preset of the panel you want to reproduce, or build your own ``Preset``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from . import frames as FR
from . import geometry as G
from . import kinematics as KI
from . import scale as SC
from . import tracking as TR
from .container import PolylineContainer


@dataclass(frozen=True)
class Preset:
    name: str
    panels: str
    r0_policy: str | None            # scale.POLICIES, or None (no R0-scaled outputs)
    variant: str                     # frames.VARIANTS: 'fig4e' | 'fig4d' | 'fig5'
    deformation_lag: str             # 'prev_frame_reset' (Fig 4) | 'last_arm_frame' (Fig 5)
    shape_order: str                 # T1.6 scaling order: 'fig4' | 'fig5'
    two_point: str                   # T1.1 two-point arms: 'zero' (export / Fig 4) | 'nan' (Fig 5)
    curvature_reducer: str           # 'median' (Fig 4) | 'nanmedian' (Fig 5)
    isotropy_fallback: float         # T2.6 window value without frames: NaN (5D strata) | 0.0 (5A example)
    tip_origin_dtype: str            # T1.3 tip angle: 'float32' (flipping stats) | 'float64'
    tip_angle_dtype: str             # 'float64' | 'float32' (Fig 6C kymograph, prepare_figure_06_flip.py:117)
    tracker_origin_dtype: str        # T4 inputs: 'float64' (figure scripts) | 'float32' (runtime, stats)
    track_frames: str                # frames fed to a re-link: 'all' (frames with arms) | 'star'
    log10_floor: float | None = None # also return log10(max(deformation, floor)) (Fig 4D)
    # Fig 4D per-bout deformation (prepare_figure_04_tree_features.py:121-157): with ``windows=`` each bout
    # (start - context, stop + context) is run on its own, shapes only from star frames with
    # lo - lookback <= t <= hi. (context_s, lookback_s), or None for a whole-sequence deformation.
    deformation_window: tuple[float, float] | None = None

    def as_dict(self) -> dict:
        return asdict(self)


FIG4B = Preset('fig4b', '4B', 'per_frame', 'fig4e', 'prev_frame_reset', 'fig4', 'zero', 'median', math.nan,
               'float64', 'float64', 'float64', 'all')
FIG4D = Preset('fig4d', '4D', 'outside_windows', 'fig4d', 'prev_frame_reset', 'fig4', 'zero', 'median', math.nan,
               'float64', 'float64', 'float64', 'all', log10_floor=1e-3,
               deformation_window=(SC.OUTSIDE_CONTEXT_S, 1.0))
FIG4E = Preset('fig4e', '4E', 'pre_onset', 'fig4e', 'prev_frame_reset', 'fig4', 'zero', 'median', math.nan,
               'float64', 'float64', 'float64', 'all')
FIG5 = Preset('fig5', '5B-D', 'baseline', 'fig5', 'last_arm_frame', 'fig5', 'nan', 'nanmedian', math.nan,
              'float64', 'float64', 'float64', 'all')
FIG5_EXAMPLE = replace(FIG5, name='fig5_example', panels='5A-B', isotropy_fallback=0.0)
# Flipping Fig 6: 6A (step costs) and 6D (stats.tip_polar: float32 origin, float64 angles) share one
# preset; the 6C kymograph (prepare_figure_06_flip.py:117-118) takes angles of float32 tips with no
# float cast, so its angles are float32. Sleepy 6B casts polylines to float64 (06_example:98).
FIG6_FLIPPING = Preset('fig6_flipping', '6A,6D', None, 'fig5', 'last_arm_frame', 'fig5', 'nan', 'nanmedian',
                       math.nan, 'float32', 'float64', 'float32', 'star')
FIG6C = replace(FIG6_FLIPPING, name='fig6c', panels='6C', tip_angle_dtype='float32')
FIG6_SLEEPY = replace(FIG6_FLIPPING, name='fig6_sleepy', panels='6B,6D', tip_origin_dtype='float64',
                      tracker_origin_dtype='float64', track_frames='all')
PRESETS = {p.name: p for p in (FIG4B, FIG4D, FIG4E, FIG5, FIG5_EXAMPLE, FIG6_FLIPPING, FIG6C, FIG6_SLEEPY)}


def get(preset: str | Preset) -> Preset:
    return preset if isinstance(preset, Preset) else PRESETS[preset]


def resolve_r0(c: PolylineContainer, preset: Preset, r0=None, bout=None, windows=None, onset_s=None
               ) -> SC.R0Result | None:
    """R0 for the preset: an explicit ``r0`` wins; otherwise the preset's policy with its context."""
    if r0 is not None:
        return SC.R0Result(r0, math.nan, 0, 'explicit')
    if preset.r0_policy is None:
        return None
    if preset.r0_policy == 'pre_onset' and bout is None:
        raise ValueError(f'{preset.name}: R0 pre_onset needs bout= (first_time_s, last_time_s, flip_start_s)')
    if preset.r0_policy == 'outside_windows' and windows is None:
        raise ValueError(f'{preset.name}: R0 outside_windows needs windows= [(start_s, stop_s), ...]')
    if preset.r0_policy == 'baseline' and onset_s is None:
        raise ValueError(f'{preset.name}: R0 baseline needs onset_s=')
    return SC.r0(c, preset.r0_policy, bout=bout, windows=windows, onset_s=onset_s)


def _window_deformation(c: PolylineContainer, r0, ids: np.ndarray, p: Preset, windows, base: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Fig 4D deformation, one run per bout window: (F,) first window containing each frame (NaN outside
    every window) and (W, F) per window (NaN outside that window)."""
    context, lookback = p.deformation_window
    t = c.times()
    w = np.asarray(list(windows), np.float64).reshape(-1, 2)
    per = np.full((len(w), c.n_frames), np.nan)
    first = np.full(c.n_frames, np.nan)
    taken = np.zeros(c.n_frames, bool)
    for k, (start, stop) in enumerate(w):
        lo, hi = start - context, stop + context
        d = KI.arm_deformation(c, r0, ids, lag=p.deformation_lag, shape_order=p.shape_order,
                               shape_mask=(t >= lo - lookback) & (t <= hi), base=base)
        inside = (t >= lo) & (t <= hi)
        per[k, inside] = d[inside]
        new = inside & ~taken
        first[new] = d[new]
        taken |= inside
    return first, per


def compute_all(c: PolylineContainer, preset: str | Preset = 'fig5', *, r0=None, bout=None, windows=None,
                onset_s=None, track: str = 'auto', config: TR.TrackerConfig = TR.RELEASED,
                shapes: bool = False, isotropy_window=(-300.0, -60.0)) -> dict[str, np.ndarray]:
    """Every representation of the preset's panel on one container, as a dict of named arrays.

    Keys ending in nothing are per arm (A,) / per point (P,) / per frame (F,) as documented in
    docs/noodle-polylines.md; ``r0``/``area0`` are scalars (per-frame arrays for ``per_frame``).
    ``track``: 'auto' (stored IDs when present, else re-link), 'stored' or 'relink'. A container whose
    R0 policy cannot be resolved (e.g. a baseline with < 30 frames) returns only the untracked geometry
    and ``dropped=True``.

    Fig 4D (``deformation_window`` set and ``windows=`` given): the deformation is run once per bout
    window ``[start - 2, stop + 2]`` with shapes only from ``[start - 3, stop + 2]``, as the panel does.
    ``arm_deformation_window`` / ``log_deformation_window`` (W, F) hold each bout's values (NaN outside
    its window); ``arm_deformation`` / ``log_deformation`` (F,) take the first window containing the
    frame and are NaN outside every window.
    """
    p = get(preset)
    c.require('crop')
    out: dict[str, np.ndarray] = {'preset': np.array(p.name)}
    pts, ao, fo = c.points, c.arm_point_offsets, c.frame_arm_offsets
    # nothing is cached on c here (memory is freed on return); L, kappa and base are shared explicitly
    L, kappa = G.arm_metrics(c, two_point=p.two_point)
    out['arc_length'], out['curvature'] = L, kappa
    out['tip_source'] = G.tips_source(pts, ao, fo, c.crop)
    out['n_arms'] = FR.n_arms(c)
    if c.support_area_model_px is not None:
        out['radius_model'], out['radius_source'] = FR.body_radius(c, empty='nan')
        out['norm_length'] = G.norm_length(L, fo, out['radius_model'])
    if c.support_centroid_model is not None:
        iso = FR.tip_isotropy(c)
        out['tip_isotropy'] = iso
        if onset_s is not None:
            out['tip_isotropy_window'] = np.array(FR.window_median(iso, c.times() - onset_s, isotropy_window,
                                                                  p.isotropy_fallback))
    base = None
    if shapes:
        base = out['resample_base'] = G.base_shapes(c)
    if c.body_origin_model is None:
        return out
    angle, radius = G.tip_polar(pts, ao, fo, c.body_origin_model, p.tip_origin_dtype, p.tip_angle_dtype)
    out['tip_angle_deg'], out['tip_radius_px'] = angle, radius
    if c.identity is not None or c.has_track_id():
        frames_sel = None if p.track_frames == 'all' else np.flatnonzero(c.star_mask())
        ids, source = TR.track_ids(c, mode=track, config=config, origin_dtype=p.tracker_origin_dtype,
                                   frames=frames_sel)
        out['track_id'], out['track_id_source'] = ids, np.array(source)
    else:
        ids = None
    if p.r0_policy == 'per_frame':
        fl = FR.frame_lists_4b(c, L if p.two_point == 'zero' else None, kappa if p.two_point == 'zero' else None)
        out.update(fl)
        out['body_path'] = G.body_path(pts, ao, fo, c.body_origin_model, out['radius_model'])
        return out
    if r0 is None and p.r0_policy is None:
        return out              # preset without R0-scaled outputs (Fig 6)
    context = {'pre_onset': bout, 'outside_windows': windows, 'baseline': onset_s}
    if r0 is None and p.r0_policy in context and context[p.r0_policy] is None:
        return out              # no R0 context given: R0-free outputs only
    r = resolve_r0(c, p, r0, bout, windows, onset_s)
    if r is None:
        out['dropped'] = np.array(p.r0_policy is not None)
        return out
    out['r0'], out['area0'], out['r0_frames'] = np.asarray(r.r0), np.asarray(r.area0), np.asarray(r.n_frames)
    out['fold_asymmetry'] = FR.fold_asymmetry(c, r.r0, p.variant)
    out['arm_length_spread'] = FR.arm_length_spread(c, r.r0, p.variant, arc_length=L)
    out['arm_curvature'] = FR.arm_curvature(c, r.r0, p.variant, curvature=kappa, reducer=p.curvature_reducer)
    if ids is not None:
        if base is None:
            base = G.base_shapes(c)
        if p.deformation_window is not None and windows is not None:
            d, d_win = _window_deformation(c, r.r0, ids, p, windows, base)
            out['arm_deformation_window'] = d_win
            if p.log10_floor is not None:
                out['log_deformation_window'] = np.log10(np.maximum(d_win, p.log10_floor))
        else:
            d = KI.arm_deformation(c, r.r0, ids, lag=p.deformation_lag, shape_order=p.shape_order, base=base)
        out['arm_deformation'] = d
        if p.log10_floor is not None:
            out['log_deformation'] = np.log10(np.maximum(d, p.log10_floor))
        out['tip_motion'] = KI.tip_motion(c, r.r0, ids)
    return out
