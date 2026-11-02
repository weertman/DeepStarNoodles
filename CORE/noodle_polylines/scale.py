# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""T0.7 body-size normalisation R0 (and the matching support-area baseline) plus window masks.

R0 is in source px. The paper uses four policies, kept separate on purpose:

======================  =====  ====================================================================
policy                  panel  definition
======================  =====  ====================================================================
``per_frame``           4B     ``sqrt(area / pi)`` of each frame, model px (not source px)
``pre_onset``           4E     median source-px radius over star frames of the bout window before
                               flip onset (``first_time <= t <= last_time`` and ``t < flip_start``)
``outside_windows``     4D     per-clip median over star frames outside every ``[start-2, stop+2]``
``baseline``            5B-D   per boundary, ``rel in [-300, -60]`` s and r > 0; dropped (None) with
                               fewer than 30 frames
======================  =====  ====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .container import PolylineContainer
from .frames import body_radius, support_area_source

POLICIES = ('per_frame', 'pre_onset', 'outside_windows', 'baseline')
BASELINE_WINDOW = (-300.0, -60.0)
MIN_BASELINE_FRAMES = 30
OUTSIDE_CONTEXT_S = 2.0


@dataclass(frozen=True)
class R0Result:
    r0: float | np.ndarray        # source px (model px for per_frame)
    area0: float | np.ndarray     # source px^2 (model px^2 for per_frame)
    n_frames: int                 # frames the median used
    policy: str


def in_any_window(t: np.ndarray, windows) -> np.ndarray:
    """(n,) bool: ``lo <= t <= hi`` for any (lo, hi) in ``windows`` (closed intervals)."""
    t = np.asarray(t, np.float64)
    w = np.asarray(list(windows), np.float64).reshape(-1, 2)
    if not len(w):
        return np.zeros(t.shape, bool)
    return np.any((t[:, None] >= w[None, :, 0]) & (t[:, None] <= w[None, :, 1]), axis=1)


def r0_per_frame(c: PolylineContainer) -> R0Result:
    """Fig 4B: per-frame model-px radius (NaN for an empty support) and raw model-px area."""
    r_model, _ = body_radius(c, empty='nan')
    return R0Result(r_model, c.support_area_model_px.astype(np.float64), c.n_frames, 'per_frame')


def r0_pre_onset(c: PolylineContainer, bout: dict) -> R0Result:
    """Fig 4E: median over the bout window's star frames before ``flip_start_s`` (no r > 0 guard).

    ``bout`` needs ``first_time_s``, ``last_time_s`` and ``flip_start_s`` (prepare_figure_04_kinematics.py:130-136).
    """
    t = c.times()
    _, r_src = body_radius(c, empty='nan')
    m = (t >= bout['first_time_s']) & (t <= bout['last_time_s']) & c.star_mask() & (t < bout['flip_start_s'])
    if not m.any():
        raise ValueError('no pre-flip frames for this bout')
    area = support_area_source(c)
    return R0Result(float(np.median(r_src[m])), float(np.median(area[m])), int(m.sum()), 'pre_onset')


def r0_outside_windows(c: PolylineContainer, windows, context_s: float = OUTSIDE_CONTEXT_S) -> R0Result:
    """Fig 4D: median over star frames outside every ``[start - context, stop + context]``.

    ``windows``: (start_s, stop_s) pairs (upside-down onset, right-side-up onset of every bout of the clip).
    Source: prepare_figure_04_tree_features.py:96-101.
    """
    t = c.times()
    _, r_src = body_radius(c, empty='nan')
    wins = [(a - context_s, b + context_s) for a, b in windows]
    m = c.star_mask() & ~in_any_window(t, wins)
    area = support_area_source(c)
    return R0Result(float(np.median(r_src[m])), float(np.median(area[m])), int(m.sum()), 'outside_windows')


def r0_baseline(c: PolylineContainer, onset_s: float, window=BASELINE_WINDOW,
                min_frames: int = MIN_BASELINE_FRAMES) -> R0Result | None:
    """Fig 5: median over frames with ``window[0] <= t - onset <= window[1]`` and r > 0; None (the boundary
    is dropped) with fewer than ``min_frames``. Source: prepare_figure_05_measures.py:146-154."""
    rel = c.times() - onset_s
    _, r_src = body_radius(c, empty='zero')
    m = (rel >= window[0]) & (rel <= window[1]) & (r_src > 0)
    if m.sum() < min_frames:
        return None
    area = support_area_source(c)
    return R0Result(float(np.median(r_src[m])), float(np.median(area[m])), int(m.sum()), 'baseline')


def r0(c: PolylineContainer, policy: str, **ctx) -> R0Result | None:
    """Dispatch on ``policy``: ``bout=`` for pre_onset, ``windows=`` for outside_windows, ``onset_s=`` for baseline."""
    if policy == 'per_frame':
        return r0_per_frame(c)
    if policy == 'pre_onset':
        return r0_pre_onset(c, ctx['bout'])
    if policy == 'outside_windows':
        return r0_outside_windows(c, ctx['windows'], ctx.get('context_s', OUTSIDE_CONTEXT_S))
    if policy == 'baseline':
        return r0_baseline(c, ctx['onset_s'], ctx.get('window', BASELINE_WINDOW),
                           ctx.get('min_frames', MIN_BASELINE_FRAMES))
    raise ValueError(f'policy must be one of {POLICIES}')
