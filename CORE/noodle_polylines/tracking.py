# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""T4: optional arm association with the released FusedTracker, and its Fig 6D diagnostics.

The released tracker (legacy path of ``deepstarnoodles.fused_tracker.FusedTracker``) links arms frame
to frame on body-relative, tip-anchored 12-point polylines:

* T4.1 :func:`common_span_dist` - mean point distance over the arc span both arms share, from the tip.
* T4.2 :func:`cost_matrix` - ``shape + embed_weight * ||e_i - e_track||``, gated at ``shape_gate``.
* T4.3 :func:`assign` - Hungarian assignment augmented with "unmatched" options at ``max_match_cost / 2``.
* T4.4 :func:`relink` - the sequential tracker over a container. Frames without arms never call
  ``step`` (``max_age`` counts 16 arm-bearing updates, not frames or seconds). IDs start at 0.
* T4.5 :func:`diagnostics` - continuation, forward-reverse agreement, jumps, lifetimes, stride precision.

Released data stores ``track_id``; :func:`track_ids` uses it unless asked to re-link.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from . import _reference as R
from . import geometry as G
from .container import PolylineContainer


@dataclass(frozen=True)
class TrackerConfig:
    """Released FusedTracker settings (shape gate 120 px, embedding weight 2.0, max age 16 updates, cap 90)."""
    shape_gate: float = 120.0
    embed_weight: float = 2.0
    max_age: int = 16
    max_match_cost: float | None = 90.0
    embed_update_gate: float = 1.0
    embed_alpha: float = 0.4
    vel_alpha: float = 0.0
    max_vel_px: float = 30.0
    k: int = 12

    def tracker_kwargs(self) -> dict:
        d = asdict(self)
        d.pop('k')
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TrackerConfig':
        keys = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in keys})


RELEASED = TrackerConfig()
VARIANTS = {'fused': RELEASED,
            'shape_only': replace(RELEASED, embed_weight=0.0),
            'no_cap': replace(RELEASED, max_match_cost=None)}
STRIDES = (2, 4, 8, 16)
JUMP_DEG, JUMP_MIN_RADIUS = 30.0, 25.0


class FusedTracker(R.FusedTracker):
    """The released tracker's legacy path (``step(frame, polys, lengths, embeds) -> labels``)."""

    def __init__(self, config: TrackerConfig = RELEASED, assign_hook=None):
        super().__init__(**config.tracker_kwargs(), assign_hook=assign_hook)
        self.config = config


def common_span_dist(pa: np.ndarray, la: float, pb: np.ndarray, lb: float) -> float:
    """T4.1 shape distance (px) between two tip-first (k,2) resamples with arc lengths ``la``, ``lb``."""
    return R.common_span_dist(pa, la, pb, lb)


def cost_matrix(polys: np.ndarray, lengths: np.ndarray, embeds: np.ndarray, track_polys: np.ndarray,
                track_lengths: np.ndarray, track_embeds: np.ndarray, config: TrackerConfig = RELEASED
                ) -> dict[str, np.ndarray]:
    """T4.2 (n,m) shape distance, embedding distance and gated cost (NaN where shape > gate).

    ``polys`` (n,k,2) f32 detections, ``lengths`` (n,) f32; ``track_*`` the tracks' last observations
    (with ``vel_alpha = 0`` a track's prediction is its last shape).
    """
    n, m = len(polys), len(track_polys)
    shape = np.empty((n, m))
    emb = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            shape[i, j] = R.common_span_dist(polys[i], float(lengths[i]), track_polys[j], float(track_lengths[j]))
            emb[i, j] = float(np.linalg.norm(embeds[i] - track_embeds[j]))
    cost = np.where(shape <= config.shape_gate, shape + config.embed_weight * emb, np.nan)
    return {'shape_distance': shape, 'embedding_distance': emb, 'cost': cost}


def assign(cost: np.ndarray, max_match_cost: float | None = 90.0) -> np.ndarray:
    """T4.3 (pairs, 2) int64 (row, column) assignment of a gated cost matrix (NaN = not allowed)."""
    return np.asarray(R.assign(cost, max_match_cost), np.int64).reshape(-1, 2)


def tracker_inputs(c: PolylineContainer, origin_dtype: str = 'float64', k: int = 12
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body-relative tip-first resamples (A,k,2) f32, lengths (A,) f32 and embeddings (A,8) f32.

    ``origin_dtype`` 'float64' reproduces the figure scripts (``poly - origin`` in float64), 'float32' the
    runtime and Fig 6 stats path; on released data both give bit-identical inputs.
    """
    c.require('body_origin_model', 'identity')
    if origin_dtype not in ('float64', 'float32'):
        raise ValueError("origin_dtype must be 'float64' or 'float32'")
    origin = c.body_origin_model[c.arm_frame()].astype(origin_dtype)
    sampled, lengths = G.resample_tip(c.points, c.arm_point_offsets, k, arm_origin=origin)
    return sampled, lengths.astype(np.float32), c.identity


def _frames(c: PolylineContainer, frames, require_finite_origin: bool) -> np.ndarray:
    pos = np.arange(c.n_frames) if frames is None else np.asarray(frames, np.int64)
    n = c.arms_per_frame()[pos]
    keep = n > 0
    if require_finite_origin:
        keep &= np.all(np.isfinite(c.body_origin_model[pos]), axis=1)
    return pos[keep]


def relink(c: PolylineContainer, config: TrackerConfig = RELEASED, origin_dtype: str = 'float64',
           frames: np.ndarray | None = None, reverse: bool = False, step_index: np.ndarray | None = None,
           inputs: tuple | None = None, require_finite_origin: bool = True, assign_hook=None) -> np.ndarray:
    """T4.4 (A,) int64 track IDs from a cold-start FusedTracker run (-1 for arms of frames not processed).

    ``frames``: frame positions to feed, in time order (default all); frames without arms (or with a
    non-finite origin) are skipped and do not age tracks. ``reverse`` runs them backwards (Fig 6D).
    ``step_index`` (F,) is the value passed to ``step`` (default ``frame_index``; inert with vel_alpha 0).
    """
    sampled, lengths, embeds = inputs if inputs is not None else tracker_inputs(c, origin_dtype, config.k)
    all_pos = np.arange(c.n_frames) if frames is None else np.asarray(frames, np.int64)
    pos = _frames(c, all_pos, require_finite_origin)
    idx = c.frame_index if step_index is None else np.asarray(step_index)
    tracker = FusedTracker(config, assign_hook=assign_hook)
    out = np.full(c.n_arms, -1, np.int64)
    fo = c.frame_arm_offsets
    order = pos[::-1] if reverse else pos
    last = int(idx[all_pos[-1]]) if len(all_pos) else 0
    for f in order:
        a0, a1 = fo[f], fo[f + 1]
        step = last - int(idx[f]) if reverse else int(idx[f])
        out[a0:a1] = tracker.step(step, sampled[a0:a1], lengths[a0:a1], embeds[a0:a1])
    return out


def track_ids(c: PolylineContainer, mode: str = 'auto', **relink_kw) -> tuple[np.ndarray, str]:
    """Per-arm track IDs and where they came from: 'stored' (the container's ``track_id``) or 'relink'.

    ``mode='auto'`` uses the stored IDs when every arm has one, else re-links.
    """
    if mode not in ('auto', 'stored', 'relink'):
        raise ValueError("mode must be 'auto', 'stored' or 'relink'")
    if mode == 'stored' or (mode == 'auto' and c.has_track_id()):
        if not c.has_track_id():
            raise ValueError('container has no complete stored track_id')
        return c.track_id, 'stored'
    return relink(c, **relink_kw), 'relink'


def step_costs(c: PolylineContainer, frame: int, config: TrackerConfig = RELEASED, origin_dtype: str = 'float64',
               frames: np.ndarray | None = None, track_order: str = 'sorted') -> dict:
    """The cost matrix the tracker solves at frame position ``frame`` (Fig 6A/C), after a cold-start run
    over the earlier ``frames``. ``track_order``: 'sorted' by ID (prepare_figure_06_step.py) or
    'insertion' (prepare_figure_06_flip.py). Returns track_ids, shape/embedding distances, gated cost,
    the assignment pairs and the tracker's labels for that frame."""
    sampled, lengths, embeds = tracker_inputs(c, origin_dtype, config.k)
    pos = _frames(c, frames, True)
    tracker = FusedTracker(config)
    fo = c.frame_arm_offsets
    for f in pos[pos < frame]:
        tracker.step(int(c.frame_index[f]), sampled[fo[f]:fo[f + 1]], lengths[fo[f]:fo[f + 1]], embeds[fo[f]:fo[f + 1]])
    active = sorted(tracker.tracks.items()) if track_order == 'sorted' else list(tracker.tracks.items())
    a0, a1 = fo[frame], fo[frame + 1]
    preds = [tracker._predict(t, int(c.frame_index[frame])) for _, t in active]
    res = cost_matrix(sampled[a0:a1], lengths[a0:a1], embeds[a0:a1], preds, [t['len'] for _, t in active],
                      [t['embed'] for _, t in active], config)
    res['track_ids'] = np.array([tid for tid, _ in active], np.int64)
    res['assignment'] = assign(res['cost'], config.max_match_cost)
    res['labels'] = np.asarray(tracker.step(int(c.frame_index[frame]), sampled[a0:a1], lengths[a0:a1],
                                            embeds[a0:a1]), np.int64)
    return res


# ------------------------------------------------------------------------------------ T4.5 diagnostics
def sequence_index(c: PolylineContainer, by: str = 'frame_index') -> np.ndarray:
    """Per-frame integer index used by the diagnostics: ``frame_index`` (flipping) or
    ``round(time_s - time_s[0])`` (Sleepy sequences, prepare_figure_06_stats.py:147)."""
    if by == 'frame_index':
        return c.frame_index
    if by == 'time':
        t = c.times()
        return np.array([int(round(x - t[0])) for x in t], np.int64)
    raise ValueError("by must be 'frame_index' or 'time'")


def _labels_per_frame(ids: np.ndarray, fo: np.ndarray, pos: np.ndarray) -> list[list[int]]:
    return [[int(x) for x in ids[fo[f]:fo[f + 1]] if x >= 0] for f in pos]


def diagnostics(c: PolylineContainer, frames: np.ndarray | None = None, states=None, fps: float | None = None,
                index: np.ndarray | None = None, variants: dict | None = None, strides=STRIDES,
                origin_dtype: str = 'float64') -> dict:
    """T4.5 Fig 6D tracker self-consistency on one sequence (prepare_figure_06_stats.py:238-315).

    ``frames``: positions forming the sequence (default: star frames); ``states`` (F,) labels of the later
    frame used to split counts (None entries are skipped); ``index`` (F,) integer frame index (default
    ``frame_index``). For each tracker variant: forward and reverse runs, ``by_state`` counters
    (continuation, forward-reverse link agreement, jumps > 30 deg with both tip radii >= 25 crop px,
    births), ``n_ids`` and arm-weighted ``arm_lifetime_s``; for 'fused' also stride precision
    (g = 2, 4, 8, 16). Returns the same structure as ``analyse`` plus ``runs`` (labels per pass, (A,)).
    """
    variants = VARIANTS if variants is None else variants
    pos = np.flatnonzero(c.star_mask()) if frames is None else np.asarray(frames, np.int64)
    fps = c.fps if fps is None else fps
    idx_all = c.frame_index if index is None else np.asarray(index)
    fo = c.frame_arm_offsets
    angle, radius = G.tip_polar(c.points, c.arm_point_offsets, fo, c.body_origin_model, origin_dtype=origin_dtype)
    polar = [(angle[fo[f]:fo[f + 1]], radius[fo[f]:fo[f + 1]]) for f in pos]
    seq_frames = [{'index': int(idx_all[f]), 'arms': [None] * int(fo[f + 1] - fo[f]),
                   'state': None if states is None else states[f]} for f in pos]
    inputs = tracker_inputs(c, origin_dtype)
    idx = np.array([fr['index'] for fr in seq_frames])
    res: dict = {'n_frames': len(pos), 'n_arms': int(sum(len(fr['arms']) for fr in seq_frames)),
                 'median_arms': float(np.median([len(fr['arms']) for fr in seq_frames])),
                 'duration_s': float((idx[-1] - idx[0] + 1) / fps), 'runs': {}}
    for variant, config in variants.items():
        cfg = config if isinstance(config, TrackerConfig) else TrackerConfig.from_dict(config)
        fwd_ids = relink(c, cfg, frames=pos, inputs=inputs, step_index=idx_all, require_finite_origin=False)
        rev_ids = relink(c, cfg, frames=pos, inputs=inputs, step_index=idx_all, reverse=True,
                         require_finite_origin=False)
        res['runs'][f'{variant}_fwd'], res['runs'][f'{variant}_rev'] = fwd_ids, rev_ids
        fwd, rev = _labels_per_frame(fwd_ids, fo, pos), _labels_per_frame(rev_ids, fo, pos)
        span: dict[int, list[int]] = {}
        for k, labs in enumerate(fwd):
            for lab in labs:
                span.setdefault(lab, [idx[k], idx[k]])[1] = idx[k]
        life = {lab: (b - a + 1) / fps for lab, (a, b) in span.items()}
        by_state = {k: {kk: int(vv) for kk, vv in v.items()}
                    for k, v in R.frame_measures(seq_frames, fwd, rev, polar).items()}
        res[variant] = {'by_state': by_state, 'n_ids': len(span),
                        'arm_lifetime_s': np.array([life[lab] for labs in fwd for lab in labs], np.float32)}
        if variant == 'fused' and strides:
            st = {}
            for g in strides:
                sub_pos = pos[::g]
                sub_ids = relink(c, cfg, frames=sub_pos, inputs=inputs, step_index=idx_all,
                                 require_finite_origin=False)
                res['runs'][f'{variant}_fwd_g{g}'] = sub_ids
                sub = _labels_per_frame(sub_ids, fo, sub_pos)
                sub_idx = idx[::g]
                tp = n_sub = n_full = 0
                for m in range(len(sub) - 1):
                    if sub_idx[m + 1] - sub_idx[m] != g:
                        continue
                    k0, k1 = m * g, (m + 1) * g
                    ls, lf = R.links(sub[m], sub[m + 1]), R.links(fwd[k0], fwd[k1])
                    tp += len(ls & lf); n_sub += len(ls); n_full += len(lf)
                st[g] = {'agree': tp, 'sub_links': n_sub, 'full_links': n_full}
            res['strides'] = st
    return res


def rates(counts: list[dict]) -> dict:
    """Pooled rates from ``by_state`` counters (continuation, agreement, jumps per 1,000 links, new-ID rate)."""
    return R.rates(counts)


def pooled(results: list[dict], variant: str, states: tuple[str, ...]) -> dict:
    """Pool :func:`diagnostics` results of several sequences (prepare_figure_06_stats.py:330-341)."""
    return R.pooled(results, variant, states)
