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

Re-links with the released settings run the array-state tracker of :mod:`noodle_polylines._assoc`
(batched cost blocks, the same ``linear_sum_assignment`` calls, bit-identical labels); other settings
run the reference tracker. :func:`assignment_margin` measures how close an assignment is to a tie.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from . import _assoc as A
from . import _reference as R
from . import geometry as G
from .container import ContainerError, PolylineContainer


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
    (with ``vel_alpha = 0`` a track's prediction is its last shape). The whole (n, m) block is computed
    at once (:mod:`noodle_polylines._assoc`), with the values of :func:`common_span_dist` and
    ``np.linalg.norm`` pair by pair.
    """
    polys, track_polys = np.asarray(polys), np.asarray(track_polys)
    lengths = np.asarray(lengths, np.float64).reshape(-1)
    track_lengths = np.asarray(track_lengths, np.float64).reshape(-1)
    n, m = len(polys), len(track_polys)
    if n and m:
        k = polys.shape[1]

        def deg_fn(i, j):
            return R.common_span_dist(polys[i], float(lengths[i]), track_polys[j], float(track_lengths[j]))

        shape = A.span_dist_rows(A.arm_rows(polys, lengths, k), A.arm_rows(track_polys, track_lengths, k), k, deg_fn)
        emb = A.embed_dist(embeds, track_embeds)
    else:
        shape, emb = np.zeros((n, m)), np.zeros((n, m))
    cost = np.where(shape <= config.shape_gate, shape + config.embed_weight * emb, np.nan)
    return {'shape_distance': shape, 'embedding_distance': emb, 'cost': cost}


def assign(cost: np.ndarray, max_match_cost: float | None = 90.0) -> np.ndarray:
    """T4.3 (pairs, 2) int64 (row, column) assignment of a gated cost matrix (NaN = not allowed)."""
    return np.asarray(R.assign(cost, max_match_cost), np.int64).reshape(-1, 2)


def assignment_margin(matrix: np.ndarray, capped: bool = True, n: int | None = None) -> float:
    """Decision margin of one tracker assignment (near-tie audit): how much more the cheapest assignment
    of ``matrix`` whose *kept* pairs differ costs than the optimum (``inf`` if none exists).

    ``matrix`` is what the tracker passed to ``linear_sum_assignment`` (an ``assign_hook`` sees it): the
    augmented (n+m)^2 matrix when ``capped`` (``n`` arms; inferred as the zeros of the last, dummy row
    when omitted), else the plain (n, m) cost. Kept pairs are arm-track pairs with cost < 1e9. Every
    other assignment drops at least one edge of the optimum that touches an arm or a track, so the
    margin is the minimum over the re-solves with one such edge forbidden; alternatives that keep the
    same pairs (swaps among dummy or gated entries) are skipped. A margin below ~1e-6 means a last-bit
    change in a cost could flip the decision."""
    from scipy.optimize import linear_sum_assignment

    M = np.asarray(matrix, np.float64)
    if capped:
        N = M.shape[0]
        n = int(np.count_nonzero(M[-1] == 0.0)) if n is None else n
        m = N - n
    else:
        n, m = M.shape

    def kept(rr, cc):
        sel = (rr < n) & (cc < m)
        sel &= M[rr, cc] < A.BIG
        return set(zip(rr[sel].tolist(), cc[sel].tolist()))

    rr, cc = linear_sum_assignment(M)
    best = float(M[rr, cc].sum())
    ref = kept(rr, cc)
    margin = np.inf
    for r, col in zip(rr.tolist(), cc.tolist()):
        if r >= n and col >= m:
            continue                       # dummy-dummy edge: free to permute
        alt = M.copy()
        alt[r, col] = np.inf
        try:
            r2, c2 = linear_sum_assignment(alt)
        except ValueError:                 # no assignment avoids this edge
            continue
        if kept(r2, c2) != ref:
            margin = min(margin, float(alt[r2, c2].sum()) - best)
    return margin


def tracker_inputs(c: PolylineContainer, origin_dtype: str = 'float64', k: int = 12
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body-relative tip-first resamples (A,k,2) f32, lengths (A,) f32 and embeddings (A,8) f32.

    Inside a :meth:`PolylineContainer.caching` scope the resamples are computed once per container
    and origin dtype and returned read-only; outside one every call resamples afresh.

    ``origin_dtype`` 'float64' reproduces the figure scripts (``poly - origin`` in float64), 'float32' the
    runtime and Fig 6 stats path; on released data both give bit-identical inputs.
    """
    c.require('body_origin_model', 'identity')
    origin_dtype = resolve_origin_dtype(c, origin_dtype)
    if c.n_arms and not np.all(np.isfinite(c.identity)):
        raise ContainerError('identity has non-finite entries (arms without an identity embedding); '
                             'the tracker needs a finite (A,8) identity for every arm')

    def compute():
        origin = c.body_origin_model[c.arm_frame()].astype(origin_dtype)
        sampled, lengths = G.resample_tip(c.points, c.arm_point_offsets, k, arm_origin=origin)
        return sampled, lengths.astype(np.float32)

    # within c.caching(): one resample serves every tracker pass (relink, step costs, diagnostics)
    sampled, lengths = c.derived(('tracker_inputs', origin_dtype, k),
                                 ('points', 'arm_point_offsets', 'frame_arm_offsets', 'body_origin_model'), compute)
    return sampled, lengths, c.identity


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
    inputs = inputs if inputs is not None else tracker_inputs(c, origin_dtype, config.k)
    all_pos = np.arange(c.n_frames) if frames is None else np.asarray(frames, np.int64)
    pos = _frames(c, all_pos, require_finite_origin)
    return _run(c, config, inputs, pos, reverse, assign_hook, step_index, all_pos)


def _run(c: PolylineContainer, config: TrackerConfig, inputs: tuple, pos: np.ndarray, reverse: bool = False,
         assign_hook=None, step_index=None, all_pos=None, fast: bool | None = None) -> np.ndarray:
    """One tracker pass over frame positions ``pos`` (frames with arms). ``fast``: None checks whether the
    array tracker applies (:func:`noodle_polylines._assoc.fast_path_ok`); True/False forces the choice
    (callers pass True only after that check)."""
    sampled, lengths, embeds = inputs
    out = np.full(c.n_arms, -1, np.int64)
    fo = c.frame_arm_offsets
    order = pos[::-1] if reverse else pos
    if fast is None:
        fast = A.fast_path_ok(config, sampled, lengths, embeds, _arms_of(fo, pos))
    if fast:
        # array-state tracker: same matrices, same assignments (step_index is inert with vel_alpha 0)
        A.run(config, sampled, lengths, embeds, fo, order, out, assign_hook=assign_hook)
        return out
    all_pos = pos if all_pos is None else all_pos
    idx = c.frame_index if step_index is None else np.asarray(step_index)
    tracker = FusedTracker(config, assign_hook=assign_hook)
    last = int(idx[all_pos[-1]]) if len(all_pos) else 0
    for f in order:
        a0, a1 = fo[f], fo[f + 1]
        step = last - int(idx[f]) if reverse else int(idx[f])
        out[a0:a1] = tracker.step(step, sampled[a0:a1], lengths[a0:a1], embeds[a0:a1])
    return out


def _arms_of(fo: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Arm indices of frame positions ``pos``, in order."""
    a0, a1 = fo[pos], fo[pos + 1]
    cnt = a1 - a0
    start = np.zeros(len(pos), np.int64)
    np.cumsum(cnt[:-1], out=start[1:])
    return np.repeat(a0 - start, cnt) + np.arange(int(cnt.sum()), dtype=np.int64)


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
    fo = c.frame_arm_offsets
    prev = pos[pos < frame]
    a0, a1 = fo[frame], fo[frame + 1]
    if A.fast_path_ok(config, sampled, lengths, embeds, _arms_of(fo, np.append(prev, frame))):
        # the array tracker keeps its tracks in insertion order, which is ascending ID: both orders agree
        tr = A.run(config, sampled, lengths, embeds, fo, prev, np.full(c.n_arms, -1, np.int64))
        res = cost_matrix(sampled[a0:a1], lengths[a0:a1], embeds[a0:a1], tr.track_polys(), tr.track_lengths(),
                          tr.emb, config)
        res['track_ids'] = tr.tid.copy()
        res['assignment'] = assign(res['cost'], config.max_match_cost)
        res['labels'] = tr.step(A.arm_rows(sampled[a0:a1], lengths[a0:a1], config.k), embeds[a0:a1],
                                sampled[a0:a1], lengths[a0:a1])
        return res
    tracker = FusedTracker(config)
    for f in prev:
        tracker.step(int(c.frame_index[f]), sampled[fo[f]:fo[f + 1]], lengths[fo[f]:fo[f + 1]], embeds[fo[f]:fo[f + 1]])
    active = sorted(tracker.tracks.items()) if track_order == 'sorted' else list(tracker.tracks.items())
    preds = [tracker._predict(t, int(c.frame_index[frame])) for _, t in active]
    res = cost_matrix(sampled[a0:a1], lengths[a0:a1], embeds[a0:a1], preds, [t['len'] for _, t in active],
                      [t['embed'] for _, t in active], config)
    res['track_ids'] = np.array([tid for tid, _ in active], np.int64)
    res['assignment'] = assign(res['cost'], config.max_match_cost)
    res['labels'] = np.asarray(tracker.step(int(c.frame_index[frame]), sampled[a0:a1], lengths[a0:a1],
                                            embeds[a0:a1]), np.int64)
    return res


# ------------------------------------------------------------------------------------ T4.5 diagnostics
def resolve_index_by(c: PolylineContainer, by: str = 'auto') -> str:
    """The Fig 6D sequence index for ``c``: 'time' for Sleepy containers (``meta['source_format']``),
    'frame_index' otherwise. ``'auto'`` refuses a non-Sleepy container whose ``frame_index``
    :func:`~noodle_polylines.container.concat` renumbered: pass ``by=`` or ``index=`` explicitly."""
    if by != 'auto':
        if by not in ('frame_index', 'time'):
            raise ValueError("by must be 'auto', 'frame_index' or 'time'")
        return by
    if c.meta.get('source_format') == 'sleepy':
        return 'time'
    if c.meta.get('source_frame_index_renumbered'):
        raise ContainerError("frame_index was renumbered by concat; pass by='time' (or index=) to the diagnostics")
    return 'frame_index'


def resolve_origin_dtype(c: PolylineContainer, origin_dtype: str = 'auto') -> str:
    """Body-origin subtraction dtype: 'float32' for flipping containers (Fig 6C/6D subtract a float32
    origin, prepare_figure_06_stats.py:162-170), 'float64' otherwise (Sleepy, figure scripts)."""
    if origin_dtype == 'auto':
        return 'float32' if c.meta.get('source_format') == 'flipping' else 'float64'
    if origin_dtype not in ('float64', 'float32'):
        raise ValueError("origin_dtype must be 'auto', 'float64' or 'float32'")
    return origin_dtype


def sequence_index(c: PolylineContainer, by: str = 'auto') -> np.ndarray:
    """Per-frame integer index used by the diagnostics: ``frame_index`` (flipping) or
    ``round(time_s - time_s[0])`` (Sleepy sequences, prepare_figure_06_stats.py:147). ``'auto'``: see
    :func:`resolve_index_by`."""
    by = resolve_index_by(c, by)
    if by == 'frame_index':
        return c.frame_index
    t = c.times()
    return np.array([int(round(x - t[0])) for x in t], np.int64)


def diagnostics(c: PolylineContainer, frames: np.ndarray | None = None, states=None, fps: float | None = None,
                index: np.ndarray | None = None, variants: dict | None = None, strides=STRIDES,
                origin_dtype: str = 'auto', by: str = 'auto') -> dict:
    """T4.5 Fig 6D tracker self-consistency on one sequence (prepare_figure_06_stats.py:238-315).

    ``frames``: positions forming the sequence (default: star frames); ``states`` (F,) labels of the later
    frame used to split counts (None entries are skipped); ``index`` (F,) integer frame index, else ``by``
    ('frame_index' | 'time' | 'auto' = time for Sleepy, frame_index otherwise; :func:`resolve_index_by`).
    Sleepy sequences need the time index, or gaps of 2-16 s count as consecutive frames. ``origin_dtype``
    'auto' is float32 for flipping, float64 otherwise (:func:`resolve_origin_dtype`); the resolved
    choices are returned as ``index_by`` and ``origin_dtype``. An empty sequence gives zero counters.
    For each tracker variant: forward and reverse runs, ``by_state`` counters
    (continuation, forward-reverse link agreement, jumps > 30 deg with both tip radii >= 25 crop px,
    births), ``n_ids`` and arm-weighted ``arm_lifetime_s``; for 'fused' also stride precision
    (g = 2, 4, 8, 16). Returns the same structure as ``analyse`` plus ``runs`` (labels per pass, (A,)).
    """
    variants = VARIANTS if variants is None else variants
    pos = np.flatnonzero(c.star_mask()) if frames is None else np.asarray(frames, np.int64)
    fps = c.fps if fps is None else fps
    if fps is None or not fps > 0:
        raise ContainerError('diagnostics needs a positive fps (meta["fps"] or fps=)')
    origin_dtype = resolve_origin_dtype(c, origin_dtype)
    if index is None:
        by = resolve_index_by(c, by)
        idx_all = sequence_index(c, by)
    else:
        by, idx_all = 'index', np.asarray(index)
    if len(pos) == 0:
        return _empty_diagnostics(variants, strides, by, origin_dtype)
    fo = c.frame_arm_offsets
    angle, radius = G.tip_polar(c.points, c.arm_point_offsets, fo, c.body_origin_model, origin_dtype=origin_dtype)
    inputs = tracker_inputs(c, origin_dtype)
    seq = _Sequence(fo, pos, idx_all, states)
    res: dict = {'n_frames': len(pos), 'n_arms': int(seq.cnt.sum()), 'median_arms': float(np.median(seq.cnt)),
                 'duration_s': float((seq.idx[-1] - seq.idx[0] + 1) / fps), 'runs': {},
                 'index_by': by, 'origin_dtype': origin_dtype}
    run_pos = _frames(c, pos, False)              # every pass feeds the frames with arms
    sub_pos = {g: _frames(c, pos[::g], False) for g in (strides or ())}
    fast = {}

    def run(cfg, frames_, reverse=False):
        if cfg not in fast:   # one finiteness check per configuration (the pass frames are a subset)
            fast[cfg] = A.fast_path_ok(cfg, *inputs, seq.arms)
        return _run(c, cfg, inputs, frames_, reverse, step_index=idx_all, all_pos=pos, fast=fast[cfg])

    for variant, config in variants.items():
        cfg = config if isinstance(config, TrackerConfig) else TrackerConfig.from_dict(config)
        fwd_ids = run(cfg, run_pos)
        rev_ids = run(cfg, run_pos, reverse=True)
        res['runs'][f'{variant}_fwd'], res['runs'][f'{variant}_rev'] = fwd_ids, rev_ids
        res[variant] = seq.measures(fwd_ids[seq.arms], rev_ids[seq.arms], angle[seq.arms], radius[seq.arms], fps)
        if variant == 'fused' and strides:
            fwd = fwd_ids[seq.arms]
            st = {}
            for g in strides:
                sub_ids = run(cfg, sub_pos[g])
                res['runs'][f'{variant}_fwd_g{g}'] = sub_ids
                st[g] = seq.stride_agreement(g, fwd, sub_ids)
            res['strides'] = st
    return res


class _Sequence:
    """The frames of one diagnostics sequence, flattened: arms in sequence order (frame k, then arm
    order), with each arm's sequence position ``k`` and within-frame index. The counters of
    ``frame_measures`` / ``analyse`` (prepare_figure_06_stats.py:238-315) are computed with sorted
    (position, label) keys instead of per-frame sets; they give the same integers because a frame's
    labels are unique (one arm per track and step) and every arm of a processed frame has one."""

    def __init__(self, fo: np.ndarray, pos: np.ndarray, idx_all: np.ndarray, states):
        self.pos = pos
        self.K = K = len(pos)
        self.idx = np.array([int(idx_all[f]) for f in pos], np.int64)
        self.cnt = (fo[pos + 1] - fo[pos]).astype(np.int64)
        self.arms = _arms_of(fo, pos)
        self.k = np.repeat(np.arange(K, dtype=np.int64), self.cnt)
        start = np.zeros(K, np.int64)
        np.cumsum(self.cnt[:-1], out=start[1:])
        self.start = start
        self.within = np.arange(len(self.arms), dtype=np.int64) - start[self.k]
        # state codes in first-appearance order ('all' is code -1 and always first)
        self.state_names: list = []
        codes = np.full(K, -1, np.int64)
        if states is not None:
            seen: dict = {}
            for q, f in enumerate(pos):
                st = states[f]
                if st is None:
                    continue
                code = seen.get(st)
                if code is None:
                    code = seen[st] = len(self.state_names)
                    self.state_names.append(st)
                codes[q] = code
        self.codes = codes

    # ------------------------------------------------------------------ joins
    def _keys(self, labels: np.ndarray, k: np.ndarray, base: int) -> np.ndarray:
        return k * base + labels

    def partner(self, labels: np.ndarray, k: np.ndarray, query: np.ndarray, back: int) -> np.ndarray:
        """For each query arm (index into the flattened arms), the flattened index of the arm with the same
        label ``back`` sequence positions earlier, or -1."""
        base = int(labels.max()) + 1 if len(labels) else 1
        keys = self._keys(labels, k, base)
        order = np.argsort(keys, kind='stable')
        sk = keys[order]
        q = self._keys(labels[query], k[query] - back, base)
        at = np.searchsorted(sk, q)
        hit = at < len(sk)
        hit[hit] = sk[at[hit]] == q[hit]
        out = np.full(len(query), -1, np.int64)
        out[hit] = order[at[hit]]
        return out

    def _check(self, labels: np.ndarray) -> None:
        if len(labels) and labels.min() < 0:
            raise ContainerError('diagnostics: a processed frame has an arm without a track label')
        if len(labels) > 1:
            key = self.k * (int(labels.max()) + 1) + labels
            if len(np.unique(key)) != len(key):
                raise ContainerError('diagnostics: a label occurs twice in one frame')

    def _by_state(self, per_frame: dict[str, np.ndarray], frame_of: dict[str, np.ndarray]) -> dict:
        """Sum per-frame counters into 'all' and each state (frame_of: which frame's state counts)."""
        names = ('continued', 'continuation_total', 'link_agree', 'link_union', 'links', 'jumps', 'arms',
                 'births')
        out = {'all': {n: int(per_frame[n].sum()) for n in names}}
        S = len(self.state_names)
        if S:
            sums = {}
            for n in names:
                code = self.codes[frame_of[n]]
                ok = code >= 0
                sums[n] = np.bincount(code[ok], weights=per_frame[n][ok], minlength=S)
            for s, st in enumerate(self.state_names):
                out[st] = {n: int(round(sums[n][s])) for n in names}
        return out

    # ------------------------------------------------------------------ Fig 6D counters
    def measures(self, fwd: np.ndarray, rev: np.ndarray, angle: np.ndarray, radius: np.ndarray, fps: float) -> dict:
        """``by_state`` counters, ``n_ids`` and ``arm_lifetime_s`` of one forward / reverse pair of passes
        (labels per flattened arm)."""
        self._check(fwd)
        self._check(rev)
        K, k = self.K, self.k
        # births: arms whose label first appears at this frame; lifetimes from first / last appearance
        uniq, first, inv = np.unique(fwd, return_index=True, return_inverse=True)
        last = len(fwd) - 1 - np.unique(fwd[::-1], return_index=True)[1]
        births = np.bincount(k[first], minlength=K)
        life = (self.idx[k[last]] - self.idx[k[first]] + 1) / fps
        # consecutive frame pairs (k, k+1): index step 1, both with arms
        pair_ok = np.zeros(K, bool)
        if K > 1:
            pair_ok[1:] = (np.diff(self.idx) == 1) & (self.cnt[:-1] > 0) & (self.cnt[1:] > 0)
        q = np.flatnonzero(pair_ok[k])                       # arms of the later frame of a valid pair
        pf = self.partner(fwd, k, q, 1)
        pr = self.partner(rev, k, q, 1)
        linked = pf >= 0
        i, j = pf[linked], q[linked]
        r_ok = (radius[i] >= JUMP_MIN_RADIUS) & (radius[j] >= JUMP_MIN_RADIUS)
        jump = np.zeros(len(q), bool)
        jump[linked] = r_ok & (np.abs((angle[j] - angle[i] + 180) % 360 - 180) > JUMP_DEG)
        agree = linked & (pf == pr)
        kq = k[q]
        per_frame = {
            'continued': np.bincount(kq, weights=linked, minlength=K),
            'continuation_total': np.where(pair_ok, self.cnt, 0),
            'link_agree': np.bincount(kq, weights=agree, minlength=K),
            'link_union': np.bincount(kq, weights=linked.astype(np.int64) + (pr >= 0) - agree, minlength=K),
            'links': np.bincount(kq, weights=linked, minlength=K),
            'jumps': np.bincount(kq, weights=jump, minlength=K),
            'arms': self.cnt, 'births': births}
        frame = np.arange(K)
        by_state = self._by_state(per_frame, dict.fromkeys(per_frame, frame))
        return {'by_state': by_state, 'n_ids': len(uniq), 'arm_lifetime_s': life[inv].astype(np.float32)}

    def stride_agreement(self, g: int, fwd: np.ndarray, sub_ids: np.ndarray) -> dict:
        """Stride precision at stride ``g``: links of the ``pos[::g]`` pass between sub-frames ``g`` index
        steps apart, against the full forward pass's links between the same two frames."""
        sub_k = np.arange(0, self.K, g)
        M = len(sub_k)
        ok = np.zeros(M, bool)
        if M > 1:
            ok[1:] = (self.idx[sub_k[1:]] - self.idx[sub_k[:-1]]) == g
        # arms of the sub-sequence, in order, with their sub position m
        is_sub = (self.k % g) == 0
        sub_arm = np.flatnonzero(is_sub)
        m = self.k[sub_arm] // g
        sub_lab = sub_ids[self.arms[sub_arm]]
        self._check_sub(sub_lab, m)
        q_local = np.flatnonzero(ok[m])                      # later frame of a valid sub pair
        ps = self.partner(sub_lab, m, q_local, 1)
        pf = self.partner(fwd, self.k, sub_arm[q_local], g)
        tp = int(np.count_nonzero((ps >= 0) & (pf >= 0) & (sub_arm[np.maximum(ps, 0)] == pf)))
        return {'agree': tp, 'sub_links': int(np.count_nonzero(ps >= 0)), 'full_links': int(np.count_nonzero(pf >= 0))}

    def _check_sub(self, labels: np.ndarray, m: np.ndarray) -> None:
        if len(labels) and labels.min() < 0:
            raise ContainerError('diagnostics: a processed frame has an arm without a track label')
        if len(labels) > 1:
            key = m * (int(labels.max()) + 1) + labels
            if len(np.unique(key)) != len(key):
                raise ContainerError('diagnostics: a label occurs twice in one frame')


def _empty_diagnostics(variants: dict, strides, by: str, origin_dtype: str) -> dict:
    """Diagnostics of a sequence without frames: zero counters, no runs."""
    res: dict = {'n_frames': 0, 'n_arms': 0, 'median_arms': float('nan'), 'duration_s': 0.0, 'runs': {},
                 'index_by': by, 'origin_dtype': origin_dtype}
    for variant in variants:
        res[variant] = {'by_state': {'all': R.counters()}, 'n_ids': 0, 'arm_lifetime_s': np.zeros(0, np.float32)}
        if variant == 'fused' and strides:
            res['strides'] = {g: {'agree': 0, 'sub_links': 0, 'full_links': 0} for g in strides}
    return res


def rates(counts: list[dict]) -> dict:
    """Pooled rates from ``by_state`` counters (continuation, agreement, jumps per 1,000 links, new-ID rate)."""
    return R.rates(counts)


def pooled(results: list[dict], variant: str, states: tuple[str, ...]) -> dict:
    """Pool :func:`diagnostics` results of several sequences (prepare_figure_06_stats.py:330-341).

    Results of empty sequences (``n_frames == 0``) carry no arms or links and are skipped."""
    kept = [r for r in results if r['n_frames'] > 0]
    if not kept:
        raise ValueError('pooled: no non-empty sequences')
    return R.pooled(kept, variant, states)
