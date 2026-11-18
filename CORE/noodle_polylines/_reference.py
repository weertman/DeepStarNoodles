# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Faithful, slow ports of the paper figure-script definitions: the parity oracle.

Every function here reproduces one block of the figure scripts of the DeepStarNoodles
methods paper (``paper/deepstarnoodles_methods/figures``) and the released
``FusedTracker`` (``CORE/deepstarnoodles/fused_tracker.py``, legacy path with the
``max_match_cost=90`` cap). The arithmetic lines are kept in the same order and with
the same dtypes as the originals, so the outputs are bit-identical where the originals
are deterministic. Only the plumbing differs: the scripts parse CSV strings and pickled
polyline archives; these ports take already-parsed values (the CSV round trip is exact).

**Never optimise this module.** The public modules (``geometry``, ``frames``,
``kinematics``, ``tracking``, ``scale``) may be rewritten for speed; they are checked
against these functions. Source locations are given as ``file:lines`` relative to the
figure directory, or ``FT`` for ``fused_tracker.py``.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

MODEL_SIZE = 512
K_SHAPE = 12      # prepare_figure_05_measures.py: K_SHAPE
K_TRACK = 12      # deepstarnoodles.runtime.K_TRACK
JUMP_DEG, JUMP_MIN_RADIUS = 30.0, 25.0                       # prepare_figure_06_stats.py:77
STRIDES = (2, 4, 8, 16)                                       # prepare_figure_06_stats.py:87
LIFETIME_GRID_S = np.array([1, 2, 5, 10, 20, 30, 60, 120, 300, 600])   # prepare_figure_06_stats.py:88
RELEASED = {'shape_gate': 120.0, 'embed_weight': 2.0, 'max_age': 16, 'max_match_cost': 90.0}
VARIANTS = {'fused': RELEASED,
            'shape_only': {**RELEASED, 'embed_weight': 0.0},
            'no_cap': {**RELEASED, 'max_match_cost': None}}


# ======================================================================== T0 / T1 per arm
def resample_arm(poly: np.ndarray, k: int = K_SHAPE) -> np.ndarray:
    """prepare_figure_05_measures.py:75-82. k points evenly spaced by arc length from base to tip."""
    poly = np.asarray(poly, float)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    if arc[-1] <= 0:
        return np.repeat(poly[:1], k, axis=0)
    at = np.linspace(0, arc[-1], k)
    return np.column_stack([np.interp(at, arc, poly[:, 0]), np.interp(at, arc, poly[:, 1])])


def rigid_residual(a: np.ndarray, b: np.ndarray) -> float:
    """prepare_figure_05_measures.py:85-91. RMS distance after the best rigid fit (SVD Kabsch)."""
    ca, cb = a - a.mean(0), b - b.mean(0)
    u, _s, vt = np.linalg.svd(ca.T @ cb)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, d]) @ vt
    return float(np.sqrt(((ca @ rotation - cb) ** 2).sum(1).mean()))


def arm_metrics(poly: np.ndarray) -> tuple[float, float]:
    """prepare_figure_05_measures.py:98-104 (Fig 5). Curvature NaN under 3 points."""
    segments = np.diff(poly, axis=0)
    length = float(np.linalg.norm(segments, axis=1).sum())
    if len(segments) < 2:
        return length, np.nan
    angles = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    return length, float(np.abs(np.diff(angles)).sum() / max(length, 1e-6))


def poly_metrics(polyline: np.ndarray) -> tuple[float, float, float]:
    """retrace_released.py:586-595 (export columns, Fig 4). Curvature 0.0 for 2 points."""
    if len(polyline) < 2:
        return 0.0, math.nan, math.nan
    segments = np.diff(polyline, axis=0)
    length = float(np.linalg.norm(segments, axis=1).sum())
    chord = float(np.linalg.norm(polyline[-1] - polyline[0]))
    angles = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    curvature = float(np.abs(np.diff(angles)).sum() / max(length, 1e-6)) if len(segments) > 1 else 0.0
    return length, curvature, length / max(chord, 1e-6)


def model_to_source(points: np.ndarray, left: float, top: float, side: float) -> np.ndarray:
    """retrace_released.py:579-583."""
    out = np.asarray(points, np.float64).copy()
    out[..., 0] = left + out[..., 0] * side / MODEL_SIZE
    out[..., 1] = top + out[..., 1] * side / MODEL_SIZE
    return out


def frame_radius(area: int, side: float) -> tuple[float, float]:
    """retrace_released.py:666,673 (traced frame): NaN radius for an empty support."""
    radius_model = float(np.sqrt(area / np.pi)) if area else math.nan
    radius_source = radius_model * side / MODEL_SIZE
    return radius_model, radius_source


def normalized_arm_length(length: float, radius_model: float) -> float:
    """retrace_released.py:695."""
    return length / max(radius_model, 1e-6) if np.isfinite(radius_model) else math.nan


def flip_origin_source(origin_model: np.ndarray, left: float, top: float, side: float) -> np.ndarray:
    """trace_sources.py:253-260 (runtime origin in source px)."""
    x, y = origin_model
    return np.array([float(left) + x * side / 512.0, float(top) + y * side / 512.0])


def tip_polar_stats(frames: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    """prepare_figure_06_stats.py:225-230 (float64 tips)."""
    out = []
    for f in frames:
        tips = np.array([a['poly'][-1] for a in f['arms']], float).reshape(-1, 2)
        out.append((np.degrees(np.arctan2(-tips[:, 1], tips[:, 0])) % 360, np.linalg.norm(tips, axis=1)))
    return out


def tip_angle_fig6_copy(f: dict) -> np.ndarray:
    """prepare_figure_06_flip.py:117-118 (6C): no float cast, so float32 tips give float32 angles.

    06_example:135-138 (6B) runs the same expression on polylines it cast to float64 (line 98), so 6B
    angles are float64 (= :func:`tip_polar_stats`)."""
    tips = np.array([a['poly'][-1] for a in f['arms']]).reshape(-1, 2)
    return np.degrees(np.arctan2(-tips[:, 1], tips[:, 0])) % 360


# ======================================================================== T4 FusedTracker (legacy path)
def tip_anchored_resample(poly_base_to_tip: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """FT:62-81."""
    pts = np.asarray(poly_base_to_tip, dtype=np.float32)[::-1]  # tip first
    if len(pts) == 1:
        return np.repeat(pts, k, axis=0), 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-6:
        return np.repeat(pts[:1], k, axis=0), 0.0
    targets = np.linspace(0.0, total, k)
    out = np.empty((k, 2), dtype=np.float32)
    j = 0
    for i, d in enumerate(targets):
        while j < len(seg) - 1 and d > cum[j + 1]:
            j += 1
        t = (d - cum[j]) / max(float(seg[j]), 1e-6)
        out[i] = (1.0 - t) * pts[j] + t * pts[j + 1]
    return out, total


def common_span_dist(pa: np.ndarray, la: float, pb: np.ndarray, lb: float) -> float:
    """FT:84-99."""
    k = len(pa)
    lc = min(la, lb)
    if lc <= 1e-6:
        return float(np.linalg.norm(pa[0] - pb[0]))
    s = np.linspace(0.0, lc, k)

    def sample(p: np.ndarray, total: float) -> np.ndarray:
        idx = s / max(total, 1e-6) * (k - 1)
        i0 = np.clip(idx.astype(int), 0, k - 2)
        t = (idx - i0)[:, None]
        return (1 - t) * p[i0] + t * p[i0 + 1]

    return float(np.linalg.norm(sample(pa, la) - sample(pb, lb), axis=1).mean())


class FusedTracker:
    """FT:196-442, the legacy (released) path only: no temporal lifecycle, no velocity by default.

    ``assign_hook``, if given, is called with every augmented (or plain) matrix passed to
    ``linear_sum_assignment`` and its solution (parity recording only).
    """

    def __init__(self, *, shape_gate: float = 120.0, embed_weight: float = 2.0,
                 embed_update_gate: float = 1.0, max_age: int = 16,
                 vel_alpha: float = 0.0, embed_alpha: float = 0.4,
                 max_vel_px: float = 30.0, max_match_cost: float | None = 90.0,
                 assign_hook=None):
        if max_match_cost is not None and max_match_cost <= 0:
            raise ValueError("max_match_cost must be positive or None")
        self.shape_gate = shape_gate
        self.max_match_cost = max_match_cost
        self.embed_weight = embed_weight
        self.embed_update_gate = embed_update_gate
        self.max_age = max_age
        self.vel_alpha = vel_alpha
        self.embed_alpha = embed_alpha
        self.max_vel_px = max_vel_px
        self.tracks: dict[int, dict] = {}
        self.next_id = 0
        self.assign_hook = assign_hook

    def _new(self, poly, length, embed, frame) -> int:
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {'poly': poly.astype(np.float32).copy(), 'len': float(length),
                            'vel': np.zeros(2, dtype=np.float32), 'embed': np.asarray(embed, np.float32).copy(),
                            'age': 0, 'last_frame': int(frame)}
        return tid

    def _predict(self, t: dict, frame: int) -> np.ndarray:
        dt = float(max(frame - t['last_frame'], 1))
        v = t['vel']
        speed = float(np.linalg.norm(v))
        if speed > self.max_vel_px:
            v = v * (self.max_vel_px / speed)
        return t['poly'] + v[None, :] * dt

    def _lsa(self, matrix):
        rr, cc = linear_sum_assignment(matrix)
        if self.assign_hook is not None:
            self.assign_hook(matrix, rr, cc)
        return rr, cc

    def step(self, frame: int, polys: np.ndarray, lengths: np.ndarray, embeds: np.ndarray) -> list[int]:
        n = len(polys)
        if n == 0:
            for t in self.tracks.values():
                t['age'] += 1
            self._retire()
            return []
        active = list(self.tracks.items())
        if not active:
            return [self._new(polys[i], lengths[i], embeds[i], frame) for i in range(n)]

        preds = {tid: self._predict(t, frame) for tid, t in active}
        big = 1e9
        cost = np.full((n, len(active)), big, dtype=np.float64)
        for i in range(n):
            for j, (tid, t) in enumerate(active):
                sd = common_span_dist(polys[i], float(lengths[i]), preds[tid], t['len'])
                if sd > self.shape_gate:
                    continue
                ed = float(np.linalg.norm(embeds[i] - t['embed']))
                cost[i, j] = sd + self.embed_weight * ed
        if self.max_match_cost is None:
            rr, cc = self._lsa(cost)
        else:
            m = len(active)
            half = self.max_match_cost / 2.0
            full = np.full((n + m, m + n), big, dtype=np.float64)
            full[:n, :m] = cost
            full[:n, m:][np.diag_indices(n)] = half
            full[n:, :m][np.diag_indices(m)] = half
            full[n:, m:] = 0.0
            rr, cc = self._lsa(full)
            keep = (rr < n) & (cc < m)
            rr, cc = rr[keep], cc[keep]
        labels = [-1] * n
        matched = set()
        for i, j in zip(rr, cc):
            if cost[i, j] >= big:
                continue
            tid = active[j][0]
            labels[i] = tid
            matched.add(tid)
            t = self.tracks[tid]
            dt = max(frame - t['last_frame'], 1)
            obs_vel = (polys[i][0] - t['poly'][0]) / dt
            t['vel'] = (1 - self.vel_alpha) * t['vel'] + self.vel_alpha * obs_vel.astype(np.float32)
            t['poly'] = polys[i].astype(np.float32).copy()
            t['len'] = float(lengths[i])
            ed = float(np.linalg.norm(embeds[i] - t['embed']))
            if ed <= self.embed_update_gate:
                t['embed'] = (1 - self.embed_alpha) * t['embed'] + self.embed_alpha * np.asarray(embeds[i], np.float32)
            t['age'] = 0
            t['last_frame'] = int(frame)
        for i in range(n):
            if labels[i] < 0:
                labels[i] = self._new(polys[i], lengths[i], embeds[i], frame)
                matched.add(labels[i])
        for tid in list(self.tracks):
            if tid not in matched:
                self.tracks[tid]['age'] += 1
        self._retire()
        return labels

    def _retire(self) -> None:
        for tid in [k for k, t in self.tracks.items() if t['age'] > self.max_age]:
            del self.tracks[tid]


def assign(cost: np.ndarray, max_match_cost: float | None) -> list[tuple[int, int]]:
    """prepare_figure_06_stats.py:203-217 (NaN = gated)."""
    big = 1e9
    c = np.nan_to_num(cost, nan=big)
    n, m = c.shape
    if max_match_cost is None:
        rr, cc = linear_sum_assignment(c)
    else:
        full = np.full((n + m, m + n), big)
        full[:n, :m] = c
        full[:n, m:][np.diag_indices(n)] = max_match_cost / 2
        full[n:, :m][np.diag_indices(m)] = max_match_cost / 2
        full[n:, m:] = 0.0
        rr, cc = linear_sum_assignment(full)
    return [(int(i), int(j)) for i, j in zip(rr, cc) if i < n and j < m and c[i, j] < big]


def relinked_track_ids(arm_frames: list[tuple[int, list[np.ndarray], np.ndarray, np.ndarray]],
                       config: dict | None = None, assign_hook=None) -> list[list[int]]:
    """prepare_figure_04_kinematics.py:85-107 over pre-grouped arm frames.

    ``arm_frames``: (frame, [poly float64 (n,2) base->tip], origin float64 (2,), embeds float32 (n,8)),
    frames with at least one arm, in time order. Returns the labels of each frame.
    """
    tracker = FusedTracker(**(config or {}), assign_hook=assign_hook)
    out = []
    for frame, polys, origin, embeds in arm_frames:
        sampled, lengths = zip(*(tip_anchored_resample(p - origin, K_TRACK) for p in polys))
        labels = tracker.step(frame, np.asarray(sampled, np.float32), np.asarray(lengths, np.float32), embeds)
        out.append([int(x) for x in labels])
    return out


def run_tracker(frames: list[dict], config: dict, reverse: bool = False, assign_hook=None) -> list[list[int]]:
    """prepare_figure_06_stats.py:185-200 (frames: {'index', 'arms': [{'poly', 'identity'}]}, poly body-relative)."""
    tracker = FusedTracker(**config, assign_hook=assign_hook)
    order = range(len(frames) - 1, -1, -1) if reverse else range(len(frames))
    last = frames[-1]['index']
    labels: list[list[int]] = [[] for _ in frames]
    for k in order:
        f = frames[k]
        if not f['arms']:
            continue
        sampled, lengths = zip(*(tip_anchored_resample(a['poly'], K_TRACK) for a in f['arms']))
        labels[k] = tracker.step(last - f['index'] if reverse else f['index'], np.asarray(sampled),
                                 np.asarray(lengths, np.float32), np.asarray([a['identity'] for a in f['arms']]))
    return labels


# ======================================================================== T4.5 diagnostics (Fig 6D)
def links(labels_a: list[int], labels_b: list[int]) -> set[tuple[int, int]]:
    """prepare_figure_06_stats.py:220-222."""
    where_b = {lab: j for j, lab in enumerate(labels_b)}
    return {(i, where_b[lab]) for i, lab in enumerate(labels_a) if lab in where_b}


def counters() -> dict:
    return dict.fromkeys(('continued', 'continuation_total', 'link_agree', 'link_union', 'links', 'jumps', 'arms',
                          'births'), 0)


def frame_measures(frames, fwd, rev, polar) -> dict[str, dict]:
    """prepare_figure_06_stats.py:238-274."""
    idx = np.array([f['index'] for f in frames])
    out: dict[str, dict] = defaultdict(counters)
    seen: set[int] = set()
    for k, labs in enumerate(fwd):
        for key in ('all', frames[k]['state']):
            if key is None:
                continue
            c = out[key]
            c['arms'] += len(labs)
            c['births'] += sum(lab not in seen for lab in labs)
        seen.update(labs)
    for k in np.flatnonzero(np.diff(idx) == 1):
        if not (frames[k]['arms'] and frames[k + 1]['arms']):
            continue
        prev = set(fwd[k])
        cont = sum(lab in prev for lab in fwd[k + 1])
        lf = links(fwd[k], fwd[k + 1])
        jumps = 0
        for i, j in lf:
            (a0, r0), (a1, r1) = polar[k], polar[k + 1]
            if r0[i] >= JUMP_MIN_RADIUS and r1[j] >= JUMP_MIN_RADIUS:
                jumps += abs((a1[j] - a0[i] + 180) % 360 - 180) > JUMP_DEG
        agree = union = 0
        if rev is not None:
            lr = links(rev[k], rev[k + 1])
            agree, union = len(lf & lr), len(lf | lr)
        for key in ('all', frames[k + 1]['state']):
            if key is None:
                continue
            c = out[key]
            c['continued'] += cont; c['continuation_total'] += len(fwd[k + 1])
            c['link_agree'] += agree; c['link_union'] += union
            c['links'] += len(lf); c['jumps'] += jumps
    return dict(out)


def analyse(seq: dict, variants: dict | None = None, strides: bool = True, run=run_tracker) -> dict:
    """prepare_figure_06_stats.py:277-315 (also returns every pass's labels under 'runs')."""
    variants = VARIANTS if variants is None else variants
    frames, fps = seq['frames'], seq['fps']
    idx = np.array([f['index'] for f in frames])
    polar = tip_polar_stats(frames)
    res: dict = {'name': seq['name'], 'n_frames': len(frames),
                 'n_arms': int(sum(len(f['arms']) for f in frames)),
                 'median_arms': float(np.median([len(f['arms']) for f in frames])),
                 'duration_s': float((idx[-1] - idx[0] + 1) / fps), 'runs': {}}
    for variant, config in variants.items():
        fwd = run(frames, config)
        rev = run(frames, config, reverse=True)
        res['runs'][f'{variant}_fwd'], res['runs'][f'{variant}_rev'] = fwd, rev
        span: dict[int, list[int]] = {}
        for k, labs in enumerate(fwd):
            for lab in labs:
                span.setdefault(lab, [idx[k], idx[k]])[1] = idx[k]
        life = {lab: (b - a + 1) / fps for lab, (a, b) in span.items()}
        res[variant] = {
            'by_state': frame_measures(frames, fwd, rev, polar),
            'n_ids': len(span),
            'arm_lifetime_s': np.array([life[lab] for labs in fwd for lab in labs], np.float32),
        }
        if variant == 'fused':
            res['labels_fused'] = fwd
        if variant == 'fused' and strides:
            st = {}
            for g in STRIDES:
                sub = run(frames[::g], config)
                res['runs'][f'{variant}_fwd_g{g}'] = sub
                sub_idx = idx[::g]
                tp = n_sub = n_full = 0
                for m in range(len(sub) - 1):
                    if sub_idx[m + 1] - sub_idx[m] != g:
                        continue
                    k0, k1 = m * g, (m + 1) * g
                    ls, lf = links(sub[m], sub[m + 1]), links(fwd[k0], fwd[k1])
                    tp += len(ls & lf); n_sub += len(ls); n_full += len(lf)
                st[g] = {'agree': tp, 'sub_links': n_sub, 'full_links': n_full}
            res['strides'] = st
    return res


def rates(counts: list[dict]) -> dict:
    """prepare_figure_06_stats.py:318-327."""
    tot = defaultdict(int)
    for c in counts:
        for k, v in c.items():
            tot[k] += v
    return {'continuation': tot['continued'] / max(tot['continuation_total'], 1),
            'forward_reverse_agreement': tot['link_agree'] / max(tot['link_union'], 1),
            'jumps_per_1000_links': 1000 * tot['jumps'] / max(tot['links'], 1),
            'new_id_rate': tot['births'] / max(tot['arms'], 1),
            'n_arms': tot['arms'], 'n_links': tot['links']}


def pooled(results: list[dict], variant: str, states: tuple[str, ...]) -> dict:
    """prepare_figure_06_stats.py:330-341."""
    v = [r[variant] for r in results]
    arm_life = np.concatenate([x['arm_lifetime_s'] for x in v])
    per_arm_ids = [x['n_ids'] / max(r['median_arms'], 1) / (r['duration_s'] / 600) for r, x in zip(results, v)]
    out = rates([x['by_state'].get('all', counters()) for x in v])
    out.update({
        'arm_weighted_median_lifetime_s': float(np.median(arm_life)),
        'fraction_arms_in_ids_lasting_ge': {int(t): float(np.mean(arm_life >= t)) for t in LIFETIME_GRID_S},
        'ids_per_arm_per_10min_median': float(np.median(per_arm_ids)),
        'by_state': {s: rates([x['by_state'].get(s, counters()) for x in v]) for s in states},
    })
    return out


def step_cost_fig6_step(tracker: FusedTracker, sampled, lengths, embeds, frame_index, config=RELEASED) -> dict:
    """prepare_figure_06_step.py:75-84 (tracks sorted by id; NaN = gated)."""
    active = sorted(tracker.tracks.items())
    shape = np.array([[common_span_dist(sampled[i], float(lengths[i]), tracker._predict(t, frame_index), t['len'])
                       for _, t in active] for i in range(len(sampled))])
    emb = np.array([[float(np.linalg.norm(embeds[i] - t['embed'])) for _, t in active] for i in range(len(sampled))])
    cost = np.where(shape <= config['shape_gate'], shape + config['embed_weight'] * emb, np.nan)
    return {'track_ids': [int(tid) for tid, _ in active], 'shape_distance': shape, 'embedding_distance': emb,
            'cost': cost, 'assignment': assign(cost, config['max_match_cost'])}


def step_cost_fig6_flip(tracker: FusedTracker, sampled, lengths, embeds, frame_index, config=RELEASED) -> dict:
    """prepare_figure_06_flip.py:79-89 (tracks in insertion order; NaN = gated)."""
    active = list(tracker.tracks.items())
    cost = np.full((len(sampled), len(active)), np.nan)
    for i in range(len(sampled)):
        for j, (tid, tr) in enumerate(active):
            sd = common_span_dist(np.asarray(sampled[i]), float(lengths[i]), tracker._predict(tr, frame_index), tr['len'])
            if sd <= config['shape_gate']:
                cost[i, j] = sd + config['embed_weight'] * float(np.linalg.norm(embeds[i] - tr['embed']))
    pairs = assign(cost, config['max_match_cost'])
    return {'track_ids': [int(tid) for tid, _ in active], 'cost': cost, 'assignment': [[a, b] for a, b in pairs]}


# ======================================================================== Figure loops
def fig4e_event(rows: list[dict], arms: dict[int, list[dict]], polylines: dict[int, np.ndarray],
                track_of: dict[int, int], event: dict) -> dict:
    """prepare_figure_04_kinematics.py:121-176 for one event (rows sorted by frame).

    rows: {'frame','time_s','has_star' (bool),'crop_side','body_src' (2,) support centroid src,
    'origin_src' (2,), 'support_area_model_px','body_radius_source_px'}; arms[frame]: {'row_id','crop_side',
    'tip_source' (2,),'arm_length_source_px','curvature_model_rad_px'}; polylines float64.
    """
    flip, up, right = event['flip_start_s'], event['upside_down_start_s'], event['right_side_up_start_s']
    in_window = [r for r in rows if event['first_time_s'] <= r['time_s'] <= event['last_time_s']]
    pre = [r['body_radius_source_px'] for r in in_window if r['has_star'] and r['time_s'] < flip]
    assert pre, 'no pre-flip frames'
    r0 = float(np.median(pre))
    area_pre = float(np.median([r['support_area_model_px'] * (r['crop_side'] / MODEL_SIZE) ** 2
                                for r in in_window if r['has_star'] and r['time_s'] < flip]))
    cols = defaultdict(list)
    previous = None
    previous_shapes: dict[int, np.ndarray] = {}
    for row in rows:
        frame = row['frame']
        t = row['time_s']
        star = row['has_star']
        centre = row['body_src'] if star else None
        origin = row['origin_src'] if star else None
        shapes = {track_of[a['row_id']]: resample_arm(polylines[a['row_id']]) * a['crop_side'] / MODEL_SIZE / r0
                  for a in (arms.get(frame, []) if star else [])}
        deformation = np.nan
        if previous is not None and shapes and previous_shapes:
            bends = [rigid_residual(previous_shapes[k], v) / (t - previous[0])
                     for k, v in shapes.items() if k in previous_shapes]
            if bends:
                deformation = float(np.median(bends))
        if event['first_time_s'] <= t <= event['last_time_s']:
            frame_arms = arms.get(frame, []) if star else []
            speed = fold = spread = curvature = area = np.nan
            if star and previous is not None and previous[1] is not None:
                speed = float(np.linalg.norm(centre - previous[1]) / r0 / (t - previous[0]))
            if frame_arms:
                tips = np.array([a['tip_source'] for a in frame_arms])
                fold = float(np.linalg.norm(tips.mean(0) - origin) / r0)
                lengths = np.array([a['arm_length_source_px'] for a in frame_arms]) / r0
                if len(lengths) >= 4:
                    q1, q3 = np.percentile(lengths, [25, 75])
                    spread = float(q3 - q1)
                kappa = [a['curvature_model_rad_px'] * MODEL_SIZE / a['crop_side'] for a in frame_arms]
                curvature = float(np.median(kappa) * r0)
            if star:
                area = row['support_area_model_px'] * (row['crop_side'] / MODEL_SIZE) ** 2 / area_pre
            cols['frame'].append(frame)
            cols['relative_time_s'].append(t - flip)
            cols['phase'].append(1 if flip <= t < up else 2 if up <= t < right else 0)
            cols['n_arms'].append(len(frame_arms))
            cols['body_speed'].append(speed)
            cols['arm_deformation'].append(deformation)
            cols['fold_asymmetry'].append(fold)
            cols['arm_length_spread'].append(spread)
            cols['arm_curvature'].append(curvature)
            cols['support_area'].append(area)
        previous = (t, centre)
        previous_shapes = shapes
    return {'r0': r0, 'area_pre': area_pre, 'pre_flip_frames': len(pre), 'columns': dict(cols)}


def fig4d_clip(rows: list[dict], arms: dict[int, list[dict]], polylines: dict[int, np.ndarray],
               track_of: dict[int, int], bouts: list[dict], context_s: float = 2.0) -> dict:
    """prepare_figure_04_tree_features.py:95-172 for one clip (side index omitted; rows as fig4e_event)."""
    windows = [(b['start_s'] - context_s, b['stop_s'] + context_s) for b in bouts]
    outside = [r['body_radius_source_px'] for r in rows
               if r['has_star'] and not any(lo <= r['time_s'] <= hi for lo, hi in windows)]
    r0 = float(np.median(outside))
    area0 = float(np.median([r['support_area_model_px'] * (r['crop_side'] / MODEL_SIZE) ** 2
                             for r in rows if r['has_star']
                             and not any(lo <= r['time_s'] <= hi for lo, hi in windows)]))
    out = defaultdict(list)
    for i, (lo, hi) in enumerate(windows):
        bout = bouts[i]
        previous = None
        previous_shapes: dict[int, np.ndarray] = {}
        for r in rows:
            frame = r['frame']
            t = r['time_s']
            star = r['has_star']
            centre = r['body_src'] if star else None
            origin = r['origin_src'] if star else None
            shapes = {}
            if star and (lo - 1 <= t <= hi):
                shapes = {track_of[a['row_id']]: resample_arm(polylines[a['row_id']]) * a['crop_side']
                          / MODEL_SIZE / r0 for a in arms.get(frame, [])}
            if lo <= t <= hi:
                frame_arms = arms.get(frame, []) if star else []
                values = dict.fromkeys(('fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'log_speed',
                                        'support_area', 'log_deformation'), np.nan)
                if star:
                    values['support_area'] = r['support_area_model_px'] * (r['crop_side'] / MODEL_SIZE) ** 2 / area0
                    if previous is not None and previous[1] is not None:
                        speed = np.linalg.norm(centre - previous[1]) / r0 / (t - previous[0])
                        values['log_speed'] = float(np.log10(max(speed, 1e-3)))
                if frame_arms:
                    scale = r['crop_side'] / MODEL_SIZE
                    offset = np.array([r['crop_left'], r['crop_top']])
                    source = [polylines[a['row_id']] * scale + offset for a in frame_arms]
                    tips = np.array([p[-1] for p in source])
                    values['fold_asymmetry'] = float(np.linalg.norm(tips.mean(0) - origin) / r0)
                    lengths = np.array([a['arm_length_source_px'] for a in frame_arms]) / r0
                    if len(lengths) >= 4:
                        q1, q3 = np.percentile(lengths, [25, 75])
                        values['arm_length_spread'] = float(q3 - q1)
                    kappa = [a['curvature_model_rad_px'] * MODEL_SIZE / a['crop_side'] for a in frame_arms]
                    values['arm_curvature'] = float(np.median(kappa) * r0)
                if previous is not None and shapes and previous_shapes:
                    bends = [rigid_residual(previous_shapes[k], v) / (t - previous[0])
                             for k, v in shapes.items() if k in previous_shapes]
                    if bends:
                        values['log_deformation'] = float(np.log10(max(float(np.median(bends)), 1e-3)))
                out['bout_id'].append(i)
                out['frame'].append(frame)
                out['time_s'].append(t)
                out['n_arms'].append(len(frame_arms))
                for name, v in values.items():
                    out[name].append(v)
            previous = (t, centre)
            previous_shapes = shapes
    return {'r0': r0, 'area0': area0, 'columns': dict(out)}


def fig5_window(frames: list[dict], onset: float, r0: float | None = None, area0: float | None = None,
                pre=(-300, -60), tracker_config: dict | None = None, assign_hook=None) -> dict | None:
    """prepare_figure_05_measures.py:146-210 (one boundary; frames: {'timestamp','crop','area','centroid',
    'origin','arms': [{'polyline' f32, 'identity' f32}]} in time order). Side index omitted."""
    side = np.array([f['crop'][2] for f in frames])
    radius_src = np.sqrt(np.array([f['area'] for f in frames]) / np.pi) * side / MODEL_SIZE
    rel = np.array([f['timestamp'] - onset for f in frames])
    base = (rel >= pre[0]) & (rel <= pre[1]) & (radius_src > 0)
    area_src = np.array([f['area'] for f in frames]) * (side / MODEL_SIZE) ** 2
    if r0 is None:
        if base.sum() < 30:
            return None
        r0 = float(np.median(radius_src[base]))
        area0 = float(np.median(area_src[base]))
    tracker = FusedTracker(**(tracker_config or dict(shape_gate=120.0, embed_weight=2.0, max_age=16)),
                           assign_hook=assign_hook)
    previous_tips: dict[int, np.ndarray] = {}
    previous_shapes: dict[int, np.ndarray] = {}
    previous_t = None
    names = ('body_speed', 'fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'support_area', 'tip_motion',
             'arm_deformation')
    rows = defaultdict(list)
    labels_per_frame = []
    for k, f in enumerate(frames):
        scale = f['crop'][2] / MODEL_SIZE
        body = np.asarray(f['centroid'], float)
        centre = np.asarray(f['origin'], float)
        values = dict.fromkeys(names, np.nan)
        values['support_area'] = area_src[k] / area0 if f['area'] else np.nan
        if k and np.all(np.isfinite(body)) and np.all(np.isfinite(frames[k - 1]['centroid'])):
            dt = f['timestamp'] - frames[k - 1]['timestamp']
            a = body * scale + f['crop'][:2]
            p = np.asarray(frames[k - 1]['centroid']) * frames[k - 1]['crop'][2] / MODEL_SIZE + frames[k - 1]['crop'][:2]
            values['body_speed'] = float(np.linalg.norm(a - p) / r0 / dt)
        labels = []
        if f['arms'] and np.all(np.isfinite(centre)):
            lengths, curvatures, tips, shapes = [], [], [], []
            sampled, track_lengths, identities = [], [], []
            for arm in f['arms']:
                poly = np.asarray(arm['polyline'], np.float32)
                length, curvature = arm_metrics(poly)
                lengths.append(length * scale / r0)
                curvatures.append(curvature / scale * r0)
                tips.append(poly[-1])
                shapes.append(resample_arm(poly) * scale / r0)
                pts, tl = tip_anchored_resample(poly - centre, K_TRACK)
                sampled.append(pts); track_lengths.append(tl); identities.append(arm['identity'])
            tips = np.asarray(tips)
            values['fold_asymmetry'] = float(np.linalg.norm(tips.mean(0) - centre) * scale / r0)
            if len(lengths) >= 4:
                q1, q3 = np.percentile(lengths, [25, 75])
                values['arm_length_spread'] = float(q3 - q1)
            values['arm_curvature'] = float(np.nanmedian(curvatures))
            labels = tracker.step(int(round(f['timestamp'] - frames[0]['timestamp'])), np.asarray(sampled),
                                  np.asarray(track_lengths, np.float32), np.asarray(identities, np.float32))
            relative_tips = (tips - centre) * scale / r0
            if previous_t is not None:
                dt = f['timestamp'] - previous_t
                moves = [np.linalg.norm(relative_tips[i] - previous_tips[t]) / dt
                         for i, t in enumerate(labels) if t in previous_tips]
                if moves:
                    values['tip_motion'] = float(np.median(moves))
                bends = [rigid_residual(previous_shapes[t], shapes[i]) / dt
                         for i, t in enumerate(labels) if t in previous_shapes]
                if bends:
                    values['arm_deformation'] = float(np.median(bends))
            previous_tips = {t: relative_tips[i] for i, t in enumerate(labels)}
            previous_shapes = {t: shapes[i] for i, t in enumerate(labels)}
            previous_t = f['timestamp']
        labels_per_frame.append([int(x) for x in labels])
        rows['relative_s'].append(rel[k])
        rows['n_arms'].append(len(f['arms']))
        for m in names:
            rows[m].append(values[m])
    return {'r0_source_px': r0, 'area0': area0, 'rows': dict(rows), 'labels': labels_per_frame}


def deformation_lag_a(frames: list[dict], r0: float) -> list[float]:
    """prepare_figure_04_kinematics.py:141-156,185-186: previous RECORDED frame, shapes reset every frame.

    frames: {'time_s','has_star','arms': [(track_id, poly float64, crop_side)]} in time order.
    """
    out = []
    previous = None
    previous_shapes: dict[int, np.ndarray] = {}
    for f in frames:
        t = f['time_s']
        shapes = {tid: resample_arm(poly) * side / MODEL_SIZE / r0
                  for tid, poly, side in (f['arms'] if f['has_star'] else [])}
        deformation = np.nan
        if previous is not None and shapes and previous_shapes:
            bends = [rigid_residual(previous_shapes[k], v) / (t - previous)
                     for k, v in shapes.items() if k in previous_shapes]
            if bends:
                deformation = float(np.median(bends))
        out.append(deformation)
        previous = t
        previous_shapes = shapes
    return out


def isotropy_frame(tips_minus_centre: np.ndarray) -> float:
    """prepare_figure_05_summary.py:96-104 per frame (NaN when fewer than 4 tips)."""
    if len(tips_minus_centre) >= 4:
        ev = np.linalg.eigvalsh(np.cov(tips_minus_centre.T))
        return float(np.sqrt(max(ev[0], 0.0) / max(ev[1], 1e-9)))
    return math.nan


def isotropy_window(segments: list[dict], onset_ts: float, fallback: float, pre=(-300, -60)) -> float:
    """prepare_figure_05_summary.py:89-105 (fallback NaN) / prepare_figure_05_example.py:74-92 (fallback 0.0).

    segments: dicts with 'timestamp','centroid','arm_offsets','point_offsets','points'."""
    values = []
    for d in segments:
        for k, ts in enumerate(d['timestamp']):
            if not pre[0] <= ts - onset_ts <= pre[1] or not np.all(np.isfinite(d['centroid'][k])):
                continue
            tips = np.array([d['points'][d['point_offsets'][a + 1] - 1]
                             for a in range(d['arm_offsets'][k], d['arm_offsets'][k + 1])]) - d['centroid'][k]
            if len(tips) >= 4:
                ev = np.linalg.eigvalsh(np.cov(tips.T))
                values.append(float(np.sqrt(max(ev[0], 0.0) / max(ev[1], 1e-9))))
    return float(np.median(values)) if values else fallback


def fig4b_skeleton(frame_rows: dict, arms_by_frame: dict, polylines: dict, frames) -> list[dict]:
    """prepare_figure_04.py:116-127 (rows: 'origin' f64, 'body_radius_model_px'); also keeps the unrounded path."""
    skeleton = []
    for frame in frames:
        row = frame_rows[frame]
        cx, cy = (float(v) for v in row['origin'])
        radius = float(row['body_radius_model_px'])
        assert radius > 0
        for arm in arms_by_frame.get(frame, []):
            path = polylines[arm['row_id']]
            xy = np.column_stack(((path[:, 0] - cx) / radius, (cy - path[:, 1]) / radius))
            skeleton.append({'frame': frame, 'row_id': arm['row_id'], 'xy_body_radii': np.round(xy, 3),
                             'xy_unrounded': xy})
    return skeleton


def fig4b_timeline(frame_rows: dict, arms_by_frame: dict, window) -> list[dict]:
    """prepare_figure_04.py:128-143 (state and support area omitted)."""
    timeline = []
    for frame in window:
        row = frame_rows[frame]
        arm_rows = arms_by_frame.get(frame, [])
        timeline.append({'frame': frame, 'n_arms': len(arm_rows),
                         'arm_lengths_body_radii': [float(a['normalized_arm_length']) for a in arm_rows],
                         'arm_curvatures_rad_per_body_radius': [
                             float(a['curvature_model_rad_px']) * float(row['body_radius_model_px'])
                             for a in arm_rows],
                         'mean_normalized_arm_length': (
                             float(np.mean([float(a['normalized_arm_length']) for a in arm_rows]))
                             if arm_rows else None)})
    return timeline


def anchored_ghosts(frames: list[dict], windows: dict) -> dict:
    """prepare_figure_05_example.py:122-145 (frames: {'t','crop','origin','arms' [f32 polys],'identity'}, sorted).

    Runs the example's own tracker loop, then anchors every arm of each window to the reference frame
    nearest the window centre. Raises ValueError for an empty window (as the script)."""
    tracker = FusedTracker(shape_gate=120.0, embed_weight=2.0, max_age=16)
    for f in frames:
        f['labels'] = []
        if f['arms'] and np.all(np.isfinite(f['origin'])):
            sampled, lengths = zip(*(tip_anchored_resample(p - f['origin'], K_TRACK) for p in f['arms']))
            f['labels'] = list(tracker.step(int(round(f['t'] - frames[0]['t'])), np.asarray(sampled),
                                            np.asarray(lengths, np.float32), np.asarray(f['identity'], np.float32)))
    out = {}
    for name, (lo, hi) in windows.items():
        window = [f for f in frames if lo <= f['t'] <= hi]
        ref = min(window, key=lambda f: abs(f['t'] - (lo + hi) / 2))
        scale_ref = ref['crop'][2] / MODEL_SIZE
        centre_ref_src = ref['origin'] * scale_ref + ref['crop'][:2]
        paths: dict[int, list] = {}
        ghosts: list = []
        for f in window:
            scale = f['crop'][2] / MODEL_SIZE
            centre_src = f['origin'] * scale + f['crop'][:2]
            for poly, label in zip(f['arms'], f['labels']):
                poly_src = poly * scale + f['crop'][:2]
                anchored = centre_ref_src + (poly_src - centre_src)
                ghosts.append((anchored - ref['crop'][:2]) / scale_ref)
                paths.setdefault(int(label), []).append(((anchored[-1] - ref['crop'][:2]) / scale_ref).tolist())
        out[name] = {'t_ref': ref['t'], 'ghosts': ghosts, 'labels': [int(x) for f in window for x in f['labels']],
                     'tip_paths': {str(k): v for k, v in paths.items() if len(v) >= 10}}
    return out


def sleepy_sequences(frames_by_individual: dict[str, list[dict]], max_gap_s: float = 16,
                     min_frames: int = 30) -> list[dict]:
    """prepare_figure_06_stats.py:119-151 without the ethogram (state=None).

    frames_by_individual: per individual, frames in segment order with 'timestamp','origin',
    'arms' [{'polyline' f32,'identity'}]."""
    by_individual: dict[str, dict[float, dict]] = defaultdict(dict)
    for ind, frames in frames_by_individual.items():
        for f in frames:
            by_individual[ind][round(f['timestamp'], 1)] = f
    sequences = []
    for ind, frames in sorted(by_individual.items()):
        ordered = [frames[t] for t in sorted(frames)]
        run: list[dict] = []
        for f in ordered:
            if run and f['timestamp'] - run[-1]['timestamp'] > max_gap_s:
                sequences.append(run); run = []
            run.append(f)
        sequences.append(run)
    out = []
    for k, run in enumerate(s for s in sequences if len(s) >= min_frames):
        t0 = run[0]['timestamp']
        frames = []
        for f in run:
            arms = [{'poly': a['polyline'] - f['origin'], 'identity': a['identity']} for a in f['arms']]
            frames.append({'index': int(round(f['timestamp'] - t0)), 'arms': arms, 'state': None,
                           'origin': f['origin'], 'timestamp': f['timestamp']})
        out.append({'name': f'sleepy_{k:03d}', 'fps': 1.0, 'frames': frames, 't0': t0})
    return out


# ======================================================================== plumbing: container -> script structures
# These build the per-frame/per-arm dicts the scripts read from their CSV and npz inputs. Derived export
# columns are recomputed with the export's own expressions (retrace_released.py:664-702); the released CSV
# values equal them bit for bit.
def flip_tables(c) -> tuple[list[dict], dict[int, list[dict]], dict[int, np.ndarray]]:
    """(rows sorted by frame, arms by frame index, float64 polylines by row id) as the Fig 4 scripts read them."""
    rows, arms, polylines = [], defaultdict(list), {}
    star = c.star_mask()
    for f in range(c.n_frames):
        left, top, side = (float(v) for v in c.crop[f])
        cx, cy = c.support_centroid_model[f]
        body_src = (model_to_source(np.array([[cx, cy]]), left, top, side)[0] if np.isfinite(cx)
                    else np.array([math.nan, math.nan]))
        area = int(c.support_area_model_px[f])
        radius_model, radius_source = frame_radius(area, side) if star[f] else (math.nan, math.nan)
        frame = int(c.frame_index[f])
        rows.append({'frame': frame, 'time_s': float(c.time_s[f]), 'has_star': bool(star[f]),
                     'crop_left': left, 'crop_top': top, 'crop_side': side, 'body_src': body_src,
                     'origin': np.asarray(c.body_origin_model[f], np.float64),
                     'origin_src': flip_origin_source(c.body_origin_model[f], left, top, side),
                     'support_area_model_px': area, 'body_radius_model_px': radius_model,
                     'body_radius_source_px': radius_source})
        for a in c.frame_arms(f):
            poly32 = np.asarray(c.arm(a), np.float32)
            length, curvature, _ = poly_metrics(poly32)
            src = model_to_source(poly32, left, top, side)
            arms[frame].append({'row_id': a, 'frame': frame, 'crop_side': side, 'tip_source': src[-1],
                                'arm_length_source_px': length * side / MODEL_SIZE,
                                'curvature_model_rad_px': curvature,
                                'normalized_arm_length': normalized_arm_length(length, radius_model)})
            polylines[a] = np.asarray(poly32, float)
    return rows, dict(arms), polylines


def relink_frames(c, polylines: dict[int, np.ndarray]) -> list[tuple]:
    """Input of :func:`relinked_track_ids`: frames with arms, float64 polylines and origin."""
    out = []
    for f in range(c.n_frames):
        arms = list(c.frame_arms(f))
        if arms:
            out.append((int(c.frame_index[f]), [polylines[a] for a in arms], np.asarray(c.body_origin_model[f], float),
                        np.asarray(c.identity[arms[0]:arms[-1] + 1], np.float32)))
    return out


def fig5_frames(c) -> list[dict]:
    """Frames as prepare_figure_05_measures.load_frames builds them (float32 polylines)."""
    return [{'timestamp': float(c.time_s[f]), 'crop': c.crop[f].copy(), 'area': int(c.support_area_model_px[f]),
             'centroid': c.support_centroid_model[f].copy(), 'origin': c.body_origin_model[f].copy(),
             'arms': [{'polyline': c.arm(a).copy(), 'identity': c.identity[a].copy()} for a in c.frame_arms(f)]}
            for f in range(c.n_frames)]


def stats_frames(c, frames=None, origin_float32: bool = True, states=None, index=None) -> list[dict]:
    """Sequence frames as prepare_figure_06_stats builds them: body-relative arms ('poly'), 'identity', 'index'.

    Flipping: star frames, float32 origin (``flipping_sequences``); Sleepy: float64 origin (``sleepy_sequences``).
    """
    pos = np.flatnonzero(c.star_mask()) if frames is None else frames
    idx = c.frame_index if index is None else index
    out = []
    for f in pos:
        o = c.body_origin_model[f].astype(np.float32) if origin_float32 else c.body_origin_model[f]
        out.append({'index': int(idx[f]), 'state': None if states is None else states[f],
                    'arms': [{'poly': c.arm(a) - o, 'identity': c.identity[a]} for a in c.frame_arms(f)]})
    return out
