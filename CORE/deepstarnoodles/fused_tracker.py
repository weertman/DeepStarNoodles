# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""FusedTracker: shape + motion + identity association for arm tracking.

Released video deploy uses this tracker with temporal lifecycle modes off
(shape_gate, embed_weight, and max_age only).

1. **Association is a polyline, not a point.** The decoder produces a full
   base->tip polyline. Association cost is a polyline shape distance, so two
   arms with nearby tips but different bases/shapes no longer collide.
2. **Optional motion model.** Tracks can keep an EMA tip velocity and be
   extrapolated by ``dt`` frames before matching (``vel_alpha > 0``). Measured
   at 2 fps this HURT identity purity (switch rate 0.0123 -> 0.0078 with it
   off) -- frame-to-frame tip displacement is mostly decode noise at this frame
   rate -- so it is disabled by default. It may help at higher frame rates.
3. **Embedding EMA drift.** The identity embedding is a weighted tiebreaker,
   not the primary gate, and a track's embedding is only updated when the
   observation agrees with it (outlier-gated), so one bad association cannot
   walk a track's identity onto another arm.

Two robustness details that matter in practice:

- All coordinates are **body-relative** (crop-512 px, origin at the body
  centroid), which removes whole-star translation and crop jitter -- a crawling
  star has slowly-varying body-relative arm shapes even when its absolute
  position changes fast.
- The shape distance is **tip-anchored over the common arc length**. Decoded
  arm length is unstable on the *base* side (the walk's termination varies
  frame to frame) while the tip end is the walk's seed and always present, so
  polylines are parameterized by arc length from the tip and compared only over
  the span both share. Same arm, shorter trace -> still a small distance.

Gating: a candidate pair only needs predicted-shape distance below
``shape_gate`` (generous: motion bursts legitimately produce large one-frame
shape changes). There is deliberately NO hard embedding veto: instrumentation
showed the embedding is most disturbed exactly during the motion bursts where
association is hardest, so a veto there causes id-birth cascades. The embedding
contributes a soft cost ``embed_weight * ||de||`` and disambiguation falls out
of Hungarian competition between candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment


def tip_anchored_resample(poly_base_to_tip: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Resample an (N,2) base->tip polyline to k points uniform in arc length
    FROM THE TIP (index 0 = tip). Returns (points (k,2), total_length)."""
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
    """Mean point distance between two tip-anchored uniform polylines, evaluated
    over the arc span both share (from the tip). Robust to base-side truncation."""
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


def body_center_crop(body_cx_src: float, body_cy_src: float,
                     crop_left: float, crop_top: float, crop_side: float,
                     out_size: float = 512.0) -> np.ndarray:
    """Body centroid (source px) -> crop-512 coordinates."""
    sc = out_size / max(crop_side, 1e-6)
    return np.array([(body_cx_src - crop_left) * sc, (body_cy_src - crop_top) * sc],
                    dtype=np.float32)


@dataclass(frozen=True)
class TemporalTrackerConfig:
    """Opt-in online tracking behavior.

    Supplying this object to :class:`FusedTracker` enables lifecycle,
    confidence, and existence handling.  Omitting it preserves the original
    tracker exactly.
    """

    confirmation_hits: int = 2
    tentative_max_misses: int = 1
    lost_after_misses: int = 1
    retire_after_misses: int | None = None
    high_quality: float = 0.6
    low_quality: float = 0.1
    suppress_tentative: bool = True
    existence_mode: Literal["none", "hysteresis", "log_odds", "hmm"] = "none"
    existence_initial: float = 0.65
    existence_confirm: float = 0.60
    existence_lost: float = 0.30
    existence_retire: float = 0.05
    hit_evidence: float = 0.85
    miss_evidence: float = 0.25
    hmm_survival: float = 0.95
    hmm_birth: float = 0.02
    motion_mode: Literal["legacy", "ema", "state"] = "legacy"
    state_process_noise: float = 4.0
    state_measurement_noise: float = 9.0
    mahalanobis_gate: float = 4.0

    def __post_init__(self) -> None:
        if self.confirmation_hits < 1:
            raise ValueError("confirmation_hits must be positive")
        if self.tentative_max_misses < 0 or self.lost_after_misses < 1:
            raise ValueError("invalid lifecycle miss limits")
        if self.retire_after_misses is not None and self.retire_after_misses < 1:
            raise ValueError("retire_after_misses must be positive")
        if not 0 <= self.low_quality <= self.high_quality <= 1:
            raise ValueError("quality thresholds must satisfy 0 <= low <= high <= 1")
        probabilities = (
            self.existence_initial, self.existence_confirm, self.existence_lost,
            self.existence_retire, self.hit_evidence, self.miss_evidence,
            self.hmm_survival, self.hmm_birth,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("existence parameters must be probabilities")
        if self.existence_mode not in {"none", "hysteresis", "log_odds", "hmm"}:
            raise ValueError(f"unknown existence mode: {self.existence_mode}")
        if self.motion_mode not in {"legacy", "ema", "state"}:
            raise ValueError(f"unknown motion mode: {self.motion_mode}")
        if self.state_process_noise < 0 or self.state_measurement_noise <= 0:
            raise ValueError("state noise values must be non-negative/positive")
        if self.mahalanobis_gate <= 0:
            raise ValueError("mahalanobis_gate must be positive")


def reconcile_forward_reverse(
    forward: list[list[int]], reverse: list[list[int]]
) -> dict[int, int]:
    """Map forward IDs to reverse IDs by deterministic maximum overlap.

    ``reverse`` must be in chronological order (reverse-pass outputs should be
    reversed by the caller).  Unassigned detections (negative IDs) are ignored.
    """
    if len(forward) != len(reverse):
        raise ValueError("forward and reverse sequences must have equal length")
    counts: dict[tuple[int, int], int] = {}
    for forward_frame, reverse_frame in zip(forward, reverse):
        if len(forward_frame) != len(reverse_frame):
            raise ValueError("corresponding frames must have equal detection counts")
        for left, right in zip(forward_frame, reverse_frame):
            if left >= 0 and right >= 0:
                counts[(left, right)] = counts.get((left, right), 0) + 1
    result: dict[int, int] = {}
    used_reverse: set[int] = set()
    candidates = sorted(
        ((-count, left, right) for (left, right), count in counts.items())
    )
    for _, left, right in candidates:
        if left not in result and right not in used_reverse:
            result[left] = right
            used_reverse.add(right)
    return result


class FusedTracker:
    """Hungarian tracker over body-relative tip-anchored polylines.

    step() consumes, per frame: frame index (for dt), per-arm tip-anchored
    K-point body-relative polylines + lengths, and per-arm identity embeddings.
    Returns persistent track labels, one per detection.
    """

    def __init__(self, *, shape_gate: float = 120.0,
                 embed_weight: float = 2.0,
                 embed_update_gate: float = 1.0, max_age: int = 16,
                 vel_alpha: float = 0.0, embed_alpha: float = 0.4,
                 max_vel_px: float = 30.0,
                 temporal: TemporalTrackerConfig | None = None):
        self.shape_gate = shape_gate
        self.embed_weight = embed_weight
        self.embed_update_gate = embed_update_gate
        self.max_age = max_age
        self.vel_alpha = vel_alpha      # weight of the NEW velocity estimate
        self.embed_alpha = embed_alpha  # weight of the NEW embedding (when accepted)
        self.max_vel_px = max_vel_px    # clamp predicted motion (rejects flip teleports)
        self.temporal = temporal
        self.tracks: dict[int, dict] = {}
        self.next_id = 0
        self.last_detection_diagnostics: list[dict[str, Any]] = []
        self.last_track_diagnostics: dict[int, dict[str, Any]] = {}

    def _new(self, poly: np.ndarray, length: float, embed: np.ndarray, frame: int,
             *, timestamp: float | None = None, quality: float = 1.0) -> int:
        tid = self.next_id
        self.next_id += 1
        confirmed = self.temporal is None or self.temporal.confirmation_hits <= 1
        self.tracks[tid] = {
            'poly': poly.astype(np.float32).copy(),
            'len': float(length),
            'vel': np.zeros(2, dtype=np.float32),
            'embed': np.asarray(embed, np.float32).copy(),
            'age': 0,
            'last_frame': int(frame),
            'last_time': float(frame if timestamp is None else timestamp),
            'state': 'confirmed' if confirmed else 'tentative',
            'hits': 1,
            'misses': 0,
            'existence': 1.0 if self.temporal is None else self.temporal.existence_initial,
            'quality': float(quality),
            'tip_cov': np.eye(2, dtype=np.float64) * (
                1.0 if self.temporal is None else self.temporal.state_measurement_noise
            ),
            'shape_residual': 0.0,
        }
        return tid

    def _dt(self, t: dict, frame: int, timestamp: float | None) -> float:
        current = float(frame if timestamp is None else timestamp)
        previous = float(t.get('last_frame', frame) if timestamp is None else t.get('last_time', current))
        if self.temporal is None:
            return float(max(frame - t['last_frame'], 1))
        return max(current - previous, 1e-6)

    def _predict(self, t: dict, frame: int, timestamp: float | None = None) -> np.ndarray:
        dt = self._dt(t, frame, timestamp)
        v = t['vel']
        speed = float(np.linalg.norm(v))
        if speed > self.max_vel_px:
            v = v * (self.max_vel_px / speed)
        return t['poly'] + v[None, :] * dt

    def step(self, frame: int, polys: np.ndarray, lengths: np.ndarray,
             embeds: np.ndarray, *, qualities: np.ndarray | None = None,
             observable: bool = True, timestamp: float | None = None) -> list[int]:
        """Associate detections and return one internal ID per detection.

        Optional arguments are active only when ``temporal`` was configured.
        Visibility (including tentative suppression) is reported in
        :attr:`last_detection_diagnostics`; IDs always remain aligned with the
        input detections.  A low-quality detection which cannot be associated
        receives ``-1`` because low-confidence detections never create tracks.
        """
        if self.temporal is None:
            labels = self._step_legacy(frame, polys, lengths, embeds)
            self.last_detection_diagnostics = [
                {'detection': i, 'track_id': tid, 'visible': True,
                 'quality': 1.0, 'stage': 'legacy', 'state': 'confirmed'}
                for i, tid in enumerate(labels)
            ]
            self._snapshot_track_diagnostics()
            return labels

        polys = np.asarray(polys)
        lengths = np.asarray(lengths)
        embeds = np.asarray(embeds)
        n = len(polys)
        if qualities is None:
            quality = np.ones(n, dtype=float)
        else:
            quality = np.asarray(qualities, dtype=float)
            if quality.shape == ():
                quality = np.full(n, float(quality))
            if quality.shape != (n,):
                raise ValueError("qualities must be scalar or have shape (n,)")
            quality = np.nan_to_num(quality, nan=0.0)
        labels = [-1] * n
        stages = ['below_threshold'] * n
        match_details: list[dict[str, float | None]] = [
            {'shape_distance': None, 'embedding_distance': None, 'mahalanobis': None}
            for _ in range(n)
        ]

        if n == 0:
            self._mark_unmatched(
                {tid for tid, track in self.tracks.items() if track['state'] != 'retired'},
                observable,
            )
            self._retire_temporal()
            self.last_detection_diagnostics = []
            self._snapshot_track_diagnostics()
            return []

        active = [(tid, track) for tid, track in self.tracks.items()
                  if track['state'] != 'retired']
        unmatched_tracks = {tid for tid, _ in active}
        high = [i for i in range(n) if quality[i] >= self.temporal.high_quality]
        low = [i for i in range(n)
               if self.temporal.low_quality <= quality[i] < self.temporal.high_quality]

        for indices, stage in ((high, 'high'), (low, 'low_recovery')):
            matches = self._match(
                indices, unmatched_tracks, frame, timestamp, polys, lengths, embeds
            )
            for detection, tid, shape_distance, embed_distance, mahalanobis in matches:
                labels[detection] = tid
                stages[detection] = stage
                match_details[detection] = {
                    'shape_distance': shape_distance,
                    'embedding_distance': embed_distance,
                    'mahalanobis': mahalanobis,
                }
                unmatched_tracks.remove(tid)
                self._update_track(
                    tid, polys[detection], float(lengths[detection]), embeds[detection],
                    frame, timestamp, float(quality[detection]), shape_distance,
                )

        # Only unmatched high-confidence detections can initiate tracks.
        for detection in high:
            if labels[detection] < 0:
                labels[detection] = self._new(
                    polys[detection], lengths[detection], embeds[detection], frame,
                    timestamp=timestamp, quality=float(quality[detection]),
                )
                stages[detection] = 'birth'

        self._mark_unmatched(unmatched_tracks, observable)
        self._retire_temporal()
        self.last_detection_diagnostics = []
        for i, tid in enumerate(labels):
            track = self.tracks.get(tid)
            state = None if track is None else track['state']
            visible = bool(
                track is not None
                and state == 'confirmed'
                and self._exists(track)
            )
            if track is not None and state == 'tentative' and not self.temporal.suppress_tentative:
                visible = self._exists(track)
            self.last_detection_diagnostics.append({
                'detection': i,
                'track_id': tid,
                'visible': visible,
                'quality': float(quality[i]),
                'stage': stages[i],
                'state': state,
                **match_details[i],
            })
        self._snapshot_track_diagnostics()
        return labels

    def _step_legacy(self, frame: int, polys: np.ndarray, lengths: np.ndarray,
                     embeds: np.ndarray) -> list[int]:
        """The original implementation, intentionally kept as a separate path."""
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
        rr, cc = linear_sum_assignment(cost)
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
            obs_vel = (polys[i][0] - t['poly'][0]) / dt  # tip displacement
            t['vel'] = (1 - self.vel_alpha) * t['vel'] + self.vel_alpha * obs_vel.astype(np.float32)
            t['poly'] = polys[i].astype(np.float32).copy()
            t['len'] = float(lengths[i])
            ed = float(np.linalg.norm(embeds[i] - t['embed']))
            if ed <= self.embed_update_gate:  # outlier-gated identity update
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

    def _match(
        self, detections: list[int], track_ids: set[int], frame: int,
        timestamp: float | None, polys: np.ndarray, lengths: np.ndarray,
        embeds: np.ndarray,
    ) -> list[tuple[int, int, float, float, float | None]]:
        if not detections or not track_ids:
            return []
        ordered_tracks = sorted(track_ids)
        big = 1e9
        cost = np.full((len(detections), len(ordered_tracks)), big, dtype=np.float64)
        details: dict[tuple[int, int], tuple[float, float, float | None]] = {}
        for row, detection in enumerate(detections):
            for column, tid in enumerate(ordered_tracks):
                track = self.tracks[tid]
                prediction = self._predict(track, frame, timestamp)
                shape_distance = common_span_dist(
                    polys[detection], float(lengths[detection]), prediction, track['len']
                )
                if shape_distance > self.shape_gate:
                    continue
                mahalanobis: float | None = None
                if self.temporal.motion_mode == 'state':
                    dt = self._dt(track, frame, timestamp)
                    covariance = track['tip_cov'] + np.eye(2) * (
                        self.temporal.state_process_noise * dt
                        + self.temporal.state_measurement_noise
                    )
                    residual = np.asarray(polys[detection][0], float) - prediction[0]
                    mahalanobis = float(np.sqrt(residual @ np.linalg.solve(covariance, residual)))
                    if mahalanobis > self.temporal.mahalanobis_gate:
                        continue
                embed_distance = float(np.linalg.norm(embeds[detection] - track['embed']))
                uncertainty_penalty = 0.0 if mahalanobis is None else mahalanobis
                cost[row, column] = (
                    shape_distance + self.embed_weight * embed_distance + uncertainty_penalty
                )
                details[(row, column)] = (
                    shape_distance, embed_distance, mahalanobis
                )
        rows, columns = linear_sum_assignment(cost)
        result = []
        for row, column in zip(rows, columns):
            if cost[row, column] >= big:
                continue
            shape_distance, embed_distance, mahalanobis = details[(row, column)]
            result.append((
                detections[row], ordered_tracks[column],
                shape_distance, embed_distance, mahalanobis,
            ))
        return result

    def _update_track(
        self, tid: int, poly: np.ndarray, length: float, embed: np.ndarray,
        frame: int, timestamp: float | None, quality: float, shape_distance: float,
    ) -> None:
        track = self.tracks[tid]
        dt = self._dt(track, frame, timestamp)
        old_tip = track['poly'][0].copy()
        observed_velocity = (np.asarray(poly[0], np.float32) - old_tip) / dt
        if self.temporal.motion_mode == 'legacy':
            alpha = self.vel_alpha
        elif self.temporal.motion_mode == 'ema':
            alpha = self.vel_alpha if self.vel_alpha > 0 else 0.4
        else:
            predicted_tip = self._predict(track, frame, timestamp)[0]
            prior_covariance = (
                track['tip_cov']
                + np.eye(2) * self.temporal.state_process_noise * dt
            )
            measurement_covariance = np.eye(2) * (
                self.temporal.state_measurement_noise / max(quality, 1e-3)
            )
            gain = prior_covariance @ np.linalg.inv(
                prior_covariance + measurement_covariance
            )
            residual = np.asarray(poly[0], float) - predicted_tip
            corrected_tip = predicted_tip + gain @ residual
            observed_velocity = (corrected_tip - old_tip) / dt
            track['tip_cov'] = (np.eye(2) - gain) @ prior_covariance
            alpha = self.vel_alpha if self.vel_alpha > 0 else 0.4
        track['vel'] = (
            (1 - alpha) * track['vel'] + alpha * observed_velocity.astype(np.float32)
        )
        # Observation is copied verbatim. Predictions are never written here.
        track['poly'] = np.asarray(poly, np.float32).copy()
        track['len'] = float(length)
        embed_distance = float(np.linalg.norm(embed - track['embed']))
        if embed_distance <= self.embed_update_gate:
            track['embed'] = (
                (1 - self.embed_alpha) * track['embed']
                + self.embed_alpha * np.asarray(embed, np.float32)
            )
        track['shape_residual'] = float(shape_distance)
        track['age'] = 0
        track['misses'] = 0
        track['hits'] += 1
        track['quality'] = quality
        track['last_frame'] = int(frame)
        track['last_time'] = float(frame if timestamp is None else timestamp)
        self._update_existence(track, hit=True)
        if track['hits'] >= self.temporal.confirmation_hits and self._exists(track):
            track['state'] = 'confirmed'
        elif track['state'] == 'lost':
            track['state'] = 'confirmed'

    def _mark_unmatched(self, unmatched: set[int], observable: bool) -> None:
        if not observable:
            return
        for tid in unmatched:
            track = self.tracks[tid]
            track['age'] += 1
            track['misses'] += 1
            self._update_existence(track, hit=False)
            if track['state'] == 'tentative':
                if track['misses'] > self.temporal.tentative_max_misses:
                    track['state'] = 'retired'
            elif (
                track['misses'] >= self.temporal.lost_after_misses
                or (
                    self.temporal.existence_mode != 'none'
                    and track['existence'] < self.temporal.existence_lost
                )
            ):
                track['state'] = 'lost'

    def _update_existence(self, track: dict, *, hit: bool) -> None:
        mode = self.temporal.existence_mode
        probability = float(track['existence'])
        if mode == 'none':
            return
        if mode == 'hysteresis':
            evidence = self.temporal.hit_evidence if hit else self.temporal.miss_evidence
            probability = 0.5 * probability + 0.5 * evidence
        elif mode == 'log_odds':
            eps = 1e-6
            evidence = self.temporal.hit_evidence if hit else self.temporal.miss_evidence
            prior_odds = log(np.clip(probability, eps, 1 - eps) /
                             np.clip(1 - probability, eps, 1 - eps))
            evidence_odds = log(np.clip(evidence, eps, 1 - eps) /
                                np.clip(1 - evidence, eps, 1 - eps))
            probability = 1.0 / (1.0 + exp(-(prior_odds + evidence_odds)))
        elif mode == 'hmm':
            predicted = (
                probability * self.temporal.hmm_survival
                + (1 - probability) * self.temporal.hmm_birth
            )
            likelihood_existing = (
                self.temporal.hit_evidence if hit else self.temporal.miss_evidence
            )
            likelihood_absent = 1 - likelihood_existing
            numerator = predicted * likelihood_existing
            denominator = numerator + (1 - predicted) * likelihood_absent
            probability = numerator / max(denominator, 1e-12)
        else:
            raise ValueError(f"unknown existence mode: {mode}")
        track['existence'] = float(np.clip(probability, 0.0, 1.0))

    def _exists(self, track: dict) -> bool:
        if self.temporal.existence_mode == 'none':
            return True
        threshold = (
            self.temporal.existence_lost if track['state'] == 'lost'
            else self.temporal.existence_confirm
        )
        return track['existence'] >= threshold

    def _retire_temporal(self) -> None:
        retire_after = (
            self.max_age if self.temporal.retire_after_misses is None
            else self.temporal.retire_after_misses
        )
        for tid, track in self.tracks.items():
            if (
                track['state'] == 'retired'
                or track['misses'] > retire_after
                or (
                    self.temporal.existence_mode != 'none'
                    and track['existence'] < self.temporal.existence_retire
                )
            ):
                track['state'] = 'retired'

    def predict_gaps(
        self, frame: int, *, timestamp: float | None = None,
        include_tentative: bool = False,
    ) -> dict[int, dict[str, Any]]:
        """Return conservative predicted products without mutating observations."""
        products: dict[int, dict[str, Any]] = {}
        for tid, track in sorted(self.tracks.items()):
            if track['state'] == 'retired':
                continue
            if track['state'] == 'tentative' and not include_tentative:
                continue
            prediction = self._predict(track, frame, timestamp).copy()
            dt = self._dt(track, frame, timestamp)
            covariance = track['tip_cov'].copy()
            if self.temporal is not None and self.temporal.motion_mode == 'state':
                covariance += np.eye(2) * self.temporal.state_process_noise * dt
            products[tid] = {
                'track_id': tid,
                'poly': prediction,
                'length': float(track['len']),
                'tip_covariance': covariance,
                'source': 'prediction',
                'last_observed_frame': int(track['last_frame']),
            }
        return products

    def _snapshot_track_diagnostics(self) -> None:
        self.last_track_diagnostics = {
            tid: {
                'track_id': tid,
                'state': track.get('state', 'confirmed'),
                'hits': int(track.get('hits', 1)),
                'misses': int(track.get('misses', track.get('age', 0))),
                'existence': float(track.get('existence', 1.0)),
                'last_frame': int(track['last_frame']),
                'quality': float(track.get('quality', 1.0)),
                'shape_residual': float(track.get('shape_residual', 0.0)),
                'tip_uncertainty': float(
                    np.sqrt(np.trace(track.get('tip_cov', np.eye(2))))
                ),
            }
            for tid, track in sorted(self.tracks.items())
        }

    def detection_diagnostics(self) -> list[dict[str, Any]]:
        """Return defensive copies of diagnostics for the most recent step."""
        return [dict(item) for item in self.last_detection_diagnostics]

    def track_diagnostics(self) -> dict[int, dict[str, Any]]:
        """Return defensive copies of current lifecycle diagnostics."""
        return {tid: dict(item) for tid, item in self.last_track_diagnostics.items()}

    def _retire(self) -> None:
        for tid in [k for k, t in self.tracks.items() if t['age'] > self.max_age]:
            del self.tracks[tid]
