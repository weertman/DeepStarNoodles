# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Post-hoc star identities for records annotated frame by frame at low frame rates (0.5-2 fps).

``track-videos`` uses this tracker by default. BoT-SORT, which live replay uses, matches boxes on IoU and confirms a
new track over one or two frames; at 1-2 fps a star moves about a body width between frames, so BoT-SORT splits one
star into many IDs and holds back each new track's first detection. This tracker is for the offline pass only.

It prefers a new ID to a wrong one: an ID that jumps to another animal mixes two animals in one per-animal sequence,
while a split only shortens the sequences. Distances are in body widths: the distance between two box centres divided
by ``sqrt(s_track * s_det)``, where ``s = sqrt(box area)`` and a track's size is the median of its last ``size_hist``
observed sizes. Time is the record's ``time_s`` (or ``frame / fps``), so missing records count like any other gap.

Per record:

1. Each track predicts its centre: its last centre plus its velocity times ``min(dt, velocity_horizon_s)``, with
   ``dt`` the seconds since it was last seen. The velocity is an exponential average (weight ``velocity_alpha``) of
   the steps between its observations; it restarts at zero after a gap of more than ``recent_s``.
2. A (track, detection) pair costs ``d / sigma_t + ln(sigma_t / sigma) + size_weight * |ln(s_det / s_track)|``, where
   ``d`` is the distance to the prediction and ``sigma_t = sqrt(sigma^2 + sigma_growth * max(dt - 1, 0))`` is the
   track's position spread, which grows while it is unseen. The log term makes an old track cost more than a fresh
   one at the same distance.
3. A pair is allowed when its cost is below ``new_cost`` and the sizes differ by at most a factor ``size_gate``. A
   track unseen for more than ``recent_s`` is lost: it may take a detection only within
   ``min(reacquire_gate + reacquire_gate_per_s * dt, reacquire_gate_max)`` widths of its predicted centre, and only
   when no other track is within ``reacquire_clear`` widths of that detection.
4. One global assignment (``scipy.optimize.linear_sum_assignment``) over all pairs, where leaving a track or a
   detection unmatched costs ``new_cost / 2`` each.
5. Ambiguity check. For each matched pair that has a rival (another allowed pair for its track or its detection,
   counting lost tracks that step 3 kept away), the assignment is solved again without that pair. When the best
   alternative gives the track another detection, or the detection to another track, and costs less than
   ``ambiguity`` more, the two readings are too close to tell apart: the track ends, so does an unmatched track the
   alternative would give the detection to, and the detection starts a new ID.
6. Lone jump. A track seen within ``recent_s`` and left unmatched takes a detection up to ``jump_gate`` widths away
   (a star moved by hand) when that detection is its only candidate and no other track or detection is within
   ``jump_gate`` widths.
7. Every remaining detection starts a new ID at once (no confirmation delay); IDs are numbered from 1 in order of
   birth. A track unseen for more than ``max_gap_s`` seconds ends.

Two boxes of one record with IoU >= ``dup_iou`` are the same star detected twice: the less confident one gets no ID
(``None``; ``track-videos`` moves it to ``untracked_stars``). ``min_conf_new`` is a confidence floor for starting a
track (default 0: off).

Deterministic: no randomness; tracks are kept in ID order and detections in record order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

_BIG = 1e9


@dataclass(frozen=True)
class HungarianParams:
    """Settings of :class:`HungarianStarTracker` (the defaults are the released ``track-videos`` settings)."""

    sigma: float = 0.25                # position spread (widths) of a track seen within the last second
    sigma_growth: float = 0.05         # growth of the spread's square (widths^2) per second unseen, after the first
    new_cost: float = 3.5              # price of a new ID plus an unmatched track; dearer pairs are never used
    ambiguity: float = 0.4             # a match whose best alternative costs less than this more is ambiguous (0 = off)
    velocity_alpha: float = 0.3        # weight of the latest step in a track's velocity (0 = no motion model)
    velocity_horizon_s: float = 5.0    # the velocity extrapolates at most this many seconds
    size_weight: float = 0.5           # weight of |ln size ratio| in the cost
    size_gate: float = 2.0             # max size ratio, either way
    size_hist: int = 3                 # a track's size is the median of its last size_hist observed sizes
    recent_s: float = 4.5              # a track unseen for longer than this is lost
    reacquire_gate: float = 0.75       # a lost track takes a detection within this many widths ...
    reacquire_gate_per_s: float = 0.02  # ... plus this many per second unseen ...
    reacquire_gate_max: float = 1.5    # ... up to this many,
    reacquire_clear: float = 1.5       # and only with no other track this close to the detection (0 = off)
    jump_gate: float = 2.5             # lone-jump reach (and clear radius) of a recent unmatched track (0 = off)
    max_gap_s: float = 30.0            # a track unseen for longer than this ends
    dup_iou: float = 0.95              # same-record IoU at which the less confident box is a duplicate (0 = off)
    min_conf_new: float = 0.0          # detections below this confidence start no track

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"star-tracker parameter {item.name} must be a finite number, got {value!r}")
        for name in ("sigma", "new_cost", "recent_s", "max_gap_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"star-tracker parameter {name} must be positive")
        for name in ("sigma_growth", "ambiguity", "velocity_horizon_s", "size_weight", "reacquire_gate_per_s",
                     "reacquire_clear", "jump_gate"):
            if getattr(self, name) < 0:
                raise ValueError(f"star-tracker parameter {name} must be non-negative")
        if not 0 < self.reacquire_gate <= self.reacquire_gate_max:
            raise ValueError("star-tracker parameters need 0 < reacquire_gate <= reacquire_gate_max")
        if not self.size_gate > 1:
            raise ValueError("star-tracker parameter size_gate must be greater than 1")
        if not isinstance(self.size_hist, int) or self.size_hist < 1:
            raise ValueError("star-tracker parameter size_hist must be a positive integer")
        for name in ("velocity_alpha", "dup_iou", "min_conf_new"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"star-tracker parameter {name} must be between 0 and 1")

    @classmethod
    def from_mapping(cls, values: dict | None) -> "HungarianParams":
        """Parameters from ``{name: value}`` (values may be strings, as ``--star-param`` gives them)."""
        values = dict(values or {})
        known = {item.name: item for item in fields(cls)}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"unknown star-tracker parameters: {', '.join(unknown)} "
                             f"(known: {', '.join(known)})")
        converted = {}
        for name, value in values.items():
            kind = int if known[name].type in (int, "int") else float
            try:
                number = float(value)
                if kind is int:
                    if not number.is_integer():
                        raise ValueError
                    number = int(number)
            except (TypeError, ValueError):
                raise ValueError(f"star-tracker parameter {name} must be {'an integer' if kind is int else 'a number'}"
                                 f", got {value!r}") from None
            converted[name] = number
        return cls(**converted)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Track:
    track_id: int
    centre: np.ndarray
    time_s: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    sizes: list = field(default_factory=list)
    size: float = 0.0


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _assign(cost: np.ndarray, unmatched: float) -> tuple[dict[int, int], float]:
    """The optimal ``{track row: detection column}`` and its total; an unmatched row or column costs ``unmatched``."""
    rows, columns = cost.shape
    size = rows + columns
    matrix = np.full((size, size), _BIG)
    matrix[:rows, :columns] = cost
    matrix[:rows, columns:][np.eye(rows, dtype=bool)] = unmatched
    matrix[rows:, :columns][np.eye(columns, dtype=bool)] = unmatched
    matrix[rows:, columns:] = 0.0
    row_index, column_index = linear_sum_assignment(matrix)
    pairs = {int(r): int(c) for r, c in zip(row_index, column_index) if r < rows and c < columns}
    return pairs, float(matrix[row_index, column_index].sum())


class HungarianStarTracker:
    """Online over one video's records in time order; call :meth:`update` once per record.

    After each call, :attr:`ended` lists the IDs that ended in that call (unseen for more than ``max_gap_s``, or ended
    by the ambiguity check), and :attr:`live_ids` the IDs still open. An ended ID never comes back.
    """

    def __init__(self, params: HungarianParams | None = None) -> None:
        self.params = params or HungarianParams()
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.ended: list[int] = []
        self._last_time: float | None = None

    @property
    def live_ids(self) -> list[int]:
        return [track.track_id for track in self.tracks]

    def _observe(self, track: _Track, centre: np.ndarray, size: float, t: float, *, moving: bool) -> None:
        p = self.params
        dt = t - track.time_s
        if moving and 0 < dt <= p.recent_s:
            track.velocity = (1 - p.velocity_alpha) * track.velocity + p.velocity_alpha * (centre - track.centre) / dt
        else:
            track.velocity = np.zeros(2)
        track.centre, track.time_s = centre, t
        track.sizes = (track.sizes + [float(size)])[-p.size_hist:]
        track.size = float(np.median(track.sizes))

    def update(self, stars: list[dict], time_s: float) -> list[int | None]:
        """IDs for one record's ``stars`` (each with ``crop.bbox_xyxy`` and ``crop.confidence``), in their order.

        ``None`` marks a duplicate box or one below ``min_conf_new``. ``time_s`` must not decrease.
        """
        p = self.params
        t = float(time_s)
        if not math.isfinite(t):
            raise ValueError(f"record time must be finite, got {time_s!r}")
        if self._last_time is not None and t < self._last_time:
            raise ValueError(f"record times must not decrease; {t} came after {self._last_time}")
        self._last_time = t
        self.ended = [track.track_id for track in self.tracks if t - track.time_s > p.max_gap_s]
        if self.ended:
            self.tracks = [track for track in self.tracks if t - track.time_s <= p.max_gap_s]
        n = len(stars)
        ids: list[int | None] = [None] * n
        if n == 0:
            return ids
        boxes = np.asarray([star["crop"]["bbox_xyxy"] for star in stars], dtype=np.float64).reshape(-1, 4)
        conf = np.asarray([float(star["crop"]["confidence"]) for star in stars], dtype=np.float64)
        if not (np.all(np.isfinite(boxes)) and np.all(np.isfinite(conf))):
            raise ValueError("star boxes and confidences must be finite")
        size = np.sqrt(np.clip(boxes[:, 2] - boxes[:, 0], 1e-6, None) * np.clip(boxes[:, 3] - boxes[:, 1], 1e-6, None))
        centre = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        active = np.ones(n, dtype=bool)
        if p.dup_iou > 0 and n > 1:
            iou = _iou_matrix(boxes)
            order = sorted(range(n), key=lambda i: (-conf[i], i))
            for rank, i in enumerate(order):
                if active[i]:
                    for j in order[rank + 1:]:
                        if active[j] and iou[i, j] >= p.dup_iou:
                            active[j] = False
        detections = np.flatnonzero(active)
        claimed = np.zeros(n, dtype=bool)        # detections given an existing track's ID
        if self.tracks and len(detections):
            ended = self._match(detections, centre, size, t, ids, claimed)
            if ended:
                self.ended += [self.tracks[row].track_id for row in sorted(ended)]
                self.tracks = [track for row, track in enumerate(self.tracks) if row not in ended]
        for i in detections.tolist():
            if claimed[i] or conf[i] < p.min_conf_new:
                continue
            track = _Track(self.next_id, centre[i], t, sizes=[float(size[i])], size=float(size[i]))
            self.next_id += 1
            self.tracks.append(track)
            ids[i] = track.track_id
        return ids

    def _match(self, detections, centre, size, t, ids, claimed) -> set[int]:
        """Steps 1-6 of the module docstring; sets ``ids``/``claimed`` and returns the rows of tracks that end."""
        p = self.params
        dt = np.asarray([max(t - track.time_s, 0.0) for track in self.tracks])
        velocity = np.stack([track.velocity for track in self.tracks])
        horizon = np.minimum(dt, p.velocity_horizon_s)[:, None]
        predicted = np.stack([track.centre for track in self.tracks]) + velocity * horizon
        track_size = np.asarray([track.size for track in self.tracks])
        det_size, det_centre = size[detections], centre[detections]
        distance = (np.linalg.norm(predicted[:, None, :] - det_centre[None, :, :], axis=2)
                    / np.sqrt(track_size[:, None] * det_size[None, :]))
        log_ratio = np.abs(np.log(det_size[None, :] / track_size[:, None]))
        spread = np.sqrt(p.sigma ** 2 + p.sigma_growth * np.maximum(dt - 1.0, 0.0))[:, None]
        cost = distance / spread + np.log(spread / p.sigma) + p.size_weight * log_ratio
        size_ok = log_ratio <= math.log(p.size_gate)
        lost = dt > p.recent_s
        reach = np.where(lost, np.minimum(p.reacquire_gate + p.reacquire_gate_per_s * dt, p.reacquire_gate_max), np.inf)
        plausible = size_ok & (cost < p.new_cost) & (distance <= reach[:, None])
        allowed = plausible.copy()
        if p.reacquire_clear > 0:
            crowded = (distance <= p.reacquire_clear).sum(axis=0) > 1
            allowed &= ~(lost[:, None] & crowded[None, :])
        unmatched = p.new_cost / 2
        best, total = _assign(np.where(allowed, cost, _BIG), unmatched)
        ended: set[int] = set()
        if p.ambiguity > 0:
            rival_cost = np.where(plausible, cost, _BIG)
            for row, column in best.items():
                if plausible[:, column].sum() < 2 and plausible[row, :].sum() < 2:
                    continue
                excluded = rival_cost.copy()
                excluded[row, column] = _BIG
                alternative, alternative_total = _assign(excluded, unmatched)
                takers = [other for other, taken in alternative.items() if taken == column]
                if (row in alternative or takers) and alternative_total - total < p.ambiguity:
                    ended.add(row)
                    ended.update(other for other in takers if other not in best)
        for row, column in best.items():
            if row not in ended:
                i = int(detections[column])
                self._observe(self.tracks[row], centre[i], size[i], t, moving=p.velocity_alpha > 0)
                ids[i] = self.tracks[row].track_id
                claimed[i] = True
        if p.jump_gate > 0:
            taken = set(best) | ended
            free = [column for column in range(len(detections)) if not claimed[detections[column]]]
            for row in range(len(self.tracks)):
                if row in taken or lost[row]:
                    continue
                candidates = [column for column in free
                              if distance[row, column] <= p.jump_gate and size_ok[row, column]]
                if len(candidates) != 1:
                    continue
                column = candidates[0]
                if (distance[:, column] <= p.jump_gate).sum() > 1 or (distance[row, :] <= p.jump_gate).sum() > 1:
                    continue
                i = int(detections[column])
                self._observe(self.tracks[row], centre[i], size[i], t, moving=False)
                ids[i] = self.tracks[row].track_id
                claimed[i] = True
                free.remove(column)
        return ended
