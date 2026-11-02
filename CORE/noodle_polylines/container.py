# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""The canonical pickle-free CSR container: one sequence = one star in one video.

Layout (A arms, F frames, P points)::

    points             (P, 2) float32   base->tip polylines, crop-512 model px
    arm_point_offsets  (A+1,) int64     arm a = points[arm_point_offsets[a]:arm_point_offsets[a+1]]
    frame_arm_offsets  (F+1,) int64     frame f = arms frame_arm_offsets[f]:frame_arm_offsets[f+1]
    frame_index        (F,)   int64     strictly increasing video frame index

Optional per-frame fields: ``time_s`` (F,) f64, ``crop`` (F,3) f64 (left, top, side in source px),
``body_origin_model`` (F,2) f64, ``body_origin_code`` (F,) int8, ``support_area_model_px`` (F,) int32,
``support_centroid_model`` (F,2) f64, ``has_star`` (F,) bool. Optional per-arm fields: ``identity``
(A,8) f32, ``track_id`` (A,) int64 (-1 = absent), and pass-through ``tip_score``, ``base_score``,
``state_confidence`` (f32), ``steps`` (int16), ``stop_reason`` (int8). ``meta`` is a JSON-able dict
(``fps``, ``model_size``, ``trace_contract``, ``tracker_config``, ...), stored in the npz as a JSON
string, so every file loads with ``allow_pickle=False``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MODEL_SIZE = 512
STOP_REASON_CODES = {'base': 0, 'loop': 1, 'max_steps': 2, 'max_path': 3, 'lost': 4, 'stop': 5}
BODY_ORIGIN_CODES = {0: 'mouth peak', 1: 'support > 0.3 centroid', 2: 'crop centre', -1: 'not traced'}

FRAME_FIELDS = {'frame_index': (np.int64, ()), 'time_s': (np.float64, ()), 'crop': (np.float64, (3,)),
                'body_origin_model': (np.float64, (2,)), 'body_origin_code': (np.int8, ()),
                'support_area_model_px': (np.int32, ()), 'support_centroid_model': (np.float64, (2,)),
                'has_star': (np.bool_, ())}
ARM_FIELDS = {'identity': (np.float32, (8,)), 'track_id': (np.int64, ()), 'tip_score': (np.float32, ()),
              'base_score': (np.float32, ()), 'state_confidence': (np.float32, ()), 'steps': (np.int16, ()),
              'stop_reason': (np.int8, ())}


class ContainerError(ValueError):
    """The container violates the data model."""


@dataclass
class PolylineContainer:
    points: np.ndarray
    arm_point_offsets: np.ndarray
    frame_arm_offsets: np.ndarray
    frame_index: np.ndarray
    time_s: np.ndarray | None = None
    crop: np.ndarray | None = None
    body_origin_model: np.ndarray | None = None
    body_origin_code: np.ndarray | None = None
    support_area_model_px: np.ndarray | None = None
    support_centroid_model: np.ndarray | None = None
    has_star: np.ndarray | None = None
    identity: np.ndarray | None = None
    track_id: np.ndarray | None = None
    tip_score: np.ndarray | None = None
    base_score: np.ndarray | None = None
    state_confidence: np.ndarray | None = None
    steps: np.ndarray | None = None
    stop_reason: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ sizes and derived indices
    @property
    def n_frames(self) -> int:
        return len(self.frame_arm_offsets) - 1

    @property
    def n_arms(self) -> int:
        return len(self.arm_point_offsets) - 1

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def fps(self) -> float | None:
        v = self.meta.get('fps')
        return None if v is None else float(v)

    def arms_per_frame(self) -> np.ndarray:
        """(F,) int64 number of arms per frame (T2.1)."""
        return np.diff(self.frame_arm_offsets)

    def points_per_arm(self) -> np.ndarray:
        return np.diff(self.arm_point_offsets)

    def arm_frame(self) -> np.ndarray:
        """(A,) int64 frame position (0..F-1) of every arm."""
        return np.repeat(np.arange(self.n_frames, dtype=np.int64), self.arms_per_frame())

    def point_arm(self) -> np.ndarray:
        """(P,) int64 arm of every point."""
        return np.repeat(np.arange(self.n_arms, dtype=np.int64), self.points_per_arm())

    def arm(self, a: int) -> np.ndarray:
        return self.points[self.arm_point_offsets[a]:self.arm_point_offsets[a + 1]]

    def frame_arms(self, f: int) -> range:
        return range(int(self.frame_arm_offsets[f]), int(self.frame_arm_offsets[f + 1]))

    def times(self) -> np.ndarray:
        """time_s, or frame_index / fps when time_s is absent (explicit error if neither works)."""
        if self.time_s is not None:
            return self.time_s
        if self.fps is None:
            raise ContainerError('time_s is absent and meta has no fps')
        return self.frame_index / self.fps

    def star_mask(self) -> np.ndarray:
        return np.ones(self.n_frames, bool) if self.has_star is None else self.has_star.astype(bool)

    def require(self, *names: str) -> None:
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            raise ContainerError(f'container lacks required field(s): {", ".join(missing)}')

    def has_track_id(self) -> bool:
        return self.track_id is not None and self.n_arms > 0 and bool(np.all(self.track_id >= 0))

    # ------------------------------------------------------------------ validation
    def validate(self) -> 'PolylineContainer':
        p, ao, fo = self.points, self.arm_point_offsets, self.frame_arm_offsets
        if p.ndim != 2 or p.shape[1] != 2 or p.dtype != np.float32:
            raise ContainerError(f'points must be (P,2) float32, got {p.shape} {p.dtype}')
        for name, off, total in (('arm_point_offsets', ao, len(p)), ('frame_arm_offsets', fo, len(ao) - 1)):
            if off.ndim != 1 or off.dtype != np.int64 or len(off) < 1:
                raise ContainerError(f'{name} must be a non-empty 1-D int64 array')
            if off[0] != 0 or off[-1] != total or np.any(np.diff(off) < 0):
                raise ContainerError(f'{name} must start at 0, end at {total} and be non-decreasing')
        if self.n_arms and np.any(np.diff(ao) < 1):
            raise ContainerError('every arm needs at least one point')
        F, A = self.n_frames, self.n_arms
        if self.frame_index.shape != (F,) or self.frame_index.dtype != np.int64:
            raise ContainerError('frame_index must be (F,) int64')
        if F > 1 and np.any(np.diff(self.frame_index) <= 0):
            raise ContainerError('frame_index must be strictly increasing')
        for name, (dtype, tail) in FRAME_FIELDS.items():
            v = getattr(self, name)
            if v is not None and (v.shape != (F, *tail) or v.dtype != dtype):
                raise ContainerError(f'{name} must be {(F, *tail)} {np.dtype(dtype)}, got {v.shape} {v.dtype}')
        for name, (dtype, tail) in ARM_FIELDS.items():
            v = getattr(self, name)
            if v is not None and (v.shape != (A, *tail) or v.dtype != dtype):
                raise ContainerError(f'{name} must be {(A, *tail)} {np.dtype(dtype)}, got {v.shape} {v.dtype}')
        if self.time_s is not None and F > 1 and np.any(np.diff(self.time_s) < 0):
            raise ContainerError('time_s must be non-decreasing')
        if self.crop is not None:
            bad = ~(self.crop[:, 2] > 0) & ((np.diff(fo) > 0) | self.star_mask())
            if np.any(bad):
                raise ContainerError('crop side must be positive on frames with a star or arms')
        if not np.all(np.isfinite(p)):
            raise ContainerError('points must be finite')
        if self.track_id is not None and A and np.any(self.track_id < -1):
            raise ContainerError('track_id must be >= -1')
        json.dumps(self.meta)   # must be JSON-able
        return self

    # ------------------------------------------------------------------ I/O
    def arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for f in fields(self):
            if f.name == 'meta':
                continue
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = v
        return out

    def save(self, path: str | Path, compressed: bool = True) -> Path:
        """Write a pickle-free ``.npz`` (meta as a JSON string)."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.arrays()
        arrays['meta_json'] = np.array(json.dumps(self.meta, sort_keys=True))
        (np.savez_compressed if compressed else np.savez)(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path, meta: dict | None = None, validate: bool = True,
             strict: bool = True) -> 'PolylineContainer':
        """Read a container ``.npz`` with ``allow_pickle=False``. A sibling ``<stem>.json`` is merged into meta.

        Unknown arrays raise unless ``strict=False`` (then they are ignored)."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        m = json.loads(str(arrays.pop('meta_json'))) if 'meta_json' in arrays else {}
        side = path.with_suffix('.json')
        if side.exists():
            m = {**json.loads(side.read_text()), **m}
        if meta:
            m.update(meta)
        known = {f.name for f in fields(cls)}
        unknown = set(arrays) - known
        if unknown and not strict:
            arrays = {k: v for k, v in arrays.items() if k in known}
        elif unknown:
            raise ContainerError(f'{path.name}: unknown container field(s) {sorted(unknown)}')
        c = cls(**arrays, meta=m)
        return c.validate() if validate else c

    # ------------------------------------------------------------------ slicing
    def select_frames(self, frames: np.ndarray | slice) -> 'PolylineContainer':
        """A new container with the given frame positions (bool mask, slice or increasing indices)."""
        pos = np.arange(self.n_frames)[frames]
        if len(pos) > 1 and np.any(np.diff(pos) <= 0):
            raise ContainerError('frame positions must be strictly increasing')
        counts = self.arms_per_frame()[pos]
        arm_idx = _ranges(self.frame_arm_offsets[pos], counts)
        pcounts = self.points_per_arm()[arm_idx]
        pt_idx = _ranges(self.arm_point_offsets[arm_idx], pcounts)
        kw: dict[str, Any] = {
            'points': self.points[pt_idx],
            'arm_point_offsets': _offsets(pcounts),
            'frame_arm_offsets': _offsets(counts),
            'frame_index': self.frame_index[pos],
        }
        for name in FRAME_FIELDS:
            if name != 'frame_index' and getattr(self, name) is not None:
                kw[name] = getattr(self, name)[pos]
        for name in ARM_FIELDS:
            if getattr(self, name) is not None:
                kw[name] = getattr(self, name)[arm_idx]
        return PolylineContainer(**kw, meta=dict(self.meta))

    def time_window(self, lo: float, hi: float, relative_to: float = 0.0) -> 'PolylineContainer':
        """Frames with lo <= time_s - relative_to <= hi (closed)."""
        rel = self.times() - relative_to
        return self.select_frames(np.flatnonzero((rel >= lo) & (rel <= hi)))

    def with_fields(self, **kw) -> 'PolylineContainer':
        return replace(self, **kw)


def _offsets(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenate arange(s, s+c) for every (s, c) without a Python loop."""
    counts = np.asarray(counts, np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    off = _offsets(counts)[:-1]
    rep = np.repeat(np.asarray(starts, np.int64) - off, counts)
    return rep + np.arange(total, dtype=np.int64)


def concat(containers: Iterable[PolylineContainer], sort_by_time: bool = True,
           dedupe_time_decimals: int | None = None) -> PolylineContainer:
    """Concatenate containers of one animal (e.g. the Sleepy segments of an individual).

    With ``sort_by_time`` the frames are put in a stable time order (``list.sort`` semantics). With
    ``dedupe_time_decimals`` frames whose rounded times collide keep the LAST one seen, as the Fig 6
    Sleepy sequences do (``round(timestamp, 1)`` keys in a dict). ``frame_index`` must stay strictly
    increasing; when the parts come from different videos it is renumbered 0..F-1 and the original
    values are kept in ``meta['source_frame_index_renumbered']``.
    """
    parts = [c for c in containers]
    if not parts:
        raise ContainerError('nothing to concatenate')
    names = [n for n in (*FRAME_FIELDS, *ARM_FIELDS) if all(getattr(c, n) is not None for c in parts)]
    pts = np.concatenate([c.points for c in parts]) if parts else np.zeros((0, 2), np.float32)
    ppa = np.concatenate([c.points_per_arm() for c in parts])
    apf = np.concatenate([c.arms_per_frame() for c in parts])
    kw: dict[str, Any] = {'points': pts.astype(np.float32, copy=False), 'arm_point_offsets': _offsets(ppa),
                          'frame_arm_offsets': _offsets(apf)}
    for n in names:
        kw[n] = np.concatenate([getattr(c, n) for c in parts])
    kw.setdefault('frame_index', np.concatenate([c.frame_index for c in parts]))
    meta = dict(parts[0].meta)
    out = PolylineContainer(**kw, meta=meta)
    if sort_by_time or dedupe_time_decimals is not None:
        t = out.times()
        if dedupe_time_decimals is not None:
            keys = np.array([round(float(x), dedupe_time_decimals) for x in t])
            order = np.argsort(keys, kind='stable')
            k_sorted = keys[order]
            last = np.r_[k_sorted[1:] != k_sorted[:-1], True]   # last occurrence of each key
            order = order[last]
        else:
            order = np.argsort(t, kind='stable')
        out = _reorder(out, order)
    if out.n_frames > 1 and np.any(np.diff(out.frame_index) <= 0):
        out.meta['source_frame_index_renumbered'] = True
        out.frame_index = np.arange(out.n_frames, dtype=np.int64)
    return out


def _reorder(c: PolylineContainer, order: np.ndarray) -> PolylineContainer:
    """Frames in an arbitrary order (frame_index may become non-monotonic; caller fixes it)."""
    counts = c.arms_per_frame()[order]
    arm_idx = _ranges(c.frame_arm_offsets[order], counts)
    pcounts = c.points_per_arm()[arm_idx]
    pt_idx = _ranges(c.arm_point_offsets[arm_idx], pcounts)
    kw: dict[str, Any] = {'points': c.points[pt_idx], 'arm_point_offsets': _offsets(pcounts),
                          'frame_arm_offsets': _offsets(counts)}
    for name in FRAME_FIELDS:
        if getattr(c, name) is not None:
            kw[name] = getattr(c, name)[order]
    for name in ARM_FIELDS:
        if getattr(c, name) is not None:
            kw[name] = getattr(c, name)[arm_idx]
    return PolylineContainer(**kw, meta=dict(c.meta))


def split_at_gaps(c: PolylineContainer, max_gap_s: float, min_frames: int = 1) -> list[PolylineContainer]:
    """Split where consecutive frames are more than ``max_gap_s`` apart; keep parts with >= min_frames.

    Fig 6D Sleepy sequences use ``max_gap_s=16`` (the tracker max age reused as seconds) and
    ``min_frames=30`` (prepare_figure_06_stats.py:119-151).
    """
    t = c.times()
    if c.n_frames == 0:
        return []
    cuts = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    bounds = np.r_[0, cuts, c.n_frames]
    return [c.select_frames(slice(a, b)) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= min_frames]


def from_arrays(points, arm_point_offsets, frame_arm_offsets, frame_index=None, meta=None, **optional
                ) -> PolylineContainer:
    """Build a container from arrays, casting to the canonical dtypes."""
    fo = np.asarray(frame_arm_offsets, np.int64)
    kw: dict[str, Any] = {'points': np.asarray(points, np.float32).reshape(-1, 2),
                          'arm_point_offsets': np.asarray(arm_point_offsets, np.int64),
                          'frame_arm_offsets': fo,
                          'frame_index': (np.arange(len(fo) - 1, dtype=np.int64) if frame_index is None
                                          else np.asarray(frame_index, np.int64))}
    for name, value in optional.items():
        spec = FRAME_FIELDS.get(name) or ARM_FIELDS.get(name)
        if spec is None:
            raise ContainerError(f'unknown field {name}')
        kw[name] = None if value is None else np.asarray(value, spec[0])
    return PolylineContainer(**kw, meta=dict(meta or {})).validate()


def from_polylines(frames: list[list[np.ndarray]], **kw) -> PolylineContainer:
    """Build a container from a list of frames, each a list of (n_i, 2) base->tip polylines."""
    polys = [np.asarray(p, np.float32).reshape(-1, 2) for f in frames for p in f]
    pts = np.concatenate(polys) if polys else np.zeros((0, 2), np.float32)
    return from_arrays(pts, _offsets(np.array([len(p) for p in polys], np.int64)),
                       _offsets(np.array([len(f) for f in frames], np.int64)), **kw)
