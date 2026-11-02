# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""One-time, offline converters from the released trace formats to :class:`PolylineContainer`.

* :func:`from_flipping_clip` - the flipping re-trace (``<stem>_frames.csv``, ``<stem>_arms.csv`` and the
  pickled ``<stem>_polylines.npz``).
* :func:`from_sleepy_segment` - a Sleepy Stars re-trace segment (``seg_*.npz``, already CSR).
* :func:`from_inference_json` - ``SCRIPTS/infer_video.py`` output (single-star JSON, multi-star
  schema-2 JSON, or JSON Lines with one frame record per line).

Values are parsed exactly as the figure scripts parse them (``float()`` of CSV text), so a converted
container reproduces the paper's numbers bit for bit.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np

from .container import STOP_REASON_CODES, ContainerError, PolylineContainer, _offsets

FLIPPING_FPS = 2.0     # not stored in the flipping contract; prepare_figure_06_stats.py:180 hard-codes it
SLEEPY_FPS = 1.0
TRACE_CONTRACT = {'tracer': 'released full-corpus GRU Noodle Tracer', 'invalid_seed_threshold': 0.6,
                  'resample_spacing_px': 4.0, 'smooth_window': 5}
TRACKER_CONFIG = {'shape_gate': 120.0, 'embed_weight': 2.0, 'max_age': 16, 'max_match_cost': 90.0,
                  'k_track': 12, 'body_origin': 'body_origin_model'}


def _stop_code(value) -> int:
    return STOP_REASON_CODES.get(str(value), -1)


# ------------------------------------------------------------------------------------------ flipping
def flipping_paths(folder: str | Path) -> tuple[Path, Path, Path]:
    folder = Path(folder)
    stem = folder.name
    return folder / f'{stem}_frames.csv', folder / f'{stem}_arms.csv', folder / f'{stem}_polylines.npz'


def from_flipping_clip(folder: str | Path, frame_range: tuple[int, int] | None = None,
                       body_origin: str = 'runtime') -> PolylineContainer:
    """Convert one released flipping clip folder (``.../clips/<session>/<animal>/<stem>``).

    .. warning:: The clip's ``_polylines.npz`` is a **pickled** object array and is read with
       ``allow_pickle=True``. Only call this on the released re-trace you produced yourself; never on
       files from an untrusted source. The container written from it is pickle-free.

    ``frame_range`` (inclusive frame indices) cuts an excerpt; ``body_origin='runtime'`` uses the
    runtime body origin (``runtime_body_c{x,y}_model``, the released trace set), ``'support'`` the
    support > 0.5 centroid. ``track_id`` is the stored association (equal to a full-clip re-link).
    """
    frame_path, arm_path, poly_path = flipping_paths(folder)
    with frame_path.open(newline='') as fh:
        frames = list(csv.DictReader(fh))
    with arm_path.open(newline='') as fh:
        arms = list(csv.DictReader(fh))
    if frame_range is not None:
        lo, hi = frame_range
        frames = [r for r in frames if lo <= int(r['frame']) <= hi]
        arms = [a for a in arms if lo <= int(a['frame']) <= hi]
    arms.sort(key=lambda a: (int(a['frame']), int(a['arm_index'])))
    with np.load(poly_path, allow_pickle=True) as z:      # internal input only (see warning)
        poly = {int(r): np.asarray(p, np.float32) for r, p in zip(z['row_id'], z['polylines'])}
    fidx = np.array([int(r['frame']) for r in frames], np.int64)
    pos = {int(f): k for k, f in enumerate(fidx)}
    counts = np.bincount(np.array([pos[int(a['frame'])] for a in arms], np.int64), minlength=len(fidx))
    P = [poly[int(a['row_id'])] for a in arms]
    f64 = lambda rows, key: np.array([float(r[key]) for r in rows], np.float64)   # noqa: E731
    ox, oy = (('runtime_body_cx_model', 'runtime_body_cy_model') if body_origin == 'runtime'
              else ('body_cx_model', 'body_cy_model'))
    c = PolylineContainer(
        points=(np.concatenate(P) if P else np.zeros((0, 2))).astype(np.float32),
        arm_point_offsets=_offsets(np.array([len(p) for p in P], np.int64)),
        frame_arm_offsets=_offsets(counts.astype(np.int64)),
        frame_index=fidx,
        time_s=f64(frames, 'time_s'),
        crop=np.column_stack([f64(frames, 'crop_left'), f64(frames, 'crop_top'), f64(frames, 'crop_side')]),
        body_origin_model=np.column_stack([f64(frames, ox), f64(frames, oy)]),
        body_origin_code=(np.array([int(r['runtime_body_source']) for r in frames], np.int8)
                          if body_origin == 'runtime' and frames and 'runtime_body_source' in frames[0] else None),
        support_area_model_px=np.array([int(r['support_area_model_px']) for r in frames], np.int32),
        support_centroid_model=np.column_stack([f64(frames, 'body_cx_model'), f64(frames, 'body_cy_model')]),
        has_star=np.array([r['has_star'] == '1' for r in frames], bool),
        identity=np.array([[float(a[f'identity_{i}']) for i in range(8)] for a in arms], np.float32).reshape(-1, 8),
        track_id=np.array([int(a['track_id']) for a in arms], np.int64),
        tip_score=np.array([float(a['tip_score']) for a in arms], np.float32),
        base_score=np.array([float(a['base_score']) for a in arms], np.float32),
        state_confidence=np.array([float(a['state_confidence']) for a in arms], np.float32),
        steps=np.array([int(a['steps']) for a in arms], np.int16),
        stop_reason=np.array([_stop_code(a['stop_reason']) for a in arms], np.int8),
        meta={'fps': FLIPPING_FPS, 'model_size': 512, 'trace_contract': TRACE_CONTRACT,
              'tracker_config': TRACKER_CONFIG, 'body_origin': body_origin, 'source_format': 'flipping'},
    )
    return c.validate()


# ------------------------------------------------------------------------------------------ sleepy
def from_sleepy_segment(path: str | Path, body_origin: str = 'runtime') -> PolylineContainer:
    """Convert one Sleepy Stars re-trace segment (``sleepy_q_to_sa/seg_*.npz``; pickle-free).

    Only traced frames are stored (``has_star`` is all true); there is no ``track_id`` (run
    :func:`noodle_polylines.tracking.relink`). ``time_s`` is unix time; ``frame_index`` is the source
    video frame. Pass-through scores are cast to float32.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    F = len(d['timestamp'])
    fidx = np.asarray(d['source_frame'] if 'source_frame' in d else d['row'], np.int64)
    if F > 1 and np.any(np.diff(fidx) <= 0):
        fidx = np.asarray(d['row'], np.int64)
    if body_origin == 'runtime':
        if 'body_center_runtime' not in d:
            raise ValueError(f'{path.name} lacks body_center_runtime (needed for the runtime body origin)')
        origin = np.asarray(d['body_center_runtime'], np.float64)
    else:
        origin = np.asarray(d['centroid'], np.float64)
    A = len(d['point_offsets']) - 1
    meta = {'fps': SLEEPY_FPS, 'model_size': 512, 'trace_contract': TRACE_CONTRACT, 'tracker_config': TRACKER_CONFIG,
            'body_origin': body_origin, 'source_format': 'sleepy', 'segment': path.stem}
    c = PolylineContainer(
        points=np.asarray(d['points'], np.float32),
        arm_point_offsets=np.asarray(d['point_offsets'], np.int64),
        frame_arm_offsets=np.asarray(d['arm_offsets'], np.int64),
        frame_index=fidx,
        time_s=np.asarray(d['timestamp'], np.float64),
        crop=np.asarray(d['crop'], np.float64),
        body_origin_model=origin,
        body_origin_code=(np.asarray(d['body_center_source'], np.int8)
                          if body_origin == 'runtime' and 'body_center_source' in d else None),
        support_area_model_px=np.asarray(d['support_area'], np.int32),
        support_centroid_model=np.asarray(d['centroid'], np.float64),
        has_star=np.ones(F, bool),
        identity=np.asarray(d['identity'], np.float32).reshape(-1, 8),
        tip_score=np.asarray(d['arm_tip_score'], np.float32) if 'arm_tip_score' in d else None,
        base_score=np.asarray(d['arm_base_score'], np.float32) if 'arm_base_score' in d else None,
        state_confidence=(np.asarray(d['arm_state_confidence'], np.float32)
                          if 'arm_state_confidence' in d else None),
        steps=np.asarray(d['arm_steps'], np.int16) if 'arm_steps' in d else None,
        stop_reason=(np.array([_stop_code(s) for s in d['arm_stop_reason']], np.int8).reshape(A)
                     if 'arm_stop_reason' in d else None),
        meta=meta,
    )
    return c.validate()


# ------------------------------------------------------------------------------------------ inference
def _frame_records(path: Path) -> Iterator[dict]:
    text = path.read_text()
    payload = None
    if path.suffix != '.jsonl':
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if 'frames' in rec:          # a whole payload on one line
                yield from rec['frames']
            else:
                yield rec
        return
    yield from (payload['frames'] if isinstance(payload, dict) else payload)


def from_inference_json(path: str | Path, fps: float | None = None) -> dict[int | None, PolylineContainer]:
    """Split an ``infer_video.py`` output into one container per star.

    Single-star records (``{"frame", "detected", "crop", "arms"}``) give one container keyed ``None``;
    multi-star records (``{"frame", "stars": [{"star_track_id", "crop", "arms"}]}``) give one per
    ``star_track_id``. Each arm record supplies ``polyline_crop`` (crop-512 px, base->tip),
    ``identity`` and an optional ``track_id`` (missing -> -1). If any arm of a star lacks ``identity``,
    that star's ``identity`` is left ``None`` (listed in ``meta['missing_fields']``), so a re-link fails
    with a clear :class:`ContainerError` instead of on NaN costs.

    The inference record has **no** body origin, timestamp or support area: those fields are left
    ``None`` (``time_s`` falls back to ``frame_index / fps`` when ``fps`` is given). Only the
    non-body-relative T0/T1 outputs and the association (T4, after setting ``body_origin_model``)
    can be computed from it.
    """
    path = Path(path)
    stars: dict[int | None, dict[str, list]] = {}
    for rec in _frame_records(path):
        if 'stars' in rec:
            entries = []
            for s in rec['stars']:
                sid = s.get('star_track_id')
                if sid is None:
                    raise ContainerError(f'{path.name}: frame {rec.get("frame")}: multi-star record without '
                                         'a star_track_id (null or missing)')
                entries.append((int(sid), s.get('crop'), s.get('arms', [])))
        else:
            entries = [(None, rec.get('crop'), rec.get('arms', []))] if (rec.get('detected', True)
                                                                         or rec.get('crop')) else []
            if not entries:
                entries = [(None, None, [])]
        for star, crop, arms in entries:
            s = stars.setdefault(star, {'frame': [], 'crop': [], 'has_star': [], 'polys': [], 'n': [],
                                        'identity': [], 'track_id': [], 'identity_missing': False})
            s['frame'].append(int(rec['frame']))
            s['has_star'].append(crop is not None)
            s['crop'].append([math.nan] * 3 if crop is None else [float(crop['left']), float(crop['top']),
                                                                   float(crop['side'])])
            s['n'].append(len(arms))
            for a in arms:
                if 'polyline_crop' not in a:
                    raise ValueError(f'{path.name}: frame {rec["frame"]}: arm record lacks polyline_crop')
                s['polys'].append(np.asarray(a['polyline_crop'], np.float32).reshape(-1, 2))
                ident = a.get('identity')
                s['identity_missing'] |= ident is None
                s['identity'].append(np.full(8, np.nan, np.float32) if ident is None
                                     else np.asarray(ident, np.float32).reshape(8))
                s['track_id'].append(int(a['track_id']) if a.get('track_id') is not None else -1)
    out = {}
    for star, s in stars.items():
        order = np.argsort(np.asarray(s['frame']), kind='stable')
        if not np.array_equal(order, np.arange(len(order))):
            raise ValueError(f'{path.name}: frames are not in increasing order')
        polys = s['polys']
        missing = ['body_origin_model', 'time_s', 'support_area_model_px']
        if s['identity_missing']:
            missing.append('identity')
        meta = {'model_size': 512, 'source_format': 'infer_video', 'star_track_id': star,
                'track_id_scope': 'star_track_id', 'missing_fields': missing}
        if fps is not None:
            meta['fps'] = float(fps)
        c = PolylineContainer(
            points=(np.concatenate(polys) if polys else np.zeros((0, 2))).astype(np.float32),
            arm_point_offsets=_offsets(np.array([len(p) for p in polys], np.int64)),
            frame_arm_offsets=_offsets(np.asarray(s['n'], np.int64)),
            frame_index=np.asarray(s['frame'], np.int64),
            crop=np.asarray(s['crop'], np.float64).reshape(-1, 3),
            has_star=np.asarray(s['has_star'], bool),
            identity=None if s['identity_missing'] else np.asarray(s['identity'], np.float32).reshape(-1, 8),
            track_id=np.asarray(s['track_id'], np.int64),
            meta=meta)
        out[star] = c.validate()
    return out


def load_any(path: str | Path, **kw) -> PolylineContainer:
    """Load a container ``.npz``, a Sleepy segment ``.npz`` or a flipping clip folder by sniffing the path."""
    path = Path(path)
    if path.is_dir():
        return from_flipping_clip(path, **kw)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
        if 'arm_point_offsets' in files:
            return PolylineContainer.load(path)
        if 'point_offsets' in files and 'timestamp' in files:
            return from_sleepy_segment(path, **kw)
    if path.suffix in ('.json', '.jsonl'):
        stars = from_inference_json(path, **kw)
        if len(stars) != 1:
            raise ValueError(f'{path.name} holds {len(stars)} stars; use from_inference_json')
        return next(iter(stars.values()))
    raise ValueError(f'cannot tell the format of {path}')
