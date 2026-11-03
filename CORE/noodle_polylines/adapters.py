# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""One-time, offline converters from the released trace formats to :class:`PolylineContainer`.

* :func:`from_flipping_clip` - the flipping re-trace (``<stem>_frames.csv``, ``<stem>_arms.csv`` and the
  pickled ``<stem>_polylines.npz``).
* :func:`from_sleepy_segment` - a Sleepy Stars re-trace segment (``seg_*.npz``, already CSR).
* :func:`from_inference_json` - ``SCRIPTS/infer_video.py`` output (single-star JSON, multi-star
  schema-2 JSON, or JSON Lines with one frame record per line; plain or ``.gz`` / ``.bz2`` / ``.xz`` /
  ``.zst``, as the backcast runner writes ``*.jsonl.zst``).

Values are parsed exactly as the figure scripts parse them (``float()`` of CSV text), so a converted
container reproduces the paper's numbers bit for bit. The CSV and JSON paths parse column-wise and
per star rather than per cell and per arm (round 5); the containers are byte-identical to the
per-cell readers' on every released clip and segment.
"""
from __future__ import annotations

import csv
from itertools import chain
import json
import math
from pathlib import Path
import re
import warnings
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


def _stop_codes(values) -> np.ndarray:
    """``[_stop_code(v) for v in values]`` as int8, mapping each distinct text once."""
    v = np.asarray(values)
    if v.size == 0:
        return np.zeros(v.shape, np.int8)
    uniq, inv = np.unique(v.reshape(-1), return_inverse=True)
    return np.array([_stop_code(u) for u in uniq], np.int8)[inv].reshape(v.shape)


# ------------------------------------------------------------------------------------------ flipping
def flipping_paths(folder: str | Path) -> tuple[Path, Path, Path]:
    folder = Path(folder)
    stem = folder.name
    return folder / f'{stem}_frames.csv', folder / f'{stem}_arms.csv', folder / f'{stem}_polylines.npz'


def _read_columns(path: Path, spec: dict[str, str], optional: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Read the named columns of a header-first CSV in one C pass (``np.loadtxt``, structured dtype).

    ``spec`` maps a column name to a numpy dtype: ``'f8'`` (parsed with the same correctly rounded
    conversion as Python ``float()``; checked bit for bit against ``float()`` of the text, ``nan``
    included), ``'i8'`` (``int()``), or ``'U'`` (the raw text). A column in ``optional`` that the
    header lacks is left out of the result; any other missing column raises ``KeyError``, as a
    ``csv.DictReader`` lookup would; of duplicate header names the last column wins, as in a
    ``DictReader`` row. Quoting follows the csv module (``"``), and ``#`` is not a comment. The one
    text ``float()`` / ``int()`` accept that this rejects (``ValueError``) is digit grouping with
    underscores (``1_000``), which no CSV writer emits.
    """
    with path.open(newline='') as fh:
        header = next(csv.reader(fh), [])
    col = {name: i for i, name in enumerate(header)}
    missing = [n for n in spec if n not in col and n not in optional]
    if missing:
        raise KeyError(missing[0])
    names = sorted((n for n in spec if n in col), key=col.__getitem__)
    if not names:
        return {}
    dt = np.dtype([(n, 'U64' if spec[n] == 'U' else spec[n]) for n in names])
    with warnings.catch_warnings():         # a header-only file is an empty table, not a warning
        warnings.filterwarnings('ignore', 'loadtxt: input contained no data', UserWarning)
        rows = np.loadtxt(path, delimiter=',', skiprows=1, usecols=[col[n] for n in names], dtype=dt,
                          comments=None, quotechar='"', encoding='utf-8', ndmin=1)
    return {n: np.ascontiguousarray(rows[n]) for n in names}


_FRAME_COLS = {'frame': 'i8', 'time_s': 'f8', 'has_star': 'U', 'crop_left': 'f8', 'crop_top': 'f8',
               'crop_side': 'f8', 'body_cx_model': 'f8', 'body_cy_model': 'f8', 'support_area_model_px': 'i8',
               'runtime_body_cx_model': 'f8', 'runtime_body_cy_model': 'f8', 'runtime_body_source': 'i8'}
_ARM_COLS = {'frame': 'i8', 'row_id': 'i8', 'arm_index': 'i8', 'track_id': 'i8', 'tip_score': 'f8',
             'base_score': 'f8', 'state_confidence': 'f8', 'steps': 'i8', 'stop_reason': 'U',
             **{f'identity_{i}': 'f8' for i in range(8)}}


def from_flipping_clip(folder: str | Path, frame_range: tuple[int, int] | None = None,
                       body_origin: str = 'runtime') -> PolylineContainer:
    """Convert one released flipping clip folder (``.../clips/<session>/<animal>/<stem>``).

    .. warning:: The clip's ``_polylines.npz`` is a **pickled** object array and is read with
       ``allow_pickle=True``. Only call this on the released re-trace you produced yourself; never on
       files from an untrusted source. The container written from it is pickle-free.

    ``frame_range`` (inclusive frame indices) cuts an excerpt; ``body_origin='runtime'`` uses the
    runtime body origin (``runtime_body_c{x,y}_model``, the released trace set), ``'support'`` the
    support > 0.5 centroid. ``track_id`` is the stored association (equal to a full-clip re-link).

    The CSVs are parsed column-wise in C (:func:`_read_columns`); every value equals ``float()`` /
    ``int()`` of its text, as the figure scripts parse it.
    """
    frame_path, arm_path, poly_path = flipping_paths(folder)
    fr = _read_columns(frame_path, _FRAME_COLS, optional=('runtime_body_cx_model', 'runtime_body_cy_model',
                                                          'runtime_body_source'))
    ar = _read_columns(arm_path, _ARM_COLS)
    if body_origin == 'runtime':
        for k in ('runtime_body_cx_model', 'runtime_body_cy_model'):
            if k not in fr:
                raise KeyError(k)
    if frame_range is not None:
        lo, hi = frame_range
        keep = (fr['frame'] >= lo) & (fr['frame'] <= hi)
        fr = {k: v[keep] for k, v in fr.items()}
        keep = (ar['frame'] >= lo) & (ar['frame'] <= hi)
        ar = {k: v[keep] for k, v in ar.items()}
    order = np.lexsort((ar['arm_index'], ar['frame']))       # stable, = sort by (frame, arm_index)
    ar = {k: v[order] for k, v in ar.items()}
    with np.load(poly_path, allow_pickle=True) as z:      # internal input only (see warning)
        row_ids = np.asarray(z['row_id']).astype(np.int64)
        polylines = z['polylines']
    # row_id -> polyline, the last one winning on a duplicate row_id (as a dict comprehension)
    uniq, last_rev = np.unique(row_ids[::-1], return_index=True)
    src = len(row_ids) - 1 - last_rev
    want = ar['row_id']
    at = np.searchsorted(uniq, want)
    ok = at < len(uniq)
    ok[ok] = uniq[at[ok]] == want[ok]
    if not ok.all():
        raise KeyError(int(want[~ok][0]))
    P = [np.asarray(polylines[i], np.float32) for i in src[at].tolist()]
    fidx = fr['frame'].astype(np.int64)
    arm_frames = ar['frame']
    fsort = np.argsort(fidx, kind='stable')
    # frame -> position, the last row winning on a duplicate frame (as the dict it replaces)
    fs = fidx[fsort]
    lastpos = np.r_[fs[1:] != fs[:-1], True] if len(fs) else np.zeros(0, bool)
    fu, fpos = fs[lastpos], fsort[lastpos]
    k = np.searchsorted(fu, arm_frames)
    ok = k < len(fu)
    ok[ok] = fu[k[ok]] == arm_frames[ok]
    if not ok.all():
        raise KeyError(int(arm_frames[~ok][0]))
    counts = np.bincount(fpos[k], minlength=len(fidx))
    ox, oy = (('runtime_body_cx_model', 'runtime_body_cy_model') if body_origin == 'runtime'
              else ('body_cx_model', 'body_cy_model'))
    c = PolylineContainer(
        points=(np.concatenate(P) if P else np.zeros((0, 2))).astype(np.float32),
        arm_point_offsets=_offsets(np.array([len(p) for p in P], np.int64)),
        frame_arm_offsets=_offsets(counts.astype(np.int64)),
        frame_index=fidx,
        time_s=fr['time_s'],
        crop=np.column_stack([fr['crop_left'], fr['crop_top'], fr['crop_side']]),
        body_origin_model=np.column_stack([fr[ox], fr[oy]]),
        body_origin_code=(fr['runtime_body_source'].astype(np.int8)
                          if body_origin == 'runtime' and len(fidx) and 'runtime_body_source' in fr else None),
        support_area_model_px=fr['support_area_model_px'].astype(np.int32),
        support_centroid_model=np.column_stack([fr['body_cx_model'], fr['body_cy_model']]),
        has_star=fr['has_star'] == '1',
        identity=np.column_stack([ar[f'identity_{i}'] for i in range(8)]).astype(np.float32).reshape(-1, 8),
        track_id=ar['track_id'].astype(np.int64),
        tip_score=ar['tip_score'].astype(np.float32),
        base_score=ar['base_score'].astype(np.float32),
        state_confidence=ar['state_confidence'].astype(np.float32),
        steps=ar['steps'].astype(np.int16),
        stop_reason=_stop_codes(ar['stop_reason']),
        meta={'fps': FLIPPING_FPS, 'model_size': 512, 'trace_contract': TRACE_CONTRACT,
              'tracker_config': TRACKER_CONFIG, 'body_origin': body_origin, 'source_format': 'flipping'},
    )
    return c.validate()



# ------------------------------------------------------------------------------------------ sleepy
_SLEEPY_KEYS = ('timestamp', 'source_frame', 'row', 'body_center_runtime', 'centroid', 'point_offsets', 'points',
                'arm_offsets', 'crop', 'body_center_source', 'support_area', 'identity', 'arm_tip_score',
                'arm_base_score', 'arm_state_confidence', 'arm_steps', 'arm_stop_reason')


def from_sleepy_segment(path: str | Path, body_origin: str = 'runtime') -> PolylineContainer:
    """Convert one Sleepy Stars re-trace segment (``sleepy_q_to_sa/seg_*.npz``; pickle-free).

    Only traced frames are stored (``has_star`` is all true); there is no ``track_id`` (run
    :func:`noodle_polylines.tracking.relink`). ``time_s`` is unix time; ``frame_index`` is the source
    video frame. Pass-through scores are cast to float32.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:     # decompress only the members the container uses
        d = {k: z[k] for k in _SLEEPY_KEYS if k in z.files}
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
        stop_reason=_stop_codes(d['arm_stop_reason']).reshape(A) if 'arm_stop_reason' in d else None,
        meta=meta,
    )
    return c.validate()


# ------------------------------------------------------------------------------------------ inference
_COMPRESSED = ('.gz', '.bz2', '.xz', '.zst')


def _read_text(path: Path) -> str:
    """The file's text, decompressing ``.gz`` / ``.bz2`` / ``.xz`` (stdlib) and ``.zst`` (the backcast
    runner's ``*.jsonl.zst``; needs Python >= 3.14 or the optional ``zstandard`` package)."""
    suffix = path.suffix.lower()
    if suffix not in _COMPRESSED:
        return path.read_text()
    raw = path.read_bytes()
    if suffix == '.gz':
        import gzip
        data = gzip.decompress(raw)
    elif suffix == '.bz2':
        import bz2
        data = bz2.decompress(raw)
    elif suffix == '.xz':
        import lzma
        data = lzma.decompress(raw)
    else:
        data = _zstd_decompress(raw, path)
    return data.decode('utf-8')


def _zstd_decompress(raw: bytes, path: Path) -> bytes:
    try:
        from compression import zstd           # Python >= 3.14
        return zstd.decompress(raw)
    except ImportError:
        pass
    try:
        import zstandard
    except ImportError:
        raise ImportError(f'{path.name}: reading .zst needs Python >= 3.14 or the zstandard package '
                          '(pip install zstandard); or decompress it first (zstd -d)') from None
    with zstandard.ZstdDecompressor().stream_reader(raw) as fh:   # frames without a content size too
        return fh.read()


def _inner_suffix(path: Path) -> str:
    """``.jsonl`` for ``x.jsonl`` and ``x.jsonl.zst``."""
    s = path.suffix.lower()
    return Path(path.stem).suffix.lower() if s in _COMPRESSED else s


# Arrays of numbers the adapter never reads (source-pixel copies of the polyline and the star's mask
# polygon): about half the numbers of a backcast record. They are cut from the text before json.loads.
# The value group runs greedily over number / bracket / comma / whitespace characters and backs off to
# the last "]]" before the first other character.
_UNUSED_ARRAY = re.compile(r'"(?:polyline_source|mask_polygon_source)"\s*:\s*(\[[0-9.eE+\-,\[\]\s]*\]\s*\])')


def _drop_unused_arrays(text: str) -> str:
    """Replace the values of ``"polyline_source"`` / ``"mask_polygon_source"`` with ``null``.

    In a valid JSON document the pattern can only match a real key (a ``"`` inside a string is escaped)
    and the matched span is exactly the key's array value when that value ends in ``]]`` (an array of
    points): the greedy run covers the value and then at most a comma and whitespace before the next
    key's ``"`` or the object's ``}``, and backs off to the value's own closing brackets. The key must
    also follow ``{`` or ``,``. Values of any other form (``null``, ``NaN``, flat arrays) are left to
    ``json.loads``, and every other value parses exactly as before. Malformed text inside a cut value
    is not reported (it is never parsed).
    """
    out: list[str] = []
    pos = 0
    for m in _UNUSED_ARRAY.finditer(text):
        k = m.start() - 1
        prev = text[k] if k >= 0 else ''
        if prev.isspace():
            prev = text[max(0, k - 4096):k].rstrip()[-1:]
        if not prev or prev not in '{,':
            continue
        v0, v1 = m.span(1)
        out.append(text[pos:v0])
        out.append('null')
        pos = v1
    if not out:
        return text
    out.append(text[pos:])
    return ''.join(out)


def _frame_records(path: Path, skip_unused: bool = True) -> Iterator[dict]:
    text = _read_text(path)
    if skip_unused:
        text = _drop_unused_arrays(text)
    payload = None
    if _inner_suffix(path) != '.jsonl':
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


def _stack_polylines(polys: list) -> tuple[np.ndarray, np.ndarray]:
    """All arms' ``polyline_crop`` lists as ((P, 2) float32 points, (A,) int64 point counts).

    Fast path, when every polyline is a list of [x, y] lists: one ``np.fromiter`` over the numbers
    (float64, then cast to float32), which converts every number exactly as the per-arm
    ``np.asarray(p, np.float32)`` does (one rounding to float32 of the number's double value).
    Otherwise the per-arm ``np.asarray(p, np.float32).reshape(-1, 2)`` (and its errors)."""
    if all(type(p) is list for p in polys):
        pairs = list(chain.from_iterable(polys))
        if not pairs or (set(map(type, pairs)) == {list} and set(map(len, pairs)) == {2}):
            try:
                flat = np.fromiter(chain.from_iterable(pairs), np.float64, 2 * len(pairs))
            except (ValueError, TypeError):
                flat = None
            if flat is not None:
                counts = np.fromiter(map(len, polys), np.int64, len(polys))
                return flat.astype(np.float32).reshape(-1, 2), counts
    arrays = [np.asarray(p, np.float32).reshape(-1, 2) for p in polys]
    points = (np.concatenate(arrays) if arrays else np.zeros((0, 2))).astype(np.float32)
    return points, np.array([len(p) for p in arrays], np.int64)


def _stack_identity(idents: list) -> np.ndarray:
    try:
        out = np.array(idents, np.float32)
        if out.shape == (len(idents), 8):
            return out
    except (ValueError, TypeError):
        pass
    return np.asarray([np.asarray(i, np.float32).reshape(8) for i in idents], np.float32).reshape(-1, 8)


def from_inference_json(path: str | Path, fps: float | None = None, skip_unused: bool = True
                        ) -> dict[int | None, PolylineContainer]:
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

    Reads ``.json``, ``.jsonl`` and their ``.gz`` / ``.bz2`` / ``.xz`` / ``.zst`` compressions. With
    ``skip_unused`` (default) the unused ``polyline_source`` and ``mask_polygon_source`` arrays are cut
    from the text before parsing (:func:`_drop_unused_arrays`; about 40% faster on backcast records).
    The container is identical either way; the only difference is that a malformed *number inside a
    skipped array* is not reported. Pass ``skip_unused=False`` to parse every value.
    """
    path = Path(path)
    stars: dict[int | None, dict[str, list]] = {}
    for rec in _frame_records(path, skip_unused):
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
            s = stars.get(star)
            if s is None:
                s = stars[star] = {'frame': [], 'crop': [], 'has_star': [], 'polys': [], 'n': [],
                                   'identity': [], 'track_id': [], 'identity_missing': False}
            s['frame'].append(int(rec['frame']))
            s['has_star'].append(crop is not None)
            s['crop'].append([math.nan] * 3 if crop is None else [float(crop['left']), float(crop['top']),
                                                                   float(crop['side'])])
            s['n'].append(len(arms))
            polys, idents, tids = s['polys'], s['identity'], s['track_id']
            for a in arms:
                if 'polyline_crop' not in a:
                    raise ValueError(f'{path.name}: frame {rec["frame"]}: arm record lacks polyline_crop')
                polys.append(a['polyline_crop'])
                ident = a.get('identity')
                if ident is None:
                    s['identity_missing'] = True
                    ident = _NAN8
                idents.append(ident)
                tid = a.get('track_id')
                tids.append(-1 if tid is None else int(tid))
    out = {}
    for star, s in stars.items():
        order = np.argsort(np.asarray(s['frame']), kind='stable')
        if not np.array_equal(order, np.arange(len(order))):
            raise ValueError(f'{path.name}: frames are not in increasing order')
        points, counts = _stack_polylines(s['polys'])
        identity = _stack_identity(s['identity'])      # also when missing: a malformed one still raises
        missing = ['body_origin_model', 'time_s', 'support_area_model_px']
        if s['identity_missing']:
            missing.append('identity')
        meta = {'model_size': 512, 'source_format': 'infer_video', 'star_track_id': star,
                'track_id_scope': 'star_track_id', 'missing_fields': missing}
        if fps is not None:
            meta['fps'] = float(fps)
        c = PolylineContainer(
            points=points,
            arm_point_offsets=_offsets(counts),
            frame_arm_offsets=_offsets(np.asarray(s['n'], np.int64)),
            frame_index=np.asarray(s['frame'], np.int64),
            crop=np.asarray(s['crop'], np.float64).reshape(-1, 3),
            has_star=np.asarray(s['has_star'], bool),
            identity=None if s['identity_missing'] else identity,
            track_id=np.asarray(s['track_id'], np.int64),
            meta=meta)
        out[star] = c.validate()
    return out


_NAN8 = [math.nan] * 8


CONTAINER_LOAD_KW = ('mmap', 'fields', 'validate', 'strict', 'sidecar', 'meta')


def load_any(path: str | Path, **kw) -> PolylineContainer:
    """Load a container ``.npz``, a Sleepy segment ``.npz``, an inference JSON or a flipping clip folder
    by sniffing the path. ``kw`` goes to the adapter; for a container ``.npz`` only the
    :meth:`PolylineContainer.load` keywords (``mmap``, ``fields``, ``validate``, ...) are used."""
    path = Path(path)
    if path.is_dir():
        return from_flipping_clip(path, **kw)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
        if 'arm_point_offsets' in files:
            return PolylineContainer.load(path, **{k: v for k, v in kw.items() if k in CONTAINER_LOAD_KW})
        if 'point_offsets' in files and 'timestamp' in files:
            return from_sleepy_segment(path, **kw)
    if _inner_suffix(path) in ('.json', '.jsonl'):
        stars = from_inference_json(path, **kw)
        if len(stars) != 1:
            raise ValueError(f'{path.name} holds {len(stars)} stars; use from_inference_json')
        return next(iter(stars.values()))
    raise ValueError(f'cannot tell the format of {path}')
