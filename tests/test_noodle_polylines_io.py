"""Round-5 I/O: column-wise CSV adapter, JSON skipping, mapped container loads, pooled streaming.

Every fast path is compared with the per-cell implementation it replaced (kept below as ``_old_*``),
field by field and bit for bit, including dtypes and meta.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

import noodle_polylines as NP
from noodle_polylines import ContainerError, PolylineContainer, adapters
from noodle_polylines.adapters import (FLIPPING_FPS, STOP_REASON_CODES, TRACE_CONTRACT, TRACKER_CONFIG,
                                       _drop_unused_arrays, _read_columns)
from noodle_polylines.container import ARM_FIELDS, FRAME_FIELDS, _offsets

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples' / 'noodle_polylines' / 'data' / 'flip_example_01.npz'


def same_container(a: PolylineContainer, b: PolylineContainer) -> None:
    for n in ('points', 'arm_point_offsets', 'frame_arm_offsets', *FRAME_FIELDS, *ARM_FIELDS):
        x, y = getattr(a, n), getattr(b, n)
        if x is None or y is None:
            assert x is None and y is None, n
            continue
        assert x.dtype == y.dtype and x.shape == y.shape, (n, x.dtype, y.dtype, x.shape, y.shape)
        assert x.tobytes() == y.tobytes(), n
    assert a.meta == b.meta


# ------------------------------------------------------------------------------------------ old oracles
def _old_stop_code(value) -> int:
    return STOP_REASON_CODES.get(str(value), -1)


def _old_from_flipping_clip(folder, frame_range=None, body_origin='runtime'):
    """The round-4 adapter: csv.DictReader and float() / int() per cell."""
    frame_path, arm_path, poly_path = adapters.flipping_paths(folder)
    with frame_path.open(newline='') as fh:
        frames = list(csv.DictReader(fh))
    with arm_path.open(newline='') as fh:
        arms = list(csv.DictReader(fh))
    if frame_range is not None:
        lo, hi = frame_range
        frames = [r for r in frames if lo <= int(r['frame']) <= hi]
        arms = [a for a in arms if lo <= int(a['frame']) <= hi]
    arms.sort(key=lambda a: (int(a['frame']), int(a['arm_index'])))
    with np.load(poly_path, allow_pickle=True) as z:
        poly = {int(r): np.asarray(p, np.float32) for r, p in zip(z['row_id'], z['polylines'])}
    fidx = np.array([int(r['frame']) for r in frames], np.int64)
    pos = {int(f): k for k, f in enumerate(fidx)}
    counts = np.bincount(np.array([pos[int(a['frame'])] for a in arms], np.int64), minlength=len(fidx))
    P = [poly[int(a['row_id'])] for a in arms]
    f64 = lambda rows, key: np.array([float(r[key]) for r in rows], np.float64)   # noqa: E731
    ox, oy = (('runtime_body_cx_model', 'runtime_body_cy_model') if body_origin == 'runtime'
              else ('body_cx_model', 'body_cy_model'))
    return PolylineContainer(
        points=(np.concatenate(P) if P else np.zeros((0, 2))).astype(np.float32),
        arm_point_offsets=_offsets(np.array([len(p) for p in P], np.int64)),
        frame_arm_offsets=_offsets(counts.astype(np.int64)),
        frame_index=fidx, time_s=f64(frames, 'time_s'),
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
        stop_reason=np.array([_old_stop_code(a['stop_reason']) for a in arms], np.int8),
        meta={'fps': FLIPPING_FPS, 'model_size': 512, 'trace_contract': TRACE_CONTRACT,
              'tracker_config': TRACKER_CONFIG, 'body_origin': body_origin, 'source_format': 'flipping'},
    ).validate()


def _old_frame_records(path: Path):
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
            if 'frames' in rec:
                yield from rec['frames']
            else:
                yield rec
        return
    yield from (payload['frames'] if isinstance(payload, dict) else payload)


def _old_from_inference_json(path, fps=None):
    """The round-4 adapter: one np.asarray per arm polyline and identity."""
    path = Path(path)
    stars = {}
    for rec in _old_frame_records(path):
        if 'stars' in rec:
            entries = []
            for s in rec['stars']:
                sid = s.get('star_track_id')
                if sid is None:
                    raise ContainerError('multi-star record without a star_track_id')
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
                    raise ValueError('arm record lacks polyline_crop')
                s['polys'].append(np.asarray(a['polyline_crop'], np.float32).reshape(-1, 2))
                ident = a.get('identity')
                s['identity_missing'] |= ident is None
                s['identity'].append(np.full(8, np.nan, np.float32) if ident is None
                                     else np.asarray(ident, np.float32).reshape(8))
                s['track_id'].append(int(a['track_id']) if a.get('track_id') is not None else -1)
    out = {}
    for star, s in stars.items():
        polys = s['polys']
        missing = ['body_origin_model', 'time_s', 'support_area_model_px']
        if s['identity_missing']:
            missing.append('identity')
        meta = {'model_size': 512, 'source_format': 'infer_video', 'star_track_id': star,
                'track_id_scope': 'star_track_id', 'missing_fields': missing}
        if fps is not None:
            meta['fps'] = float(fps)
        out[star] = PolylineContainer(
            points=(np.concatenate(polys) if polys else np.zeros((0, 2))).astype(np.float32),
            arm_point_offsets=_offsets(np.array([len(p) for p in polys], np.int64)),
            frame_arm_offsets=_offsets(np.asarray(s['n'], np.int64)),
            frame_index=np.asarray(s['frame'], np.int64),
            crop=np.asarray(s['crop'], np.float64).reshape(-1, 3),
            has_star=np.asarray(s['has_star'], bool),
            identity=None if s['identity_missing'] else np.asarray(s['identity'], np.float32).reshape(-1, 8),
            track_id=np.asarray(s['track_id'], np.int64), meta=meta).validate()
    return out


# ------------------------------------------------------------------------------------------ CSV
TRICKY = ['0', '-0.0', '0.1', '1e-310', '4.9e-324', '2.2250738585072014e-308', '1.7976931348623157e308',
          '123456789.123456789123456789', '-3.4028234663852886e+38', 'nan', 'inf', '-inf', '1E5', '.5', '5.',
          '0.30000000000000004', '9007199254740993', '1.00000005960464477539062500000001']


def test_read_columns_parses_like_float_and_int(tmp_path):
    path = tmp_path / 't.csv'
    with path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['label', 'x', 'n', 'note'])
        for i, v in enumerate(TRICKY):
            w.writerow([f'r#{i}', v, -i * 7, 'a,"quoted" # not a comment'])
    cols = _read_columns(path, {'x': 'f8', 'n': 'i8', 'note': 'U', 'label': 'U'})
    want = np.array([float(v) for v in TRICKY])
    assert cols['x'].tobytes() == want.tobytes()             # bit for bit, sign of zero and NaN included
    assert cols['n'].tolist() == [-i * 7 for i in range(len(TRICKY))]
    assert set(cols['note']) == {'a,"quoted" # not a comment'} and cols['label'][3] == 'r#3'
    with pytest.raises(KeyError):
        _read_columns(path, {'absent': 'f8'})
    assert 'absent' not in _read_columns(path, {'x': 'f8', 'absent': 'f8'}, optional=('absent',))
    dup = tmp_path / 'd.csv'
    dup.write_text('x,x\n1.5,2.5\n')
    assert _read_columns(dup, {'x': 'f8'})['x'].tolist() == [2.5] == [float(r['x']) for r in csv.DictReader(dup.open())]
    head = tmp_path / 'h.csv'
    head.write_text('x,n\r\n')
    empty = _read_columns(head, {'x': 'f8', 'n': 'i8'})
    assert empty['x'].shape == (0,) and empty['n'].dtype == np.int64


def _write_clip(folder: Path, frames: list[list], arms: list[list], polys: list[np.ndarray], row_ids,
                frame_fields=None) -> Path:
    stem = folder.name
    folder.mkdir(parents=True)
    ffields = frame_fields or ['frame', 'time_s', 'has_star', 'crop_left', 'crop_top', 'crop_side', 'body_cx_model',
                               'body_cy_model', 'support_area_model_px', 'runtime_body_cx_model',
                               'runtime_body_cy_model', 'runtime_body_source']
    with (folder / f'{stem}_frames.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['clip_id', *ffields])
        for r in frames:
            w.writerow(['S/x/' + stem, *r[:len(ffields)]])
    afields = ['frame', 'row_id', 'arm_index', 'track_id', 'tip_score', 'base_score', 'state_confidence', 'steps',
               'stop_reason'] + [f'identity_{i}' for i in range(8)]
    with (folder / f'{stem}_arms.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['clip_id', 'condition', *afields, 'tip_model_x'])
        for r in arms:
            w.writerow(['S/x/' + stem, 'R1', *r, 1.0])
    obj = np.empty(len(polys), dtype=object)
    obj[:] = polys
    with (folder / f'{stem}_polylines.npz').open('wb') as fh:
        np.savez_compressed(fh, polylines=obj, row_id=np.asarray(row_ids))
    return folder


@pytest.mark.filterwarnings('ignore:overflow encountered in cast')   # a TRICKY score exceeds float32
def test_flipping_adapter_matches_the_per_cell_reader(tmp_path):
    rng = np.random.default_rng(5)
    frames = []
    for f in range(6):
        frames.append([f, f * 0.5, int(f != 3), rng.random() * 100, rng.random() * 100, 300 + f, 250 + rng.random(),
                       'nan' if f == 3 else 251 + rng.random(), 4000 + f, repr(252 + rng.random()), 253.25, f % 3 - 1])
    arms, polys, row_ids = [], [], []
    stops = ['base', 'loop', 'lost', 'weird', 'max_steps']
    for k, (frame, ai) in enumerate([(4, 1), (0, 0), (4, 0), (0, 1), (5, 0), (2, 0), (4, 2), (2, 1), (2, 1)]):
        arms.append([frame, 100 + k, ai, k % 4, rng.random(), TRICKY[k], rng.random(), 7 + k, stops[k % 5]]
                    + [repr(float(v)) for v in rng.normal(size=8)])
        polys.append((rng.normal(250, 30, (3 + k, 2))).astype(np.float32 if k % 2 else np.float64))
        row_ids.append(100 + k)
    polys.append(np.zeros((2, 2), np.float32))        # duplicate row_id: the last one wins, as in a dict
    row_ids.append(103)
    folder = _write_clip(tmp_path / 'clips' / 'S' / 'x' / 'clip-a', frames, arms, polys, row_ids)
    for kw in ({}, {'body_origin': 'support'}, {'frame_range': (2, 4)}, {'frame_range': (9, 12)}):
        same_container(adapters.from_flipping_clip(folder, **kw), _old_from_flipping_clip(folder, **kw))
    # without the runtime body columns: the support origin still works, the runtime one raises KeyError
    short = _write_clip(tmp_path / 'clips' / 'S' / 'x' / 'clip-b', frames, arms, polys, row_ids,
                        frame_fields=['frame', 'time_s', 'has_star', 'crop_left', 'crop_top', 'crop_side',
                                      'body_cx_model', 'body_cy_model', 'support_area_model_px'])
    same_container(adapters.from_flipping_clip(short, body_origin='support'),
                   _old_from_flipping_clip(short, body_origin='support'))
    with pytest.raises(KeyError):
        adapters.from_flipping_clip(short)
    with pytest.raises(KeyError):
        _old_from_flipping_clip(short)


def test_flipping_adapter_unknown_rows_raise(tmp_path):
    frames = [[0, 0.0, 1, 1, 1, 300, 250, 250, 10, 250, 250, 0]]
    arms = [[1, 7, 0, 0, .5, .5, .5, 3, 'base'] + [0.0] * 8]          # frame 1 is not in the frames table
    folder = _write_clip(tmp_path / 'c' / 'S' / 'x' / 'clip-c', frames, arms, [np.zeros((2, 2), np.float32)], [7])
    with pytest.raises(KeyError):
        adapters.from_flipping_clip(folder)
    arms[0][0] = 0
    arms[0][1] = 8                                                     # row 8 has no polyline
    folder = _write_clip(tmp_path / 'd' / 'S' / 'x' / 'clip-d', frames, arms, [np.zeros((2, 2), np.float32)], [7])
    with pytest.raises(KeyError):
        adapters.from_flipping_clip(folder)


def test_stop_codes_vectorised():
    v = np.array(['base', 'loop', 'zzz', 'stop', 'base', 'max_path', ''])
    assert adapters._stop_codes(v).tolist() == [adapters._stop_code(x) for x in v]
    b = np.array([b'base', b'loop'])
    assert adapters._stop_codes(b).tolist() == [adapters._stop_code(x) for x in b] == [-1, -1]
    assert adapters._stop_codes(np.array([], '<U4')).dtype == np.int8


# ------------------------------------------------------------------------------------------ JSON
def _records(c: PolylineContainer, odd: bool = False) -> list[dict]:
    """Backcast-like multi-star records from a container; ``odd`` adds the shapes the skipper must leave."""
    recs = []
    for f in range(c.n_frames):
        arms = []
        for a in c.frame_arms(f):
            p = c.arm(a)
            src = (p * 0.8 + 100).astype(np.float32).tolist()
            if odd and a % 7 == 1:
                src = None
            elif odd and a % 7 == 2:
                src = [1.0, 2.0]                               # flat: left to json
            elif odd and a % 7 == 3:
                src = [[[1.0, 2.0]], [[3.0, 4.0]]]            # deeper: cut as one value
            elif odd and a % 7 == 4:
                src = []
            arms.append({'polyline_crop': p.tolist(), 'polyline_source': src,
                         'identity': c.identity[a].tolist(), 'track_id': int(c.track_id[a]),
                         'note': 'text with "polyline_source": [[1, 2]] inside' if odd else 'x'})
        stars = [{'star_track_id': 3, 'crop': {'left': float(c.crop[f, 0]), 'top': float(c.crop[f, 1]),
                                               'side': float(c.crop[f, 2])},
                  'mask_polygon_source': (None if odd and f % 5 == 0 else [[1.5, 2.5], [3.5, 4.5 + f]]),
                  'arms': arms}]
        if odd and f % 4 == 0:
            stars.append({'star_track_id': 9, 'crop': {'left': 1.0, 'top': 2.0, 'side': 3.0}, 'mask_polygon_source': [[float('nan'), 1.0]],
                          'arms': arms[:1]})
        recs.append({'frame': int(c.frame_index[f]), 'stars': stars})
    return recs


@pytest.fixture(scope='module')
def micro():
    return PolylineContainer.load(EXAMPLE, sidecar=True)


@pytest.mark.parametrize('odd', [False, True])
def test_inference_adapter_matches_the_per_arm_reader(tmp_path, micro, odd):
    c = micro.select_frames(slice(0, 60))
    recs = _records(c, odd)
    files = {'compact.jsonl': '\n'.join(json.dumps(r, separators=(',', ':')) for r in recs) + '\n',
             'indented.json': json.dumps({'schema_version': 2, 'frames': recs}, indent=2),
             'spaced.jsonl': '\n'.join(json.dumps(r) for r in recs) + '\n'}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
        want = _old_from_inference_json(tmp_path / name, fps=2.0)
        for skip in (True, False):
            got = adapters.from_inference_json(tmp_path / name, fps=2.0, skip_unused=skip)
            assert got.keys() == want.keys()
            for k in want:
                same_container(got[k], want[k])
    (tmp_path / 'compact.jsonl.gz').write_bytes(gzip.compress(files['compact.jsonl'].encode()))
    got = adapters.from_inference_json(tmp_path / 'compact.jsonl.gz', fps=2.0)
    for k, v in _old_from_inference_json(tmp_path / 'compact.jsonl', fps=2.0).items():
        same_container(got[k], v)
    if not odd:                                                   # one star: load_any can sniff it
        same_container(adapters.load_any(tmp_path / 'compact.jsonl.gz'), _old_from_inference_json(tmp_path / 'compact.jsonl')[3])


def test_drop_unused_arrays_cuts_only_whole_numeric_values():
    rec = {'a': 'he said "polyline_source": [[1, 2]] here', 'polyline_source': [[1.5, -2e-3], [3, 4]],
           'arms': [{'polyline_source': [], 'mask_polygon_source': [[1, 2]], 'polyline_crop': [[5, 6]]}],
           'mask_polygon_source': [[float('nan'), 1]], 'k': [[1, 2]]}
    for text in (json.dumps(rec), json.dumps(rec, indent=3), json.dumps(rec, separators=(',', ':'))):
        got = json.loads(_drop_unused_arrays(text))
        want = json.loads(text)
        want['polyline_source'] = None
        want['arms'][0]['mask_polygon_source'] = None
        assert got['a'] == want['a'] and got['k'] == want['k'] and got['arms'][0]['polyline_crop'] == [[5, 6]]
        assert got['arms'][0]['polyline_source'] == []                 # flat/empty: left to json
        assert math.isnan(got['mask_polygon_source'][0][0])            # NaN: left to json
        assert got['polyline_source'] is None and got['arms'][0]['mask_polygon_source'] is None
    assert _drop_unused_arrays('{"x": 1}') == '{"x": 1}'


@pytest.mark.parametrize('value', [
    [[[1, 2], [3, 4]], [[None, 5]]], [[[1, 2]], [[True, 5]]], [[[1, 2]], {'x': 1}],
    [[[1, 2]], [[float('nan'), 3]]], [[1, 2], [None, 3]], [[1, 2], 'x]]', [[3, 4]]],
    [[[1, 2]], 3], [[1, 2], [float('inf'), 3]], [[1, 2], {'k': [[1, 2]]}], [], [1, 2], None])
def test_drop_unused_arrays_leaves_other_values_to_json(tmp_path, value):
    """A value holding a non-number (also after an inner ']]') or not ending in ']]' is left to json."""
    rec = {'frame': 0, 'detected': True, 'crop': {'left': 0, 'top': 0, 'side': 9},
           'arms': [{'polyline_crop': [[1, 2], [3, 4]], 'identity': [0.5] * 8, 'polyline_source': value}],
           'mask_polygon_source': value}
    for kw in ({}, {'indent': 2}, {'separators': (',', ':')}):
        text = json.dumps(rec, **kw)
        assert _drop_unused_arrays(text) == text
        path = tmp_path / 'x.json'
        path.write_text(json.dumps({'frames': [rec]}, **kw))
        got, want = (adapters.from_inference_json(path, skip_unused=s) for s in (True, False))
        same_container(got[None], want[None])
    deep = {'polyline_source': [[[1, 2]], [[3, 4]]], 'k"\\:': [[[1]]]}           # numbers only: cut
    for kw in ({}, {'indent': 2}):
        assert json.loads(_drop_unused_arrays(json.dumps(deep, **kw))) == {**deep, 'polyline_source': None}


def test_inference_polyline_fallbacks_match(tmp_path):
    """Shapes the batched conversion hands to the per-arm one: flat lists, deeper nests, None values."""
    cases = [
        [[[1.0, 2.0], [3.0, 4.0]], [[5, 6]]],                        # ints and floats
        [[1.0, 2.0, 3.0, 4.0], [[5.0, 6.0]]],                         # a flat polyline
        [[[[1.0, 2.0]], [[3.0, 4.0]]]],                               # (n, 1, 2)
        [[[None, 2.0], [3.0, True]]],                                 # None -> NaN: both fail validation
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],                         # (2, 3) reshaped to (3, 2)
    ]
    outcomes = []
    for i, polys in enumerate(cases):
        recs = [{'frame': f, 'detected': True, 'crop': {'left': 0, 'top': 0, 'side': 400},
                 'arms': [{'polyline_crop': p, 'identity': [0.5] * 8}]} for f, p in enumerate(polys)]
        path = tmp_path / f'case{i}.json'
        path.write_text(json.dumps({'frames': recs}))
        try:
            want = _old_from_inference_json(path)
        except (ValueError, ContainerError) as e:
            with pytest.raises(type(e)):
                adapters.from_inference_json(path)
            outcomes.append('raised')
            continue
        got = adapters.from_inference_json(path)
        for k in want:
            same_container(got[k], want[k])
        outcomes.append('equal')
    assert outcomes == ['equal', 'equal', 'equal', 'raised', 'equal']
    bad = tmp_path / 'bad_ident.json'
    bad.write_text(json.dumps({'frames': [{'frame': 0, 'detected': True, 'crop': {'left': 0, 'top': 0, 'side': 9},
                                           'arms': [{'polyline_crop': [[1, 2]], 'identity': [1, 2]},
                                                    {'polyline_crop': [[1, 2]]}]}]}))
    with pytest.raises(ValueError):
        _old_from_inference_json(bad)
    with pytest.raises(ValueError):
        adapters.from_inference_json(bad)


def test_zst_without_a_decoder_says_how(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def no_zstd(name, *a, **k):
        if name in ('compression', 'zstandard') or name.startswith('compression.'):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', no_zstd)
    p = tmp_path / 'x.jsonl.zst'
    p.write_bytes(b'\x28\xb5\x2f\xfd')
    with pytest.raises(ImportError, match='zstandard'):
        adapters.from_inference_json(p)


# ------------------------------------------------------------------------------------------ container
def test_mapped_and_partial_loads(tmp_path, micro):
    raw = micro.save(tmp_path / 'raw.npz', compressed=False)
    packed = micro.save(tmp_path / 'packed.npz')
    ref = PolylineContainer.load(raw)
    for path in (raw, packed):
        m = PolylineContainer.load(path, mmap=True)
        same_container(m, ref)
        assert type(m.points) is np.ndarray
        assert (not m.points.flags.writeable) == (path == raw)      # compressed members are read as usual
    lazy = PolylineContainer.load(raw, mmap=True, validate=False)
    assert lazy.points.base is not None and np.array_equal(lazy.points, micro.points)
    part = PolylineContainer.load(raw, fields=('time_s', 'track_id'))
    assert part.identity is None and part.crop is None and np.array_equal(part.track_id, micro.track_id)
    assert np.array_equal(part.points, micro.points) and part.meta == ref.meta
    with pytest.raises(ContainerError):
        PolylineContainer.load(raw, fields=('not_a_field',))
    empty = NP.from_polylines([[], []], meta={'fps': 1.0})
    same_container(PolylineContainer.load(empty.save(tmp_path / 'e.npz', compressed=False), mmap=True), empty)
    out_a = NP.compute_all(PolylineContainer.load(raw, mmap=True), 'fig5', r0=50.0)
    out_b = NP.compute_all(micro, 'fig5', r0=50.0)
    for k, v in out_b.items():
        if isinstance(v, np.ndarray):
            assert v.tobytes() == out_a[k].tobytes(), k


def test_load_any_passes_container_options(tmp_path, micro):
    raw = micro.save(tmp_path / 'raw.npz', compressed=False)
    c = adapters.load_any(raw, mmap=True, fields=('track_id',), body_origin='ignored-here')
    assert not c.points.flags.writeable and c.identity is None


# ------------------------------------------------------------------------------------------ pipeline
def _seqs(micro, tmp_path, n=5):
    paths = []
    for i in range(n):
        part = micro.select_frames(slice(30 * i, 30 * i + 40 + 7 * i))
        paths.append(part.save(tmp_path / f'seq{i}.npz', compressed=bool(i % 2)))
    return paths


def test_compute_many_pool_equals_serial(tmp_path, micro):
    paths = _seqs(micro, tmp_path)
    kw = {'r0': 50.0, 'context': {'seq2': {'r0': 40.0}}}
    serial = list(NP.compute_many(paths, 'fig5', **kw))
    pooled = list(NP.compute_many(paths, 'fig5', workers=3, max_pending=2, mp_context='fork',
                                  load_kw={'mmap': True}, **kw))
    assert [n for n, _ in pooled] == [n for n, _ in serial] == [f'seq{i}' for i in range(5)]
    for (_, a), (_, b) in zip(serial, pooled):
        assert set(a) == set(b)
        for k, v in a.items():
            if isinstance(v, np.ndarray):
                assert v.dtype == b[k].dtype and v.tobytes() == b[k].tobytes(), k
    assert serial[2][1]['r0'] == 40.0 != serial[1][1]['r0']
    written = list(NP.compute_many(paths, 'fig5', workers=2, out_dir=tmp_path / 'out', mp_context='fork', **kw))
    for (name, path), (_, a) in zip(written, serial):
        with np.load(path, allow_pickle=False) as z:
            assert z['arm_deformation'].tobytes() == a['arm_deformation'].tobytes()


def test_compute_many_pool_reports_failures(tmp_path, micro):
    paths = _seqs(micro, tmp_path, 2) + [tmp_path / 'missing.npz']
    with pytest.raises(FileNotFoundError):
        list(NP.compute_many(paths, 'fig5', r0=50.0, workers=2, mp_context='fork'))
