# Vendored from DeepStarNoodles tests/test_noodle_polylines_container.py at f749a49; only imports and fixture paths
# changed (see UPSTREAM.json "noodle_polylines").
"""Container data model, pickle-free I/O, slicing, adapters and the torch-free import."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from deepstarnoodles_inference._vendor.noodle_polylines import (ContainerError, PolylineContainer, adapters, concat, from_arrays, from_polylines,
                              split_at_gaps)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'polylines'
DATA = FIXTURES


def test_imports_without_torch():
    # The vendored copy must stay torch-free and must not register a top-level noodle_polylines.
    import deepstarnoodles_inference
    parent = Path(deepstarnoodles_inference.__file__).resolve().parents[1]
    code = ('import importlib, sys; '
            'import deepstarnoodles_inference._vendor.noodle_polylines, '
            'deepstarnoodles_inference._vendor.noodle_polylines._reference; '
            'bad = [m for m in ("torch", "deepstarnoodles", "cv2", "pandas", "noodle_polylines") '
            'if m in sys.modules]; '
            'print(",".join(bad)); sys.exit(1 if bad else 0)')
    r = subprocess.run([sys.executable, '-s', '-c', code], cwd=parent, capture_output=True, text=True,
                       env={'PYTHONPATH': str(parent), 'PATH': '/usr/bin:/bin'})
    assert r.returncode == 0, r.stdout + r.stderr


def test_example_data_contract():
    npz, meta = DATA / 'flip_example_01.npz', DATA / 'flip_example_01.json'
    assert npz.stat().st_size < 500_000
    with np.load(npz, allow_pickle=False) as z:
        assert z['points'].dtype == np.float32 and not any(z[k].dtype == object for k in z.files)
    text = meta.read_text()
    for bad in ('/home', '/media', 'weertman', '.mp4', 'sha256'):
        assert bad not in text
    # no session code or source video stem (patterns, so this file does not name the source either)
    assert not re.search(r'R\d_W\d|\d{1,2}-\d\d_\d\d_\d\d-\d\d_\d\d_\d\d', text)
    m = json.loads(text)
    assert m['license'] == 'CC-BY-4.0' and m['fps'] == 2.0 and len(m['bouts']) == 2
    c = PolylineContainer.load(npz, sidecar=True)
    assert (c.n_frames, c.n_arms, c.n_points) == (181, 1226, 34144)
    assert c.fps == 2.0 and c.has_track_id()


def test_roundtrip_and_slicing(tmp_path):
    c = PolylineContainer.load(DATA / 'flip_example_01.npz', sidecar=True)
    p = c.save(tmp_path / 'x.npz')
    with np.load(p, allow_pickle=False) as z:
        assert 'meta_json' in z.files
    back = PolylineContainer.load(p)
    for k, v in c.arrays().items():
        assert np.array_equal(getattr(back, k), v, equal_nan=v.dtype.kind == 'f')
    assert back.meta['fps'] == 2.0
    w = c.time_window(330.0, 340.0)
    assert w.n_frames == 21 and np.all((w.time_s >= 330) & (w.time_s <= 340))
    a0 = c.frame_arm_offsets[np.searchsorted(c.time_s, 330.0)]
    assert np.array_equal(w.arm(0), c.arm(a0)) and np.array_equal(w.track_id, c.track_id[a0:a0 + w.n_arms])
    parts = [c.select_frames(slice(0, 90)), c.select_frames(slice(90, None))]
    joined = concat(parts)
    for k, v in c.arrays().items():
        assert np.array_equal(getattr(joined, k), v, equal_nan=v.dtype.kind == 'f'), k
    rev = concat(parts[::-1])             # sorted back into time order
    assert np.array_equal(rev.points, c.points)
    dup = concat([parts[0], parts[0]], dedupe_time_decimals=1)
    assert dup.n_frames == 90
    gaps = split_at_gaps(c.select_frames(np.r_[0:50, 80:181]), max_gap_s=5.0, min_frames=30)
    assert [g.n_frames for g in gaps] == [50, 101]


def test_validation_errors():
    good = dict(points=np.zeros((3, 2), np.float32), arm_point_offsets=np.array([0, 1, 3]),
                frame_arm_offsets=np.array([0, 2]))
    from_arrays(**good)
    with pytest.raises(ContainerError):
        from_arrays(**{**good, 'arm_point_offsets': np.array([0, 2, 2])})       # empty arm
    with pytest.raises(ContainerError):
        from_arrays(**{**good, 'frame_arm_offsets': np.array([0, 3])})          # past the end
    with pytest.raises(ContainerError):
        from_arrays(**good, frame_index=[5], time_s=[0.0, 1.0])
    with pytest.raises(ContainerError):
        from_arrays(points=np.full((1, 2), np.nan), arm_point_offsets=[0, 1], frame_arm_offsets=[0, 1])
    c = from_arrays(**good)
    with pytest.raises(ContainerError):
        c.require('body_origin_model')
    with pytest.raises(ContainerError):
        c.times()                                                                # no time_s, no fps
    assert np.allclose(from_arrays(**good, meta={'fps': 2.0}).times(), [0.0])


def test_sleepy_adapter(tmp_path):
    rng = np.random.default_rng(0)
    po = np.array([0, 5, 9, 15])
    np.savez(tmp_path / 'seg_test.npz', row=np.arange(2), timestamp=np.array([100.0, 101.0]),
             crop=np.array([[0, 0, 400.0], [1, 1, 410.0]]), support_area=np.array([900, 0]),
             centroid=np.array([[256, 256], [np.nan, np.nan]], float), arm_offsets=np.array([0, 2, 3]),
             point_offsets=po, points=rng.normal(256, 20, (15, 2)).astype(np.float32),
             identity=rng.normal(size=(3, 8)).astype(np.float32), source_frame=np.array([10, 11]),
             body_center_runtime=np.array([[250.0, 255.0], [251.0, 256.0]]), body_center_source=np.array([0, 1], np.int8),
             arm_stop_reason=np.array(['base', 'loop', 'weird']), arm_steps=np.array([3, 4, 5]),
             arm_tip_score=np.array([.9, .8, .7]))
    c = adapters.from_sleepy_segment(tmp_path / 'seg_test.npz')
    assert c.n_arms == 3 and list(c.frame_index) == [10, 11] and c.track_id is None
    assert list(c.stop_reason) == [0, 1, -1] and c.fps == 1.0
    assert np.array_equal(c.body_origin_model, [[250.0, 255.0], [251.0, 256.0]])
    assert adapters.load_any(tmp_path / 'seg_test.npz').n_arms == 3


def _infer_records():
    arm = lambda tid, off: {'track_id': tid, 'polyline_crop': [[10 + off, 10], [20 + off, 12], [30 + off, 15]],  # noqa: E731
                            'polyline_source': [[0, 0]] * 3, 'identity': [0.1] * 8}
    crop = {'left': 5.0, 'top': 6.0, 'side': 400.0, 'confidence': .9, 'bbox_xyxy': [0, 0, 1, 1]}
    single = [{'frame': 0, 'detected': True, 'crop': crop, 'arms': [arm(0, 0), arm(1, 5)]},
              {'frame': 1, 'detected': False, 'crop': None, 'arms': []},
              {'frame': 2, 'detected': True, 'crop': crop, 'arms': [{**arm(0, 1), 'track_id': None}]}]
    multi = [{'frame': 0, 'detected': True, 'stars': [{'star_track_id': 3, 'crop': crop, 'arms': [arm(0, 0)]},
                                                      {'star_track_id': 7, 'crop': crop, 'arms': [arm(0, 2), arm(1, 4)]}]},
             {'frame': 1, 'detected': True, 'stars': [{'star_track_id': 7, 'crop': crop, 'arms': [arm(1, 4)]}]}]
    return single, multi


def test_inference_adapter(tmp_path):
    single, multi = _infer_records()
    (tmp_path / 'single.json').write_text(json.dumps({'video': 'v.mp4', 'frames': single}))
    stars = adapters.from_inference_json(tmp_path / 'single.json', fps=2.0)
    c = stars[None]
    assert list(c.arms_per_frame()) == [2, 0, 1] and list(c.track_id) == [0, 1, -1]
    assert list(c.has_star) == [True, False, True] and c.body_origin_model is None and c.time_s is None
    assert np.allclose(c.times(), [0.0, 0.5, 1.0])
    (tmp_path / 'multi.json').write_text(json.dumps({'schema_version': 2, 'frames': multi}))
    stars = adapters.from_inference_json(tmp_path / 'multi.json')
    assert sorted(stars) == [3, 7] and stars[7].n_arms == 3 and list(stars[7].frame_index) == [0, 1]
    (tmp_path / 'multi.jsonl').write_text('\n'.join(json.dumps(r) for r in multi) + '\n')
    stars2 = adapters.from_inference_json(tmp_path / 'multi.jsonl')
    assert np.array_equal(stars2[7].points, stars[7].points)


def test_flipping_adapter(tmp_path):
    stem = 'clip-x1'
    folder = tmp_path / 'clips' / 'S' / 'x1' / stem
    folder.mkdir(parents=True)
    fields = ['frame', 'time_s', 'has_star', 'crop_left', 'crop_top', 'crop_side', 'body_cx_model', 'body_cy_model',
              'support_area_model_px', 'runtime_body_cx_model', 'runtime_body_cy_model', 'runtime_body_source']
    with (folder / f'{stem}_frames.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        w.writerow(dict(zip(fields, [0, 0.0, 1, 10.5, 20.25, 300.0, 250.1, 251.2, 4000, 252.0, 253.0, 0])))
        w.writerow(dict(zip(fields, [1, 0.5, 0, 0, 0, 0, 'nan', 'nan', 0, 'nan', 'nan', -1])))
        w.writerow(dict(zip(fields, [2, 1.0, 1, 11.5, 21.25, 301.0, 250.0, 251.0, 4100, 252.5, 253.5, 1])))
    afields = ['frame', 'row_id', 'arm_index', 'track_id', 'tip_score', 'base_score', 'state_confidence', 'steps',
               'stop_reason'] + [f'identity_{i}' for i in range(8)]
    with (folder / f'{stem}_arms.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=afields); w.writeheader()
        for row_id, (frame, ai) in enumerate([(2, 0), (0, 1), (0, 0)]):     # out of order on purpose
            w.writerow(dict(zip(afields, [frame, row_id, ai, row_id + 4, .5, .4, .3, 7, 'base'] + [0.1 * i for i in range(8)])))
    polys = np.empty(3, dtype=object)
    polys[:] = [np.array([[1, 2], [3, 4]], np.float32) * (i + 1) for i in range(3)]
    with (folder / f'{stem}_polylines.npz').open('wb') as fh:
        np.savez_compressed(fh, polylines=polys, row_id=np.arange(3))
    c = adapters.from_flipping_clip(folder)
    assert list(c.arms_per_frame()) == [2, 0, 1]
    assert list(c.track_id) == [6, 5, 4]                        # sorted by (frame, arm_index)
    assert np.array_equal(c.arm(0), polys[2]) and c.has_star.tolist() == [True, False, True]
    assert c.body_origin_model[0].tolist() == [252.0, 253.0] and c.fps == 2.0
    sub = adapters.from_flipping_clip(folder, frame_range=(1, 2))
    assert sub.n_frames == 2 and sub.n_arms == 1
    with pytest.raises(ValueError):
        np.load(folder / f'{stem}_polylines.npz', allow_pickle=False)['polylines']   # the source really is pickled


def test_from_polylines():
    c = from_polylines([[np.zeros((3, 2)), np.ones((1, 2))], []])
    assert c.n_frames == 2 and c.n_arms == 2 and list(c.points_per_arm()) == [3, 1]
