"""Regression tests for review findings: per-panel presets, Sleepy diagnostics defaults, the metadata
sidecar, missing identities, concat of separately tracked parts, empty sequences, the parity
comparator, the benchmark tiling and de-identification of the committed files."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from noodle_polylines import (FIG6_FLIPPING, FIG6C, ContainerError, PolylineContainer, adapters, compute_all, concat,
                              from_polylines)
from noodle_polylines import _reference as R
from noodle_polylines import geometry as G, kinematics as KI, scale as SC, tracking as TR

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'examples' / 'noodle_polylines' / 'data'
EXAMPLE = DATA / 'flip_example_01.npz'

pytestmark = pytest.mark.filterwarnings('ignore::RuntimeWarning')


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')


@pytest.fixture(scope='module')
def c():
    return PolylineContainer.load(EXAMPLE, sidecar=True)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------ presets
def test_fig6c_preset_has_float32_kymograph_angles(c):
    assert '6C' not in FIG6_FLIPPING.panels and FIG6C.panels == '6C'
    out = compute_all(c, 'fig6c')
    a32, _ = G.tip_polar(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.body_origin_model, 'float32', 'float32')
    assert out['tip_angle_deg'].dtype == np.float32
    same(out['tip_angle_deg'], a32)
    frames = R.stats_frames(c, frames=np.arange(c.n_frames), origin_float32=True)
    same(out['tip_angle_deg'], np.concatenate([R.tip_angle_fig6_copy(f) for f in frames]))
    # 6A/6D keep the float64 angles of stats.tip_polar (float32 origin)
    d = compute_all(c, 'fig6_flipping')
    assert d['tip_angle_deg'].dtype == np.float64
    same(d['tip_angle_deg'], np.concatenate([a for a, _ in R.tip_polar_stats(frames)]))
    same(d['track_id'], out['track_id'])


def test_fig4d_deformation_is_per_bout_window(c):
    """A gap > 1 s right before a window makes its first frame NaN in Fig 4D; compute_all must agree."""
    bouts = [{'start_s': b['upside_down_start_s'], 'stop_s': b['right_side_up_start_s']} for b in c.meta['bouts']]
    lo = bouts[0]['start_s'] - 2.0
    t = c.times()
    g = c.select_frames(np.flatnonzero(~((t > lo - 1.6) & (t < lo))))
    rows, arms, polys = R.flip_tables(g)
    ids = TR.relink(g)
    ref = R.fig4d_clip(rows, arms, polys, {a: int(i) for a, i in enumerate(ids)}, bouts)
    out = compute_all(g, 'fig4d', windows=[(b['start_s'], b['stop_s']) for b in bouts], track='relink')
    same(out['track_id'], ids)
    tg = g.times()
    assert out['log_deformation_window'].shape == (len(bouts), g.n_frames)
    for i, b in enumerate(bouts):
        win = (tg >= b['start_s'] - 2) & (tg <= b['stop_s'] + 2)
        sel = np.asarray(ref['columns']['bout_id']) == i
        same(out['log_deformation'][win], np.asarray(ref['columns']['log_deformation'])[sel])
        same(out['log_deformation_window'][i][win], np.asarray(ref['columns']['log_deformation'])[sel])
        assert np.all(np.isnan(out['log_deformation_window'][i][~win]))
    first = np.flatnonzero(tg >= lo)[0]
    assert np.isnan(out['log_deformation'][first])
    whole = KI.arm_deformation(g, float(out['r0']), ids, lag='prev_frame_reset')
    assert np.isfinite(whole[first])            # the old whole-sequence value the preset used to return


# ------------------------------------------------------------------------------------------ diagnostics
def _sleepy_like(c: PolylineContainer) -> PolylineContainer:
    """1-s frames with 3-8 s gaps, renumbered frame_index, Sleepy metadata (as concat + split_at_gaps give)."""
    keep = np.arange(0, c.n_frames, 2)
    keep = keep[~(((keep > 30) & (keep < 40)) | ((keep > 100) & (keep < 116)))]
    s = c.select_frames(keep)
    meta = {**s.meta, 'source_format': 'sleepy', 'fps': 1.0, 'source_frame_index_renumbered': True}
    return s.with_fields(frame_index=np.arange(s.n_frames, dtype=np.int64), track_id=None,
                         has_star=np.ones(s.n_frames, bool), meta=meta)


def test_sleepy_diagnostics_default_to_time_index(c):
    s = _sleepy_like(c)
    v = {'fused': TR.RELEASED}
    d = TR.diagnostics(s, variants=v, strides=())
    assert d['index_by'] == 'time' and d['origin_dtype'] == 'float64'
    t = s.times()
    idx = np.array([int(round(x - t[0])) for x in t])
    seq = {'name': 's', 'fps': 1.0, 'frames': R.stats_frames(s, origin_float32=False, index=idx)}
    ref = R.analyse(seq, variants={'fused': R.RELEASED}, strides=False)
    assert d['fused']['by_state'] == {k: {kk: int(vv) for kk, vv in x.items()} for k, x in ref['fused']['by_state'].items()}
    assert d['duration_s'] == ref['duration_s'] and d['fused']['n_ids'] == ref['fused']['n_ids']
    same(d['fused']['arm_lifetime_s'], np.array([x for x in ref['fused']['arm_lifetime_s']], np.float32))
    wrong = TR.diagnostics(s, variants=v, strides=(), by='frame_index')
    assert wrong['fused']['by_state']['all'] != d['fused']['by_state']['all']   # the gaps matter
    # a renumbered container of unknown format must say which index to use
    other = s.with_fields(meta={**s.meta, 'source_format': 'infer_video'})
    with pytest.raises(ContainerError):
        TR.diagnostics(other, variants=v, strides=())


def test_flipping_defaults_follow_the_figure_path(c):
    assert c.meta['source_format'] == 'flipping'
    assert TR.resolve_origin_dtype(c) == 'float32' and TR.resolve_index_by(c) == 'frame_index'
    tr = KI.tip_angle_track(c)
    a32, r32 = G.tip_polar(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.body_origin_model, 'float32', 'float32')
    same(tr['tip_angle_deg'], a32)
    plain = c.with_fields(meta={k: v for k, v in c.meta.items() if k != 'source_format'})
    assert KI.tip_angle_track(plain)['tip_angle_deg'].dtype == np.float64
    assert TR.resolve_origin_dtype(plain) == 'float64'


def test_diagnostics_empty_and_without_fps():
    empty = from_polylines([], meta={'fps': 2.0})
    empty = empty.with_fields(body_origin_model=np.zeros((0, 2)), identity=np.zeros((0, 8), np.float32))
    d = TR.diagnostics(empty)
    assert d['n_frames'] == 0 and d['fused']['by_state']['all']['arms'] == 0 and d['strides'][2]['agree'] == 0
    starless = from_polylines([[], []], meta={'fps': 2.0}, has_star=[False, False],
                              body_origin_model=np.zeros((2, 2)), identity=np.zeros((0, 8)))
    assert TR.diagnostics(starless)['n_arms'] == 0
    with pytest.raises(ContainerError):
        TR.diagnostics(starless.with_fields(meta={}))
    one = from_polylines([[np.array([[0., 0.], [4., 0.], [8., 0.]])]] * 3, meta={'fps': 1.0},
                         body_origin_model=np.zeros((3, 2)), identity=np.zeros((3, 8)))
    full = TR.diagnostics(one, variants={'fused': TR.RELEASED})
    pooled = TR.pooled([full, TR.diagnostics(starless, variants={'fused': TR.RELEASED})], 'fused', ())
    assert pooled == TR.pooled([full], 'fused', ())


# ------------------------------------------------------------------------------------------ container I/O
def test_sidecar_is_opt_in_and_rejects_payloads(c, tmp_path):
    plain = PolylineContainer.load(EXAMPLE)
    assert 'bouts' not in plain.meta and c.meta['bouts']
    payload = {'video': '/abs/path/source.mp4', 'frames': [{'frame': 0, 'detected': True,
                                                             'crop': {'left': 0, 'top': 0, 'side': 512}, 'arms': []}]}
    (tmp_path / 'clip.json').write_text(json.dumps(payload))
    out = adapters.from_inference_json(tmp_path / 'clip.json', fps=2.0)[None]
    back = PolylineContainer.load(out.save(tmp_path / 'clip.npz'))
    assert 'video' not in back.meta and 'frames' not in back.meta
    assert adapters.load_any(tmp_path / 'clip.npz').meta == back.meta
    with pytest.raises(ContainerError):
        PolylineContainer.load(tmp_path / 'clip.npz', sidecar=True)
    (tmp_path / 'clip.json').write_text(json.dumps([1, 2]))
    assert PolylineContainer.load(tmp_path / 'clip.npz').n_frames == 1
    with pytest.raises(ContainerError):
        PolylineContainer.load(tmp_path / 'clip.npz', sidecar=True)


def _arm(rng, ident=True, tid=None):
    a = {'polyline_crop': (256 + np.cumsum(rng.normal(size=(10, 2)) * 4, 0)).tolist()}
    if ident:
        a['identity'] = rng.normal(size=8).tolist()
    if tid is not None:
        a['track_id'] = tid
    return a


def test_inference_without_identity_and_null_star_id(tmp_path):
    rng = np.random.default_rng(0)
    crop = {'left': 10, 'top': 20, 'side': 300}
    recs = [{'frame': f, 'detected': True, 'crop': crop, 'arms': [_arm(rng, ident=(i != 1)) for i in range(4)]}
            for f in range(6)]
    (tmp_path / 'noident.json').write_text(json.dumps({'frames': recs}))
    s = adapters.from_inference_json(tmp_path / 'noident.json')[None]
    assert s.identity is None and 'identity' in s.meta['missing_fields']
    s = s.with_fields(body_origin_model=np.full((s.n_frames, 2), 256.0))
    with pytest.raises(ContainerError):
        TR.relink(s)
    out = compute_all(s.with_fields(time_s=np.arange(s.n_frames, dtype=float)), 'fig6_flipping')
    assert 'track_id' not in out and out['tip_angle_deg'].shape == (s.n_arms,)
    nan_ident = s.with_fields(identity=np.full((s.n_arms, 8), np.nan, np.float32))
    with pytest.raises(ContainerError):
        TR.relink(nan_ident)
    lines = [json.dumps({'frame': f, 'stars': [{'star_track_id': None, 'crop': crop, 'arms': [_arm(rng)]}]})
             for f in range(3)]
    (tmp_path / 'nullstar.jsonl').write_text('\n'.join(lines))
    with pytest.raises(ContainerError, match='frame 0'):
        adapters.from_inference_json(tmp_path / 'nullstar.jsonl')


def test_concat_of_separately_tracked_recordings(c):
    a = c.select_frames(slice(0, 90))
    b = c.select_frames(slice(90, None))
    a2 = a.with_fields(track_id=TR.relink(a))
    b2 = b.with_fields(track_id=TR.relink(b), frame_index=b.frame_index - b.frame_index[0], time_s=b.time_s + 1000)
    cc = concat([a2, b2])
    assert cc.meta['source_frame_index_renumbered'] and cc.meta['track_id_concat'] == 'offset'
    same(cc.source_frame_index, np.r_[a2.frame_index, b2.frame_index])
    same(cc.frame_index, np.arange(cc.n_frames))
    ta, tb = cc.track_id[:a2.n_arms], cc.track_id[a2.n_arms:]
    same(ta, a2.track_id)
    assert not set(ta.tolist()) & set(tb.tolist())
    same(tb, b2.track_id + a2.track_id.max() + 1)
    ids, source = TR.track_ids(cc)
    assert source == 'stored'
    join = a2.n_frames
    assert np.isnan(KI.arm_deformation(cc, 50.0)[join]) and np.isnan(KI.tip_motion(cc, 50.0)[join])
    # survives slicing, saving and loading
    sub = cc.select_frames(slice(80, 100))
    same(sub.source_frame_index, cc.source_frame_index[80:100])
    assert concat([a2, b2], track_ids='keep').track_id.tolist() == np.r_[a2.track_id, b2.track_id].tolist()
    assert concat([a2, b2], track_ids='drop').track_id is None
    # one tracker run split and re-joined keeps its IDs, with no renumbering
    rejoined = concat([c.select_frames(slice(0, 90)), c.select_frames(slice(90, None))])
    same(rejoined.track_id, c.track_id)
    assert rejoined.source_frame_index is None and rejoined.meta['track_id_concat'] == 'keep'


def test_source_frame_index_roundtrip(c, tmp_path):
    a, b = c.select_frames(slice(0, 50)), c.select_frames(slice(50, 100))
    cc = concat([a, b.with_fields(frame_index=b.frame_index - 1000)])
    back = PolylineContainer.load(cc.save(tmp_path / 'cc.npz'))
    same(back.source_frame_index, cc.source_frame_index)
    assert back.meta['source_frame_index_renumbered']


# ------------------------------------------------------------------------------------------ harness
def test_parity_comparator_rejects_infinities():
    pc = _load_module('parity_check_mod', ROOT / 'benchmarks' / 'noodle_polylines' / 'parity_check.py')
    chk = pc.Checks('x')
    chk.add('a', [np.inf, 1.0], [5.0, 1.0])
    chk.add('b', [np.inf], [-np.inf])
    chk.add('c', [np.inf, 2.0], [5.0, 2.0], 'resample')
    chk.add('d', [np.inf, np.nan, 1.0], [np.inf, np.nan, 1.0])
    assert [r['ok'] for r in chk.rows] == [False, False, False, True]


def test_bench_tiling_keeps_missing_ids(c):
    bench = _load_module('bench_mod', ROOT / 'benchmarks' / 'noodle_polylines' / 'bench.py')
    origin = c.body_origin_model.copy()
    origin[0] = np.nan                         # a re-link leaves the arms of this frame at -1
    src = c.with_fields(track_id=None, body_origin_model=origin)
    n0 = int(src.arms_per_frame()[0])
    assert n0 and np.all(TR.relink(src)[:n0] == -1)
    out = bench.synthetic([src], 3 * c.n_arms, seed=1)[0]
    reps = out.n_arms // c.n_arms
    assert reps == 3
    for k in range(reps):
        part = out.track_id[k * c.n_arms:(k + 1) * c.n_arms]
        assert np.all(part[:n0] == -1) and np.all(part[n0:] >= 0)


SOURCE_PATTERNS = (re.compile(r'R\d_W\d'), re.compile(r'\d{1,2}-\d\d_\d\d_\d\d-\d\d_\d\d_\d\d'),
                   re.compile(r'/(?:home|media)/'))


def test_committed_files_do_not_name_the_example_source():
    files = subprocess.run(['git', 'ls-files', 'CORE/noodle_polylines', 'benchmarks/noodle_polylines',
                            'examples/noodle_polylines', 'docs/noodle-polylines.md', 'tests/fixtures/noodle_polylines',
                            'tests/test_noodle_polylines*.py'], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if not files:
        pytest.skip('not a git checkout')
    for f in files + ['tests/test_noodle_polylines_regressions.py']:
        p = ROOT / f
        if not p.exists() or p.suffix in ('.npz',):
            continue
        text = p.read_text(errors='ignore')
        for pat in SOURCE_PATTERNS:
            assert not pat.search(text), f'{f}: {pat.pattern}'
