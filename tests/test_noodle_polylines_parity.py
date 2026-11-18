"""Public noodle_polylines API == _reference (figure-script ports) == golden fixtures, on the micro example."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from noodle_polylines import PolylineContainer, compute_all
from noodle_polylines import _reference as R
from noodle_polylines import frames as FR, geometry as G, kinematics as KI, scale as SC, tracking as TR

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples' / 'noodle_polylines' / 'data' / 'flip_example_01.npz'
FIX = ROOT / 'tests' / 'fixtures' / 'noodle_polylines'
STATES = {0: 'Right Side Up', 1: 'Being flipped', 2: 'Upside Down'}


def same(a, b):
    """Bit-identical, NaN positions included."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'), np.nanmax(np.abs(a.astype(float) - b.astype(float)))


def close(a, b, rtol=1e-9):
    """The closed-form Kabsch default: NaN/inf positions exact, finite values within the gate's rtol 1e-9."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    fa, fb = np.isfinite(a), np.isfinite(b)
    assert np.array_equal(fa, fb) and np.array_equal(a[~fa], b[~fb], equal_nan=True)
    assert np.all(np.abs(a[fa] - b[fb]) <= rtol * np.abs(b[fb])), np.max(np.abs(a[fa] - b[fb]) / np.abs(b[fb]))


@pytest.fixture(scope='module')
def c():
    return PolylineContainer.load(EXAMPLE, sidecar=True)


@pytest.fixture(scope='module')
def fx():
    return (dict(np.load(FIX / 'micro_arms.npz')), dict(np.load(FIX / 'micro_frames.npz')),
            json.loads((FIX / 'micro_meta.json').read_text()))


@pytest.fixture(scope='module')
def tables(c):
    return R.flip_tables(c)


@pytest.fixture(scope='module')
def relink_ids(c):
    return TR.relink(c)


# ------------------------------------------------------------------------------------------ T0 / T1
def test_resample_and_metrics(c, fx):
    arms, _, _ = fx
    n = len(arms['T0_2_base'])
    sub = c.select_frames(np.arange(c.n_frames))      # identity selection keeps everything
    pts, ao = sub.points, sub.arm_point_offsets
    same(G.resample_base(pts, ao[:n + 1]), arms['T0_2_base'])
    tip, L = G.resample_tip(pts, ao[:n + 1])
    same(tip, arms['T0_2_tip_raw']); same(L, arms['T0_2_tip_raw_L'])
    tip, L = G.resample_tip(pts, ao[:n + 1], arm_origin=c.body_origin_model[c.arm_frame()[:n]])
    same(tip, arms['T0_2_tip_rel']); same(L, arms['T0_2_tip_rel_L'])
    for rule in ('zero', 'nan'):
        L, k = G.arc_length_curvature(pts, ao, rule)
        same(L, arms[f'T0_3_arc_length_{rule}']); same(k, arms[f'T1_1_curvature_{rule}'])
    same(G.tips_source(pts, ao, c.frame_arm_offsets, c.crop), arms['T0_4_tip_source'])
    rm, _ = FR.body_radius(c)
    same(G.norm_length(arms['T0_3_arc_length_zero'], c.frame_arm_offsets, rm), arms['T1_2_norm_length'])


def test_public_resample_equals_reference_per_arm(c):
    for a in range(0, c.n_arms, 37):
        p = c.arm(a)
        same(G.resample_base(p, np.array([0, len(p)]))[0], R.resample_arm(p))
        t, L = R.tip_anchored_resample(p, 12)
        t2, L2 = G.resample_tip(p, np.array([0, len(p)]))
        same(t2[0], t); assert L2[0] == L
        assert tuple(G.arc_length_curvature(p, np.array([0, len(p)]), 'zero')[i][0] for i in (0, 1)) == \
            R.poly_metrics(p)[:2]


def test_tip_polar(c, fx):
    arms, _, _ = fx
    ang, rad = G.tip_polar(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.body_origin_model, 'float32')
    same(ang, arms['T1_3_tip_angle_f32origin']); same(rad, arms['T1_3_tip_radius'])
    a32, _ = G.tip_polar(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.body_origin_model, 'float32', 'float32')
    same(a32, arms['T1_3_tip_angle_f32'])
    frames = R.stats_frames(c, origin_float32=True)
    ref = np.concatenate([x for x, _ in R.tip_polar_stats(frames)])
    same(ang, ref)
    same(a32, np.concatenate([R.tip_angle_fig6_copy(f) for f in frames]))


def test_body_path_and_4b_lists(c, tables):
    rows, arms, polys = tables
    by_frame = {r['frame']: r for r in rows}
    sk = R.fig4b_skeleton(by_frame, arms, polys, sorted(arms))
    rm, _ = FR.body_radius(c)
    got = G.body_path(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.body_origin_model, rm)
    same(got, np.concatenate([s['xy_unrounded'] for s in sk]))
    same(G.body_path(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.body_origin_model, rm, 3),
         np.concatenate([s['xy_body_radii'] for s in sk]))
    tl = R.fig4b_timeline(by_frame, arms, list(c.frame_index))
    fl = FR.frame_lists_4b(c)
    same(fl['arm_norm_length'], np.concatenate([t['arm_lengths_body_radii'] for t in tl]))
    same(fl['arm_curvature_body_radius'], np.concatenate([t['arm_curvatures_rad_per_body_radius'] for t in tl]))
    same(fl['mean_norm_length'], np.array([np.nan if t['mean_normalized_arm_length'] is None
                                           else t['mean_normalized_arm_length'] for t in tl]))


# ------------------------------------------------------------------------------------------ T2
def test_frames_t2(c, fx):
    _, fr, _ = fx
    same(FR.n_arms(c), fr['T2_T2_1_n_arms'])
    rm, rs = FR.body_radius(c)
    same(rm, fr['T2_T0_6_radius_model']); same(rs, fr['T2_T0_6_radius_source'])
    same(FR.tip_isotropy(c), fr['T2_T2_6_isotropy_frame_support_centroid'])
    same(FR.frame_lists_4b(c)['mean_norm_length'], fr['T2_T2_5_mean_normalized_arm_length'])


def test_isotropy_window_equals_reference(c):
    seg = {'timestamp': c.time_s, 'centroid': c.support_centroid_model, 'arm_offsets': c.frame_arm_offsets,
           'point_offsets': c.arm_point_offsets, 'points': c.points}
    iso = FR.tip_isotropy(c)
    for onset in (400.0, 700.0):
        for fb in (np.nan, 0.0):
            same(FR.window_median(iso, c.time_s - onset, (-300, -60), fb), R.isotropy_window([seg], onset, fb))


# ------------------------------------------------------------------------------------------ T4
def test_relink_matches_reference_and_fixture(c, fx, tables, relink_ids):
    arms, _, _ = fx
    same(relink_ids, arms['T4_4_relink'])
    _, _, polys = tables
    ref = R.relinked_track_ids(R.relink_frames(c, polys))
    same(relink_ids, np.concatenate(ref))
    same(c.track_id, arms['stored_track_id'])
    ids, src = TR.track_ids(c)
    assert src == 'stored' and ids is c.track_id
    # the cold start differs from the warm-started stored IDs on some arms: 1,179 / 1,226 under the best mapping
    best = sum(np.bincount(relink_ids[c.track_id == s]).max() for s in np.unique(c.track_id))
    assert best == 1179


def test_assignment_matrices(c, fx):
    arms, _, _ = fx
    mats = []
    TR.relink(c, assign_hook=lambda m, r, cc: mats.append(m.copy()))
    shapes = arms['lsa_first_shape']
    vals = np.concatenate([m.ravel() for m in mats[:len(shapes)]])
    same(vals, arms['lsa_first_values'])
    assert all(m.shape == tuple(s) for m, s in zip(mats, shapes))


def test_step_costs_fig6(c, fx):
    _, _, meta = fx
    star = np.flatnonzero(c.star_mask())
    for key, order in (('fig6_step_369.5', 'sorted'), ('fig6_flip_371.5', 'insertion')):
        st = meta['fig6_steps'][key]
        k = int(np.flatnonzero(c.frame_index == st['frame_t'])[0])
        nxt = int(star[star > k][0])
        sc = TR.step_costs(c, nxt, origin_dtype='float32', frames=star, track_order=order)
        same(sc['track_ids'], np.array(st['track_ids']))
        same(sc['cost'], np.array(st['cost'], float))
        same(sc['assignment'], np.array(st['assignment']).reshape(-1, 2))
        same(sc['labels'], np.array(st['labels_t1']))
        # the reference copies give the same matrix
        tracker = R.FusedTracker(**R.RELEASED)
        sampled, lengths, embeds = TR.tracker_inputs(c, 'float32')
        fo = c.frame_arm_offsets
        for f in star[star < nxt]:
            if fo[f + 1] > fo[f]:
                tracker.step(int(c.frame_index[f]), sampled[fo[f]:fo[f + 1]], lengths[fo[f]:fo[f + 1]], embeds[fo[f]:fo[f + 1]])
        fn = R.step_cost_fig6_step if order == 'sorted' else R.step_cost_fig6_flip
        ref = fn(tracker, sampled[fo[nxt]:fo[nxt + 1]], lengths[fo[nxt]:fo[nxt + 1]], embeds[fo[nxt]:fo[nxt + 1]],
                 int(c.frame_index[nxt]))
        same(sc['cost'], ref['cost'])


def test_common_span_dist_degenerate():
    p = np.zeros((12, 2), np.float32)
    q = np.ones((12, 2), np.float32)
    assert TR.common_span_dist(p, 0.0, q, 5.0) == pytest.approx(np.sqrt(2))
    assert TR.common_span_dist(q, 3.0, q, 5.0) == 0.0


def test_diagnostics_equal_reference_and_fixture(c, fx):
    _, fr, meta = fx
    states = [STATES.get(int(x)) for x in fr['T2_label_state_code']]
    d = TR.diagnostics(c, states=states, origin_dtype='float32')
    seq = {'name': 'x', 'fps': 2.0, 'frames': R.stats_frames(c, states=states)}
    ref = R.analyse(seq)
    star = np.flatnonzero(c.star_mask())
    fo = c.frame_arm_offsets
    for key, ids in d['runs'].items():
        pos = star[::int(key.rsplit('_g', 1)[1])] if '_g' in key else star
        got = np.concatenate([ids[fo[f]:fo[f + 1]] for f in pos])
        same(got, np.concatenate([np.asarray(x, np.int64) for x in ref['runs'][key]]))
        same(got, fr[f'T4_5_labels_{key}'])
    for v in TR.VARIANTS:
        assert d[v]['by_state'] == {k: {kk: int(vv) for kk, vv in s.items()} for k, s in ref[v]['by_state'].items()}
        assert json.loads(json.dumps(d[v]['by_state'])) == meta['T4_5']['per_variant'][v]['by_state']
        assert d[v]['n_ids'] == meta['T4_5']['per_variant'][v]['n_ids']
        same(d[v]['arm_lifetime_s'], fr[f'T4_5_arm_lifetime_s_{v}'])
    assert {str(k): v for k, v in d['strides'].items()} == meta['T4_5']['strides_result']
    pooled = json.loads(json.dumps(TR.pooled([d], 'fused', tuple(STATES.values()))))
    assert pooled == meta['T4_5']['pooled_fused']


# ------------------------------------------------------------------------------------------ R0 + Fig 4E / 4D / 5
def test_fig4e_equals_reference_and_fixture(c, fx, tables, relink_ids):
    _, fr, meta = fx
    rows, arms, polys = tables
    track_of = {a: int(i) for a, i in enumerate(relink_ids)}
    t = c.times()
    for e, (ev, r0ref) in enumerate(zip(meta['fig4e_events'], meta['fig4e_r0'])):
        res = SC.r0_pre_onset(c, ev)
        assert res.r0 == r0ref['r0_source_px'] and res.n_frames == r0ref['pre_flip_frames']
        ref = R.fig4e_event(rows, arms, polys, track_of, ev)
        assert ref['r0'] == res.r0
        win = (t >= ev['first_time_s']) & (t <= ev['last_time_s'])
        sel = fr['fig4e_event_id'] == e
        got = {'fold_asymmetry': FR.fold_asymmetry(c, res.r0, 'fig4e'),
               'arm_length_spread': FR.arm_length_spread(c, res.r0, 'fig4e'),
               'arm_curvature': FR.arm_curvature(c, res.r0, 'fig4e'),
               'arm_deformation': KI.arm_deformation(c, res.r0, relink_ids, lag='prev_frame_reset', kabsch='svd')}
        for name, v in got.items():
            same(v[win], np.asarray(ref['columns'][name]))
            same(v[win], fr[f'fig4e_{name}'][sel])
        close(KI.arm_deformation(c, res.r0, relink_ids, lag='prev_frame_reset')[win], fr['fig4e_arm_deformation'][sel])


def test_fig4d_equals_reference_and_fixture(c, fx, tables, relink_ids):
    _, fr, meta = fx
    rows, arms, polys = tables
    track_of = {a: int(i) for a, i in enumerate(relink_ids)}
    bouts = meta['fig4d_bouts']
    ref = R.fig4d_clip(rows, arms, polys, track_of, bouts)
    res = SC.r0_outside_windows(c, [(b['start_s'], b['stop_s']) for b in bouts])
    assert res.r0 == ref['r0'] == meta['fig4d_r0'] and res.area0 == ref['area0'] == meta['fig4d_area0']
    out = compute_all(c, 'fig4d', windows=[(b['start_s'], b['stop_s']) for b in bouts], track='relink', kabsch='svd')
    out_c = compute_all(c, 'fig4d', windows=[(b['start_s'], b['stop_s']) for b in bouts], track='relink')
    t = c.times()
    for i, b in enumerate(bouts):
        lo, hi = b['start_s'] - 2, b['stop_s'] + 2
        win = (t >= lo) & (t <= hi)
        sel = np.asarray(ref['columns']['bout_id']) == i
        fsel = fr['fig4d_bout_id'] == i
        for name in ('fold_asymmetry', 'arm_length_spread', 'arm_curvature'):
            same(out[name][win], np.asarray(ref['columns'][name])[sel])
            same(out[name][win], fr[f'fig4d_{name}'][fsel])
        d = KI.arm_deformation(c, res.r0, relink_ids, lag='prev_frame_reset', shape_mask=(t >= lo - 1) & (t <= hi),
                               kabsch='svd')
        logd = np.array([np.nan if not np.isfinite(x) else float(np.log10(max(float(x), 1e-3))) for x in d])
        same(logd[win], np.asarray(ref['columns']['log_deformation'])[sel])
        same(logd[win], fr['fig4d_log_deformation'][fsel])
        same(out['log_deformation'][win], fr['fig4d_log_deformation'][fsel])
        same(out['log_deformation_window'][i][win], fr['fig4d_log_deformation'][fsel])
        close(out_c['log_deformation'][win], fr['fig4d_log_deformation'][fsel])


def test_fig5_definitions_equal_reference_and_fixture(c, fx, relink_ids):
    _, fr, meta = fx
    frames = R.fig5_frames(c)
    for tag, r0 in meta['fig5_r0'].items():
        ref = R.fig5_window(frames, 331.0, r0=r0, area0=1.0)
        same(relink_ids, np.concatenate([np.asarray(x, np.int64) for x in ref['labels']]))
        out = compute_all(c, 'fig5', r0=r0, track='relink', kabsch='svd')
        for name in ('fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'tip_motion', 'arm_deformation'):
            same(out[name], np.asarray(ref['rows'][name]))
            same(out[name], fr[f'fig5_{tag}_{name}'])
        out_c = compute_all(c, 'fig5', r0=r0, track='relink')
        close(out_c['arm_deformation'], fr[f'fig5_{tag}_arm_deformation'])
        for name in ('fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'tip_motion'):
            same(out_c[name], out[name])


def test_fig5_baseline_drop_and_r0(c):
    # 181 frames at 2 Hz: an onset 400 s puts 160..340 s in the baseline window
    frames = R.fig5_frames(c)
    ref = R.fig5_window(frames, 400.0)
    res = SC.r0_baseline(c, 400.0)
    assert ref['r0_source_px'] == res.r0 and ref['area0'] == res.area0
    assert R.fig5_window(frames, 1000.0) is None and SC.r0_baseline(c, 1000.0) is None
    out = compute_all(c, 'fig5', onset_s=1000.0)
    assert bool(out['dropped']) and 'fold_asymmetry' not in out


def test_lag_variants_reference(c, relink_ids):
    """Lag A (Fig 4) and lag B (Fig 5) agree on a gap-free clip; lag A resets after a frame without arms."""
    r0 = 50.0
    a = KI.arm_deformation(c, r0, relink_ids, lag='prev_frame_reset', kabsch='svd')
    b = KI.arm_deformation(c, r0, relink_ids, lag='last_arm_frame', shape_order='fig4', kabsch='svd')
    same(a, b)
    same(KI.arm_deformation(c, r0, relink_ids, lag='prev_frame_reset'),
         KI.arm_deformation(c, r0, relink_ids, lag='last_arm_frame', shape_order='fig4'))
    close(KI.arm_deformation(c, r0, relink_ids, lag='prev_frame_reset'), a)
    frames = [{'time_s': float(c.time_s[f]), 'has_star': bool(c.has_star[f]),
               'arms': [(int(relink_ids[x]), np.asarray(c.arm(x), float), float(c.crop[f, 2])) for x in c.frame_arms(f)]}
              for f in range(c.n_frames)]
    same(a, np.array(R.deformation_lag_a(frames, r0)))
    gap = c.select_frames(np.r_[0:50, 51:c.n_frames]).with_fields()
    star = gap.has_star.copy(); star[49] = False         # drop a star frame -> lag A NaN on the next frame
    gap = gap.with_fields(has_star=star)
    ids = np.r_[relink_ids[:c.frame_arm_offsets[50]], relink_ids[c.frame_arm_offsets[51]:]]
    la = KI.arm_deformation(gap, r0, ids, lag='prev_frame_reset')
    lb = KI.arm_deformation(gap, r0, ids, lag='last_arm_frame')
    assert np.isnan(la[49]) and np.isnan(la[50]) and np.isfinite(lb[50])


def test_ghosts_equal_reference(c, relink_ids):
    onset = 365.0
    frames = [{'t': float(c.time_s[f]) - onset, 'crop': c.crop[f].copy(), 'origin': c.body_origin_model[f].copy(),
               'arms': [c.arm(a).copy() for a in c.frame_arms(f)],
               'identity': [c.identity[a].copy() for a in c.frame_arms(f)]} for f in range(c.n_frames)]
    ref = R.anchored_ghosts(frames, {'Q': (-30.0, -20.0), 'SA': (20.0, 30.0)})
    for name, win in (('Q', (-30.0, -20.0)), ('SA', (20.0, 30.0))):
        g = G.anchored_ghosts(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.crop, c.body_origin_model,
                              c.time_s - onset, win, labels=relink_ids)
        same(g['points'], np.concatenate(ref[name]['ghosts']))
        same(g['labels'], np.array(ref[name]['labels']))
        assert g['t_ref'] == ref[name]['t_ref']
        assert sorted(g['tip_paths']) == sorted(int(k) for k in ref[name]['tip_paths'])
        for k, v in ref[name]['tip_paths'].items():
            same(g['tip_paths'][int(k)], np.asarray(v))


def test_compute_all_presets(c):
    ev = c.meta['bouts'][0]
    for preset, kw in (('fig4b', {}), ('fig4e', {'bout': ev}), ('fig5', {'r0': 50.0}),
                       ('fig6_flipping', {}), ('fig5_example', {'onset_s': 400.0})):
        out = compute_all(c, preset, **kw)
        assert out['arc_length'].shape == (c.n_arms,)
        assert out['n_arms'].shape == (c.n_frames,)
        assert out['track_id'].shape == (c.n_arms,) and str(out['track_id_source']) == 'stored'
    out = compute_all(c, 'fig4b')
    assert out['body_path'].shape == (c.n_points, 2) and out['mean_norm_length'].shape == (c.n_frames,)
    out = compute_all(c, 'fig5_example', onset_s=1000.0)
    assert float(out['tip_isotropy_window']) == 0.0
    out = compute_all(c, 'fig5', onset_s=1000.0)
    assert np.isnan(float(out['tip_isotropy_window']))
