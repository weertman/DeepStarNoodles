#!/usr/bin/env python3
"""Parity gate: compare the public noodle_polylines API with the parity reference store.

The reference store (not in the repo) holds outputs of the paper's figure-script code on the released
re-trace: the micro example, 7 full flipping clips and a 3-animal Sleepy Stars subset (18 Q->SA
boundaries, 36 segments). Every optimisation round must pass this gate unchanged.

    python benchmarks/noodle_polylines/parity_check.py --ref <store>/ref \
        --data-dir RUNS/paper_retrace_f39dc82b [--sources <store>/inputs/sources.json] \
        [--only micro,flip,sleepy] [--quick] [--jobs 16]

The released clip behind each store entry (and the source of the micro example) is read from the
store's own ``sources.json`` ({"flip_clips": {name: clip_id}, "micro": {"clip": clip_id,
"frame_range": [lo, hi]}}), which is not committed. Without it the adapter cross-check is skipped and
the store's input containers are used.

Tolerances (never loosened silently): integer, label, mask and decision outputs exact; tip-first
resample <= 1e-5 px; every other float rtol 1e-9 with identical NaN and +-inf positions. The table also
says whether a float output is bit-identical.

Near-tie audit (information, never a failure): for every tracker assignment of the flipping relinks
(fused, shape-only and uncapped) and of the Sleepy Fig 5 event relinks, the decision margin
(``tracking.assignment_margin``) is computed and margins below 1e-6 are listed, so that a future change
that is not bit-identical can be judged against the assignments it could flip.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'CORE'))

import numpy as np  # noqa: E402

import noodle_polylines as NP  # noqa: E402
from noodle_polylines import _reference as R  # noqa: E402
from noodle_polylines import frames as FR, geometry as G, kinematics as KI, scale as SC, tracking as TR  # noqa: E402
from noodle_polylines.container import PolylineContainer, concat, split_at_gaps  # noqa: E402

def load_sources(path: Path | None) -> dict:
    """{'flip_clips': {store name: clip id}, 'micro': {'clip', 'frame_range'}} from the store (not committed)."""
    if path is None or not Path(path).exists():
        return {'flip_clips': {}, 'micro': None}
    d = json.loads(Path(path).read_text())
    return {'flip_clips': dict(d.get('flip_clips', {})), 'micro': d.get('micro')}
STATE_NAMES = {0: 'Right Side Up', 1: 'Being flipped', 2: 'Upside Down'}
SLEEPY_STATES = ('quiescent', 'SA', 'moving')
RESAMPLE_ATOL = 1e-5
RTOL = 1e-9
NEAR_TIE = 1e-6        # near-tie audit: LSA decision margins below this are flagged (reported, not failed)


# ------------------------------------------------------------------------------------------ comparison
class Checks:
    def __init__(self, scope: str):
        self.scope = scope
        self.rows: list[dict] = []

    def add(self, rep: str, got, ref, kind: str = 'float', note: str = '') -> None:
        """kind: 'exact' (ints, labels, masks, decisions), 'resample' (<= 1e-5 abs), 'float' (rtol 1e-9)."""
        row = {'scope': self.scope, 'rep': rep, 'kind': kind, 'note': note}
        try:
            g, r = np.asarray(got), np.asarray(ref)
            if g.shape != r.shape:
                row.update(ok=False, bit=False, detail=f'shape {g.shape} != {r.shape}')
            elif kind == 'exact':
                eq = np.array_equal(g, r, equal_nan=g.dtype.kind == 'f')
                row.update(ok=bool(eq), bit=bool(eq), detail='' if eq else f'{int(np.sum(g != r))} differ')
            else:
                g64, r64 = g.astype(np.float64), r.astype(np.float64)
                gf, rf = np.isfinite(g64), np.isfinite(r64)
                # NaN, +inf and -inf must sit at the same positions with the same values
                nan_ok = np.array_equal(gf, rf) and np.array_equal(g64[~gf], r64[~rf], equal_nan=True)
                m = rf & gf
                bit = nan_ok and np.array_equal(g64, r64, equal_nan=True)
                d = np.abs(g64[m] - r64[m])
                mx = float(d.max()) if d.size else 0.0
                if kind == 'resample':
                    ok = nan_ok and mx <= RESAMPLE_ATOL
                    detail = f'max abs {mx:.3g}'
                else:
                    rel = d / np.maximum(np.abs(r64[m]), 1e-300)
                    mr = float(rel.max()) if rel.size else 0.0
                    ok = nan_ok and bool(np.all(d <= RTOL * np.abs(r64[m]) + 1e-300) or mr <= RTOL)
                    detail = f'max rel {mr:.3g}'
                if not nan_ok:
                    detail += ' non-finite (NaN/inf) entries differ'
                row.update(ok=bool(ok), bit=bool(bit), detail=detail, n=int(r.size))
        except Exception as exc:   # noqa: BLE001
            row.update(ok=False, bit=False, detail=f'error {exc!r}')
        self.rows.append(row)

    def fail(self, rep: str, msg: str) -> None:
        self.rows.append({'scope': self.scope, 'rep': rep, 'kind': 'run', 'ok': False, 'bit': False, 'detail': msg})


def near_tie_audit(chk: Checks, rep: str, matrices: list, capped: bool = True) -> None:
    """Record the decision margin of every tracker LSA call (tracking.assignment_margin) as an info row.

    A margin below NEAR_TIE means a last-bit difference in one cost could change which pairs are kept;
    such calls are listed (by call number) so a future non-bit-identical change can be judged."""
    margins = np.array([TR.assignment_margin(m, capped) for m in matrices], np.float64)
    near = np.flatnonzero(margins < NEAR_TIE)
    mn = float(margins.min()) if len(margins) else float('inf')
    chk.rows.append({'scope': chk.scope, 'rep': rep, 'kind': 'info', 'audit': 'near_tie', 'ok': True, 'bit': True,
                     'n_lsa': int(len(margins)), 'n_near': int(len(near)), 'min_margin': mn,
                     'near_calls': near.tolist(), 'near_margins': margins[near].tolist(),
                     'detail': f'{len(margins)} LSA calls, {len(near)} with margin < {NEAR_TIE:g}, min {mn:.3g}'})


def _jd(o, **kw):
    """Canonical JSON (numpy scalars as Python numbers, keys as sorted strings)."""
    kw.setdefault('sort_keys', True)
    plain = json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    return json.dumps(plain, **kw)


def ragged_eq(values, offsets):
    return [values[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


# ------------------------------------------------------------------------------------------ flipping
def flip_container(name: str, ref: Path, data_dir: Path | None, chk: Checks, sources: dict) -> PolylineContainer:
    rc = PolylineContainer.load(ref / name / 'input_container.npz', validate=False, strict=False)
    if name == 'micro':
        mi = sources.get('micro')
        clip, rng = (mi['clip'], tuple(mi['frame_range'])) if mi else (None, None)
    else:
        clip, rng = sources.get('flip_clips', {}).get(name), None
    if data_dir is not None and clip is not None:
        folder = data_dir / 'flipping' / 'clips' / clip
        if folder.exists():
            c = NP.adapters.from_flipping_clip(folder, frame_range=rng)
            for k in ('points', 'arm_point_offsets', 'frame_arm_offsets', 'frame_index', 'time_s', 'crop',
                      'body_origin_model', 'body_origin_code', 'support_area_model_px', 'support_centroid_model',
                      'has_star', 'identity', 'track_id'):
                chk.add(f'A-FLIP adapter.{k}', getattr(c, k), getattr(rc, k), 'exact')
            return c
    rc.meta.setdefault('fps', 2.0)
    return rc.validate()


def check_flip(name: str, ref_root: str, data_dir: str | None, quick: bool, sources: dict | None = None) -> list[dict]:
    ref = Path(ref_root)
    d = ref / name
    chk = Checks(name)
    t0 = time.perf_counter()
    c = flip_container(name, ref, Path(data_dir) if data_dir else None, chk, sources or {})
    pts, ao, fo = c.points, c.arm_point_offsets, c.frame_arm_offsets
    af = c.arm_frame()
    a = np.load(d / 'T0_T1_arms.npz')
    # T0.2
    chk.add('T0.2 resample_base', G.resample_base(pts, ao), a['T0_2_base'])
    tip, L = G.resample_tip(pts, ao)
    chk.add('T0.2 resample_tip raw', tip, a['T0_2_tip_raw'], 'resample')
    chk.add('T0.2 resample_tip raw L', L, a['T0_2_tip_raw_L'])
    for tag, dt in (('f64origin', np.float64), ('f32origin', np.float32)):
        tip, L = G.resample_tip(pts, ao, arm_origin=c.body_origin_model[af].astype(dt))
        chk.add(f'T0.2 resample_tip body-rel {tag}', tip, a[f'T0_2_tip_rel_{tag}'], 'resample')
        chk.add(f'T0.2 resample_tip body-rel {tag} L', L, a[f'T0_2_tip_rel_{tag}_L'])
    # T0.3 / T1.1
    Lz, kz = G.arc_length_curvature(pts, ao, 'zero')
    Ln, kn = G.arc_length_curvature(pts, ao, 'nan')
    chk.add('T0.3 arc_length (poly_metrics)', Lz, a['T0_3_arc_length_poly_metrics_f32'])
    chk.add('T1.1 curvature (poly_metrics)', kz, a['T1_1_curvature_poly_metrics_f32'])
    chk.add('T0.3 arc_length (arm_metrics)', Ln, a['T0_3_arc_length_arm_metrics_f32'])
    chk.add('T1.1 curvature (arm_metrics)', kn, a['T1_1_curvature_arm_metrics_f32'])
    chk.add('T0.3 == export csv', Lz, a['csv_arm_length_model_px'])
    chk.add('T1.1 == export csv', kz, a['csv_curvature_model_rad_px'])
    # T0.4 / T0.5 / T1.2
    chk.add('T0.4 to_source points', G.to_source(pts, ao, fo, c.crop), a['T0_4_source_points'])
    chk.add('T0.4 tips_source', G.tips_source(pts, ao, fo, c.crop), a['T0_4_source_tip'])
    chk.add('T0.4 tips_source == export csv', G.tips_source(pts, ao, fo, c.crop), a['csv_tip_source'])
    chk.add('T0.5 body_relative', G.body_relative(pts, ao, fo, c.body_origin_model), a['T0_5_body_relative_points'])
    rm, rs = FR.body_radius(c, 'nan')
    chk.add('T1.2 norm_length', G.norm_length(Lz, fo, rm), a['T1_2_norm_length'])
    # T1.4 (every frame with arms)
    bp = np.load(d / 'T1_4_body_path.npz')
    chk.add('T1.4 body_path', G.body_path(pts, ao, fo, c.body_origin_model, rm), bp['xy_body_radii'])
    chk.add('T1.4 body_path round3', G.body_path(pts, ao, fo, c.body_origin_model, rm, 3), bp['xy_body_radii_round3'])
    # T2 / T0.6
    fz = np.load(d / 'T2_frames.npz')
    chk.add('T2.1 n_arms', FR.n_arms(c), fz['T2_1_n_arms'], 'exact')
    chk.add('T0.6 radius_model', rm, fz['T0_6_radius_model'])
    chk.add('T0.6 radius_source', rs, fz['T0_6_radius_source'])
    chk.add('T0.7 per_frame R0', SC.r0_per_frame(c).r0, fz['T0_7_R0_per_frame'])
    chk.add('T2.6 isotropy (support centroid)', FR.tip_isotropy(c), fz['T2_6_isotropy_frame_support_centroid'])
    chk.add('T2.6 isotropy (runtime origin frames)', FR.tip_isotropy(c, 'body_origin_model'),
            fz['T2_6_isotropy_frame_runtime_origin'])
    fl = FR.frame_lists_4b(c, Lz, kz)
    chk.add('T2.5 mean norm length', fl['mean_norm_length'], fz['T2_5_mean_normalized_arm_length'])
    chk.add('T2.5 arm lengths (body radii)', fl['arm_norm_length'], fz['T2_5_arm_lengths_body_radii'])
    chk.add('T2.5 arm curvature x r_model', fl['arm_curvature_body_radius'], fz['T2_5_arm_curvatures_rad_per_body_radius'])
    # T4.4 relink (f64 origin, frames with arms) + every assignment matrix
    rl = np.load(d / 'T4_relink.npz')
    mats = []
    ids = TR.relink(c, assign_hook=lambda m, r_, c_: mats.append((m.copy(), r_.copy(), c_.copy())))
    chk.add('T4.4 relink ids', ids, rl['T4_4_relink_ids'], 'exact')
    if name != 'micro':
        chk.add('T4.4 stored track_id == relink', c.track_id, rl['T4_4_relink_ids'], 'exact')
    vals = np.concatenate([m.ravel() for m, _, _ in mats]) if mats else np.zeros(0)
    chk.add('T4.2/4.3 LSA matrices (augmented cost)', vals, rl['lsa_full_values'])
    chk.add('T4.3 LSA rows', np.concatenate([r_ for _, r_, _ in mats]), rl['lsa_rows'], 'exact')
    chk.add('T4.3 LSA cols', np.concatenate([c_ for _, _, c_ in mats]), rl['lsa_cols'], 'exact')
    near_tie_audit(chk, 'T4.3 near-tie audit (fused relink)', [m for m, _, _ in mats])
    for v in ('shape_only', 'no_cap'):
        vm = []
        TR.relink(c, TR.VARIANTS[v], assign_hook=lambda m, r_, c_: vm.append(m.copy()))
        near_tie_audit(chk, f'T4.3 near-tie audit ({v} relink)', vm, capped=TR.VARIANTS[v].max_match_cost is not None)
    if name == 'micro':
        cs = np.load(d / 'T4_1_common_span_dist_calls.npz')
        got = np.array([TR.common_span_dist(cs['pa'][i], float(cs['la'][i]), cs['pb'][i], float(cs['lb'][i]))
                        for i in range(len(cs['d']))])
        chk.add('T4.1 common_span_dist (9,547 calls)', got, cs['d'])
    # T3.3 / T1.3 (stats path: star frames, f32 origin)
    tt = np.load(d / 'T3_3_tip_angle_track.npz')
    tr = KI.tip_angle_track(c, ids, origin_dtype='float32', angle_dtype='float64')
    chk.add('T1.3/T3.3 tip angle (f32 origin, f64 angle)', tr['tip_angle_deg'], tt['tip_angle_deg'])
    chk.add('T1.3 tip radius', tr['tip_radius_px'], tt['tip_radius_px'])
    a32, _ = G.tip_polar(pts, ao, fo, c.body_origin_model, 'float32', 'float32')
    chk.add('T1.3 tip angle float32 (Fig 6C copy)', a32, tt['tip_angle_deg_fig6_flip_copy_f32'])
    # Fig 4E (pre-onset R0, lag A) per event
    kj = json.loads((d / 'fig4e_kinematics.json').read_text())
    kz_ = np.load(d / 'fig4e_kinematics.npz')
    t = c.times()
    for e, (ev, rr) in enumerate(zip(kj['events'], kj['r0_by_event_T0_7_pre_onset'])):
        res = SC.r0_pre_onset(c, ev)
        chk.add(f'Fig4E ev{e} R0 pre_onset', res.r0, rr['r0_source_px'])
        chk.add(f'Fig4E ev{e} R0 frames', res.n_frames, rr['pre_flip_frames'], 'exact')
        sel = kz_['event_id'] == e
        win = (t >= ev['first_time_s']) & (t <= ev['last_time_s'])
        star = c.star_mask()
        chk.add(f'Fig4E ev{e} n_arms', np.where(star, c.arms_per_frame(), 0)[win], kz_['n_arms'][sel], 'exact')
        chk.add(f'Fig4E ev{e} fold_asymmetry', FR.fold_asymmetry(c, res.r0, 'fig4e')[win], kz_['fold_asymmetry'][sel])
        chk.add(f'Fig4E ev{e} arm_length_spread', FR.arm_length_spread(c, res.r0, 'fig4e', Lz)[win],
                kz_['arm_length_spread'][sel])
        chk.add(f'Fig4E ev{e} arm_curvature', FR.arm_curvature(c, res.r0, 'fig4e', kz)[win], kz_['arm_curvature'][sel])
        if quick and e < len(kj['events']) - 1:
            continue
        chk.add(f'Fig4E ev{e} arm_deformation (lag A)',
                KI.arm_deformation(c, res.r0, ids, lag='prev_frame_reset')[win], kz_['arm_deformation'][sel])
    # Fig 4D (outside-windows R0, windows +-2 s)
    tj = json.loads((d / 'fig4d_tree_features.json').read_text())
    tz = np.load(d / 'fig4d_tree_features.npz')
    wins = [(b['start_s'], b['stop_s']) for b in tj['bouts']]
    res = SC.r0_outside_windows(c, wins)
    chk.add('Fig4D R0 outside_windows', res.r0, tj['T0_7_R0_outside_windows'])
    chk.add('Fig4D area0 outside_windows', res.area0, tj['area0_outside_windows'])
    fold = FR.fold_asymmetry(c, res.r0, 'fig4d')
    spread = FR.arm_length_spread(c, res.r0, 'fig4d', Lz)
    curv = FR.arm_curvature(c, res.r0, 'fig4d', kz)
    base = G.resample_base(pts, ao)
    for i, (lo, hi) in enumerate((s - 2.0, e_ + 2.0) for s, e_ in wins):
        sel = tz['bout_id'] == i
        win = (t >= lo) & (t <= hi)
        chk.add(f'Fig4D bout{i} frames', c.frame_index[win], tz['frame'][sel].astype(np.int64), 'exact')
        chk.add(f'Fig4D bout{i} fold_asymmetry', fold[win], tz['fold_asymmetry'][sel])
        chk.add(f'Fig4D bout{i} arm_length_spread', spread[win], tz['arm_length_spread'][sel])
        chk.add(f'Fig4D bout{i} arm_curvature', curv[win], tz['arm_curvature'][sel])
        dfm = KI.arm_deformation(c, res.r0, ids, lag='prev_frame_reset', shape_mask=(t >= lo - 1) & (t <= hi), base=base)
        logd = np.array([np.nan if not np.isfinite(x) else float(np.log10(max(float(x), 1e-3))) for x in dfm])
        chk.add(f'Fig4D bout{i} log_deformation', logd[win], tz['log_deformation'][sel])
    # Fig 5 definitions on flipping (lag B, tip motion) with the named R0
    fj = json.loads((d / 'fig5_measures.json').read_text())
    for tag, run in fj['runs'].items():
        z5 = np.load(d / f'fig5_measures_{tag}.npz')
        r0 = run['r0']
        chk.add(f'Fig5[{tag}] relink labels', ids[ids >= 0], z5['labels'], 'exact')
        chk.add(f'Fig5[{tag}] fold_asymmetry', FR.fold_asymmetry(c, r0, 'fig5'), z5['fold_asymmetry'])
        chk.add(f'Fig5[{tag}] arm_length_spread', FR.arm_length_spread(c, r0, 'fig5', Ln), z5['arm_length_spread'])
        chk.add(f'Fig5[{tag}] arm_curvature', FR.arm_curvature(c, r0, 'fig5', kn), z5['arm_curvature'])
        chk.add(f'Fig5[{tag}] tip_motion', KI.tip_motion(c, r0, ids), z5['tip_motion'])
        chk.add(f'Fig5[{tag}] arm_deformation (lag B)', KI.arm_deformation(c, r0, ids, base=base), z5['arm_deformation'])
        if quick:
            break
    if name == 'micro':
        s12 = np.load(d / 'T1_6_shape12.npz')
        for e, r0 in enumerate(s12['r0']):
            chk.add(f'T1.6 shape12 ev{e}', G.shape12(base, c.crop[af, 2], r0, 'fig5'), s12[f'shape12_event{e}'])
    # Presets: compute_all must reproduce each panel on the full clip (stored IDs = the full-clip re-link)
    check_presets_flip(chk, c, d, kj, kz_, tj, tz, fj, tt, ids, 'relink' if name == 'micro' else 'stored')
    # Fig 6A / 6C step cost blocks
    sp = d / 'T4_2_T4_3_fig6_steps.json'
    if sp.exists():
        sj = json.loads(sp.read_text())
        star_pos = np.flatnonzero(c.star_mask())
        for key, order in (('fig6_step_369.5', 'sorted'), ('fig6_flip_371.5', 'insertion')):
            st = sj[key]
            k_t = int(np.flatnonzero(c.frame_index == st['frame_t'])[0])
            nxt = star_pos[star_pos > k_t][0]
            sc = TR.step_costs(c, int(nxt), origin_dtype='float32', frames=star_pos, track_order=order)
            chk.add(f'{key} track_ids', sc['track_ids'], np.array(st['track_ids']), 'exact')
            chk.add(f'{key} cost', sc['cost'], np.array(st['cost'], float))
            if 'shape_distance' in st:
                chk.add(f'{key} shape_distance', sc['shape_distance'], np.array(st['shape_distance']))
                chk.add(f'{key} embedding_distance', sc['embedding_distance'], np.array(st['embedding_distance']))
            chk.add(f'{key} assignment', sc['assignment'], np.array(st['assignment'], np.int64).reshape(-1, 2), 'exact')
            chk.add(f'{key} labels t+1', sc['labels'], np.array(st['labels_t1']), 'exact')
    # T4.5 diagnostics (stats path: star frames, f32 origin, 10 passes)
    if not quick:
        dz = np.load(d / 'T4_5_stats_runs.npz')
        dj = json.loads((d / 'T4_5_diagnostics.json').read_text())
        codes = fz['label_state_code']
        states = [STATE_NAMES.get(int(x)) for x in codes]
        diag = TR.diagnostics(c, states=states, origin_dtype='float32', fps=2.0)
        star_pos = np.flatnonzero(c.star_mask())
        fo_ = c.frame_arm_offsets
        for key, ids_ in diag['runs'].items():
            if '_g' in key:
                g = int(key.rsplit('_g', 1)[1])
                sub = star_pos[::g]
                got = np.concatenate([ids_[fo_[f]:fo_[f + 1]] for f in sub]) if len(sub) else np.zeros(0, np.int64)
            else:
                got = np.concatenate([ids_[fo_[f]:fo_[f + 1]] for f in star_pos])
            chk.add(f'T4.5 labels {key}', got, dz[f'labels_{key}'], 'exact')
        for v in TR.VARIANTS:
            chk.add(f'T4.5 {v} arm_lifetime_s', diag[v]['arm_lifetime_s'], dz[f'arm_lifetime_s_{v}'], 'exact')
            chk.add(f'T4.5 {v} by_state+n_ids',
                    _jd({'by_state': diag[v]['by_state'], 'n_ids': diag[v]['n_ids']}, sort_keys=True),
                    _jd(dj['per_variant'][v], sort_keys=True), 'exact')
        chk.add('T4.5 strides', _jd({str(k): v for k, v in diag['strides'].items()}, sort_keys=True),
                _jd(dj['strides_result'], sort_keys=True), 'exact')
        pooled = TR.pooled([diag], 'fused', tuple(STATE_NAMES.values()))
        chk.add('T4.5 pooled fused', _jd(pooled, sort_keys=True), _jd(dj['pooled_fused'], sort_keys=True),
                'exact')
    chk.rows.append({'scope': name, 'rep': '(wall s)', 'kind': 'info', 'ok': True, 'bit': True,
                     'detail': f'{time.perf_counter() - t0:.1f}'})
    return chk.rows


def check_presets_flip(chk: Checks, c: PolylineContainer, d: Path, kj, kz_, tj, tz, fj, tt, ids, track: str) -> None:
    """compute_all per preset against the store (the same reference values as the per-function checks).

    ``track``: 'stored' on full clips (stored IDs == the full-clip re-link, checked above); 'relink' on the
    micro excerpt, whose store values come from a cold re-link of the excerpt."""
    t = c.times()
    for e, ev in enumerate(kj['events']):
        out = NP.compute_all(c, 'fig4e', bout=ev, track=track)
        win = (t >= ev['first_time_s']) & (t <= ev['last_time_s'])
        sel = kz_['event_id'] == e
        for k in ('fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'arm_deformation'):
            chk.add(f'preset fig4e ev{e} {k}', out[k][win], kz_[k][sel])
    wins = [(b['start_s'], b['stop_s']) for b in tj['bouts']]
    out = NP.compute_all(c, 'fig4d', windows=wins, track=track)
    chk.add('preset fig4d R0', float(out['r0']), tj['T0_7_R0_outside_windows'])
    for i, (s_, e_) in enumerate(wins):
        win = (t >= s_ - 2.0) & (t <= e_ + 2.0)
        sel = tz['bout_id'] == i
        for k in ('fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'log_deformation'):
            chk.add(f'preset fig4d bout{i} {k}', out[k][win], tz[k][sel])
        chk.add(f'preset fig4d bout{i} log_deformation_window', out['log_deformation_window'][i][win],
                tz['log_deformation'][sel])
    for tag, run in fj['runs'].items():
        z5 = np.load(d / f'fig5_measures_{tag}.npz')
        out = NP.compute_all(c, 'fig5', r0=run['r0'], track=track)
        for k in ('fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'tip_motion', 'arm_deformation'):
            chk.add(f'preset fig5[{tag}] {k}', out[k], z5[k])
    out = NP.compute_all(c, 'fig6_flipping', track=track)
    chk.add('preset fig6_flipping tip_angle (6D)', out['tip_angle_deg'], tt['tip_angle_deg'])
    chk.add('preset fig6_flipping tip_radius', out['tip_radius_px'], tt['tip_radius_px'])
    chk.add('preset fig6_flipping track_id', out['track_id'], ids, 'exact')
    out = NP.compute_all(c, 'fig6c', track=track)
    chk.add('preset fig6c tip_angle (6C, f32)', out['tip_angle_deg'], tt['tip_angle_deg_fig6_flip_copy_f32'])
    chk.add('preset fig6c tip_angle dtype', str(out['tip_angle_deg'].dtype), 'float32', 'exact')


def check_oracle(name: str, ref_root: str, quick: bool) -> list[dict]:
    """The _reference ports themselves (figure-script loops on script-shaped inputs) against the store."""
    ref = Path(ref_root)
    d = ref / name
    chk = Checks(name + '/oracle')
    c = PolylineContainer.load(d / 'input_container.npz', validate=False, strict=False)
    rows, arms, polys = R.flip_tables(c)
    ids = np.concatenate(R.relinked_track_ids(R.relink_frames(c, polys)))
    chk.add('oracle T4.4 relinked_track_ids', ids, np.load(d / 'T4_relink.npz')['T4_4_relink_ids'], 'exact')
    track_of = {a: int(i) for a, i in enumerate(ids)}
    kj = json.loads((d / 'fig4e_kinematics.json').read_text())
    kz_ = np.load(d / 'fig4e_kinematics.npz')
    for e, ev in enumerate(kj['events']):
        res = R.fig4e_event(rows, arms, polys, track_of, ev)
        chk.add(f'oracle Fig4E ev{e} R0', res['r0'], kj['r0_by_event_T0_7_pre_onset'][e]['r0_source_px'])
        sel = kz_['event_id'] == e
        for m in ('n_arms', 'phase', 'relative_time_s', 'body_speed', 'arm_deformation', 'fold_asymmetry',
                  'arm_length_spread', 'arm_curvature', 'support_area'):
            chk.add(f'oracle Fig4E ev{e} {m}', np.asarray(res['columns'][m]), kz_[m][sel],
                    'exact' if m in ('n_arms', 'phase') else 'float')
    tj = json.loads((d / 'fig4d_tree_features.json').read_text())
    tz = np.load(d / 'fig4d_tree_features.npz')
    res = R.fig4d_clip(rows, arms, polys, track_of, tj['bouts'])
    chk.add('oracle Fig4D R0', res['r0'], tj['T0_7_R0_outside_windows'])
    for m in ('fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'log_speed', 'support_area', 'log_deformation'):
        chk.add(f'oracle Fig4D {m}', np.asarray(res['columns'][m]), tz[m])
    fj = json.loads((d / 'fig5_measures.json').read_text())
    frames = R.fig5_frames(c)
    for tag, run in fj['runs'].items():
        z5 = np.load(d / f'fig5_measures_{tag}.npz')
        res = R.fig5_window(frames, 0.0, r0=run['r0'], area0=1.0)
        for m in ('body_speed', 'fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'support_area', 'tip_motion',
                  'arm_deformation'):
            chk.add(f'oracle Fig5[{tag}] {m}', np.asarray(res['rows'][m]), z5[m])
        if quick:
            break
    if not quick:
        fz = np.load(d / 'T2_frames.npz')
        states = [STATE_NAMES.get(int(x)) for x in fz['label_state_code']]
        seq = {'name': name, 'fps': 2.0, 'frames': R.stats_frames(c, states=states)}
        res = R.analyse(seq)
        dz = np.load(d / 'T4_5_stats_runs.npz')
        dj = json.loads((d / 'T4_5_diagnostics.json').read_text())
        for key, labs in res['runs'].items():
            chk.add(f'oracle T4.5 labels {key}', np.concatenate([np.asarray(x, np.int64) for x in labs]),
                    dz[f'labels_{key}'], 'exact')
        for v in R.VARIANTS:
            chk.add(f'oracle T4.5 {v} by_state+n_ids', _jd({'by_state': res[v]['by_state'], 'n_ids': res[v]['n_ids']}),
                    _jd(dj['per_variant'][v]), 'exact')
        chk.add('oracle T4.5 pooled fused', _jd(R.pooled([res], 'fused', tuple(STATE_NAMES.values()))),
                _jd(dj['pooled_fused']), 'exact')
    return chk.rows


# ------------------------------------------------------------------------------------------ sleepy
def sleepy_meta(inputs: Path):
    subset = json.loads((inputs / 'sleepy_subset_manifest.json').read_text())
    index = json.loads((inputs / 'sleepy_index.json').read_text())
    return subset, index


def check_sleepy_segments(segs: list[str], ref_root: str, data_dir: str) -> list[dict]:
    chk = Checks('sleepy/segments')
    for s in segs:
        c = NP.adapters.from_sleepy_segment(Path(data_dir) / 'sleepy_q_to_sa' / f'{s}.npz')
        z = np.load(Path(ref_root) / 'sleepy' / 'segments' / f'{s}.npz')
        pts, ao, fo = c.points, c.arm_point_offsets, c.frame_arm_offsets
        if c.n_arms == 0:
            continue
        af = c.arm_frame()
        chk.add('T0.2 resample_base', G.resample_base(pts, ao), z['T0_2_base'].reshape(-1, 12, 2))
        tip, L = G.resample_tip(pts, ao)
        chk.add('T0.2 resample_tip raw', tip, z['T0_2_tip_raw'].reshape(-1, 12, 2), 'resample')
        chk.add('T0.2 resample_tip raw L', L, z['T0_2_tip_raw_L'])
        for tag, dt in (('f64origin', np.float64), ('f32origin', np.float32)):
            tip, L = G.resample_tip(pts, ao, arm_origin=c.body_origin_model[af].astype(dt))
            chk.add(f'T0.2 resample_tip body-rel {tag}', tip, z[f'T0_2_tip_rel_{tag}'].reshape(-1, 12, 2), 'resample')
            chk.add(f'T0.2 resample_tip body-rel {tag} L', L, z[f'T0_2_tip_rel_{tag}_L'])
        Lz, kz = G.arc_length_curvature(pts, ao, 'zero')
        Ln, kn = G.arc_length_curvature(pts, ao, 'nan')
        chk.add('T0.3 arc_length (poly_metrics)', Lz, z['T0_3_arc_length_poly_metrics_f32'])
        chk.add('T1.1 curvature (poly_metrics)', kz, z['T1_1_curvature_poly_metrics_f32'])
        chk.add('T0.3 arc_length (arm_metrics)', Ln, z['T0_3_arc_length_arm_metrics_f32'])
        chk.add('T1.1 curvature (arm_metrics)', kn, z['T1_1_curvature_arm_metrics_f32'])
        chk.add('T0.4 to_source points', G.to_source(pts, ao, fo, c.crop), z['T0_4_source_points'])
        chk.add('T0.5 body_relative', G.body_relative(pts, ao, fo, c.body_origin_model), z['T0_5_body_relative_points'])
        ang, rad = G.tip_polar(pts, ao, fo, c.body_origin_model, 'float64')
        chk.add('T1.3 tip angle (f64)', ang, z['T1_3_tip_angle_deg'])
        chk.add('T1.3 tip radius', rad, z['T1_3_tip_radius_px'])
        rm, rs = FR.body_radius(c, 'nan')
        chk.add('T0.6 radius_model', rm, z['T0_6_radius_model'])
        chk.add('T0.6 radius_source', rs, z['T0_6_radius_source'])
        chk.add('T0.6 radius_source (Fig 5 expr, 0 for empty)', FR.body_radius(c, 'zero')[1],
                z['T0_6_radius_source_fig5_expr'])
        chk.add('T2.1 n_arms', FR.n_arms(c), z['T2_1_n_arms'], 'exact')
        chk.add('T2.6 isotropy per frame', FR.tip_isotropy(c), z['T2_6_isotropy_frame_support_centroid'])
    return chk.rows


def _individual(ind: str, subset, index, data_dir: str) -> PolylineContainer:
    segs = [s for s in subset['segments'] if s in {x['segment'] for x in index[ind]}]
    parts = [NP.adapters.from_sleepy_segment(Path(data_dir) / 'sleepy_q_to_sa' / f'{s}.npz') for s in segs]
    return concat(parts, sort_by_time=True)


def check_sleepy_events(events: list[int], ref_root: str, data_dir: str, inputs: str) -> list[dict]:
    chk = Checks('sleepy/fig5')
    subset, index = sleepy_meta(Path(inputs))
    ref = Path(ref_root) / 'sleepy'
    m = np.load(ref / 'fig5_measures.npz')
    lab = np.load(ref / 'fig5_labels.npz')
    lag = np.load(ref / 'T3_1_lag_a_vs_b.npz')
    gh = np.load(ref / 'T1_5_anchored_ghosts.npz')
    lsa = np.load(ref / 'T4_2_T4_3_lsa_fig5.npz')
    ev_json = json.loads((ref / 'fig5_events.json').read_text())
    # LSA calls are recorded in event order; find each event's slice
    lsa_counts = []
    cache: dict[str, PolylineContainer] = {}
    for b, event in enumerate(subset['events']):
        ind = event['individual_id']
        if ind not in cache:
            cache[ind] = _individual(ind, subset, index, data_dir)
        w = cache[ind].time_window(-subset['half_window_s'], subset['half_window_s'], event['onset_ts'])
        lsa_counts.append(int(np.sum(w.arms_per_frame() > 0)) - 1)
    lsa_start = np.r_[0, np.cumsum(lsa_counts)]
    for b in events:
        event = subset['events'][b]
        c_ind = cache[event['individual_id']]
        w = c_ind.time_window(-subset['half_window_s'], subset['half_window_s'], event['onset_ts'])
        sel = m['boundary'] == b
        res = SC.r0_baseline(w, event['onset_ts'])
        ej = ev_json['events'][b]
        if res is None:
            chk.add(f'b{b} dropped', True, 'dropped' in ej, 'exact')
            continue
        chk.add(f'b{b} R0 baseline', res.r0, ej['T0_7_R0_baseline'])
        chk.add(f'b{b} area0 baseline', res.area0, ej['area0_baseline'])
        chk.add(f'b{b} relative_s', w.times() - event['onset_ts'], m['relative_s'][sel])
        chk.add(f'b{b} n_arms', FR.n_arms(w).astype(float), m['n_arms'][sel], 'exact')
        mats = []
        ids = TR.relink(w, assign_hook=lambda mm, r_, c_: mats.append(mm.copy()))
        chk.add(f'b{b} relink labels', ids[ids >= 0], lab['labels'][lab['event_offsets'][b]:lab['event_offsets'][b + 1]],
                'exact')
        lo, hi = lsa_start[b], lsa_start[b + 1]
        ref_vals = lsa['full_values'][lsa['full_offsets'][lo]:lsa['full_offsets'][hi]]
        chk.add(f'b{b} LSA matrices', np.concatenate([x.ravel() for x in mats]) if mats else np.zeros(0), ref_vals)
        near_tie_audit(chk, f'b{b} T4.3 near-tie audit (fused relink)', mats)
        r0 = res.r0
        Ln, kn = G.arc_length_curvature(w.points, w.arm_point_offsets, 'nan')
        base = G.resample_base(w.points, w.arm_point_offsets)
        chk.add(f'b{b} fold_asymmetry', FR.fold_asymmetry(w, r0, 'fig5'), m['fold_asymmetry'][sel])
        chk.add(f'b{b} arm_length_spread', FR.arm_length_spread(w, r0, 'fig5', Ln), m['arm_length_spread'][sel])
        chk.add(f'b{b} arm_curvature', FR.arm_curvature(w, r0, 'fig5', kn), m['arm_curvature'][sel])
        chk.add(f'b{b} tip_motion', KI.tip_motion(w, r0, ids), m['tip_motion'][sel])
        chk.add(f'b{b} arm_deformation lag B', KI.arm_deformation(w, r0, ids, base=base), m['arm_deformation'][sel])
        pre = NP.compute_all(w, 'fig5', onset_s=event['onset_ts'], track='relink')
        chk.add(f'b{b} preset fig5 R0', float(pre['r0']), ej['T0_7_R0_baseline'])
        chk.add(f'b{b} preset fig5 track_id', pre['track_id'], ids, 'exact')
        for k in ('fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'tip_motion', 'arm_deformation'):
            chk.add(f'b{b} preset fig5 {k}', pre[k], m[k][sel])
        lsel = lag['boundary'] == b
        chk.add(f'b{b} arm_deformation lag A', KI.arm_deformation(w, r0, ids, lag='prev_frame_reset', base=base),
                lag['lag_a_prev_frame_reset'][lsel])
        # T2.6 window isotropy over the individual's frames (both fallbacks)
        iso = FR.tip_isotropy(c_ind)
        rel = c_ind.times() - event['onset_ts']
        isoj = next(x for x in ev_json['isotropy'] if x['boundary'] == b)
        ref_nan = np.nan if isoj['summary_nan_fallback'] is None else isoj['summary_nan_fallback']
        chk.add(f'b{b} T2.6 isotropy window (NaN fallback)', FR.window_median(iso, rel, (-300, -60), np.nan), ref_nan)
        chk.add(f'b{b} T2.6 isotropy window (0.0 fallback)', FR.window_median(iso, rel, (-300, -60), 0.0),
                isoj['example_zero_fallback'])
        # T1.5 ghosts (example tracker = the window re-link)
        trel = w.times() - event['onset_ts']
        for wname, win in (('Q', (-180, -120)), ('SA', (120, 180))):
            if f'b{b}_{wname}_points' not in gh.files:
                continue
            g = G.anchored_ghosts(w.points, w.arm_point_offsets, w.frame_arm_offsets, w.crop, w.body_origin_model,
                                  trel, win, labels=ids)
            chk.add(f'b{b} T1.5 ghosts {wname} points', g['points'], gh[f'b{b}_{wname}_points'])
            chk.add(f'b{b} T1.5 ghosts {wname} offsets', g['arm_point_offsets'], gh[f'b{b}_{wname}_offsets'], 'exact')
            chk.add(f'b{b} T1.5 ghosts {wname} labels', g['labels'], gh[f'b{b}_{wname}_labels'], 'exact')
            chk.add(f'b{b} T1.5 ghosts {wname} t_ref', g['t_ref'], ej[f'ghost_{wname}']['t_ref'])
            tp = ej[f'ghost_{wname}']['tip_paths']
            chk.add(f'b{b} T1.5 tip_paths {wname} ids', sorted(int(k) for k in g['tip_paths']),
                    sorted(int(k) for k in tp), 'exact')
            if sorted(g['tip_paths']) == sorted(int(k) for k in tp):
                chk.add(f'b{b} T1.5 tip_paths {wname}', np.concatenate([g['tip_paths'][int(k)] for k in tp]),
                        np.concatenate([np.asarray(v) for v in tp.values()]))
    return chk.rows


def check_sleepy_sequences(names: list[str] | None, ref_root: str, data_dir: str, inputs: str) -> list[dict]:
    chk = Checks('sleepy/fig6d')
    subset, index = sleepy_meta(Path(inputs))
    ref = Path(ref_root) / 'sleepy'
    z = np.load(ref / 'T4_5_T3_3_sequences.npz')
    sj = json.loads((ref / 'T4_5_sequences.json').read_text())
    seqs = []
    for ind in sorted({e['individual_id'] for e in subset['events']} | set(subset.get('individuals', []))):
        segs = [s for s in subset['segments'] if s in {x['segment'] for x in index[ind]}]
        if not segs:
            continue
        parts = [NP.adapters.from_sleepy_segment(Path(data_dir) / 'sleepy_q_to_sa' / f'{s}.npz') for s in segs]
        c_ind = concat(parts, dedupe_time_decimals=1)
        seqs += split_at_gaps(c_ind, max_gap_s=16, min_frames=30)
    chk.add('sequence count', len(seqs), len(sj['sequences']), 'exact')
    for k, (seq, meta) in enumerate(zip(seqs, sj['sequences'])):
        nm = meta['name']
        if names is not None and nm not in names:
            continue
        idx = TR.sequence_index(seq, by='time')
        chk.add(f'{nm} index', idx, z[f'{nm}__frame_index'], 'exact')
        chk.add(f'{nm} n_arms', seq.arms_per_frame(), z[f'{nm}__n_arms'], 'exact')
        ang, rad = G.tip_polar(seq.points, seq.arm_point_offsets, seq.frame_arm_offsets, seq.body_origin_model)
        chk.add(f'{nm} T1.3 tip angle', ang, z[f'{nm}__tip_angle_deg'])
        chk.add(f'{nm} T1.3 tip radius', rad, z[f'{nm}__tip_radius_px'])
        diag = TR.diagnostics(seq, frames=np.arange(seq.n_frames), index=idx, fps=1.0, origin_dtype='float64')
        for key, ids in diag['runs'].items():
            if '_g' in key:
                g = int(key.rsplit('_g', 1)[1])
                fo = seq.frame_arm_offsets
                got = np.concatenate([ids[fo[f]:fo[f + 1]] for f in range(0, seq.n_frames, g)])
            else:
                got = ids
            chk.add(f'{nm} labels {key}', got, z[f'{nm}__labels_{key}'], 'exact')
        for v in TR.VARIANTS:
            chk.add(f'{nm} {v} lifetimes', diag[v]['arm_lifetime_s'], z[f'{nm}__arm_lifetime_s_{v}'], 'exact')
            chk.add(f'{nm} {v} n_ids', diag[v]['n_ids'], meta['per_variant'][v]['n_ids'], 'exact')
            chk.add(f'{nm} {v} counters[all]', _jd(diag[v]['by_state'].get('all'), sort_keys=True),
                    _jd(meta['per_variant'][v]['by_state'].get('all'), sort_keys=True), 'exact')
        chk.add(f'{nm} strides', _jd({str(k): v for k, v in diag['strides'].items()}, sort_keys=True),
                _jd(meta['strides'], sort_keys=True), 'exact')
        # the defaults must pick the Sleepy time index and float64 origin on their own
        dflt = TR.diagnostics(seq, variants={'fused': TR.RELEASED}, strides=())
        chk.add(f'{nm} default-args index_by/origin', f"{dflt['index_by']}/{dflt['origin_dtype']}", 'time/float64',
                'exact')
        chk.add(f'{nm} default-args fused counters[all]', _jd(dflt['fused']['by_state'].get('all'), sort_keys=True),
                _jd(meta['per_variant']['fused']['by_state'].get('all'), sort_keys=True), 'exact')
        chk.add(f'{nm} default-args duration_s', dflt['duration_s'], diag['duration_s'], 'exact')
        pre = NP.compute_all(seq, 'fig6_sleepy', track='relink')
        chk.add(f'{nm} preset fig6_sleepy tip_angle', pre['tip_angle_deg'], z[f'{nm}__tip_angle_deg'])
        chk.add(f'{nm} preset fig6_sleepy track_id', pre['track_id'], z[f'{nm}__labels_fused_fwd'], 'exact')
    return chk.rows


# ------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', required=True, type=Path, help='parity store ref/ directory')
    ap.add_argument('--inputs', type=Path, default=None, help='parity store inputs/ (default: <ref>/../inputs)')
    ap.add_argument('--data-dir', type=Path, default=None, help='released re-trace root (paper_retrace_f39dc82b)')
    ap.add_argument('--sources', type=Path, default=None,
                    help='store entry -> released clip map (default: <inputs>/sources.json; not committed)')
    ap.add_argument('--only', default='micro,flip,sleepy')
    ap.add_argument('--quick', action='store_true', help='skip the 10-pass Fig 6D diagnostics and most events')
    ap.add_argument('--jobs', type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument('--oracle', action='store_true', help='also check the _reference ports on the flipping inputs')
    ap.add_argument('--json', type=Path, default=None, help='write every row as JSON')
    ap.add_argument('--verbose', action='store_true', help='print every row, not only failures and a summary')
    args = ap.parse_args(argv)
    inputs = args.inputs or args.ref.parent / 'inputs'
    sources = load_sources(args.sources or inputs / 'sources.json')
    only = set(args.only.split(','))
    data = str(args.data_dir) if args.data_dir else None
    jobs = []
    if 'micro' in only:
        jobs.append((check_flip, ('micro', str(args.ref), data, args.quick, sources)))
        if args.oracle:
            jobs.append((check_oracle, ('micro', str(args.ref), args.quick)))
    if 'flip' in only:
        for name in sorted(p.name for p in args.ref.glob('flip_*') if (p / 'input_container.npz').exists()):
            jobs.append((check_flip, (name, str(args.ref), data, args.quick, sources)))
            if args.oracle:
                jobs.append((check_oracle, (name, str(args.ref), args.quick)))
    if 'sleepy' in only:
        if data is None:
            print('sleepy checks need --data-dir (segments are read with the adapter)', file=sys.stderr)
        else:
            subset, _ = sleepy_meta(inputs)
            segs = subset['segments']
            for i in range(0, len(segs), 6):
                jobs.append((check_sleepy_segments, (segs[i:i + 6], str(args.ref), data)))
            evs = list(range(len(subset['events'])))
            if args.quick:
                evs = evs[:3]
            for i in range(0, len(evs), 3):
                jobs.append((check_sleepy_events, (evs[i:i + 3], str(args.ref), data, str(inputs))))
            sj = json.loads((args.ref / 'sleepy' / 'T4_5_sequences.json').read_text())
            names = [s['name'] for s in sj['sequences']]
            if args.quick:
                names = names[:1]
            for nm in names:
                jobs.append((check_sleepy_sequences, ([nm], str(args.ref), data, str(inputs))))
    t0 = time.perf_counter()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(fn, *a) for fn, a in jobs]
        for f in futs:
            rows.extend(f.result())
    wall = time.perf_counter() - t0
    # summary table: representation x scope
    by = defaultdict(lambda: {'n': 0, 'fail': 0, 'bit': 0, 'worst': ''})
    for r in rows:
        if r['kind'] == 'info':
            continue
        key = (r['scope'].split('/')[0] if not r['scope'].startswith('flip_') else 'flip(7)', _rep_key(r['rep']))
        s = by[key]
        s['n'] += 1
        s['bit'] += bool(r.get('bit'))
        if not r['ok']:
            s['fail'] += 1
            s['worst'] = f"{r['scope']}: {r['rep']}: {r['detail']}"
        elif not r.get('bit') and r['kind'] != 'exact' and not s['worst']:
            s['worst'] = f"(within tol) {r['detail']}"
    print(f"{'scope':<10} {'representation':<46} {'checks':>6} {'bit':>5} {'fail':>5}  note")
    for (scope, rep), s in sorted(by.items()):
        status = 'FAIL' if s['fail'] else 'pass'
        print(f"{scope:<10} {rep:<46} {s['n']:>6} {s['bit']:>5} {s['fail']:>5}  {status} {s['worst']}")
    if args.verbose:
        for r in rows:
            print(r)
    n_fail = sum(not r['ok'] for r in rows)
    n = sum(r['kind'] != 'info' for r in rows)
    print(f'\n{n} checks, {n_fail} failed, {sum(bool(r.get("bit")) for r in rows if r["kind"] != "info")} bit-identical;'
          f' wall {wall:.1f} s')
    for r in rows:
        if not r['ok']:
            print('FAIL', r['scope'], r['rep'], r['detail'])
    audit = [r for r in rows if r.get('audit') == 'near_tie']
    if audit:
        n_lsa = sum(r['n_lsa'] for r in audit)
        n_near = sum(r['n_near'] for r in audit)
        mn = min(r['min_margin'] for r in audit)
        print(f'near-tie audit: {n_lsa} LSA calls, {n_near} with decision margin < {NEAR_TIE:g} '
              f'(reported, not failed), smallest margin {mn:.3g}')
        for r in audit:
            if r['n_near']:
                print('NEAR-TIE', r['scope'], r['rep'], r['detail'], 'calls', r['near_calls'][:10])
    if args.json:
        args.json.write_text(_jd(rows, indent=1))
    return 1 if n_fail else 0


def _rep_key(rep: str) -> str:
    import re
    rep = re.sub(r'\b(ev|bout|b)\d+\b', '', rep)
    rep = re.sub(r'sleepy_\d+', '', rep)
    rep = re.sub(r'\[[^\]]*\]', '', rep)
    return ' '.join(rep.split())[:46]


if __name__ == '__main__':
    sys.exit(main())
