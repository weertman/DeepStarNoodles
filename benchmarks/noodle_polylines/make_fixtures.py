#!/usr/bin/env python3
"""Write the small golden test fixtures (tests/fixtures/noodle_polylines/) from the parity store's micro input.

    python benchmarks/noodle_polylines/make_fixtures.py --ref <store>/ref

Only the micro example (flip_example_01: 181 frames, 1,226 arms) is used; every array comes from the
figure-script code as captured in the store. Total size is kept well under 1 MB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'tests' / 'fixtures' / 'noodle_polylines'
N_SHAPES = 48      # arms whose full 12-point resamples are kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', type=Path, required=True)
    d = ap.parse_args().ref / 'micro'
    OUT.mkdir(parents=True, exist_ok=True)
    a = np.load(d / 'T0_T1_arms.npz')
    t3 = np.load(d / 'T3_3_tip_angle_track.npz')
    rl = np.load(d / 'T4_relink.npz')
    np.savez_compressed(
        OUT / 'micro_arms.npz',
        T0_2_base=a['T0_2_base'][:N_SHAPES], T0_2_tip_raw=a['T0_2_tip_raw'][:N_SHAPES],
        T0_2_tip_raw_L=a['T0_2_tip_raw_L'][:N_SHAPES], T0_2_tip_rel=a['T0_2_tip_rel_f64origin'][:N_SHAPES],
        T0_2_tip_rel_L=a['T0_2_tip_rel_f64origin_L'][:N_SHAPES],
        T0_3_arc_length_zero=a['T0_3_arc_length_poly_metrics_f32'], T1_1_curvature_zero=a['T1_1_curvature_poly_metrics_f32'],
        T0_3_arc_length_nan=a['T0_3_arc_length_arm_metrics_f32'], T1_1_curvature_nan=a['T1_1_curvature_arm_metrics_f32'],
        T0_4_tip_source=a['T0_4_source_tip'], T1_2_norm_length=a['T1_2_norm_length'],
        T1_3_tip_angle_f32origin=t3['tip_angle_deg'], T1_3_tip_angle_f32=t3['tip_angle_deg_fig6_flip_copy_f32'],
        T1_3_tip_radius=t3['tip_radius_px'],
        T4_4_relink=rl['T4_4_relink_ids'], stored_track_id=rl['stored_track_id'],
        T4_4_stats_fused=t3['track_id_stats_fused'],
        lsa_first_values=rl['lsa_full_values'][:rl['lsa_full_offsets'][5]],
        lsa_first_shape=rl['lsa_full_shape'][:5])
    f = np.load(d / 'T2_frames.npz')
    k = np.load(d / 'fig4e_kinematics.npz')
    t = np.load(d / 'fig4d_tree_features.npz')
    frames = {f'T2_{key}': f[key] for key in ('T2_1_n_arms', 'T0_6_radius_model', 'T0_6_radius_source',
                                              'T2_6_isotropy_frame_support_centroid', 'T2_5_mean_normalized_arm_length',
                                              'label_state_code')}
    frames.update({f'fig4e_{key}': k[key] for key in k.files})
    frames.update({f'fig4d_{key}': t[key] for key in t.files if key != 'side_index'})
    for tag in ('event0', 'event1', 'event2'):
        z = np.load(d / f'fig5_measures_{tag}.npz')
        for key in ('fold_asymmetry', 'arm_length_spread', 'arm_curvature', 'tip_motion', 'arm_deformation'):
            frames[f'fig5_{tag}_{key}'] = z[key]
    s = np.load(d / 'T4_5_stats_runs.npz')
    frames.update({f'T4_5_{key}': s[key] for key in s.files})
    np.savez_compressed(OUT / 'micro_frames.npz', **frames)
    kj = json.loads((d / 'fig4e_kinematics.json').read_text())
    tj = json.loads((d / 'fig4d_tree_features.json').read_text())
    fj = json.loads((d / 'fig5_measures.json').read_text())
    dj = json.loads((d / 'T4_5_diagnostics.json').read_text())
    sj = json.loads((d / 'T4_2_T4_3_fig6_steps.json').read_text())
    strip = lambda e: {k_: v for k_, v in e.items() if k_ != 'video_id'}   # noqa: E731
    meta = {
        'source': 'parity store ref/micro (figure-script outputs on flip_example_01)',
        'fig4e_events': [strip(e) for e in kj['events']],
        'fig4e_r0': [{'r0_source_px': r['r0_source_px'], 'pre_flip_frames': r['pre_flip_frames']}
                     for r in kj['r0_by_event_T0_7_pre_onset']],
        'fig4d_bouts': [strip(b) for b in tj['bouts']], 'fig4d_r0': tj['T0_7_R0_outside_windows'],
        'fig4d_area0': tj['area0_outside_windows'],
        'fig5_r0': {tag: run['r0'] for tag, run in fj['runs'].items()},
        'T4_5': {key: dj[key] for key in ('per_variant', 'strides_result', 'rates_fused_all', 'pooled_fused',
                                          'n_frames', 'n_arms', 'median_arms', 'duration_s')},
        'fig6_steps': {key: v for key, v in sj.items() if key.startswith('fig6_')},
    }
    (OUT / 'micro_meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f'{OUT}: {total / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
