#!/usr/bin/env python3
"""Compute every polyline representation of the paper figures on the micro example.

    python examples/noodle_polylines/run_example.py [--out DIR]

The example (data/flip_example_01.npz + .json) is 90 s of one sea star righting itself twice
(181 frames at 2 Hz, 1,226 traced arms). Two tracked paths are shown: the stored ``track_id``
(warm-started over the whole original recording) and a cold-start re-link of the excerpt alone.
Needs only numpy and scipy.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'CORE'))

import noodle_polylines as NP                                                      # noqa: E402
from noodle_polylines import frames as FR, geometry as G, kinematics as KI         # noqa: E402
from noodle_polylines import scale as SC, tracking as TR                          # noqa: E402
from noodle_polylines.pipeline import save_outputs                                # noqa: E402


def best_mapping_agreement(a: np.ndarray, b: np.ndarray) -> int:
    """Arms whose ID in ``b`` is the most common partner of their ID in ``a``."""
    return int(sum(np.bincount(b[a == x]).max() for x in np.unique(a)))


def nanstats(x) -> str:
    x = np.asarray(x, float)
    f = x[np.isfinite(x)]
    return f'median {np.median(f):8.4f}  [{f.min():.4f}, {f.max():.4f}]  finite {len(f)}/{len(x)}' if len(f) else 'all NaN'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=HERE / 'data' / 'flip_example_01.npz')
    ap.add_argument('--out', type=Path, default=None, help='write every output array as .npz files here')
    args = ap.parse_args()

    # 1. load + validate (pickle-free; the sibling .json supplies fps, bouts and configs)
    c = NP.PolylineContainer.load(args.data)
    bouts = c.meta['bouts']
    t = c.times()
    print(f'{c.meta["sequence_id"]}: {c.n_frames} frames ({t[0]:.1f}-{t[-1]:.1f} s at {c.fps:g} Hz), '
          f'{c.n_arms} arms, {c.n_points} points; licence {c.meta["license"]}')
    pts, ao, fo = c.points, c.arm_point_offsets, c.frame_arm_offsets
    outputs: dict[str, dict] = {}

    # 2. T0 / T1 per arm
    base = G.resample_base(pts, ao)                                         # T0.2 (A,12,2) base first
    tip, tip_len = G.resample_tip(pts, ao)                                  # T0.2 (A,12,2) tip first
    L, kappa = G.arc_length_curvature(pts, ao, 'zero')                       # T0.3, T1.1
    r_model, r_src = FR.body_radius(c)                                       # T0.6
    angle, radius = G.tip_polar(pts, ao, fo, c.body_origin_model, 'float32')  # T1.3 (Fig 6 stats path)
    outputs['per_arm'] = {
        'resample_base': base, 'resample_tip': tip, 'arc_length_model_px': L, 'curvature_rad_per_model_px': kappa,
        'tip_source_px': G.tips_source(pts, ao, fo, c.crop), 'norm_length_body_radii': G.norm_length(L, fo, r_model),
        'tip_angle_deg': angle, 'tip_radius_px': radius,
        'body_path_body_radii': G.body_path(pts, ao, fo, c.body_origin_model, r_model)}
    print('\nT0/T1 per arm')
    print(f'  T0.3 arc length (model px)        {nanstats(L)}')
    print(f'  T1.1 curvature (rad / model px)   {nanstats(kappa)}')
    print(f'  T1.2 length (body radii)          {nanstats(outputs["per_arm"]["norm_length_body_radii"])}')
    print(f'  T1.3 tip radius (crop px)         {nanstats(radius)}')
    print(f'  T0.2 tip-first vs base-first max |diff| {np.abs(tip - base[:, ::-1]).max():.2e} px')

    # 3. T2 per frame (untracked) with the Fig 4E pre-onset R0 of bout 1
    r0 = SC.r0_pre_onset(c, bouts[0])
    print(f'\nT0.7 R0 pre-onset bout 1: {r0.r0:.3f} source px from {r0.n_frames} frames; '
          f'outside windows: {SC.r0_outside_windows(c, [(b["upside_down_start_s"], b["right_side_up_start_s"]) for b in bouts]).r0:.3f}')
    per_frame = {'n_arms': FR.n_arms(c), 'radius_model_px': r_model, 'radius_source_px': r_src,
                 'tip_isotropy': FR.tip_isotropy(c),
                 'fold_asymmetry_R0': FR.fold_asymmetry(c, r0.r0, 'fig4e'),
                 'arm_length_spread_R0': FR.arm_length_spread(c, r0.r0, 'fig4e', L),
                 'arm_curvature_rad_per_R0': FR.arm_curvature(c, r0.r0, 'fig4e', kappa),
                 **FR.frame_lists_4b(c, L, kappa)}
    print('T2 per frame')
    for k in ('n_arms', 'fold_asymmetry_R0', 'arm_length_spread_R0', 'arm_curvature_rad_per_R0', 'tip_isotropy'):
        print(f'  {k:<30} {nanstats(per_frame[k])}')

    # 4. T4 association: stored IDs vs a cold-start re-link of the excerpt
    stored, source = TR.track_ids(c)
    relinked = TR.relink(c)
    agree = best_mapping_agreement(stored, relinked)
    print(f'\nT4.4 track IDs: {source} {len(np.unique(stored))} IDs; cold re-link {len(np.unique(relinked))} IDs; '
          f'agreement under the best mapping {agree}/{c.n_arms} ({agree / c.n_arms:.1%})')

    # 5. T3 tracked kinematics for both bouts and both ID sets (lag A = Fig 4, lag B = Fig 5)
    print('T3 tracked kinematics (median over each bout window)')
    for i, b in enumerate(bouts):
        r = SC.r0_pre_onset(c, b).r0
        win = (t >= b['first_time_s']) & (t <= b['last_time_s'])
        for tag, ids in (('stored', stored), ('relink', relinked)):
            lag_a = KI.arm_deformation(c, r, ids, lag='prev_frame_reset', base=base)
            lag_b = KI.arm_deformation(c, r, ids, lag='last_arm_frame', base=base)
            motion = KI.tip_motion(c, r, ids)
            per_frame[f'bout{i}_{tag}_deformation_lagA'] = lag_a
            per_frame[f'bout{i}_{tag}_deformation_lagB'] = lag_b
            per_frame[f'bout{i}_{tag}_tip_motion'] = motion
            print(f'  bout {i + 1} [{tag:>6}] deformation A {np.nanmedian(lag_a[win]):.4f}  '
                  f'B {np.nanmedian(lag_b[win]):.4f}  tip motion {np.nanmedian(motion[win]):.4f} R0/s')
    outputs['per_frame'] = per_frame

    # 6. T3.3 tip angles per track in the Fig 6C window (360-402 s)
    track = KI.tip_angle_track(c, stored, origin_dtype='float32')
    w6 = (track['time_s'] >= 360) & (track['time_s'] <= 402)
    print(f'\nT3.3 Fig 6C window: {w6.sum()} arm tips on {len(np.unique(track["track_id"][w6]))} tracks')
    outputs['tip_angle_track'] = track

    # 7. T4.2/T4.3: the cost matrix solved at t = 372.0 s (Fig 6C step 371.5 -> 372.0)
    star = np.flatnonzero(c.star_mask())
    k = int(np.flatnonzero(t == 372.0)[0])
    step = TR.step_costs(c, k, origin_dtype='float32', frames=star, track_order='insertion')
    print(f'T4.2 cost matrix at {t[k]:.1f} s: {step["cost"].shape[0]} arms x {step["cost"].shape[1]} tracks, '
          f'{int(np.isfinite(step["cost"]).sum())} pairs under the gate, {len(step["assignment"])} assigned')
    outputs['step_372s'] = {kk: v for kk, v in step.items()}

    # 8. T4.5 Fig 6D diagnostics on the excerpt (lifetimes are capped by the 90-s excerpt)
    diag = TR.diagnostics(c)
    rates = TR.rates([diag['fused']['by_state']['all']])
    print(f'T4.5 diagnostics: continuation {rates["continuation"]:.3f}, forward-reverse agreement '
          f'{rates["forward_reverse_agreement"]:.3f}, jumps {rates["jumps_per_1000_links"]:.1f}/1,000 links, '
          f'new-ID rate {rates["new_id_rate"]:.4f}, stride-2 precision '
          f'{diag["strides"][2]["agree"] / max(diag["strides"][2]["sub_links"], 1):.3f}')

    # 9. the same through presets (one call per panel)
    print('\nPresets (compute_all):')
    for name, kw in (('fig4b', {}), ('fig4d', {'windows': [(b['upside_down_start_s'], b['right_side_up_start_s'])
                                                           for b in bouts]}),
                     ('fig4e', {'bout': bouts[0]}), ('fig5', {'r0': r0.r0}), ('fig6_flipping', {})):
        out = NP.compute_all(c, name, **kw)
        outputs[f'preset_{name}'] = out
        print(f'  {name:<14} {len(out)} arrays: {", ".join(sorted(out)[:8])}, ...')

    if args.out:
        for group, arrays in outputs.items():
            save_outputs(args.out / f'{group}.npz', arrays)
        print(f'\nwrote {len(outputs)} files to {args.out}')


if __name__ == '__main__':
    main()
