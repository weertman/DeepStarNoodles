#!/usr/bin/env python3
"""Benchmark harness for noodle_polylines: per-kernel and end-to-end throughput, projected to a camera-year.

Inputs (any combination):
  micro       the committed micro example (181 frames, 1,226 arms)
  subset      --data-dir: N flipping clips + M Sleepy segments read through the adapters (not committed)
  synthetic   the micro example (or the subset) tiled with jitter to --synthetic-arms arms, fixed seed

    OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs micro,subset,synthetic \
        --data-dir RUNS/paper_retrace_f39dc82b --synthetic-arms 1000000 --repeats 5 --json bench.json

Each kernel is timed in its own single-threaded process (BLAS threads pinned to 1) as the best of
``--repeats`` runs (``--synthetic-repeats`` for the synthetic input); wall, CPU and peak RSS are reported.
Metrics: arms/s and frames/s of the input set; the projection multiplies the per-arm cost by
3e8 arms (one camera-year at ~1 fps) and reports CPU-hours.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import cProfile
import csv
import io
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'CORE'))

import numpy as np  # noqa: E402

import noodle_polylines as NP  # noqa: E402
from noodle_polylines import frames as FR, geometry as G, kinematics as KI, scale as SC, tracking as TR  # noqa: E402
from noodle_polylines.container import PolylineContainer, _offsets  # noqa: E402

EXAMPLE = ROOT / 'examples' / 'noodle_polylines' / 'data' / 'flip_example_01.npz'
CAMERA_YEAR_ARMS = 3e8
R0 = 50.0


# ------------------------------------------------------------------------------------------ inputs
def load_micro() -> list[PolylineContainer]:
    return [PolylineContainer.load(EXAMPLE)]


def load_subset(data_dir: Path, n_flip: int, n_sleepy: int) -> list[PolylineContainer]:
    out = []
    rows = list(csv.DictReader((data_dir / 'flipping' / 'manifest.csv').open()))
    for r in sorted(rows, key=lambda r: r['clip_id'])[:n_flip]:
        out.append(NP.adapters.from_flipping_clip(data_dir / 'flipping' / 'clips' / r['clip_id']))
    segs = sorted((data_dir / 'sleepy_q_to_sa').glob('seg_*.npz'))[:n_sleepy]
    for s in segs:
        c = NP.adapters.from_sleepy_segment(s)
        if c.n_arms:
            out.append(c)
    return out


def synthetic(source: list[PolylineContainer], target_arms: int, seed: int = 20261102,
              jitter_px: float = 0.25) -> list[PolylineContainer]:
    """Tile the source containers frame-wise (time-shifted, point jitter, fresh track IDs) to ``target_arms``.

    Returns one long sequence per source container, so per-sequence costs (tracker state, windows) scale
    like long recordings. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    total = sum(c.n_arms for c in source)
    reps = int(np.ceil(target_arms / max(total, 1)))
    out = []
    for c in source:
        span = float(c.times()[-1] - c.times()[0]) + 1.0 / (c.fps or 1.0)
        F = c.n_frames
        tid = c.track_id if c.has_track_id() else TR.relink(c)
        tid_span = int(tid.max()) + 1 if len(tid) else 0
        pts = np.concatenate([c.points + rng.normal(0, jitter_px, c.points.shape).astype(np.float32)
                              for _ in range(reps)])
        ident = np.concatenate([c.identity + rng.normal(0, 0.01, c.identity.shape).astype(np.float32)
                                for _ in range(reps)])
        tile = lambda a: np.concatenate([a] * reps)  # noqa: E731
        kw = {'points': pts, 'arm_point_offsets': _offsets(tile(c.points_per_arm())),
              'frame_arm_offsets': _offsets(tile(c.arms_per_frame())),
              'frame_index': np.arange(F * reps, dtype=np.int64),
              'time_s': np.concatenate([c.times() + k * span for k in range(reps)]),
              'identity': ident, 'track_id': np.concatenate([tid + k * tid_span for k in range(reps)])}
        for name in ('crop', 'body_origin_model', 'support_area_model_px', 'support_centroid_model', 'has_star'):
            kw[name] = tile(getattr(c, name))
        out.append(PolylineContainer(**kw, meta=dict(c.meta)).validate())
    return out


def describe(cs: list[PolylineContainer]) -> dict:
    return {'containers': len(cs), 'arms': int(sum(c.n_arms for c in cs)), 'frames': int(sum(c.n_frames for c in cs)),
            'points': int(sum(c.n_points for c in cs))}


# ------------------------------------------------------------------------------------------ kernels
def _ids(c):
    return c.track_id if c.has_track_id() else TR.relink(c)


def _fig5(c):
    return NP.compute_all(c, 'fig5', r0=R0, track='stored' if c.has_track_id() else 'relink')


KERNELS = {
    'T0.2 resample_base': lambda c: G.resample_base(c.points, c.arm_point_offsets),
    'T0.2 resample_tip (body-rel)': lambda c: TR.tracker_inputs(c),
    'T0.3+T1.1 arc_length_curvature': lambda c: G.arc_length_curvature(c.points, c.arm_point_offsets),
    'T0.4 to_source': lambda c: G.to_source(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.crop),
    'T1.3 tip_polar': lambda c: G.tip_polar(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.body_origin_model),
    'T2.2 fold_asymmetry (fig5)': lambda c: FR.fold_asymmetry(c, R0, 'fig5'),
    'T2.3 arm_length_spread (fig5)': lambda c: FR.arm_length_spread(c, R0, 'fig5'),
    'T2.4 arm_curvature (fig5)': lambda c: FR.arm_curvature(c, R0, 'fig5'),
    'T2.6 tip_isotropy': lambda c: FR.tip_isotropy(c),
    'T0.7 R0 policies': lambda c: (SC.r0_per_frame(c), SC.r0_outside_windows(c, [(c.times()[0] + 10, c.times()[0] + 40)])),
    'T4.4 relink (1 pass)': lambda c: TR.relink(c),
    'T3.1 arm_deformation lag B': lambda c: KI.arm_deformation(c, R0, _ids(c)),
    'T3.1 arm_deformation lag A': lambda c: KI.arm_deformation(c, R0, _ids(c), lag='prev_frame_reset'),
    'T3.2 tip_motion': lambda c: KI.tip_motion(c, R0, _ids(c)),
    'T4.5 diagnostics (10 passes)': lambda c: TR.diagnostics(c, frames=np.arange(c.n_frames), fps=c.fps or 1.0),
    'E2E compute_all fig5 (stored IDs)': _fig5,
    'E2E compute_all fig5 + relink': lambda c: NP.compute_all(c, 'fig5', r0=R0, track='relink'),
}
EXPENSIVE = {'T4.5 diagnostics (10 passes)'}
TRACKED = {'T3.1 arm_deformation lag B', 'T3.1 arm_deformation lag A', 'T3.2 tip_motion'}


def _build(spec: dict) -> list[PolylineContainer]:
    if spec['input'] == 'micro':
        return load_micro()
    if spec['input'] == 'subset':
        return load_subset(Path(spec['data_dir']), spec['n_flip'], spec['n_sleepy'])
    src = load_subset(Path(spec['data_dir']), spec['n_flip'], spec['n_sleepy']) if spec.get('synthetic_source') == 'subset' \
        else load_micro()
    return synthetic(src, spec['synthetic_arms'], spec['seed'])


def run_kernel(spec: dict, kernel: str, repeats: int) -> dict:
    """Time one kernel on one input set in this (fresh, single-threaded) process."""
    t_load = time.perf_counter()
    cs = _build(spec)
    load_wall = time.perf_counter() - t_load
    if kernel in TRACKED:          # IDs are an input of the tracked kernels, not part of their cost
        cs = [c if c.has_track_id() else c.with_fields(track_id=TR.relink(c)) for c in cs]
    fn = KERNELS[kernel]
    walls, cpus = [], []
    for _ in range(repeats):
        t0, c0 = time.perf_counter(), time.process_time()
        for c in cs:
            fn(c)
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
    d = describe(cs)
    best = min(walls)
    return {'input': spec['input'], 'kernel': kernel, 'repeats': repeats, 'wall_s_best': best, 'wall_s_all': walls,
            'cpu_s_best': min(cpus), 'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            'arms_per_s': d['arms'] / best, 'frames_per_s': d['frames'] / best, 'us_per_arm': 1e6 * best / d['arms'],
            'load_wall_s': load_wall, **d}


def run_io(spec: dict, repeats: int) -> list[dict]:
    """Adapter / container I/O timings."""
    out = []
    if spec['input'] == 'micro':
        fn = lambda: load_micro()  # noqa: E731
        name = 'IO container npz load (allow_pickle=False)'
    elif spec['input'] == 'subset':
        fn = lambda: load_subset(Path(spec['data_dir']), spec['n_flip'], spec['n_sleepy'])  # noqa: E731
        name = 'IO adapters (flipping CSV+pickle, Sleepy npz)'
    else:
        return out
    walls = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        cs = fn()
        walls.append(time.perf_counter() - t0)
    d = describe(cs)
    out.append({'input': spec['input'], 'kernel': name, 'repeats': repeats, 'wall_s_best': min(walls),
                'wall_s_all': walls, 'cpu_s_best': None, 'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                'arms_per_s': d['arms'] / min(walls), 'frames_per_s': d['frames'] / min(walls),
                'us_per_arm': 1e6 * min(walls) / d['arms'], **d})
    return out


def profile(spec: dict, top: int = 25) -> str:
    cs = _build(spec)
    pr = cProfile.Profile()
    pr.enable()
    for c in cs:
        _fig5(c)
        TR.relink(c)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(top)
    return s.getvalue()


def projection(results: list[dict]) -> dict:
    """CPU-hours per camera-year (3e8 arms) per kernel, from the largest input measured for that kernel."""
    best: dict[str, dict] = {}
    for r in results:
        if r['kernel'] not in best or r['arms'] > best[r['kernel']]['arms']:
            best[r['kernel']] = r
    return {k: {'from_input': r['input'], 'us_per_arm': r['us_per_arm'],
                'cpu_h_per_camera_year': r['us_per_arm'] * 1e-6 * CAMERA_YEAR_ARMS / 3600}
            for k, r in best.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inputs', default='micro')
    ap.add_argument('--data-dir', type=Path, default=None)
    ap.add_argument('--n-flip', type=int, default=7)
    ap.add_argument('--n-sleepy', type=int, default=36)
    ap.add_argument('--synthetic-arms', type=int, default=1_000_000)
    ap.add_argument('--synthetic-source', choices=('micro', 'subset'), default='micro')
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--synthetic-repeats', type=int, default=None, help='default: --repeats')
    ap.add_argument('--kernels', default=None, help='comma-separated substrings to select kernels')
    ap.add_argument('--expensive-repeats', type=int, default=1, help='repeats of the 10-pass T4.5 diagnostics')
    ap.add_argument('--skip-expensive-on', default='synthetic', help='inputs on which T4.5 diagnostics is skipped')
    ap.add_argument('--jobs', type=int, default=min(32, os.cpu_count() or 1), help='kernel processes in parallel')
    ap.add_argument('--profile', type=Path, default=None, help='write a cProfile top-25 (fig5 E2E + relink) here')
    ap.add_argument('--json', type=Path, default=None)
    ap.add_argument('--label', default='')
    args = ap.parse_args(argv)
    inputs = args.inputs.split(',')
    skip = set(args.skip_expensive_on.split(',')) if args.skip_expensive_on else set()
    specs = []
    for name in inputs:
        if name in ('subset', 'synthetic') and args.data_dir is None and not (name == 'synthetic'
                                                                              and args.synthetic_source == 'micro'):
            ap.error(f'{name} needs --data-dir')
        specs.append({'input': name, 'data_dir': str(args.data_dir) if args.data_dir else None, 'n_flip': args.n_flip,
                      'n_sleepy': args.n_sleepy, 'synthetic_arms': args.synthetic_arms,
                      'synthetic_source': args.synthetic_source, 'seed': args.seed})
    kernels = [k for k in KERNELS if not args.kernels or any(s in k for s in args.kernels.split(','))]
    jobs = []
    for spec in specs:
        reps = (args.synthetic_repeats or args.repeats) if spec['input'] == 'synthetic' else args.repeats
        for k in kernels:
            if k in EXPENSIVE and spec['input'] in skip:
                continue
            jobs.append((spec, k, min(reps, args.expensive_repeats) if k in EXPENSIVE else reps))
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_kernel, *j) for j in jobs]
        io_futs = [ex.submit(run_io, spec, args.repeats) for spec in specs]
        for f in futs:
            r = f.result()
            results.append(r)
            print(f"{r['input']:<10} {r['kernel']:<40} {r['us_per_arm']:10.2f} us/arm {r['arms_per_s']:12.0f} arms/s "
                  f"{r['frames_per_s']:10.0f} frames/s  best {r['wall_s_best']:8.3f} s  RSS {r['peak_rss_mb']:7.0f} MB",
                  flush=True)
        for f in io_futs:
            for r in f.result():
                results.append(r)
                print(f"{r['input']:<10} {r['kernel']:<40} {r['us_per_arm']:10.2f} us/arm  best {r['wall_s_best']:.3f} s")
    proj = projection(results)
    print('\nCamera-year projection (3e8 arms, single core):')
    for k, v in proj.items():
        print(f"  {k:<40} {v['us_per_arm']:10.2f} us/arm -> {v['cpu_h_per_camera_year']:10.1f} CPU-h ({v['from_input']})")
    report = {'label': args.label, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'env': {'python': platform.python_version(), 'numpy': np.__version__,
                      'scipy': __import__('scipy').__version__, 'cpu': platform.processor() or platform.machine(),
                      'cores': os.cpu_count(), 'omp_threads': os.environ['OMP_NUM_THREADS']},
              'inputs': {s['input']: s for s in specs}, 'results': results, 'projection': proj,
              'total_wall_s': time.perf_counter() - t0}
    if args.profile:
        spec = next(s for s in specs if s['input'] == ('subset' if 'subset' in inputs else inputs[0]))
        txt = profile(spec)
        args.profile.write_text(txt)
        report['profile_input'] = spec['input']
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
