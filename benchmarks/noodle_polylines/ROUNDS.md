# noodle_polylines optimisation rounds

Every round must meet two conditions:

- `parity_check.py` passes, unchanged, against the parity reference store;
- `tests/test_noodle_polylines*.py` passes.

Each round also reports `bench.py` timings against the v0 baseline. Timings are single
threaded (`OMP_NUM_THREADS=1`), with each kernel in its own process. The camera-year
projection uses 3e8 arms at about 1 fps.

## Baseline (v0)

v0 is a straight port. The public functions loop over arms and frames and call the
faithful kernels in `_reference.py`.

### Run

| | |
|---|---|
| Date | 2026-11-02 |
| CPU | Intel Xeon W-3365, 2.7 GHz, 64 cores |
| Software | Python 3.12.13, numpy 2.4.3, scipy 1.17.1 |

```
python benchmarks/noodle_polylines/bench.py --inputs micro,subset,synthetic \
    --data-dir RUNS/paper_retrace_f39dc82b --synthetic-arms 1000000 \
    --repeats 5 --synthetic-repeats 1 --jobs 40 --profile profile_v0.txt --json bench_v0.json
```

- **Repeats.** Each value is the best of 5 runs on micro and subset. The synthetic input
  and the 10-pass T4.5 diagnostics were run once: at v0 speed, best of 5 on 1e6 arms
  takes hours. Later rounds should use `--synthetic-repeats 5`.
- **Micro is noisy.** The micro values (about 0.1 s per kernel) are inflated by start-up
  contention among the 40 kernel processes. The same kernels run alone take 30 µs/arm
  (resample_base), 495 µs/arm (relink) and 132 µs/arm (fig5 end to end).
- **Projection input.** Use the subset and synthetic columns for the projection.

### Inputs

| Input | Arms | Frames | Contents |
|---|---:|---:|---|
| micro | 1,226 | 181 | the committed example |
| subset | 173,434 | 21,473 | 7 flipping clips plus 36 Sleepy segments, read with the adapters |
| synthetic | 1,000,416 | 147,696 | micro tiled 816×, time-shifted, 0.25-px point jitter, seed 20261102 |

### Per-kernel timings

| Kernel | Micro µs/arm | Subset µs/arm | Synthetic µs/arm | Peak RSS MB | CPU-h per camera-year |
|---|---:|---:|---:|---:|---:|
| T0.2 resample_base | 56.13 | 31.01 | 31.61 | 520 | 2.6 |
| T0.2 resample_tip (body-relative) | 82.23 | 71.47 | 72.04 | 482 | 6.0 |
| T0.3 + T1.1 arc_length_curvature | 60.98 | 30.50 | 30.66 | 482 | 2.6 |
| T0.4 to_source | 2.36 | 0.98 | 1.28 | 2038 | 0.1 |
| T1.3 tip_polar | 0.16 | 0.10 | 0.18 | 482 | 0.0 |
| T2.2 fold_asymmetry (fig5) | 3.54 | 1.63 | 2.44 | 482 | 0.2 |
| T2.3 arm_length_spread (fig5, includes T0.3) | 64.76 | 33.29 | 35.46 | 483 | 3.0 |
| T2.4 arm_curvature (fig5, includes T1.1) | 58.47 | 30.34 | 30.87 | 482 | 2.6 |
| T2.6 tip_isotropy | 10.34 | 5.61 | 7.53 | 482 | 0.6 |
| T0.7 R0 policies | 0.13 | 0.04 | 0.01 | 483 | 0.0 |
| **T4.4 relink (1 tracker pass)** | 569.20 | 660.32 | 521.47 | 482 | **43.5** |
| T3.1 arm_deformation, lag B | 90.12 | 76.42 | 78.74 | 912 | 6.6 |
| T3.1 arm_deformation, lag A | 94.01 | 76.51 | 77.80 | 904 | 6.5 |
| T3.2 tip_motion | 8.19 | 6.46 | 7.57 | 482 | 0.6 |
| **T4.5 diagnostics (10 tracker passes)** | 3528.78 | 4117.66 | — | 131 | **343.1** |
| E2E compute_all fig5, stored IDs | 151.79 | 581.45* | 131.53 | 999 | 11.0 |
| **E2E compute_all fig5 + relink** | 702.57 | 785.77 | 647.99 | 999 | **54.0** |
| IO: container npz load (`allow_pickle=False`) | 3.34 | — | — | 55 | 0.3 |
| IO: adapters (flipping CSV + pickle, Sleepy npz) | — | 9.02 | — | 201 | 0.8 |

\* Sleepy segments have no stored IDs, so for the subset this row includes a re-link.

### Camera-year projection (3e8 arms, one core)

| Workload | CPU-h |
|---|---:|
| Fig 5 representations with stored IDs | about 11 |
| Fig 5 representations with a re-link | about 54 |
| One tracker pass | about 44 |
| The Fig 6D diagnostics (10 passes) | about 340 |

The figure scripts' own baseline (parity store README) was 45–80 CPU-h per tracker pass
and about 10× that for the diagnostics, so v0 is at script speed, as expected.

### Hotspots (cProfile)

The profile covers `compute_all('fig5')` plus `relink` on the subset: 333 s profiled,
`tottime` order.

| Function | Calls | Self s | Cumulative s | Note |
|---|---:|---:|---:|---|
| `_reference.common_span_dist` → inner `sample` | 3.19 M / 6.37 M | 16.1 + 81.9 | **237.6 (71%)** | n×m Python loop of the tracker cost (T4.1/T4.2); `linspace` (41.8 s cumulative), `clip` (47.3 s), `norm` and `mean` overheads per 12-point pair |
| `FusedTracker.step` | 31,749 | 13.6 | 265.7 (80%) | per-frame Python loop over the cost matrix; `linear_sum_assignment` itself is minor |
| `tip_anchored_resample` | 280,915 | 16.7 | 26.3 (8%) | per-point `while` loop (T0.2 tip) |
| `rigid_residual` | 164,291 | 2.7 | 12.3 (4%) | 2×2 SVD Kabsch per linked pair (T3.1) |
| `np.unwrap`, `np.diff` (per-arm metrics) | 173 k, 975 k | 6.3 | 8.5 | T0.3/T1.1, once per arm and per call site |

Peak memory: `to_source` reaches 2 GB on 1e6 arms (28 M points) because it builds per-point
int64 index arrays; a CSR `np.repeat` of crop rows would avoid them.

### Parity at v0

`parity_check.py` covers micro, the 7 flipping clips and the Sleepy subset (36 segments,
18 boundaries, 18 Fig 6D sequences). Result: **3,237 checks, 0 failed, 3,237
bit-identical**. With `--oracle`, the `_reference` ports themselves were checked on micro
and the flipping clips: **2,625 checks, 0 failed, all bit-identical**.

### Harness corrections after v0 (review)

The v0 run predates these fixes to `bench.py`; read its numbers with them in mind:

- **Peak RSS column is not per kernel.** v0 reused pool workers and reported the worker's
  lifetime `ru_maxrss`, including the input build and any earlier kernel in that worker
  (hence the flat 482 MB). `bench.py` now runs every job in a fresh process and reports
  the RSS after the build and the kernel's own peak above it (VmHWM reset).
- **Timings were concurrent and cold.** v0 ran 40 jobs at once with no warm-up, and the
  T4.5 diagnostics once. The defaults are now `--jobs 1`, one untimed warm-up call and no
  cap on the diagnostics repeats.
- **The projection underestimated the tracker.** It took the largest input (the synthetic
  micro tiling, 4–8 arms per frame, no gaps) and wall time. It now uses CPU time and
  prefers the real subset. On v0 numbers the tracker pass is 660 µs/arm (subset), so
  about 55 CPU-h per camera-year rather than 43.5. The synthetic input now tiles the
  subset by default when `--data-dir` is given.
- **The subset is not the parity subset.** It is the first 7 clips of `manifest.csv` and
  the first 36 Sleepy segments in sort order; only 3 of those clips and 2 of those
  segments are in the parity gate.

### Targets for the rounds

1. **Tracker cost.** Use a batched (n, m, 12) `common_span_dist` and cost matrix per
   step, and cache each frame's tip resample across passes. This is 80% of tracked time.
2. **CSR kernels.** Rewrite tip and base resample with `searchsorted` and `interp` over
   CSR, and compute arc length and curvature with segment reductions. The float32
   arithmetic must stay.
3. **Per-frame reductions.** Batch percentile, median and nanmedian with lexsort segment
   ops, and compute isotropy from segment covariance.
4. **Kinematics.** Replace the dict joins with sorted (track, frame) joins, use
   closed-form 2-D Kabsch, and express the lags as index arithmetic.
5. **Scale-out.** Stream sequences through a process pool with bounded memory.

## Round 1: batched CSR geometry kernels

### What changed and why

The v0 profile put the per-arm Python loops of T0.2/T0.3/T1.1 at about 30–70 µs per arm.
They are now batched over arms. The figure-script code works on one arm at a time and its float32
sums depend on the order of the additions, so a plain `np.add.reduceat` or a global `cumsum` would
change the last bits. The new `_csr` module reproduces numpy's order for every arm at once.

- **`_csr.seg_cumsum`** is `np.cumsum` per segment. It loops over positions within an arm (at most
  119 on the released data), not over arms.
- **`_csr.seg_sum`** is `np.sum` per segment, using numpy's pairwise summation: sequential under 8
  values, eight interleaved accumulators up to 128 and recursive halves above (those rare arms fall
  back to `np.add.reduce`).
- **`_csr.seg_search`** is an exact branch-free bisection in each arm's own range, with
  `searchsorted` right/left semantics.
- **T0.2 base resample.** Float64 arc length from `seg_cumsum`, the targets as `np.linspace` computes
  them, then `np.interp`'s search and formula (node hits, last node and the NaN fallback).
- **T0.2 tip resample.** Each arm is reversed in place in the global layout. The float32 cumulative
  length comes from `seg_cumsum`, and the per-arm walk becomes a strict `seg_search`. The
  interpolation is done in float64 and the result cast to float32, as in the tracker. The
  body-relative origin is broadcast with `np.repeat`; no per-arm loop is left.
- **T0.3 / T1.1 arc length and curvature.** Segment norms are computed as `sqrt(dx*dx + dy*dy)`,
  which is bit-equal to the 2-element `linalg.norm` reduce and avoids its slow short-axis reduction. Lengths and curvature
  sums come from `seg_sum`. `np.unwrap` is applied in two steps. Its elementwise steps (`mod`,
  boundary fix, `|dd| < pi` mask) run on the global angle differences. The sequential correction
  `cumsum` runs only on arms that have a non-zero correction. On the others the unwrapped
  differences equal the raw ones, up to the sign of zero, which `abs` removes.
  The final division is float32, as in the scripts (`np.float32 / float`).
- **T0.4 / T0.5 / T1.4.** Per-frame crop and origin values are broadcast to points with `np.repeat`
  over points per frame. This avoids the (P,) int64 gather indices behind the 2 GB peak noted at v0.
- **Blocks of 8,192 arms.** The kernels run over blocks of arms. A sweep over 1k–64k arms per block
  found 4k–8k fastest, because temporaries stay in cache, and 8k keeps the kernel's own peak at tens
  of MB.
- **Per-container cache.** `PolylineContainer.derived` memoises derived arrays and drops them when a
  dependency field is rebound. The cache is never pickled and its arrays are read-only. Three
  cached helpers are built on it:
  - `geometry.base_shapes(c)`, the T0.2 base resample, for T1.6/T3.1;
  - `geometry.arm_metrics(c, two_point)`, T0.3/T1.1 for the T2 aggregates;
  - `tracking.tracker_inputs(c)`, the T0.2 tip resample, for every tracker pass, step cost and
    diagnostic.

  `compute_all` computes each of them once per container. `bench.py` now clears the cache before
  every timed call, so each timing below is cold.

Not changed (later rounds): the tracker's per-pair cost (T4.1/T4.2), the per-frame reduction loops
(T2.2–T2.6) and the Kabsch / dict joins of T3.

### Run

| | |
|---|---|
| Date | 2026-11-02 |
| CPU | Intel Xeon W-3365, 64 cores (shared workstation) |
| Software | Python 3.12.13, numpy 2.4.3, scipy 1.17.1 |
| Base commit | `518f47e` (before) and this commit (after) |

Before and after were both run with the post-v0 harness, on the same inputs and with the same
settings:

```
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs micro,subset,synthetic \
    --data-dir RUNS/paper_retrace_f39dc82b --synthetic-arms 1000000 --repeats 5 --synthetic-repeats 5 \
    --jobs 14 --kernels "T0.,T1.,T2.,T3.,E2E compute_all fig5 (stored"
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs micro,subset \
    --data-dir RUNS/paper_retrace_f39dc82b --repeats 5 --jobs 4 --kernels "T4.4,fig5 + relink"
```

- **Timings.** Each value is the best of 5 runs after one warm-up, each job in a fresh
  single-threaded process. The two commands ran together, so 18 of 64 cores were busy; both runs had
  the same contention.
- **Inputs.** The synthetic input is the subset tiled to 1,000,416 arms. The 10-pass T4.5 diagnostics
  were not re-timed. Only their one-off tracker-input resample changed, so they stay at about 4,100
  µs/arm.
- **Memory.** The before column of kernel peak RSS reads 0 MB for the per-arm loops. Their small
  allocations reused heap freed after the input build. The after values include the output arrays
  (for example 96 MB of tip resamples on 1e6 arms) and the cached shapes.

### Per kernel (before → after)

| Kernel | Micro µs/arm | Subset µs/arm | Synthetic µs/arm | Speed-up, subset / synthetic (CPU) | Kernel peak RSS MB above build | CPU-h per camera-year |
|---|---:|---:|---:|---:|---:|---:|
| T0.2 resample_base | 30.38 → 2.52 | 30.65 → 1.86 | 30.06 → 1.57 | 16.4× / 19.1× | 0 → 3 (synth.) | 2.6 → 0.16 |
| T0.2 resample_tip (body-relative, tracker input) | 71.45 → 3.36 | 70.40 → 2.84 | 69.86 → 2.66 | 24.8× / 26.3× | 0 → 99 (synth.) | 5.9 → 0.24 |
| T0.3 + T1.1 arc_length_curvature | 30.51 → 1.75 | 30.41 → 1.53 | 30.57 → 1.22 | 19.9× / 25.1× | 0 → 0 (synth.) | 2.5 → 0.13 |
| T0.4 to_source | 1.12 → 0.46 | 0.86 → 0.33 | 0.73 → 0.37 | 2.6× / 2.0× | 106 → 67 (synth.) | 0.1 → 0.03 |
| T1.3 tip_polar | 0.07 → 0.07 | 0.09 → 0.09 | 0.08 → 0.08 | 1.0× / 1.0× | 0 → 0 (synth.) | 0.0 → 0.01 |
| T2.2 fold_asymmetry (fig5) | 1.72 → 1.74 | 1.47 → 1.48 | 1.45 → 1.47 | 1.0× / 1.0× | 0 → 0 (synth.) | 0.1 → 0.12 |
| T2.3 arm_length_spread (fig5, includes T0.3) | 34.63 → 9.29 | 32.82 → 7.21 | 33.24 → 6.86 | 4.6× / 4.8× | 0 → 0 (synth.) | 2.7 → 0.60 |
| T2.4 arm_curvature (fig5, includes T1.1) | 30.74 → 5.22 | 30.16 → 4.32 | 30.19 → 3.81 | 7.0× / 7.9× | 0 → 0 (synth.) | 2.5 → 0.36 |
| T2.6 tip_isotropy | 6.89 → 6.92 | 5.74 → 5.74 | 5.55 → 5.60 | 1.0× / 1.0× | 0 → 0 (synth.) | 0.5 → 0.48 |
| T0.7 R0 policies | 0.07 → 0.07 | 0.02 → 0.02 | 0.01 → 0.01 | 1.0× / 0.9× | 0 → 0 (synth.) | 0.0 → 0.00 |
| T3.1 arm_deformation, lag B (includes T0.2 base) | 78.33 → 50.61 | 78.40 → 46.84 | 75.59 → 47.44 | 1.7× / 1.6× | 45 → 185 (synth.) | 6.5 → 3.90 |
| T3.1 arm_deformation, lag A (includes T0.2 base) | 77.64 → 49.98 | 76.10 → 46.81 | 75.18 → 45.69 | 1.6× / 1.6× | 31 → 186 (synth.) | 6.3 → 3.90 |
| T3.2 tip_motion | 7.85 → 7.17 | 6.45 → 6.40 | 6.36 → 6.41 | 1.0× / 1.0× | 0 → 0 (synth.) | 0.5 → 0.53 |
| **E2E compute_all fig5 (stored IDs)** | 136.90 → 78.37 | 584.00 → 480.10 | 124.62 → 69.56 | 1.2× / 1.8× | 46 → 194 (synth.) | 48.7 → 40.00 |
| **T4.4 relink (1 pass)** | 501.43 → 437.38 | 659.22 → 586.63 | — | 1.1× / — | 1 → 39 (subset) | 54.9 → 48.87 |
| **E2E compute_all fig5 + relink** | 638.87 → 516.79 | 795.97 → 656.33 | — | 1.2× / — | 7 → 71 (subset) | 66.3 → 54.69 |

The subset's "stored IDs" end-to-end row re-links its 36 Sleepy segments, which have no stored IDs,
so the tracker dominates it. The synthetic row has stored IDs throughout: 124.6 → 69.6 µs/arm.

### Camera-year projection (3e8 arms, one core)

| Workload | v0 harness, before | Round 1 |
|---|---:|---:|
| Geometry kernels T0.2 base + T0.2 tip + T0.3/T1.1 | 11.0 CPU-h | **0.5 CPU-h** |
| Fig 5 representations with stored IDs (synthetic basis) | 10.4 CPU-h | **5.8 CPU-h** |
| Fig 5 representations, subset basis (re-links Sleepy) | 48.7 CPU-h | 40.0 CPU-h |
| Fig 5 representations with a re-link | 66.3 CPU-h | 54.7 CPU-h |
| One tracker pass | 54.9 CPU-h | 48.9 CPU-h |
| Fig 6D diagnostics (10 passes, not re-timed) | about 340 CPU-h | about 335 CPU-h (estimate) |

### Hotspots after round 1

The profile covers `compute_all('fig5')` plus `relink` on the subset: 287 s profiled. The
geometry kernels no longer appear in the top 25.

- `_reference.common_span_dist` and its inner `sample` take 237 s cumulative (83%), and
  `FusedTracker.step` 265 s (92%). This is round 2's target.
- The next largest item is `rigid_residual`: 12.5 s, called 164k times by T3.1 (round 4).
- Within the stored-ID Fig 5 path, deformation is now about 60% of the time. Its cost is the per-pair
  SVD and the dict joins, not the resample.

### Parity and tests at round 1

- `parity_check.py` (unchanged, full gate: micro, 7 flipping clips, Sleepy subset with all 18
  Fig 6D sequences): **4,213 checks, 0 failed, 4,213 bit-identical**. The HEAD before this round
  gave the same.
- The gate's T0.2, T0.3 and T1.1 outputs are bit-identical on the micro example and the 7 flipping
  clips (67,694 arms). `tests/test_noodle_polylines_kernels.py` adds bit-exact checks on cases the released data rarely
  hits: 1- and 2-point arms, repeated points, zero-length arms, angle wraps, arms over 128 points,
  f32 and f64 origins, and any block size. It also tests the cache: sharing, invalidation,
  read-only arrays and pickling.
- `pytest tests/test_noodle_polylines*.py`: 69 passed.
