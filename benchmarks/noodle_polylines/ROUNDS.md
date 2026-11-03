# noodle_polylines optimisation rounds

Every round must meet two conditions:

- `parity_check.py` passes against the parity reference store, with its checks, pass/fail rules and
  tolerances (`RESAMPLE_ATOL = 1e-5`, `RTOL = 1e-9`) unchanged. A round may add an
  information-only report; round 3 added the near-tie audit;
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
- **Opt-in per-container cache.** `PolylineContainer.derived` memoises derived arrays, but only
  inside a `with c.caching():` scope; outside one it computes and returns fresh arrays and keeps
  nothing. In a scope the arrays are read-only, are dropped when a dependency field is rebound or the
  outermost scope exits, and are never pickled. Three helpers are built on it:
  - `geometry.base_shapes(c)`, the T0.2 base resample, for T1.6/T3.1;
  - `geometry.arm_metrics(c, two_point)`, T0.3/T1.1 for the T2 aggregates;
  - `tracking.tracker_inputs(c)`, the T0.2 tip resample, for every tracker pass, step cost and
    diagnostic.

  `compute_all` opens no scope. It shares its arc length, curvature and base shapes through local
  variables, as before, so they are freed when it returns. The first version of this commit kept
  them on the container instead; see "Review repair" below.

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
- **Memory.** The before column of kernel peak RSS often reads 0 MB for the per-arm loops. Their
  many small allocations reused heap freed after the input build. A batched kernel allocates its
  output as one fresh array (for example 96 MB of tip resamples on 1e6 arms), so part of the
  after-minus-before difference is the output itself, not extra working memory. The memory column
  below is from the repaired code (see "Review repair").
- **Timings are from a contended run.** Both commands ran at once (18 jobs on 64 cores). A review
  rerun with the same settings measured the synthetic geometry kernels 20–25% slower than the table
  (resample_base 1.97, resample_tip 3.25, arc_length_curvature 1.47 µs/arm); a clean `--jobs 1`
  rerun was back within ±15% of it. Treat single values as ±25%.

### Per kernel (before → after)

| Kernel | Micro µs/arm | Subset µs/arm | Synthetic µs/arm | Speed-up, subset / synthetic (CPU) | Kernel peak RSS MB above build | CPU-h per camera-year |
|---|---:|---:|---:|---:|---:|---:|
| T0.2 resample_base | 30.38 → 2.52 | 30.65 → 1.86 | 30.06 → 1.57 | 16.4× / 19.1× | 0 → 0 (synth.)† | 2.6 → 0.16 |
| T0.2 resample_tip (body-relative, tracker input) | 71.45 → 3.36 | 70.40 → 2.84 | 69.86 → 2.66 | 24.8× / 26.3× | 0 → 9 (synth.)†; 0 → 28 (subset)† | 5.9 → 0.24 |
| T0.3 + T1.1 arc_length_curvature | 30.51 → 1.75 | 30.41 → 1.53 | 30.57 → 1.22 | 19.9× / 25.1× | 0 → 0 (synth.) | 2.5 → 0.13 |
| T0.4 to_source | 1.12 → 0.46 | 0.86 → 0.33 | 0.73 → 0.37 | 2.6× / 2.0× | 106 → 67 (synth.) | 0.1 → 0.03 |
| T1.3 tip_polar | 0.07 → 0.07 | 0.09 → 0.09 | 0.08 → 0.08 | 1.0× / 1.0× | 0 → 0 (synth.) | 0.0 → 0.01 |
| T2.2 fold_asymmetry (fig5) | 1.72 → 1.74 | 1.47 → 1.48 | 1.45 → 1.47 | 1.0× / 1.0× | 0 → 0 (synth.) | 0.1 → 0.12 |
| T2.3 arm_length_spread (fig5, includes T0.3) | 34.63 → 9.29 | 32.82 → 7.21 | 33.24 → 6.86 | 4.6× / 4.8× | 0 → 0 (synth.) | 2.7 → 0.60 |
| T2.4 arm_curvature (fig5, includes T1.1) | 30.74 → 5.22 | 30.16 → 4.32 | 30.19 → 3.81 | 7.0× / 7.9× | 0 → 0 (synth.) | 2.5 → 0.36 |
| T2.6 tip_isotropy | 6.89 → 6.92 | 5.74 → 5.74 | 5.55 → 5.60 | 1.0× / 1.0× | 0 → 0 (synth.) | 0.5 → 0.48 |
| T0.7 R0 policies | 0.07 → 0.07 | 0.02 → 0.02 | 0.01 → 0.01 | 1.0× / 0.9× | 0 → 0 (synth.) | 0.0 → 0.00 |
| T3.1 arm_deformation, lag B (includes T0.2 base) | 78.33 → 50.61 | 78.40 → 46.84 | 75.59 → 47.44 | 1.7× / 1.6× | 33 → 33 (synth.)† | 6.5 → 3.90 |
| T3.1 arm_deformation, lag A (includes T0.2 base) | 77.64 → 49.98 | 76.10 → 46.81 | 75.18 → 45.69 | 1.6× / 1.6× | 32 → 35 (synth.)† | 6.3 → 3.90 |
| T3.2 tip_motion | 7.85 → 7.17 | 6.45 → 6.40 | 6.36 → 6.41 | 1.0× / 1.0× | 0 → 0 (synth.) | 0.5 → 0.53 |
| **E2E compute_all fig5 (stored IDs)** | 136.90 → 78.37 | 584.00 → 480.10 | 124.62 → 69.56 | 1.2× / 1.8× | 35 → 33 (synth.)† | 48.7 → 40.00 |
| **T4.4 relink (1 pass)** | 501.43 → 437.38 | 659.22 → 586.63 | — | 1.1× / — | 1 → 28 (subset)† | 54.9 → 48.87 |
| **E2E compute_all fig5 + relink** | 638.87 → 516.79 | 795.97 → 656.33 | — | 1.2× / — | 6 → 29 (subset)† | 66.3 → 54.69 |

† Memory from the review-repair run (518f47e vs the repaired code, same settings; see "Review
repair"). The other rows were not re-measured.

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
  f32 and f64 origins, and any block size. It also tests the cache: nothing kept outside a scope,
  sharing and nesting inside one, invalidation, read-only arrays, pickling, and that `compute_all`
  leaves nothing on the container.
- `pytest tests/test_noodle_polylines*.py`: 70 passed.

### Review repair

**Problem.** The first version of this commit cached derived arrays on the container with no scope.
`compute_all`, `arm_deformation`, the T2 aggregates and `tracker_inputs` all left their resamples on
the container after returning: 192 B per arm of float64 base shapes, 16 B of arc length and
curvature, and 100 B of tip resamples and lengths. That is about as much as the input points
(223 B per arm). A reviewer measured whole-process peak RSS against 518f47e and found it 25–40%
higher. For example, `compute_all` fig5 with stored IDs on the 1e6-arm synthetic input went from
447 to 627 MB. At 3e8 arms in one container, the retained arrays would come to about 62 GB. The
commit message gave the larger kernel deltas (194, 185 and 99 MB) but did not call them a
regression.

**Fix.** The cache is now opt-in (`with c.caching():`), and `compute_all` shares its arrays through
local variables. Outputs are bit-identical. `compute_all` no longer copies its arc length and
curvature, so its returned arrays are writable again, as at 518f47e.

**Memory after the fix.** The table gives whole-process peak RSS in MB (build plus kernel; best of 2,
`--jobs` 3 + 2 per version, `OMP_NUM_THREADS=1`):

| Kernel | Input | 518f47e | First version | Repaired | Repaired / 518f47e |
|---|---|---:|---:|---:|---:|
| E2E compute_all fig5, stored IDs | synthetic | 447 | 625 | 446 | 1.00 |
| E2E compute_all fig5, stored IDs | subset | 163 | 218 | 176 | 1.08 |
| E2E compute_all fig5 + relink | subset | 158 | 226 | 184 | 1.16 |
| T3.1 arm_deformation, lag B | synthetic | 447 | 607 | 448 | 1.00 |
| T3.1 arm_deformation, lag A | synthetic | 442 | 608 | 451 | 1.02 |
| T3.1 arm_deformation, lag B | subset | 160 | 198 | 170 | 1.06 |
| T4.4 relink | subset | 156 | 197 | 182 | 1.17 |
| T0.2 resample_tip | synthetic | 412 | 522 | 430 | 1.04 |
| T0.2 resample_tip | subset | 155 | 191 | 187 | 1.21 |
| T0.2 resample_base | synthetic | 413 | 422 | 414 | 1.00 |

The remaining increases on the subset (+13 to +32 MB) come mostly from output arrays. At 518f47e the
tip resample was a list of about 173k small arrays that reused heap freed after the input build, so
it read as +0 MB. Now it is one fresh 17 MB array. The relink and tip-resample rows hold that array,
plus the 3 MB per-arm origin and the 8,192-arm block temporaries. None of this outlives the call.

**Timing check.** A clean `--jobs 1` rerun of the repaired code (best of 5, single-threaded) gave
these synthetic values in µs per arm: resample_base 1.47, resample_tip 1.95, arc_length_curvature
1.10, arm_deformation 47.3 (lag B) and 47.3 (lag A), and `compute_all` fig5 with stored IDs 70.6.
The subset values were 1.72, 2.59, 1.41, 47.3, 47.0 and 479.4. All are within 15% of the table
above, or faster. The parity gate gave the same result as before: 4,213 checks, 0 failed, 4,213
bit-identical.


## Round 2: batched per-frame reductions and tracked kinematics

### What changed and why

Before this round the per-frame aggregates (T2.2–T2.6) and the tracked kinematics (T3.1, T3.2) were
still Python loops over frames. At round 1 they cost about 16 µs/arm (T2) and 54 µs/arm (T3) on
the subset. The profile of the stored-ID Fig 5 path at `b41bd72` (subset, 71.5 µs/arm) confirmed
where the time went. `rigid_residual` took 12.1 of the 20.5 s profiled, with SVD, `det`, `diag`
and `mean` overheads per linked pair. Then came the per-frame `np.median` / `np.percentile` /
`np.cov` calls, with 0.6 M `_mean` calls, 62 k `_median` and 21 k `_quantile`. T3.1 lay outside the
round's brief (per-frame reductions), but the task allows "anything else that the profile shows
dominates", and its per-frame median is one of those reductions, so it is included. Everything
stays bit-identical to the per-frame loops. No change uses a closed form that would only agree to
within a tolerance.

- **New `_segments` module** (exact segment ops over frames):
  - `seg_sort` sorts every frame's values as NaN-padded rows in power-of-two width classes, with
    `np.sort(axis=1)`. This was 5× faster than `lexsort` on 1e6 values.
  - `seg_median` and `seg_percentile` evaluate numpy's own expressions on the sorted segment: the
    mean of the one or two middle values (`np.median`), the median of the non-NaN prefix
    (`np.nanmedian`), and the `'linear'` virtual index with numpy's index clamping and two-sided
    `_lerp`. Order statistics are exact input values, so these are bit-identical to numpy, ties,
    NaN and ±inf included.
  - `seg_rowsum` is `x.sum(0)` / `mean(0)` of (n, 2) blocks. numpy accumulates these row by row
    from zero, not pairwise, and `np.add.reduceat` does not match it (2,706 of 3,000 random
    blocks differed).
  - `row_norm` is `np.linalg.norm` of 2-vectors. numpy computes it as `sqrt(x.dot(x))` through BLAS
    `ddot`, which fuses the multiply-add, so `sqrt(x*x + y*y)` differs in the last bit (16,058 of
    200,000 random vectors). `np.vecdot` takes the same dot path and matched all of them.
- **T2.2 fold asymmetry, T2.3 IQR, T2.4 median curvature, T2.5 mean.** Each variant's per-arm
  expression (`fig4e`, `fig4d`, `fig5` scaling orders, and float32 tip means in Fig 5) is computed
  for all arms at once. The per-frame reduction is a segment op, and the frame masks are applied
  afterwards. T2.5's per-frame mean uses round 1's pairwise `_csr.seg_sum`.
- **T2.6 tip isotropy: batched LAPACK, not closed form.** A closed-form 2×2 eigen solve was tried
  first. It agreed with the scripts only to about 3e-16 relative and failed the bit-exact unit
  tests, so it was dropped. Instead, frames with the same tip count are stacked, and `np.cov` is
  repeated literally: the tip-by-tip mean, centring, `X @ X.T` as `matmul(X, swapaxes(X))`, then
  `* 1/(n-1)`. The 2×2 matrices then go through `np.linalg.eigvalsh` in one stacked call. The
  second operand of the product must be a transposed view of the same buffer. A separate
  contiguous copy takes a different BLAS path and differed on one Sleepy frame (17 tips) in the
  parity gate. With the view, the results are bit-identical.
- **T3.1 / T3.2: sorted-key joins and a batched exact Kabsch fit.** The scripts' per-frame dicts
  (`{tid: shape}`, where a later arm of the same track overwrites an earlier one, followed by
  `tid in prev` lookups) are replaced by one sort of (frame, track) keys and a `searchsorted`. The
  key is the frame ordinal for `last_arm_frame` and T3.2, and the frame position for
  `prev_frame_reset`, which also de-duplicates the current side. `kinematics.rigid_residuals`
  runs `_reference.rigid_residual`'s expression on stacked pairs. Each step (`mean(0)`,
  `ca.T @ cb`, the 2×2 `svd`, `sign(det(u @ vt))`, `u @ diag(1, d) @ vt`, the residual and
  numpy's pairwise `.mean()`) is a stacked `matmul` or LAPACK gufunc that runs the same kernel on
  every slice. It matched `R.rigid_residual` on 100,000 random pairs, bit for bit, at 2.2 µs per
  pair instead of 47. The tolerance allowed for closed-form Kabsch (1e-10 abs) is not used. Shapes
  are built only for linked arms, in blocks of 32,768 pairs. A NaN R0 on a linked frame still
  raises `LinAlgError` from the SVD, as the scripts do.
- **Window masks (T0.7).** `scale.in_any_window` merges the windows into disjoint closed intervals
  and uses `searchsorted`, which is O((n + W) log W) instead of an (n, W) broadcast. Windows with a
  NaN bound or `lo > hi` are dropped, since they never contain a point. The R0 policies were
  already one vectorised median each.
- **Fig 4D per-bout deformation.** Each bout window now runs on its own masked frame span
  (`select_frames`), not the whole sequence. Frames outside the shape mask only reset or skip the
  lag reference, so the values are identical. A clip with W bouts now costs O(span) per bout
  instead of O(F).
- **Bench harness.** Two kernels were added: `T2.5 frame_lists_4b` and `E2E compute_all fig4d (bout
  windows, stored IDs)`. The second uses a 30-s window every 120 s and an explicit R0, because a
  short Sleepy segment can lie wholly inside its own windows, leaving a NaN outside-window R0.
  Both the old and the new code then fail in the SVD.

### Run

| | |
|---|---|
| Date | 2026-11-02 |
| CPU | Intel Xeon W-3365, 64 cores (shared workstation, load average about 4) |
| Software | Python 3.12.13, numpy 2.4.3, scipy 1.17.1 |
| Before | `b41bd72` (round 1 plus review repair), run with this round's `bench.py` |
| After | this commit |

```
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs micro,subset,synthetic \
    --data-dir RUNS/paper_retrace_f39dc82b --synthetic-arms 1000000 --repeats 5 --synthetic-repeats 5 \
    --jobs 2 --kernels "T0.,T1.,T2.,T3.,E2E compute_all fig5 (stored,fig4d"
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs micro,subset \
    --data-dir RUNS/paper_retrace_f39dc82b --repeats 5 --jobs 1 --kernels "fig5 + relink"
```

- **Timings.** Each value is the best of 5 runs after one warm-up, with each job in a fresh
  single-threaded process.
- **Concurrency.** The before and after runs went at the same time, so 4 single-threaded processes
  shared 64 cores. The unchanged geometry kernels came out within 3% of round 1's clean `--jobs 1`
  rerun (synthetic resample_base 1.44/1.46 µs/arm, against 1.47).
- **Not re-timed.** The tracker kernels (T4.4 relink, T4.5 diagnostics) did not change in this
  round.
- **Inputs.** The same as round 1: micro is 1,226 arms; the subset is 173,434 arms (7 flipping
  clips and 36 Sleepy segments); the synthetic input is the subset tiled to 1,000,416 arms.
- **Memory.** Peak RSS is whole-process VmHWM in MB (input build plus kernel), shown as before → after.

### Per kernel (before → after)

| Kernel | Micro µs/arm | Subset µs/arm | Synthetic µs/arm | Speed-up, subset / synthetic (CPU) | Peak RSS MB, subset; synthetic | CPU-h per camera-year |
|---|---:|---:|---:|---:|---:|---:|
| T2.2 fold_asymmetry (fig5) | 1.73 → 0.11 | 1.55 → 0.08 | 1.44 → 0.05 | 20.1× / 26.7× | 156 → 161; 415 → 415 | 0.13 → 0.006 |
| T2.3 arm_length_spread (fig5, includes T0.3) | 8.98 → 1.93 | 7.35 → 1.57 | 6.83 → 1.18 | 4.7× / 5.8× | 168 → 169; 415 → 416 | 0.61 → 0.13 |
| T2.4 arm_curvature (fig5, includes T1.1) | 5.07 → 1.84 | 4.35 → 1.58 | 3.82 → 1.15 | 2.7× / 3.3× | 174 → 166; 414 → 415 | 0.36 → 0.13 |
| T2.5 frame_lists_4b (includes T0.3) | 2.78 → 1.83 | 2.25 → 1.51 | 1.91 → 1.14 | 1.5× / 1.7× | 174 → 168; 416 → 413 | 0.19 → 0.13 |
| T2.6 tip_isotropy | 6.84 → 0.31 | 5.72 → 0.20 | 5.63 → 0.15 | 28.5× / 38.8× | 163 → 156; 414 → 416 | 0.48 → 0.017 |
| T0.7 R0 policies | 0.07 → 0.08 | 0.02 → 0.03 | 0.01 → 0.01 | 0.8× / 0.8× | 156 → 156; 414 → 415 | 0.00 → 0.00 |
| T3.1 arm_deformation, lag B (includes T0.2 base) | 49.41 → 4.84 | 47.05 → 4.11 | 47.28 → 3.77 | 11.4× / 12.5× | 172 → 172; 462 → 446 | 3.92 → 0.34 |
| T3.1 arm_deformation, lag A (includes T0.2 base) | 49.80 → 4.97 | 46.03 → 4.27 | 46.35 → 3.84 | 10.8× / 12.1× | 171 → 178; 446 → 453 | 3.84 → 0.36 |
| T3.2 tip_motion | 7.20 → 0.40 | 6.41 → 0.31 | 6.42 → 0.25 | 20.6× / 25.3× | 157 → 162; 416 → 413 | 0.53 → 0.026 |
| **E2E compute_all fig5 (stored IDs)** | 78.24 → 8.15 | 482.54 → 415.63* | 70.16 → 5.56 | 1.2×* / **12.6×** | 176 → 174; 446 → 448 | 40.2* → 34.6* |
| **E2E compute_all fig4d (bout windows, stored IDs)** | 49.62 → 7.04 | 40.51 → 5.24 | 41.52 → 4.39 | 7.7× / 9.5× | 174 → 171; 452 → 421 | 3.38 → 0.44 |
| **E2E compute_all fig5 + relink** | 506.26 → 443.62 | 661.41 → 592.71 | — | 1.1× / — | 183 → 183 | 55.1 → 49.4 |
| *unchanged:* T0.2 resample_base | 2.32 → 2.09 | 1.87 → 1.75 | 1.44 → 1.46 | 1.1× / 1.0× | 177 → 169; 416 → 429 | 0.16 → 0.15 |
| *unchanged:* T0.2 resample_tip (tracker input) | 3.09 → 3.04 | 2.76 → 2.79 | 1.93 → 1.95 | 1.0× / 1.0× | 184 → 184; 422 → 431 | 0.23 → 0.23 |
| *unchanged:* T0.3 + T1.1 arc_length_curvature | 1.77 → 1.75 | 1.43 → 1.46 | 1.10 → 1.13 | 1.0× / 1.0× | 169 → 167; 415 → 415 | 0.12 → 0.12 |
| *unchanged:* T0.4 to_source | 0.45 → 0.46 | 0.31 → 0.33 | 0.28 → 0.27 | 0.9× / 1.0× | 164 → 165; 479 → 483 | 0.03 → 0.03 |
| *unchanged:* T1.3 tip_polar | 0.07 → 0.07 | 0.08 → 0.09 | 0.08 → 0.08 | 1.0× / 1.0× | 156 → 152; 417 → 414 | 0.01 → 0.01 |

\* The subset's "stored IDs" row re-links its 36 Sleepy segments, which have no stored IDs, so the
tracker dominates it (about 586 µs/arm per pass at round 1). The synthetic row has stored IDs
throughout.

The T2.3, T2.4 and T2.5 rows include the T0.3/T1.1 arm metrics (1.1–1.5 µs/arm), so their own
reductions now cost 0.05–0.1 µs/arm. In `compute_all` the metrics are computed once and shared.

Peak RSS did not change beyond noise: every row moved by 13 MB or less either way, except two
drops. T3.1 lag B on synthetic dropped from 462 to 446 MB, because shapes are built per block of linked
pairs rather than for all arms at once (192 B per arm). Fig 4D on synthetic dropped from 452 to
421 MB, because each bout's shapes are now built only over that bout's span.

### Camera-year projection (3e8 arms, one core)

| Workload | Round 1 (`b41bd72`, this run) | Round 2 |
|---|---:|---:|
| Fig 5 representations with stored IDs (synthetic basis) | 5.85 CPU-h | **0.46 CPU-h** |
| Fig 4D representations with stored IDs, 30-s bout every 120 s (synthetic basis) | 3.46 CPU-h | **0.37 CPU-h** |
| Fig 5 representations with a re-link (subset basis) | 55.1 CPU-h | 49.4 CPU-h |
| One tracker pass (not re-timed) | about 49 CPU-h | about 49 CPU-h |
| Fig 6D diagnostics (10 passes, not re-timed) | about 335 CPU-h | about 335 CPU-h |

Against `518f47e` under the post-v0 harness (124.6 µs/arm, 10.4 CPU-h; see round 1), the Fig 5
stored-ID path is now 22× faster (5.56 µs/arm).
Every workload that re-links is now about 99% tracker. That is the target for the next tracker
round (batched `common_span_dist` cost matrix, T4.1/T4.2).

### Hotspots after round 2

The profile covers `compute_all('fig5')` with stored IDs on the subset (Sleepy segments re-linked
beforehand): 1.14 s profiled, against 20.5 s at `b41bd72`.

| Function | Self s | Note |
|---|---:|---|
| `np.linalg.svd` (stacked 2×2) | 0.21 | LAPACK on 164 k linked pairs, about 1.3 µs per pair; replacing it with a closed form would give up bit-identity |
| `_csr.seg_search`, `_resample_base_block` | 0.19 | T0.2 base resample for the T3.1 shapes |
| `rigid_residuals` (own) | 0.10 | stacked matmuls and temporaries |
| `_arc_curv_block`, `seg_sum`, `seg_cumsum` | 0.21 | T0.3/T1.1 arm metrics |
| everything per frame (T2 + T3 joins and medians) | < 0.1 | |

### Parity and tests at round 2

- `parity_check.py`, unchanged, full gate (micro, 7 flipping clips, and the Sleepy subset with all
  18 Fig 6D sequences): **4,213 checks, 0 failed, 4,213 bit-identical**, the same as round 1.
  Its float rows include Fig 4D/4E lag A, Fig 5 lag B, tip motion, isotropy (support-centroid and
  runtime-origin frames, both window fallbacks), the R0 policies and every preset.
- The intermediate closed-form isotropy and the separate-copy covariance product were not committed:
  they gave 4,212 bit-identical (one Sleepy segment at 1.2e-16 relative, within tolerance).
- New `tests/test_noodle_polylines_segments.py` (16 tests) checks the batched code bit for bit
  against the per-frame loops it replaced. Those loops are kept in the test as oracles and call
  `_reference.rigid_residual` and `isotropy_frame`. The cases cover:
  - frames with 0–3 and 150 arms, duplicate arms (ties), 1- and 2-point arms;
  - NaN origins and centroids, frames without a star, and a per-frame R0 with NaN and 0;
  - duplicate track IDs within a frame, untracked arms (`-1`) and gaps;
  - both lags, both shape orders and a shape mask, with every T2 variant, reducer and `min_arms`;
  - degenerate isotropy frames (collinear and identical tips);
  - overlapping, touching, reversed and NaN windows;
  - Fig 4D per-bout windows against whole-sequence runs, also through `compute_all`;
  - the `_segments` ops against numpy, with NaN, ±inf, -0.0, ties and segments up to 300 long.
- `pytest tests/test_noodle_polylines*.py`: **86 passed** (70 + 16). `_reference.py`, the
  fixtures, the tolerances and `parity_check.py` are unchanged. The micro example runs.

## Round 3: batched association (T4)

### What changed and why

The profile of a re-link at `a4ae126` (3 Sleepy segments, 9,424 arms, 623 µs/arm) confirmed the
hotspot named at round 2. `_reference.common_span_dist` took 4.12 of the 4.59 s profiled (90%), as
56,770 per-pair calls. Inside it, the inner `sample` took 2.34 s, `linspace` 0.64 s, `clip` 0.82 s,
and the per-pair `np.linalg.norm` and `mean` overheads the rest. A Sleepy frame has about 10 arms
against 11–13 active tracks, so each step evaluated about 120 pairs in Python. Every workload that
re-links was about 99% tracker. The changes below keep every matrix, assignment and label
bit-identical to the reference tracker. `linear_sum_assignment` is called on the same matrices, the
frame loop stays sequential, and nothing is a closed form that only agrees within a tolerance.

- **New `_assoc` module: the cost matrix as one block.**
  - `common_span_dist` samples both arms at `linspace(0, lc, 12)` with `lc = min(la, lb)`. For the
    shorter arm, `lc` is its own length, so its sample depends on that arm only. `arm_rows`
    computes it once per arm, together with the arm's grid, its points as float64, `max(L, 1e-6)`
    and a degenerate flag. Only the longer arm is interpolated per pair.
  - `span_dist_rows` evaluates all n × m pairs of a step at once. It picks short and long rows with
    `np.where` and computes the reference's `idx`, `i0 = clip(int(idx), 0, 10)`, `t` and
    `(1 - t) * p[i0] + t * p[i0 + 1]` for the long arm. It then takes the point distance
    `sqrt(dx*dx + dy*dy)`, the same unfused `add.reduce` of squares that `norm(axis=1)` uses, and
    the mean as `add.reduce(axis=-1) / 12`, numpy's pairwise sum over the contiguous last axis, as in
    `ndarray.mean`. x and y sit in separate column planes, so every elementwise step runs over
    contiguous (n, m, 12) blocks; with x and y interleaved (an inner axis of length 2) a re-link was
    about 25% slower. Points are gathered with flat `take` indices. The first version used
    `take_along_axis`, which builds broadcast index grids and took about a third of a step.
  - Pairs whose shorter arm has length ≤ 1e-6 (single-point and zero-length arms) go through the
    reference function itself. These pairs use the tip-distance branch, which is a float32 BLAS dot.
  - `embed_dist` is `sqrt(np.vecdot(D, D))` over the (n, m, 8) float32 differences. `np.linalg.norm`
    of a float32 vector is `sqrt(x.dot(x))`, a BLAS `sdot`, and `np.vecdot` calls the same dot
    routine for every pair: it matched 200,000 of 200,000 random vectors. An emulated FMA chain
    matched 80%, a plain float32 sum 81% and `einsum` 86%.
  - `augmented` builds the (n+m)² matrix with strided slice writes instead of `diag_indices`.
- **Array-state tracker (`_assoc.ArrayTracker`, `_assoc.run`).** With `vel_alpha = 0` (the released
  setting) and finite, bounded inputs, the reference velocity stays exactly zero, and a prediction
  is the track's last shape (`poly + 0·dt`; only the sign of a zero can change, which no distance
  sees). The tracks are then arrays in insertion order: arm rows, float32 embedding, age and ID.
  Insertion order is ascending ID, because tracks are appended with increasing IDs and only ever
  removed. Matching, the embedding EMA (`(1 - α)·e + α·e_i` in float32, gated on the same distance
  as the cost), ageing, births and retirement are array updates. The augmented matrix's row
  solution is `arange(n + m)`, as scipy documents for square input, so the kept pairs are the first
  n columns below m. The tracker checks the inputs once per pass, in 65,536-arm blocks. A velocity
  term, float64 or non-finite inputs run the reference tracker unchanged. Arm rows are built per
  block of about 32,768 arms (whole frames), so the extra memory is about 16 MB, not 500 B per arm
  of the sequence.
- **`relink`, `step_costs`, `cost_matrix` and `diagnostics` use it.** `cost_matrix` is now the
  batched block (the per-pair loop is gone). `step_costs` runs the array tracker up to the frame;
  the 'sorted' and 'insertion' track orders are the same order.
- **Stored-ID fast path.** `compute_all` with stored IDs already skipped the tracker. A new test
  replaces `tracker_inputs` and the tracker pass with functions that raise, to prove it.
- **T4.5 counters as sorted-key joins (`tracking._Sequence`).** Continuation, forward–reverse link
  agreement, jumps, births, arm-weighted lifetimes and stride precision were per-frame Python sets
  (`frame_measures`, `links`). They are now computed over the whole sequence. Each arm's partner,
  the arm one (or g) sequence positions earlier with the same label, is a `searchsorted` on (position,
  label) keys. Births and lifetimes come from first and last appearances (`np.unique`). The counts
  per state are `bincount`s, in the same key order: 'all', then the states in order of first
  appearance. These give the same integers because a frame's labels are unique. The code checks
  this and raises if a processed frame has a missing or repeated label, which the tracker cannot
  produce. Each pass's finiteness check runs once per configuration instead of once per pass.
- **Near-tie audit.** `tracking.assignment_margin(matrix)` gives the decision margin of one tracker
  assignment: how much more the cheapest assignment with a different set of kept (arm, track)
  pairs costs. Any other assignment drops at least one edge of the optimum that touches an arm or
  a track. So the margin is the minimum over re-solves with one such edge forbidden; swaps among
  dummy or gated entries are skipped. `parity_check.py` gains an information-only audit. It
  computes the margin of every LSA call in the flipping re-links (fused, shape-only and uncapped)
  and the Sleepy Fig 5 event re-links, and lists margins below 1e-6. No existing check, tolerance or
  pass/fail rule changed.

### Run

| | |
|---|---|
| Date | 2026-11-02 |
| CPU | Intel Xeon W-3365, 64 cores (shared workstation, load average 2–6) |
| Software | Python 3.12.13, numpy 2.4.3, scipy 1.17.1 (scipy-openblas 0.3.31) |
| Before | `a4ae126` (round 2), run with the same `bench.py` |
| After | this commit |

```
# before (a4ae126): the tracker kernels take ~100 s per subset call, so 4 jobs, 1 repeat on synthetic
# and on the 10-pass diagnostics
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs micro,subset,synthetic \
    --data-dir RUNS/paper_retrace_f39dc82b --synthetic-arms 1000000 --repeats 5 --synthetic-repeats 1 \
    --expensive-repeats 1 --kernels "T4.4,T4.5,E2E compute_all fig5" --jobs 4
# after
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs micro,subset,synthetic \
    --data-dir RUNS/paper_retrace_f39dc82b --synthetic-arms 1000000 --repeats 5 --synthetic-repeats 5 \
    --kernels "T0.2 resample_tip,T4.4,T4.5,E2E compute_all fig5" --jobs 1 --profile profile_after.txt
```

- **Timings.** After: best of 5 after one warm-up, each job in a fresh single-threaded process,
  `--jobs 1`. Before: best of 5 on micro and subset, but a single timed run (after one warm-up) for
  the synthetic input and for T4.5. At ~600 µs/arm a 1e6-arm call takes 10 minutes. The before
  run used 4 concurrent jobs, and the after run overlapped part of it, so up to 5 single-threaded
  processes shared 64 cores. The unchanged kernels in the after run (resample_tip 1.99 µs/arm
  synthetic, against 1.95 at round 2; fig5 stored-ID synthetic 5.62, against 5.56) are within 3% of
  round 2. Read the speed-ups as ±10%.
- **Inputs.** The same as rounds 1 and 2: micro is 1,226 arms, the subset 173,434 arms (7 flipping
  clips and 36 Sleepy segments), and the synthetic input the subset tiled to 1,000,416 arms. T4.5
  is not run on the synthetic input (`--skip-expensive-on synthetic`, the default).
- **Memory.** Peak RSS is whole-process VmHWM in MB (input build plus kernel), before → after.

### Per kernel (before → after, CPU µs/arm)

| Kernel | Micro | Subset | Synthetic | Speed-up, subset / synthetic | Peak RSS MB, subset; synthetic | CPU-h per camera-year |
|---|---:|---:|---:|---:|---:|---:|
| **T4.4 relink (1 pass, includes the T0.2 tip resample)** | 434.3 → 19.4 | 589.3 → 17.0 | 598.9 → 16.3 | **34.7× / 36.7×** | 186 → 184; 417 → 461 | 49.1 → **1.42** |
| **T4.5 diagnostics (10 passes)** | 2,950.9 → 109.6 | 3,968.6 → 100.6 | — | **39.5×** / — | 178 → 178 | 330.7 → **8.4** |
| **E2E compute_all fig5 + relink** | 442.5 → 26.9 | 604.8 → 23.0 | 596.6 → 21.8 | **26.3× / 27.4×** | 185 → 188; 446 → 466 | 50.4 → **1.92** |
| E2E compute_all fig5 (stored IDs)* | 8.01 → 8.46 | 415.9 → 16.1* | 5.58 → 5.62 | 25.8×* / 1.0× | 176 → 176; 444 → 449 | 0.47 → 0.47 (synthetic) |
| *unchanged:* T0.2 resample_tip (tracker input) | — → 3.04 | — → 2.60 | — → 1.99 | (round 2: 2.79 / 1.95) | 183; 424 | 0.22 |

\* The subset's "stored IDs" row re-links its 36 Sleepy segments, which have no stored IDs, so it
moves with the tracker. On the synthetic input every arm has a stored ID and the tracker is not run;
that row did not change (5.6 µs/arm).

One re-link pass is now about 14 µs/arm of tracker on top of the 2–3 µs/arm tip resample, about
110 µs per subset frame (8 arms on average). Under cProfile a step splits into about 75 µs for the
shape block, 9 µs for the embedding block, 6 µs for `linear_sum_assignment` and 29 µs of
bookkeeping. The 10-pass diagnostics are 6.9 full passes of work, because the stride passes feed
1/2 to 1/16 of the frames. The counters were per-frame Python sets. All of them (three variants plus
four strides) now take about 1.7 µs/arm on 12 Sleepy segments.

Peak RSS rose by up to 44 MB on the 1e6-arm synthetic re-link (417 → 461 MB), and by 0–20 MB
elsewhere. The cause is the per-block arm rows (32,768 arms × 63 float64 = 16.5 MB) and their
temporaries, plus the (A,) int64 arm list of the processed frames (8 B per arm). Apart from that
list, the increase is bounded by the block, not the sequence.

### Camera-year projection (3e8 arms, one core)

| Workload | Round 2 (`a4ae126`, this run) | Round 3 |
|---|---:|---:|
| Fig 5 representations with stored IDs (synthetic basis) | 0.47 CPU-h | 0.47 CPU-h |
| Fig 4D representations with stored IDs (synthetic basis, not re-timed) | 0.37 CPU-h | 0.37 CPU-h |
| Fig 5 representations with a re-link (subset basis) | 50.4 CPU-h | **1.9 CPU-h** |
| One tracker pass (subset basis, includes the tip resample) | 49.1 CPU-h | **1.4 CPU-h** |
| Fig 6D diagnostics (10 passes, subset basis) | 330.7 CPU-h | **8.4 CPU-h** |

Against v0 (518f47e, round-1 harness), a camera-year re-link plus Fig 5 went from 54 to 1.9 CPU-h,
and the Fig 6D diagnostics from 343 to 8.4 CPU-h.

### Hotspots after round 3

The profile covers `compute_all('fig5')` plus `relink` on the subset: 6.97 s profiled (31,670
tracker steps), against 287 s at round 1.

| Function | Self s | Cumulative s | Note |
|---|---:|---:|---|
| `_assoc.span_dist_rows` | 1.19 | 2.36 | the (n, m, 12) shape block; about 30 numpy calls per step on ~1,500-element arrays |
| `_assoc.ArrayTracker.step` | 0.91 | 4.84 | bookkeeping: about 40 small array ops per step |
| `_assoc._index`, `_lerp` | 0.67 | 0.71 | part of the shape block |
| `_assoc.embed_dist` | 0.26 | 0.30 | (n, m, 8) float32 differences and `vecdot` |
| `linear_sum_assignment` | 0.19 | 0.19 | 6 µs per call, unchanged |
| `_csr.seg_search`, `_resample_tip_block` | 0.56 | 0.92 | the T0.2 tip resample (tracker input) |
| `np.linalg.svd` (T3.1) | 0.21 | 0.21 | the stacked 2×2 Kabsch fit of the Fig 5 path |

The tracker is now Python-overhead bound: per step, about half the time is numpy call overhead on
small blocks and half arithmetic. The next large gain would need a compiled step loop, which the
numpy + scipy dependency rule excludes, or batching across independent sequences. Several
sequences can step in lock-step with padded blocks, because their trackers do not interact.

### Parity and tests at round 3

- `parity_check.py`, full gate (micro, 7 flipping clips, and the Sleepy subset with all 18 Fig 6D
  sequences): **4,213 checks, 0 failed, 4,213 bit-identical**, the same as rounds 1 and 2. The gate
  compares every LSA matrix value, row and column of the re-links against the figure-script
  references. The T4.5 rows (counters, lifetimes, n_ids, strides and pooled rates, for all three
  variants) all match exactly.
- **Near-tie audit (new, information only):** 39,953 LSA calls, **0 with a decision margin below
  1e-6**; the smallest margin is 7.6e-4 cost units. No assignment in the gate is within reach of a
  last-bit change.
- New `tests/test_noodle_polylines_assoc.py` (23 tests, including the review fix below) checks the batched code against the
  per-pair and per-frame code it replaced, bit for bit. Those loops are kept in the test as oracles
  and call the reference tracker, `common_span_dist`, `frame_measures` and `links`. The cases cover:
  - the shape block for k = 12, 5 and 2, with equal lengths, zero lengths and lengths just below and
    above 1e-6, and the embedding block in float32 and float64;
  - `cost_matrix` with NaN, +inf and -inf lengths and NaN points on either side;
  - `cost_matrix`, the augmented matrix, and every variant's re-link in both directions, comparing
    each LSA matrix and solution;
  - a 300-arm frame (more than 256 rows), strided and sparse frame sets, and block sizes of 1, 37
    and 32,768 arms;
  - the reference fallbacks: a velocity term, NaN-origin frames (both implementations raise the
    same `linear_sum_assignment` error) and a float64 identity;
  - `step_costs` against the Fig 6A script function in both track orders;
  - the T4.5 counters with states that include None, time and frame indices, gaps and empty frames;
  - `assignment_margin` on constructed ties, near ties and gated cases;
  - the stored-ID path, which never touches the tracker.
- `pytest tests/test_noodle_polylines*.py`: **109 passed** (86 + 23). `_reference.py`, the
  fixtures and the tolerances are unchanged. `parity_check.py` only gained the audit. The micro
  example runs.

### Fix after review (round 3)

An independent check found a regression in the public `tracking.cost_matrix` (T4.2) on non-finite
lengths. Round 2 looped over `common_span_dist` pair by pair, which returns NaN for a NaN or inf
length. The first round-3 version raised `IndexError` instead, for two reasons:

- `_assoc._index` applied only the upper clip. Its docstring assumed `idx >= 0`, but a NaN or inf
  length makes `idx` NaN, whose int64 cast is INT64_MIN. The reference's `clip(..., 0, k - 2)` maps
  that to 0, and `t` is then NaN. The fast code passed the value to `take` as a large negative
  index. Both clips are now applied, as two in-place ufuncs (`np.clip` costs more per call).
- The shorter arm was picked as `la <= lb`. For a NaN length that differs from the reference's
  `min(la, lb)`, which keeps `la` unless `lb < la`. With `la = NaN` and `lb = 0` the reference
  computes NaN, but the old test chose the degenerate tip-distance branch. The test is now
  `lb < la`, with the `np.where` arguments swapped, so it adds no operation.

`relink`, `diagnostics` and the fast branch of `step_costs` were not affected, because
`fast_path_ok` sends non-finite inputs to the reference tracker. The fallback branch of
`step_costs` (a frame with a NaN body origin) calls `cost_matrix`. It raised `IndexError` and now
raises the reference's `ValueError` from `linear_sum_assignment` again, as round 2 did. A new
test compares `cost_matrix` with the per-pair reference for NaN, +inf and -inf lengths and NaN
points on both sides; it fails on the first round-3 version. After the fix: `parity_check.py`
4,213 checks, 0 failed, 4,213 bit-identical, near-tie audit unchanged (39,953 calls, smallest
margin 7.6e-4); 109 tests pass. A re-link of the micro example takes 19.0–20.0 µs/arm after the
fix and 18.7–19.2 µs/arm before it (best of 15, three alternating runs). That is within run-to-run
noise, so the round-3 timings above stand.

## Round 4: tracked kinematics (T3)

### What changed and why

Round 2 had already replaced the scripts' per-frame dicts with sorted (frame, track) key joins
(`searchsorted` on `np.unique`-coded keys; both lag variants are index arithmetic on frame
ordinals) and vectorised T3.2. So this round started from a profile, not from the plan's list.
`arm_deformation` (lag B) on the 1e6-arm synthetic input at `bf65377` took 3.86 s:

- the stacked SVD Kabsch fit took 2.0 s (51%): `svd` 1.22 s, `det` 0.09 s, and the
  `rigid_residuals` matmuls and sums the rest;
- the T0.2 base resample of every arm took 1.52 s (40%);
- the joins and medians took about 0.15 s.

Changes:

- **Closed-form 2-D Kabsch (`kabsch='closed'`, the new default).** For centred shapes, the best
  proper rotation of row vectors, `[[c, s], [-s, c]]`, maximises `c·p + s·q`, where
  `p = Σ(x_a x_b + y_a y_b)` and `q = Σ(x_a y_b − y_a x_b)`. So `(c, s) = (p, q) / hypot(p, q)`,
  with the identity when `p = q = 0` (the SVD of a zero matrix also gives the identity). Only proper
  rotations are parameterised, so this is the same fit as the scripts' SVD with its
  `sign(det(u @ vt))` reflection guard. The residual is still formed explicitly, as
  `sqrt(mean(|ca·R − cb|²))`, not as `|ca|² + |cb|² − 2·hypot(p, q)`, which would cancel
  catastrophically for small residuals. The centred shapes are the scripts' own (`mean(0)` summed
  row by row), so only the rotation differs. The arithmetic runs on contiguous (n, 12) x/y planes:
  `einsum` row dots, in blocks of 8,192 pairs that stay in cache. It costs about 0.3 µs per pair,
  against 2.2 µs for the stacked SVD. A block with a non-finite value runs the SVD instead, which
  raises `LinAlgError` exactly as the scripts do (a NaN R0 on a linked frame, for example).
  - **Accuracy audit (new, full corpora).** Every linked pair of all 44 flipping clips and all 191
    non-empty Sleepy segments, under both lags, was compared with the SVD: 1,876,495 pairs. The
    largest difference is 1.9e-15 absolute and 5.7e-12 relative; only 22 pairs differ by more
    than 1e-12 relative. The spec's bound is 1e-10 absolute. No pair had identical or near-zero
    shapes, where a relative comparison would be meaningless.
  - `kabsch='svd'` (on `rigid_residuals`, `arm_deformation` and `compute_all`) keeps the round-2
    stacked SVD. It is bit-identical to the scripts.
- **Each linked arm's shape is built once.** The (ref, cur) pairs are mapped to the sorted set
  of linked arms. Each arm's T1.6 shape is written straight into the plane layout and centred
  once; before, an arm in two pairs was scaled and centred twice. The elementwise operations are
  those of `shape12`, so the values are the same.
- **Only linked arms are resampled.** Without a `base=` argument, `arm_deformation` no longer
  resamples every arm of the container. It works in blocks of 32,768 pairs and resamples only the
  linked arms of the block's arm span: by slicing the contiguous span when at least 80% of it is
  linked, else by gathering the linked arms' points. The full (A, 12, 2) float64 base array (192 B
  per arm) is no longer allocated, which is where the memory drop below comes from. Each arm's
  resample depends only on its own points, so the rows are bit-identical to a whole-container
  resample; a test checks this. Inside a caching scope the container's cached base is used.
- **Requested frames only (`frames=`).** `arm_deformation` and `tip_motion` take an optional bool
  mask or list of frame positions and compute only those output frames; the others are NaN. The lag
  reference may lie outside the request, and the requested values equal the whole-sequence run's.
  This is the Fig 4E case (the script loops over the whole clip but writes only the event window).
  Shapes are built only for the pairs that land in the request.
- **`compute_all` changes.** It no longer resamples every arm before the deformation (unless
  `shapes=True` asks for the resample). Fig 4D therefore resamples only the arms of its bout spans.
  For lag B (Fig 5), it builds the `last_arm_frame` links once
  (`kinematics.last_arm_frame_pairs`) and passes them to both the deformation and the tip motion.

`_reference.py`, the fixtures, `parity_check.py` and its tolerances are unchanged.

**Tests.** The existing bit-exact deformation assertions keep their strictness: they now pass
`kabsch='svd'` explicitly. The test files are `segments`, `edge`, `parity` and `regressions`, and
they compare against the per-frame dict loops, `_reference` and the fixtures. Each one gained a
companion check of the closed-form default against the same reference. That check requires
identical NaN and inf positions and finite values within the gate's rtol 1e-9.

New tests:

- closed form against the SVD at ≤ 1e-10 absolute on mirrored shapes (where the guard flips),
  collinear (rank-1) shapes, single-point shapes on either side, exact rotations and 1e6 offsets;
- non-finite shapes raise like the scripts;
- `frames=` against the whole-sequence run for both lags and both fits;
- the subset resample against a whole-container resample, bit for bit, and `base=`, a caching
  scope and the lazy path giving the same deformation;
- pair blocks of 37 and closed-form blocks of 5;
- the shared lag-B links in `compute_all`.

### Run

| | |
|---|---|
| Date | 2026-11-03 |
| CPU | Intel Xeon W-3365, 64 cores (shared workstation, load average 1.3–1.8) |
| Software | Python 3.12.13, numpy 2.4.3, scipy 1.17.1 |
| Before | `bf65377` (round 3 plus its review fix), in a separate worktree, with the same `bench.py` |
| After | this commit |

```
# before, then after (sequential, never overlapping)
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs micro,subset,synthetic \
    --data-dir RUNS/paper_retrace_f39dc82b --synthetic-arms 1000000 --repeats 5 --jobs 1 \
    --kernels "T0.2 resample_base,T3.1,T3.2,E2E compute_all fig5,E2E compute_all fig4d"
```

- **Timings.** Best of 5 after one warm-up, each job in a fresh single-threaded process,
  `--jobs 1`. The before and after runs ran one after the other, with no overlap. The kernels
  this round did not change came out equal in both runs: resample_base 1.66 → 1.66 µs/arm on the
  subset and 1.44 → 1.48 on synthetic, and tip_motion 0.31 → 0.31 and 0.26 → 0.25. Read the
  speed-ups as ±5%.
- **Inputs.** The same as rounds 1–3: micro is 1,226 arms, the subset 173,434 arms (7 flipping
  clips and 36 Sleepy segments), and the synthetic input the subset tiled to 1,000,416 arms. The
  T3 kernels receive their track IDs before timing (`_ids`), so they time the kinematics alone.
- **Memory.** "Kernel +MB" is the growth of VmHWM over the built input during the kernel (VmHWM is
  reset after the build), before → after.

### Per kernel (before → after, CPU µs/arm)

| Kernel | Micro | Subset | Synthetic | Speed-up, subset / synthetic | Kernel +MB, subset; synthetic |
|---|---:|---:|---:|---:|---:|
| **T3.1 arm_deformation lag B (Fig 5)** | 4.90 → 3.21 | 4.01 → 2.23 | 3.77 → 2.01 | **1.80× / 1.87×** | 10.8 → 10.2; **29.7 → 6.2** |
| **T3.1 arm_deformation lag A (Fig 4)** | 4.89 → 3.25 | 4.04 → 2.29 | 3.76 → 2.03 | **1.76× / 1.85×** | 10.3 → 10.6; **28.7 → 12.0** |
| T3.2 tip_motion (unchanged) | 0.41 → 0.41 | 0.31 → 0.31 | 0.26 → 0.25 | 1.0× | 0.4 → 0.4; 0.5 → 0.5 |
| **E2E compute_all fig5 (stored IDs)** | 8.36 → 6.28 | 16.09 → 14.34* | 5.42 → 3.54 | 1.12×* / **1.53×** | 21.5 → 20.6; **32.4 → 11.2** |
| **E2E compute_all fig4d (bout windows, stored IDs)** | 7.13 → 4.66 | 5.16 → 3.67 | 4.38 → 3.03 | **1.40× / 1.45×** | 11.4 → 7.8; 4.3 → 1.0 |
| E2E compute_all fig5 + relink | 27.07 → 25.14 | 23.22 → 21.30 | 22.04 → 20.25 | 1.09× / 1.09× | 29.2 → 29.0; 49.8 → 49.0 |
| *unchanged:* T0.2 resample_base | 2.30 → 2.40 | 1.66 → 1.66 | 1.44 → 1.48 | — | 16.5 → 16.5; 2.8 → 2.6 |

\* As in round 3, the subset's "stored IDs" row re-links its 36 Sleepy segments, which have no
stored IDs, so the tracker dominates it. On synthetic every arm has a stored ID.

Whole-process peak RSS on synthetic (input build plus kernel): lag B 445 → 424 MB, lag A
444 → 430 MB, and fig5 with stored IDs 450 → 429 MB. The kernel's own growth drops by 20–23 MB
per 1e6 arms, because the full base array is gone. That saving scales with the sequence: at
1e8 arms in one container it would be about 19 GB less. The re-link path is unchanged; the
tracker's per-block arm rows dominate it.

### Camera-year projection (3e8 arms, one core)

| Workload | Round 3 (`bf65377`, this run) | Round 4 |
|---|---:|---:|
| Fig 5 representations with stored IDs (synthetic basis) | 0.45 CPU-h | **0.30 CPU-h** |
| Fig 4D representations with stored IDs (subset basis) | 0.43 CPU-h | **0.31 CPU-h** |
| T3.1 deformation alone, lag B (subset basis) | 0.33 CPU-h | **0.19 CPU-h** |
| Fig 5 representations with a re-link (subset basis) | 1.94 CPU-h | **1.78 CPU-h** |
| One tracker pass; Fig 6D diagnostics (not re-timed) | 1.4; 8.4 CPU-h | 1.4; 8.4 CPU-h |

Against v0 (518f47e, 124.6 µs/arm on synthetic), the Fig 5 stored-ID path is now 35× faster
(3.54 µs/arm). A camera-year of Fig 5 with a re-link went from 54 to 1.8 CPU-h.

### Hotspots after round 4

T3.1 lag B on the synthetic input took 2.05 s profiled, against 3.86 s before:

| Function | Self s | Cumulative s | Note |
|---|---:|---:|---|
| `geometry._resample_base_block` + `_csr.seg_search` + `seg_cumsum` + `_seg_norm` | 1.33 | 1.48 | T0.2 base resample of the linked arms (98% of arms on this input): 72% |
| `kinematics._closed_rows` | 0.09 | 0.14 | the closed-form fit (was 2.0 s as SVD) |
| `kinematics._centre_planes`, `_shape_planes` | 0.15 | — | T1.6 shapes in plane layout, centred once per arm |
| joins (`argsort`, `searchsorted`, `unique`) and medians | ≈0.15 | — | unchanged |

T3 is now bound by the round-1 base resample. Its per-segment bisection (`seg_search`, 0.65 s)
is the largest item. An exact counting alternative was tried in this round: each node is placed
among the 12 uniform targets with a corrected `ceil(arc/step)`, then `bincount` and `cumsum`. It
gave identical indices but took 0.92 µs/arm against 0.61 for the bisection, so it was dropped.
In the stored-ID Fig 5 path, the base resample (1.5 µs/arm) and `arc_length_curvature`
(1.2 µs/arm) now make up most of the remaining 3.5 µs/arm. The re-link path is still about 90%
tracker (`_assoc.span_dist_rows` and `ArrayTracker.step`, as at round 3).

### Parity and tests at round 4

- `parity_check.py`, unchanged, full gate (micro, 7 flipping clips, and the Sleepy subset with all
  18 Fig 6D sequences): **4,213 checks, 0 failed, 3,773 bit-identical**. The 440 checks that are
  not bit-identical are all T3.1 deformation outputs (Fig 4D log deformation, Fig 4E lag A, Fig 5
  lag B, and their `compute_all` preset rows). Their largest relative difference is 7.7e-14,
  against the gate's 1e-9, and every NaN position is identical. All track IDs, assignments, drop
  decisions, masks and other floats remain bit-identical. Near-tie audit unchanged: 39,953 LSA
  calls, 0 below 1e-6, smallest margin 7.6e-4.
- The same gate with the rigid fit forced to `kabsch='svd'` (a wrapper that patches the fit in an
  in-process `--jobs 1` run; `parity_check.py` itself untouched): **4,213 checks, 0 failed, 4,213
  bit-identical**, as in rounds 1–3.
- `pytest tests/test_noodle_polylines*.py`: **116 passed** (109 + 7 new cases from 4 new test functions). The
  micro example runs, and its printed output is identical to round 3's.

## Round 5: I/O and scale-out

### Where the time went

Rounds 1–4 made the representations cheap: Fig 5 with stored IDs is 3.5 µs/arm. Profiling the
whole path from a file on disk then showed that reading the inputs cost more than computing
from them:

| Path (round 4, CPU µs/arm) | Read | Compute (Fig 5, stored IDs) |
|---|---:|---:|
| Backcast runner records (`*.jsonl`, round-16 pods; 60 files, 190,861 arms) | 53.5 | 3.5 |
| Released flipping clip (CSV + pickled npz; 7 clips, 65,953 arms) | 16.2 | 3.5 |
| Sleepy segment (compressed npz; 36 segments, 107,481 arms) | 2.6 | (re-link) |
| Container `.npz`, compressed (synthetic, 1e6 arms) | 1.3 | 3.6 |

The backcast records are what a camera-year will arrive as, so their adapter is the largest
remaining cost: about 4.5 CPU-h per camera-year against 0.3 CPU-h for the Fig 5 representations.

The real records' profile (round-4 adapter, 10.8 s under cProfile): `json.loads` 73%,
per-arm `np.asarray` of the polyline and identity lists 10%, the adapter's own loop 9%. About half of the numbers in a record are `polyline_source` and
`mask_polygon_source` (≈ 26 points per polyline and ≈ 250 per star mask), which the adapter
never reads. Parsing decimal text to doubles costs about 0.2 µs per number whatever does it
(per 1e6 float32-valued numbers: `json.loads` 0.23 s, `np.fromstring` 0.21 s, `np.loadtxt` 0.23 s),
so the only lever is not to parse what is not needed.

### What changed and why

All in `CORE/noodle_polylines/adapters.py`, `container.py` and `pipeline.py`; no kernel changed.

- **Inference adapter.**
  - The values of `"polyline_source"` and `"mask_polygon_source"` are cut from the text before
    `json.loads` (`_drop_unused_arrays`, one regex pass, `skip_unused=True` by default). The key
    cannot match inside a string (quotes there are escaped) and must follow `{` or `,`; the matched
    span runs greedily over number, bracket, comma and whitespace characters and backs off to the
    last `]]`. As committed, that was *not* always the value's own end (see "Fix after review
    (round 5)" below): the fixed pattern also requires the end of the object member after the span.
    `null`, `NaN`, flat or empty arrays are left to `json.loads`. The one behaviour change is that
    malformed text inside a cut array is no longer reported; `skip_unused=False` parses
    everything.
  - Polylines and identities are converted once per star, not once per arm: every number goes
    through one `np.fromiter` (float64, then cast to float32), the same single rounding as the
    per-arm `np.asarray(p, np.float32)`. Any shape the per-arm code would treat differently (flat
    lists, deeper nesting, wrong widths, non-lists) falls back to the per-arm conversion and its
    errors.
  - `.gz`, `.bz2`, `.xz` (stdlib) and `.zst` (Python 3.14's `compression.zstd`, else the optional
    `zstandard` package, else a clear `ImportError`) are read directly; the runner writes
    `*.jsonl.zst`.
- **Flipping adapter.** One `np.loadtxt` call per CSV with a structured dtype of only the needed
  columns replaces `csv.DictReader` and a Python `float()` per cell. numpy's text-to-double
  conversion is correctly rounded like `float()`; the CSV reader follows the csv module's quoting,
  treats `#` as text, and lets the last of duplicate header names win as a `DictReader` row does.
  The sort by (frame, arm_index) is a stable `lexsort`, and the row → polyline and frame →
  position dicts are `unique`/`searchsorted` lookups that keep the dicts' last-one-wins rule and
  raise `KeyError` on a missing key.
- **Sleepy adapter.** Only the 17 members the container uses are decompressed (not `mouth`,
  `anus`, `arm_length_px`, …), and stop reasons are mapped once per distinct text.
- **Container loads.** `PolylineContainer.load(path, mmap=True)` memory-maps every member stored
  uncompressed (`save(..., compressed=False)`) read-only, from the member's offset in the zip; a
  compressed member is read as before. `fields=(...)` reads only the named optional fields.
  `load_any` passes `mmap`, `fields` and `validate` on to `load`.
- **Pool.** `compute_many(..., workers=N)` runs sequences in a process pool and yields results in
  input order, with at most `max_pending` (default `2 * N`) sequences in flight or waiting. Each
  sequence is loaded and computed by the same function serially or pooled, so the outputs are
  identical (checked bit for bit). `out_dir=` makes each worker write `<name>.npz` and return only
  the path.
- **Bench.** `bench.py` gained I/O kernels (adapters, container loads, file → Fig 5), a
  `--pool-workers` scaling run that checks every pooled output against the serial run, written
  bytes per arm, and `NOODLE_POLYLINES_CORE=<dir>` to time another checkout with the same harness.
  `write_backcast_jsonl` writes runner-shaped records (with source polylines and a 250-point mask
  per star) from any container, so the JSON path is benchmarked without private data.

### Exactness

- **All released data, adapters old against new** (scratch scripts, not committed): all 44
  flipping clips × {runtime origin, support origin, frames 640–820} and all 192 Sleepy segments ×
  {runtime, support} give byte-identical containers (every field, dtype and meta).
- **Real backcast records**: 60 files of round-16 pod output (190,861 arms; `star_track_id` set
  from `detection_index`, see below), plus indented, compact and multi-star JSON, give
  byte-identical containers with `skip_unused` on and off, and from `.gz`.
- **New tests** (`tests/test_noodle_polylines_io.py`, 13 cases): the round-4 per-cell and per-arm
  readers are kept in the test file as oracles. They cover tricky decimal texts (signed zero,
  subnormals, 17+ digits, `nan`, `inf`), quoting, `#`, duplicate headers and header-only files;
  out-of-order rows, duplicate `row_id`s, frame ranges and missing columns; skipper edge cases
  (the key text inside a string, `null`, `NaN`, flat, empty and deeper arrays, indentation);
  polyline fallbacks; a missing `.zst` decoder; mapped and partial loads, including Fig 5 outputs
  from a mapped container; and pooled `compute_many` against serial, its ordering, `out_dir` and
  error propagation.
- `parity_check.py`, unchanged, full gate: **4,213 checks, 0 failed, 3,773 bit-identical**,
  the same as round 4. Its `A-FLIP adapter.*` rows (13 fields, on micro and on the 7 clips)
  compare the new flipping adapter with the store's input containers, exactly. Near-tie audit unchanged (39,953 LSA calls, smallest margin
  7.6e-4).
- `pytest tests/test_noodle_polylines*.py`: **129 passed** (116 + 13). The micro example runs.

### Float32 / float64 audit

- **Storage is lossless.** All 10,015,474 `polyline_crop` numbers and 1,526,888 identity numbers
  of the 60 backcast files are exact float32 values (the runner writes float32 arrays with
  `tolist()`), so the container's float32 points and identities lose nothing. The released
  flipping polylines are float32 in their npz, and the Sleepy points are float32.
- **Memory per arm** (subset, 26.4 points per arm): 280 B in RAM, of which points 212 B,
  identity 32 B, arm offsets 8 B, track ID 8 B for arms that have one, and the pass-through scores
  and per-frame fields the rest. On disk: 270–282 B uncompressed and 213–221 B compressed
  (float32 coordinates compress poorly). Offsets stay int64: a camera-year holds about 8e9 points.
- **Compute precision** is the figure scripts' (unchanged): float32 for the per-arm arc length and
  curvature and the tip-first cumulative length; float64 for the base-first resample (cast block by
  block, 8,192 arms at a time, never the whole point array), the rigid fit, medians and every
  per-frame output. Fig 5 outputs take 73 B per arm (float64 and int64); halving that with float32
  would break the gate's bit-identity, so it is not offered.

### Run

| | |
|---|---|
| Date | 2026-11-03 |
| CPU | Intel Xeon W-3365, 64 cores (shared workstation, load average 1.0–2.3) |
| Software | Python 3.12.13, numpy 2.4.3, scipy 1.17.1 |
| Before | `ab5db53` (round 4) `CORE`, timed by this round's `bench.py` (`NOODLE_POLYLINES_CORE`) |
| After | this commit |

```
# before, then after, sequential (K = "IO ,E2E file,E2E compute_all fig5 (stored,T0.2 resample_base")
NOODLE_POLYLINES_CORE=<ab5db53>/CORE OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py \
    --inputs micro,subset,synthetic --data-dir RUNS/paper_retrace_f39dc82b --synthetic-arms 1000000 \
    --repeats 5 --jobs 1 --kernels "$K"
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py ... (same) --profile profile_after.txt
# large synthetic: 1e7 arms, then fig5 + relink and the pool
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs synthetic --synthetic-arms 10000000 \
    --repeats 5 --jobs 1 --kernels "T0.2 resample_base,T0.3+T1.1,T3.1 arm_deformation lag B,E2E compute_all fig5 (stored,E2E compute_all fig4d,IO container,E2E file"
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs synthetic --synthetic-arms 10000000 \
    --repeats 3 --jobs 1 --kernels "E2E compute_all fig5 + relink" --pool-workers 1,8,32 --pool-repeats 3
```

- **Timings.** CPU µs/arm, best of 5 after one warm-up, each job in a fresh single-threaded
  process, `--jobs 1`, runs never overlapping. I/O files are written untimed to a temporary
  directory first, so timed reads come from the page cache. Kernels this round did not touch came
  out equal: resample_base 1.66 → 1.67 (subset) and 1.43 → 1.42 (synthetic), Fig 5 with stored IDs
  3.58 → 3.59 (synthetic). Read the speed-ups as ±5%, except the JSON row: a second "before" run
  of it gave 64.6 instead of 57.4 µs/arm, so read that one as ±12%.
- **Inputs.** As in rounds 1–4 (micro 1,226 arms; subset 173,434 arms; synthetic 1,040,604 arms),
  plus a 1e7 synthetic (the subset tiled 58×: 10,059,172 arms in 43 sequences). Each adapter row is
  per arm it reads (flipping 65,953, Sleepy 107,481). The first "before" run divided the two
  adapter rows by all subset arms; they were rerun with the fixed harness and those reruns are
  quoted.

### Per kernel (before → after, CPU µs/arm)

| Kernel | Micro | Subset | Synthetic 1e6 | Speed-up | Kernel +MB, subset |
|---|---:|---:|---:|---:|---:|
| **IO adapter: inference JSONL (runner-shaped records)** | 60.15 → 33.08 | 57.36 → 35.41 | — | **1.82× / 1.62×** | 62 → 103 |
| IO adapter: real round-16 records (60 files, scratch script) | | 53.5 → 33.9 | | **1.58×** | |
| **IO adapter: flipping clips (CSV + pickled npz)** | — | 16.22 → 6.10 | — | **2.66×** | 23 → −1 |
| IO adapter: Sleepy segments (npz) | — | 2.60 → 2.05 | — | 1.27× | 7 → 0 |
| IO container load, compressed (unchanged path) | 2.88 → 2.97 | 1.77 → 1.73 | 1.31 → 1.27 | — | |
| IO container load, uncompressed (unchanged path) | 1.34 → 1.52 | 0.58 → 0.55 | 0.26 → 0.25 | — | |
| *new:* IO container load, mmap (no validation) | 1.51 | 0.44 | **0.06** | 22× vs compressed | |
| *new:* IO container load, mmap + validation | 1.70 | 0.53 | 0.09 | 14× vs compressed | |
| **E2E file → Fig 5 (stored IDs)**: compressed before → mmap after | 8.92 → 8.19 | 15.78 → 14.03* | 4.94 → 3.73 | **1.32×** (synthetic) | |

\* The subset row re-links its 36 Sleepy segments, so the tracker dominates it.

The inference adapter's subset memory growth rises from 62 to 103 MB: the polylines of a star are
now held as Python lists until one conversion at the end, rather than as small arrays. That is
about 0.24 kB per arm of the file being read, released when its containers are built.

Whole-process peak RSS with the subset built by the adapters fell from 173–177 MB to 149–156 MB
(the CSV rows are no longer held as dicts).

### Large synthetic (1e7 arms, 43 sequences)

| Kernel | CPU µs/arm at 1e7 | at 1e6 (this round or round 4) | Peak RSS, MB (build 2,742) | Kernel +MB |
|---|---:|---:|---:|---:|
| T0.2 resample_base | 1.43 | 1.42 | 2,892 | 149 |
| T0.3+T1.1 arc_length_curvature | 1.08 | 1.10 (round 1) | 2,743 | 0 |
| T3.1 arm_deformation lag B | 1.93 | 2.01 (round 4) | 2,812 | 69 |
| E2E compute_all fig5 (stored IDs) | 3.54 | 3.59 | 2,843 | 100 |
| E2E compute_all fig4d (bout windows) | 3.42 | 3.03 (round 4) | 3,698 | **956** |
| E2E compute_all fig5 + relink | 20.04 | 20.25 (round 4) | 2,895 | 153 |
| IO container load: compressed / uncompressed | 1.24 / 0.22 | 1.27 / 0.25 | | 1,766 / 1,825 |
| IO container load: mmap / mmap + validation | **0.006** / 0.034 | 0.06 / 0.09 | | 0 / 2,241† |
| E2E file → Fig 5: compressed / mmap | 4.73 / **3.59** | 4.91 / 3.73 | | 131 / 121 |

Per-arm costs are flat from 1e6 to 1e7 arms: every kernel scales linearly, and memory grows with
the largest sequence, not with the input set. Loading a mapped 1e7-arm set takes 58 ms (headers
only). † Validation touches every mapped page, which RSS counts; those pages are file-backed page
cache, shared across processes and reclaimable, unlike the 1.8 GB of private arrays of a normal load.

Fig 4D is the exception. Its per-window outputs (`arm_deformation_window`, `log_deformation_window`)
are dense (W, F) arrays. The bench puts a bout every 120 s, so the longest tiled sequence (97k
frames, 406 windows) needs about 0.3 GB per array. The paper's panels have a handful of bouts,
but a camera-year run with many windows on day-long sequences would need a sparse form. That is
an output-shape change, so it is left for a decision rather than made here.

### Pool scaling (1e7 arms, Fig 5 with stored IDs, mmap loads, wall clock)

| Workers | Wall µs/arm | Speed-up | Identical to serial |
|---:|---:|---:|---|
| 1 | 3.60 | 1.0× | yes |
| 8 | 0.72 | 5.0× | yes (every array, bit for bit) |
| 32 | 0.60 (0.52 in a repeat) | 6.0–6.9× | yes |

The ceiling here is the input, not the pool: the largest of the 43 sequences holds 6.5% of the
arms, which caps the speed-up at about 15×. Results also travel back to the parent (73 B per
arm). `out_dir=` avoids that transfer but costs more than it saves, because each worker writes a
compressed npz (1.11 and 0.70 µs/arm at 8 and 32 workers). Ordering the paths largest first
changed nothing at this size. A camera-year split by video file, at thousands of sequences, does
not have this ceiling.

### Camera-year projection (3e8 arms)

| Workload (CPU-h, one core) | Round 4 | Round 5 |
|---|---:|---:|
| **Convert backcast records → containers** (runner-shaped / real records) | 4.8 / 4.5 | **3.0 / 2.8** |
| Load the containers (compressed npz → mmap of uncompressed npz; 1e7 basis) | 0.10 | **0.0005** |
| Fig 5 representations from container files, stored IDs (1e7 basis) | 0.39 | **0.30** |
| Fig 4D representations, stored IDs (1e7 basis; code unchanged) | 0.29 | 0.29 |
| Fig 5 representations with a re-link (1e7 basis; code unchanged) | 1.67 | 1.67 |
| **Whole path from backcast records**: convert, re-link, Fig 5 | 6.5 | **4.6** |

The 1e7 basis replaces round 4's subset and 1e6 bases for the unchanged rows (round 4 quoted
0.31 and 1.78 CPU-h on those); the per-arm costs agree within 6–13%, the subset's Sleepy
segments being a little dearer. Files are independent, so the whole path splits across workers:
ideally 4.6 / 32 ≈ 0.15 wall-hours per camera-year on 32 workers when the files are many and
similar in size (this bench's 43 unequal sequences reached 6×, above). A converted container
set is read back for any further preset at 0.3 CPU-h per pass (stored IDs).

### Hotspots after round 5

- **Text parsing**, 22 of the 34 µs/arm of the backcast conversion. What remains is
  `json.loads` of `polyline_crop`, `identity` and the record skeleton. Text-to-double conversion is
  the floor (≈ 0.2 µs per number), so the next step would be a binary inference output (float32
  CSR, the container itself) or an optional fast parser, both outside this package's scope.
- **The tracker**: `_assoc.span_dist_rows` and `ArrayTracker.step` (≈ 73% of the subset
  profile of Fig 5 + re-link), unchanged since round 3.
- **The representations**: 3.5 µs/arm (Fig 5, stored IDs), led by the base resample and the arc
  metrics as at round 4.

### Fix after review (round 5)

A review found that the default `skip_unused=True` path failed on some valid JSON that
`skip_unused=False` and the round-4 reader both read. When a `polyline_source` or
`mask_polygon_source` value was nested three or more deep and held a non-number after an inner
`]]` (for example `[[[1,2],[3,4]],[[null,5]]]`, `[[[1,2]],[[true,5]]]`, `[[[1,2]],{"x":1}]`,
or Python's `[[[1,2]],[[NaN,3]]]`), the greedy span backed off to the inner `]]`, only part of the
value was cut, and `json.loads` raised `JSONDecodeError` on the rest. The commit's claim that the
cut span is always exactly the value was wrong. It never produced a wrong container (the leftover
text always made the parse fail), and the current producers (`runtime.py` and the backcast runner)
write two-level float lists, which were cut correctly; but it was a correctness bug.

- **Fix.** The pattern now also requires, by a lookahead, that the span be followed by the end of
  an object member: `}`, or `,` and the next key's `"...":`. Inside an array a `]` is followed by
  `,` or `]`, and a string element is never followed by `:`, so in valid JSON only the value's
  own end qualifies. A value holding anything but arrays and numbers is now left whole to
  `json.loads`. Values made only of arrays and numbers and ending in `]]` are cut at any depth.
  (A stricter pattern accepting only arrays of flat numeric arrays was tried first: it was also
  correct but ran the regex at about half the speed, +4.5 µs/arm on the real records; a
  bracket-count check after the old pattern cost +2 µs/arm.)
- **Checks.** A randomized test of `_drop_unused_arrays` (318,058 documents, nine seeds: random nesting,
  `NaN`, `Infinity`, `null`, booleans, strings holding the key text, `]]` or escaped quotes, keys
  with escaped quotes, objects inside the values, compact and indented dumps) gives exactly
  `json.loads` of the uncut text with each numbers-only value ending in `]]` replaced by `null`.
  An end-to-end randomized comparison of `skip_unused=True` with `False` (2,128 files: JSON and
  JSONL, single- and multi-star, compact and indented, plain, `.gz`, `.bz2`, `.xz`) gives 0
  mismatches; the same comparison on the round-5 commit gave 792. All 211,587 source arrays of the
  60 real backcast files are still cut, and their containers are byte-identical with
  `skip_unused` on and off. New test: `test_drop_unused_arrays_leaves_other_values_to_json`
  (12 cases; 5 fail on the round-5 commit).
- **Cost.** Real round-16 records 33.4 → 33.7 µs/arm (round-5 commit → fixed, best of 5, three
  alternating runs each); runner-shaped records (subset) 35.3 → 35.7. Both are within the ±5%
  stated above, so the round's numbers stand: 53.5 → 33.7 (1.59×) on the real records and
  57.4 → 35.7 (1.61×) on runner-shaped ones.
- **Gate.** `parity_check.py`: 4,213 checks, 0 failed, 3,773 bit-identical (unchanged).
  `pytest tests/test_noodle_polylines*.py`: 141 passed (129 + 12).

### Findings for the camera-year (not changed here)

- The round-16 backcast records have `"star_track_id": null` (the runner runs without its live
  star tracker) and carry `arm_index` instead of `track_id`. `from_inference_json` refuses
  null star IDs, as it did before, because stars cannot be told apart in multi-star frames (3,433 of
  7,340 frames in the 60 files have two or more detections). These records need a star association
  before conversion, and then an arm re-link (the "with a re-link" rows above).
- Fig 4D's dense (W, F) window outputs (above).
