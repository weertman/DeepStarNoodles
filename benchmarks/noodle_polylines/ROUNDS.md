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
