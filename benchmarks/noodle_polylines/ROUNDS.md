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
