# noodle_polylines: polyline representations of the paper figures

`CORE/noodle_polylines` computes every arm-polyline quantity that the DeepStarNoodles
methods paper (Figs 1–6) draws or depends on. It starts from the per-arm output of the
released GRU Noodle Tracer and has three kinds of output:

- per-arm geometry;
- per-frame aggregates;
- tracked kinematics, with the optional FusedTracker association.

The package needs only numpy and scipy. It does not import `deepstarnoodles`, whose
`__init__` pulls in torch. It has no VAE, VAME or latent code.

It ships inside the `deepstarnoodles` distribution, whose `pip` dependencies include
ultralytics (and so torch), opencv and pandas. For a numpy + scipy-only environment, put
`CORE/` on `PYTHONPATH` (or copy `CORE/noodle_polylines`), or install the repository with
`pip install --no-deps .` next to `numpy` and `scipy`.

Every definition reproduces the figure scripts bit for bit. Where two panels define a
quantity differently, the library keeps both as an explicit, named option.
`noodle_polylines._reference` holds slow, line-by-line ports of the figure-script code,
and the tests and the parity gate compare the public functions against it.

## Data model

One container is one sequence: one star in one video, with frames in time order. It is
stored as CSR arrays in an `.npz` file that loads with `allow_pickle=False`. Below, A is
the number of arms, F the number of frames and P the number of points.

| Field | Shape, dtype | Required | Meaning |
|---|---|---|---|
| `points` | (P,2) f32 | yes | base→tip polylines, crop-512 model px, about 4 px spacing |
| `arm_point_offsets` | (A+1,) i64 | yes | arm `a` is `points[o[a]:o[a+1]]` |
| `frame_arm_offsets` | (F+1,) i64 | yes | frame `f` holds arms `o[f]:o[f+1]`; a frame may have 0 arms |
| `frame_index` | (F,) i64 | yes | strictly increasing |
| `time_s` | (F,) f64 | T3, R0 | otherwise `frame_index / meta['fps']` |
| `crop` | (F,3) f64 | source px, R0 | left, top, side in source px; `scale = side / 512` |
| `body_origin_model` | (F,2) f64 | body-relative, T4 | runtime origin: mouth peak, else support > 0.3 centroid, else (256, 256) |
| `body_origin_code` | (F,) i8 | no | 0 mouth, 1 support, 2 crop centre, −1 not traced |
| `support_area_model_px` | (F,) i32 | R0 | count of support > 0.5 pixels |
| `support_centroid_model` | (F,2) f64 | T2.6 | support > 0.5 centroid |
| `has_star` | (F,) bool | no (default true) | the frame was traced |
| `source_frame_index` | (F,) i64 | no | original video frame when `concat` renumbered `frame_index` |
| `identity` | (A,8) f32 | T4 | arm identity embedding |
| `track_id` | (A,) i64 | no | stored association; −1 = none |
| `tip_score`, `base_score`, `state_confidence`, `steps`, `stop_reason` | (A,) | no | pass-through; no figure uses them |
| `meta_json` | str | yes | `fps`, `model_size`, `trace_contract`, `tracker_config`, `source_format`, … |

`PolylineContainer.load(path)` reads only the `.npz`. `load(path, sidecar=True)` also merges
the sibling `<stem>.json` (as the micro example ships its metadata); it is never read
implicitly, because a sibling JSON may be an unrelated file such as the `infer_video`
output the container was converted from.

For large data, save containers uncompressed (`c.save(path, compressed=False)`, about
280 B per arm against about 215 B compressed) and load them with `load(path, mmap=True)`:
every stored array is then memory-mapped read-only, so loading reads only the headers and a
pool of workers shares one page-cache copy. `validate=False` keeps the load lazy (validation
reads every point). `fields=('time_s', 'track_id')` reads only the named optional fields;
the CSR arrays and `frame_index` are always read.

Precision. The tracer's polylines and identities are float32 values (all 10.0 million
polyline numbers and 1.5 million identity numbers of the round-16 backcast records are exact
float32 values), so float32 storage loses nothing. Computations use the dtype of the figure
script they reproduce: float32 for the per-arm arc length and curvature and the tip-first
resample's cumulative length, float64 for the base-first resample, the rigid fit, the medians
and every per-frame output. Offsets are int64 because a camera-year holds about 8e9 points.

The data have two "centres" per frame, and the difference is deliberate:

- The **runtime body origin** (`body_origin_model`) is the origin of every arm-relative
  quantity: tracker inputs, tip angle, fold asymmetry, tip motion, body paths and ghosts.
- The **support centroid** (`support_centroid_model`), together with `support_area`, is used
  for R0 and tip isotropy. Tip isotropy is translation-invariant, so the centroid only
  decides which frames count.

### Adapters

Each adapter runs once, offline.

| Function | Input |
|---|---|
| `adapters.from_flipping_clip(folder, frame_range=None)` | released flipping clip: `_frames.csv`, `_arms.csv` and `_polylines.npz` |
| `adapters.from_sleepy_segment(path)` | Sleepy Stars `seg_*.npz` (already CSR) |
| `adapters.from_inference_json(path, fps=None, skip_unused=True)` | `SCRIPTS/infer_video.py` output, as single-star JSON, multi-star schema-2 JSON or JSON Lines, plain or `.gz`, `.bz2`, `.xz`, `.zst` |
| `adapters.load_any(path, **kw)` | picks one of the above; for a container `.npz` it passes `mmap`, `fields` and `validate` to `load` |

- **Flipping.** The clip's `_polylines.npz` is a pickled object array, so this adapter is the
  only place that reads with `allow_pickle=True`. Use it only on re-traces you produced
  yourself.
- **Sleepy.** The segments have no `track_id`; re-link them.
- **Inference.** The output is split into one container per `star_track_id`. Inference records
  have no body origin, timestamp or support area, so only non-body-relative T0/T1 and the
  association can be computed from them. If any arm lacks `identity`, that star's
  `identity` is `None`, and a re-link raises `ContainerError`. The `polyline_source` and
  `mask_polygon_source` arrays are never used; by default their text is cut before
  `json.loads` (about half the numbers of a backcast record), which gives the same container.
  Only a malformed number inside a cut array goes unreported; `skip_unused=False` parses
  everything. `.zst` needs Python 3.14 or the optional `zstandard` package. Records whose
  `star_track_id` is `null` (the backcast runner without its live star tracker) are refused,
  as before: they need a star association first.

The CSV adapter parses the columns it needs in one `np.loadtxt` pass. Every value equals
`float()` or `int()` of its text, as the figure scripts parse it, and the converted
containers of all 44 released flipping clips and all 192 Sleepy segments are byte-identical to
the per-cell readers' (checked in round 5).

Helpers:

| Function | Purpose |
|---|---|
| `container.concat(parts, sort_by_time=True, dedupe_time_decimals=None, track_ids='auto')` | joins the segments of one animal; renumbers `frame_index` when the parts are separate videos (originals kept in `source_frame_index`), and then offsets each part's stored `track_id` so separately tracked parts never share an ID |
| `container.split_at_gaps(c, max_gap_s, min_frames)` | builds the Fig 6D Sleepy sequences (gap 16 s, at least 30 frames); their diagnostics index is `round(t − t0)` (`tracking.sequence_index(c, by='time')`), which `diagnostics` picks by default for Sleepy containers |
| `c.time_window(lo, hi, relative_to)` | cuts a window, e.g. a Fig 5 ±300-s boundary |
| `c.select_frames(...)` | selects frames by index |

## Representation catalogue

Level: A = per arm, P = per point, F = per frame, W = per window or bout, S = per sequence.
"Trk" means the representation needs `track_id`. Source px = `x · side/512 + left`.
R0 is in source px.

### T0 primitives

| ID | Level | Trk | Function | Definition | Units | Panels |
|---|---|---|---|---|---|---|
| T0.1 | A | no | container `points` | raw trace, base→tip | crop px | 1C, 2D, 3C, 4A |
| T0.2 base | A | no | `geometry.resample_base` | 12 points uniform in arc length, `np.interp`, f64; degenerate arms repeat the base | crop px | 4D, 4E, 5B–D (deformation) |
| T0.2 tip | A | no | `geometry.resample_tip` | tip first, f32, plus length L; degenerate arms (n = 1 or L ≤ 1e-6) repeat the tip with L = 0 | crop px | 6A–D, every re-link |
| T0.3 | A | no | `geometry.arc_length_curvature` | Σ‖ΔP‖ in float32 arithmetic (not the tracer's `length_px`) | model px | 4B, 4D, 4E, 5C |
| T0.4 | P/A | no | `geometry.to_source`, `tips_source` | `left + x·side/512` | source px | 4E, 5A |
| T0.5 | P | no | `geometry.body_relative` | `P − body_origin_model` | crop px | tracker input |
| T0.6 | F | no | `frames.body_radius(empty='nan'\|'zero')` | `sqrt(area/π)`, model and source px; empty support gives NaN (export) or 0 (Fig 5) | px | 4B, R0 |
| T0.7 | W/S | no | `scale.r0(policy, …)` | see the R0 policies below | source px | 4B, 4D, 4E, 5B–D |

### T1 per-arm geometry

| ID | Level | Trk | Function | Definition | Units | Panels |
|---|---|---|---|---|---|---|
| T1.1 | A | no | `geometry.arc_length_curvature(two_point=…)` | Σ\|Δ unwrap(atan2 seg)\| / max(L, 1e-6); a 2-point arm gives 0.0 (`'zero'`, export / Fig 4) or NaN (`'nan'`, Fig 5) | rad / model px | 4B, 4D, 4E, 5C |
| T1.2 | A | no | `geometry.norm_length` | L / max(r_model, 1e-6) | body radii | 4B |
| T1.3 | A | no | `geometry.tip_polar(origin_dtype, angle_dtype)` | `deg(atan2(−y, x)) mod 360` and ‖tip − origin‖ | deg, crop px | 6A–D |
| T1.4 | P | no | `geometry.body_path` | `((x − cx)/r, (cy − y)/r)` | body radii | 4B left |
| T1.5 | A | no | `geometry.anchored_ghosts` | arms of a window, re-anchored to the origin of the frame nearest the window centre | reference crop px | 5A |
| T1.6 | A | no | `geometry.shape12(order='fig4'\|'fig5')` | T0.2 base × side/512 / R0 | R0 | deformation input |

### T2 per-frame aggregates (untracked)

`variant` is `'fig4e'`, `'fig4d'` or `'fig5'`; see the presets below.

| ID | Level | Trk | Function | Definition | Units | Panels |
|---|---|---|---|---|---|---|
| T2.1 | F | no | `frames.n_arms` | arms per frame | count | 4B |
| T2.2 | F | no | `frames.fold_asymmetry(c, r0, variant)` | ‖mean tip − origin‖ / R0 | R0 | 4D, 4E, 5B, 5C |
| T2.3 | F | no | `frames.arm_length_spread` | IQR of L·scale / R0; NaN with fewer than 4 arms | R0 | 4D, 4E, 5C |
| T2.4 | F | no | `frames.arm_curvature(reducer=…)` | median of κ / scale × R0; `median` (Fig 4) or `nanmedian` (Fig 5) | rad / R0 | 4D, 4E, 5C |
| T2.5 | F | no | `frames.frame_lists_4b` | per-arm T1.2 and κ × r_model, and the per-frame mean of T1.2 | body radii | 4B right |
| T2.6 | F→W | no | `frames.tip_isotropy`, `frames.window_median` | `sqrt(max(λmin, 0)/max(λmax, 1e-9))` of the tip covariance (at least 4 tips); window (−300, −60) s; fallback NaN or 0.0 | — | 5A/B selection, 5D strata |

### T3 tracked kinematics

| ID | Level | Trk | Function | Definition | Units | Panels |
|---|---|---|---|---|---|---|
| T3.1 | F | yes | `kinematics.arm_deformation(lag=…, kabsch=…)` | median over linked arms of the Kabsch RMS residual of T1.6 / dt (closed-form rotation by default, `kabsch='svd'` bit-identical to the scripts). `last_arm_frame` (default, Fig 5) bridges gaps; `prev_frame_reset` (Fig 4) resets after a frame with no star or no arms | R0/s | 4D (as log10 with floor 1e-3), 4E, 5B, 5C |
| T3.2 | F | yes | `kinematics.tip_motion` | median ‖Δ((tip − origin)·scale/R0)‖ / dt over arms whose track was present in the last arm-bearing frame | R0/s | 5B, 5C |
| T3.3 | A | yes | `kinematics.tip_angle_track` | T1.3 with time and `track_id`; by default float32 origin and angles for flipping (6C), float64 otherwise (6B) | deg | 6B, 6C |

### T4 association (optional, FusedTracker)

| ID | Level | Function | Definition | Panels |
|---|---|---|---|---|
| T4.1 | pair | `tracking.common_span_dist` | mean point distance over the arc span both arms share, from the tip | 6A |
| T4.2 | F | `tracking.cost_matrix`, `tracking.step_costs` | `d + 2.0·‖e_i − e_j‖`; NaN where d > 120 | 6A, 6C |
| T4.3 | F | `tracking.assign` | Hungarian assignment augmented with "unmatched" options at 90/2 | 6A |
| (audit) | F | `tracking.assignment_margin` | how much more the cheapest assignment with different kept pairs costs; a near-tie check, not a representation | — |
| T4.4 | S | `tracking.relink`, `tracking.track_ids` | sequential tracker; frames without arms are skipped, so `max_age = 16` counts arm-bearing updates; IDs start at 0 | 4D, 4E, 5B–D, 6A–D |
| T4.5 | S | `tracking.diagnostics`, `rates`, `pooled` | continuation, forward–reverse Jaccard, jumps (> 30° with both radii ≥ 25 crop px), arm-weighted lifetime, stride precision (g = 2, 4, 8, 16), new-ID rate. Frame index: `frame_index` for flipping, `round(t − t0)` for Sleepy (`by='auto'` reads `meta['source_format']`); origin float32 for flipping, float64 otherwise (`origin_dtype='auto'`). The resolved choices are returned. | 6D |

**Which IDs.** `track_ids(c, mode='auto')` uses the stored IDs when every arm has one, and
re-links otherwise. On released flipping data the stored IDs equal a full-clip re-link
exactly. A cold-start re-link of an excerpt differs: on the micro example it agrees with
the stored IDs on 1,179 of 1,226 arms. Every output records which IDs it used
(`track_id_source`). With stored IDs, `compute_all` does not touch the tracker or its inputs.

**How the tracker runs.** With the released settings (`vel_alpha = 0`, float32 inputs, all finite)
a re-link uses an array-state port of the tracker (`noodle_polylines._assoc`). Each step computes the
whole (arms × tracks) shape and embedding blocks at once, builds the same augmented matrix and calls
`linear_sum_assignment` on it, so every matrix, assignment and label equals the reference tracker's
bit for bit (the parity gate compares all of them). The shorter arm of a pair is sampled on its own
grid, which is computed once per arm; only the longer arm is interpolated per pair. Embedding
distances use `np.vecdot`, which calls the same BLAS dot as `np.linalg.norm`. Arms whose length is
at most 1e-6 take the reference's tip-distance branch through the reference function. Any other
setting (a velocity term, non-finite or float64 inputs) runs the reference tracker unchanged. The
frame loop stays sequential; arm rows are built per block of about 32,768 arms. The Fig 6D counters
(continuation, agreement, jumps, births, lifetimes, stride precision) are sorted-key joins over the
whole sequence instead of per-frame sets.

### R0 policies (T0.7)

| Policy | Panel | Function | Definition |
|---|---|---|---|
| `per_frame` | 4B | `scale.r0_per_frame` | per-frame `sqrt(area/π)` in model px |
| `pre_onset` | 4E | `scale.r0_pre_onset(c, bout)` | median source-px radius over star frames with `first_time ≤ t ≤ last_time` and `t < flip_start` (no r > 0 guard) |
| `outside_windows` | 4D | `scale.r0_outside_windows(c, windows)` | median over star frames outside every `[start − 2, stop + 2]` |
| `baseline` | 5B–D | `scale.r0_baseline(c, onset_s)` | `t − onset ∈ [−300, −60]` s with r > 0; `None` (boundary dropped) with fewer than 30 frames |

Each policy also returns the matching support-area baseline, in source px².
`scale.in_any_window(t, windows)` is the closed-window mask: the windows are merged into disjoint
intervals and the times located with `searchsorted`, so it costs O((n + W) log W) for n times and
W windows.

## Presets

`compute_all(container, preset, ...)` returns a dict of named arrays for one panel's
definitions. The `ctx` column lists the context each preset needs.

| Preset | Panels | R0 | Variant | Lag | 2-pt κ | Reducer | Isotropy fallback | Tip origin | Tracker origin | ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| `FIG4B` | 4B | per_frame | fig4e | — | 0.0 | — | — | f64 | f64 | — |
| `FIG4D` | 4D | outside_windows | fig4d | prev_frame_reset | 0.0 | median | — | f64 | f64 | `windows=` |
| `FIG4E` | 4E | pre_onset | fig4e | prev_frame_reset | 0.0 | median | — | f64 | f64 | `bout=` |
| `FIG5` | 5B–D | baseline | fig5 | last_arm_frame | NaN | nanmedian | NaN | f64 | f64 | `onset_s=` |
| `FIG5_EXAMPLE` | 5A/B | baseline | fig5 | last_arm_frame | NaN | nanmedian | 0.0 | f64 | f64 | `onset_s=` |
| `FIG6_FLIPPING` | 6A, 6D | — | — | — | — | — | — | f32 | f32 (star frames) | — |
| `FIG6C` | 6C | — | — | — | — | — | — | f32 (f32 angles) | f32 (star frames) | — |
| `FIG6_SLEEPY` | 6B, 6D | — | — | — | — | — | — | f64 | f64 | — |

Panels differ in the last bits in a few places, and each preset reproduces its own panel:

- **Fold asymmetry.** Fig 5 averages float32 tips; Fig 4 averages float64 source-px tips.
  They differ by up to 3e-5 relative.
- **Order of scaling.** Fig 4 computes `L·side/512/R0`; Fig 5 computes `L·(side/512)/R0`.
- **Tip angle.** The Fig 6C kymograph (`prepare_figure_06_flip.py`) computes float32 angles of
  float32 tips (`FIG6C`); Fig 6D stats and the Fig 6B example (which casts polylines to float64)
  use float64 angles. The two differ by up to 2.2e-5 deg.

An explicit `r0=` overrides the preset's policy. Without the R0 context, only the R0-free
outputs are returned.

**Fig 4D deformation is per bout.** With `windows=`, `FIG4D` runs the deformation once per bout
window `[start − 2, stop + 2]`, taking shapes only from star frames in `[start − 3, stop + 2]`
(`prepare_figure_04_tree_features.py:121-157`). A frame whose predecessor lies more than 1 s
before the window is therefore NaN, as in the panel. `arm_deformation_window` and
`log_deformation_window` (W, F) hold each bout's values; the (F,) `arm_deformation` and
`log_deformation` take the first window that contains the frame and are NaN outside every
window.

Streaming over many files:

```python
from noodle_polylines import compute_many
for name, out in compute_many(paths, 'fig5', context={'seg_x': {'onset_s': 1.78e9}}):
    ...
```

This loads one container at a time, so memory is bounded by the largest sequence. Parallel
work splits by file; the tracker is sequential within a sequence:

```python
for name, path in compute_many(paths, 'fig5', r0=50.0, workers=32, load_kw={'mmap': True},
                               out_dir='outputs/'):
    ...
```

With `workers` > 1 the sequences run in a process pool. Results are bit-identical to the
serial run and come back in input order; at most `max_pending` (default `2 * workers`)
sequences are in flight, so memory is bounded by about that many sequences. With `out_dir`
each worker writes `<name>.npz` and only the path comes back. Set `OMP_NUM_THREADS=1` before
starting Python so the workers do not oversubscribe the cores.

The per-arm kernels (`resample_base`, `resample_tip`, `arc_length_curvature`) work on the whole
CSR array at once, in blocks of 8,192 arms, and give bit-identical results to the per-arm script code.
The per-frame aggregates (T2) and the tracked kinematics (T3.1, T3.2) are batched over frames in the
same way. Each frame's median, nanmedian or percentile is taken from a segmented sort. Tip means are
summed row by row, as `ndarray.mean(0)` does. Covariances run as stacked `matmul` and LAPACK
`eigvalsh` calls, which apply the same kernel to every slice. The track joins of T3 are sorted-key
lookups, which reproduce the scripts' per-frame dicts (the last arm of a track in a frame wins).
Apart from the T3.1 rigid fit, the results are bit-identical to the per-frame loops (a closed-form
2×2 eigen solve differs from LAPACK in the last bits, so it is not used).

The T3.1 rigid fit is the one exception, and it can be switched. By default (`kabsch='closed'`)
the 2-D Kabsch rotation is solved in closed form: for centred shapes, the best proper rotation has
`θ = atan2(Σ(x_a y_b − y_a x_b), Σ(x_a x_b + y_a y_b))`. That is the rotation the scripts' SVD with
its `sign(det)` reflection guard finds, but it needs about a quarter of the time. Over the 1.9M linked
pairs of the released corpora it agrees with the SVD to 2e-15 absolute and 5.7e-12 relative at
worst; the per-frame deformations agree to 8e-14 relative, with identical NaN positions. A block with
a non-finite shape runs the SVD, which raises `LinAlgError` as the scripts do. `kabsch='svd'`
(also on `compute_all`) evaluates the scripts' SVD expression on stacked pairs and is bit-identical
to them. Each linked arm's shape is built and centred once, and only linked arms are resampled.
`arm_deformation(frames=…)` and `tip_motion(frames=…)` compute only the requested output frames (the
lag reference may lie outside them), and Fig 4D runs each bout window on its own frame span, not
on the whole sequence. `compute_all` joins the lag-B links once for deformation and tip motion.
No call keeps derived arrays on a container, so memory is bounded by the call (`compute_all` shares
its resamples internally and frees them on return). To share them across several calls, open a
caching scope:

```python
with c.caching():
    ids = tracking.relink(c)                 # one tip resample ...
    res = tracking.step_costs(c, frame)      # ... reused here
```

Inside the scope `geometry.base_shapes(c)` (T0.2 base, T1.6/T3.1), `geometry.arm_metrics(c, two_point)`
(T0.3/T1.1, T2) and `tracking.tracker_inputs(c)` (T0.2 tip, every tracker pass) are computed once
and returned read-only. Per arm, the cache holds 192 B of base shapes (float64), 16 B of arc length
and curvature and 100 B of tip resamples and lengths (float32). It is dropped when the outermost scope
exits or a field is rebound (as `with_fields` does), and is never pickled. After an in-place edit of
a field's array inside a scope, call `c.clear_cache()`.

## Running

```bash
# the micro example: every representation, both ID paths
python examples/noodle_polylines/run_example.py [--out outputs/]

# tests: public API == _reference == golden fixtures; edge cases; torch-free import
python -m pytest -q tests/test_noodle_polylines*.py

# benchmark: micro, released subset, synthetic scale-up; camera-year projection
OMP_NUM_THREADS=1 python benchmarks/noodle_polylines/bench.py --inputs micro,subset,synthetic \
    --data-dir RUNS/paper_retrace_f39dc82b --synthetic-arms 1000000 --json bench.json

# parity gate against the reference store (figure-script outputs; not in the repo)
python benchmarks/noodle_polylines/parity_check.py --ref <store>/ref --data-dir RUNS/paper_retrace_f39dc82b
```

The parity gate has three tolerance classes:

- **Exact:** integer, label, mask and decision outputs.
- **≤ 1e-5 px:** the float32 tip-first resample.
- **rtol 1e-9:** every other float, with identical NaN positions.

The gate also reports how many outputs are bit-identical, and runs a near-tie audit: the decision
margin (`tracking.assignment_margin`) of every tracker assignment in its re-links, with margins below
1e-6 listed. The audit informs; it never fails the gate.
`benchmarks/noodle_polylines/ROUNDS.md` records the timings of each optimisation round.

## Not included

- Inference itself.
- Dense companion columns: side index, support area as a measure, and body speed.
- Statistics, bootstraps, running medians and plots.
- The VAE / VAME latent pipelines.
- The archived DTW tree.
- The older `deepstarnoodles.representations` (`arm_bag`, `arm_motion`), which use other definitions.
