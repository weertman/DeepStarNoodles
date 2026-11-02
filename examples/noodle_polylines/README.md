# noodle_polylines micro example

`data/flip_example_01.npz` (with `data/flip_example_01.json`) holds 90 s of one sea star
in a lab self-righting video, sampled at 2 Hz. The animal turns over twice in that time.

| | |
|---|---|
| Frames | 181 |
| Traced arms | 1,226 (4–8 per frame) |
| Polyline points | 34,144 |

It contains GRU Noodle Tracer arm polylines (base→tip, 512-px crop coordinates) and the
per-frame context: crop, body origin, support area and centroid. It also has identity
embeddings and the stored `track_id`. There are no image pixels.

The file is the canonical `noodle_polylines` container. It loads with
`allow_pickle=False`. The JSON file gives the frame rate, the tracer and tracker settings,
code tables, and the two righting bouts on the 1-s grid. Load both with
`PolylineContainer.load('data/flip_example_01.npz', sidecar=True)`:

| Bout | Flip | Upside-down | Right-side-up | Analysis window |
|---|---|---|---|---|
| 1 | 331.0 s | 334.0 s | 365.0 s | 327–369 s |
| 2 | 370.0 s | 375.0 s | 398.0 s | 365–403 s |

## Run

```bash
python examples/noodle_polylines/run_example.py            # prints a summary
python examples/noodle_polylines/run_example.py --out out/ # also writes every output array (.npz)
```

It needs only numpy and scipy and runs in a few seconds. The script computes every
representation below twice:

- with the **stored** track IDs, which came from a tracker warm-started over the whole
  original recording;
- with a **cold-start re-link** of the excerpt alone, which agrees with the stored IDs on
  1,179 of the 1,226 arms under the best ID mapping.

## What each representation is

Units:

- **Model px:** the 512-px crop.
- **Source px:** the video frame.
- **R0:** the body radius `sqrt(support area / π)` in source px, fixed per bout or window.

See `docs/noodle-polylines.md` for the full definitions and figure sources.

| ID | Output | Unit | Figure panel |
|---|---|---|---|
| T0.2 | 12-point resample, base-first (deformation) and tip-first (tracker) | model px | 4D, 4E, 5B–D; 6A–D |
| T0.3 | arc length of each arm | model px | 4B, 4D, 4E, 5C |
| T0.4 | polylines and tips in video coordinates | source px | 4E, 5A |
| T0.6 / T0.7 | body radius and the R0 policies (per frame, pre-onset, outside windows, baseline) | px | 4B, 4D, 4E, 5B–D |
| T1.1 | curvature: summed turning angle per unit length | rad / model px | 4B, 4D, 4E, 5C |
| T1.2 | arm length in body radii | body radii | 4B |
| T1.3 | tip angle and radius about the body origin | deg, model px | 6A–D |
| T1.4 | arm paths in body radii, y up | body radii | 4B |
| T2.1 | arms per frame | count | 4B |
| T2.2 | fold asymmetry: mean tip offset from the body origin | R0 | 4D, 4E, 5B, 5C |
| T2.3 | arm-length spread (IQR, at least 4 arms) | R0 | 4D, 4E, 5C |
| T2.4 | median arm curvature | rad / R0 | 4D, 4E, 5C |
| T2.6 | tip isotropy (at least 4 tips) | 0–1 | 5A/B, 5D |
| T3.1 | arm deformation after rigid alignment; lag A (Fig 4) and lag B (Fig 5) | R0 / s | 4D, 4E, 5B, 5C |
| T3.2 | body-relative tip motion | R0 / s | 5B, 5C |
| T3.3 | tip angle per track over time (Fig 6C window, 360–402 s) | deg | 6B, 6C |
| T4.2 / T4.3 | tracker cost matrix and assignment at t = 372.0 s | model px | 6A, 6C |
| T4.4 | FusedTracker track IDs (stored and re-linked) | — | 4D, 4E, 5B–D, 6A–D |
| T4.5 | tracker self-consistency: continuation, forward–reverse agreement, jumps, stride precision | — | 6D |

The excerpt cannot show the following, which need longer recordings:

- Fig 5's ±300-s windows at 1 Hz;
- Fig 6D's long track lifetimes, which are capped at 90 s here;
- statistics across bouts and animals.

## Licence

The code is MIT (see `LICENSE` at the repository root). The example data in
`data/` (`flip_example_01.npz` and `flip_example_01.json`) are licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Please cite DeepStarNoodles
(`CITATION.cff`) when you reuse them.
