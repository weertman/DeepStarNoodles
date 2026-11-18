# Representations

`deepstarnoodles-inference representations` turns inference output into the polyline representations of the DeepStarNoodles paper figures. It uses `noodle_polylines`, vendored unchanged from DeepStarNoodles (`UPSTREAM.json` pins the commit and file hashes). The command needs no GPU, PyTorch or OpenCV, so it runs anywhere the package installs.

```bash
deepstarnoodles-inference representations results/ --output reps/ --csv
deepstarnoodles-inference representations clip.jsonl --output reps/ --preset fig5 --onset-s 331
zstd -dc video.jsonl.zst | deepstarnoodles-inference representations - --name video --output reps/ --untracked single
```

## What the records need

Output from 0.5.0.dev11 onward (record `schema_version` 3, or 4 from 0.5.0.dev12, which adds landmarks the presets don't use) carries every field the presets use:

| Record field | Where | Container field |
|---|---|---|
| `crop` (`left`, `top`, `side`) | star | `crop` |
| `arms[].polyline_crop`, `identity`, `track_id` | arm | points, `identity`, `track_id` |
| `arms[].tip_score`, `base_score`, `state_confidence`, `steps`, `stop_reason` | arm | pass-through fields |
| `body_origin_model`, `body_origin_code` | star | runtime body origin and its source (0 mouth peak, 1 support > 0.3 centroid, 2 crop centre) |
| `support_area_model_px`, `support_centroid_model` | star | support > 0.5 area and centroid |
| `time_s`, `fps` | record | `time_s`, `meta["fps"]` |

The body origin is the runtime body centre that arm association uses and that the paper's re-trace exported. All of these are in 512-pixel crop coordinates. Inference computes them on the GPU from the Hydra maps it already holds, and they equal the host rules bit for bit (see `body_fields.py`). They match the paper's re-trace exactly only when the Hydra maps are identical: the same Hydra checkpoint, eager full mode and the same batch composition. Compiled or graph builds give values that agree closely but not bit for bit.

Older records (0.5.0.dev10 and earlier) have no body fields or time. They still load, but they give only `arc_length`, `curvature`, `tip_source` and `n_arms`, plus time from `--fps`. The body origin is never approximated from arm bases, because that would not match the paper. `--require-complete` turns a missing field into an error.

## Stars and tracking

| Records | Command | Containers |
|---|---|---|
| `live`, `live-replay` (`star_track_id` set) | default | one per `star_track_id`; stored arm `track_id`s |
| `archive`, `serve-videos`, `staged-archive`, one star per frame | `--untracked single` | one, `<input>__starsingle`; arms are re-linked |
| `archive` etc., several stars per frame | `--untracked detections` | one row per detection; no re-link, no windowed R0 |

Archive records have `star_track_id: null`, and by default the command refuses them, as upstream does. For `serve-videos` output with several stars per frame, run `deepstarnoodles-inference track-videos` first. It assigns star and arm IDs as live replay does, and its output reads with the default. `--untracked single` treats the file as one animal and makes frames without a detection `has_star=False` rows. It fails on the first frame with two detections. `--untracked detections` keeps every detection as an independent row: `frame_index` is renumbered and the video frame is kept in `source_frame_index`. Per-detection outputs (T0 to T2 and Fig 4B) are valid. Re-linking and the windowed R0 policies would mix animals, so they give an error row.

For a tracked star, every frame between its first and last appearance where it is missing becomes a `has_star=False` row, as in the paper re-trace, so Fig 4 frame-to-frame deformation resets across the gap instead of bridging it. `--no-fill-gaps` turns this off.

## Frame rate

`time_s` comes from the records. `serve-videos` and `live --video` read the video's rate, and image-index commands take `--fps` when inference runs. For records without a rate, pass `--fps` here. The command refuses a value that disagrees with the records, and it refuses records that disagree with one another.

## Presets and R0 context

`--preset` takes one or more of `fig4b`, `fig4d`, `fig4e`, `fig5`, `fig5_example`, `fig6_flipping`, `fig6c` and `fig6_sleepy`, or `all` (the default). Presets whose R0 needs context compute their R0-free outputs without it. Supply the context with one of these options:

- `--onset-s S` gives the Fig 5 baseline onset.
- `--bout FIRST_S LAST_S FLIP_START_S` gives the Fig 4E bout.
- `--windows-json FILE` gives a list of Fig 4D `[start_s, stop_s]` windows.
- `--r0 R` gives an explicit R0 in source pixels.
- `--context JSON` maps input or sequence names to per-sequence `compute_all` keywords, for example `{"clip__star3": {"onset_s": 331.0}}`.

`--track` chooses arm IDs: `auto` uses the stored IDs when every arm has one and re-links otherwise; `stored` and `relink` force one or the other. `--kabsch svd` reproduces the figure scripts' rigid fit bit for bit. `--shapes` also writes the resampled base shapes.

## Inputs and outputs

Inputs can be files (`.json`, `.jsonl`, plain or `.gz`, `.bz2`, `.xz` or `.zst`), directories, globs or `-` for stdin. Directories are searched recursively for `*.jsonl*`, skipping `.part`, `report.json`, `worker.*.json` and `service.json`. Reading `.zst` needs Python 3.14 or `pip install deepstarnoodles-inference[representations]` (zstandard); otherwise pipe through `zstd -dc`. An input is named by its stem; `predictions.jsonl` takes its directory's name. `--name NAME` overrides this when there is exactly one input (stdin or one file) and is refused with several. Each star becomes a sequence named `<input>__star<id>`.

The output directory must be new unless `--overwrite` is given. It holds:

- `containers/<sequence>.npz`: the pickle-free container (`PolylineContainer.load`), unless `--no-containers` is given.
- `<preset>/<sequence>.npz`: the `compute_all` outputs (`np.load`, no pickle).
- With `--csv`, `<preset>/<sequence>__frames.csv` (one row per container frame) and `<preset>/<sequence>__arms.csv` (one row per arm). They hold the container fields plus every per-frame or per-arm output. Multi-column arrays become `_0`, `_1`, ... columns, Fig 4D window arrays become one column per window, and NaN is written as an empty cell. Per-point arrays (`body_path`) stay in the `.npz`.
- `manifest.json` and `manifest.csv`, which record versions (including the vendored commit), the command line, every input, and each sequence and preset. Each sequence entry gives its frame and arm counts, frame rate, missing fields, the keys written and skipped, whether R0 was dropped, the R0 context, scalars, and its status and error.

A failing input or preset becomes an error row, and the run continues; the exit code is 1 if anything failed. `--workers N` processes input files in parallel, with output identical to a serial run. Set `OMP_NUM_THREADS=1` so the workers do not oversubscribe the cores.

## Python

```python
from deepstarnoodles_inference.representations import load_inference
from deepstarnoodles_inference._vendor.noodle_polylines import compute_all

stars = load_inference("live_run/predictions.jsonl")          # {star_track_id: PolylineContainer}
outputs = compute_all(stars[3], "fig5", onset_s=331.0)
```

See DeepStarNoodles `docs/noodle-polylines.md` for what each output means.
