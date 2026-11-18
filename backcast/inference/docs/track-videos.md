# Tracking after annotation

`deepstarnoodles-inference track-videos` gives the stars and arms in `serve-videos` records IDs that persist through a video. It needs no GPU, model weights or video, and runs one video per process:

```bash
deepstarnoodles-inference track-videos annotations/ --output tracked/ --jobs 8
deepstarnoodles-inference track-videos annotations/ --output tracked/ --star-param max_gap_s=60
deepstarnoodles-inference track-videos annotations/ --output tracked_dev12/ --star-tracker botsort
```

## What it does

1. **Star IDs.** By default (0.5.0.dev13 on) the post-hoc Hungarian star tracker (`star_tracking.py`) assigns them. `--star-tracker botsort` gives the 0.5.0.dev12 IDs instead (see [BoT-SORT](#bot-sort-the-050dev12-behaviour)).
2. **Arm IDs.** One arm tracker per star ID follows its arms, measured from the star's saved `body_origin_model`. It is the released FusedTracker (shape gate 120 px, embedding weight 2.0, max age 16 arm-bearing updates, match-cost cap 90, 12-point tip-first resample), run through the vendored `noodle_polylines` array tracker. It gives the same arm IDs as live replay's `runtime.StarArmTrackerRegistry` bit for bit, at a fraction of the CPU. Arm IDs are scoped to their star. A record where a star has no arms does not age that star's arm tracks.

Each star gets `star_track_id`, and each arm gets `track_id`; `arm_index` stays. A detection without a star ID moves to the record's `untracked_stars`, with null IDs, so `stars` plus `untracked_stars` are always exactly the record's detections. The output reads in `representations` without `--untracked`.

## The Hungarian star tracker (default)

At 1–2 fps a moving star travels about a body width between records. BoT-SORT matches boxes on overlap (IoU) and waits a frame or two before confirming a new track. At these rates it therefore splits one star into many IDs and holds each new track's first detection back. On the two 4-minute, 2 fps flipping clips of the backcast rehearsal, one star got 7 and 11 IDs. The Hungarian tracker was built for these records.

It prefers a new ID to a wrong one. An ID that moves to another animal puts that animal's frames into the first one's sequence, which no later step can undo; a split only shortens the sequences. Distances are in body widths: the distance between box centres divided by `sqrt(s_track * s_det)`, where `s = sqrt(box area)` and a track's size is the median of its last `size_hist` sizes. Time comes from the records' `time_s`, or `frame / fps`, or the frame number when neither is present, so missing records count like any other gap. For each record:

1. **Prediction.** Each track predicts its centre from its last centre and its velocity, an exponential average (weight `velocity_alpha`) of its steps, extrapolated for at most `velocity_horizon_s`. After a gap longer than `recent_s` the velocity restarts at zero.
2. **Cost.** A (track, detection) pair costs `d / sigma_t + ln(sigma_t / sigma) + size_weight * |ln(s_det / s_track)|`. Here `d` is the distance to the prediction and `sigma_t = sqrt(sigma^2 + sigma_growth * max(dt - 1, 0))` is the track's spread, which grows while it goes unseen. The log term makes a track seen long ago cost more than a fresh one at the same distance.
3. **Gates.** A pair is allowed when its cost is below `new_cost` and the sizes differ by at most `size_gate`. A track unseen for more than `recent_s` is lost. It may take a detection only within `min(reacquire_gate + reacquire_gate_per_s * dt, reacquire_gate_max)` widths of its predicted centre, and only when no other track is within `reacquire_clear` widths of that detection.
4. **Assignment.** One global assignment (`scipy.optimize.linear_sum_assignment`) over all pairs, where leaving a track or a detection unmatched costs `new_cost / 2` each.
5. **Ambiguity.** For each match with a rival (another allowed pair for its track or detection, including lost tracks that the gates kept away), the assignment is solved again without that match. If the best alternative gives the track another detection, or the detection to another track, and costs less than `ambiguity` more, the two readings are too close to call. The track ends, so does an unmatched track the alternative would hand the detection to, and the detection starts a new ID. This is what happens where two stars cross with their boxes on top of each other, where two boxes merge into one, or where one detection sits between two stars.
6. **Lone jumps.** A track seen within `recent_s` and left unmatched takes a detection up to `jump_gate` widths away when that detection is its only candidate and no other track or detection is within `jump_gate`. This keeps a star that is picked up and moved by hand, with nothing else nearby.
7. **New IDs.** Every detection still without an ID starts a new one in the record it appears in. IDs are numbered from 1 in order of birth. A track unseen for more than `max_gap_s` ends, and its arm tracker is dropped. An ended ID never comes back.

- **Duplicate boxes.** When two boxes in one record have IoU >= `dup_iou`, the less confident one gets no ID and goes to `untracked_stars`.
- **Determinism.** The tracker has no randomness. The order of detections within a record does not change which detections share an ID.

| Parameter | Default | Meaning |
|---|---|---|
| `sigma` | 0.25 | position spread, in widths, of a track seen within the last second |
| `sigma_growth` | 0.05 | growth of the spread's square (widths²) per second unseen, after the first |
| `new_cost` | 3.5 | price of a new ID plus an unmatched track; a dearer pair is never used |
| `ambiguity` | 0.4 | a match whose best alternative costs less than this more is ambiguous; 0 turns the check off |
| `velocity_alpha` | 0.3 | weight of the latest step in a track's velocity; 0 turns the motion model off |
| `velocity_horizon_s` | 5.0 | the velocity extrapolates at most this many seconds |
| `size_weight` | 0.5 | weight of the log size ratio in the cost |
| `size_gate` | 2.0 | max size ratio, either way |
| `size_hist` | 3 | a track's size is the median of its last N sizes |
| `recent_s` | 4.5 | a track unseen for longer is lost |
| `reacquire_gate` | 0.75 | a lost track takes a detection within this many widths, |
| `reacquire_gate_per_s` | 0.02 | plus this many per second unseen, |
| `reacquire_gate_max` | 1.5 | up to this many, |
| `reacquire_clear` | 1.5 | and only with no other track this close to the detection; 0 turns this off |
| `jump_gate` | 2.5 | lone-jump reach and clear radius; 0 turns lone jumps off |
| `max_gap_s` | 30.0 | a track unseen for longer ends |
| `dup_iou` | 0.95 | same-record IoU at which the less confident box is a duplicate; 0 turns it off |
| `min_conf_new` | 0.0 | detections below this confidence start no track |

`--star-param KEY=VALUE` overrides one parameter and may be repeated. Values are checked before any video starts, and unknown keys are refused.

### How it was checked

- **Single stars.** On the rehearsal's flip clips (two 4-minute clips at 2 fps, each also at 1 and 0.5 fps, with a 15 s gap and with 5% of detections removed), R2 keeps 1 ID in all five variants, and R1 has 2 in all five, where 0.5.0.dev12 gave 4–11. R1 splits once, where an experimenter moves the star 3.2 body widths after 2 s out of view. With other stars around, that move could not be told from another animal.
- **Several stars.** A synthetic set built from the flip clips' real detections (real misses, box jitter and size changes) moved along scripted paths: crossings, touching, rotating position swaps, a star out of view for 20 s while another walks over its spot, one animal leaving and another arriving, and crowds of four in 6×4 or 10×7 body widths, at 0.5, 1 and 2 fps, with both stars detected, one hidden, random drops or merged boxes. At 1 fps, against 0.5.0.dev12, wrong-animal detections fell from 6.3% to 1.0% in dense crowds (2.0% to 0.3% sparse, 1.0% to 0 on crossings), with fewer or as many ID switches in every scenario except hand-overs (6 against 3, below) and rotating swaps (3 against 0, on 3 detections). A second crowd set with other random walks gave 4.7% to 0.7% and 2.6% to 0.1%. With the clips' full natural motion, which includes experimenters moving stars by up to 2 widths, crowds improved as much, while crossings, a star out of view and rotating swaps came out about even with dev12 (wrong-animal detections within about 2 percentage points either way). BENCHMARK_RESULTS.md has the tables.

### Known costs

- **More, shorter IDs in crowds.** Where two stars could be confused, the tracker starts new IDs instead of guessing. In the dense synthetic crowds that gave about 2.5 times dev12's extra IDs (858 against 337 over 80 clips at 1 fps), and 1.4 times in sparse ones. Each is pure, so per-animal sequences are shorter but not mixed.
- **Hand-overs can merge.** With no appearance cue, a star that appears where another left looks like that star coming back. In 20 synthetic hand-overs at 1 fps, 6 merged (dev12: 3): a newcomer at the same spot within 30 s, or one width away within 3 s or after 20 s. Further away or later, the newcomer gets a new ID.
- **One-frame false positives get IDs.** BoT-SORT's confirmation delay used to hide them in `untracked_stars`. They are now short tracks of their own.
- **Not yet checked on real multi-star 1 fps footage.** The synthetic scenes use real detections but scripted motion. Check the default against a hand-checked real 1 fps lab segment with several stars before a large run.

## BoT-SORT (the 0.5.0.dev12 behaviour)

`--star-tracker botsort` runs BoT-SORT with the configuration live replay uses (`--tracker-config FILE` to change it). It has no camera-motion compensation, which needs the frames. Its output is byte-identical to 0.5.0.dev12 on current data; the tests pin it on two real fixtures. Stars BoT-SORT has not confirmed, which includes each star in the first record it appears in after the video's first, go to `untracked_stars`, as live replay never crops them. A star unseen for more than 30 frames loses its arm tracker. `--star-param` with `botsort`, and `--tracker-config` with `hungarian`, are refused.

One fix applies to this path. In ultralytics 8.4.51, a BoT-SORT row's detection index counts within the high- or low-confidence subset it came from, not within the record. `track-videos` now carries each box's record index through BoT-SORT's unused class column. With the shipped configuration (`fuse_score: true`) and `serve-videos`' 0.25 confidence floor no low-confidence box is ever matched, so the output is unchanged. Live replay's own BoT-SORT path (`crop.py`) is not changed.

## Recording what was used

- **Hungarian output.** Each tracked record gets a last key, `star_tracking`, of the form `{"tracker": "hungarian", "params": {...}, "version": "0.5.0.dev13"}`. Keys are only appended, as for the output schemas.
- **BoT-SORT output.** It gets no such key, to stay byte-identical to dev12. A tracked record that has `untracked_stars` but no `star_tracking` came from BoT-SORT, from dev12 or from `--star-tracker botsort`.
- **Summaries.** Each video's summary line on stdout gives its counts, the tracker and its parameters or BoT-SORT config. The line is also appended, with the package version, to `_track-videos.log` in the output directory (JSON lines; not named `.jsonl`, so that 0.5.0.dev12's `representations` does not read it as records). Files starting with `_` are never read as inputs by either command.

## Guarantees and refusals

- **Run once, on annotations.** Records that already have star IDs (`untracked_stars`, `star_tracking`, or a non-null `star_track_id`, as `track-videos`, `live` and `live-replay` write them) are refused, not tracked again. Records without a `stars` list are refused. Records from before 0.5.0.dev11 have no body origin and are refused.
- **Resuming and idempotence.** A video whose output exists is skipped unless `--overwrite` is given. Tracking is deterministic, so a rerun with `--overwrite` writes the same bytes.
- **Layout.** Directories are searched recursively, following directory symlinks, and mirrored under `--output`. So `cam1/C0001.mp4.jsonl.zst` and `cam2/C0001.mp4.jsonl.zst` stay apart. The fleet's `_manifest.jsonl`, the run log and `worker.*`, `report.*` and `service.*` files are skipped; any other record file starting with `_` is skipped with a note on stderr (pass it by name to track it). An `--output` inside an input directory is not read back on the next run, and another run's output inside an input directory (a subdirectory holding `_track-videos.log`) is skipped with a note. An `--output` equal to an input directory is refused. So are one video in two formats (`x.jsonl` next to `x.jsonl.zst`) and two inputs that would land on the same output. A file reached twice (named twice, or by name and through its directory) is tracked once. Finding no input is an error (exit 2).
- **Bad files.** A file without records is an error. So is a file with another record count than the fleet's `_manifest.jsonl` lists for it (the nearest one above the file, its last row for that video), which catches a cut exactly at a record boundary. A line that is not JSON, such as a truncated last line, is reported with the file, line number and last good frame. A truncated `.jsonl.zst` reports `zstd`'s own error. Blank lines are skipped. A failed video leaves no `.part` file, and the run goes on with the next video and exits 1 at the end.
- **Compressed files.** `.jsonl.zst` is read and written through `zstd` on `PATH`. Input goes to it on stdin, so symlinked files work.
