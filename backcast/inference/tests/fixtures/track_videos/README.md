# track-videos fixtures

Cut from real 0.5.0.dev12 `serve-videos` annotations (schema 4) of the backcast rehearsal:

- `flip_r1_f030-089.jsonl.gz`: 60 records (frames 30-89) of a 2 fps flipping clip with one star. BoT-SORT
  splits it (frames 39-40 untracked, a second ID from frame 41) and frame 75 has a near-duplicate box.
- `lab_stress_f000-039.jsonl.gz`: 40 records (frames 0-39) of a synthetic 1 fps lab clip, validation stills
  held 5 s each, 3-4 stars per frame.

To keep them small, `image_path`, `mask_polygon_source` and the arms' `polyline_source` are dropped,
`polyline_crop` is rounded to 0.01 px, `identity` is rounded to 4 decimals, and each record's `video` is replaced by
`flip_r1` or `lab_stress`. Nothing else is changed.

`dev12_botsort.json` holds, per fixture, the SHA-256 of the 0.5.0.dev12 `track-videos` output (BoT-SORT plus
`runtime.StarArmTrackerRegistry`, ultralytics 8.4.51) and its star and arm IDs per frame:
`[frame, [[star_track_id, [arm track_id, ...]], ...], number of untracked stars]`.
