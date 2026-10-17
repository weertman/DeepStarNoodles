# Published arm tracer (selected 2026-10-17)

## Decision

The arm tracer DeepStarNoodles publishes is the **full-corpus recurrent GRU, cycle 30, seed
20261011**. It runs with its own state rules and an invalid-seed threshold of 0.6, on the
endpoint-refined Hydra that the pipeline already uses.

| Item | Value |
|---|---|
| Tracer checkpoint | `MODELS/tracer/noodle_tracer_fullcorpus_seed20261011_calibrated.pt` |
| SHA-256 | `f39dc82b9bc5d38e2053c2535ca30055a9d03140a713dbb991cb7058130d5287` |
| Inference Hydra (unchanged) | `MODELS/hydra/endpoint_mse_best.pt`, SHA-256 `dcf7140c…3e02` |
| Invalid-seed threshold (stored in the checkpoint) | 0.6 |
| Replaces | `noodle_tracer_seed20260807_endpoint_mse_calibrated.pt` (Sleepy-only, threshold 0.4) |

The weights are byte-identical to the training checkpoint
`DeepNoodleTracker/runs/gru_corpus_vs_sleepy_local_20261011/arms/full/training/seed20261011/g0030/last.pt`
(SHA-256 `d546eea4…926a`). The published file adds the threshold and a `publication` metadata
block, and drops the optimizer state. The DeepStarNoodles runtime
(`deepstarnoodles.noodle_tracer`) loads it offline and reproduces the evaluated outputs
exactly: same arms, 0 px difference.

## What the model is

- **Architecture:** the same v2 multimodal CNN-GRU as the previous release (18 Hydra
  channels, 8-D arm embedding, 33 px patch, frozen MobileNetV3 RGB features, hidden size
  128).
- **Training data:** all 4,062 annotated training images, not just Sleepy Stars:
  - starseg: 2,470
  - starseg_lab: 749
  - arm_pit: 413
  - axis_detector: 27
  - sleepy_stars: 403
- **Training recipe:**
  - 30 cycles of 4,000 updates each, at batch size 1.
  - Every cycle draws a fresh augmentation of each image and reruns frozen Hydra
    (`epoch_060.pt`) on the augmented image.
  - Learning-rate warmup, then cosine decay from 3e-4 to 3e-5.
  - Rollout training ramps to 30% by cycle 12.
- **Normalization:** fixed, computed from non-Sleepy training pixels.

## Why this model

All selection was done on validation, by a rule written before the numbers existed
(`MODEL_SELECTION_PROTOCOL.md`). The test set was used once, after a lock.

1. **Full corpus over Sleepy-only (user decision, supported by the data).** On fixed
   non-Sleepy validation (444 images, 177 groups), the full-corpus GRU scored 0.855 against
   0.835 for a matched Sleepy-only GRU. On Sleepy validation the two were within noise (34
   images, 3 groups).
2. **Cycle 30 is fully trained.** Training continued to cycle 64: 25 cycles at a constant
   learning rate of 1e-4, then an anneal back to 3e-5, stopped partway through cycle 65 at
   the user's request.
   - Non-Sleepy validation F1 stayed flat at about 0.854 ± 0.002.
   - Cycle 64 minus cycle 30: −0.001 [−0.004, +0.002].
   - By the tie rule, cycle 30 is kept.
3. **Hydra version.** The original and endpoint-refined Hydra give the same tracer accuracy
   (within 0.001 F1). The refined one is kept, so the inference stack doesn't change.
4. **Threshold 0.6.** Chosen on non-Sleepy validation (mean of 3 seeds) from
   {default, 0.3, 0.4, 0.5, 0.6}.
   - Gain over the default: +0.003 F1 [+0.001, +0.006]. It trades about 1.4 points of
     precision for about 2 points of recall.
   - A check beyond the grid found a flat plateau: 0.7 and 0.8 are within 0.002, and
     accepting every tip is worse.
   - It makes no difference on Sleepy validation.
5. **Better than the incumbent on validation.** Against the released tracer as shipped,
   non-Sleepy validation F1 is +0.011 [+0.004, +0.017] for the shipped seed. On Sleepy
   validation it's +0.008 [−0.003, +0.022], with only 3 groups.

## Locked test result

Scored once, on CPU in float32, under `selection/analysis_lock.json` (written 2026-10-17
05:58 UTC). Both models use the endpoint-refined Hydra.

| Test set | Model | F1 | Group-macro F1 | Precision | Recall | Chamfer px |
|---|---|---|---|---|---|---|
| Sleepy, 63 images, 5 groups | **Published (full-corpus)** | **0.868** | 0.855 | 0.912 | 0.828 | 3.14 |
| Sleepy | Previous release | 0.838 | 0.824 | 0.859 | 0.817 | 3.40 |
| Non-Sleepy, 439 images, 178 groups | **Published (full-corpus)** | **0.854** | 0.836 | 0.888 | 0.822 | 2.72 |
| Non-Sleepy | Previous release | 0.837 | 0.818 | 0.859 | 0.816 | 2.85 |

Published minus previous release, paired group bootstrap, 95% intervals:

- Sleepy: **+0.030 F1 [+0.005, +0.054]**, 5 groups
- Non-Sleepy: **+0.017 F1 [+0.009, +0.026]**, 178 groups

This evaluation reproduces the previous release's original locked Sleepy test report
(F1 0.8366, precision 0.859, recall 0.815) as 0.8375, 0.859 and 0.817. The gap is about one
arm, which confirms the pipeline matches.

## Caveats

- **The 63-image Sleepy test set has now been viewed four times:** the original release, the
  2026-10-03 RunPod readout, and this readout, which scored two models. Its interval rests
  on 5 groups. The non-Sleepy test set (178 groups) is the stronger evidence.
- **One shipped seed out of three.** All three seeds score within about 0.005 F1 of each
  other on validation.
- **Selection used validation only.** Nothing was changed after the test.
- **Inference speed** on an RTX A6000, from the decoder benchmark: about 50 ms per image at
  batch 16, plus Hydra.
- **The one-shot decoder** was evaluated and deliberately left out of DeepStarNoodles; it's
  kept for a follow-up paper.

## Release changes (2026-10-17)

- `CORE/deepstarnoodles/paths.py`: `DEFAULT_TRACER_CHECKPOINT` now points to the new file.
- `config/artifacts.json`:
  - the new tracer has role `released-tracer`;
  - the previous tracer is kept as `previous-tracer` and marked optional;
  - the new lock, selection and test records are `recipe-lock` and `heldout-eval`.
- `config/locked/`: `fullcorpus_tracer_analysis_lock.json`,
  `fullcorpus_tracer_selection.json` and `fullcorpus_tracer_test_readout.json`, with local
  paths redacted.
- Hugging Face `weertman/deepstarnoodles`: the new checkpoint and an updated model card. The
  previous tracer file stays.
- README: the results and recipe sections are updated.

Not changed: the manuscript workspace under `paper/`. Its arm-recovery numbers (0.8366 on
63 images, 0.8219 on 502 images) still describe the previous tracer.

## Evidence

Everything is under `DeepNoodleTracker/runs/gru_corpus_vs_sleepy_local_20261011/`:

- `MODEL_SELECTION_PROTOCOL.md`: the rule, written before the sweep, with its amendment.
- `selection/SELECTION.md` and `selection.json`: the validation sweep (152 results) and the
  choice.
- `selection/threshold_sensitivity.json`: the thresholds beyond the grid.
- `selection/analysis_lock.json`, `TEST_READOUT.md` and `test_readout.json`: the locked test.
- `CONTINUATION_PLAN.md` and `continuation/curve.csv`: training to cycle 64.
- `non_sleepy_validation_curve.json`: the original 30-cycle curve.
- The comparison with the Sleepy-only arm and the one-shot decoder:
  `DeepNoodleTracker/runs/oneshot_polyline_decoder_20261011/readout/FINAL_COMPARISON.md`.
