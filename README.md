# DeepStarNoodles

DeepStarNoodles is research software for extracting sunflower-star pose and
arm trajectories from images and video. Given footage of *Pycnopodia
helianthoides*, it detects each animal, estimates a dense pose field, traces
visible arms as base-to-tip polylines, and associates those observations
across video frames. It is intended for researchers studying posture and
movement from recorded imagery.

**Status:** research release, version 0.1.0. The released evaluation supports
per-image arm tracing. Video `track_id` values are causal associations for
visualization and downstream analysis; they are not validated measurements of
biological arm identity.

## Pipeline

```text
full frame
    -> StarSeg detector and square crop
    -> Hydra dense pose network
    -> Noodle Tracer arm polylines
    -> causal video association
```

- **StarSeg** detects sunflower stars and creates a square crop around each
  detection.
- **Hydra** combines a DINOv3 ConvNeXt-Base backbone with an FPN and prediction
  heads to produce an 18-channel dense field: ten geometry, landmark, and
  endpoint channels plus an eight-channel arm embedding.
- **Noodle Tracer** uses the image crop and dense field to generate one
  base-to-tip polyline for each detected arm.
- The **fused tracker** assigns video-local association IDs to arm
  observations.

## Quick start

Run these commands from the repository root. Python 3.11 is the tested
environment; a CUDA-capable GPU is recommended.

### 1. Create the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch 2.6 and torchvision 0.21 using the
[official 2.6 installation instructions](https://pytorch.org/get-started/previous-versions/#v260)
so the wheel matches the host CUDA version. Then install the remaining
dependencies and the Hugging Face CLI:

```bash
python -m pip install -r requirements.txt
python -m pip install --upgrade huggingface_hub
export PYTHONPATH="$PWD/CORE"
```

### 2. Download data and checkpoints

The runtime checkpoints are hosted in a gated
[Hugging Face model repository](https://huggingface.co/weertman/deepstarnoodles).
Request access, then download them into the checkout:

```bash
hf auth login
hf download weertman/deepstarnoodles --local-dir MODELS
```

The released Hydra checkpoints already contain the DINOv3 ConvNeXt-Base
backbone weights, so **Hydra inference (DNT, `HydraPredictor`, evaluation,
frozen-output caching) does not need any extra download**.

Only Hydra training from scratch (`CORE/deepstarnoodles/train.py`) initializes
the backbone from Meta's pretrained file, which is distributed separately under
Meta's DINOv3 License. If you intend to train, follow the
[official DINOv3 access instructions](https://github.com/facebookresearch/dinov3)
and place this file in `MODELS/dinov3/`:

```text
dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
SHA-256 801f2ba95801bdcc9a8b77e293d38b35b355dbc813ef609cd82e18c3a185e8da
```

Validate the runtime artifacts against the
[artifact manifest](config/artifacts.json) (add `pretrained-backbone` to the
roles only if you downloaded the Meta file for training):

```bash
python SCRIPTS/validate_artifacts.py \
  --roles runtime-detector released-dense released-tracer
```

Hydra and Noodle Tracer training and evaluation require the
[dense-pose dataset](https://huggingface.co/datasets/weertman/deepstarnoodles-data):

```bash
hf download weertman/deepstarnoodles-data \
  --repo-type dataset \
  --local-dir DATA/deepstarnoodles
```

StarSeg retraining requires the separate
[StarSeg dataset](https://huggingface.co/datasets/weertman/starseg):

```bash
hf download weertman/starseg \
  --repo-type dataset \
  --local-dir DATA/starseg
```

After downloading both datasets, validate the complete checkout:

```bash
python SCRIPTS/validate_artifacts.py
```

Full validation skips absent optional artifacts, including the YOLO26-L
training initializer and temporal VAE. Pass `--include-optional` to require
them.

### 3. Process a video

```bash
python SCRIPTS/infer_video.py \
  --video path/to/video.mp4 \
  --output RUNS/infer/tracks.json \
  --overlay RUNS/infer/overlay.mp4
```

The default mode follows one focal animal and writes frame-aligned arm
polylines and association IDs to `tracks.json`. The optional overlay provides
a visual check of the detector, crop, and traced arms. Each frame record
contains the frame number, detection status, crop metadata, and arm records.
Each arm includes its `track_id`, crop- and source-coordinate polylines, and
identity embedding.

## Reported results

### Released Noodle Tracer (October 2026)

The released tracer was selected on validation by a rule written before any
test access, then scored once on the held-out test splits under an analysis
lock. Both test splits are separated from training by `split_group_id`. Arms
are greedily matched one-to-one when their symmetric Chamfer distance is at
most 8 model-crop pixels; distances are in model-crop pixels.

| Test split | Arm F1 | Group-macro F1 | Precision | Recall | Mean Chamfer |
| --- | --- | --- | --- | --- | --- |
| Sleepy Stars, `source_family=sleepy_stars` (63 images, 5 groups) | **0.868** | 0.855 | 0.912 | 0.828 | 3.14 |
| All other source families (439 images, 178 groups) | **0.854** | 0.836 | 0.888 | 0.822 | 2.72 |

On the same images and Hydra outputs, the previous Sleepy-only tracer scored
0.838 and 0.837. The paired group-bootstrap differences in arm F1 are
**+0.030** on Sleepy Stars (95% CI +0.005 to +0.054) and **+0.017** on the
other families (95% CI +0.009 to +0.026).

- The Sleepy test split was viewed during earlier development, and its
  interval rests on five groups. The 178-group split is the stronger evidence.
- The readout ran on CPU in float32. It reproduces the previous tracer's
  archived Sleepy result (F1 0.8366) as 0.8375.
- Crossing-subset, base-termination, and geometric-baseline figures have not
  been computed for the released tracer.

The records are
[`fullcorpus_tracer_analysis_lock.json`](config/locked/fullcorpus_tracer_analysis_lock.json),
[`fullcorpus_tracer_selection.json`](config/locked/fullcorpus_tracer_selection.json),
and
[`fullcorpus_tracer_test_readout.json`](config/locked/fullcorpus_tracer_test_readout.json)
in `config/locked/`. The decision record,
[`docs/published-tracer.md`](docs/published-tracer.md), explains the selection
and its caveats.

### Previous tracer (August 2026)

The previous Sleepy-only tracer remains available as
`MODELS/tracer/noodle_tracer_seed20260807_endpoint_mse_calibrated.pt`. Its
archived report, [`config/locked/heldout_test.json`](config/locked/heldout_test.json),
covers the same 63-image Sleepy test split:

- Aggregate precision, recall, and F1 were **0.859**, **0.815**, and
  **0.837**. Mean and p90 Chamfer distances were **3.389** and **5.761**.
- Equal-group macro precision, recall, and F1 were **0.850**, **0.802**, and
  **0.824**. Group-macro mean and p90 Chamfer distances were **3.487** and
  **5.645**.
- Of 617 emitted traces, 616 terminated at the model's base condition
  (**99.84%**). This is a decoder termination statistic, not an independent
  biological accuracy measure.
- The embedding-aware Dijkstra geometric baseline had aggregate F1 **0.777**
  and group-macro F1 **0.757** on the same cached Hydra outputs. A paired
  bootstrap over the five split groups estimated an aggregate F1 improvement
  of **0.062** over that baseline (95% CI **0.043–0.097**; 2,000 resamples).
- Fifty of the 63 test images are marked as containing an arm crossing.
  Aggregate F1 on that subset was **0.795**, with mean Chamfer distance
  **3.663**.

That JSON report contains the artifact, cache, evaluator, and decoder hashes
used to produce it. It is archival provenance rather than a claim that the
current checkout can regenerate the report bit for bit.

## Limitations

- The released tracer was tested on held-out images from the dataset's own
  source families: 63 Sleepy Stars images and 439 from the other families.
  It may not transfer to other imaging conditions, life stages, or species.
- Performance is lower when arms cross or overlap. For the previous tracer,
  F1 on test images with a crossing was 0.795, against 0.837 overall. The
  released tracer's crossing subset has not been measured.
- Retraining reproduces the released recipe, not the released weights; see
  [Noodle Tracer training](#noodle-tracer).
- Geometric errors are reported in model-crop pixels. No physical-length
  calibration is provided by the inference pipeline.
- Video association has not been validated as biological arm identity.
- StarSeg inference depends on Ultralytics, and its training data and class
  definitions constrain detector generalization.

## Released recipe and data splits

The inference recipe consists of:

1. `MODELS/starseg/starseg_best.pt`, a YOLOv8 StarSeg detector for class
   `Pycnopodia_helianthoides`;
2. `MODELS/hydra/endpoint_mse_best.pt`, an endpoint-head fine-tune of the
   18-channel parent checkpoint `MODELS/hydra/epoch_060.pt`;
3. `MODELS/tracer/noodle_tracer_fullcorpus_seed20261011_calibrated.pt`, a
   Noodle Tracer trained on all source families, with a 33×33 dense patch,
   frozen MobileNetV3-Small RGB encoder, 128-unit GRU, single-scale context,
   8-pixel lookahead, 0.5–6-pixel step range, and invalid-seed threshold 0.6.
   The previous Sleepy-only tracer,
   `MODELS/tracer/noodle_tracer_seed20260807_endpoint_mse_calibrated.pt`
   (invalid-seed threshold 0.4), remains available.

Full SHA-256 values and artifact roles are maintained in
[`config/artifacts.json`](config/artifacts.json).

Hydra uses grouped splits created with seed 20260718: **4,062 training**, **478
validation**, and **502 test** images. Related captures remain in the same
group. The released Noodle Tracer trains on all 4,062 training images. The
previous tracer used only the Sleepy rows within those splits: **403
training**, **34 validation**, and **63 test** images.

## Repository layout

| Path | Contents |
| --- | --- |
| `CORE/deepstarnoodles/` | Models, data, decoding, tracking, representations |
| `SCRIPTS/` | Training, evaluation, validation, and inference commands |
| `config/` | Artifact hashes, evaluation locks, and tracking settings |
| `docs/` | Detailed API and representation documentation |
| `tests/` | Automated tests |
| `external/dinov3/` | Vendored DINOv3 source under Meta's license |
| `DATA/` | Downloaded datasets; excluded from git |
| `MODELS/` | Downloaded and trained checkpoints; excluded from git |
| `RUNS/` | Generated training, evaluation, and inference outputs |

## Training and evaluation

These workflows create new artifacts under `RUNS/`. They do not recreate the
released checkpoint hashes unless the complete original training environment
and state are reproduced.

### StarSeg

The locked runtime detector is `MODELS/starseg/starseg_best.pt`. To train the
three-class YOLO26-L detector, obtain `yolo26l-seg.pt` from Ultralytics and
place it in `MODELS/starseg/`. The training classes are `sunflower_star`,
`other_sea_star`, and `not_sea_star`. With the virtual environment activated:

```bash
SCRIPTS/train_starseg.sh
```

The script trains on `DATA/starseg/data.yaml` and uses the first visible CUDA
device.

### Hydra

Hydra defaults to the grouped training and validation indexes, 512-pixel
inputs, and an eight-channel arm-identity embedding. Start a new 18-channel
training run with:

```bash
python -m deepstarnoodles.train \
  --out-dir RUNS/hydra_18ch \
  --embedding-dim 8
```

To run the endpoint-head fine-tune from the released parent checkpoint:

```bash
python -m deepstarnoodles.train \
  --out-dir RUNS/hydra_endpoint_mse \
  --endpoint-only \
  --epochs 8 \
  --init-checkpoint MODELS/hydra/epoch_060.pt
```

Hydra checkpoints are written to a timestamped subdirectory of the selected
output directory.

### Noodle Tracer

#### Released recipe: refreshed training on every source family

One command trains the released tracer's recipe from the downloaded dataset
and the parent Hydra checkpoint:

```bash
python SCRIPTS/train_noodle_tracer_refreshed.py \
  --output RUNS/noodle_tracer_fullcorpus
```

The command:

- prepares a study of all 4,062 training images (`--cohort all`) with the
  frozen parent Hydra, `MODELS/hydra/epoch_060.pt`, and caches the fixed
  34-image Sleepy validation split that is scored after every cycle;
- runs 30 cycles. Each cycle regenerates the training data: every image gets
  fresh flips, quarter turns, and brightness, contrast, and saturation changes,
  applied to the pixels and the annotated arm polylines alike, and Hydra is
  rerun on the augmented image. The tracer then makes exactly 4,000 updates at
  batch size 1, resuming from the previous cycle;
- normalizes the dense channels with the fixed statistics in
  [`config/recipes/noodle_tracer_fullcorpus_normalization.json`](config/recipes/noodle_tracer_fullcorpus_normalization.json);
- warms the learning rate up over two cycles, then decays it from 3e-4 to 3e-5
  on a cosine schedule, and ramps rollout training to 30% by cycle 12. The RGB
  encoder is torchvision's ImageNet-pretrained MobileNetV3-Small, kept frozen;
  the first run downloads its weights if they are not cached.

Each cycle caches about 27 GB, deleted once the cycle has trained unless
`--keep-generations` is set. The next cycle's data is generated while the
current cycle trains. On one RTX A6000 a cycle takes roughly 35–45 minutes, so
the 30 cycles take about a day. Re-running the same command resumes after the
last completed cycle; the final checkpoint is
`RUNS/noodle_tracer_fullcorpus/training/g0030/last.pt`.

Then select the invalid-seed threshold on the non-Sleepy validation split,
cached with the endpoint-refined Hydra that inference uses, and write the
calibrated checkpoint:

```bash
python SCRIPTS/cache_fixed_tracer_split.py \
  --cohort non_sleepy \
  --checkpoint MODELS/hydra/endpoint_mse_best.pt \
  --output RUNS/noodle_tracer_fullcorpus/calibration/non_sleepy

python SCRIPTS/calibrate_noodle_tracer.py \
  --study-root RUNS/noodle_tracer_fullcorpus \
  --validation-root RUNS/noodle_tracer_fullcorpus/calibration/non_sleepy \
  --checkpoint RUNS/noodle_tracer_fullcorpus/training/g0030/last.pt \
  --selection-metric f1 \
  --invalid-seed-thresholds none,0.3,0.4,0.5,0.6 \
  --base-state-thresholds none \
  --lost-state-thresholds none \
  --write-calibrated-checkpoint RUNS/noodle_tracer_fullcorpus/calibrated.pt
```

Score the calibrated checkpoint on that split, here against the released
tracer:

```bash
python SCRIPTS/evaluate_tracer_split.py \
  --split-root RUNS/noodle_tracer_fullcorpus/calibration/non_sleepy \
  --checkpoint RUNS/noodle_tracer_fullcorpus/calibrated.pt \
  --checkpoint MODELS/tracer/noodle_tracer_fullcorpus_seed20261011_calibrated.pt
```

The evaluator pools precision, recall, and F1 over images, averages F1 over
`split_group_id` groups, and gives paired group-bootstrap F1 differences
against the first checkpoint.

The released threshold, 0.6, was selected over three training seeds
(20261011, 20261012, and 20261013): repeat `--checkpoint` once per seed to
select by the mean, and the first checkpoint is written. The released
checkpoint is seed 20261011.

A retrained model has similar, not identical, weights: Hydra inference and
training use mixed precision on the GPU and are not bit-deterministic. On the
CPU, this pipeline reproduces the original development code's generations,
weights, and optimizer state exactly.

To reproduce the matched Sleepy-only comparison, run
`--cohort sleepy --draws-per-image 10` with a new `--output`: each cycle then
uses ten augmented draws of each of the 403 Sleepy training images and the same
4,000-update budget. To recompute the normalization statistics from the data,
see [`SCRIPTS/write_tracer_normalization.py`](SCRIPTS/write_tracer_normalization.py).

#### Adapting the tracer to new scenes

The same pipeline trains on your own annotated images. You need:

- one annotation file per image: a JSON list of shapes in which each arm is
  `{"label": "arm", "type": "polyline", "points": [x0, y0, x1, y1, ...]}` in
  source-image pixels, ordered from the central disk to the arm tip. CVAT
  polyline annotations exported to this layout work directly;
- `train_index.csv`, `validation_index.csv`, and `test_index.csv` in one
  directory, with the columns `image_path`, `shapes_path`, `width`, `height`,
  and `split_group_id`. Paths may be absolute or relative to the repository
  root. Put related captures (the same animal, session, or video) in one
  group; no group may span two splits.

Train with `--cohort all --validation-cohort all`, because the `sleepy` and
`non_sleepy` cohorts check the distributed split counts. To start from the
released weights instead of a random initialization, add `--init-checkpoint`:

```bash
python SCRIPTS/train_noodle_tracer_refreshed.py \
  --output RUNS/noodle_tracer_my_scene \
  --split-dir path/to/my_splits \
  --cohort all --validation-cohort all \
  --init-checkpoint MODELS/tracer/noodle_tracer_fullcorpus_seed20261011_calibrated.pt \
  --cycles 10
```

- The learning rate still warms up and then decays over `--cycles`; pass, for
  example, `--trainer-args "--lr 1e-4"` to lower its peak. Fine-tuning budgets
  have not been benchmarked, so choose the number of cycles on your validation
  split.
- For a small dataset, `--draws-per-image 4` gives each image four different
  augmentations per cycle, so the 4,000 updates of a cycle (`--steps-per-cycle`)
  see more distinct views.
- To keep accuracy on the original scenes, append your rows to the distributed
  split indexes instead of replacing them. In our experiments, training on
  every source family generalized better to other scenes than training on the
  Sleepy subset alone.
- The tracer starts from Hydra's tip detections and reads its dense fields. If
  Hydra misses arms in the new scene, fine-tune Hydra first (see
  [Hydra](#hydra)), pass it with `--checkpoint`, and write a matching
  normalization file with
  [`SCRIPTS/write_tracer_normalization.py`](SCRIPTS/write_tracer_normalization.py);
  the shipped statistics describe `epoch_060.pt`.

Calibrate the threshold and score the result on your own validation split with
`cache_fixed_tracer_split.py --cohort all --split-dir path/to/my_splits`,
followed by `calibrate_noodle_tracer.py` and `evaluate_tracer_split.py` as
above.

#### Previous recipe: fixed Sleepy cache

The commands below reproduce the previous Sleepy-only tracer: one fixed cache
of the Sleepy training and validation rows, with augmentation applied to the
cached outputs during training.

Prepare the Sleepy training and validation rows and cache the frozen parent
Hydra outputs:

```bash
python SCRIPTS/cache_frozen_outputs.py \
  --stage all \
  --checkpoint MODELS/hydra/epoch_060.pt \
  --cache-splits train,validation \
  --output RUNS/noodle_tracer_sleepy18
```

Train the single-context tracer:

```bash
python SCRIPTS/train_noodle_tracer.py \
  --study-root RUNS/noodle_tracer_sleepy18 \
  --context-mode single \
  --out-dir RUNS/noodle_tracer_sleepy18/training/primary
```

Select seed and termination thresholds on validation and write the calibrated
checkpoint:

```bash
python SCRIPTS/calibrate_noodle_tracer.py \
  --study-root RUNS/noodle_tracer_sleepy18 \
  --checkpoint RUNS/noodle_tracer_sleepy18/training/primary/best.pt \
  --invalid-seed-thresholds none,0.2,0.3,0.4,0.5,0.6 \
  --base-state-thresholds none \
  --lost-state-thresholds none \
  --write-calibrated-checkpoint \
    RUNS/noodle_tracer_sleepy18/training/primary/calibrated.pt
```

Evaluate the resulting checkpoint on validation:

```bash
python SCRIPTS/evaluate_noodle_tracer.py \
  --study-root RUNS/noodle_tracer_sleepy18 \
  --checkpoint RUNS/noodle_tracer_sleepy18/training/primary/calibrated.pt \
  --split validation \
  --out-json RUNS/noodle_tracer_sleepy18/evaluation/validation.json
```

#### Held-out test access

Test images are cached only after the analysis is locked.
`cache_fixed_tracer_split.py --split test` requires `--analysis-lock`, a JSON
file that sets `test_access_locked` and records the SHA-256 of the test index
(`test_source_index_sha256`), the Hydra checkpoint
(`frozen_checkpoint_sha256`), and the normalization file
(`normalization_sha256`). `evaluate_tracer_split.py --split test` takes the
same lock and scores only checkpoints whose hashes the lock records under keys
ending in `checkpoint_sha256`. For the previous recipe,
`evaluate_noodle_tracer.py` writes a validation lock with `--lock` and
`--lock-manifest`, and a test run requires `--split test`, that manifest, and
`--test-lock-token` set to its `lock_id`. The archived records,
[`config/locked/heldout_test.json`](config/locked/heldout_test.json) for the
previous tracer and `config/locked/fullcorpus_tracer_*.json` for the released
one, cannot be reused as locks for newly trained models.

### Dense geometric decoding

The following evaluates Hydra's standard and embedding-aware geometric
decoders on all 502 images in the grouped test index:

```bash
python -m deepstarnoodles.evaluate_decode \
  --mode model \
  --checkpoint MODELS/hydra/endpoint_mse_best.pt \
  --index DATA/deepstarnoodles/splits/test_index.csv \
  --split all \
  --out-json RUNS/evaluate_decode/grouped_test.json
```

This measures Hydra's geometric decoders, not the Noodle Tracer. The tracer
results reported above come from their own locked evaluation.

## Advanced video inference

Single-video inference uses a dense/tracer batch size of one for minimum live
latency. The default arm-association parameters are a shape gate of 120,
embedding weight of 2.0, and maximum age of 16 frames; each is configurable
through `infer_video.py`.

### Multiple input streams

`infer_videos.py` processes independent videos round-robin while batching
their Hydra and Noodle Tracer work:

```bash
python SCRIPTS/infer_videos.py \
  --video path/to/a.mp4 \
  --video path/to/b.mp4 \
  --video path/to/c.mp4 \
  --video path/to/d.mp4 \
  --batch-size 4 \
  --output-dir RUNS/infer/multi
```

Each input retains independent cropper and tracker state. Batch size controls
the maximum number of crops in a GPU micro-batch and does not change model
weights or tracing thresholds.

### Multiple stars

Use `--multi-star` to detect and trace every sunflower star in a frame:

```bash
python SCRIPTS/infer_video.py \
  --video path/to/multiple_stars.mp4 \
  --output RUNS/infer/multiple_stars_tracks.json \
  --overlay RUNS/infer/multiple_stars_overlay.mp4 \
  --multi-star \
  --overlay-support \
  --batch-size 4
```

BoT-SORT assigns each animal a video-local `star_track_id` using the pinned
[`config/star_tracking/botsort.yaml`](config/star_tracking/botsort.yaml)
configuration. Each animal has a separate arm associator, so an arm
`track_id` is scoped to its enclosing `star_track_id`. Multi-star output uses
schema version 2 and stores detections under each frame's `stars` collection.
The optional support overlay defaults to a 0.3 threshold and 0.28 opacity.

## Behavior representations

`deepstarnoodles.representations` converts native dense, decoded-arm, and
tracked-arm outputs into configurable analysis features:

- `arm_bag(...)` produces a default 102-dimensional,
  permutation-invariant arm-shape summary.
- `arm_motion(...)` produces causal tracked-arm motion features at default
  lags of 1, 3, and 5 seconds.
- `ProgressSequenceEncoder` produces a default 64-dimensional temporal latent
  representation from a 60-second progress-field window.

Representation results include validity masks and deterministic recipe
identifiers. See
[`docs/representations.md`](docs/representations.md) for API examples,
calibration requirements, training, and sidecar formats.

The optional temporal VAE was trained on a different Hydra checkpoint with
the same channel contract. Its use with the released Hydra is available for
experimentation but has not been scientifically revalidated.

## Tests

```bash
python -m pytest -q
```

Tests that require downloaded datasets or checkpoints are skipped when those
artifacts are unavailable.

## License, data use, and citation

First-party source code and first-party released weights are available under
the [MIT License](LICENSE). Third-party components retain their own terms; see
[`NOTICE`](NOTICE) for the complete dependency and redistribution summary.

- The vendored DINOv3 source and separately downloaded backbone remain under
  Meta's DINOv3 License. Publications using DINOv3 must acknowledge it:
  Siméoni et al., “[DINOv3](https://arxiv.org/abs/2508.10104),” 2025.
- StarSeg training and inference use Ultralytics software under AGPL-3.0.
  Distribution of a combined work may therefore create AGPL obligations.
- The gated imagery datasets are licensed CC-BY-4.0. Access is recorded
  because *Pycnopodia helianthoides* is a listed species.

If you use DeepStarNoodles, cite the software using
[`CITATION.cff`](CITATION.cff) and cite DINOv3 when using its backbone.
Contact information is provided in `CITATION.cff`.
