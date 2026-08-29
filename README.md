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

The DINOv3 ConvNeXt-Base backbone is distributed separately under Meta's
DINOv3 License. Follow the
[official DINOv3 access instructions](https://github.com/facebookresearch/dinov3)
and place this file in `MODELS/dinov3/`:

```text
dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
SHA-256 801f2ba95801bdcc9a8b77e293d38b35b355dbc813ef609cd82e18c3a185e8da
```

Validate the runtime artifacts against the
[artifact manifest](config/artifacts.json):

```bash
python SCRIPTS/validate_artifacts.py \
  --roles pretrained-backbone runtime-detector released-dense released-tracer
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

The release result is recorded in
[`config/locked/heldout_test.json`](config/locked/heldout_test.json). It covers
the complete 63-image test split of the **Sleepy** data-collection subset,
identified as `source_family=sleepy_stars` in the dataset metadata. Those
images belong to five leakage-separated `split_group_id` groups.

Arms are greedily matched one-to-one when their symmetric Chamfer distance is
at most 8 model-crop pixels. All reported distances below are in model-crop
pixels.

- Noodle Tracer aggregate precision, recall, and F1 are **0.859**, **0.815**,
  and **0.837**. Mean and p90 Chamfer distances are **3.389** and **5.761**.
- Equal-group macro precision, recall, and F1 are **0.850**, **0.802**, and
  **0.824**. Group-macro mean and p90 Chamfer distances are **3.487** and
  **5.645**.
- Of 617 emitted traces, 616 terminated at the model's base condition
  (**99.84%**). This is a decoder termination statistic, not an independent
  biological accuracy measure.
- The embedding-aware Dijkstra geometric baseline has aggregate F1 **0.777**
  and group-macro F1 **0.757** on the same cached Hydra outputs.
- A paired bootstrap over the five split groups estimates an aggregate F1
  improvement of **0.062** over that baseline (95% CI **0.043–0.097**; 2,000
  resamples).

Arm crossings remain difficult. Fifty of the 63 test images are marked as
containing a crossing; aggregate F1 on that subset is **0.795**, with mean
Chamfer distance **3.663**.

The JSON report is the canonical record for this release and contains the
artifact, cache, evaluator, and decoder hashes used to produce it. It is
archival provenance rather than a claim that the current checkout can
regenerate the report bit for bit.

## Limitations

- The released arm-tracing evaluation is limited to the Sleepy subset and may
  not represent other imaging conditions, life stages, or species.
- Performance is lower when arms cross or overlap. The crossing subset result
  above should be considered when selecting downstream analyses.
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
3. `MODELS/tracer/noodle_tracer_seed20260807_endpoint_mse_calibrated.pt`, a
   Noodle Tracer with a 33×33 dense patch, frozen MobileNetV3-Small RGB
   encoder, 128-unit GRU, single-scale context, 8-pixel lookahead, 0.5–6-pixel
   step range, and invalid-seed threshold 0.4.

Full SHA-256 values and artifact roles are maintained in
[`config/artifacts.json`](config/artifacts.json).

Hydra uses grouped splits created with seed 20260718: **4,062 training**, **478
validation**, and **502 test** images. Related captures remain in the same
group. Noodle Tracer uses the Sleepy rows within those splits: **403
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

Held-out test access requires a lock generated from an exactly matching
validation configuration. The archived release result is provided in
[`config/locked/heldout_test.json`](config/locked/heldout_test.json); its lock
token is not a reusable token for newly trained models.

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

This 502-image evaluation has a different scope from the archived
63-image Sleepy comparison reported above.

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
