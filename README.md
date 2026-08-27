# DeepStarNoodles

Dense pose estimation, **Noodle Tracer** arm polylines, and temporal tracking
for sunflower stars (*Pycnopodia helianthoides*).

```
full frame
    -> StarSeg YOLO crop (2x bbox square)
    -> DINOv3 ConvNeXt-Base + FPN + heads  (18-channel dense field)
    -> Noodle Tracer                       (per-arm polylines)
    -> fused tracker                       (temporal IDs)
```

Held-out 63 Sleepy test images: **arm F1 0.837**, group-macro F1 **0.824**,
mean/p90 Chamfer 3.389 / 5.761, 99.84% base completion. Embedding-Dijkstra on
the same Hydra outputs is a geometric baseline (arm F1 **0.777**), not a
second released recipe. See `config/locked/heldout_test.json`.

Author: Weertman (`wlweert@gmail.com`). First-party code is MIT; see
`LICENSE` and `NOTICE`. DINOv3 in `external/dinov3/` remains under Meta's
DINOv3 License and must be cited in publications that use it (Siméoni et al.,
2025, https://arxiv.org/abs/2508.10104). StarSeg depends on Ultralytics
(AGPL-3.0).

## Locked recipe

1. 18-channel Hydra (`legacy10` geometry + 8-D embedding) from
   `MODELS/hydra/epoch_060.pt`.
2. Endpoint-head-only MSE fine-tune
   (`MODELS/hydra/endpoint_mse_best.pt`, SHA-256 `dcf7140c…`).
3. Noodle Tracer, 33×33, frozen MobileNetV3-Small RGB, 128-unit GRU,
   `context_mode=single`, 8 px lookahead, 0.5–6 px step,
   `invalid_seed_threshold=0.4`
   (`MODELS/tracer/noodle_tracer_seed20260807_endpoint_mse_calibrated.pt`,
   SHA-256 `bb7f1b5d…`).

Video deploy uses this pair. StarSeg runtime weight is
`MODELS/starseg/starseg_best.pt` (YOLOv8, class `Pycnopodia_helianthoides`).

## Splits

**Hydra dense pose** uses grouped image splits (seed 20260718): **4062 / 478 /
502** train / validation / test. Related captures share `split_group_id` and
do not cross boundaries.

**Noodle Tracer** is trained and locked on the Sleepy subset of those grouped
splits: **403 / 34 / 63**.

## Layout

| Path | Contents |
| --- | --- |
| `CORE/deepstarnoodles/` | Model, dataset, decode, Noodle Tracer, cropper, fused tracker |
| `SCRIPTS/` | Train / cache / calibrate / evaluate / infer CLIs |
| `DATA/deepstarnoodles/` | Dense images, shapes, grouped splits (seed 20260718) |
| `DATA/starseg/` | 3-class YOLO train/val corpus |
| `MODELS/` | DINOv3, StarSeg, Hydra, tracer checkpoints |
| `RUNS/` | Training outputs |
| `external/dinov3/` | Vendored DINOv3 hub source |
| `config/artifacts.json` | SHA-256 pins |
| `config/locked/` | Immutable tracer validation lock and held-out eval |

## Environment

Python 3.11. Install a CUDA PyTorch build first, then:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
export PYTHONPATH=CORE
```

Large files are not in git. They live on gated Hugging Face repos
(CC-BY-4.0 for imagery). Log in with `hf auth login` and accept the access
form, then:

```bash
hf download weertman/deepstarnoodles --local-dir MODELS
hf download weertman/deepstarnoodles-data --repo-type dataset --local-dir DATA/deepstarnoodles
hf download weertman/starseg --repo-type dataset --local-dir DATA/starseg
```

DINOv3 ConvNeXt-Base is not included. Put Meta's
`dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` at
`MODELS/dinov3/` (SHA-256 `801f2ba95801bdcc9a8b77e293d38b35b355dbc813ef609cd82e18c3a185e8da`).

Confirm checksums (skips Ultralytics `yolo26l-seg.pt` unless it is present):

```bash
.venv/bin/python SCRIPTS/validate_artifacts.py
```

Override the checkout root with `DEEPSTARNOODLES_ROOT` only when needed.

## Train

**StarSeg (optional retrain).** Locked inference uses `starseg_best.pt`. The
3-class YOLO26-L recipe is the from-scratch detector path. Put Ultralytics
`yolo26l-seg.pt` at `MODELS/starseg/` first (not on Hugging Face), then:

```bash
SCRIPTS/train_starseg.sh
```

**Hydra 18-channel, then endpoint fine-tune.** Grouped splits, ignore masks,
and `--embedding-dim 8` are the defaults:

```bash
export PYTHONPATH=CORE
.venv/bin/python -m deepstarnoodles.train \
  --out-dir RUNS/hydra_18ch \
  --init-checkpoint MODELS/hydra/epoch_060.pt

.venv/bin/python -m deepstarnoodles.train \
  --out-dir RUNS/hydra_endpoint_mse \
  --endpoint-only \
  --epochs 8 \
  --init-checkpoint MODELS/hydra/epoch_060.pt
```

**Noodle Tracer.** Freeze Hydra outputs on the Sleepy 403/34/63 split, train
with `--context-mode single` (the locked recipe; this is also the CLI default),
then calibrate `invalid_seed_threshold` to 0.4:

```bash
.venv/bin/python SCRIPTS/cache_frozen_outputs.py
.venv/bin/python SCRIPTS/train_noodle_tracer.py \
  --context-mode single \
  --out-dir RUNS/noodle_tracer_sleepy18/training/primary
.venv/bin/python SCRIPTS/calibrate_noodle_tracer.py \
  --invalid-seed-thresholds none,0.2,0.3,0.4,0.5,0.6 \
  --base-state-thresholds none \
  --lost-state-thresholds none
```

## Evaluate

Noodle Tracer first (locked recipe). Dijkstra eval of the Hydra field is the
geometric baseline.

```bash
export PYTHONPATH=CORE
.venv/bin/python SCRIPTS/evaluate_noodle_tracer.py \
  --study-root RUNS/noodle_tracer_sleepy18 \
  --checkpoint MODELS/tracer/noodle_tracer_seed20260807_endpoint_mse_calibrated.pt \
  --split test

.venv/bin/python -m deepstarnoodles.evaluate_decode \
  --mode model \
  --checkpoint MODELS/hydra/endpoint_mse_best.pt
```

`evaluate_decode` defaults to the grouped test index and `--split all`.

## Deploy

```bash
.venv/bin/python SCRIPTS/infer_video.py \
  --video path/to/video.mp4 \
  --output RUNS/infer/tracks.json \
  --overlay RUNS/infer/overlay.mp4 \
  --batch-size 1
```

Crop → 18-channel Hydra → Noodle Tracer → fused tracker (shape_gate 120,
embed_weight 2.0, max_age 16). Temporal lifecycle modes are not enabled.
Batch size 1 is the low-latency default. For four independent 1 fps streams,
use one round-robin process so Hydra and all live RNN walkers share a B=4
GPU batch:

```bash
.venv/bin/python SCRIPTS/infer_videos.py \
  --video path/to/a.mp4 \
  --video path/to/b.mp4 \
  --video path/to/c.mp4 \
  --video path/to/d.mp4 \
  --batch-size 4 \
  --output-dir RUNS/infer/multi
```

This is an inference-only optimization: no checkpoint, architecture, recurrent
step, threshold, or trace semantics changes. Benchmark the complete path on a
target GPU with:

```bash
.venv/bin/python SCRIPTS/benchmark_inference.py \
  --device cuda:0 \
  --batch-size 4 \
  --engine batch \
  --include-starseg \
  --output RUNS/inference_benchmark/gpu_b4_full.json
```

## Tests

```bash
export PYTHONPATH="CORE:SCRIPTS"
.venv/bin/python -m pytest -q tests
```

## Non-goals

This extract does not include BAM, spatial VAE, VAME, Sleepy 5M-frame archives,
StarFlipping behavior analysis, SLEAP, pixel-instance KMeans, the June 9/10
10-channel ensemble, 768-px Hydra ablations, or experimental offline tracklet
reconciliation.
