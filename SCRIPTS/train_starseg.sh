#!/usr/bin/env bash
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
# Train a StarSeg detector with the recipe of the published detectors:
# YOLO26-L segmentation from COCO-pretrained weights, AdamW, 50 epochs at
# 1024 px, nominal batch 64, terminal checkpoint (train/weights/last.pt).
#
#   SCHEME=B SEED=0 SCRIPTS/train_starseg.sh    # starseg_binary.pt
#   SCHEME=F SEED=0 SCRIPTS/train_starseg.sh    # starseg_fine.pt
#   SCHEME=C SEED=0 SCRIPTS/train_starseg.sh    # starseg_coarse.pt
#
# Retraining reproduces the recipe, not the published weights.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEME="${SCHEME:-B}"
SEED="${SEED:-0}"
BATCH="${BATCH:-4}"
DATA_DIR="$REPO_ROOT/DATA/starseg_yolo26_20261106/arms/$SCHEME"
MODEL="$REPO_ROOT/MODELS/starseg/yolo26l-seg.pt"
PROJECT="$REPO_ROOT/RUNS/starseg_train"
NAME="${NAME:-yolo26l_${SCHEME}_seed${SEED}}"

if ! command -v yolo >/dev/null 2>&1; then
  echo "Install ultralytics so the 'yolo' CLI is on PATH." >&2
  exit 1
fi
if [[ ! -f "$DATA_DIR/data.yaml" ]]; then
  echo "Missing detection dataset: $DATA_DIR/data.yaml" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Missing YOLO26-L-seg init weights: $MODEL" >&2
  echo "Download yolo26l-seg.pt from Ultralytics into MODELS/starseg/. It is not on Hugging Face." >&2
  exit 1
fi
if (( BATCH <= 0 || 64 % BATCH )); then
  echo "BATCH must be a positive divisor of the nominal batch 64." >&2
  exit 1
fi

# Ultralytics resolves a relative dataset path against its own datasets
# folder, so write a copy of data.yaml with an absolute path.
mkdir -p "$PROJECT/$NAME"
DATA_YAML="$PROJECT/$NAME/data.yaml"
sed "s#^path: .*#path: $DATA_DIR#" "$DATA_DIR/data.yaml" > "$DATA_YAML"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec yolo segment train \
  model="$MODEL" \
  data="$DATA_YAML" \
  project="$PROJECT" \
  name="$NAME" \
  exist_ok=True \
  device=0 \
  imgsz=1024 \
  epochs=50 \
  batch="$BATCH" \
  nbs=64 \
  workers=4 \
  seed="$SEED" \
  deterministic=True \
  optimizer=AdamW \
  lr0=0.000244 \
  lrf=0.01 \
  momentum=0.9 \
  weight_decay=0.0005 \
  warmup_epochs=3.0 \
  warmup_momentum=0.8 \
  warmup_bias_lr=0.0 \
  cos_lr=True \
  close_mosaic=10 \
  patience=0 \
  amp=True \
  box=7.5 cls=0.5 dfl=1.5 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=0.0 translate=0.1 scale=0.5 shear=0.0 perspective=0.0 \
  flipud=0.0 fliplr=0.5 bgr=0.0 mosaic=1.0 mixup=0.0 cutmix=0.0 copy_paste=0.0 \
  overlap_mask=True mask_ratio=4 rect=False \
  cache=False val=True save=True plots=False end2end=True
