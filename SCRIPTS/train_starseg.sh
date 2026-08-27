#!/usr/bin/env bash
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
# Retrain the 3-class StarSeg detector. The locked inference weight remains
# MODELS/starseg/starseg_best.pt (YOLOv8). This recipe is the current
# from-scratch path on DATA/starseg.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_YAML="$REPO_ROOT/DATA/starseg/data.yaml"
MODEL="$REPO_ROOT/MODELS/starseg/yolo26l-seg.pt"
PROJECT="$REPO_ROOT/RUNS/starseg_three_class"
NAME="${NAME:-starseg_three_class_yolo26l_seg_1024}"

if ! command -v yolo >/dev/null 2>&1; then
  echo "Install ultralytics so the 'yolo' CLI is on PATH." >&2
  exit 1
fi
if [[ ! -f "$DATA_YAML" ]]; then
  echo "Missing YOLO data yaml: $DATA_YAML" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Missing YOLO26-L-seg init weights: $MODEL" >&2
  echo "Download yolo26l-seg.pt from Ultralytics into MODELS/starseg/. It is not on Hugging Face." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec yolo segment train \
  model="$MODEL" \
  data="$DATA_YAML" \
  device=0 \
  imgsz=1024 \
  epochs=150 \
  batch=4 \
  workers=8 \
  project="$PROJECT" \
  name="$NAME" \
  pretrained=True \
  amp=True \
  cache=False \
  patience=25 \
  close_mosaic=10 \
  cos_lr=True \
  plots=True
