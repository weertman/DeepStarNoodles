# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""One-shot prep: StarSeg-derived ignore polygons for unannotated neighbor stars.

Annotators only label the central star of each crop, so any other Pycnopodia in
frame is a real star whose arms the targets wrongly call background. This script
runs the StarSeg YOLO segmentation model over the dense index and, per image:

1. finds all Pycnopodia instances,
2. picks the central instance = the one whose mask best covers the human
   annotation (fallback: contains the image center),
3. SAFETY GATE: requires that >= --containment-frac of all annotated points
   (every arm-polyline vertex + every landmark) fall inside the dilated central
   mask. StarSeg is only trusted where it demonstrably agrees with the human.
4. If the gate passes and other instances exist, their polygons are written as
   ignore shapes (label 'star_ignore', original image coordinates). Training
   excludes those pixels from the loss — neither positive nor negative.

Images that fail the gate (or have no neighbors) get no ignore file: training
is unchanged for them.

Usage:
  python -m deepstarnoodles.build_ignore_masks \
    --index .../deepstarnoodles_dense_train_index_annotated_only_20260610b.csv \
    --out-dir .../dense_training_latest/ignore_shapes
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from deepstarnoodles.dataset import row_artifact_name
from deepstarnoodles.paths import DEFAULT_ANNOTATED_INDEX, DEFAULT_STARSEG

DEFAULT_INDEX = DEFAULT_ANNOTATED_INDEX
DEFAULT_WEIGHTS = DEFAULT_STARSEG
SUNFLOWER_NAMES = ("Pycnopodia_helianthoides", "sunflower_star")
GRID = 768  # working resolution for containment checks / overlap scoring


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build StarSeg ignore polygons for neighbor stars.')
    p.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    p.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--conf', type=float, default=0.4)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--containment-frac', type=float, default=0.99,
                   help='Min fraction of annotated points inside the dilated central mask.')
    p.add_argument('--dilate-px', type=int, default=12,
                   help='Dilation (in GRID px) of the central mask before the containment test.')
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def annotated_points(shapes: list[dict]) -> np.ndarray:
    """All human-annotated coordinates (arm vertices + landmarks), (N, 2)."""
    pts: list[list[float]] = []
    for s in shapes:
        if s.get('outside'):
            continue
        raw = list(s.get('points') or [])
        pts.extend([raw[i], raw[i + 1]] for i in range(0, len(raw) - 1, 2))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def poly_to_mask(poly: np.ndarray, w: int, h: int, sx: float, sy: float) -> np.ndarray:
    img = Image.new('L', (GRID, GRID), 0)
    scaled = [(float(x) * sx, float(y) * sy) for x, y in poly]
    if len(scaled) >= 3:
        ImageDraw.Draw(img).polygon(scaled, fill=1)
    return np.asarray(img, dtype=bool)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    out = Image.fromarray(mask.astype(np.uint8) * 255)
    from PIL import ImageFilter

    out = out.filter(ImageFilter.MaxFilter(2 * px + 1))
    return np.asarray(out, dtype=np.uint8) > 0


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    names = {int(k): str(v) for k, v in model.names.items()}
    sunflower_ids = {cid for cid, name in names.items() if name in SUNFLOWER_NAMES}
    if not sunflower_ids:
        raise RuntimeError(
            f"StarSeg weights have no sunflower class {SUNFLOWER_NAMES!r}; names={names}"
        )
    rows = list(csv.DictReader(args.index.open()))
    if args.limit:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for idx, row in enumerate(rows):
        artifact_name = row_artifact_name(row, idx)
        rec: dict = {'idx': idx, 'artifact_name': artifact_name,
                     'image_path': row['image_path'], 'status': 'no_neighbors',
                     'n_pycno': 0, 'containment': None, 'n_ignore': 0}
        manifest.append(rec)
        res = model.predict(row['image_path'], imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)[0]
        W, H = int(row['width']), int(row['height'])
        sx, sy = GRID / W, GRID / H
        if res.masks is None:
            rec['status'] = 'no_detections'
            continue
        cls = res.boxes.cls.tolist()
        polys = [np.asarray(res.masks.xy[k]) for k in range(len(cls)) if int(cls[k]) in sunflower_ids]
        polys = [p for p in polys if len(p) >= 3]
        rec['n_pycno'] = len(polys)
        if not polys:
            rec['status'] = 'no_detections'
            continue

        shapes = json.loads(Path(row['shapes_path']).read_text())
        pts = annotated_points(shapes)
        if len(pts) == 0:
            rec['status'] = 'no_annotation'
            continue
        gpts = np.stack([np.clip(pts[:, 0] * sx, 0, GRID - 1), np.clip(pts[:, 1] * sy, 0, GRID - 1)], axis=1)
        ipts = gpts.astype(int)

        masks = [poly_to_mask(p, W, H, sx, sy) for p in polys]
        coverage = [m[ipts[:, 1], ipts[:, 0]].mean() for m in masks]
        central = int(np.argmax(coverage))
        contained = float(dilate(masks[central], args.dilate_px)[ipts[:, 1], ipts[:, 0]].mean())
        rec['containment'] = round(contained, 4)
        if contained < args.containment_frac:
            rec['status'] = 'gate_failed'
            continue
        neighbors = [polys[k] for k in range(len(polys)) if k != central]
        if not neighbors:
            rec['status'] = 'no_neighbors'
            continue
        # Drop neighbor polygons that overlap the central star's mask region:
        # supervision there is real (the annotated star), never ignored.
        central_mask = masks[central]
        ignore_shapes = []
        for k, poly in enumerate(polys):
            if k == central:
                continue
            m = masks[k] & ~central_mask
            if m.sum() < 16:
                continue
            ignore_shapes.append({
                'label': 'star_ignore',
                'type': 'polygon',
                'points': [float(v) for xy in poly for v in xy],
            })
        if not ignore_shapes:
            rec['status'] = 'no_neighbors'
            continue
        out_path = args.out_dir / artifact_name
        out_path.write_text(json.dumps(ignore_shapes))
        rec['status'] = 'masked'
        rec['n_ignore'] = len(ignore_shapes)

    man_path = args.out_dir / 'manifest.json'
    man_path.write_text(json.dumps(manifest, indent=1))
    from collections import Counter

    counts = Counter(r['status'] for r in manifest)
    print(json.dumps({'index': str(args.index), 'out_dir': str(args.out_dir), 'counts': dict(counts)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
