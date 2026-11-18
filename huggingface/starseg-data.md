---
license: cc-by-4.0
task_categories:
  - image-segmentation
  - object-detection
pretty_name: StarSeg detection dataset
tags:
  - marine-biology
  - pycnopodia
  - sunflower-star
  - instance-segmentation
  - yolo
extra_gated_prompt: >
  Laboratory, field and web images with instance masks of sunflower sea stars,
  other sea stars and prey, used to train and score the StarSeg detectors of
  DeepStarNoodles. Pycnopodia helianthoides is a listed species. Access is
  gated so we know who holds the imagery. The licence is CC BY 4.0; cite
  DeepStarNoodles if you use this set. Detector training and inference depend
  on Ultralytics (AGPL-3.0).
extra_gated_fields:
  Affiliation: text
  I will cite DeepStarNoodles if I use this dataset: checkbox
---

# StarSeg detection dataset

Images of sunflower sea stars (*Pycnopodia helianthoides*), other sea stars and their prey, with one instance mask per animal or prey item. This is the frozen version `starseg_yolo26_20261106` (frozen 2026-11-07, split `starseg-20261106-64f5f09f0ced`), on which the three StarSeg detectors of [DeepStarNoodles](https://github.com/weertman/DeepStarNoodles) were trained and scored. The trained models are at [weertman/deepstarnoodles](https://huggingface.co/weertman/deepstarnoodles).

| | Train | Validation | Test | All |
| --- | --- | --- | --- | --- |
| Images | 5,600 | 1,129 | 1,249 | 7,978 |
| Laboratory | 1,951 | 341 | 352 | 2,644 |
| Field | 1,221 | 248 | 395 | 1,864 |
| Web | 2,428 | 540 | 502 | 3,470 |
| Instance masks | 18,345 | 3,966 | 4,083 | 26,394 |
| Sunflower sea star masks | 5,806 | 1,213 | 1,413 | 8,432 |
| Images without labels | 84 | 0 | 8 | 92 |

Field images include 878 frames from the dive surveys of one cruise in Prince William Sound, Alaska (PWS; 243 of them in the test split). The paper reports the field images with and without PWS separately. The 92 images without labels were checked by eye and show no sunflower sea star.

Splits keep groups of related images together (3,494 groups, formed from capture sessions, recording days, source videos and duplicate pairs). `groups.json` and `manifest.csv` record each image's group and the rule that formed it.

## Label schemes

The same masks are written in four YOLO datasets that differ in how objects other than sunflower sea stars are labelled:

| Folder | Classes | Detector trained on it |
| --- | --- | --- |
| `arms/B` | 2: `sunflower_star`, `not_sunflower` | `starseg_binary.pt`, the detector of the paper |
| `arms/F` | 37: `Pycnopodia_helianthoides`, 26 other sea star species, urchins, bivalves, sand dollars, dead prey, unidentified | `starseg_fine.pt` |
| `arms/C` | 4: `sunflower_star`, `other_sea_star`, `live_prey`, `dead_prey` | `starseg_coarse.pt` |
| `arms/P` | 1: `sunflower_star` (other objects unlabelled) | none published |

## Layout

```
starseg_yolo26_20261106/
  arms/{B,F,C,P}/data.yaml, images/{train,val,test}/, labels/{train,val,test}/
  images/all/                   the 7,978 prepared JPEGs (longest side at most 1024 px)
  manifest.csv, splits/*.csv    one row per image: source, group, split, label counts
  canonical_annotations.jsonl   the CVAT shapes in source pixels; use these for scoring
  groups.json, duplicate_image_edges.csv
  summary.json, label_maps.json, masks_conversion.json, image_preparation.json
  hashes.json, payload_hashes.json, freeze.json, validation.json, provenance.json
  AMENDMENT.md, amendment_report.json   the labelling fixes of November 2026
  code/                         the builder and its settings
```

Download it into the `DATA` folder of a checkout of the code, where the detector training script expects it:

```bash
hf download weertman/starseg --repo-type dataset --local-dir DATA
SCHEME=B SEED=0 SCRIPTS/train_starseg.sh
```

The images in `arms/*/images/` are the same files as `images/all/`.

## Notes

- Masks are polygons; 68 masks drawn as rasters are kept as run-length encodings in `canonical_annotations.jsonl` and approximated by polygons for training.
- Most images (7,881) were used in earlier StarSeg training runs before this version was frozen, so the test split is held out only for the runs of the paper.
- The origin of the web photographs is not recorded.
- Animals recorded at Friday Harbor Laboratories come from its conservation breeding programme.

## Earlier contents

Until 2026-11-17 this repository held an earlier three-class version (`sunflower_star`, `other_sea_star`, `not_sea_star`) in the folders `images/` and `labels/`. It was removed from the current revision and remains in the history of this repository (last revision with it: `c61eabd0f8af8ae051019e27df7d8e102d477c39`).
