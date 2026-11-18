---
license: cc-by-4.0
task_categories:
  - image-segmentation
pretty_name: DeepStarNoodles arm dataset
tags:
  - marine-biology
  - pycnopodia
  - sunflower-star
  - pose-estimation
extra_gated_prompt: >
  Crops of sunflower sea stars with one polyline per arm and landmark points,
  used to train the Hydra field network and the Noodle Tracer. Pycnopodia
  helianthoides is a listed species. Access is gated so we know who holds the
  imagery. The licence is CC BY 4.0; cite DeepStarNoodles if you use this set.
extra_gated_fields:
  Affiliation: text
  I will cite DeepStarNoodles if I use this dataset: checkbox
---

# DeepStarNoodles arm dataset

Crops of single sunflower sea stars (*Pycnopodia helianthoides*) with one polyline per arm, drawn from base to tip, and points on the mouth, madreporite and anus. This is the frozen version `noodles_final_20261026`, on which the Hydra field network and the Noodle Tracer of [DeepStarNoodles](https://github.com/weertman/DeepStarNoodles) were trained and scored. The trained models are at [weertman/deepstarnoodles](https://huggingface.co/weertman/deepstarnoodles).

| | Train | Validation | Test | All |
| --- | --- | --- | --- | --- |
| Crops | 4,062 | 478 | 553 | 5,093 |

The crops carry 57,055 arm polylines and 6,346 landmark points (4,693 anus, 1,254 madreporite, 399 mouth). The anus has the label `butthole` in the files.

## Sources

| `source_family` | What it is | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| `starseg` | Crops of sunflower sea stars from images of the [detection dataset](https://huggingface.co/datasets/weertman/starseg): web photographs, field and laboratory images | 2,366 | 255 | 248 |
| `starseg_lab` | Crops from laboratory images | 732 | 138 | 138 |
| `arm_pit` | Crops from an earlier annotation project | 394 | 51 | 44 |
| `axis_detector` | Crops from an earlier annotation project | 27 | 0 | 0 |
| `sleepy_stars` | Sleepy Stars rig, one animal per camera | 403 | 34 | 63 |
| `stardome` | StarDome rig, monochrome | 70 | 0 | 30 |
| `startrack_usb2` | StarTrack rig, colour and near-infrared cameras | 70 | 0 | 30 |

Splits keep acquisition groups together (`split_group_id`). The 493 test crops of the first five sources were scored repeatedly while the method was built. The 60 test crops of the StarDome and StarTrack rigs were not. For those two rigs the column `split_limitation` records what the split does not guarantee: the same animals may appear in training and test crops.

## Layout

```
noodles_final_20261026/
  splits/train_index.csv, validation_index.csv, test_index.csv, split_contract.json
  shapes/<uid>.json            annotations of one crop
  ignore_masks/<uid>.json      polygons over neighbouring stars that are not annotated (329 crops)
  dataset_manifest.json        version, counts and hashes
  annotation_adjudications.json  the six crops whose landmark points were reviewed
noodles_cvat_20261026/extracted/images/default/   the 5,093 images (4,393 JPEG, 700 PNG)
```

The index files store paths relative to a folder `DATA` in the checkout of the code, for example `DATA/noodles_cvat_20261026/extracted/images/default/<name>.jpg`. Download into that folder and the paths resolve:

```bash
hf download weertman/deepstarnoodles-data --repo-type dataset --local-dir DATA
```

Each index row has the image and annotation paths, the image size, the counts of arms and landmarks, three hashes (file bytes, pixels, annotation), the split, the acquisition group and the rule that formed it. A shapes file is a list of objects with a `label` (`arm`, `mouth`, `madreporite`, `butthole`), a `type` (`polyline` or `points`) and `points` as a flat list of x, y in image pixels. Arm polylines run from base to tip.

## Notes

- Before the freeze, 149 crops were removed because an image in another split had the same perceptual hash. About 47 test crops still have a near-identical training crop in another group.
- Animals recorded at Friday Harbor Laboratories come from its conservation breeding programme.
- The CVAT export archive is not included. The index column `source_shapes_path` records the prepared CVAT file that each shapes file was copied from; those files are not included and no code reads them.

## Earlier contents

Until 2026-11-17 this repository held an earlier version (5,792 images, 5,042 of them annotated, split 4,062 / 478 / 502, in the folders `images/`, `shapes/`, `ignore_masks/` and `splits/`). The frozen version drops the 750 images without annotation and the 149 removed crops, and adds 200 crops from the StarDome and StarTrack rigs. The earlier files remain in the history of this repository (last revision with them: `9590c93f0eac2ad39924ad81902ccafa626e77ec`).
