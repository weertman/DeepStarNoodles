---
license: cc-by-4.0
task_categories:
  - object-detection
pretty_name: StarSeg
tags:
  - marine-biology
  - pycnopodia
  - sunflower-star
  - yolo
extra_gated_prompt: >
  Three-class YOLO corpus for StarSeg (sunflower_star, other_sea_star,
  not_sea_star). Access is gated so we know who holds the imagery. License is
  CC-BY-4.0. Detector training and inference depend on Ultralytics (AGPL-3.0).
extra_gated_fields:
  Affiliation: text
  I will cite DeepStarNoodles if I use this dataset: checkbox
---

# StarSeg detector corpus

YOLO images, labels, and `data.yaml` for retraining StarSeg. Locked inference
uses `starseg_best.pt` from [`weertman/deepstarnoodles`](https://huggingface.co/weertman/deepstarnoodles); this corpus is the from-scratch detector path.

Classes: `sunflower_star`, `other_sea_star`, `not_sea_star`.

```bash
hf download weertman/starseg --repo-type dataset --local-dir DATA/starseg
```
