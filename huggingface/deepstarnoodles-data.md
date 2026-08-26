---
license: cc-by-4.0
task_categories:
  - image-segmentation
pretty_name: DeepStarNoodles dense pose
tags:
  - marine-biology
  - pycnopodia
  - sunflower-star
  - pose-estimation
extra_gated_prompt: >
  Dense sunflower-star images and polyline annotations for Hydra and Noodle
  Tracer training. Pycnopodia helianthoides is a listed species. Access is
  gated so we know who holds the imagery. License is CC-BY-4.0; cite
  DeepStarNoodles if you use this set.
extra_gated_fields:
  Affiliation: text
  I will cite DeepStarNoodles if I use this dataset: checkbox
---

# DeepStarNoodles dense pose data

Grouped splits (seed 20260718): **4062 / 478 / 502** train / validation / test.
5042 annotated dense images, 5792 images total.

Layout matches `DATA/deepstarnoodles/` in [the GitHub extract](https://github.com/weertman/DeepStarNoodles):

- `images/`
- `shapes/`
- `ignore_masks/`
- `splits/`
- `deepstarnoodles_dense_train_index.csv`
- `deepstarnoodles_dense_metadata.csv`
- `deepstarnoodles_all_image_metadata.csv`
- `metadata_summary.json`

The CVAT export archive is not included.

Index CSVs store repo-relative paths such as `DATA/deepstarnoodles/images/...`.
The GitHub loaders resolve them from the checkout root.

```bash
hf download weertman/deepstarnoodles-data --repo-type dataset --local-dir DATA/deepstarnoodles
```
