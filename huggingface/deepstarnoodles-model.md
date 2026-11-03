---
license: mit
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - pose-estimation
  - marine-biology
  - pycnopodia
  - sunflower-star
extra_gated_prompt: >
  These are trained DeepStarNoodles weights (StarSeg, Hydra, Noodle Tracer,
  temporal VAE).
  Access is gated so we know who has the locked recipe. DINOv3 backbone
  weights are not included; download those from Meta under the DINOv3 License.
extra_gated_fields:
  Affiliation: text
  I will cite DeepStarNoodles and DINOv3 where I use them: checkbox
---

# DeepStarNoodles weights

Locked inference and retrain weights for [DeepStarNoodles](https://github.com/weertman/DeepStarNoodles).

| File | Role | SHA-256 |
| --- | --- | --- |
| `starseg/starseg_best.pt` | StarSeg detector (YOLOv8) | `0252e49d3b201f5fe557536d37648ff6ff8ac1ca2c303ec7b5c707f1973a9414` |
| `hydra/epoch_060.pt` | 18-channel Hydra parent | `b3423e9bab886777e04825945434b8d42609b3c3fde0cd270d2fe2668c411eec` |
| `hydra/endpoint_mse_best.pt` | Released dense Hydra | `dcf7140c9ef0129d8f66b33b05af6d0ebb064f84b41ce61e51e67061799e3f02` |
| `tracer/noodle_tracer_fullcorpus_seed20261011_calibrated.pt` | Noodle Tracer (released, full corpus) | `f39dc82b9bc5d38e2053c2535ca30055a9d03140a713dbb991cb7058130d5287` |
| `tracer/noodle_tracer_seed20260807_endpoint_mse_calibrated.pt` | Previous Noodle Tracer (Sleepy-only, August 2026) | `bb7f1b5da4f998006163b0bfbba005b417dfd6e8ab313d8e3869b7224fcc5c6c` |
| `temporal_vae/dnt_progress_temporal_vae_60s_sleepy.pt` | Optional 60-second progress temporal VAE | `182de7d66d3b670c94e25d4f78070d7ffa5851acd76200cc7c059849b1abddc9` |

## Released Noodle Tracer (October 2026)

The released tracer was trained on all 4,062 annotated training images, from every source family (not only Sleepy Stars). It runs on `hydra/endpoint_mse_best.pt` with an invalid-seed threshold of 0.6, stored in the checkpoint. It was selected on validation by a rule fixed in advance, then scored once on the held-out test splits under an analysis lock:

| Test split | Arm F1 | Group-macro F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| Sleepy Stars (63 images, 5 groups) | 0.868 | 0.855 | 0.912 | 0.828 |
| Other source families (439 images, 178 groups) | 0.854 | 0.836 | 0.888 | 0.822 |

Against the previous Sleepy-only tracer on the same images, arm F1 rose by +0.030 (95% CI +0.005 to +0.054) on Sleepy Stars and by +0.017 (+0.009 to +0.026) on the other families. The previous tracer stays here for reproducing the August 2026 release. See [`docs/published-tracer.md`](https://github.com/weertman/DeepStarNoodles/blob/main/docs/published-tracer.md) for the selection record and caveats.

First-party weights are MIT, matching the GitHub code. StarSeg inference uses Ultralytics (AGPL-3.0). The Hydra checkpoints include the fine-tuned DINOv3 ConvNeXt-Base backbone weights, so inference needs nothing else. Meta's original pretrained file `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` (SHA-256 `801f2ba95801bdcc9a8b77e293d38b35b355dbc813ef609cd82e18c3a185e8da`) is **not** in this repo and is only needed to train Hydra from scratch; it is distributed by Meta under the DINOv3 License.

The temporal VAE is an inference-only export of the frozen DeepNoodleTracker
checkpoint at step 10,250. It expects one support-weighted 64×64 progress raster
per second and produces a 64-D posterior mean from a centered window spanning
−30 through +29 seconds. It was trained from Hydra checkpoint
`3f691daa14a891a5f9ad163355d969f8922892c662d4d9b7864a54fd815eecee`.
The currently released Hydra checkpoint has the same channel contract, but its
output distribution has not yet been scientifically revalidated with this VAE.
Use the released VAE as the default reproducible recipe or fine-tune a new model
through the configurable representation API.

## Download into a DeepStarNoodles checkout

```bash
hf download weertman/deepstarnoodles --local-dir MODELS
```

Then run `SCRIPTS/validate_artifacts.py --roles runtime-detector released-dense released-tracer`. Add the `pretrained-backbone` role only if you placed Meta's DINOv3 file at `MODELS/dinov3/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` for Hydra training. The check skips Ultralytics `yolo26l-seg.pt` unless you add it for StarSeg from-scratch training.
