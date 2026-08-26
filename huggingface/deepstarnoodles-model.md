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
  These are trained DeepStarNoodles weights (StarSeg, Hydra, Noodle Tracer).
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
| `tracer/noodle_tracer_seed20260807_endpoint_mse_calibrated.pt` | Noodle Tracer | `bb7f1b5da4f998006163b0bfbba005b417dfd6e8ab313d8e3869b7224fcc5c6c` |

First-party weights are MIT, matching the GitHub code. StarSeg inference uses Ultralytics (AGPL-3.0). The DINOv3 ConvNeXt-Base backbone is **not** in this repo; get `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` from Meta (SHA-256 `801f2ba95801bdcc9a8b77e293d38b35b355dbc813ef609cd82e18c3a185e8da`).

## Download into a DeepStarNoodles checkout

```bash
hf download weertman/deepstarnoodles --local-dir MODELS
```

Then place the Meta DINOv3 file at `MODELS/dinov3/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` and run `SCRIPTS/validate_artifacts.py`.
