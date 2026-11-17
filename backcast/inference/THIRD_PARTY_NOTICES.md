# Third-party notices

DeepStarNoodles runtime code is copyright 2026 Weertman, provided under the MIT license in `LICENSE`.

The unmodified `src/deepstarnoodles_inference/_vendor/dinov3/convnext.py` is copyright Meta Platforms, Inc. and affiliates. It is distributed under the DINOv3 License Agreement, reproduced in `licenses/DINOV3-LICENSE.md` and beside the vendored source. DINOv3 code and model derivatives retain that agreement; the project MIT license does not replace it. The upstream source is [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).

`config/botsort.yaml` reproduces the pinned Ultralytics BoT-SORT configuration under AGPL-3.0. The full license is in `licenses/ULTRALYTICS-AGPL-3.0.txt`. The installed Ultralytics 8.4.51 runtime is separately distributed under its upstream [license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). The combined runtime includes these separately licensed components.

PyTorch, torchvision, NumPy, SciPy, Pillow, OpenCV, and other installed dependencies retain their own licenses and notices. The torchvision MobileNetV3 architecture is used by the tracer; its trained encoder weights come from the explicitly supplied tracer checkpoint. No pretrained model weights are distributed in this repository or wheel, and inference does not download initialization weights.

`src/deepstarnoodles_inference/_vendor/noodle_polylines/` is an unmodified copy of DeepStarNoodles `CORE/noodle_polylines` (copyright 2026 Weertman, MIT).

`UPSTREAM.json` records source-file hashes at extraction. Runtime changes remove research-checkout paths and training imports while retaining checkpoint tensor names and normalization/decoding behavior.
