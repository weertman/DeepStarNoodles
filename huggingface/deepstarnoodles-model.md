---
license: mit
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - pose-estimation
  - instance-segmentation
  - marine-biology
  - pycnopodia
  - sunflower-star
extra_gated_prompt: >
  These are the trained DeepStarNoodles models (three StarSeg detectors, the
  Hydra field network and the Noodle Tracer).
  Access is gated so we know who uses them. The original DINOv3 backbone
  weights are not included; download those from Meta under the DINOv3 License.
extra_gated_fields:
  Affiliation: text
  I will cite DeepStarNoodles and DINOv3 where I use them: checkbox
---

# DeepStarNoodles models

Trained models of [DeepStarNoodles](https://github.com/weertman/DeepStarNoodles), which traces every arm of a sunflower sea star (*Pycnopodia helianthoides*) as a curve from base to tip. A detector finds the animals, a field network (Hydra) predicts 18 dense maps for each crop, and a recurrent tracer (Noodle Tracer) walks each arm from tip to base.

All models were trained with seed 0 on the frozen datasets: the [detection dataset](https://huggingface.co/datasets/weertman/starseg) `starseg_yolo26_20261106` (frozen 2026-11-07) and the [arm dataset](https://huggingface.co/datasets/weertman/deepstarnoodles-data) `noodles_final_20261026` (frozen 2026-10-26).

## Files

| File | Model | Parameters | SHA-256 |
| --- | --- | --- | --- |
| `starseg_binary.pt` | Detector, YOLO26-L segmentation, 2 classes: sunflower sea star, anything else annotated. The detector of the paper. | 31,375,206 | `1a38e2dab1a3e6937297e397fc7778f05746a8db2b64b851b561d6379d1f2592` |
| `starseg_fine.pt` | Detector, 37 classes: sunflower sea star, 26 other sea star species, urchins, bivalves, sand dollars, dead prey, unidentified animals | 31,438,171 | `d46917b7a5e8eaecf337c2ddbd0e128aa2b24d54a6b6a98392118a44b5254699` |
| `starseg_coarse.pt` | Detector, 4 classes: sunflower sea star, other sea star, live prey, dead prey | 31,378,804 | `5a268a704e677d463221a2c262c8ae1f43ed4db401dd7a68fb93f6988a34323c` |
| `hydra.pt` | Field network, DINOv3 ConvNeXt-Base backbone, 18 output maps | 90,883,602 | `aa056f933c1a0994fb299203aa20423c5208637d2eb4ede7e942abcc3aa91d7c` |
| `tracer.pt` | Tracer, GRU with 128 hidden units | 194,463 | `e6c1506067fc5864ab1d0d648dec2f7932632017d952753cb68f80fed2c4812f` |
| `manifest.json`, `manifest.fine.json`, `manifest.coarse.json` | Model manifests for the [inference package](https://github.com/weertman/deepstarnoodles-inference): one detector each, with Hydra and the tracer | | |
| `SHA256SUMS` | Hashes of all files above | | |

The same files are assets of the GitHub release [`models-20261117`](https://github.com/weertman/deepstarnoodles-inference/releases/tag/models-20261117), which needs no login.

## Which detector

The three detectors differ in how objects other than sunflower sea stars were labelled in training. With any of them, only detections of the sunflower sea star class are cropped and traced. Use the binary detector to reproduce the paper. The fine and coarse detectors also report the other classes, which the pipeline ignores.

Detection of sunflower sea stars on the test split (1,249 images). A prediction and an annotated mask match one to one at a mask intersection over union of at least 0.5, at a confidence of at least 0.25. Values are precision / recall / F1.

| Detector | All | Laboratory | Field without PWS | PWS |
| --- | --- | --- | --- | --- |
| binary | 0.920 / 0.903 / 0.911 | 0.906 / 0.924 / 0.915 | 0.918 / 0.668 / 0.774 | 0.976 / 1.000 / 0.988 |
| fine | 0.941 / 0.902 / 0.921 | 0.932 / 0.925 / 0.928 | 0.924 / 0.658 / 0.768 | 0.992 / 0.996 / 0.994 |
| coarse | 0.934 / 0.909 / 0.921 | 0.926 / 0.927 / 0.926 | 0.920 / 0.690 / 0.789 | 0.976 / 1.000 / 0.988 |

PWS is the dive surveys of one cruise in Prince William Sound, Alaska (243 test images). It is reported apart from the other field images (152 test images) and is not pooled with them. The detectors miss about one third of the sunflower sea stars in field images outside PWS, so field use needs more annotated field images. Each detector is a single training run: differences of about one point between them are within what a change of seed produces. The accuracy of the other classes of the fine and coarse detectors was not evaluated.

## Arm tracing

With Hydra and the tracer on annotated test crops, a traced arm and an annotated arm match when their symmetric Chamfer distance is at most 8 px in the 512 px crop.

| Test crops | Crops | Annotated arms | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| All | 553 | 5,685 | 0.821 | 0.798 | 0.809 |
| Development test set | 493 | 4,890 | 0.807 | 0.784 | 0.795 |
| StarDome rig | 30 | 419 | 0.927 | 0.905 | 0.915 |
| StarTrack rig | 30 | 376 | 0.882 | 0.854 | 0.868 |

The development test set was scored repeatedly while the method was built. The two rigs were not, but both are in our laboratory. These scores use crops cut from the annotations, so they do not include detector errors. Arm accuracy with each detector in the loop was not measured separately.

The tracer drops a starting point when its probability of being invalid is at least 0.6, a threshold chosen on the validation split. The checkpoint does not store this value. The manifests set it, and the inference package (0.5.0.dev14 or later) applies it. If you load `tracer.pt` yourself, pass 0.6.

## Use

With the [inference package](https://github.com/weertman/deepstarnoodles-inference):

```bash
curl -fL -o manifest.json \
  https://github.com/weertman/deepstarnoodles-inference/releases/download/models-20261117/manifest.json
deepstarnoodles-inference fetch-models --models manifest.json --destination models
deepstarnoodles-inference verify-models --models models/
```

Or download this repository (after accepting the access conditions and `hf auth login`):

```bash
hf download weertman/deepstarnoodles --local-dir MODELS
cd MODELS && sha256sum -c SHA256SUMS
```

Checkpoints are PyTorch pickles. Load only files whose hashes match the table.

## Notes on the files

The detector files are the terminal checkpoints of their training runs with one change: the absolute paths of the training machines stored in the checkpoint were replaced by relative ones. Every weight tensor is bit-identical to the file as trained, and each manifest gives the hash of both files.

`hydra.pt` includes the fine-tuned DINOv3 ConvNeXt-Base backbone, so inference needs nothing else. Meta's pretrained file `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` (SHA-256 `801f2ba95801bdcc9a8b77e293d38b35b355dbc813ef609cd82e18c3a185e8da`) is not in this repository. It is needed only to train Hydra and is distributed by Meta under the DINOv3 License.

## Licence

Our weights are released under the MIT licence, like the code. The detectors are run with Ultralytics, which is licensed under AGPL-3.0. The backbone of Hydra derives from DINOv3 and is subject to the DINOv3 License.

## Earlier files

Until 2026-11-17 this repository held development models from August and October 2026 (a YOLOv8 detector, two Hydra checkpoints, two tracers and a temporal VAE). They were trained before the datasets were frozen and are not the models of the paper. They were removed from the current revision and remain in the history of this repository (last revision with them: `0881c427eb0f1f564ef76fbd8399173121169132`).
