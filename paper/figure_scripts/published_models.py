#!/usr/bin/env python3
"""The three published models used by the figure scripts in this folder.

* detector: YOLO26-L-seg, two classes (scheme B), seed 0, terminal weight. The scripts were run
  with the file as trained (SHA-256 ``e48c5ce4...``); the released ``starseg_binary.pt`` has the
  same weights with the training paths made relative.
* dense model ("Hydra"): DINOv3 ConvNeXt-Base, 18 output maps, seed 0, terminal EMA weight
* tracer: GRU, hidden size 128, cycle 30, seed 0; the weights of the released ``tracer.pt``
  (``e6c15060...``) with the invalid-seed threshold 0.6 written into ``trace_params``

Set ``DSN_MODEL_DIR`` to the folder holding ``starseg.pt``, ``hydra.pt`` and ``tracer.pt``.
"""
from __future__ import annotations

import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get('DSN_MODEL_DIR', Path(__file__).resolve().parents[2] / 'MODELS'))
MANIFEST = MODEL_DIR / 'manifest.json'

DETECTOR = MODEL_DIR / 'starseg.pt'
DETECTOR_SHA256 = 'e48c5ce4c0ca9f9617e0856a2d3bdac17a4326bd463d5df4775c650f85bee0ba'
DETECTOR_CLASS = 'sunflower_star'          # class 0 of scheme B; class 1 is not_sunflower

HYDRA = MODEL_DIR / 'hydra.pt'
HYDRA_SHA256 = 'aa056f933c1a0994fb299203aa20423c5208637d2eb4ede7e942abcc3aa91d7c'

TRACER = MODEL_DIR / 'tracer.pt'
TRACER_SHA256 = '8e346e2192541fb917cb42333fd797ce31ba32390fe07dbec9ba22d6a041c64c'
TRACER_SOURCE_SHA256 = 'e6c1506067fc5864ab1d0d648dec2f7932632017d952753cb68f80fed2c4812f'
INVALID_SEED_THRESHOLD = 0.6

# Development models that must not appear in any regenerated figure.
DEVELOPMENT_SHA256 = {
    'tracer_f39dc82b': 'f39dc82b9bc5d38e2053c2535ca30055a9d03140a713dbb991cb7058130d5287',
    'tracer_bb7f1b5d': 'bb7f1b5da4f998006163b0bfbba005b417dfd6e8ab313d8e3869b7224fcc5c6c',
    'hydra_dcf7140c': 'dcf7140c9ef0129d8f66b33b05af6d0ebb064f84b41ce61e51e67061799e3f02',
    'detector_e150': 'e2127b2a320ba8d9a67d70794ca5f61c7ef124907cad5dc59c5715bfcc322250',
}
