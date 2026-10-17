# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Image-space augmentation for refreshed Noodle Tracer training.

The released tracer is trained on *refreshed* generations: every cycle, each
training image is augmented in pixel space, its annotated arm polylines are
moved with the same transform, and the frozen Hydra model is rerun on the
augmented image. This differs from transforming dense outputs that were cached
once from the original image.

The augmentation of one image is a deterministic function of the augmentation
seed, the cycle number, and the image's row in the cohort's training index, so a
generation can be regenerated exactly.
"""
from __future__ import annotations

import hashlib
import random
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance

REFRESH_IMAGE_SIZE = 512
BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE = (0.85, 1.15)
SATURATION_RANGE = (0.85, 1.15)


def draw_augmentation(seed: int, cycle: int, row_index: int) -> dict:
    """Augmentation parameters for one image of one generation.

    Flips and quarter turns are exact isometries of the square crop grid, so
    pixels and polyline coordinates stay aligned without resampling.
    """
    digest = hashlib.sha256(f"{seed}:{cycle}:{row_index}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return {
        "flip_x": rng.random() < 0.5,
        "flip_y": rng.random() < 0.5,
        "rotate_k": rng.randrange(4),
        "brightness": rng.uniform(*BRIGHTNESS_RANGE),
        "contrast": rng.uniform(*CONTRAST_RANGE),
        "saturation": rng.uniform(*SATURATION_RANGE),
    }


def augment_image_and_arms(
    image: Image.Image,
    arms: Sequence[np.ndarray],
    params: dict,
    *,
    size: int = REFRESH_IMAGE_SIZE,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Apply one augmentation to an RGB image and its arm polylines.

    ``arms`` are ``(N, 2)`` ``(x, y)`` arrays in the coordinates of the image
    resized to ``size`` × ``size``. Returns the augmented ``uint8`` RGB array
    of shape ``(size, size, 3)`` and the transformed polylines. Photometric
    changes affect pixels only.
    """
    pixels = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
    paths = [np.asarray(arm, dtype=np.float32).copy() for arm in arms]
    if params["flip_x"]:
        pixels = np.flip(pixels, axis=1)
        for path in paths:
            path[:, 0] = size - 1 - path[:, 0]
    if params["flip_y"]:
        pixels = np.flip(pixels, axis=0)
        for path in paths:
            path[:, 1] = size - 1 - path[:, 1]
    for _ in range(int(params["rotate_k"]) % 4):
        pixels = np.rot90(pixels)
        for path in paths:
            old_x = path[:, 0].copy()
            path[:, 0] = path[:, 1]
            path[:, 1] = size - 1 - old_x
    pil = Image.fromarray(np.ascontiguousarray(pixels))
    pil = ImageEnhance.Brightness(pil).enhance(params["brightness"])
    pil = ImageEnhance.Contrast(pil).enhance(params["contrast"])
    pil = ImageEnhance.Color(pil).enhance(params["saturation"])
    for path in paths:
        np.clip(path, 0, size - 1, out=path)
    return np.asarray(pil, dtype=np.uint8), paths
