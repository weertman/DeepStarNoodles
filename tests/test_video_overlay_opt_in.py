# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from deepstarnoodles.crop import CropResult


@pytest.mark.parametrize("module_name", ["infer_video", "infer_videos"])
@pytest.mark.parametrize("multi_star", [False, True])
def test_video_overlays_are_opt_in_without_changing_tracks(
    module_name: str,
    multi_star: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = importlib.import_module(module_name)
    crop = CropResult(
        crop_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        bbox_xyxy=(0.0, 0.0, 16.0, 16.0),
        center=(8.0, 8.0),
        confidence=0.9,
        crop_left=0.0,
        crop_top=0.0,
        crop_side=16.0,
        mask_polygon=None,
        star_track_id=7 if multi_star else None,
    )

    class Capture:
        def __init__(self, _path):
            self.remaining = 2

        def isOpened(self):
            return True

        def get(self, _property):
            return 30.0

        def read(self):
            if not self.remaining:
                return False, None
            self.remaining -= 1
            return True, np.full((16, 16, 3), self.remaining, dtype=np.uint8)

        def release(self):
            pass

    class Cropper:
        def __init__(self, *_args, **_kwargs):
            self.model = object()
            self.tracker_config = Path("botsort.yaml")

        def crop(self, frame):
            return crop if frame[0, 0, 0] else None

        def track_crops(self, frame):
            return [crop] if frame[0, 0, 0] else []

        def reset(self):
            pass

    class Hydra:
        device = torch.device("cpu")
        image_size = 16

        def __init__(self, *_args):
            pass

        def preprocess_batch(self, crops):
            return torch.zeros(len(crops), 3, 16, 16)

        def predict_tensor_batch(self, images):
            fields = torch.zeros(len(images), 18, 16, 16)
            fields[:, 5, 8, 8] = 1.0
            return fields, fields

    def trace_batch(raw, *_args):
        return [[{
            "polyline": np.asarray([[8.0, 8.0], [12.0, 8.0]]),
            "identity": np.zeros(8),
        }] for _ in raw]

    written_frames: list[np.ndarray] = []

    class Writer:
        def __init__(self, *_args):
            pass

        def write(self, frame):
            written_frames.append(frame)

        def release(self):
            pass

    monkeypatch.setattr(cli.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(cli.cv2, "VideoWriter", Writer)
    monkeypatch.setattr(cli, "StarCropper", Cropper)
    monkeypatch.setattr(cli, "TrackedStarCropper", Cropper)
    monkeypatch.setattr(cli, "HydraPredictor", Hydra)
    monkeypatch.setattr(cli, "load_noodle_tracer_checkpoint", lambda *_a, **_k: (None,) * 4)
    monkeypatch.setattr(cli, "resolve_invalid_seed_threshold", lambda params, _threshold=None: params)
    monkeypatch.setattr(cli, "trace_arms_noodle_batch", trace_batch)

    payloads = []
    for enabled in (False, True):
        draw_calls = []

        def draw(frame, *_args, **kwargs):
            assert enabled, "overlay rendering must be skipped without an output"
            if multi_star:
                assert len(kwargs["support_masks"]) == (1 if frame[0, 0, 0] else 0)
            draw_calls.append(frame)
            return frame.copy()

        monkeypatch.setattr(cli, "draw_overlay", draw)
        monkeypatch.setattr(cli, "draw_multi_overlay", draw)
        argv = [module_name, "--video", "input.mp4", "--device", "cpu"]
        destination = tmp_path / str(enabled)
        if module_name == "infer_video":
            output = destination / "tracks.json"
            argv += ["--output", str(output)]
            if enabled:
                argv += ["--overlay", str(destination / "overlay.mp4")]
        else:
            output = destination / "00_input_tracks.json"
            argv += ["--output-dir", str(destination)]
            if enabled:
                argv += ["--overlay-dir", str(destination / "overlays")]
        if multi_star:
            argv += ["--multi-star", "--overlay-support"]
        monkeypatch.setattr("sys.argv", argv)

        assert cli.main() == 0
        assert len(draw_calls) == (2 if enabled else 0)
        payloads.append(json.loads(output.read_text()))

    assert payloads[0] == payloads[1]
    assert [frame["detected"] for frame in payloads[0]["frames"]] == [True, False]
    assert len(written_frames) == 2
