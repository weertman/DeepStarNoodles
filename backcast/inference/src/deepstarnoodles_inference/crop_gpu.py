# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Byte-identical GPU crop preparation for Hydra.

The CPU path cuts a zero-padded square crop from the BGR frame, converts it to
RGB, resizes it with Pillow's bilinear filter and scales it to float32 in
[0, 1] (``HydraPredictor.preprocess_batch``). For 8-bit images Pillow
resamples in two separable passes, horizontal then vertical, with fixed-point
coefficients of 22 fractional bits, rounding each pass to uint8 (Pillow
``src/libImaging/Resample.c``). This module computes the same coefficients in
float64 on the host, in Pillow's order of operations, and applies both passes
with int32 arithmetic on the GPU, so the Hydra input matches the CPU path bit
for bit. Only the frame is uploaded: border pixels are zero, so taps that fall
outside the frame are dropped instead of padded, which leaves every sum
unchanged. Resampling acts on each channel alone, so channels are reordered
after resizing, and the final scaling is a lookup of the CPU path's own
float32 quotients.
"""
from __future__ import annotations

import numpy as np
import torch

PRECISION_BITS = 32 - 8 - 2
# Bounds the int32 tap workspace of one resampling chunk.
WORKSPACE_BYTES = 64 * 1024**2


def bilinear_coefficients(in_size: int, out_size: int):
    """Pillow's ``precompute_coeffs`` and ``normalize_coeffs_8bpc`` for its bilinear filter.

    Returns ``(first, counts, fixed)``: the first input index, the number of
    taps and the int32 fixed-point weights ``(out_size, ksize)`` of each output.
    """
    if in_size < 1 or out_size < 1:
        raise ValueError("resample sizes must be positive")
    scale = float(in_size) / out_size
    filterscale = max(scale, 1.0)
    support = 1.0 * filterscale
    ksize = int(np.ceil(support)) * 2 + 1
    center = (np.arange(out_size, dtype=np.float64) + 0.5) * scale
    first = np.maximum(np.trunc(center - support + 0.5), 0).astype(np.int64)
    last = np.minimum(np.trunc(center + support + 0.5), in_size).astype(np.int64)
    counts = last - first
    taps = np.arange(ksize, dtype=np.int64)
    position = ((taps[None, :] + first[:, None]).astype(np.float64) - center[:, None] + 0.5) * (1.0 / filterscale)
    distance = np.abs(position)
    weights = np.where(distance < 1.0, 1.0 - distance, 0.0)
    weights[taps[None, :] >= counts[:, None]] = 0.0
    # Pillow accumulates the normalizer tap by tap; cumsum keeps that order.
    total = np.cumsum(weights, axis=1)[:, -1]
    weights = np.divide(weights, total[:, None], out=weights.copy(), where=total[:, None] != 0.0)
    fixed = np.trunc(0.5 + weights * (1 << PRECISION_BITS)).astype(np.int32)
    return first, counts, fixed


def _tap_tables(first, fixed, offset: int, limit: int, device):
    """Source indices and weights for taps; taps outside ``[0, limit)`` read zero pixels."""
    index = first[:, None] + np.arange(fixed.shape[1], dtype=np.int64)[None, :] + offset
    weights = np.where((index >= 0) & (index < limit), fixed, 0).astype(np.int32)
    index = np.clip(index, 0, limit - 1)
    return (torch.from_numpy(index).to(device, non_blocking=False),
            torch.from_numpy(weights).to(device, non_blocking=False))


def _resample(source: torch.Tensor, index: torch.Tensor, weights: torch.Tensor, dim: int) -> torch.Tensor:
    """One fixed-point pass of uint8 ``source`` along ``dim`` (0 rows, 1 columns)."""
    out_size, ksize = index.shape
    shape = list(source.shape)
    shape[dim] = out_size
    acc = torch.full(shape, 1 << (PRECISION_BITS - 1), dtype=torch.int32, device=source.device)
    per_tap = acc.numel() * 4
    step = max(1, min(ksize, WORKSPACE_BYTES // max(per_tap, 1)))
    for start in range(0, ksize, step):
        stop = min(ksize, start + step)
        taps = source.index_select(dim, index[:, start:stop].reshape(-1)).to(torch.int32)
        if dim == 0:
            taps = taps.view(out_size, stop - start, *shape[1:])
            acc += (taps * weights[:, start:stop].view(out_size, stop - start, 1, 1)).sum(dim=1, dtype=torch.int32)
        else:
            taps = taps.view(shape[0], out_size, stop - start, shape[2])
            acc += (taps * weights[:, start:stop].view(1, out_size, stop - start, 1)).sum(dim=2, dtype=torch.int32)
    # Pillow's clip8: values at or above 255.5 saturate; the rounding offset keeps sums positive.
    return (acc >> PRECISION_BITS).clamp_(0, 255).to(torch.uint8)


class GpuCropPreparer:
    """Hydra inputs ``(B, 3, S, S)`` float32 from uploaded BGR frames, equal to the CPU path."""

    def __init__(self, size: int, device) -> None:
        self.size = int(size)
        self.device = torch.device(device)
        quotients = np.arange(256, dtype=np.float32) / 255.0
        self._scale = torch.from_numpy(quotients).to(self.device)
        self._coefficients = {}
        self.crops = 0
        self.uploaded_bytes = 0

    def _coefficients_for(self, side: int):
        cached = self._coefficients.get(side)
        if cached is None:
            cached = self._coefficients[side] = bilinear_coefficients(side, self.size)
        return cached

    def upload(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Copy one contiguous HxWx3 uint8 BGR frame to the device."""
        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frames must be HxWx3 BGR uint8 arrays")
        self.uploaded_bytes += frame_bgr.nbytes
        return torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(self.device)

    def crop(self, frame: torch.Tensor, left: int, top: int, side: int) -> torch.Tensor:
        """One ``(3, S, S)`` float32 RGB crop from an uploaded BGR frame."""
        height, width = int(frame.shape[0]), int(frame.shape[1])
        rows = slice(max(0, top), min(height, top + side))
        if side == self.size:
            # The CPU path skips resizing; gather the padded crop directly.
            index = torch.arange(side, device=self.device)
            row_ok = (index + top >= 0) & (index + top < height)
            col_ok = (index + left >= 0) & (index + left < width)
            out = frame.index_select(0, (index + top).clamp(0, height - 1))
            out = out.index_select(1, (index + left).clamp(0, width - 1))
            out = out * (row_ok.view(-1, 1, 1) & col_ok.view(1, -1, 1))
        else:
            first, _counts, fixed = self._coefficients_for(side)
            if rows.start >= rows.stop or max(0, left) >= min(width, left + side):
                out = torch.zeros(self.size, self.size, 3, dtype=torch.uint8, device=self.device)
            else:
                col_index, col_weights = _tap_tables(first, fixed, left, width, self.device)
                horizontal = _resample(frame[rows], col_index, col_weights, dim=1)
                # Rows above the frame are zero in the padded crop; index the uploaded rows.
                row_index, row_weights = _tap_tables(first, fixed, top - rows.start, rows.stop - rows.start,
                                                     self.device)
                out = _resample(horizontal, row_index, row_weights, dim=0)
        self.crops += 1
        rgb = out.flip(-1)
        return self._scale[rgb.long()].permute(2, 0, 1)

    def prepare(self, uploads: dict, crop_group) -> torch.Tensor:
        """Stack the crops of ``crop_group`` (``(i, j, CropResult)``) from ``uploads[i]``."""
        crops = [self.crop(uploads[i], int(crop.crop_left), int(crop.crop_top), int(crop.crop_side))
                 for i, _j, crop in crop_group]
        return torch.stack(crops).contiguous()

    def summary(self) -> dict:
        return {"crops": self.crops, "uploaded_bytes": self.uploaded_bytes,
                "coefficient_tables": len(self._coefficients)}
