# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Experimental bounded, ordered image decoding for independent archives.

The reservation bound covers arrays retained by this reader, including jobs
being decoded. Consumer-owned frames, codec scratch space, encoded file buffers,
and OS page cache are outside that bound and require the process RSS guard.
Inputs must remain immutable. Native reads are not forcibly cancellable; the
existing process watchdog remains necessary for a blocked filesystem/codec.
"""
from __future__ import annotations
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from itertools import islice
import time

import cv2
import numpy as np
from PIL import Image


def expected_bytes(path):
    with Image.open(path) as image:
        width, height = image.size
    if min(width, height) <= 0:
        raise ValueError("invalid image dimensions")
    return width * height * 3


def read_frame(path, size):
    started = time.perf_counter()
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"could not decode source image: {path}")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or frame.nbytes != size:
        raise ValueError("decoded size differs from reserved header size; keep inputs immutable")
    return frame, (time.perf_counter() - started) * 1000


class OrderedDecoder:
    def __init__(self, paths, *, workers=2, max_pending=4,
                 byte_budget=256 * 1024**2, max_frame_bytes=64 * 1024**2):
        for name, value in (("workers", workers), ("max_pending", max_pending),
                            ("byte_budget", byte_budget), ("max_frame_bytes", max_frame_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if workers > 4 or max_pending < workers or max_pending > 64 or max_frame_bytes > byte_budget:
            raise ValueError("require workers<=4, workers<=max_pending<=64, max_frame_bytes<=byte_budget")
        self.paths = iter(paths)
        self.workers, self.max_pending = workers, max_pending
        self.byte_budget, self.max_frame_bytes = byte_budget, max_frame_bytes
        self._pool = None
        self._pending = deque()
        self._next = None
        self._exhausted = self._closed = False
        self.reserved_bytes = self.peak_reserved_bytes = self.peak_pending = 0
        self.submitted = self.delivered = 0
        self.decode_ms = self.wait_ms = 0.0

    def __enter__(self):
        if self._pool is not None or self._closed:
            raise RuntimeError("OrderedDecoder cannot be reopened")
        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="archive-decode")
        # Header discovery happens on demand: errors belong to the next source
        # position, and no frames are decoded merely by entering the context.
        return self

    def _fill(self):
        while not self._exhausted and len(self._pending) < self.max_pending:
            if self._next is None:
                try:
                    path = Path(next(self.paths))
                except StopIteration:
                    self._exhausted = True
                    return
                try:
                    size = expected_bytes(path)
                    if size > self.max_frame_bytes:
                        raise MemoryError(f"frame exceeds max_frame_bytes before decoding: {path}")
                    self._next = (path, size, None)
                except Exception as exc:
                    # Delay metadata errors until earlier sources are yielded.
                    self._next = (path, 0, exc)
            path, size, error = self._next
            if self.reserved_bytes + size > self.byte_budget:
                return
            self._next = None
            if error is not None:
                self._pending.append((path, size, None, error))
                self._exhausted = True
            else:
                future = self._pool.submit(read_frame, path, size)
                self._pending.append((path, size, future, None))
                self.reserved_bytes += size
                self.submitted += 1
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
            self.peak_pending = max(self.peak_pending, len(self._pending))

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self._pool is None:
            raise RuntimeError("use OrderedDecoder as a context manager")
        self._fill()
        if not self._pending:
            raise StopIteration
        path, size, future, error = self._pending.popleft()
        if error is not None:
            self.close()
            raise error
        started = time.perf_counter()
        try:
            frame, elapsed = future.result()
        except BaseException:
            self.reserved_bytes -= size
            self.close()
            raise
        self.wait_ms += (time.perf_counter() - started) * 1000
        self.decode_ms += elapsed
        self.reserved_bytes -= size
        self.delivered += 1
        del future
        # Returned pixels are now consumer-owned. A caller assembling a large
        # detector batch still pays that batch's existing resident RAM cost.
        self._fill()
        return path, frame

    def close(self):
        if self._closed:
            return
        self._closed = True
        for _, _, future, _ in self._pending:
            if future is not None:
                future.cancel()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._next = None
        self.reserved_bytes = 0

    def __exit__(self, *_exc):
        self.close()

    def report(self):
        return {"workers": self.workers, "max_pending": self.max_pending,
                "byte_budget": self.byte_budget, "max_frame_bytes": self.max_frame_bytes,
                "peak_reserved_bytes": self.peak_reserved_bytes, "peak_pending": self.peak_pending,
                "submitted": self.submitted, "delivered": self.delivered,
                "sum_worker_decode_ms": self.decode_ms, "consumer_wait_ms": self.wait_ms,
                "consumer_wait_scope": "Future.result only; archive_decode_wait profiling also includes headers and batch assembly",
                "scope": "reader-retained decoded arrays; excludes consumer frames, native codec scratch and page cache"}


class ArchiveSourceGroups:
    """Yield unchanged source batches, optionally decoded by bounded workers.

    With workers disabled, frames are None and the existing synchronous caller
    remains responsible for decode. The reader never invokes a model or CUDA.
    """
    def __init__(self, paths, batch_size, args, timer):
        self.paths, self.batch_size, self.timer = iter(paths), batch_size, timer
        workers = getattr(args, "decode_workers", 0)
        self.reader = (OrderedDecoder(
            self.paths, workers=workers,
            max_pending=getattr(args, "decode_prefetch_frames", 4),
            byte_budget=getattr(args, "decode_prefetch_mib", 256) * 1024**2,
            max_frame_bytes=getattr(args, "decode_max_frame_mib", 64) * 1024**2,
        ) if workers else None)

    def __enter__(self):
        if self.reader is not None:
            self.reader.__enter__()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.reader is None:
            paths = list(islice(self.paths, self.batch_size))
            if not paths:
                raise StopIteration
            return paths, None
        if self.reader._exhausted and not self.reader._pending:
            raise StopIteration
        # This is consumer wait/header work, not a serial decode stage. Worker
        # decode overlaps inference and its duration is reported separately.
        with self.timer.stage("archive_decode_wait"):
            values = list(islice(self.reader, self.batch_size))
        if not values:
            raise StopIteration
        paths, frames = zip(*values)
        return list(paths), list(frames)

    def report(self):
        return self.reader.report() if self.reader is not None else None

    def __exit__(self, *exc):
        if self.reader is not None:
            self.reader.__exit__(*exc)


def accumulate_decode_report(previous, current):
    """Bounded summary across staged shards; worker times overlap their consumer."""
    if previous is None:
        return {**current, "shards": 1}
    result = {**previous, "shards": previous["shards"] + 1}
    for name in ("peak_reserved_bytes", "peak_pending"):
        result[name] = max(previous[name], current[name])
    for name in ("submitted", "delivered", "sum_worker_decode_ms", "consumer_wait_ms"):
        result[name] = previous[name] + current[name]
    return result
