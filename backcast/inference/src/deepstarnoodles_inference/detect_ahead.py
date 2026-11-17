# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Experimental archive overlap: detect and prepare crops ahead of Hydra/GRU.

One producer thread takes ordered source groups, runs StarSeg and training-time
crop preparation, and hands each group to the consumer through a bounded FIFO
queue. The consumer runs Hydra, the GRU and output writing in the original
order. Every model call, crop and tensor is unchanged; only host scheduling
differs, so decode, StarSeg pre/postprocessing and crop resizing overlap
Hydra/GRU work. Native-resolution crop pixels are released once their Hydra
inputs exist.

The producer issues CUDA work on its own stream and leaves that stream idle
before releasing CAPTURE_LOCK and handing a group off. Without that handoff
synchronization, rare illegal memory accesses occurred locally under
multi-process load (about 3 in 20 processes with concurrent GRU streams); with
it, 1 in 48, and a shared stream did no better (3 in 64). Faults abort the
process; no run produced different predictions. The root cause was not
isolated; single-process runs never faulted. Every fault has been an Xid 31 MMU
fault reading through the constant cache (GPCCLIENT_GCC), in processes
time-slicing one GPU; none has occurred under CUDA MPS.
"""
from __future__ import annotations

import contextlib
import queue
import threading
import time
from dataclasses import replace

import cv2
import torch

from .capture_lock import CAPTURE_LOCK

_END = object()


class _Failure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


class DetectAhead:
    """Bounded producer of ``(paths, records, crop count, crop groups)`` items."""

    EMPTY = object()

    def __init__(self, pipeline, source, *, depth: int, crop_batch_size: int, timer, stream=None) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 8:
            raise ValueError("detect-ahead depth must be an integer between 1 and 8")
        self.pipeline = pipeline
        self.source = source
        self.crop_batch_size = int(crop_batch_size)
        self.timer = timer
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self.device = pipeline.device
        # A caller running many sources passes one stream for all of them (see video_service.run_worker).
        self.stream = stream if stream is not None or self.device.type != "cuda" else torch.cuda.Stream(device=self.device)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, name="detect-ahead", daemon=True)
        self.depth = depth
        self.produced_groups = 0
        self.producer_ms = 0.0
        self.consumer_wait_ms = 0.0
        self.peak_queued = 0
        self._ended = False

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        while self._thread.is_alive():
            try:
                self.queue.get(timeout=0.05)
            except queue.Empty:
                pass
        self._thread.join()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                self.peak_queued = max(self.peak_queued, self.queue.qsize())
                return True
            except queue.Full:
                continue
        return False

    def _prepare(self, paths, frames, start_index):
        pipeline = self.pipeline
        if frames is None:
            frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
            if any(frame is None for frame in frames):
                raise ValueError("could not decode one or more source images")
        frames, records, observations = pipeline.detect_frames(
            frames, start_index, self.timer, source_paths=paths)
        uploads = pipeline.upload_frames(frames, observations) if observations else None
        groups = []
        for offset in range(0, len(observations), self.crop_batch_size):
            group = observations[offset:offset + self.crop_batch_size]
            crop_group = pipeline.crop_group(frames, group)
            images = pipeline.preprocess_crops(crop_group, uploads)
            event = None
            if self.stream is not None:
                event = torch.cuda.Event()
                event.record(self.stream)
            # Hydra inputs exist; the consumer needs only crop geometry.
            crop_group = [(i, j, replace(crop, crop_rgb=None)) for i, j, crop in crop_group]
            groups.append((crop_group, images, event))
        if self.stream is not None:
            # Idle before CAPTURE_LOCK is released (see module docstring).
            self.stream.synchronize()
        return records, len(observations), groups

    def _produce(self) -> None:
        start_index = 0
        try:
            stream = torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()
            with torch.inference_mode(), stream:
                while not self._stop.is_set():
                    try:
                        paths, frames = next(self.source)
                    except StopIteration:
                        break
                    started = time.perf_counter()
                    with CAPTURE_LOCK:
                        records, crops, groups = self._prepare(paths, frames, start_index)
                    del frames
                    self.producer_ms += (time.perf_counter() - started) * 1000
                    self.produced_groups += 1
                    if not self._put((paths, records, crops, groups)):
                        return
                    start_index += len(paths)
            self._put(_END)
        except BaseException as error:  # surfaced by the consumer
            self._put(_Failure(error))

    def get(self):
        """Next prepared group in source order, or None after the last group."""
        if self._ended:
            return None
        started = time.perf_counter()
        item = self.queue.get()
        self.consumer_wait_ms += (time.perf_counter() - started) * 1000
        if item is _END:
            self._ended = True
            return None
        if isinstance(item, _Failure):
            raise item.error
        return item

    def poll(self):
        """Next prepared group without waiting: an item, ``EMPTY``, or None at the end."""
        if self._ended:
            return None
        try:
            item = self.queue.get_nowait()
        except queue.Empty:
            return self.EMPTY
        if item is _END:
            self._ended = True
            return None
        if isinstance(item, _Failure):
            raise item.error
        return item

    def consume(self, groups):
        """Yield crop groups once their inputs are ready on the consumer stream."""
        for crop_group, images, event in groups:
            if event is not None:
                current = torch.cuda.current_stream(self.device)
                current.wait_event(event)
                # The producer stream allocated these inputs; keep the block
                # reserved until consumer-stream work using it completes.
                images.record_stream(current)
            yield crop_group, images

    def job(self, group):
        """Crop-stream job: make one prepared group usable on a workspace stream."""
        crop_group, images, event = group

        def ready(workspace):
            if event is not None:
                workspace.stream.wait_event(event)
                images.record_stream(workspace.stream)
            return crop_group, images
        return ready

    def report(self) -> dict:
        return {"depth": self.depth, "produced_groups": self.produced_groups,
                "producer_ms": self.producer_ms, "consumer_wait_ms": self.consumer_wait_ms,
                "peak_queued": self.peak_queued}
