# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Experimental resident archive crop accumulation across singleton detections.

Only crop pixels and ordered source records survive across source frames. The
crop queue is bounded by count and bytes; completed leading records are emitted
immediately. No model, CUDA, or tracker work is moved to a background thread.
"""
from __future__ import annotations

from collections import Counter
import json
import time

import cv2
import numpy as np

from . import benchmark as b
from .archive_decode import ArchiveSourceGroups
from .crop import _crop_result, _square_crop_geometry
from .records import finalize_frame


class ArchiveCropBuffer:
    def __init__(self, pipeline, args, timer, sink):
        window = getattr(args, "archive_frame_window", 0)
        budget_mib = getattr(args, "archive_crop_buffer_mib", 256)
        if (args.mode != "archive" or args.detector_batch_size != 1
                or pipeline.live_cropper is not None):
            raise ValueError("buffered archives require independent images and detector batch size 1")
        if not 1 <= window <= 256 or not 1 <= budget_mib <= 1024 or not 1 <= args.crop_batch_size <= 128:
            raise ValueError("invalid archive window, crop buffer budget, or crop batch size")
        self.pipeline, self.args, self.timer, self.sink = pipeline, args, timer, sink
        self.window, self.byte_budget = window, budget_mib * 1024**2
        self.records, self.remaining = {}, {}
        self.queue, self.queue_bytes = [], 0
        self.crops = self.arms = self.emitted = self.empty = 0
        self.peak_frames = self.peak_crops = self.peak_bytes = 0
        self.flush_reasons = dict.fromkeys(("crop_batch", "frame_window", "byte_budget", "eof"), 0)
        self.batch_sizes = Counter()
        self.closed = False

    def make_room(self):
        if len(self.records) >= self.window:
            self.flush("frame_window")
            if len(self.records) >= self.window:
                raise RuntimeError("archive window cannot advance with unfinished source observations")

    def admit(self, record, detections):
        self.make_room()
        index = record["frame"]
        if index in self.records or index != self.emitted + len(self.records):
            raise ValueError("source records must be contiguous and admitted exactly once")
        self.records[index], self.remaining[index] = record, detections
        self.peak_frames = max(self.peak_frames, len(self.records))
        self.emit_ready()

    def add_crop(self, frame, source_index, detection_index, detection):
        # Use exactly the cropper's integer geometry, before cv2 allocation.
        _, _, side = _square_crop_geometry(detection["xyxy"], self.pipeline.crop_config.context_scale)
        polygon = detection.get("polygon")
        polygon_bytes = 0 if polygon is None else polygon.nbytes
        reserved = side * side * 3 + polygon_bytes
        if reserved > self.byte_budget:
            raise MemoryError("one crop exceeds --archive-crop-buffer-mib before allocation; increase the budget or disable buffering")
        if self.queue_bytes + reserved > self.byte_budget:
            self.flush("byte_budget")
        with self.timer.stage("archive_crop_create"):
            # Own polygon storage too: a view must not retain an uncharged
            # backing array from an earlier source frame.
            owned = dict(detection, polygon=None if polygon is None else np.array(polygon, copy=True))
            crop = _crop_result(frame, owned, context_scale=self.pipeline.crop_config.context_scale)
        actual_bytes = crop.crop_rgb.nbytes + (0 if crop.mask_polygon is None else crop.mask_polygon.nbytes)
        if actual_bytes != reserved:
            raise RuntimeError("crop allocation differs from its reserved geometry")
        self.queue.append((source_index, detection_index, crop))
        self.queue_bytes += reserved
        self.crops += 1
        self.peak_crops = max(self.peak_crops, len(self.queue))
        self.peak_bytes = max(self.peak_bytes, self.queue_bytes)
        if len(self.queue) >= self.args.crop_batch_size:
            self.flush("crop_batch")

    def flush(self, reason):
        if not self.queue:
            self.emit_ready()
            return
        with self.timer.stage("crop_preprocess_transfer"):
            images = self.pipeline.preprocess_crops(self.queue)
        try:
            arms = self.pipeline.infer_crops(self.queue, images, self.records, 0, self.timer)
        finally:
            del images
        self.arms += arms
        self.flush_reasons[reason] += 1
        self.batch_sizes[len(self.queue)] += 1
        for index, _, _ in self.queue:
            self.remaining[index] -= 1
        self.queue.clear()
        self.queue_bytes = 0
        self.emit_ready()
        self.check_ram()

    def emit_ready(self):
        # FIFO crops mean only the leading unfinished source can retain partial
        # star output. Later empty records wait without growing past the window.
        if not self.records or self.remaining[next(iter(self.records))]:
            return
        with self.timer.stage("serialize_write"):
            written = False
            while self.records:
                index = next(iter(self.records))
                if self.remaining[index]:
                    break
                record = finalize_frame(self.records[index], getattr(self.args, "fps", None))
                self.sink.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                self.emitted += 1
                self.empty += not record["stars"]
                del self.records[index], self.remaining[index]
                written = True
            if written:
                self.sink.flush()

    def check_ram(self):
        if b.host_peak_bytes() > self.args.ram_limit_gib * 1024**3:
            raise MemoryError("process high-water RSS exceeded --ram-limit-gib")

    def report(self):
        return {"frame_window": self.window, "crop_byte_budget": self.byte_budget,
                "peak_pending_frames": self.peak_frames, "peak_pending_crops": self.peak_crops,
                "peak_crop_bytes": self.peak_bytes, "emitted_source_images": self.emitted,
                "flush_reasons": self.flush_reasons, "crop_batch_histogram": dict(self.batch_sizes),
                "scope": "retained RGB crop and owned polygon arrays plus source-record count; excludes current full frame, detector results, partial output for one source and decoded/output data for the active crop group, model tensors, and native scratch"}

    def close(self):
        if self.closed:
            return
        self.queue.clear()
        self.queue_bytes = 0
        self.records.clear()
        self.remaining.clear()
        self.closed = True


def measure_buffered_archive(pipeline, args, count, sink, profile):
    timer = b.Timer(pipeline.device, profile)
    buffer = ArchiveCropBuffer(pipeline, args, timer, sink)
    groups_ms, sources = [], 0
    timer.sync()
    started = time.perf_counter()
    try:
        with ArchiveSourceGroups(b.input_paths(args.index, count), 1, args, timer) as source:
            while sources < count:
                timer.sync()
                start = time.perf_counter()
                buffer.make_room()
                try:
                    paths, frames = next(source)
                except StopIteration:
                    break
                if frames is None:
                    with timer.stage("read_decode"):
                        frames = [cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)]
                        if frames[0] is None:
                            raise ValueError("could not decode source image")
                frames, records, observations = pipeline.detect_frames(frames, sources, timer, source_paths=paths)
                buffer.admit(records[0], len(observations))
                for local_source, detection_index, detection in observations:
                    if local_source != 0:
                        raise RuntimeError("buffered detector returned a non-singleton source index")
                    buffer.add_crop(frames[0], sources, detection_index, detection)
                # The loop target otherwise retains the last original polygon
                # (and possibly a large backing array) through later empty frames.
                if observations:
                    del detection
                # No full source frame or detection descriptor survives ingestion.
                del frames, records, observations
                buffer.check_ram()
                timer.sync()
                groups_ms.append((time.perf_counter() - start) * 1000)
                sources += 1
            buffer.flush("eof")
        timer.sync()
        if sources != count or buffer.emitted != sources:
            raise ValueError(f"requested {count} images, read {sources}, and emitted {buffer.emitted}")
        buffer.close()
        elapsed = time.perf_counter() - started
        return {"schedule": "resident-cross-frame-crops", "source_images": sources, "star_crops": buffer.crops, "arms": buffer.arms,
                "empty_source_images": buffer.empty, "elapsed_seconds": elapsed,
                "source_images_per_second": sources / elapsed if elapsed else None,
                "star_crops_per_second": buffer.crops / elapsed if elapsed else None,
                "source_group_latency": b.summary(groups_ms), "group_ms": groups_ms,
                "source_group_latency_scope": "source ingestion including triggered flushes; excludes final EOF flush; not per-image completion latency",
                "stages": {k: b.summary(v) for k, v in timer.values.items()},
                "archive_decode": source.report(), "archive_crop_buffer": buffer.report()}
    finally:
        buffer.close()
