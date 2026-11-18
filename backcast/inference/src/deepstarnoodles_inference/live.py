# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Resident ndarray inference and bounded camera/video capture.

Timestamps measure completion of OpenCV read(), not sensor exposure. Camera
capture keeps one pending frame; video files process every frame sequentially.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from .benchmark import Cascade, Timer, host_peak_bytes, sha256, verify_models
from .hydra import validate_hydra_batch_size
from .records import OUTPUT_SCHEMA_VERSION, finalize_frame, validate_fps


@dataclass(frozen=True)
class LiveConfig:
    device: str = "cuda:0"
    crop_batch_size: int = 1
    allocator_limit_gib: float = 7.0
    ram_limit_gib: float = 24.0
    starseg_half: bool = False
    hydra_fp32: bool = False
    channels_last: bool = False
    hydra_output_mode: str = "full"
    gru_execution: str = "dynamic"
    gru_check_interval: int = 8
    hydra_batch_size: int | None = None
    hydra_execution: str = "eager"
    crop_streams: int = 1
    gru_streams: int = 1
    gru_compile: bool = False

    def __post_init__(self):
        validate_hydra_batch_size(self.hydra_batch_size)
        if isinstance(self.crop_batch_size, bool) or not isinstance(self.crop_batch_size, int) or self.crop_batch_size < 1:
            raise ValueError("crop_batch_size must be a positive integer")
        for name in ("allocator_limit_gib", "ram_limit_gib"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        device = torch.device(self.device)
        if self.starseg_half and device.type != "cuda":
            raise ValueError("starseg_half requires CUDA")
        if self.hydra_output_mode not in ("full", "minimal"):
            raise ValueError("hydra_output_mode must be full or minimal")
        if self.gru_execution not in ("dynamic", "fixed", "graph"):
            raise ValueError("gru_execution must be dynamic, fixed or graph")
        if self.gru_execution == "graph" and device.type != "cuda":
            raise ValueError("gru_execution graph requires CUDA")
        if (isinstance(self.crop_streams, bool) or not isinstance(self.crop_streams, int)
                or not 1 <= self.crop_streams <= 4 or (self.crop_streams > 1 and device.type != "cuda")):
            raise ValueError("crop_streams must be an integer from 1 to 4; above 1 requires CUDA")
        if (isinstance(self.gru_streams, bool) or not isinstance(self.gru_streams, int) or not 1 <= self.gru_streams <= 8
                or (self.gru_streams > 1 and (self.gru_execution != "graph" or self.crop_streams > 1))):
            raise ValueError("gru_streams must be an integer from 1 to 8; above 1 requires graph GRU without crop streams")
        if self.gru_compile is not False and (self.gru_compile is not True or self.gru_execution != "graph"):
            raise ValueError("gru_compile requires gru_execution graph")
        if self.hydra_execution not in ("eager", "graph"):
            raise ValueError("hydra_execution must be eager or graph")
        if self.hydra_execution == "graph" and (device.type != "cuda" or self.hydra_output_mode != "minimal"):
            raise ValueError("hydra_execution graph requires CUDA and minimal Hydra outputs")
        if (isinstance(self.gru_check_interval, bool) or not isinstance(self.gru_check_interval, int)
                or not 1 <= self.gru_check_interval <= 128):
            raise ValueError("gru_check_interval must be an integer between 1 and 128")


class RollingTiming:
    """Exact lifetime mean/extrema and explicitly windowed quantiles."""
    def __init__(self, window=1024):
        self._lock = threading.Lock()
        self.values = deque(maxlen=window)
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = 0.0

    def add(self, value):
        with self._lock:
            self.values.append(value)
            self.count += 1
            self.total += value
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)

    def report(self):
        # A native reader can resume after close() reaches its bounded timeout.
        # Snapshot the deque and all aggregate counters under the same lock.
        with self._lock:
            values = list(self.values)
            count, total, minimum, maximum = self.count, self.total, self.minimum, self.maximum
            capacity = self.values.maxlen
        result = {"samples": count, "quantile_window_samples": len(values),
                  "quantile_window_capacity": capacity}
        if count:
            result.update(mean_ms=total / count, min_ms=minimum, max_ms=maximum,
                          **{f"window_p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)})
        return result


class LivePipeline:
    """Single-stream synchronous BGR-array API; models remain resident.

    Create with from_manifest(). predict() does not retain the caller's pixels
    after returning. Do not mutate the array during prediction. Call reset()
    between unrelated streams. Concurrent predict/reset calls are rejected.
    """
    def __init__(self, cascade, config, *, manifest=None, manifest_sha256=None):
        if cascade.live_cropper is None:
            raise ValueError("LivePipeline requires a temporal Cascade")
        self.cascade, self.config = cascade, config
        self.manifest, self.manifest_sha256 = manifest, manifest_sha256
        self._lock = threading.Lock()
        self._reset_counts()

    @classmethod
    def from_manifest(cls, models, config=None):
        config = LiveConfig() if config is None else config
        manifest, paths = verify_models(models)
        manifest_path = Path(models).expanduser().resolve()
        if manifest_path.is_dir():
            manifest_path /= "manifest.json"
        device = torch.device(config.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(config.allocator_limit_gib * 1024**3 / total, 1.0), device)
        cascade = Cascade(SimpleNamespace(**asdict(config), mode="live-replay", detector_batch_size=1), paths)
        Timer(device, False).sync()
        if host_peak_bytes() > config.ram_limit_gib * 1024**3:
            raise MemoryError("model loading exceeded ram_limit_gib")
        return cls(cascade, config, manifest=manifest, manifest_sha256=sha256(manifest_path))

    def _reset_counts(self):
        self.processed = self.star_crops = self.arms = 0
        self._last_source_index = None
        self._failed = False
        self.timings = {name: RollingTiming() for name in ("queue_wait_ms", "processing_ms", "post_acquisition_result_ms")}

    def reset(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("LivePipeline is already in use")
        try:
            self.cascade.reset()
            self._reset_counts()
        finally:
            self._lock.release()

    def predict(self, frame_bgr, *, source_index=None, acquired_monotonic_ns=None,
                source_timestamp_s=None, acquisition_clock=None, fps=None):
        """Process one frame; the record ends with ``time_s`` (``source_index / fps``, or null without
        ``fps``), ``fps`` and the output ``schema_version``."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("LivePipeline is already in use")
        try:
            if self._failed:
                raise RuntimeError("inference failed previously; reset the stream before continuing")
            if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                    or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or min(frame_bgr.shape[:2]) < 1):
                raise ValueError("frame must be a nonempty HxWx3 BGR uint8 array")
            source_index = (0 if self._last_source_index is None else self._last_source_index + 1) if source_index is None else source_index
            if (isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0
                    or (self._last_source_index is not None and source_index <= self._last_source_index)):
                raise ValueError("source_index must be a nonnegative, strictly increasing integer")
            now = time.monotonic_ns()
            if acquired_monotonic_ns is None:
                acquired_monotonic_ns = now
                acquisition_clock = "caller_received"
            elif acquisition_clock is None:
                acquisition_clock = "caller_supplied_monotonic"
            if (isinstance(acquired_monotonic_ns, bool) or not isinstance(acquired_monotonic_ns, int)
                    or not 0 <= acquired_monotonic_ns <= now):
                raise ValueError("acquired_monotonic_ns must use this process's monotonic clock and not be in the future")
            if source_timestamp_s is not None and (not math.isfinite(source_timestamp_s) or source_timestamp_s < 0):
                raise ValueError("source_timestamp_s must be finite and nonnegative")
            if not isinstance(acquisition_clock, str):
                raise ValueError("acquisition_clock must be a descriptive string")
            fps = validate_fps(fps)
            timer = Timer(self.cascade.device, False)
            timer.sync()
            started = time.monotonic_ns()
            try:
                with torch.inference_mode():
                    records, crops, arms = self.cascade.run_frames([frame_bgr], self.processed, timer)
                timer.sync()
                ended = time.monotonic_ns()
                if host_peak_bytes() > self.config.ram_limit_gib * 1024**3:
                    raise MemoryError("process high-water RSS exceeded ram_limit_gib")
            except BaseException:
                # Detector/association state may have advanced before failure.
                self._failed = True
                raise
            record = records[0]
            record.update(frame=source_index, source_index=source_index, processed_index=self.processed,
                          skipped_source_indices_since_previous=(0 if self._last_source_index is None else source_index - self._last_source_index - 1),
                          source_timestamp_s=source_timestamp_s,
                          acquisition_clock=acquisition_clock,
                          acquired_monotonic_ns=acquired_monotonic_ns,
                          inference_started_monotonic_ns=started, inference_ended_monotonic_ns=ended)
            record["timing"] = {"queue_wait_ms": (started - acquired_monotonic_ns) / 1e6,
                                "processing_ms": (ended - started) / 1e6,
                                "post_acquisition_result_ms": (ended - acquired_monotonic_ns) / 1e6}
            finalize_frame(record, fps)
            for name, value in record["timing"].items():
                self.timings[name].add(value)
            self.processed += 1
            self.star_crops += crops
            self.arms += arms
            self._last_source_index = source_index
            return record
        finally:
            self._lock.release()

    def report(self):
        return {"processed_frames": self.processed, "star_crops": self.star_crops, "arms": self.arms,
                "configuration": asdict(self.config),
                "timing": {key: value.report() for key, value in self.timings.items()},
                "tracker_age_basis": "processed observations, not source frame sequence"}


@dataclass(frozen=True)
class FramePacket:
    frame_bgr: np.ndarray
    source_index: int
    acquired_monotonic_ns: int
    source_timestamp_s: float | None
    read_ms: float


class LatestFrameQueue:
    """One pending frame, independent of the frame currently being inferred."""
    def __init__(self):
        self._condition = threading.Condition()
        self._pending = None
        self._finished = False
        self._error = None
        self.overwritten = self.delivered = self.discarded_on_close = 0

    def put(self, packet):
        with self._condition:
            if self._finished:
                self.discarded_on_close += 1
                return False
            if self._pending is not None:
                self.overwritten += 1
            self._pending = packet
            self._condition.notify_all()
            return True

    def take(self, timeout=None):
        with self._condition:
            ready = self._condition.wait_for(lambda: self._pending is not None or self._finished, timeout)
            if not ready:
                raise TimeoutError("waiting for capture")
            if self._pending is not None:
                packet, self._pending = self._pending, None
                self.delivered += 1
                return packet
            if self._error is not None:
                raise self._error
            raise StopIteration

    def finish(self, error=None):
        with self._condition:
            self._finished = True
            self._error = error
            self._condition.notify_all()

    def cancel(self):
        with self._condition:
            if self._pending is not None:
                self.discarded_on_close += 1
                self._pending = None
            self._finished = True
            self._condition.notify_all()


class CaptureSource:
    """OpenCV source with sequential video or latest-frame camera policy.

    Native backend buffers are outside our queue bound. A blocking native read
    cannot safely be forcibly cancelled; close() reports a reader that has not
    stopped after the bounded join. That daemon reader releases its own capture
    when the backend returns. The consumer never releases a concurrently read
    VideoCapture object.
    """
    def __init__(self, *, camera=None, video=None, queue_policy=None, capture_factory=None):
        if (camera is None) == (video is None):
            raise ValueError("provide exactly one camera index or video path")
        if camera is not None and (isinstance(camera, bool) or not isinstance(camera, int) or camera < 0):
            raise ValueError("camera must be a nonnegative integer")
        self.kind = "camera" if camera is not None else "video"
        self.policy = queue_policy or ("latest" if camera is not None else "sequential")
        if self.policy not in ("latest", "sequential"):
            raise ValueError("queue_policy must be latest or sequential")
        if self.kind == "video" and self.policy != "sequential":
            raise ValueError("video files require sequential capture; unpaced decoding is not live arrival")
        self.source = camera if camera is not None else str(Path(video).expanduser().resolve())
        self._factory = cv2.VideoCapture if capture_factory is None else capture_factory
        self._capture = self._thread = None
        self._stop = threading.Event()
        self._queue = LatestFrameQueue()
        self.captured = self.delivered = 0
        self.eof = False
        self.reader_stopped = True
        self.read_timing = RollingTiming()
        self._closed = False
        self.nominal_fps = None

    def __enter__(self):
        if self._capture is not None or self._closed:
            raise RuntimeError("CaptureSource cannot be reopened")
        capture = self._factory(self.source)
        try:
            if not capture.isOpened():
                raise OSError(f"could not open {self.kind} source")
            fps = capture.get(cv2.CAP_PROP_FPS)
            self.nominal_fps = float(fps) if math.isfinite(fps) and fps > 0 else None
        except BaseException:
            capture.release()
            raise
        self._capture = capture
        if self.policy == "latest":
            self.reader_stopped = False
            self._thread = threading.Thread(target=self._produce, name="deepstarnoodles-capture", daemon=True)
            self._thread.start()
        return self

    def _read(self):
        started = time.monotonic_ns()
        ok, frame = self._capture.read()
        acquired = time.monotonic_ns()
        if not ok:
            if self.kind == "camera" and not self._stop.is_set():
                raise OSError("camera read failed")
            self.eof = self.kind == "video"
            raise StopIteration
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
            raise ValueError("capture returned an invalid BGR uint8 frame")
        timestamp = None
        if self.kind == "video":
            milliseconds = self._capture.get(cv2.CAP_PROP_POS_MSEC)
            if math.isfinite(milliseconds) and milliseconds >= 0:
                timestamp = float(milliseconds) / 1000
        packet = FramePacket(np.array(frame, copy=True, order="C"), self.captured, acquired,
                             timestamp, (acquired - started) / 1e6)
        self.captured += 1
        self.read_timing.add(packet.read_ms)
        return packet

    def _produce(self):
        error = None
        try:
            while not self._stop.is_set():
                if not self._queue.put(self._read()):
                    break
        except StopIteration:
            pass
        except BaseException as exc:
            error = exc
        finally:
            try:
                self._capture.release()
            except BaseException as exc:
                if error is None:
                    error = exc
            finally:
                self.reader_stopped = True
                self._queue.finish(error)

    def read(self, timeout=None):
        if self._capture is None or self._closed:
            raise RuntimeError("capture is not open")
        if self.policy == "latest":
            packet = self._queue.take(timeout)
        else:
            packet = self._read()
        self.delivered += 1
        return packet

    def close(self, timeout=1.0):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._queue.cancel()
        if self._thread is not None:
            self._thread.join(timeout)
            self.reader_stopped = not self._thread.is_alive()
        elif self._capture is not None:
            self._capture.release()

    def __exit__(self, *_exc):
        self.close()

    def report(self):
        return {"kind": self.kind, "queue_policy": self.policy, "pending_frame_capacity": 1 if self.policy == "latest" else 0,
                "captured_frames": self.captured, "delivered_frames": self.delivered,
                "overwritten_pending_frames": self._queue.overwritten,
                "discarded_on_close": self._queue.discarded_on_close,
                "reader_stopped": self.reader_stopped, "statistics_final": self._closed and self.reader_stopped,
                "capture_error": (None if self._queue._error is None else
                                  f"{type(self._queue._error).__name__}: {self._queue._error}"),
                "video_end_or_decode_failure": self.eof,
                "nominal_fps": self.nominal_fps, "read_timing": self.read_timing.report()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for streaming JSONL and final report")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="OpenCV camera index; defaults to one pending latest frame")
    source.add_argument("--video", type=Path, help="Video file; every decoded frame is processed sequentially")
    parser.add_argument("--queue-policy", choices=("latest", "sequential"))
    parser.add_argument("--max-frames", type=int, help="Stop after N processed frames; default is unlimited")
    parser.add_argument("--max-seconds", type=float, help="Capture-run deadline, checked between reads/inference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-batch-size", type=int, default=1)
    parser.add_argument("--hydra-batch-size", type=int, default=None,
                        help="Experimental: cap Hydra forwards independently of the GRU crop batch")
    parser.add_argument("--allocator-limit-gib", type=float, default=7.0)
    parser.add_argument("--ram-limit-gib", type=float, default=24.0)
    parser.add_argument("--starseg-half", action="store_true")
    parser.add_argument("--hydra-fp32", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--hydra-output-mode", choices=("full", "minimal"), default="full",
                        help="Opt-in minimal mode retains raw GRU fields and only two maps for live arm tracking")
    parser.add_argument("--gru-execution", choices=("dynamic", "fixed", "graph"), default="dynamic",
                        help="Experimental fixed mode keeps all seed slots; graph replays fixed-slot steps from "
                             "CUDA graphs with device peak candidates; dynamic remains the reference")
    parser.add_argument("--gru-check-interval", type=int, default=8,
                        help="Fixed/graph modes check aggregate termination every N steps (1-128); ignored by dynamic mode")
    parser.add_argument("--hydra-execution", choices=("eager", "graph"), default="eager",
                        help="Experimental: replay minimal-mode Hydra forwards from CUDA graphs")
    parser.add_argument("--gru-streams", type=int, default=1,
                        help="Experimental: roll out a frame's crops' GRU graphs concurrently on CUDA streams (1-8)")
    parser.add_argument("--crop-streams", type=int, default=1,
                        help="Experimental: run up to N crop groups of one frame concurrently on CUDA streams (1-4)")
    parser.add_argument("--gru-compile", action="store_true",
                        help="Experimental: torch.compile the GRU step before graph capture (requires --gru-execution "
                             "graph); not byte-identical to the eager step")
    args = parser.parse_args(argv)
    try:
        config = LiveConfig(**{name: getattr(args, name) for name in LiveConfig.__dataclass_fields__})
        if args.camera is not None and args.camera < 0:
            raise ValueError("camera index must be nonnegative")
        if args.video is not None and not args.video.is_file():
            raise ValueError("video must be an existing local file")
        if args.video is not None and args.queue_policy == "latest":
            raise ValueError("video files require sequential capture")
        if args.max_frames is not None and args.max_frames < 1:
            raise ValueError("max-frames must be positive")
        if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
            raise ValueError("max-seconds must be finite and positive")
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return args, config


def main(argv=None):
    started = time.perf_counter()
    args, config = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    pipeline = source = None
    report = {"schema_version": 1, "output_schema_version": OUTPUT_SCHEMA_VERSION, "mode": "live",
              "status": "running", "configuration": asdict(config),
              "source": {"camera_index": args.camera, "video_name": None if args.video is None else args.video.name},
              "inventory": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                            "device": config.device, "python": sys.version,
                            "ultralytics": importlib.metadata.version("ultralytics"),
                            "numpy": np.__version__, "opencv": cv2.__version__,
                            "torch_cpu_threads": torch.get_num_threads()},
              "limitations": ["Acquisition time is OpenCV read completion, not sensor exposure; backend buffering is unmeasured.",
                              "Video POS_MSEC is a best-effort backend timestamp, not a wall clock; read failure cannot distinguish EOF from corruption.",
                              "Queue age and inference end exclude subsequent JSONL write/flush; durability fsync is excluded.",
                              "First inference is included; there is no warmup or camera FPS/RTX 5060 certification.",
                              "Native open/read may block; deadlines are checked between operations and shutdown reports any stuck reader.",
                              "Allocator guard covers PyTorch only; RSS is process high-water including model loading."]}
    old_sigterm = None
    if threading.current_thread() is threading.main_thread():
        old_sigterm = signal.getsignal(signal.SIGTERM)
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
    write_timing = RollingTiming()
    try:
        load_started = time.perf_counter()
        pipeline = LivePipeline.from_manifest(args.models, config)
        report.update(model_load_seconds=time.perf_counter() - load_started, models=pipeline.manifest,
                      models_manifest_sha256=pipeline.manifest_sha256,
                      hydra_state_dict_key=getattr(pipeline.cascade.hydra, "selected_state_dict_key", None))
        report["crop_config"] = asdict(pipeline.cascade.crop_config)
        report["trace_params"] = asdict(pipeline.cascade.trace_params)
        tracer = pipeline.cascade.tracer
        if hasattr(tracer, "dense_mean") and hasattr(tracer, "dense_std"):
            normalization = {name: getattr(tracer, name).detach().cpu().numpy().astype("<f4")
                             for name in ("dense_mean", "dense_std")}
            report["normalization"] = {
                "source": "checkpoint_buffers", "digest_encoding": "dense_mean then dense_std, little-endian float32",
                "sha256": hashlib.sha256(b"".join(normalization[name].tobytes() for name in ("dense_mean", "dense_std"))).hexdigest(),
                **{name: value.reshape(-1).tolist() for name, value in normalization.items()},
            }
        if torch.device(config.device).type == "cuda":
            report["inventory"].update(gpu_name=torch.cuda.get_device_name(config.device),
                                      gpu_total_memory_bytes=torch.cuda.get_device_properties(config.device).total_memory,
                                      gpu_capability=list(torch.cuda.get_device_capability(config.device)))
        source = CaptureSource(camera=args.camera, video=args.video, queue_policy=args.queue_policy)
        capture_started = time.perf_counter()
        with source, (args.output / "predictions.jsonl").open("x") as sink:
            # A video's nominal rate times its frames; a camera's nominal rate is not a clock.
            frame_rate = source.nominal_fps if source.kind == "video" else None
            while True:
                if args.max_frames is not None and pipeline.processed >= args.max_frames:
                    report["stop_reason"] = "max_frames"
                    break
                if args.max_seconds is not None and time.perf_counter() - capture_started >= args.max_seconds:
                    report["stop_reason"] = "max_seconds"
                    break
                try:
                    packet = source.read(timeout=0.25)
                except TimeoutError:
                    continue
                except StopIteration:
                    report["stop_reason"] = "video_end_or_decode_failure"
                    break
                record = pipeline.predict(packet.frame_bgr, source_index=packet.source_index,
                                          acquired_monotonic_ns=packet.acquired_monotonic_ns,
                                          source_timestamp_s=packet.source_timestamp_s,
                                          acquisition_clock="opencv_read_completed", fps=frame_rate)
                record["capture_read_ms"] = packet.read_ms
                record["queue_overwritten_total_at_result"] = source._queue.overwritten
                finalize_frame(record, frame_rate)
                write_started = time.perf_counter()
                sink.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                sink.flush()
                write_timing.add((time.perf_counter() - write_started) * 1000)
        report["capture_run_seconds"] = time.perf_counter() - capture_started
        report["status"] = "complete" if source.reader_stopped else "incomplete_shutdown"
        return 0 if source.reader_stopped else 2
    except KeyboardInterrupt:
        report.update(status="interrupted", stop_reason="signal_or_keyboard_interrupt")
        return 130
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
        if source is not None:
            source.close()
            report["capture"] = source.report()
        if pipeline is not None:
            report["measurement"] = pipeline.report()
        report["serialize_write"] = write_timing.report()
        report["total_job_seconds"] = time.perf_counter() - started
        report["host_peak_rss_bytes"] = host_peak_bytes()
        if torch.device(config.device).type == "cuda" and torch.cuda.is_initialized():
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(config.device)
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(config.device)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
