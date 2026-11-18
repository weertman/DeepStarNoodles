# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Turn inference output into the DeepStarNoodles paper's polyline representations (torch-free).

:func:`load_inference` reads ``archive`` / ``live`` / ``serve-videos`` / ``staged-archive`` records
(JSON, JSON Lines, optionally ``.gz`` / ``.bz2`` / ``.xz`` / ``.zst``, or a stream) into one
:class:`PolylineContainer` per star. Schema-3 records (0.5.0.dev11 and later) carry the runtime body
origin, support area and centroid, and ``time_s``, so every preset of the vendored
``noodle_polylines`` can be computed; legacy records give the body-free outputs only. The
``representations`` command writes containers, per-preset outputs and optional CSV tables.

The vendored library is used unchanged; this module adds what inference output needs on top of
``noodle_polylines.adapters.from_inference_json``: the schema-3 fields, record and header ``fps``,
gap rows for tracked stars, the per-arm pass-through fields and archive records without a
``star_track_id`` (``untracked=``).
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import contextlib
import csv
import fnmatch
import glob
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator
import warnings

import numpy as np

from ._vendor.noodle_polylines import PRESETS, compute_all
from ._vendor.noodle_polylines import __version__ as POLYLINES_VERSION
from ._vendor.noodle_polylines.adapters import (_drop_unused_arrays, _stack_identity, _stack_polylines, _stop_codes,
                                                _zstd_decompress)
from ._vendor.noodle_polylines.container import ContainerError, PolylineContainer, _offsets
from ._vendor.noodle_polylines.pipeline import save_outputs
from .input_tree import expected_records as manifest_records, record_files

VENDORED_COMMIT = "f749a4906de639b9a36c08ff0def015dbe558ff1"
UNTRACKED_MODES = ("error", "single", "detections")
BODY_FIELDS = ("body_origin_model", "body_origin_code", "support_area_model_px", "support_centroid_model")
PASS_THROUGH = {"tip_score": np.float32, "base_score": np.float32, "state_confidence": np.float32,
                "steps": np.int16}
COMPRESSED = (".gz", ".bz2", ".xz", ".zst")
WINDOWED_R0 = ("pre_onset", "outside_windows", "baseline")

# Which axis each compute_all output runs along (derived from compute_all on the flipping example and
# re-derived by a test): per frame (F, ...), per arm (A, ...), per bout window (W, F), per point (P, 2),
# or a scalar. Unknown keys are left out of the CSV tables and listed in the manifest.
AXIS = {
    "n_arms": "frame", "radius_model": "frame", "radius_source": "frame", "tip_isotropy": "frame",
    "mean_norm_length": "frame", "fold_asymmetry": "frame", "arm_length_spread": "frame",
    "arm_curvature": "frame", "arm_deformation": "frame", "log_deformation": "frame", "tip_motion": "frame",
    "arc_length": "arm", "curvature": "arm", "tip_source": "arm", "norm_length": "arm",
    "arm_norm_length": "arm", "arm_curvature_body_radius": "arm", "tip_angle_deg": "arm",
    "tip_radius_px": "arm", "track_id": "arm", "resample_base": "arm",
    "arm_deformation_window": "window", "log_deformation_window": "window",
    "body_path": "point",
    "preset": "scalar", "r0": "scalar", "area0": "scalar", "r0_frames": "scalar",
    "track_id_source": "scalar", "tip_isotropy_window": "scalar", "dropped": "scalar",
}


# ------------------------------------------------------------------------------------------ reading
def inner_suffix(path: Path) -> str:
    """``.jsonl`` for ``x.jsonl`` and ``x.jsonl.zst``."""
    suffix = path.suffix.lower()
    return Path(path.stem).suffix.lower() if suffix in COMPRESSED else suffix


class _ZstdFrames(io.RawIOBase):
    """Every frame of a zstd stream, decompressed; a stream that ends inside a frame raises :class:`EOFError`.

    python-zstandard's ``stream_reader`` returns the data of a truncated frame without an error, so a cut-off
    ``.jsonl.zst`` would read as a shorter, complete video. This reader decompresses frame by frame with
    ``decompressobj`` and checks that the last frame ended.
    """

    def __init__(self, raw, decompressor, name):
        super().__init__()
        self._raw, self._decompressor, self._name = raw, decompressor, name
        self._frame = None          # the open frame's decompressobj
        self._pending = b""         # compressed bytes after the last finished frame
        self._out = memoryview(b"")
        self._eof = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not len(self._out) and not self._eof:
            chunk, self._pending = self._pending or self._raw.read(1 << 20), b""
            if not chunk:
                if self._frame is not None:
                    raise EOFError(f"{self._name}: the zstd stream ends inside a frame; the file is truncated")
                self._eof = True
                break
            if self._frame is None:
                self._frame = self._decompressor.decompressobj()
            self._out = memoryview(self._frame.decompress(chunk))
            if self._frame.eof:
                self._pending, self._frame = self._frame.unused_data, None
        n = min(len(buffer), len(self._out))
        buffer[:n] = self._out[:n]
        self._out = self._out[n:]
        return n

    def close(self) -> None:
        if not self.closed:
            self._raw.close()
        super().close()


def _open_zstd(path: Path):
    try:
        from compression import zstd   # Python >= 3.14; raises EOFError on a truncated frame
        return zstd.open(path, "rt", encoding="utf-8")
    except ImportError:
        pass
    try:
        import zstandard
    except ImportError:
        _zstd_decompress(b"", path)   # raises the vendored library's ImportError with install advice
        raise
    reader = _ZstdFrames(path.open("rb"), zstandard.ZstdDecompressor(), str(path))
    return io.TextIOWrapper(io.BufferedReader(reader), encoding="utf-8")


@contextlib.contextmanager
def open_text(source) -> Iterator[io.TextIOBase]:
    """A text stream over a path (decompressing by suffix), ``-`` (stdin) or an open text stream."""
    if hasattr(source, "read"):
        yield source
        return
    if str(source) == "-":
        yield sys.stdin
        return
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".gz":
        import gzip
        stream = gzip.open(path, "rt", encoding="utf-8")
    elif suffix == ".bz2":
        import bz2
        stream = bz2.open(path, "rt", encoding="utf-8")
    elif suffix == ".xz":
        import lzma
        stream = lzma.open(path, "rt", encoding="utf-8")
    elif suffix == ".zst":
        stream = _open_zstd(path)
    else:
        stream = path.open(encoding="utf-8")
    with stream:
        yield stream


def iter_records(source, *, skip_unused: bool = True) -> Iterator[tuple[dict, Any]]:
    """Yield ``(frame record, header fps)`` from a JSON payload, a JSON array or JSON Lines.

    JSON Lines are parsed one line at a time (memory is one record); a line holding a whole payload
    (``{"frames": [...]}``) yields its frames. A ``.json`` file, or a stream whose first line is not a
    JSON document, is parsed whole. ``skip_unused`` cuts the unused source-pixel arrays before parsing
    (``noodle_polylines.adapters._drop_unused_arrays``).
    """
    prepare = _drop_unused_arrays if skip_unused else (lambda text: text)
    whole = not hasattr(source, "read") and str(source) != "-" and inner_suffix(Path(source)) == ".json"
    with open_text(source) as stream:
        if whole:
            yield from _payload_records(json.loads(prepare(stream.read())))
            return
        first = None
        last_frame = None
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            where = "<stream>" if hasattr(source, "read") else str(source)
            if first is None:
                first = line
                try:
                    record = json.loads(prepare(line))
                except json.JSONDecodeError as exc:
                    # A pretty-printed JSON document parses whole; anything else is a broken first record.
                    try:
                        payload = json.loads(prepare(line + stream.read()))
                    except json.JSONDecodeError:
                        raise ContainerError(f"{where}: line {number} (before any record) is not a JSON record "
                                             f"({exc.msg}); the file is truncated or corrupt") from None
                    yield from _payload_records(payload)
                    return
            else:
                try:
                    record = json.loads(prepare(line))
                except json.JSONDecodeError as exc:
                    raise ContainerError(f"{where}: line {number} (after frame {last_frame}) is not a JSON record "
                                         f"({exc.msg}); the file is truncated or corrupt") from None
            if not isinstance(record, (dict, list)):
                raise ContainerError(f"{where}: line {number} is not a frame record (a JSON "
                                     f"{type(record).__name__})")
            if isinstance(record, dict):
                last_frame = record.get("frame", last_frame)
            if isinstance(record, dict) and "frames" in record:
                yield from _payload_records(record)
            elif isinstance(record, list):
                yield from _payload_records(record)
            else:
                yield record, None


def _payload_records(payload):
    if isinstance(payload, dict) and "frames" not in payload:
        yield payload, None          # one frame record
    elif isinstance(payload, dict):
        header_fps = payload.get("fps")
        for record in payload["frames"]:
            yield record, header_fps
    else:
        for record in payload:
            yield record, None


# ------------------------------------------------------------------------------------------ parsing
class _Star:
    """Rows and arms of one container while a file is read."""

    def __init__(self):
        self.position: list[int] = []      # index of the row's record in the file
        self.frame: list[int] = []
        self.detection: list[int | None] = []
        self.crop: list[list[float]] = []
        self.has_star: list[bool] = []
        self.n: list[int] = []
        self.body: dict[str, list] = {key: [] for key in BODY_FIELDS}
        self.body_present = dict.fromkeys(BODY_FIELDS, 0)
        self.star_rows = 0
        self.polys: list = []
        self.identity: list = []
        self.identity_missing = False
        self.track_id: list[int] = []
        self.passthrough: dict[str, list] = {key: [] for key in (*PASS_THROUGH, "stop_reason")}

    def add(self, position, frame, crop, arms, star, detection=None):
        self.position.append(position)
        self.frame.append(frame)
        self.detection.append(detection)
        self.has_star.append(crop is not None)
        self.crop.append([math.nan] * 3 if crop is None else
                         [float(crop["left"]), float(crop["top"]), float(crop["side"])])
        self.n.append(len(arms))
        if crop is not None:
            self.star_rows += 1
            for key in BODY_FIELDS:
                value = (star or {}).get(key, _ABSENT)
                if value is not _ABSENT:
                    self.body_present[key] += 1
                self.body[key].append(value)
        else:
            for key in BODY_FIELDS:
                self.body[key].append(_ABSENT)
        for arm in arms:
            if "polyline_crop" not in arm:
                raise ValueError(f"frame {frame}: arm record lacks polyline_crop")
            self.polys.append(arm["polyline_crop"])
            identity = arm.get("identity")
            if identity is None:
                self.identity_missing = True
                identity = _NAN8
            self.identity.append(identity)
            track = arm.get("track_id")
            self.track_id.append(-1 if track is None else int(track))
            for key, values in self.passthrough.items():
                values.append(arm.get(key, _ABSENT))


_ABSENT = object()
_NAN8 = [math.nan] * 8


def _star_entries(record, frame, untracked, name):
    """``[(key, crop, arms, star dict, detection index)]`` of one frame record."""
    if "stars" not in record:
        detected = record.get("detected", True) or record.get("crop")
        if not detected:
            return [(None, None, [], None, None)]
        return [(None, record.get("crop"), record.get("arms", []), record, None)]
    stars = record["stars"]
    if not stars:
        return []
    ids = [s.get("star_track_id") for s in stars]
    untracked_stars = [i for i in ids if i is None]
    if untracked_stars and untracked == "error":
        raise ContainerError(f"{name}: frame {frame}: multi-star record without a star_track_id (archive-mode "
                             "output); read it with untracked='single' (at most one star per frame: "
                             "--untracked single) or untracked='detections' (--untracked detections)")
    if untracked_stars and len(untracked_stars) != len(ids):
        raise ContainerError(f"{name}: frame {frame}: tracked and untracked stars in one file")
    if not untracked_stars:
        return [(int(s["star_track_id"]), s.get("crop"), s.get("arms", []), s, None) for s in stars]
    if untracked == "single":
        if len(stars) > 1:
            raise ContainerError(f"{name}: frame {frame}: {len(stars)} untracked stars; untracked='single' "
                                 "needs at most one star per frame. Give the stars IDs with "
                                 "`deepstarnoodles-inference track-videos` and read its output without --untracked, "
                                 "or read every detection on its own with untracked='detections'")
        s = stars[0]
        return [("single", s.get("crop"), s.get("arms", []), s, None)]
    return [("untracked", s.get("crop"), s.get("arms", []), s, s.get("detection_index", j))
            for j, s in enumerate(stars)]


def load_inference(source, *, fps: float | None = None, untracked: str = "error", skip_unused: bool = True,
                   fill_gaps: bool = True, name: str | None = None,
                   expected_records: tuple[int, Any] | None = None) -> dict:
    """Split inference output into one :class:`PolylineContainer` per star.

    Keys: the ``star_track_id`` of tracked multi-star records (``live``, ``live-replay``), ``None`` for
    single-star records, ``"single"`` or ``"untracked"`` for archive records without a
    ``star_track_id`` read with ``untracked="single"`` (at most one star per frame; frames without it
    become ``has_star=False`` rows) or ``untracked="detections"`` (one row per detection, ``frame_index``
    renumbered 0..N-1 with the video frame in ``source_frame_index``; no identity or track IDs, so no
    re-link). The default ``"error"`` refuses such records, as upstream does.

    Schema-3 star fields (body origin and code, support area and centroid) and record ``time_s`` /
    ``fps`` fill the container fields of the same names. ``fps`` is for files that carry none; one that
    disagrees with the records' raises :class:`ContainerError`. ``fill_gaps`` adds a ``has_star=False``
    row for every frame between a tracked star's first and last appearance without it, as the paper
    re-trace's no-star rows do. Rows go by frame number at the file's frame spacing, so frames with no
    record at all are filled too (``meta['missing_records']``, with a warning); without a frame rate their
    times are unknown and they are left out, with a warning. A file without records, a line that is not
    JSON and a truncated compressed file raise, as does a record count other than ``expected_records``
    (``(count, where)``, from the fleet's ``_manifest.jsonl``: a cut at a record boundary reads as a shorter
    video). Pass-through arm fields (``tip_score``, ``base_score``, ``state_confidence``, ``steps``,
    ``stop_reason``) are kept when every arm has them; inference-only
    stop reasons map to -1. Missing fields are listed in ``meta['missing_fields']``; the body origin is
    never approximated.
    """
    if untracked not in UNTRACKED_MODES:
        raise ValueError(f"untracked must be one of {UNTRACKED_MODES}")
    name = name or (str(source) if not hasattr(source, "read") else "<stream>")
    fps = None if fps is None else _check_fps(fps, name)
    stars: dict = {}
    frames: list[int] = []
    times: list = []
    fps_values: set[float] = set()
    schemas: set = set()
    kinds: set[str] = set()
    for position, (record, header_fps) in enumerate(iter_records(source, skip_unused=skip_unused)):
        value = record.get("frame") if isinstance(record, dict) else None
        if isinstance(value, bool) or not (isinstance(value, int) or isinstance(value, float) and value.is_integer()):
            raise ContainerError(f"{name}: record {position + 1} is not a frame record (it needs an integer "
                                 f"'frame'{f', after frame {frames[-1]}' if frames else ''})")
        frame = int(value)
        if frames and frame <= frames[-1]:
            raise ContainerError(f"{name}: frames are not in strictly increasing order (frame {frame} after "
                                 f"{frames[-1]})")
        frames.append(frame)
        times.append(record.get("time_s", _ABSENT))
        schemas.add(record.get("schema_version", "legacy"))
        for value in (record.get("fps"), header_fps):
            if value is not None:
                fps_values.add(_check_fps(value, name))
        entries = _star_entries(record, frame, untracked, name)
        kinds.add("multi-star" if "stars" in record else "single-star")
        if len(kinds & {"multi-star", "single-star"}) > 1:
            raise ContainerError(f"{name}: frame {frame}: single-star records (no 'stars' list) and multi-star "
                                 "records in one file")
        for key, crop, arms, star, detection in entries:
            kinds.add("tracked" if isinstance(key, int) else "untracked" if key is not None else "single-star")
            if {"tracked", "untracked"} <= kinds:
                raise ContainerError(f"{name}: frame {frame}: tracked and untracked stars in one file")
            stars.setdefault(key, _Star()).add(position, frame, crop, arms, star, detection)
    if not frames:
        raise ContainerError(f"{name}: holds no records")
    if expected_records is not None and expected_records[0] != len(frames):
        raise ContainerError(f"{name}: holds {len(frames)} records, but {expected_records[1]} lists "
                             f"{expected_records[0]} for it; the file is truncated or incomplete (move the manifest "
                             "aside if the video was annotated again elsewhere)")
    if len(fps_values) > 1:
        raise ContainerError(f"{name}: records disagree on fps: {sorted(fps_values)}")
    record_fps = next(iter(fps_values), None)
    if fps is not None and record_fps is not None and fps != record_fps:
        raise ContainerError(f"{name}: fps={fps} disagrees with the records' fps {record_fps}")
    fps = fps if fps is not None else record_fps
    record_schema = sorted(schemas, key=str)
    out = {}
    for key, star in stars.items():
        out[key] = _build(key, star, frames, times, fps, fill_gaps and isinstance(key, int), record_schema, name)
    return out


def _check_fps(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ContainerError(f"{name}: fps must be a finite positive number, got {value!r}")
    return float(value)


def _frame_step(frames: list[int]) -> int:
    """The video's frame spacing: its most common frame difference (the smallest of equally common ones; 1 for one
    record). One record off the grid does not turn a file of every second frame into one of every frame."""
    if len(frames) < 2:
        return 1
    counts = Counter(b - a for a, b in zip(frames, frames[1:]))
    return min(counts, key=lambda step: (-counts[step], step))


def _frame_span(frames: list[int], first: int, last: int) -> list[int]:
    """Every record's frame from ``first`` to ``last``, plus the frames a gap wider than the frame spacing lacks."""
    step = _frame_step(frames)
    inside = [f for f in frames if first <= f <= last]
    span = []
    for a, b in zip(inside, inside[1:]):
        span.append(a)
        span.extend(range(a + step, b, step))
    return span + inside[-1:]


def _build(key, s: _Star, frames, times, fps, fill_gaps, record_schema, name) -> PolylineContainer:
    rows = len(s.position)
    order = list(range(rows))
    row_frames = list(s.frame)
    row_times = [times[p] for p in s.position]
    filled = 0
    no_record = 0
    meta_warnings = []
    if (fill_gaps and rows) or key == "single":
        # One row per frame between the star's first and last appearance (every frame of the file for an untracked
        # single star), by frame number, at the file's frame spacing: records without the star, and frames with no
        # record at all, become no-star rows.
        present = dict(zip(s.frame, range(rows)))
        record_of = {frame: p for p, frame in enumerate(frames)}
        first, last = (frames[0], frames[-1]) if key == "single" else (s.frame[0], s.frame[-1])
        span = _frame_span(frames, first, last)
        absent = [f for f in span if f not in record_of]
        if absent and fps is None:
            meta_warnings.append(f"{len(absent)} frames between {first} and {last} have no record and the frame "
                                 "rate is unknown, so they are not filled")
            span = [f for f in span if f in record_of]
        elif absent:
            meta_warnings.append(f"{len(absent)} frames between {first} and {last} have no record (first: frame "
                                 f"{absent[0]}); they are filled as no-star rows")
        order, row_frames, row_times = [], [], []
        for f in span:
            order.append(present.get(f, -1))
            row_frames.append(f)
            p = record_of.get(f)
            row_times.append(times[p] if p is not None else f / fps)
            no_record += p is None
        filled = sum(1 for i in order if i < 0)
    detections = key == "untracked"

    def column(values, fill):
        return [values[i] if i >= 0 else fill for i in order]

    video_frame = np.asarray(row_frames, np.int64)
    counts = np.asarray(column(s.n, 0), np.int64)
    has_star = np.asarray(column(s.has_star, False), bool)
    crop = np.asarray(column(s.crop, [math.nan] * 3), np.float64).reshape(-1, 3)
    missing = []

    record_times = row_times
    if record_times and all(t is not _ABSENT and t is not None for t in record_times):
        time_s = np.asarray(record_times, np.float64)
    elif detections and fps is not None:
        time_s = video_frame / fps
    else:
        time_s = None
    if time_s is None and fps is None:
        missing.append("time_s")

    body = {}
    schema3 = all(t is not _ABSENT for t in record_times) if record_times else False
    for field in BODY_FIELDS:
        present = s.body_present[field]
        if s.star_rows and present == s.star_rows or not s.star_rows and schema3:
            values = column(s.body[field], _ABSENT)
            body[field] = _body_array(field, values)
        else:
            body[field] = None
            missing.append(field)
            if present:
                meta_warnings.append(f"{field} present on {present} of {s.star_rows} star rows; the field is left out")
    if s.identity_missing:
        missing.append("identity")

    points, point_counts = _stack_polylines(s.polys)
    identity = _stack_identity(s.identity)      # also when missing: a malformed identity still raises
    arms = len(s.polys)
    passthrough = {}
    for field, values in s.passthrough.items():
        if not arms or any(v is _ABSENT for v in values):
            passthrough[field] = None
        elif field == "stop_reason":
            passthrough[field] = _stop_codes(values).reshape(arms)
        else:
            passthrough[field] = np.asarray(values, PASS_THROUGH[field])
    if detections:
        identity_out, track_id = None, None
    else:
        identity_out = None if s.identity_missing else identity
        track_id = np.asarray(s.track_id, np.int64)
    meta = {"model_size": 512, "source_format": "infer_video",
            "star_track_id": key if isinstance(key, int) else None,
            "track_id_scope": "star_track_id" if isinstance(key, int) or key is None else None,
            "record_schema": record_schema, "missing_fields": missing,
            "untracked": {"single": "single", "untracked": "detections"}.get(key, False),
            "source_name": Path(name).name if name else None}
    if fps is not None:
        meta["fps"] = fps
    if body["body_origin_model"] is not None:
        meta["body_origin"] = "runtime"
    if fill_gaps:
        meta["gap_rows"] = filled
    if no_record:
        meta["missing_records"] = no_record
    if detections:
        meta["detection_index"] = [int(d) for d in s.detection]
    if meta_warnings:
        meta["warnings"] = meta_warnings
        for text in meta_warnings:
            warnings.warn(f"{name}: star {key}: {text}", stacklevel=3)
    container = PolylineContainer(
        points=points, arm_point_offsets=_offsets(point_counts), frame_arm_offsets=_offsets(counts),
        frame_index=np.arange(len(video_frame), dtype=np.int64) if detections else video_frame,
        time_s=time_s, crop=crop, **body, has_star=has_star,
        source_frame_index=video_frame if detections else None,
        identity=identity_out, track_id=track_id, **passthrough, meta=meta)
    return container.validate()


def _body_array(field, values):
    if field == "body_origin_model":
        return np.asarray([[math.nan, math.nan] if v is _ABSENT else v for v in values], np.float64).reshape(-1, 2)
    if field == "support_centroid_model":
        return np.asarray([[math.nan, math.nan] if v is _ABSENT or v is None else v for v in values],
                          np.float64).reshape(-1, 2)
    if field == "body_origin_code":
        return np.asarray([-1 if v is _ABSENT else v for v in values], np.int8)
    return np.asarray([0 if v is _ABSENT else v for v in values], np.int32)


# ------------------------------------------------------------------------------------------ CLI
INPUT_PATTERNS = ("predictions.jsonl", "predictions.jsonl.gz", "predictions.jsonl.bz2", "predictions.jsonl.xz",
                  "predictions.jsonl.zst", "*.jsonl", "*.jsonl.gz", "*.jsonl.bz2", "*.jsonl.xz", "*.jsonl.zst")
EXCLUDED_NAMES = ("report.json", "service.json")


def _is_input(path: Path, *, underscore: bool = False) -> bool:
    """A record file found in a directory: not a partial file, a report or (unless ``underscore``) a ``_`` file such
    as the fleet's ``_manifest.jsonl``."""
    n = path.name
    if n.endswith(".part") or n in EXCLUDED_NAMES or n.endswith(".meta.json") or (n.startswith("_") and not underscore):
        return False
    if n.startswith("worker.") and n.endswith(".json"):
        return False
    return inner_suffix(path) in (".jsonl", ".json") and path.is_file()


def _base(name: str) -> str:
    """``x`` for ``x.jsonl``, ``x.json`` and their compressed forms (``C0001.mp4.jsonl.zst`` -> ``C0001.mp4``)."""
    if Path(name).suffix.lower() in COMPRESSED:
        name = Path(name).stem
    return Path(name).stem if Path(name).suffix.lower() in (".jsonl", ".json") else name


def _found_in(directory: Path, notes: list[str] | None = None) -> list[tuple[str, str]]:
    """Record files under ``directory`` (``input_tree``: directory symlinks followed, another track-videos output
    and ``_`` files skipped, with ``notes``), named by their relative paths."""
    def accept(path: Path) -> bool:
        return (any(fnmatch.fnmatch(path.name, pattern) for pattern in INPUT_PATTERNS)
                and _is_input(path, underscore=True))

    paths, skipped = record_files(directory, accept)
    if notes is not None:
        notes.extend(skipped)
    return [(str(path), _relative_name(path, directory)) for path in paths]


def _relative_name(path: Path, root: Path) -> str:
    """Name of a file found under ``root``: its relative path without the record suffixes, ``/``-separated."""
    rel = path.relative_to(root)
    base = _base(rel.name)
    if base == "predictions":
        return rel.parent.as_posix() if rel.parent != Path(".") else root.resolve().name
    return (rel.parent / base).as_posix()


def collect_named_inputs(items: list[str], notes: list[str] | None = None) -> list[tuple[str, str]]:
    """``(path, name)`` of every input; sorted by path, de-duplicated.

    Files found in a directory are named by their path relative to it (``camA/C0001.mp4``), so files with the same
    name in different subdirectories (cameras) stay apart; files given directly, or matched by a glob, by their stem.
    ``predictions.jsonl`` takes its directory's name. Refuses one video in two formats (``x.jsonl`` and
    ``x.jsonl.zst``) and two inputs with the same name. What directory searches skipped is added to ``notes``.
    """
    found: dict[str, str] = {}
    stdin = False
    for item in items:
        if item == "-":
            stdin = True
            continue
        path = Path(item)
        if path.is_dir():
            found.update((p, n) for p, n in _found_in(path, notes) if p not in found)
        elif path.exists():
            found.setdefault(str(path), _relative_name(path, path.parent))
        else:
            matches = [Path(m) for m in glob.glob(item, recursive=True)]
            if not matches:
                raise FileNotFoundError(f"no input matches {item}")
            for match in sorted(matches):
                if match.is_dir():
                    found.update((p, n) for p, n in _found_in(match, notes) if p not in found)
                elif _is_input(match):
                    found.setdefault(str(match), _relative_name(match, match.parent))
    if stdin and found:
        raise ValueError("'-' (stdin) cannot be combined with other inputs")
    if stdin:
        return [("-", "-")]
    real: dict[str, str] = {}
    for path in sorted(found):
        if os.path.realpath(path) in real:     # the same file twice (a directory and a file inside it, a symlink)
            del found[path]
        else:
            real[os.path.realpath(path)] = path
    videos: dict[Path, str] = {}
    names: dict[str, str] = {}
    for path in sorted(found):
        video = Path(path).resolve().parent / _base(Path(path).name)
        if video in videos:
            raise ValueError(f"{videos[video]} and {path} are the same video in two formats; keep one of them")
        videos[video] = path
        name = found[path]
        if name in names:
            raise ValueError(f"{names[name]} and {path} would both be named {name!r}; pass the directory that holds "
                             "both, so inputs are named by their relative paths, or process them separately")
        names[name] = path
    return [(path, found[path]) for path in sorted(found)]


def collect_inputs(items: list[str]) -> list[str]:
    """Files, ``-``, directories (searched recursively for JSON Lines) and globs; sorted, de-duplicated."""
    return [path for path, _ in collect_named_inputs(items)]


def input_names(inputs: list[str], stdin_name: str | None = None) -> list[str]:
    """Stem without compound suffixes; ``predictions.jsonl`` takes its directory's name; ``__2`` on clashes.

    The command names inputs with :func:`collect_named_inputs` instead (relative paths, clashes refused).
    """
    names, seen = [], {}
    for item in inputs:
        if item == "-":
            base = stdin_name or "stdin"
        else:
            path = Path(item)
            base = _base(path.name)
            if base == "predictions":
                base = path.resolve().parent.name
        count = seen.get(base, 0) + 1
        seen[base] = count
        names.append(base if count == 1 else f"{base}__{count}")
    return names


def sequence_name(input_name: str, key) -> str:
    return f"{input_name}__star{'single' if key is None else key}"


def _value_text(value) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    if isinstance(value, bool):
        return str(int(value))
    return str(value)


def _columns(name: str, array: np.ndarray, rows: int) -> list[tuple[str, list]]:
    a = np.asarray(array)
    if a.dtype == bool:
        a = a.astype(np.int8)
    if a.ndim == 1:
        return [(name, a.tolist())]
    flat = a.reshape(rows, int(np.prod(a.shape[1:], dtype=np.int64)) if rows == 0 else -1)
    return [(f"{name}_{k}", flat[:, k].tolist()) for k in range(flat.shape[1])]


def _write_csv(path: Path, columns: list[tuple[str, list]], rows: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow([name for name, _ in columns])
        texts = [[_value_text(v) for v in values] for _, values in columns]
        writer.writerows(zip(*texts) if texts else [[]] * rows)


def write_tables(directory: Path, sequence: str, c: PolylineContainer, outputs: dict) -> dict:
    """``<sequence>__frames.csv`` and ``<sequence>__arms.csv``; returns the written paths and omitted keys."""
    F, A = c.n_frames, c.n_arms
    frame_cols: list = [("frame_index", c.frame_index.tolist())]
    for field in ("source_frame_index", "time_s", "crop", *BODY_FIELDS, "has_star"):
        value = getattr(c, field)
        if field == "time_s" and value is None and c.fps is not None:
            value = c.times()
        if value is not None:
            frame_cols += _columns(field, value, F)
    if "detection_index" in c.meta:
        frame_cols.append(("detection_index", list(c.meta["detection_index"])))
    arm_cols: list = [("frame_index", np.repeat(c.frame_index, c.arms_per_frame()).tolist()),
                      ("arm", (np.arange(A) - np.repeat(c.frame_arm_offsets[:-1], c.arms_per_frame())).tolist())]
    if "track_id" not in outputs and c.track_id is not None:
        arm_cols.append(("track_id", c.track_id.tolist()))
    arm_tail = [("track_id", np.asarray(outputs["track_id"]).tolist())] if "track_id" in outputs else []
    for field in ("tip_score", "base_score", "state_confidence", "steps", "stop_reason"):
        value = getattr(c, field)
        if value is not None:
            arm_cols += _columns(field, value, A)
    omitted = []
    for key, value in outputs.items():
        if key == "track_id":
            continue
        axis = AXIS.get(key)
        a = np.asarray(value) if not isinstance(value, dict) else None
        if a is None or a.dtype == object:
            omitted.append(key)
        elif axis == "frame" and a.shape[:1] == (F,):
            frame_cols += _columns(key, a, F)
        elif axis == "arm" and a.shape[:1] == (A,):
            arm_tail += _columns(key, a, A)
        elif axis == "window" and a.ndim == 2 and a.shape[1] == F:
            frame_cols += _columns(key, a.T, F)
        elif axis == "scalar" and a.ndim == 1 and a.shape == (F,):
            frame_cols += _columns(key, a, F)      # a per-frame R0
        elif axis != "scalar":
            omitted.append(key)
    frames_path = directory / f"{sequence}__frames.csv"
    arms_path = directory / f"{sequence}__arms.csv"
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(frames_path, frame_cols, F)
    _write_csv(arms_path, arm_cols + arm_tail, A)
    return {"frames_csv": f"{sequence}__frames.csv", "arms_csv": f"{sequence}__arms.csv", "csv_omitted": omitted}


def _scalars(outputs: dict) -> dict:
    found = {}
    for key, value in outputs.items():
        if AXIS.get(key) != "scalar" or isinstance(value, dict):
            continue
        a = np.asarray(value)
        if a.ndim:
            continue
        item = a.item()
        found[key] = None if isinstance(item, float) and not math.isfinite(item) else item
    return found


def _context_for(options: dict, input_name: str, sequence: str) -> dict:
    context = {}
    if options.get("onset_s") is not None:
        context["onset_s"] = options["onset_s"]
    if options.get("bout") is not None:
        context["bout"] = dict(options["bout"])
    if options.get("windows") is not None:
        context["windows"] = [tuple(w) for w in options["windows"]]
    if options.get("r0") is not None:
        context["r0"] = options["r0"]
    per = options.get("context") or {}
    for key in (input_name, sequence):
        if key in per:
            context.update(per[key])
    if "windows" in context:
        context["windows"] = [tuple(w) for w in context["windows"]]
    return context


def process_input(job: dict) -> dict:
    """One input file: parse, compute every preset for every star, write; returns manifest entries."""
    path, input_name, options, out = job["path"], job["name"], job["options"], Path(job["output"])
    info = {"path": path, "name": input_name, "status": "ok", "error": None}
    if path != "-":
        stat = Path(path).stat()
        info.update(bytes=stat.st_size, mtime=stat.st_mtime)
    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            containers = load_inference(path, fps=options["fps"], untracked=options["untracked"],
                                        skip_unused=not options["parse_all_values"],
                                        fill_gaps=options["fill_gaps"], name=path if path != "-" else input_name,
                                        expected_records=manifest_records(path) if path != "-" else None)
        except Exception as exc:   # one unreadable input is reported, the run continues
            info.update(status="error", error=f"{type(exc).__name__}: {exc}")
            return {"input": info, "rows": rows}
        info["record_schema"] = next(iter(containers.values())).meta["record_schema"] if containers else None
        info["sequences"] = len(containers)
        sequences = [sequence_name(input_name, key) for key in containers]
        if len(set(sequences)) != len(sequences):
            info.update(status="error", error=f"two stars map to one sequence name: {sorted(sequences)}")
            return {"input": info, "rows": rows}
        if not containers:
            info.update(status="empty", error="no star in any record")
        for key, c in containers.items():
            sequence = sequence_name(input_name, key)
            base = {"input": input_name, "sequence": sequence, "star": key if key is not None else "single",
                    "frames": c.n_frames, "arms": c.n_arms, "fps": c.fps,
                    "missing_fields": list(c.meta["missing_fields"]), "record_schema": c.meta["record_schema"]}
            if c.meta.get("warnings"):
                base["warnings"] = list(c.meta["warnings"])
            if c.meta.get("missing_records"):
                base["missing_records"] = c.meta["missing_records"]
            if options["require_complete"] and c.meta["missing_fields"]:
                rows.append({**base, "preset": None, "status": "error",
                             "error": "missing fields: " + ", ".join(c.meta["missing_fields"])})
                info.update(status="error", error=f"{sequence}: incomplete (--require-complete)")
                continue
            if not options["no_containers"]:
                c.save(out / "containers" / f"{sequence}.npz")
            detections = c.meta.get("untracked") == "detections"
            for preset_name in options["presets"]:
                preset = PRESETS[preset_name]
                context = _context_for(options, input_name, sequence)
                row = {**base, "preset": preset_name, "r0_context": _jsonable(context)}
                try:
                    if detections and options["track"] == "relink":
                        raise ContainerError("untracked detections cannot be re-linked (no identity or track)")
                    if detections and preset.r0_policy in WINDOWED_R0 and any(
                            k in context for k in ("onset_s", "bout", "windows")):
                        raise ContainerError(f"{preset_name}: a windowed R0 ({preset.r0_policy}) needs one "
                                             "animal; untracked detections are independent")
                    outputs = compute_all(c, preset, track=options["track"], shapes=options["shapes"],
                                          kabsch=options["kabsch"], **context)
                    directory = out / preset_name
                    save_outputs(directory / f"{sequence}.npz", outputs)
                    written = sorted(k for k, v in outputs.items()
                                     if not isinstance(v, dict) and np.asarray(v).dtype != object)
                    row.update(status="ok", error=None, outputs=f"{preset_name}/{sequence}.npz",
                               keys_written=written,
                               keys_skipped=sorted(set(outputs) - set(written)),
                               dropped=bool(np.asarray(outputs.get("dropped", False))),
                               scalars=_scalars(outputs))
                    if options["csv"]:
                        row.update(write_tables(directory, sequence, c, outputs))
                    if base.get("warnings"):
                        row["status"] = "warning"
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
    messages = sorted({str(w.message) for w in caught if not issubclass(w.category, RuntimeWarning)})
    if messages:
        info["warnings"] = messages[:20]
    if any(row["status"] == "error" for row in rows) and info["status"] in ("ok", "empty"):
        info["status"] = "error"
        info["error"] = "one or more sequences failed"
    elif info["status"] == "ok" and (messages or any(row.get("warnings") for row in rows)):
        # Data was left out (a field missing on some star rows, frames without a record): the outputs are written,
        # but the input is not reported as clean.
        info["status"] = "warning"
        info["error"] = "; ".join(messages[:3]) or "see the sequence warnings"
    return {"input": info, "rows": rows}


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return value


def run_jobs(jobs: list[dict], workers: int, mp_context: str | None, stop_on_error: bool) -> Iterator[dict]:
    """``process_input`` over the jobs, results in input order (a pool of ``workers`` when > 1)."""
    if workers <= 1:
        for job in jobs:
            result = process_input(job)
            yield result
            if stop_on_error and result["input"]["status"] == "error":
                return
        return
    ctx = multiprocessing.get_context(mp_context) if mp_context else None
    pending: deque = deque()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        try:
            for job in jobs:
                pending.append(pool.submit(process_input, job))
                if len(pending) >= 2 * workers:
                    result = pending.popleft().result()
                    yield result
                    if stop_on_error and result["input"]["status"] == "error":
                        return
            while pending:
                result = pending.popleft().result()
                yield result
                if stop_on_error and result["input"]["status"] == "error":
                    return
        finally:
            for future in pending:
                future.cancel()


MANIFEST_CSV = ("input", "sequence", "star", "preset", "status", "frames", "arms", "fps", "missing_fields",
                "dropped", "outputs", "frames_csv", "arms_csv", "error", "missing_records")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="deepstarnoodles-inference representations",
        description="Turn inference output (JSON / JSON Lines records) into the DeepStarNoodles paper's polyline "
                    "representations: one container per star, the outputs of each figure preset and optional "
                    "CSV tables. Needs no GPU, torch or OpenCV.")
    p.add_argument("inputs", nargs="+", metavar="INPUT",
                   help="record files (.json, .jsonl, optionally .gz/.bz2/.xz/.zst), directories (searched for "
                        "*.jsonl*), globs, or - for stdin")
    p.add_argument("--output", type=Path, required=True, help="new output directory")
    p.add_argument("--preset", nargs="+", default=["all"], choices=[*PRESETS, "all"],
                   help="figure presets to compute (default all)")
    p.add_argument("--fps", type=float, default=None, help="frame rate for records that carry none")
    p.add_argument("--untracked", choices=UNTRACKED_MODES, default="error",
                   help="archive records without star_track_id: refuse (default), one star per frame (single), "
                        "or independent detections")
    p.add_argument("--no-fill-gaps", action="store_true",
                   help="do not add no-star rows where a tracked star is missing between appearances")
    r0 = p.add_mutually_exclusive_group()
    r0.add_argument("--context", type=str, default=None,
                    help="JSON object (inline or a file) mapping an input or sequence name to compute_all "
                         "context, e.g. {\"clip__star3\": {\"onset_s\": 331.0}}")
    r0.add_argument("--onset-s", type=float, default=None, help="Fig 5 baseline onset (s) for every sequence")
    r0.add_argument("--bout", type=float, nargs=3, metavar=("FIRST_S", "LAST_S", "FLIP_START_S"),
                    help="Fig 4E bout window and flip start (s) for every sequence")
    r0.add_argument("--windows-json", type=Path, default=None,
                    help="Fig 4D bout windows: a JSON list of [start_s, stop_s]")
    r0.add_argument("--r0", type=float, default=None, help="explicit R0 (source px) for every sequence")
    p.add_argument("--track", choices=("auto", "stored", "relink"), default="auto",
                   help="arm IDs: stored when complete else re-link (auto), stored, or always re-link")
    p.add_argument("--kabsch", choices=("closed", "svd"), default="closed")
    p.add_argument("--shapes", action="store_true", help="also write the resampled base shapes")
    p.add_argument("--workers", type=int, default=1, help="input files processed in parallel (processes)")
    p.add_argument("--mp-context", choices=("fork", "spawn", "forkserver"), default=None)
    p.add_argument("--csv", action="store_true", help="also write per-frame and per-arm CSV tables")
    p.add_argument("--no-containers", action="store_true", help="do not write the container .npz files")
    p.add_argument("--require-complete", action="store_true",
                   help="fail (exit 1) at the first star whose records lack a field some preset needs")
    p.add_argument("--parse-all-values", action="store_true",
                   help="parse the unused source-pixel arrays too (slower; reports malformed numbers in them)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an earlier representations run's outputs in --output (they are deleted first)")
    p.add_argument("--name", default=None,
                   help="name for the one input (stdin or a single file) instead of its stem; refused with several")
    args = p.parse_args(argv)
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        p.error("--fps must be finite and positive")
    if args.workers < 1:
        p.error("--workers must be positive")
    if "all" in args.preset:
        args.preset = list(PRESETS)
    args.preset = list(dict.fromkeys(args.preset))
    return p, args


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser, args = parse_args(argv)
    notes: list[str] = []
    try:
        named = collect_named_inputs(args.inputs, notes)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for note in notes:
        print(f"representations: {note}", file=sys.stderr)
    inputs = [path for path, _ in named]
    if not inputs:
        parser.error("no input files found")
    if args.name is not None:
        if len(inputs) != 1:
            parser.error(f"--name names a single input, but {len(inputs)} inputs were found")
        if not args.name or args.name in (".", "..") or "/" in args.name or "\\" in args.name:
            parser.error("--name must be a plain file-name stem (no path separators)")
    if "-" in inputs and args.workers > 1:
        args.workers = 1
    context = None
    if args.context is not None:
        text = Path(args.context).read_text() if Path(args.context).is_file() else args.context
        context = json.loads(text)
        if not isinstance(context, dict) or not all(isinstance(v, dict) for v in context.values()):
            parser.error("--context must map names to objects")
    windows = None
    if args.windows_json is not None:
        windows = json.loads(args.windows_json.read_text())
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} exists; choose a new directory or pass --overwrite")
    if args.output.exists():
        try:
            clear_outputs(args.output)
        except ValueError as exc:
            parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    options = {"fps": args.fps, "untracked": args.untracked, "fill_gaps": not args.no_fill_gaps,
               "parse_all_values": args.parse_all_values, "presets": args.preset, "track": args.track,
               "kabsch": args.kabsch, "shapes": args.shapes, "csv": args.csv, "no_containers": args.no_containers,
               "require_complete": args.require_complete, "onset_s": args.onset_s, "r0": args.r0,
               "bout": None if args.bout is None else dict(zip(("first_time_s", "last_time_s", "flip_start_s"),
                                                                args.bout)),
               "windows": windows, "context": context}
    names = [args.name] if args.name is not None else [
        name if path != "-" else "stdin" for path, name in named]
    jobs = [{"path": path, "name": name, "options": options, "output": str(args.output)}
            for path, name in zip(inputs, names)]
    manifest = {"schema_version": 1, "command": "representations", "argv": argv,
                "versions": _versions(), "options": _jsonable(options), "inputs": [], "sequences": [],
                "status": "running"}
    # Written first, so an interrupted run's directory is recognisable (and --overwrite can clear it).
    _write_manifest(args.output, manifest)
    failed = warned = False
    for result in run_jobs(jobs, args.workers, args.mp_context, args.require_complete):
        manifest["inputs"].append(result["input"])
        manifest["sequences"].extend(result["rows"])
        status = result["input"]["status"]
        if status == "error":
            failed = True
        warned |= status == "warning"
        if status != "ok":
            print(f"{result['input']['name']}: {status}: {result['input']['error']}", file=sys.stderr)
    manifest["status"] = "failed" if failed else "complete_with_warnings" if warned else "complete"
    _write_manifest(args.output, manifest)
    return 1 if failed else 0


def _write_manifest(output: Path, manifest: dict) -> None:
    (output / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=1, allow_nan=False) + "\n")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, MANIFEST_CSV, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in manifest["sequences"]:
            writer.writerow({**row, "missing_fields": ";".join(row.get("missing_fields") or [])})
        for item in manifest["inputs"]:
            # every input appears: one without sequence rows (unreadable, or no star in any record) gets its own row
            if item["status"] != "ok" and not any(r["input"] == item["name"] for r in manifest["sequences"]):
                writer.writerow({"input": item["name"], "status": item["status"], "error": item["error"]})


def clear_outputs(output: Path) -> None:
    """Delete an earlier representations run's outputs from ``output`` (``--overwrite``), so none goes stale.

    Only a directory a representations run wrote (its ``manifest.json`` says so) or an empty one is cleared; the
    containers, every preset's directory and the manifests are removed, anything else is left alone.
    """
    manifest = output / "manifest.json"
    if not output.is_dir():
        raise ValueError(f"{output} is not a directory")
    if not any(output.iterdir()):
        return
    try:
        command = json.loads(manifest.read_text()).get("command")
    except (OSError, ValueError, AttributeError):
        command = None
    if command != "representations":
        raise ValueError(f"{output} is not a representations output directory (no representations manifest.json); "
                         "refusing to clear it for --overwrite")
    for name in ("containers", *PRESETS):
        if (output / name).is_dir():
            shutil.rmtree(output / name)
    for name in ("manifest.json", "manifest.csv"):
        (output / name).unlink(missing_ok=True)


def _versions() -> dict:
    import platform
    from . import __version__
    import scipy
    return {"deepstarnoodles_inference": __version__, "noodle_polylines": POLYLINES_VERSION,
            "noodle_polylines_commit": VENDORED_COMMIT, "numpy": np.__version__, "scipy": scipy.__version__,
            "python": platform.python_version()}


if __name__ == "__main__":
    raise SystemExit(main())
