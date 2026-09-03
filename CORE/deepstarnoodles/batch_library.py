# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Inventory, state, safeguards, and timeline contracts for library inference."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from deepstarnoodles.roi import ROI, ROIError, load_roi


def atomic_json(path: Path, payload: dict, *, compact: bool = False) -> None:
    """Write JSON through a same-directory temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":") if compact else None, indent=None if compact else 2)
        + "\n"
    )
    os.replace(temporary, path)


SOURCE_ROOT = Path("/home/starbot/Videos/Startracked Loop Recordings")
OUTPUT_VOLUME = Path("/media/starbot/7479-B887")
OUTPUT_ROOT = OUTPUT_VOLUME / "DeepStarNoodles Outputs" / "timelapse-batch"
OUTPUT_PROJECTS_ROOT = OUTPUT_VOLUME / "DeepStarNoodles Outputs"
STATE_ROOT = Path("RUNS/batch")
MIN_FREE_BYTES = 500 * 1024**3
ARTIFACTS = ("tracks.json", "overlay.mp4", "hydra_channels.mp4", "metadata.json", "status.json")


@dataclass(frozen=True)
class Clip:
    key: str
    camera: str
    trial: str
    date: str
    stem: str
    video: Path
    metadata_path: Path
    metadata: dict
    roi: ROI

    @property
    def output_relative(self) -> Path:
        return Path(self.camera) / self.trial / self.stem


@dataclass(frozen=True)
class InventoryIssue:
    key: str
    video: str
    status: str
    error: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def contract_hash(checkpoints: Iterable[Path] = ()) -> tuple[str, dict[str, str]]:
    contract = {
        "schema": 2,
        "roi": "starseg_roi.absolute/source-mask/candidate-center/crop-mask",
        "tracker": {"shape_gate": 120.0, "embed_weight": 2.0, "max_age": 16},
        "channels": "legacy10+embedding8/fixed-scale-3",
    }
    checkpoint_hashes: dict[str, str] = {}
    for path in checkpoints:
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            checkpoint_hashes[str(path)] = digest.hexdigest()
    body = json.dumps(
        {"contract": contract, "checkpoints": checkpoint_hashes},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode()).hexdigest(), checkpoint_hashes


def _date(metadata: dict) -> str:
    iso = str(metadata.get("started_iso") or metadata.get("recording_started_iso") or "")
    return iso[:10] if len(iso) >= 10 else "unknown"


def inventory(source_root: Path = SOURCE_ROOT) -> tuple[list[Clip], list[InventoryIssue]]:
    clips: list[Clip] = []
    issues: list[InventoryIssue] = []
    for camera in ("A", "B", "C", "D"):
        camera_root = source_root / camera
        if not camera_root.is_dir():
            continue
        for video in camera_root.glob("*/*/*.mp4"):
            stem = video.stem
            # Only the canonical finalized artifact is an input. Preview,
            # overlay, and other MP4 sidecars in the segment folder are ignored.
            if video.name != f"{video.parent.name}.mp4":
                continue
            key = f"{camera}:{stem}"
            metadata_path = video.with_suffix(".json")
            try:
                roi, metadata = load_roi(metadata_path)
            except ROIError as exc:
                issues.append(InventoryIssue(key, str(video), exc.status, str(exc)))
                continue
            clips.append(
                Clip(
                    key=key,
                    camera=camera,
                    # Derive output path components from the canonical source
                    # hierarchy, never from mutable metadata strings.
                    trial=video.parents[1].name,
                    date=_date(metadata),
                    stem=stem,
                    video=video,
                    metadata_path=metadata_path,
                    metadata=metadata,
                    roi=roi,
                )
            )
    clips.sort(key=lambda clip: (clip.camera, clip.metadata.get("start_ts", 0), clip.stem))
    return clips, issues


def balanced(clips: Iterable[Clip]) -> list[Clip]:
    queues: dict[str, deque[Clip]] = defaultdict(deque)
    for clip in clips:
        queues[clip.camera].append(clip)
    result: list[Clip] = []
    while any(queues.values()):
        for camera in ("A", "B", "C", "D"):
            if queues[camera]:
                result.append(queues[camera].popleft())
    return result


def require_output_volume(
    volume: Path = OUTPUT_VOLUME,
    *,
    min_free_bytes: int = MIN_FREE_BYTES,
) -> int:
    if not volume.is_dir() or not os.path.ismount(volume):
        raise RuntimeError(f"output volume is not mounted: {volume}")
    free = shutil.disk_usage(volume).free
    if free < min_free_bytes:
        raise RuntimeError(
            f"disk guard paused: {free / 1024**3:.1f} GiB free; "
            f"requires {min_free_bytes / 1024**3:.0f} GiB"
        )
    return free


def validate_output_root(output_root: Path) -> Path:
    """Confine batch writes to a dedicated project below the approved root."""
    resolved = output_root.resolve()
    approved = OUTPUT_PROJECTS_ROOT.resolve()
    try:
        relative = resolved.relative_to(approved)
    except ValueError as exc:
        raise RuntimeError(f"output root must be inside {approved}") from exc
    if not relative.parts:
        raise RuntimeError("output root must be a dedicated project directory")
    if relative.parts[0] in {"A", "B", "C", "D"}:
        raise RuntimeError("top-level LaCie camera directories are reserved and must not be altered")
    return resolved


def is_complete(output_dir: Path, expected_hash: str) -> bool:
    try:
        status = json.loads((output_dir / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        status.get("status") == "complete"
        and status.get("contract_hash") == expected_hash
        and all(
            (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0
            for name in ARTIFACTS
        )
    )


def promote_directory(partial: Path, output: Path) -> None:
    """Atomically swap a completed directory, preserving rollback on failure."""
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    had_output = output.exists()
    if had_output:
        os.replace(output, backup)
    try:
        os.replace(partial, output)
    except Exception:
        if had_output and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


class BatchState:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA busy_timeout=5000")
            self.connection.execute(
                """CREATE TABLE IF NOT EXISTS clips (
                key TEXT PRIMARY KEY, camera TEXT, trial TEXT, date TEXT, source TEXT,
                output TEXT, status TEXT, frame INTEGER DEFAULT 0, total INTEGER,
                attempts INTEGER DEFAULT 0, error TEXT, contract_hash TEXT, updated_at TEXT)"""
            )
            self.connection.commit()

    def upsert(self, clip: Clip, output: Path, status: str = "queued", **fields: object) -> None:
        with self._lock:
            row = self.connection.execute("SELECT attempts FROM clips WHERE key=?", (clip.key,)).fetchone()
            attempts = int(fields.get("attempts", row[0] if row else 0))
            self.connection.execute(
                """INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET camera=excluded.camera, trial=excluded.trial,
                date=excluded.date, source=excluded.source, output=excluded.output,
                status=excluded.status, frame=excluded.frame, total=excluded.total,
                attempts=excluded.attempts, error=excluded.error,
                contract_hash=excluded.contract_hash, updated_at=excluded.updated_at""",
                (
                    clip.key, clip.camera, clip.trial, clip.date, str(clip.video), str(output),
                    status, fields.get("frame", 0), fields.get("total"), attempts,
                    fields.get("error"), fields.get("contract_hash"), utc_now(),
                ),
            )
            self.connection.commit()

    def rows(self) -> list[dict]:
        with self._lock:
            cursor = self.connection.execute("SELECT * FROM clips ORDER BY updated_at DESC")
            names = [item[0] for item in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]

    def request_retry(self, key: str) -> None:
        with self._lock:
            self.connection.execute(
                "UPDATE clips SET status='retry_requested', error=NULL, updated_at=? WHERE key=?",
                (utc_now(), key),
            )
            self.connection.commit()

    def summary(self) -> dict:
        rows = self.rows()
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["status"]] += 1
        return {"updated_at": utc_now(), "counts": dict(counts), "clips": rows}

    def rebuild(self, output_root: Path) -> int:
        count = 0
        for path in output_root.glob("*/*/*/status.json"):
            try:
                status = json.loads(path.read_text())
                source = Path(status["source"])
                metadata_path = source.with_suffix(".json")
                roi, metadata = load_roi(metadata_path)
                camera = str(status["camera"])
                clip = Clip(
                    status["key"], camera, str(status["trial"]), _date(metadata),
                    source.stem, source, metadata_path, metadata, roi,
                )
                self.upsert(clip, path.parent, status=status["status"], **{
                    key: status.get(key) for key in ("frame", "total", "error", "contract_hash")
                })
                count += 1
            except (KeyError, OSError, json.JSONDecodeError, ROIError):
                continue
        return count


def write_status(path: Path, clip: Clip, status: str, contract: str, **fields: object) -> None:
    atomic_json(
        path,
        {
            "schema_version": 2,
            "key": clip.key,
            "camera": clip.camera,
            "trial": clip.trial,
            "date": clip.date,
            "source": str(clip.video),
            "roi_sha256": clip.roi.sha256,
            "status": status,
            "contract_hash": contract,
            "updated_at": utc_now(),
            **fields,
        },
    )


def transcode_h264(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.mp4")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an", str(temporary),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"H.264 transcode failed: {result.stderr.strip()}")
    os.replace(temporary, destination)


def timeline(rows: Iterable[dict]) -> dict:
    clips = []
    for row in rows:
        if row.get("status") != "complete":
            continue
        metadata_path = Path(row["source"]).with_suffix(".json")
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        stamps = metadata.get("frame_timestamps") or []
        if not stamps:
            continue
        clips.append(
            {
                "key": row["key"],
                "camera": row["camera"],
                "trial": row["trial"],
                "date": row["date"],
                "start": stamps[0],
                "end": stamps[-1],
                "frame_timestamps": stamps,
            }
        )
    clips.sort(key=lambda item: (item["camera"], item["start"]))
    for index, clip in enumerate(clips):
        previous = clips[index - 1] if index else None
        clip["gap_before_s"] = (
            max(0.0, clip["start"] - previous["end"])
            if previous and previous["camera"] == clip["camera"]
            else None
        )
    return {"schema_version": 1, "clips": clips}
