"""Find the videos to annotate and read their length and shape."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fnmatch
import json
import os
from pathlib import Path
import subprocess


def discover(roots, extensions, exclude=()) -> list[tuple[str, str, int]]:
    """(absolute path, path relative to its root's parent, bytes) for every video, sorted."""
    found = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"footage root is not a directory: {root}")
        for folder, _dirs, names in os.walk(root, followlinks=True):
            for name in names:
                path = Path(folder) / name
                if path.suffix.lower() not in extensions or name.startswith("."):
                    continue
                relative = path.relative_to(root.parent).as_posix()
                if any(fnmatch.fnmatch(relative, pattern) for pattern in exclude):
                    continue
                found.append((str(path), relative, path.stat().st_size))
    found.sort(key=lambda row: row[1])
    relatives = [row[1] for row in found]
    if len(set(relatives)) != len(relatives):
        raise ValueError("two footage roots hold the same relative path; output files would collide")
    return found


def probe(path: str) -> dict:
    """Frames, fps, duration and frame size from the container, without decoding."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,nb_frames,avg_frame_rate,duration:format=duration", "-of", "json", path],
        capture_output=True, text=True, timeout=120)
    if result.returncode:
        return {"error": result.stderr.strip()[-200:] or "ffprobe failed"}
    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    number = lambda value: float(value) if value not in (None, "N/A", "") else None
    duration = number(stream.get("duration")) or number((data.get("format") or {}).get("duration"))
    top, _, bottom = str(stream.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(top) / float(bottom) if bottom and float(bottom) else None
    frames = number(stream.get("nb_frames"))
    if frames is None and duration and fps:
        frames = round(duration * fps)
    return {"codec": stream.get("codec_name"), "width": stream.get("width"), "height": stream.get("height"),
            "fps": fps, "duration": duration, "frames": int(frames) if frames else None}


def probe_many(paths, workers: int = 8) -> dict:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(paths, pool.map(probe, paths)))
