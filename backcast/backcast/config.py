"""Job configuration: one TOML file, validated up front."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

IMAGE = "runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853"
MODELS_URL = "https://github.com/weertman/deepstarnoodles-inference/releases/download/models-seed0-20261028/manifest.json"
VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi")


@dataclass(frozen=True)
class GpuChoice:
    id: str
    max_usd_per_hour: float
    processes: int
    max_pods: int | None = None
    expected_images_per_second: float | None = None
    family: str | None = None


@dataclass(frozen=True)
class Config:
    footage_roots: tuple[Path, ...]
    output_dir: Path
    run_dir: Path
    gpus: tuple[GpuChoice, ...]
    pods: int
    max_total_usd: float
    cloud: str = "SECURE"
    gpu_family: str = "any"
    lease_hours: float = 12.0
    extensions: tuple[str, ...] = VIDEO_SUFFIXES
    exclude: tuple[str, ...] = ()
    upload_limit_mbit: float = 400.0
    uploads_per_pod: int = 3
    ready_videos: int = 4
    min_vcpu: int = 8
    min_memory_gb: int = 32
    container_gb: int = 80
    image: str = IMAGE
    models_url: str = MODELS_URL
    max_attempts: int = 3
    stall_minutes: float = 30.0
    pod_root: str = "/root/backcast"
    source: Path | None = field(default=None, compare=False)


def _require(table: dict, key: str, kind, where: str):
    if key not in table or not isinstance(table[key], kind):
        raise ValueError(f"{where}: `{key}` is required and must be {getattr(kind, '__name__', kind)}")
    return table[key]


def load(path: Path) -> Config:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    footage = _require(data, "footage", dict, "config")
    output = _require(data, "output", dict, "config")
    fleet = _require(data, "fleet", dict, "config")
    pod = data.get("pod", {})
    roots = tuple(Path(p).expanduser() for p in _require(footage, "roots", list, "[footage]"))
    if not roots:
        raise ValueError("[footage]: list at least one directory in `roots`")
    gpus = []
    for item in _require(fleet, "gpu", list, "[fleet]"):
        gpus.append(GpuChoice(id=_require(item, "id", str, "[[fleet.gpu]]"),
                              max_usd_per_hour=float(_require(item, "max_usd_per_hour", (int, float), "[[fleet.gpu]]")),
                              processes=int(_require(item, "processes", int, "[[fleet.gpu]]")),
                              max_pods=item.get("max_pods"),
                              expected_images_per_second=item.get("expected_images_per_second"),
                              family=str(item["family"]).lower() if "family" in item else None))
    if not gpus:
        raise ValueError("[fleet]: add at least one [[fleet.gpu]] entry")
    family = str(fleet.get("gpu_family", "any")).lower()
    if family != "any":
        known = sorted({g.family for g in gpus if g.family})
        gpus = [g for g in gpus if g.family == family]
        if not gpus:
            raise ValueError(f"[fleet]: no [[fleet.gpu]] has family {family!r}; `gpu_family` can be \"any\" or one of {known}")
    if any(not 1 <= g.processes <= 16 for g in gpus):
        raise ValueError("[[fleet.gpu]]: `processes` must be between 1 and 16")
    run_dir = Path(output.get("run_dir", "runs/backcast")).expanduser()
    if not run_dir.is_absolute():
        run_dir = path.parent / run_dir
    config = Config(
        footage_roots=roots,
        output_dir=Path(_require(output, "dir", str, "[output]")).expanduser(),
        run_dir=run_dir,
        gpus=tuple(gpus),
        pods=int(_require(fleet, "pods", int, "[fleet]")),
        max_total_usd=float(_require(fleet, "max_total_usd", (int, float), "[fleet]")),
        cloud=fleet.get("cloud", "SECURE"),
        gpu_family=family,
        lease_hours=float(fleet.get("lease_hours", 12.0)),
        extensions=tuple(s.lower() for s in footage.get("extensions", VIDEO_SUFFIXES)),
        exclude=tuple(footage.get("exclude", ())),
        upload_limit_mbit=float(fleet.get("upload_limit_mbit", 400.0)),
        uploads_per_pod=int(fleet.get("uploads_per_pod", 3)),
        ready_videos=int(fleet.get("ready_videos", 4)),
        max_attempts=int(fleet.get("max_attempts", 3)),
        stall_minutes=float(fleet.get("stall_minutes", 30.0)),
        min_vcpu=int(pod.get("min_vcpu", 8)),
        min_memory_gb=int(pod.get("min_memory_gb", 32)),
        container_gb=int(pod.get("container_gb", 80)),
        image=pod.get("image", IMAGE),
        models_url=pod.get("models_url", MODELS_URL),
        source=path,
    )
    unsupported = set(config.extensions) - set(VIDEO_SUFFIXES)
    if unsupported:
        raise ValueError(f"[footage]: `serve-videos` only claims {VIDEO_SUFFIXES}; add {sorted(unsupported)} to SUFFIXES in "
                         "inference/src/deepstarnoodles_inference/video_service.py first, if OpenCV can decode them")
    if config.cloud not in ("SECURE", "COMMUNITY"):
        raise ValueError("[fleet]: `cloud` must be SECURE or COMMUNITY")
    if config.pods < 1 or config.max_total_usd <= 0:
        raise ValueError("[fleet]: `pods` and `max_total_usd` must be positive")
    if config.lease_hours < 1.5:
        raise ValueError("[fleet]: `lease_hours` must be at least 1.5, or pods expire before they finish a long video")
    return config
