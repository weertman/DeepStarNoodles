# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Choose a Hydra batch size from measurement, not a naive B=1 default."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepstarnoodles.paths import REPO_ROOT

DEFAULT_BATCH_SIZE = 4
DEFAULT_MEMORY_FRACTION = 0.45
YOLO_P99_LIMIT_MS = 200.0
CANDIDATES = (8, 4, 1)
DEFAULT_BUDGET_PATH = REPO_ROOT / "config" / "hydra_budget.default.json"
MEASURED_BUDGET_PATH = REPO_ROOT / "RUNS" / "hydra_budget.json"


def choose_batch_size(
    results: dict[int, dict[str, Any]],
    *,
    candidates: tuple[int, ...] = CANDIDATES,
    yolo_p99_limit_ms: float = YOLO_P99_LIMIT_MS,
) -> int:
    """Largest candidate that ran and keeps concurrent YOLO p99 under budget."""
    runnable: list[int] = []
    for batch in candidates:
        item = results.get(batch)
        if not item or item.get("error"):
            continue
        yolo = item.get("yolo_p99_ms")
        if yolo is not None and float(yolo) >= yolo_p99_limit_ms:
            continue
        runnable.append(int(batch))
    return max(runnable) if runnable else 1


def recommend_memory_fraction(results: dict[int, dict[str, Any]], batch_size: int) -> float:
    item = results.get(batch_size) or {}
    used = item.get("hydra_vram_bytes")
    total = item.get("gpu_total_bytes")
    if not used or not total:
        return DEFAULT_MEMORY_FRACTION
    # Leave a reserved slab for live YOLO; clamp to a usable band.
    fraction = (float(used) / float(total)) + 0.08
    return max(0.20, min(0.60, fraction))


def budget_payload(
    results: dict[int, dict[str, Any]],
    *,
    status: str,
    device: str,
    yolo_p99_limit_ms: float = YOLO_P99_LIMIT_MS,
    note: str = "",
) -> dict[str, Any]:
    batch = choose_batch_size(results, yolo_p99_limit_ms=yolo_p99_limit_ms)
    return {
        "status": status,
        "device": device,
        "recommended_batch_size": batch,
        "memory_fraction": recommend_memory_fraction(results, batch),
        "yolo_p99_limit_ms": yolo_p99_limit_ms,
        "results": {str(key): value for key, value in results.items()},
        "note": note,
    }


def load_budget(path: Path | None = None) -> dict[str, Any]:
    for candidate in (
        path,
        MEASURED_BUDGET_PATH if path is None else None,
        DEFAULT_BUDGET_PATH,
    ):
        if candidate is None or not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {
        "status": "unmeasured",
        "recommended_batch_size": DEFAULT_BATCH_SIZE,
        "memory_fraction": DEFAULT_MEMORY_FRACTION,
        "yolo_p99_limit_ms": YOLO_P99_LIMIT_MS,
    }


def write_budget(payload: dict[str, Any], path: Path = MEASURED_BUDGET_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
