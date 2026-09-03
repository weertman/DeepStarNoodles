# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Read-only recorder status gate so Hydra yields before archival StarSeg blocks."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

DEFAULT_STATUS_URL = "http://127.0.0.1:8000/status.json"
DEFAULT_QUEUE_WATERMARK = 8
DEFAULT_POLL_S = 0.25

FetchFn = Callable[[str], dict]


@dataclass(frozen=True)
class RecorderSnapshot:
    reachable: bool
    recording: bool = False
    recorded_queue_depth: int = 0
    recorded_waits: int = 0


def fetch_status(url: str, timeout: float = 0.4) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def snapshot_from_status(payload: dict) -> RecorderSnapshot:
    starseg = payload.get("starseg") or {}
    return RecorderSnapshot(
        reachable=True,
        recording=bool(payload.get("recording")),
        recorded_queue_depth=int(starseg.get("recorded_queue_depth") or 0),
        recorded_waits=int(starseg.get("recorded_waits") or 0),
    )


class RecorderGuard:
    """Pause Hydra when archival queue depth or recorded_waits indicate pressure.

    A missing recorder is treated as no live victim: Hydra keeps the measured batch.
    """

    def __init__(
        self,
        url: str = DEFAULT_STATUS_URL,
        *,
        queue_watermark: int = DEFAULT_QUEUE_WATERMARK,
        poll_s: float = DEFAULT_POLL_S,
        fetch: FetchFn | None = None,
    ) -> None:
        self.url = url
        self.queue_watermark = queue_watermark
        self.poll_s = poll_s
        self._fetch = fetch or fetch_status
        self._last_waits: int | None = None

    def poll(self) -> RecorderSnapshot:
        try:
            payload = self._fetch(self.url)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
            return RecorderSnapshot(reachable=False)
        if not isinstance(payload, dict):
            return RecorderSnapshot(reachable=False)
        return snapshot_from_status(payload)

    def should_pause(self, snap: RecorderSnapshot) -> bool:
        if not snap.reachable:
            return False
        if snap.recorded_queue_depth > self.queue_watermark:
            return True
        if self._last_waits is not None and snap.recorded_waits > self._last_waits:
            return True
        return False

    def note(self, snap: RecorderSnapshot) -> None:
        if snap.reachable:
            self._last_waits = snap.recorded_waits

    def wait_until_clear(self, on_pause: Callable[[], None] | None = None) -> None:
        if on_pause is not None:
            on_pause()
        while True:
            snap = self.poll()
            if not snap.reachable or snap.recorded_queue_depth <= self.queue_watermark:
                self.note(snap)
                return
            time.sleep(self.poll_s)

    def yield_if_needed(self, on_pause: Callable[[], None] | None = None) -> RecorderSnapshot:
        snap = self.poll()
        if self.should_pause(snap):
            self.wait_until_clear(on_pause)
            snap = self.poll()
        self.note(snap)
        return snap
