# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Serialize CUDA-graph capture with other in-process CUDA work.

``CAPTURE_LOCK`` is held by graph capture and by threads that issue other CUDA
work concurrently with a capturing thread (detect-ahead). ``GRAPH_GATE``
additionally separates capture from replay: PyTorch registers the default CUDA
generator with every capture, and a replay on another thread while that
generator is in capture mode fails even when no random numbers are used.
Replays share the gate; a capture takes it exclusively.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

CAPTURE_LOCK = threading.RLock()


class _ReplayCaptureGate:
    """Shared replay / exclusive capture gate, preferring waiting captures."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._replays = 0
        self._capturing = False
        self._waiting_captures = 0

    @contextmanager
    def replay(self):
        with self._condition:
            while self._capturing or self._waiting_captures:
                self._condition.wait()
            self._replays += 1
        try:
            yield
        finally:
            with self._condition:
                self._replays -= 1
                if not self._replays:
                    self._condition.notify_all()

    @contextmanager
    def capture(self):
        with self._condition:
            self._waiting_captures += 1
            try:
                while self._capturing or self._replays:
                    self._condition.wait()
            finally:
                self._waiting_captures -= 1
            self._capturing = True
        try:
            yield
        finally:
            with self._condition:
                self._capturing = False
                self._condition.notify_all()


GRAPH_GATE = _ReplayCaptureGate()
