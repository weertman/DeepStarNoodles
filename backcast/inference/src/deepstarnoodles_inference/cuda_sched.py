# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Opt-in blocking host waits for the CUDA primary context.

When a process has fewer CUDA contexts than visible logical CPUs, the driver
spins a host core while a thread waits for the GPU (stream or device
synchronization, device-to-host copies). Containers that show many CPUs but
enforce a small CPU quota pay for every spinning waiter, which starves image
decoding and crop preparation in the other archive processes. Blocking
synchronization puts waiting threads to sleep instead. It changes only how the
host waits, never what the GPU computes; a woken thread may resume a few
microseconds later than a spinning one.

``Event.synchronize()`` follows the event's own flag rather than the context's,
so events used for host waits are created with ``event()``.
"""
from __future__ import annotations

import ctypes

import torch

CU_CTX_SCHED_MASK = 0x07
CU_CTX_SCHED_BLOCKING_SYNC = 0x04
_requested = False


def _check(code: int, call: str) -> None:
    if code != 0:
        raise RuntimeError(f"{call} failed with CUDA driver error {code}")


def request_blocking_sync(device) -> dict:
    """Make host waits on ``device``'s primary context sleep; returns the context state.

    The driver accepts the flag before or after the context becomes active.
    """
    global _requested
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("blocking synchronization requires a CUDA device")
    driver = ctypes.CDLL("libcuda.so.1")
    _check(driver.cuInit(0), "cuInit")
    handle = ctypes.c_int()
    _check(driver.cuDeviceGet(ctypes.byref(handle), 0 if device.index is None else device.index), "cuDeviceGet")
    flags, active = ctypes.c_uint(), ctypes.c_int()
    _check(driver.cuDevicePrimaryCtxGetState(handle, ctypes.byref(flags), ctypes.byref(active)),
           "cuDevicePrimaryCtxGetState")
    was_active = bool(active.value)
    _check(driver.cuDevicePrimaryCtxSetFlags_v2(handle, (flags.value & ~CU_CTX_SCHED_MASK) | CU_CTX_SCHED_BLOCKING_SYNC),
           "cuDevicePrimaryCtxSetFlags_v2")
    _check(driver.cuDevicePrimaryCtxGetState(handle, ctypes.byref(flags), ctypes.byref(active)),
           "cuDevicePrimaryCtxGetState")
    if flags.value & CU_CTX_SCHED_MASK != CU_CTX_SCHED_BLOCKING_SYNC:
        raise RuntimeError(f"primary context kept scheduling flags {flags.value:#x}")
    _requested = True
    return {"primary_context_flags": flags.value, "context_active_when_set": was_active}


def event(**kwargs) -> torch.cuda.Event:
    """A CUDA event whose host waits sleep once blocking synchronization was requested."""
    return torch.cuda.Event(blocking=_requested, **kwargs)
