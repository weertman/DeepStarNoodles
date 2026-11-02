# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Exact segmented reductions over CSR (ragged) arrays.

The per-arm figure-script code sums and accumulates small float32 arrays one arm at a time. A plain
``np.add.reduceat`` or a global ``cumsum`` gives the same numbers only up to rounding, because it
groups the additions differently. These helpers reproduce numpy's own order of operations for every
segment at once, so their results are bit-identical to the per-arm calls:

* :func:`seg_cumsum` - ``np.cumsum`` of each segment (strictly sequential accumulation);
* :func:`seg_sum` - ``np.sum`` of each contiguous 1-D segment (numpy's pairwise summation: sequential
  under 8 values, eight interleaved accumulators up to 128, recursive halves above);
* :func:`seg_search` - per-segment binary search (``searchsorted``) of many sorted queries.

Each helper loops over positions within a segment (at most the longest segment), not over segments,
so the Python overhead is independent of the number of arms.
"""
from __future__ import annotations

import numpy as np

_PW_BLOCK = 128      # numpy's PW_BLOCKSIZE (pairwise summation)


def seg_cumsum(vals: np.ndarray, starts: np.ndarray, counts: np.ndarray, out: np.ndarray | None = None
               ) -> np.ndarray:
    """Inclusive per-segment cumulative sum, sequential within a segment exactly as ``np.cumsum``.

    ``vals`` (N,) is any 1-D array; segment ``s`` is ``vals[starts[s]:starts[s] + counts[s]]``. Entries
    outside every segment are copied unchanged. Returns ``out`` (a copy of ``vals`` by default).
    """
    out = vals.copy() if out is None else out
    counts = np.asarray(counts)
    if not len(counts):
        return out
    cmax = int(counts.max())
    if cmax <= 1:
        return out
    order = np.argsort(-counts, kind='stable')
    s_sorted = np.asarray(starts)[order]
    c_sorted = counts[order]
    # active[j]: segments with more than j elements (a prefix of the sorted order)
    active = np.searchsorted(-c_sorted, -np.arange(1, cmax), side='left')
    for j in range(1, cmax):
        idx = s_sorted[:active[j - 1]] + j
        out[idx] += out[idx - 1]
    return out


def seg_sum(vals: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``np.sum`` of every contiguous 1-D segment, bit-identical (numpy pairwise summation).

    Returns (S,) in the dtype of ``vals``; an empty segment sums to 0. Assumes no segment value is
    ``-0.0`` where the sign of a zero result matters (sums of norms and absolute values are safe).
    """
    starts = np.asarray(starts, np.int64)
    counts = np.asarray(counts, np.int64)
    out = np.zeros(len(counts), vals.dtype)
    if not len(counts):
        return out
    # n < 8: sequential from zero
    small = np.flatnonzero(counts < 8)
    if len(small):
        st, c = starts[small], counts[small]
        res = np.zeros(len(small), vals.dtype)
        for i in range(int(c.max()) if len(c) else 0):
            sel = c > i
            res[sel] += vals[st[sel] + i]
        out[small] = res
    # 8 <= n <= 128: eight accumulators, combined pairwise, then the < 8 remainder sequentially
    mid = np.flatnonzero((counts >= 8) & (counts <= _PW_BLOCK))
    if len(mid):
        st, c = starts[mid], counts[mid]
        lane = np.arange(8)
        r = vals[st[:, None] + lane]
        nb = c - c % 8
        for i in range(8, int(nb.max()), 8):
            sel = np.flatnonzero(nb > i)
            r[sel] += vals[st[sel, None] + i + lane]
        res = ((r[:, 0] + r[:, 1]) + (r[:, 2] + r[:, 3])) + ((r[:, 4] + r[:, 5]) + (r[:, 6] + r[:, 7]))
        rem = c - nb
        for t in range(int(rem.max())):
            sel = np.flatnonzero(rem > t)
            res[sel] += vals[st[sel] + nb[sel] + t]
        out[mid] = res
    # n > 128: rare; numpy itself on each segment
    for s in np.flatnonzero(counts > _PW_BLOCK):
        out[s] = np.add.reduce(vals[starts[s]:starts[s] + counts[s]])
    return out


def seg_search(arr: np.ndarray, x: np.ndarray, lo: np.ndarray, hi: np.ndarray, strict: bool) -> np.ndarray:
    """Per-query binary search in ``arr[lo:hi]`` (sorted, non-decreasing within each range).

    Returns the largest ``j`` in ``[lo, hi - 1]`` with ``arr[j] <= x`` (``strict=False``) or
    ``arr[j] < x`` (``strict=True``), and ``lo`` when no entry qualifies. ``x``, ``lo``, ``hi`` broadcast
    to one shape; comparisons are exact.
    """
    x, lo, hi = np.broadcast_arrays(x, lo, hi)
    span = int((hi - lo).max()) if hi.size else 0
    idt = np.int32 if len(arr) < 2 ** 31 - 1 else np.int64
    lo = lo.astype(idt, copy=True)
    hi = hi.astype(idt, copy=True)
    # Branch-free bisection: once hi - lo <= 1 an element stays put (mid == lo), so a fixed number of
    # halvings serves every query.
    for _ in range(max(span - 1, 0).bit_length()):
        mid = (lo + hi) >> 1
        ok = (arr[mid] < x) if strict else (arr[mid] <= x)
        np.copyto(lo, mid, where=ok)
        np.copyto(hi, mid, where=~ok)
    return lo
