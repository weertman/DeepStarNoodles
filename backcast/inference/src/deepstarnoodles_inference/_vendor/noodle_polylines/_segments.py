# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Exact per-segment order statistics and row sums for the per-frame aggregates (T2).

A segment is a contiguous run ``vals[starts[s]:starts[s] + counts[s]]``; for the T2 aggregates it is one
frame's arms. The figure scripts reduce one frame at a time with ``np.median``, ``np.nanmedian``,
``np.percentile`` (linear), ``ndarray.mean(0)`` and ``np.linalg.norm``. The helpers here give the same
numbers for every segment at once:

* :func:`seg_sort` - each segment sorted ascending, NaN last (as ``np.sort`` / ``np.partition`` see it),
  in the original layout;
* :func:`seg_median` - ``np.median`` (NaN-propagating) or ``np.nanmedian`` of each sorted segment;
* :func:`seg_percentile` - ``np.percentile(x, q)`` with the default ``'linear'`` method, including numpy's
  index clamping and its two-sided ``_lerp``;
* :func:`seg_rowsum` - ``x.sum(0)`` of (n, k) blocks, which numpy accumulates row by row from zero;
* :func:`row_norm` - ``np.linalg.norm`` of each row of an (N, 2) float64 array (``sqrt(x.dot(x))``, which
  goes through BLAS ``ddot`` and so may differ in the last bit from ``sqrt(x*x + y*y)``).

Order statistics are exact values of the input, so any correct sort gives them; a median or percentile
is then the same IEEE expression numpy evaluates. Only the sign of a zero result may differ.
"""
from __future__ import annotations

import numpy as np

_SORT_ROWS = 1 << 16          # rows per padded block in seg_sort


def seg_starts(counts: np.ndarray) -> np.ndarray:
    """(S,) int64 start of each segment of a contiguous layout with these ``counts``."""
    counts = np.asarray(counts, np.int64)
    out = np.zeros(len(counts), np.int64)
    np.cumsum(counts[:-1], out=out[1:])
    return out


def seg_sort(vals: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``vals`` with every segment sorted ascending (NaN last); positions outside segments are copied.

    Segments are grouped by length into power-of-two width classes and sorted as NaN-padded rows
    (``np.sort(axis=1)``), which is several times faster than a lexsort for the short segments of the
    released data (<= ~20 arms per frame) and keeps the padding below 2x per class.
    """
    vals = np.asarray(vals)
    out = vals.copy()
    starts = np.asarray(starts, np.int64)
    counts = np.asarray(counts, np.int64)
    if not len(counts) or int(counts.max(initial=0)) < 2:
        return out
    cmax = int(counts.max())
    w = 2
    while True:
        lo = w // 2
        sel = np.flatnonzero((counts > lo) & (counts <= w))
        for b in range(0, len(sel), _SORT_ROWS):
            seg = sel[b:b + _SORT_ROWS]
            n = counts[seg]
            rows = np.repeat(np.arange(len(seg)), n)
            cols = np.arange(int(n.sum()), dtype=np.int64) - np.repeat(seg_starts(n), n)
            src = np.repeat(starts[seg], n) + cols
            pad = np.full((len(seg), int(n.max())), np.nan, vals.dtype)
            pad[rows, cols] = vals[src]
            pad.sort(axis=1)
            out[src] = pad[rows, cols]
        if w >= cmax:
            return out
        w *= 2


def _nan_counts(svals: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    cs = np.zeros(len(svals) + 1, np.int64)
    np.cumsum(np.isnan(svals), out=cs[1:])
    return cs[starts + counts] - cs[starts]


def seg_median(svals: np.ndarray, starts: np.ndarray, counts: np.ndarray, skipna: bool = False) -> np.ndarray:
    """(S,) float64 median of each segment of :func:`seg_sort` output.

    ``skipna=False`` is ``np.median`` (NaN when the segment holds a NaN); ``True`` is ``np.nanmedian``
    (median of the non-NaN values). Empty (or all-NaN) segments give NaN.
    """
    starts = np.asarray(starts, np.int64)
    counts = np.asarray(counts, np.int64)
    out = np.full(len(counts), np.nan)
    if skipna:
        m = counts - _nan_counts(svals, starts, counts)
        ok = m > 0
    else:
        m = counts
        ok = counts > 0
        ok[ok] = ~np.isnan(svals[starts[ok] + counts[ok] - 1])     # NaN sorts last
    st, m = starts[ok], m[ok]
    hi = svals[st + m // 2]
    lo = svals[st + (m - 1) // 2]
    # np.median: mean of the one or two middle values; (a + b) / 2 for an even count
    out[ok] = np.where(m % 2 == 1, hi, (lo + hi) / 2)
    return out


def seg_percentile(svals: np.ndarray, starts: np.ndarray, counts: np.ndarray, q: float) -> np.ndarray:
    """(S,) ``np.percentile(segment, q)`` ('linear') of each segment of :func:`seg_sort` output.

    NaN for an empty segment or one holding a NaN, as numpy. ``q`` is in percent.
    """
    starts = np.asarray(starts, np.int64)
    counts = np.asarray(counts, np.int64)
    out = np.full(len(counts), np.nan)
    ok = counts > 0
    ok[ok] = ~np.isnan(svals[starts[ok] + counts[ok] - 1])
    st, n = starts[ok], counts[ok]
    qf = np.true_divide(np.float64(q), 100)
    virtual = (n - 1) * qf                                     # numpy: (n - 1) * quantiles
    prev = np.floor(virtual)
    nxt = prev + 1
    above = virtual >= n - 1                                   # numpy: both indexes -> -1 (the last value)
    below = virtual < 0
    prev[above] = -1
    nxt[above] = -1
    prev[below] = 0
    nxt[below] = 0
    gamma = virtual - prev                                     # numpy uses the clamped previous index
    pi = prev.astype(np.int64)
    ni = nxt.astype(np.int64)
    pi = np.where(pi < 0, n + pi, pi)
    ni = np.where(ni < 0, n + ni, ni)
    a = svals[st + pi]
    b = svals[st + ni]
    diff = b - a                                               # numpy _lerp, both branches
    res = a + diff * gamma
    hi = gamma >= 0.5
    res[hi] = b[hi] - diff[hi] * (1 - gamma[hi])
    out[ok] = res
    return out


def seg_rowsum(vals: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """(S, k) ``vals[s0:s0+n].sum(0)`` of every segment of an (N, k) array, in the dtype of ``vals``.

    numpy reduces a C-contiguous (n, k) block over axis 0 row by row, starting from zero (no pairwise
    summation across rows); this repeats that order for all segments at once. Empty segments give 0.
    """
    vals = np.asarray(vals)
    starts = np.asarray(starts, np.int64)
    counts = np.asarray(counts, np.int64)
    out = np.zeros((len(counts),) + vals.shape[1:], vals.dtype)
    if not len(counts):
        return out
    cmax = int(counts.max())
    order = np.argsort(-counts, kind='stable')
    s_sorted = starts[order]
    c_sorted = counts[order]
    active = np.searchsorted(-c_sorted, -np.arange(cmax), side='left')      # segments longer than j
    acc = np.zeros_like(out)
    for j in range(cmax):
        k = active[j]
        acc[:k] += vals[s_sorted[:k] + j]
    out[order] = acc
    return out


def row_norm(v: np.ndarray) -> np.ndarray:
    """(N,) ``np.linalg.norm(v[i])`` for every row of a float64 (N, 2) array, bit-identical.

    ``np.linalg.norm`` of a 1-D vector is ``sqrt(x.dot(x))``; ``np.vecdot`` takes the same dot product
    path, whereas ``sqrt(x*x + y*y)`` can differ in the last bit where the BLAS dot fuses the multiply-add.
    """
    v = np.ascontiguousarray(v, np.float64)
    return np.sqrt(np.vecdot(v, v))
