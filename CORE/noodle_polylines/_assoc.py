# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Batched T4 association: the common-span shape distance over whole cost matrices, and an
array-state FusedTracker that runs the released legacy path one frame at a time.

Every number is the one :mod:`noodle_polylines._reference` computes pair by pair, bit for bit:

* ``common_span_dist`` (FT:84-99) samples both arms at ``linspace(0, lc, k)`` with ``lc = min(la, lb)``.
  The shorter arm (``lc == its own length``) is sampled on its own grid, which depends on that arm
  only, so it is computed once per arm (:func:`arm_rows`); only the longer arm is interpolated per
  pair. The sampled point is ``(1 - t) * p[i0] + t * p[i0 + 1]`` in float64 from float32 points, the
  point distance ``sqrt(dx*dx + dy*dy)`` (``np.linalg.norm(axis=1)`` is an unfused ``add.reduce`` of the
  squares) and the mean numpy's pairwise sum of 12 values along a contiguous last axis.
* The embedding distance ``float(np.linalg.norm(e_i - e_j))`` is ``sqrt`` of a BLAS ``sdot``; ``np.vecdot``
  on float32 rows calls the same dot routine for every pair, so it is used for the whole (n, m) block.
* Pairs whose shorter arm has length <= 1e-6 (single-point or zero-length arms) take the reference's
  tip-distance branch through the reference function itself.

The tracker keeps the active tracks as arrays in insertion order (which is ascending ID: tracks are
appended with increasing IDs and only ever removed), and calls ``linear_sum_assignment`` on exactly
the matrices the reference builds. It covers ``vel_alpha == 0`` (the released setting) with finite,
bounded float32 inputs; there a track's prediction is its last shape and the velocity stays zero.
Anything else is left to the reference tracker by the caller (:func:`fast_path_ok`).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import _reference as R

BIG = 1e9
_CHECK_BLOCK = 1 << 16
_BOUND = 1e18          # |coordinate| below this keeps every difference finite (zero velocity stays zero)


class Layout:
    """Column layout of the per-arm rows (float64), x and y in separate planes so that every
    elementwise step runs over contiguous (..., k) blocks: self sample Qx, Qy (k each), own grid G (k),
    degenerate flag (L <= 1e-6), points Sx, Sy (k each), max(L, 1e-6), L."""

    def __init__(self, k: int):
        self.k = k
        self.qx, self.qy, self.g = slice(0, k), slice(k, 2 * k), slice(2 * k, 3 * k)
        self.DEG = 3 * k
        self.sx, self.sy = slice(3 * k + 1, 4 * k + 1), slice(4 * k + 1, 5 * k + 1)
        self.LM, self.L = 5 * k + 1, 5 * k + 2
        self.width = 5 * k + 3


_LAYOUTS: dict[int, Layout] = {}
_PLANES: dict[int, np.ndarray] = {}


def layout(k: int) -> Layout:
    lay = _LAYOUTS.get(k)
    if lay is None:
        lay = _LAYOUTS[k] = Layout(k)
        _PLANES[k] = np.array([lay.sx.start, lay.sy.start]).reshape(2, 1, 1, 1)
    return lay


_AR = np.arange(256, dtype=np.int64)
_AR.flags.writeable = False


def _arange(n: int) -> np.ndarray:
    """``np.arange(n)`` (int64), a read-only view for small ``n``."""
    return _AR[:n] if n <= len(_AR) else np.arange(n, dtype=np.int64)


def _index(grid: np.ndarray, lmax: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``idx = s / max(total, 1e-6) * (k - 1)``; ``i0 = clip(int(idx), 0, k - 2)``; ``t = idx - i0``.

    Returns ``(i0, 1 - t, t)``. ``idx >= 0`` (grids and lengths are non-negative), so only the upper
    clip can bind."""
    idx = grid / lmax[..., None]
    idx *= k - 1
    i0 = idx.astype(np.int64)
    np.minimum(i0, k - 2, out=i0)
    t = idx - i0
    return i0, 1 - t, t


def _lerp(omt: np.ndarray, t: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """``(1 - t) * p[i0] + t * p[i0 + 1]`` (float64) for one coordinate."""
    out = omt * p0
    out += t * p1
    return out


def arm_rows(sampled: np.ndarray, lengths: np.ndarray, k: int | None = None) -> np.ndarray:
    """(N, 5k+3) float64 per-arm rows (see :class:`Layout`) of tip-first shapes ``sampled`` (N,k,2)."""
    sampled = np.asarray(sampled)
    n = len(sampled)
    k = sampled.shape[1] if k is None else k
    lay = layout(k)
    rows = np.empty((n, lay.width))
    S = sampled.reshape(n, k, 2)
    rows[:, lay.sx] = S[:, :, 0]
    rows[:, lay.sy] = S[:, :, 1]
    L = np.asarray(lengths).astype(np.float64).reshape(n)
    rows[:, lay.L] = L
    lm = np.maximum(L, 1e-6)
    rows[:, lay.LM] = lm
    rows[:, lay.DEG] = L <= 1e-6
    # np.linspace(0.0, L, k): arange(k) * (L / (k - 1)) (+ 0.0), last entry exactly L
    G = np.arange(k, dtype=np.float64)[None, :] * (L / (k - 1))[:, None]
    G[:, -1] = L
    rows[:, lay.g] = G
    # the self sample: each arm on its own grid
    i0, omt, t = _index(G, lm, k)
    i0 += (np.arange(n) * lay.width)[:, None]
    flat = rows.reshape(-1)
    for q, sl in ((lay.qx, lay.sx), (lay.qy, lay.sy)):
        f0 = i0 + sl.start
        rows[:, q] = _lerp(omt, t, flat[f0], flat[f0 + 1])
    return rows


def row_points(rows: np.ndarray, k: int) -> np.ndarray:
    """(N, k, 2) float32 shapes stored in ``rows`` (exact: they were float32)."""
    lay = layout(k)
    return np.stack((rows[:, lay.sx], rows[:, lay.sy]), axis=-1).astype(np.float32)


def span_dist_rows(Ra: np.ndarray, Rb: np.ndarray, k: int, deg_fn=None) -> np.ndarray:
    """T4.1 for every pair of rows ``Ra`` (n, W) x ``Rb`` (m, W): (n, m) float64.

    ``deg_fn(i, j)`` returns the reference value for a pair whose shorter arm is degenerate."""
    lay = layout(k)
    n, m = len(Ra), len(Rb)
    if n == 0 or m == 0:
        return np.zeros((n, m))
    U = np.concatenate((Ra, Rb))
    Lc = U[:, lay.L]
    a_short = Lc[:n, None] <= Lc[None, n:]                      # min(la, lb) is la unless lb < la
    ar = _arange(n + m)
    ia = ar[:n, None]
    jb = ar[None, n:]
    short = np.where(a_short, ia, jb)
    long_ = np.where(a_short, jb, ia)
    Rs = U[short]                                                 # (n, m, W) rows of the shorter arm
    # the longer arm sampled on the shorter arm's grid; x and y points gathered by flat index as one
    # (2, n, m, k) block (the x and y planes are k columns apart)
    i0, omt, t = _index(Rs[..., lay.g], U[:, lay.LM][long_], k)
    i0 += (long_ * lay.width)[..., None]
    f0 = i0[None] + _PLANES[k]
    Uf = U.reshape(-1)
    d = _lerp(omt, t, Uf.take(f0), Uf.take(f0 + 1))
    np.subtract(Rs[..., lay.qx.start:lay.qy.stop].reshape(n, m, 2, k).transpose(2, 0, 1, 3), d, out=d)
    d *= d
    dist = d[0]
    dist += d[1]                                                  # norm(axis=1): x*x + y*y, unfused
    np.sqrt(dist, out=dist)
    out = np.add.reduce(dist, axis=-1)                            # ndarray.mean: add.reduce, then / count
    out /= k
    if U[:, lay.DEG].any():
        deg = Rs[..., lay.DEG] != 0
        for i, j in zip(*np.nonzero(deg)):
            out[i, j] = deg_fn(int(i), int(j))
    return out


def embed_dist(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """(n, m) float64 ``float(np.linalg.norm(ea[i] - eb[j]))`` (BLAS dot through ``np.vecdot``)."""
    D = np.subtract(np.asarray(ea)[:, None, :], np.asarray(eb)[None, :, :])
    return np.sqrt(np.vecdot(D, D)).astype(np.float64)


def augmented(cost: np.ndarray, max_match_cost: float | None) -> np.ndarray:
    """The matrix the released tracker hands to ``linear_sum_assignment`` (FT:400-414)."""
    if max_match_cost is None:
        return cost
    n, m = cost.shape
    N = n + m
    half = max_match_cost / 2.0
    full = np.full((N, N), BIG, dtype=np.float64)
    full[:n, :m] = cost
    flat = full.reshape(-1)
    flat[m:n * N:N + 1] = half                  # full[i, m + i]
    flat[n * N:N * N:N + 1] = half              # full[n + j, j]
    full[n:, m:] = 0.0
    return full


# ------------------------------------------------------------------------------------- tracker
def fast_path_ok(config, sampled: np.ndarray, lengths: np.ndarray, embeds: np.ndarray, arms: np.ndarray) -> bool:
    """True when :class:`ArrayTracker` reproduces the reference on these arms: ``vel_alpha == 0``,
    float32 shapes and embeddings, and finite, bounded inputs (the reference's zero velocity then
    never changes and a prediction is the last shape)."""
    if config.vel_alpha != 0 or sampled.dtype != np.float32 or embeds.dtype != np.float32:
        return False
    if sampled.ndim != 3 or sampled.shape[2] != 2 or lengths.dtype.kind != 'f':
        return False
    for b in range(0, len(arms), _CHECK_BLOCK):      # blocks: no full-size temporaries
        a = arms[b:b + _CHECK_BLOCK]
        # |x| < bound is False for NaN and +-inf, so one comparison checks finiteness too
        if not (np.all(np.abs(sampled[a]) < _BOUND) and np.all(np.isfinite(lengths[a]))
                and np.all(np.isfinite(embeds[a]))):
            return False
    return True


class ArrayTracker:
    """The released FusedTracker (legacy path, ``vel_alpha == 0``) with array state.

    ``step(rows, embeds, polys32, lengths)`` takes the frame's arm rows (:func:`arm_rows`), float32
    embeddings, and float32 shapes / lengths (used only by the degenerate tip-distance branch), and
    returns the (n,) int64 labels the reference ``FusedTracker.step`` returns.
    """

    def __init__(self, config, k: int, assign_hook=None):
        if config.max_match_cost is not None and config.max_match_cost <= 0:
            raise ValueError("max_match_cost must be positive or None")
        self.cfg = config
        self.k = k
        self.lay = layout(k)
        self.hook = assign_hook
        self.tid = np.zeros(0, np.int64)
        self.rows = np.zeros((0, self.lay.width))
        self.emb = np.zeros((0, 0), np.float32)
        self.age = np.zeros(0, np.int64)
        self.next_id = 0

    # state as the reference exposes it -------------------------------------------------------
    def track_polys(self) -> np.ndarray:
        return row_points(self.rows, self.k)

    def track_lengths(self) -> np.ndarray:
        return self.rows[:, self.lay.L].copy()

    def _retire(self) -> None:
        if len(self.age) and self.age.max() > self.cfg.max_age:
            keep = self.age <= self.cfg.max_age
            self.tid, self.rows, self.emb, self.age = (self.tid[keep], self.rows[keep], self.emb[keep],
                                                       self.age[keep])

    def costs(self, rows: np.ndarray, embeds: np.ndarray, polys: np.ndarray, lengths: np.ndarray):
        """(shape, embedding distance, cost) blocks of the frame against the active tracks; the cost is
        ``shape + embed_weight * embedding distance``, or ``BIG`` where ``shape > shape_gate``."""
        lay, k = self.lay, self.k
        trows = self.rows

        def deg_fn(i, j):
            pb = row_points(trows[j:j + 1], k)[0]
            return R.common_span_dist(polys[i], float(lengths[i]), pb, float(trows[j, lay.L]))

        shape = span_dist_rows(rows, trows, k, deg_fn)
        ed = embed_dist(embeds, self.emb)
        cost = self.cfg.embed_weight * ed
        cost += shape
        cost[shape > self.cfg.shape_gate] = BIG
        return shape, ed, cost

    def step(self, rows: np.ndarray, embeds: np.ndarray, polys: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        n, m = len(rows), len(self.tid)
        cfg = self.cfg
        if n == 0:
            self.age = self.age + 1
            self._retire()
            return np.zeros(0, np.int64)
        if m == 0:
            ids = np.arange(self.next_id, self.next_id + n, dtype=np.int64)
            self.next_id += n
            self.tid, self.rows, self.age = ids, rows.copy(), np.zeros(n, np.int64)
            self.emb = np.array(embeds, np.float32)
            return ids.copy()
        _, ed, cost = self.costs(rows, embeds, polys, lengths)
        full = augmented(cost, cfg.max_match_cost)
        rr, cc = linear_sum_assignment(full)
        if self.hook is not None:
            self.hook(full, rr, cc)
        if cfg.max_match_cost is not None:
            # square matrix: row_ind is arange(n + m) (scipy's contract), so rows < n come first
            cc = cc[:n]
            rr = np.flatnonzero(cc < m)
            cc = cc[rr]
        ok = cost[rr, cc] < BIG                     # NaN never occurs on the fast path (finite inputs)
        if not ok.all():
            rr, cc = rr[ok], cc[ok]
        labels = np.full(n, -1, np.int64)
        labels[rr] = self.tid[cc]
        upd = ed[rr, cc] <= cfg.embed_update_gate
        cu, ru = cc[upd], rr[upd]
        a = cfg.embed_alpha
        self.emb[cu] = (1 - a) * self.emb[cu] + a * embeds[ru]
        self.rows[cc] = rows[rr]
        age = self.age + 1
        age[cc] = 0
        new = np.flatnonzero(labels < 0)
        if len(new):
            ids = np.arange(self.next_id, self.next_id + len(new), dtype=np.int64)
            self.next_id += len(new)
            labels[new] = ids
            self.tid = np.concatenate((self.tid, ids))
            self.rows = np.concatenate((self.rows, rows[new]))
            self.emb = np.concatenate((self.emb, embeds[new]))
            age = np.concatenate((age, np.zeros(len(new), np.int64)))
        self.age = age
        self._retire()
        return labels


def run(config, sampled: np.ndarray, lengths: np.ndarray, embeds: np.ndarray, fo: np.ndarray,
        order: np.ndarray, out: np.ndarray, assign_hook=None, chunk_arms: int = 1 << 15,
        tracker: ArrayTracker | None = None) -> ArrayTracker:
    """Step an :class:`ArrayTracker` through frame positions ``order`` (arms ``fo[f]:fo[f+1]``) and
    write the labels into ``out``. Arm rows are built for blocks of about ``chunk_arms`` arms (whole
    frames, at least one), so the extra memory is bounded by the block (~500 B per arm), not the
    sequence."""
    k = sampled.shape[1]
    tr = ArrayTracker(config, k, assign_hook) if tracker is None else tracker
    order = np.asarray(order, np.int64)
    a0_all, a1_all = fo[order], fo[order + 1]
    cum = np.cumsum(a1_all - a0_all)
    b = 0
    while b < len(order):
        before = int(cum[b - 1]) if b else 0
        e = max(int(np.searchsorted(cum, before + chunk_arms, side='right')), b + 1)
        a0, a1 = a0_all[b:e], a1_all[b:e]
        cnt = a1 - a0
        offs = np.zeros(e - b + 1, np.int64)
        np.cumsum(cnt, out=offs[1:])
        arms = np.repeat(a0 - offs[:-1], cnt) + np.arange(int(offs[-1]), dtype=np.int64)
        rows = arm_rows(sampled[arms], lengths[arms], k)
        for q in range(e - b):
            s0, s1 = int(offs[q]), int(offs[q + 1])
            g0, g1 = int(a0[q]), int(a1[q])
            out[g0:g1] = tr.step(rows[s0:s1], embeds[g0:g1], sampled[g0:g1], lengths[g0:g1])
        b = e
    return tr
