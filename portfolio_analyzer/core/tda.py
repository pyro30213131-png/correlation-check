"""
core/tda.py
───────────
Topological Data Analysis for portfolio structural monitoring.

Distance metric:  d(i,j) = 1 − |ρ_{ij}|    ∈ [0, 1]

H₀  Connected components (clusters)
    Built via Kruskal's algorithm on the correlation-distance graph.
    Birth = 0, Death = ε at which the cluster merges.

H₁  Loops / cycles (approximate)
    Random-triangle sampling — exact boundary-reduction is O(N³)
    which is infeasible at N = 300.

Regime change:
    Wasserstein-1 distance between consecutive persistence diagrams.
    Spike → market topology has shifted.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BarPoint:
    birth:     float
    death:     float
    dimension: int

    @property
    def persistence(self) -> float:
        return self.death - self.birth


@dataclass
class PersistenceDiagram:
    bars:      list[BarPoint]
    dimension: int

    def to_array(self) -> np.ndarray:
        if not self.bars:
            return np.empty((0, 2), dtype=float)
        return np.array([[b.birth, b.death] for b in self.bars], dtype=float)

    @property
    def betti(self) -> int:
        """Number of features with persistence above median."""
        if not self.bars:
            return 0
        pvals = [b.persistence for b in self.bars]
        med   = float(np.median(pvals))
        return sum(1 for p in pvals if p > med)

    @property
    def max_persistence(self) -> float:
        return max((b.persistence for b in self.bars), default=0.0)


@dataclass
class TDAResult:
    h0:               PersistenceDiagram
    h1:               PersistenceDiagram
    dist_matrix:      np.ndarray
    n_assets:         int
    betti_0:          int
    betti_1:          int
    max_persist_h0:   float
    max_persist_h1:   float
    interpretation:   str
    topology_summary: dict


# ─────────────────────────────────────────────────────────────────────────────
#  Union-Find  (for H₀ via Kruskal)
# ─────────────────────────────────────────────────────────────────────────────

class _UF:
    def __init__(self, n: int):
        self.p    = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  H₀ — connected components
# ─────────────────────────────────────────────────────────────────────────────

def _h0_bars(dist: np.ndarray) -> PersistenceDiagram:
    N = dist.shape[0]
    edges = sorted(
        ((dist[i, j], i, j) for i in range(N) for j in range(i + 1, N)),
        key=lambda e: e[0],
    )
    uf   = _UF(N)
    bars = []
    for d, i, j in edges:
        if uf.union(i, j):
            bars.append(BarPoint(birth=0.0, death=float(d), dimension=0))
    return PersistenceDiagram(bars=bars, dimension=0)


# ─────────────────────────────────────────────────────────────────────────────
#  H₁ — cycles (approximate via triangle sampling)
# ─────────────────────────────────────────────────────────────────────────────

def _h1_bars(
    dist:      np.ndarray,
    n_samples: int = 600,
    max_edge:  float = 0.85,
    seed:      int = 42,
) -> PersistenceDiagram:
    N   = dist.shape[0]
    rng = np.random.default_rng(seed)
    seen: set[tuple] = set()
    bars: list[BarPoint] = []

    for _ in range(n_samples):
        i, j, k = rng.choice(N, size=3, replace=False)
        key = tuple(sorted([int(i), int(j), int(k)]))
        if key in seen:
            continue
        seen.add(key)

        d = [dist[i, j], dist[j, k], dist[i, k]]
        d.sort()   # d[0] ≤ d[1] ≤ d[2]

        if d[2] > max_edge:
            continue

        birth = d[1]     # triangle opens when 2nd edge appears
        death = d[2]     # triangle closes when longest edge appears

        if death - birth > 1e-6:
            bars.append(BarPoint(birth=float(birth), death=float(death), dimension=1))

    bars.sort(key=lambda b: b.persistence, reverse=True)
    return PersistenceDiagram(bars=bars, dimension=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Main TDA entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_tda(correlation: np.ndarray) -> TDAResult:
    """
    Run the full TDA pipeline on an RMT-filtered correlation matrix.

    Parameters
    ----------
    correlation : np.ndarray, shape (N, N)
    """
    C = np.asarray(correlation, dtype=float)
    N = C.shape[0]

    dist = np.clip(1.0 - np.abs(C), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    h0 = _h0_bars(dist)
    h1 = _h1_bars(dist)

    interp = _interpret(h0.betti, h1.betti, h1.max_persistence)

    summary = {
        "n_h0_bars":        len(h0.bars),
        "n_h1_bars":        len(h1.bars),
        "betti_0":          h0.betti,
        "betti_1":          h1.betti,
        "max_h0_persist":   round(h0.max_persistence, 4),
        "max_h1_persist":   round(h1.max_persistence, 4),
        "mean_merge_dist":  round(float(np.mean([b.death for b in h0.bars])), 4)
                            if h0.bars else None,
        "interpretation":   interp,
    }

    return TDAResult(
        h0             = h0,
        h1             = h1,
        dist_matrix    = dist,
        n_assets       = N,
        betti_0        = h0.betti,
        betti_1        = h1.betti,
        max_persist_h0 = h0.max_persistence,
        max_persist_h1 = h1.max_persistence,
        interpretation = interp,
        topology_summary = summary,
    )


def _interpret(b0: int, b1: int, max_h1: float) -> str:
    if b0 <= 2 and b1 <= 1:
        return "Stable — low topological complexity."
    if b1 >= 5 or max_h1 > 0.4:
        return "Elevated — loop structure detected; possible contagion."
    if b0 >= 8:
        return "Fragmented — many clusters; diverse but potentially unstable."
    return "Moderate — monitor for regime shift."


# ─────────────────────────────────────────────────────────────────────────────
#  Wasserstein-1 distance between persistence diagrams
# ─────────────────────────────────────────────────────────────────────────────

def wasserstein1(d1: PersistenceDiagram, d2: PersistenceDiagram) -> float:
    """
    Earth-mover distance between two persistence diagrams.
    Points unpaired to diagram get matched to their diagonal projection.
    """
    pts1 = d1.to_array()
    pts2 = d2.to_array()

    if len(pts1) == 0 and len(pts2) == 0:
        return 0.0

    def diag_proj(pts: np.ndarray) -> np.ndarray:
        mid = pts.mean(axis=1, keepdims=True)
        return np.hstack([mid, mid])

    aug1 = np.vstack([pts1, diag_proj(pts2)]) if len(pts2) > 0 else pts1
    aug2 = np.vstack([pts2, diag_proj(pts1)]) if len(pts1) > 0 else pts2

    n = max(len(aug1), len(aug2))
    pad = np.zeros((1, 2))
    while len(aug1) < n:
        aug1 = np.vstack([aug1, pad])
    while len(aug2) < n:
        aug2 = np.vstack([aug2, pad])

    cost = np.linalg.norm(aug1[:, None, :] - aug2[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].sum())


# ─────────────────────────────────────────────────────────────────────────────
#  Rolling regime-change timeline
# ─────────────────────────────────────────────────────────────────────────────

def rolling_topology(
    returns: np.ndarray,
    window:  int = 60,
    step:    int = 10,
) -> list[dict]:
    """
    Compute rolling TDA snapshots and flag regime changes.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    window  : rolling window in trading days
    step    : step size between windows

    Returns
    -------
    list of dicts — one per window — with betti numbers, Wasserstein
    distances, and a boolean `regime_alert`.
    """
    from .covariance import ewma_covariance, Horizon
    from .rmt_filter import rmt_filter

    T = returns.shape[0]
    out: list[dict] = []
    prev_h0: Optional[PersistenceDiagram] = None
    prev_h1: Optional[PersistenceDiagram] = None

    for start in range(0, T - window + 1, step):
        end = start + window
        r   = returns[start:end]

        ewma = ewma_covariance(r, Horizon.TACTICAL, shrink=True)
        rmt  = rmt_filter(ewma.correlation, T=window)
        tda  = compute_tda(rmt.filtered_correlation)

        w0 = wasserstein1(prev_h0, tda.h0) if prev_h0 else 0.0
        w1 = wasserstein1(prev_h1, tda.h1) if prev_h1 else 0.0

        alert = (w0 > 2.0) or (w1 > 1.0) or (tda.betti_1 >= 5)

        out.append({
            "t_start":            start,
            "t_end":              end,
            "betti_0":            tda.betti_0,
            "betti_1":            tda.betti_1,
            "wasserstein_h0":     round(w0, 4),
            "wasserstein_h1":     round(w1, 4),
            "max_persist_h0":     round(tda.max_persist_h0, 4),
            "max_persist_h1":     round(tda.max_persist_h1, 4),
            "regime_alert":       bool(alert),
            "interpretation":     tda.interpretation,
        })

        prev_h0 = tda.h0
        prev_h1 = tda.h1

    return out
