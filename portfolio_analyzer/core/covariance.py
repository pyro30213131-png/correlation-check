"""
core/covariance.py
──────────────────
EWMA covariance and correlation engine.

Bug fixes vs previous version
──────────────────────────────
- avg_correlation now computed ONLY on upper triangle (off-diagonal),
  never on the full matrix (which inflates/deflates with diagonal 1s).
- Shrinkage target uses trace(S)/N not arbitrary mu.
- Effective observations formula corrected to 1/(1-λ²).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Horizon(str, Enum):
    TACTICAL = "tactical"   # 60-day,  λ = 0.94
    CYCLICAL = "cyclical"   # 1-year,  λ = 0.97
    MACRO    = "macro"      # 5-year,  λ = 0.99


LAMBDA_MAP: dict[Horizon, float] = {
    Horizon.TACTICAL: 0.94,
    Horizon.CYCLICAL: 0.97,
    Horizon.MACRO:    0.99,
}

WINDOW_MAP: dict[Horizon, int] = {
    Horizon.TACTICAL: 60,
    Horizon.CYCLICAL: 252,
    Horizon.MACRO:    1260,
}


@dataclass
class EWMAResult:
    covariance:       np.ndarray   # (N, N)  PSD + shrinkage
    correlation:      np.ndarray   # (N, N)  diagonal = 1 exactly
    volatilities:     np.ndarray   # (N,)    annualised σ
    lambda_:          float
    horizon:          Horizon
    n_assets:         int
    n_obs:            int
    effective_obs:    float        # 1/(1-λ²)
    shrinkage_alpha:  float        # Ledoit-Wolf intensity
    # ── Correct off-diagonal average ──────────────────────────────────────
    avg_correlation:  float        # mean of upper-triangle only (NO diagonal)


def ewma_covariance(
    returns:     np.ndarray,
    horizon:     Horizon = Horizon.TACTICAL,
    lambda_:     Optional[float] = None,
    shrink:      bool = True,
    min_periods: int = 20,
) -> EWMAResult:
    """
    Compute EWMA covariance and correlation for a returns matrix.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Log-returns. Rows = time (oldest first), columns = assets.
    horizon : Horizon
        Selects the default λ.
    lambda_ : float, optional
        Override the decay constant.
    shrink : bool
        Apply Ledoit-Wolf shrinkage toward scaled identity.
    min_periods : int
        Minimum rows required.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns[:, None]

    T, N = returns.shape
    if T < min_periods:
        raise ValueError(f"Need ≥ {min_periods} observations, got {T}.")

    lam = lambda_ if lambda_ is not None else LAMBDA_MAP[horizon]
    if not (0.0 < lam < 1.0):
        raise ValueError(f"λ must be in (0,1), got {lam}.")

    # ── Exponential weights: w_t = (1-λ)·λ^{T-1-t}, normalised ──────────
    t_idx   = np.arange(T)
    weights = (1.0 - lam) * lam ** (T - 1 - t_idx)
    weights /= weights.sum()

    # ── Weighted mean ─────────────────────────────────────────────────────
    mu       = (weights[:, None] * returns).sum(axis=0)   # (N,)
    demeaned = returns - mu                                # (T, N)

    # ── Weighted outer-product: Cov = (√w · X)ᵀ (√w · X) ────────────────
    scaled  = demeaned * np.sqrt(weights)[:, None]
    cov_raw = scaled.T @ scaled                            # (N, N)  guaranteed PSD

    # ── Ledoit-Wolf shrinkage ─────────────────────────────────────────────
    alpha = 0.0
    if shrink:
        cov_raw, alpha = _ledoit_wolf(cov_raw, T)

    # ── Correlation matrix ────────────────────────────────────────────────
    std       = np.sqrt(np.diag(cov_raw))
    std_safe  = np.where(std > 1e-12, std, 1.0)
    corr      = cov_raw / np.outer(std_safe, std_safe)
    np.fill_diagonal(corr, 1.0)
    corr      = np.clip(corr, -1.0, 1.0)

    # ── Average correlation: UPPER TRIANGLE ONLY (i < j), no diagonal ────
    i_upper, j_upper = np.triu_indices(N, k=1)
    avg_corr = float(corr[i_upper, j_upper].mean()) if len(i_upper) > 0 else 0.0

    return EWMAResult(
        covariance      = cov_raw,
        correlation     = corr,
        volatilities    = std_safe * np.sqrt(252),
        lambda_         = lam,
        horizon         = horizon,
        n_assets        = N,
        n_obs           = T,
        effective_obs   = 1.0 / (1.0 - lam ** 2),
        shrinkage_alpha = alpha,
        avg_correlation = avg_corr,
    )


def _ledoit_wolf(cov: np.ndarray, T: int) -> tuple[np.ndarray, float]:
    """
    Analytical Ledoit-Wolf shrinkage toward μ·I  (μ = trace(S)/N).

    α* = min(1, max(0,  ‖S - F‖²_F / (‖S‖²_F · T) ))
    """
    N  = cov.shape[0]
    mu = np.trace(cov) / N
    F  = mu * np.eye(N)

    delta = np.linalg.norm(cov - F, "fro") ** 2
    gamma = np.linalg.norm(cov,     "fro") ** 2

    if gamma < 1e-12:
        return cov, 0.0

    alpha    = float(np.clip(delta / (gamma * T), 0.0, 1.0))
    shrunken = (1.0 - alpha) * cov + alpha * F
    return shrunken, alpha


def multi_horizon_covariance(
    returns:  np.ndarray,
    horizons: Optional[list[Horizon]] = None,
    **kwargs,
) -> dict[Horizon, EWMAResult]:
    """
    Compute EWMA covariance for multiple horizons in one call.
    Each horizon uses its canonical window (60 / 252 / 1260 days).
    """
    if horizons is None:
        horizons = list(Horizon)

    T = returns.shape[0]
    return {
        h: ewma_covariance(returns[-min(WINDOW_MAP[h], T):], horizon=h, **kwargs)
        for h in horizons
    }
