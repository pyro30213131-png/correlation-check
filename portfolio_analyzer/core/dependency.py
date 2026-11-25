"""
core/dependency.py
──────────────────
Non-linear dependency and tail-risk analysis.

  1. Distance Correlation (dCor)   — Székely, Rizzo & Bakirov (2007)
  2. Student-t Copula tail dep     — symmetric λ_L = λ_U
  3. Effective rank                — entropy of eigenvalue spectrum
  4. Diversification Ratio         — (wᵀσ) / √(wᵀΣw)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.stats import t as t_dist
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  1. Distance Correlation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DCovResult:
    dcor:         float    # ∈ [0, 1]
    dcov_sq:      float
    dvar_x_sq:    float
    dvar_y_sq:    float
    pearson:      float    # linear Pearson ρ
    nonlinearity: float    # dCor − |ρ|  > 0 means hidden nonlinear dep


def _double_centre(D: np.ndarray) -> np.ndarray:
    """Doubly-centred distance matrix: A_ij = D_ij − D̄_i· − D̄_·j + D̄_··"""
    row = D.mean(axis=1, keepdims=True)
    col = D.mean(axis=0, keepdims=True)
    return D - row - col + D.mean()


def distance_covariance(x: np.ndarray, y: np.ndarray) -> DCovResult:
    """
    Compute distance correlation between two univariate return series.

    dCor = 0  ⟺  X ⊥ Y  (independence, not just linearity)

    Parameters
    ----------
    x, y : np.ndarray, shape (T,)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: {len(x)} vs {len(y)}.")
    if len(x) < 4:
        raise ValueError("Need ≥ 4 observations.")

    n  = len(x)
    Dx = np.abs(x[:, None] - x[None, :])
    Dy = np.abs(y[:, None] - y[None, :])

    Ax = _double_centre(Dx)
    Ay = _double_centre(Dy)

    dcov_sq = float((Ax * Ay).mean())
    dvar_x  = float((Ax * Ax).mean())
    dvar_y  = float((Ay * Ay).mean())

    denom = np.sqrt(max(dvar_x * dvar_y, 0.0))
    dcor  = float(np.sqrt(max(dcov_sq / denom, 0.0))) if denom > 1e-12 else 0.0

    pearson = float(np.corrcoef(x, y)[0, 1])

    return DCovResult(
        dcor         = dcor,
        dcov_sq      = dcov_sq,
        dvar_x_sq    = dvar_x,
        dvar_y_sq    = dvar_y,
        pearson      = pearson,
        nonlinearity = dcor - abs(pearson),
    )


def distance_correlation_matrix(
    returns:    np.ndarray,
    max_assets: int = 50,
) -> np.ndarray:
    """
    Compute N×N distance correlation matrix.

    O(N² · T²) — expensive. Capped at max_assets columns by default.
    Pass max_assets=None to run on the full universe (slow for N > 100).

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    max_assets : int or None
    """
    returns = np.asarray(returns, dtype=float)
    T, N    = returns.shape

    if max_assets is not None:
        N = min(N, max_assets)
        returns = returns[:, :N]

    mat = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            r = distance_covariance(returns[:, i], returns[:, j])
            mat[i, j] = r.dcor
            mat[j, i] = r.dcor
    return mat


# ─────────────────────────────────────────────────────────────────────────────
#  2. Student-t Copula Tail Dependence
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TailDepResult:
    rho:                   float
    nu:                    float
    lambda_lower:          float    # λ_L = λ_U for t-copula
    lambda_upper:          float
    joint_crash_prob_1pct: float    # ≈ λ · 0.01 (first-order)
    excess_vs_gaussian:    float    # λ_t − 0  (Gaussian baseline = 0)


def t_copula_tail_dependence(rho: float, nu: float = 5.0) -> TailDepResult:
    """
    Symmetric tail dependence for a bivariate Student-t copula.

    λ = 2 · t_{ν+1}( −√((ν+1)(1−ρ)/(1+ρ)) )

    Parameters
    ----------
    rho : float  ∈ (−1, 1)
    nu  : float  degrees of freedom (ν = 5 → fat tails; ν → ∞ → Gaussian)
    """
    rho = float(np.clip(rho, -0.9999, 0.9999))
    nu  = float(max(nu, 2.001))

    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    lam = float(np.clip(2.0 * t_dist.cdf(arg, df=nu + 1.0), 0.0, 1.0))

    return TailDepResult(
        rho                   = rho,
        nu                    = nu,
        lambda_lower          = lam,
        lambda_upper          = lam,
        joint_crash_prob_1pct = lam * 0.01,
        excess_vs_gaussian    = lam,
    )


def tail_dependence_matrix(
    correlation: np.ndarray,
    nu:          float = 5.0,
) -> np.ndarray:
    """
    Build N×N tail-dependence matrix from an RMT-filtered correlation matrix.

    Parameters
    ----------
    correlation : np.ndarray, shape (N, N)
    nu : float  degrees of freedom for the t-copula
    """
    C = np.asarray(correlation, dtype=float)
    N = C.shape[0]
    L = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            lam        = t_copula_tail_dependence(C[i, j], nu).lambda_lower
            L[i, j]    = lam
            L[j, i]    = lam
    return L


# ─────────────────────────────────────────────────────────────────────────────
#  3. Effective Rank  (true diversification metric)
# ─────────────────────────────────────────────────────────────────────────────

def effective_rank(correlation: np.ndarray) -> float:
    """
    Effective rank = exp( H(p) )   where p_i = λ_i / Σλ_j

    Range: [1, N]
      N → fully uncorrelated portfolio (maximum diversification)
      1 → single-factor portfolio (zero diversification)

    Parameters
    ----------
    correlation : np.ndarray, shape (N, N)
    """
    evals = np.linalg.eigvalsh(np.asarray(correlation, dtype=float))
    evals = np.maximum(evals, 0.0)
    total = evals.sum()
    if total < 1e-12:
        return 1.0
    p       = evals / total
    p       = p[p > 1e-12]
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


# ─────────────────────────────────────────────────────────────────────────────
#  4. Diversification Ratio
# ─────────────────────────────────────────────────────────────────────────────

def diversification_ratio(
    weights:      np.ndarray,
    volatilities: np.ndarray,
    covariance:   np.ndarray,
) -> float:
    """
    DR = (wᵀσ) / √(wᵀΣw)

    = 1   → no diversification benefit
    > 1   → portfolio vol < weighted-average vol  (diversification gain)

    Parameters
    ----------
    weights      : (N,)  must sum to 1
    volatilities : (N,)  individual annualised σ
    covariance   : (N, N)
    """
    w   = np.asarray(weights,      dtype=float)
    sig = np.asarray(volatilities,  dtype=float)
    cov = np.asarray(covariance,   dtype=float)

    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(max(port_var, 1e-12)))
    wtd_avg  = float(w @ sig)
    return float(wtd_avg / port_vol)
