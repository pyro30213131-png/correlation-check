"""
core/analyzer.py
────────────────
High-level orchestrator.  Wires together all modules.

Bug fixes
─────────
- avg_correlation uses upper-triangle only  (np.triu_indices(N, k=1))
- avg_tail_dep same fix applied
- summary() rounds to 4dp but does NOT multiply by 100 — caller decides
- diversification_pct capped to 100%
- pair_analysis passes correct window slice to rmt_filter
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .covariance  import ewma_covariance, multi_horizon_covariance, Horizon, EWMAResult, WINDOW_MAP
from .rmt_filter  import rmt_filter, pseudoinverse_correlation, RMTResult
from .dependency  import (
    distance_covariance, distance_correlation_matrix,
    t_copula_tail_dependence, tail_dependence_matrix,
    effective_rank, diversification_ratio,
)
from .tda import compute_tda, rolling_topology, TDAResult


# ─────────────────────────────────────────────────────────────────────────────
#  Per-horizon container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HorizonReport:
    horizon:                  Horizon
    ewma:                     EWMAResult
    rmt:                      RMTResult
    tda:                      TDAResult
    tail_dep_matrix:          np.ndarray   # (N, N)
    effective_rank:           float
    # ── Correct averages (off-diagonal only) ──────────────────────────────
    avg_correlation:          float        # mean of upper triangle of filtered corr
    avg_tail_dep:             float        # mean of upper triangle of tail dep matrix
    # ── Top pairs ─────────────────────────────────────────────────────────
    top_corr_pairs:           list[dict]   # 10 most correlated
    top_tail_pairs:           list[dict]   # 10 highest tail dependence
    diversification_ratio_ew: float        # equally-weighted portfolio


# ─────────────────────────────────────────────────────────────────────────────
#  Full report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FullReport:
    tickers:         list[str]
    n_assets:        int
    n_obs:           int
    horizons:        dict[str, HorizonReport]    # key = Horizon.value string
    regime_timeline: list[dict]
    dcor_matrix:     Optional[np.ndarray]
    timestamp:       float = field(default_factory=time.time)

    def summary(self) -> dict:
        """JSON-serialisable summary dict — all values are plain Python types."""
        out: dict = {
            "n_assets": self.n_assets,
            "n_obs":    self.n_obs,
            "horizons": {},
        }
        for k, h in self.horizons.items():
            out["horizons"][k] = {
                # Diversification
                "effective_rank":       round(h.effective_rank, 3),
                "diversification_pct":  round(
                    min(h.effective_rank / self.n_assets * 100.0, 100.0), 2
                ),
                "diversification_ratio_ew": round(h.diversification_ratio_ew, 4),

                # Correlation (fraction, NOT percent — fix for "0%" bug)
                "avg_correlation":      round(h.avg_correlation, 4),  # e.g. 0.3142
                "avg_tail_dependence":  round(h.avg_tail_dep,    4),

                # RMT
                "signal_factors":       h.rmt.n_signal,
                "noise_factors":        h.rmt.n_noise,
                "lambda_max_mp":        round(h.rmt.lambda_max,       4),
                "lambda_min_mp":        round(h.rmt.lambda_min,       4),
                "signal_var_frac":      round(h.rmt.signal_var_frac,  4),
                "condition_number":     round(h.rmt.condition_number, 2),
                "shrinkage_alpha":      round(h.ewma.shrinkage_alpha, 4),

                # Topology
                "betti_0":              h.tda.betti_0,
                "betti_1":              h.tda.betti_1,
                "topology_alert":       h.tda.interpretation,
                "max_persist_h0":       round(h.tda.max_persist_h0, 4),
                "max_persist_h1":       round(h.tda.max_persist_h1, 4),
            }

        out["regime_changes_detected"] = sum(
            1 for r in self.regime_timeline if r.get("regime_alert", False)
        )
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main analyser class
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioAnalyzer:
    """
    Full quantitative correlation and diversity analyser.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Daily LOG-returns.  T = trading days (oldest first), N = assets.
        → Use core.data_loader to produce this from raw prices.
    tickers : list[str]
        N ticker symbols — same order as columns in `returns`.
    nu : float
        Student-t copula degrees of freedom (default 5 = fat tails).

    Example
    -------
        from core.data_loader import load_from_csv
        from core.analyzer import PortfolioAnalyzer

        data     = load_from_csv("prices.csv")
        analyzer = PortfolioAnalyzer(data.returns, data.tickers)
        report   = analyzer.full_report()
        print(report.summary())
    """

    MAX_ASSETS = 300

    def __init__(
        self,
        returns: np.ndarray,
        tickers: list[str],
        nu:      float = 5.0,
    ) -> None:
        self.returns = np.asarray(returns, dtype=float)
        self.tickers = list(tickers)
        self.nu      = nu

        T, N = self.returns.shape

        if N > self.MAX_ASSETS:
            raise ValueError(f"Universe {N} > {self.MAX_ASSETS} asset limit.")
        if len(self.tickers) != N:
            raise ValueError(
                f"tickers length {len(self.tickers)} ≠ return columns {N}."
            )
        if T < 20:
            raise ValueError(f"Need ≥ 20 observations, got {T}.")

        self.T = T
        self.N = N

    # ── Single horizon ────────────────────────────────────────────────────

    def analyse_horizon(self, horizon: Horizon) -> HorizonReport:
        win   = WINDOW_MAP[horizon]
        r_win = self.returns[-min(win, self.T):]
        T_win = len(r_win)

        # 1. EWMA covariance
        ewma = ewma_covariance(r_win, horizon=horizon, shrink=True)

        # 2. RMT filter
        rmt = rmt_filter(ewma.correlation, T=T_win)

        # 3. Tail dependence matrix
        tdm = tail_dependence_matrix(rmt.filtered_correlation, nu=self.nu)

        # 4. TDA
        tda = compute_tda(rmt.filtered_correlation)

        # 5. Effective rank
        eff = effective_rank(rmt.filtered_correlation)

        # 6. ── Correct average: upper triangle only, k=1 skips diagonal ──
        ui, uj = np.triu_indices(self.N, k=1)
        avg_corr = float(rmt.filtered_correlation[ui, uj].mean())
        avg_tail = float(tdm[ui, uj].mean())

        # 7. Equally-weighted diversification ratio
        ew  = np.full(self.N, 1.0 / self.N)
        dr  = diversification_ratio(ew, ewma.volatilities, ewma.covariance)

        # 8. Top pairs
        top_corr = self._top_pairs(rmt.filtered_correlation, "correlation")
        top_tail = self._top_pairs(tdm, "tail_dependence")

        return HorizonReport(
            horizon                  = horizon,
            ewma                     = ewma,
            rmt                      = rmt,
            tda                      = tda,
            tail_dep_matrix          = tdm,
            effective_rank           = eff,
            avg_correlation          = avg_corr,
            avg_tail_dep             = avg_tail,
            top_corr_pairs           = top_corr,
            top_tail_pairs           = top_tail,
            diversification_ratio_ew = dr,
        )

    def _top_pairs(
        self,
        matrix:     np.ndarray,
        value_name: str,
        n:          int = 10,
    ) -> list[dict]:
        """Return n largest off-diagonal pairs from a symmetric matrix."""
        ui, uj = np.triu_indices(self.N, k=1)
        vals   = matrix[ui, uj]
        # argsort descending
        order  = np.argsort(vals)[::-1][:n]
        return [
            {
                "asset_a":  self.tickers[ui[idx]],
                "asset_b":  self.tickers[uj[idx]],
                "idx_a":    int(ui[idx]),
                "idx_b":    int(uj[idx]),
                value_name: round(float(vals[idx]), 5),
            }
            for idx in order
        ]

    # ── Full multi-horizon report ─────────────────────────────────────────

    def full_report(
        self,
        horizons:       Optional[list[Horizon]] = None,
        include_dcor:   bool = False,
        regime_window:  int  = 60,
        regime_step:    int  = 10,
    ) -> FullReport:
        """
        Run the complete analysis.

        Parameters
        ----------
        horizons : list of Horizon, default all three
        include_dcor : bool
            Compute the N×N distance-correlation matrix.
            O(N²·T²) — slow for N > 60.  Default False.
        regime_window, regime_step : rolling TDA parameters
        """
        if horizons is None:
            horizons = list(Horizon)

        h_reports: dict[str, HorizonReport] = {
            h.value: self.analyse_horizon(h) for h in horizons
        }

        regime = rolling_topology(
            self.returns,
            window = regime_window,
            step   = regime_step,
        )

        dcor_mat = (
            distance_correlation_matrix(self.returns)
            if include_dcor else None
        )

        return FullReport(
            tickers         = self.tickers,
            n_assets        = self.N,
            n_obs           = self.T,
            horizons        = h_reports,
            regime_timeline = regime,
            dcor_matrix     = dcor_mat,
        )

    # ── Single-pair deep dive ─────────────────────────────────────────────

    def pair_analysis(
        self,
        ticker_a: str,
        ticker_b: str,
        horizon:  Horizon = Horizon.TACTICAL,
    ) -> dict:
        """
        Full dependency analysis for a single asset pair.

        Returns
        -------
        dict with Pearson, EWMA correlation, RMT-filtered correlation,
        dCor, nonlinearity excess, tail-dependence λ, joint crash prob.
        """
        missing = [t for t in (ticker_a, ticker_b) if t not in self.tickers]
        if missing:
            raise ValueError(f"Unknown tickers: {missing}")

        i = self.tickers.index(ticker_a)
        j = self.tickers.index(ticker_b)

        win   = WINDOW_MAP[horizon]
        r_win = self.returns[-min(win, self.T):]

        ewma = ewma_covariance(r_win, horizon=horizon, shrink=True)
        rmt  = rmt_filter(ewma.correlation, T=len(r_win))
        dcov = distance_covariance(r_win[:, i], r_win[:, j])
        tail = t_copula_tail_dependence(rmt.filtered_correlation[i, j], nu=self.nu)

        return {
            "asset_a":                  ticker_a,
            "asset_b":                  ticker_b,
            "horizon":                  horizon.value,
            "pearson_raw":              round(dcov.pearson,                         5),
            "ewma_correlation":         round(float(ewma.correlation[i, j]),        5),
            "rmt_filtered_correlation": round(float(rmt.filtered_correlation[i, j]),5),
            "distance_correlation":     round(dcov.dcor,                            5),
            "nonlinearity_excess":      round(dcov.nonlinearity,                    5),
            "tail_dependence_lambda":   round(tail.lambda_lower,                    5),
            "joint_crash_prob_1pct":    round(tail.joint_crash_prob_1pct,           6),
            "copula_nu":                self.nu,
        }

    # ── Convenience: get just the correlation matrix for one horizon ──────

    def correlation_matrix(
        self,
        horizon: Horizon = Horizon.TACTICAL,
        filtered: bool = True,
    ) -> np.ndarray:
        """
        Return the (N, N) correlation matrix for one horizon.

        Parameters
        ----------
        filtered : bool
            If True, returns the RMT-filtered matrix.
            If False, returns the raw EWMA correlation matrix.
        """
        win   = WINDOW_MAP[horizon]
        r_win = self.returns[-min(win, self.T):]
        ewma  = ewma_covariance(r_win, horizon=horizon, shrink=True)
        if not filtered:
            return ewma.correlation
        rmt = rmt_filter(ewma.correlation, T=len(r_win))
        return rmt.filtered_correlation
