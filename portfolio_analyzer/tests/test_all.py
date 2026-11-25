"""
tests/test_all.py
─────────────────
Full test suite — runs without any external data.
Synthetic returns are generated inside each test.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.covariance import ewma_covariance, multi_horizon_covariance, Horizon
from core.rmt_filter import rmt_filter, marchenko_pastur_pdf
from core.dependency import (
    distance_covariance, distance_correlation_matrix,
    t_copula_tail_dependence, tail_dependence_matrix,
    effective_rank, diversification_ratio,
)
from core.tda     import compute_tda, wasserstein1, rolling_topology
from core.analyzer import PortfolioAnalyzer


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_returns(T=120, N=10, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 2-factor model for realistic correlation structure
    F  = rng.standard_normal((T, 2)) * 0.01
    B  = rng.uniform(-0.5, 0.5, (N, 2))
    e  = rng.standard_normal((T, N)) * 0.005
    return F @ B.T + e   # (T, N)


def make_tickers(N=10):
    return [f"STOCK_{i:02d}" for i in range(N)]


# ── 1. EWMA Covariance ────────────────────────────────────────────────────────

class TestEWMACovariance:
    def test_shape(self):
        r = make_returns(120, 10)
        res = ewma_covariance(r, Horizon.TACTICAL)
        assert res.covariance.shape  == (10, 10)
        assert res.correlation.shape == (10, 10)
        assert res.volatilities.shape == (10,)

    def test_diagonal_ones(self):
        r = make_returns(120, 10)
        res = ewma_covariance(r, Horizon.TACTICAL)
        np.testing.assert_allclose(np.diag(res.correlation), 1.0, atol=1e-10)

    def test_symmetry(self):
        r = make_returns(120, 10)
        res = ewma_covariance(r, Horizon.TACTICAL)
        np.testing.assert_allclose(res.correlation, res.correlation.T, atol=1e-12)

    def test_psd(self):
        r = make_returns(120, 10)
        res = ewma_covariance(r, Horizon.TACTICAL)
        evals = np.linalg.eigvalsh(res.covariance)
        assert np.all(evals >= -1e-10), f"Covariance not PSD: min eval = {evals.min()}"

    def test_avg_correlation_range(self):
        """avg_correlation must be in (-1, 1) — the previous 0% bug."""
        r   = make_returns(120, 10)
        res = ewma_covariance(r, Horizon.TACTICAL)
        assert -1.0 < res.avg_correlation < 1.0, (
            f"avg_correlation out of range: {res.avg_correlation}"
        )

    def test_avg_correlation_excludes_diagonal(self):
        """With synthetic correlated data it should be meaningfully above 0."""
        r   = make_returns(120, 10, seed=1)
        res = ewma_covariance(r, Horizon.TACTICAL)
        # should be non-trivially positive for a 2-factor model
        assert res.avg_correlation > 0.01, (
            f"avg_correlation suspiciously low: {res.avg_correlation}"
        )

    def test_multi_horizon(self):
        r = make_returns(300, 10)
        results = multi_horizon_covariance(r)
        assert set(results.keys()) == set(Horizon)

    def test_too_few_obs(self):
        r = make_returns(5, 10)
        with pytest.raises(ValueError, match="observations"):
            ewma_covariance(r, Horizon.TACTICAL)

    def test_lambda_range(self):
        r = make_returns(120, 10)
        with pytest.raises(ValueError):
            ewma_covariance(r, lambda_=1.5)


# ── 2. RMT Filter ─────────────────────────────────────────────────────────────

class TestRMTFilter:
    def test_shape(self):
        r   = make_returns(120, 10)
        res = ewma_covariance(r, Horizon.TACTICAL)
        rmt = rmt_filter(res.correlation, T=120)
        assert rmt.filtered_correlation.shape == (10, 10)

    def test_diagonal_ones(self):
        r   = make_returns(120, 10)
        res = ewma_covariance(r, Horizon.TACTICAL)
        rmt = rmt_filter(res.correlation, T=120)
        np.testing.assert_allclose(np.diag(rmt.filtered_correlation), 1.0, atol=1e-10)

    def test_bounded(self):
        r   = make_returns(120, 10)
        res = ewma_covariance(r, Horizon.TACTICAL)
        rmt = rmt_filter(res.correlation, T=120)
        assert rmt.filtered_correlation.max() <= 1.0 + 1e-10
        assert rmt.filtered_correlation.min() >= -1.0 - 1e-10

    def test_signal_noise_partition(self):
        r   = make_returns(120, 10)
        res = ewma_covariance(r, Horizon.TACTICAL)
        rmt = rmt_filter(res.correlation, T=120)
        assert rmt.n_signal + rmt.n_noise == 10

    def test_mp_pdf_normalises(self):
        """MP density integrates to 1/q over its support for q < 1."""
        q, sigma_sq = 0.3, 1.0
        lmin = sigma_sq * (1 - np.sqrt(1/q)) ** 2
        lmax = sigma_sq * (1 + np.sqrt(1/q)) ** 2
        x    = np.linspace(lmin + 1e-6, lmax - 1e-6, 50000)
        pdf  = marchenko_pastur_pdf(x, q=q, sigma_sq=sigma_sq)
        dx   = x[1] - x[0]
        mass = float((pdf * dx).sum())
        expected = 1.0 / q
        assert abs(mass - expected) < 0.05, f"MP PDF mass = {mass:.4f}, expected {expected:.4f}"


# ── 3. Distance Correlation ────────────────────────────────────────────────────

class TestDistanceCorrelation:
    def test_independence_zero(self):
        rng = np.random.default_rng(0)
        x   = rng.standard_normal(200)
        y   = rng.standard_normal(200)
        r   = distance_covariance(x, y)
        # Independent normals — dCor should be near 0
        assert r.dcor < 0.25, f"dCor for independent series = {r.dcor:.4f}"

    def test_identical_series(self):
        rng = np.random.default_rng(1)
        x   = rng.standard_normal(100)
        r   = distance_covariance(x, x)
        assert r.dcor > 0.9

    def test_bounded(self):
        rng = np.random.default_rng(2)
        x   = rng.standard_normal(80)
        y   = rng.standard_normal(80)
        r   = distance_covariance(x, y)
        assert 0.0 <= r.dcor <= 1.0

    def test_matrix_shape(self):
        r   = make_returns(80, 5)
        mat = distance_correlation_matrix(r, max_assets=5)
        assert mat.shape == (5, 5)
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-10)


# ── 4. Tail Dependence ─────────────────────────────────────────────────────────

class TestTailDependence:
    def test_gaussian_limit(self):
        """As ν → ∞ tail dependence → 0."""
        r = t_copula_tail_dependence(rho=0.8, nu=1000.0)
        assert r.lambda_lower < 0.01

    def test_fat_tails(self):
        """At ν = 3, moderate correlation should give meaningful tail dep."""
        r = t_copula_tail_dependence(rho=0.5, nu=3.0)
        assert r.lambda_lower > 0.05

    def test_symmetry(self):
        r = t_copula_tail_dependence(rho=0.6, nu=5.0)
        assert r.lambda_lower == r.lambda_upper

    def test_matrix_shape(self):
        rng  = np.random.default_rng(3)
        corr = np.eye(8)
        for i in range(8):
            for j in range(i + 1, 8):
                v = rng.uniform(0.1, 0.7)
                corr[i, j] = v
                corr[j, i] = v
        mat = tail_dependence_matrix(corr, nu=5.0)
        assert mat.shape == (8, 8)
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-10)

    def test_higher_rho_more_tail_dep(self):
        r1 = t_copula_tail_dependence(rho=0.3, nu=5.0)
        r2 = t_copula_tail_dependence(rho=0.8, nu=5.0)
        assert r2.lambda_lower > r1.lambda_lower


# ── 5. Effective Rank ──────────────────────────────────────────────────────────

class TestEffectiveRank:
    def test_identity(self):
        """Identity matrix → effective rank = N (fully uncorrelated)."""
        N = 12
        er = effective_rank(np.eye(N))
        assert abs(er - N) < 0.1, f"Expected {N}, got {er:.3f}"

    def test_rank_one(self):
        """Ones matrix → effective rank ≈ 1."""
        N  = 8
        C  = np.ones((N, N))
        np.fill_diagonal(C, 1.0)
        er = effective_rank(C)
        assert er < 2.0, f"Expected ~1, got {er:.3f}"

    def test_bounded(self):
        r  = make_returns(120, 15)
        e  = ewma_covariance(r, Horizon.TACTICAL)
        rmt= rmt_filter(e.correlation, T=120)
        er = effective_rank(rmt.filtered_correlation)
        assert 1.0 <= er <= 15.0


# ── 6. TDA ─────────────────────────────────────────────────────────────────────

class TestTDA:
    def test_h0_count(self):
        r   = make_returns(120, 10)
        e   = ewma_covariance(r, Horizon.TACTICAL)
        rmt = rmt_filter(e.correlation, T=120)
        tda = compute_tda(rmt.filtered_correlation)
        # H₀ should have exactly N-1 bars (one per merge in spanning tree)
        assert len(tda.h0.bars) == 9

    def test_betti_non_negative(self):
        r   = make_returns(120, 10)
        e   = ewma_covariance(r, Horizon.TACTICAL)
        rmt = rmt_filter(e.correlation, T=120)
        tda = compute_tda(rmt.filtered_correlation)
        assert tda.betti_0 >= 0
        assert tda.betti_1 >= 0

    def test_wasserstein_self_zero(self):
        r   = make_returns(120, 10)
        e   = ewma_covariance(r, Horizon.TACTICAL)
        rmt = rmt_filter(e.correlation, T=120)
        tda = compute_tda(rmt.filtered_correlation)
        assert wasserstein1(tda.h0, tda.h0) < 1e-6

    def test_rolling_returns_list(self):
        r      = make_returns(200, 8)
        result = rolling_topology(r, window=60, step=20)
        assert isinstance(result, list)
        assert len(result) > 0
        assert "betti_0" in result[0]
        assert "regime_alert" in result[0]


# ── 7. PortfolioAnalyzer (integration) ───────────────────────────────────────

class TestPortfolioAnalyzer:
    def setup_method(self):
        self.r  = make_returns(T=200, N=12)
        self.t  = make_tickers(12)
        self.pa = PortfolioAnalyzer(self.r, self.t, nu=5.0)

    def test_full_report_runs(self):
        rep = self.pa.full_report()
        s   = rep.summary()
        assert "horizons" in s
        assert "tactical" in s["horizons"]

    def test_avg_correlation_correct(self):
        """Core regression test for the avg_correlation bug."""
        rep = self.pa.full_report(horizons=[Horizon.TACTICAL])
        avg = rep.horizons["tactical"].avg_correlation
        # Must be a genuine fraction, not 0.0 or 1.0
        assert -1.0 < avg < 1.0, f"avg_correlation = {avg}"
        # For a factor model it should be non-negative
        assert avg >= 0.0, f"avg_correlation unexpectedly < 0: {avg}"

    def test_effective_rank_range(self):
        rep = self.pa.full_report(horizons=[Horizon.TACTICAL])
        er  = rep.horizons["tactical"].effective_rank
        assert 1.0 <= er <= 12.0 + 1e-9

    def test_diversification_pct_capped(self):
        rep = self.pa.full_report()
        s   = rep.summary()
        for h in s["horizons"].values():
            assert h["diversification_pct"] <= 100.0

    def test_pair_analysis(self):
        res = self.pa.pair_analysis("STOCK_00", "STOCK_01")
        assert "distance_correlation" in res
        assert 0.0 <= res["distance_correlation"] <= 1.0
        assert 0.0 <= res["tail_dependence_lambda"] <= 1.0

    def test_unknown_ticker(self):
        with pytest.raises(ValueError):
            self.pa.pair_analysis("STOCK_00", "DOES_NOT_EXIST")

    def test_too_many_assets(self):
        big_r = np.random.randn(100, 301)
        big_t = [f"X{i}" for i in range(301)]
        with pytest.raises(ValueError, match="limit"):
            PortfolioAnalyzer(big_r, big_t)

    def test_correlation_matrix_shape(self):
        mat = self.pa.correlation_matrix(Horizon.TACTICAL, filtered=True)
        assert mat.shape == (12, 12)
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-10)

    def test_top_pairs_length(self):
        rep = self.pa.full_report(horizons=[Horizon.TACTICAL])
        top = rep.horizons["tactical"].top_corr_pairs
        assert len(top) == 10
        # Verify they are sorted descending
        vals = [p["correlation"] for p in top]
        assert vals == sorted(vals, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
