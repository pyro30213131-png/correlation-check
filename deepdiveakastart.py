"""
nifty100_deep_dive.py
─────────────────────
Full quantitative deep-dive on every stock and every pair in the Nifty 100.

Outputs (saved as CSV + printed to terminal):
  1.  Stock-level report          → nifty100_stock_report.csv
  2.  All-pairs deep dive         → nifty100_pairs_report.csv
  3.  Regime timeline             → nifty100_regime_timeline.csv
  4.  Cluster membership          → nifty100_clusters.csv
  5.  Factor decomposition        → nifty100_factors.csv

Usage:
    pip install yfinance numpy scipy pandas
    python nifty100_deep_dive.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from itertools import combinations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from portfolio_analyzer.core.data_loader  import load_from_dataframe
from portfolio_analyzer.core.covariance   import ewma_covariance, Horizon, WINDOW_MAP
from portfolio_analyzer.core.rmt_filter   import rmt_filter, pseudoinverse_correlation
from portfolio_analyzer.core.dependency   import (
    distance_covariance, t_copula_tail_dependence,
    effective_rank, diversification_ratio,
)
from portfolio_analyzer.core.tda          import compute_tda, rolling_topology, wasserstein1
from portfolio_analyzer.core.analyzer     import PortfolioAnalyzer


# ══════════════════════════════════════════════════════════════════════════════
#  NIFTY 100 UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

NIFTY100 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","BHARTIARTL.NS","ICICIBANK.NS",
    "INFY.NS","SBIN.NS","HINDUNILVR.NS","ITC.NS","LT.NS",
    "BAJFINANCE.NS","HCLTECH.NS","MARUTI.NS","SUNPHARMA.NS","KOTAKBANK.NS",
    "ONGC.NS","TITAN.NS","AXISBANK.NS","NTPC.NS","ASIANPAINT.NS",
    "WIPRO.NS","ULTRACEMCO.NS","POWERGRID.NS","BAJAJFINSV.NS","COALINDIA.NS",
    "NESTLEIND.NS","TATAMOTORS.NS","INDUSINDBK.NS","GRASIM.NS","ADANIPORTS.NS",
    "TECHM.NS","BRITANNIA.NS","JSWSTEEL.NS","TATACONSUM.NS","DRREDDY.NS",
    "CIPLA.NS","HINDALCO.NS","BPCL.NS","TATASTEEL.NS","EICHERMOT.NS",
    "HEROMOTOCO.NS","DIVISLAB.NS","SBILIFE.NS","HDFCLIFE.NS","BAJAJ-AUTO.NS",
    "APOLLOHOSP.NS","ADANIENT.NS","PIDILITIND.NS","SIEMENS.NS","HAVELLS.NS",
    "DMART.NS","ICICIPRULI.NS","ICICIGI.NS","SHRIRAMFIN.NS","TRENT.NS",
    "BANKBARODA.NS","CHOLAFIN.NS","ZOMATO.NS","IRCTC.NS","HAL.NS",
    "BEL.NS","INDIGO.NS","BERGEPAINT.NS","AMBUJACEM.NS","ACC.NS",
    "LUPIN.NS","TORNTPHARM.NS","AUROPHARMA.NS","GAIL.NS","IOC.NS",
    "MCDOWELL-N.NS","COLPAL.NS","DABUR.NS","MARICO.NS","GODREJCP.NS",
    "EMAMILTD.NS","JUBLFOOD.NS","DLF.NS","VOLTAS.NS","PAGEIND.NS",
    "MUTHOOTFIN.NS","PFC.NS","M&M.NS","ADANIGREEN.NS","ADANIPOWER.NS",
    "NYKAA.NS","PAYTM.NS","OBEROIRLTY.NS","LODHA.NS","PRESTIGE.NS",
    "PHOENIXLTD.NS","WHIRLPOOL.NS","MANAPPURAM.NS","VEDL.NS","BHEL.NS",
    "LICI.NS","LTIM.NS","TATAELXSI.NS","PERSISTENT.NS","COFORGE.NS",
]

NU          = 5.0           # Student-t copula d.o.f.
OUT_DIR     = Path(".")     # where to save CSVs


# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_prices() -> pd.DataFrame:
    print(f"\n{'═'*60}")
    print(f"  Downloading Nifty 100 ({len(NIFTY100)} stocks)")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"{'═'*60}\n")

    raw = yf.download(
        tickers     = NIFTY100,
        start       = START_DATE,
        end         = END_DATE,
        auto_adjust = True,
        progress    = True,
        threads     = True,
    )

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    # Keep tickers with at least 80% data coverage
    thresh  = int(len(prices) * 0.80)
    prices  = prices.dropna(axis=1, thresh=thresh)
    prices  = prices.ffill().bfill()

    dropped = set(NIFTY100) - set(prices.columns)
    if dropped:
        print(f"  ⚠  Dropped {len(dropped)} tickers (insufficient data): {dropped}\n")

    print(f"  ✓  {len(prices.columns)} assets × {len(prices)} trading days loaded\n")
    return prices


# ══════════════════════════════════════════════════════════════════════════════
#  2. COMPUTE ALL THREE HORIZON MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def build_horizon_matrices(returns: np.ndarray, T: int, N: int) -> dict:
    """EWMA + RMT for all three horizons."""
    out = {}
    for h in Horizon:
        win    = min(WINDOW_MAP[h], T)
        r_win  = returns[-win:]
        ewma   = ewma_covariance(r_win, horizon=h, shrink=True)
        rmt    = rmt_filter(ewma.correlation, T=win)
        out[h] = {"ewma": ewma, "rmt": rmt, "window": win}
        print(f"  [{h.value:8s}]  λ={ewma.lambda_:.2f}  "
              f"signal_factors={rmt.n_signal:3d}  "
              f"λ_max={rmt.lambda_max:.3f}  "
              f"cond={rmt.condition_number:.1f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  3. STOCK-LEVEL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def stock_level_report(
    returns:  np.ndarray,
    tickers:  list[str],
    matrices: dict,
    tda_tac:  object,
) -> pd.DataFrame:
    """
    Per-stock metrics across all three horizons.

    Columns
    ───────
    ticker
    vol_tactical / vol_cyclical / vol_macro        — annualised σ
    avg_corr_tactical / _cyclical / _macro         — avg pairwise corr to rest
    avg_tail_dep_tactical / _cyclical / _macro     — avg tail dep to rest
    max_corr_tactical  — highest single-pair correlation (tactical)
    max_corr_peer      — which stock it's correlated to most
    factor_loading_1/2/3  — loadings on top 3 RMT signal eigenvectors
    cluster_id         — which cluster this stock belongs to (tactical)
    """
    N = len(tickers)
    rows = []

    # Cluster labels from H₀ (tactical)
    cluster_labels = _cluster_labels(tda_tac, N)

    for i, tkr in enumerate(tickers):
        row: dict = {"ticker": tkr}

        for h in Horizon:
            m    = matrices[h]
            ewma = m["ewma"]
            rmt  = m["rmt"]
            C    = rmt.filtered_correlation

            # Volatility
            row[f"vol_{h.value}"] = round(float(ewma.volatilities[i]), 5)

            # Average pairwise correlation to all other stocks (not to self)
            others = [j for j in range(N) if j != i]
            row[f"avg_corr_{h.value}"] = round(float(C[i, others].mean()), 5)

            # Average tail dependence to all others
            tail_vals = [
                t_copula_tail_dependence(C[i, j], NU).lambda_lower
                for j in others
            ]
            row[f"avg_tail_dep_{h.value}"] = round(float(np.mean(tail_vals)), 5)

        # Max correlated peer (tactical)
        C_tac = matrices[Horizon.TACTICAL]["rmt"].filtered_correlation
        peers = [(abs(C_tac[i, j]), tickers[j]) for j in range(N) if j != i]
        peers.sort(reverse=True)
        row["max_corr_tactical"] = round(peers[0][0], 5)
        row["max_corr_peer"]     = peers[0][1]

        # Top-3 factor loadings (signal eigenvectors, tactical)
        rmt_tac = matrices[Horizon.TACTICAL]["rmt"]
        for k in range(min(3, rmt_tac.n_signal)):
            row[f"factor_loading_{k+1}"] = round(
                float(rmt_tac.eigenvectors_signal[i, k]), 5
            )

        row["cluster_id"] = cluster_labels[i]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("avg_corr_tactical", ascending=False)
    return df


def _cluster_labels(tda: object, N: int) -> list[int]:
    """
    Assign cluster IDs using the H₀ persistence diagram.
    Assets that merge early (low death value) share a cluster.
    """
    labels   = list(range(N))   # start: each asset its own cluster
    dist     = tda.dist_matrix
    edges    = sorted(
        [(dist[i, j], i, j) for i in range(N) for j in range(i+1, N)]
    )
    parent   = list(range(N))
    cluster_id = list(range(N))
    next_id  = N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            # Only merge if distance < 0.5 (moderate correlation threshold)
            if d < 0.5:
                cid = cluster_id[ri]
                parent[rj] = ri
                cluster_id[ri] = cid
            else:
                break   # stop merging at weak correlations

    return [cluster_id[find(i)] for i in range(N)]


# ══════════════════════════════════════════════════════════════════════════════
#  4. ALL-PAIRS DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

def all_pairs_deep_dive(
    returns:  np.ndarray,
    tickers:  list[str],
    matrices: dict,
) -> pd.DataFrame:
    """
    Every unique pair (N*(N-1)/2 pairs) with full metrics.

    Columns
    ───────
    asset_a, asset_b
    pearson_raw                — raw Pearson ρ (tactical window)
    ewma_corr_tactical/cyclical/macro
    rmt_corr_tactical/cyclical/macro
    dcor                       — distance correlation (nonlinear dep)
    nonlinearity_excess        — dCor − |ρ|  (positive = hidden dependency)
    tail_dep_tactical/cyclical/macro   — Student-t copula λ
    joint_crash_1pct           — P(both drop >1% simultaneously)
    dispersion                 — std(rmt_corr across 3 horizons) — regime stability
    """
    N    = len(tickers)
    rows = []
    total = N * (N - 1) // 2

    # Pre-extract all correlation matrices
    C = {h: matrices[h]["rmt"].filtered_correlation for h in Horizon}
    E = {h: matrices[h]["ewma"].correlation          for h in Horizon}

    # Tactical returns window for dCor
    win_tac  = min(WINDOW_MAP[Horizon.TACTICAL], returns.shape[0])
    r_tac    = returns[-win_tac:]

    print(f"\n  Computing {total:,} pairs", end="", flush=True)
    done = 0

    for i, j in combinations(range(N), 2):
        # ── Pearson (raw tactical) ────────────────────────────────────────
        pearson = float(np.corrcoef(r_tac[:, i], r_tac[:, j])[0, 1])

        # ── EWMA and RMT correlations across horizons ─────────────────────
        rmt_vals  = {h: float(C[h][i, j]) for h in Horizon}
        ewma_vals = {h: float(E[h][i, j]) for h in Horizon}

        # ── Distance Correlation (dCor) ───────────────────────────────────
        dcov = distance_covariance(r_tac[:, i], r_tac[:, j])

        # ── Tail dependence per horizon ───────────────────────────────────
        tail = {h: t_copula_tail_dependence(rmt_vals[h], NU) for h in Horizon}

        # ── Horizon dispersion (stability signal) ─────────────────────────
        rmt_arr    = np.array(list(rmt_vals.values()))
        dispersion = float(rmt_arr.std())

        rows.append({
            "asset_a":                    tickers[i],
            "asset_b":                    tickers[j],
            "pearson_raw":                round(pearson, 5),
            "ewma_corr_tactical":         round(ewma_vals[Horizon.TACTICAL],  5),
            "ewma_corr_cyclical":         round(ewma_vals[Horizon.CYCLICAL],  5),
            "ewma_corr_macro":            round(ewma_vals[Horizon.MACRO],     5),
            "rmt_corr_tactical":          round(rmt_vals[Horizon.TACTICAL],   5),
            "rmt_corr_cyclical":          round(rmt_vals[Horizon.CYCLICAL],   5),
            "rmt_corr_macro":             round(rmt_vals[Horizon.MACRO],      5),
            "dcor":                       round(dcov.dcor,          5),
            "nonlinearity_excess":        round(dcov.nonlinearity,  5),
            "tail_dep_tactical":          round(tail[Horizon.TACTICAL].lambda_lower, 5),
            "tail_dep_cyclical":          round(tail[Horizon.CYCLICAL].lambda_lower, 5),
            "tail_dep_macro":             round(tail[Horizon.MACRO].lambda_lower,    5),
            "joint_crash_1pct":           round(tail[Horizon.TACTICAL].joint_crash_prob_1pct, 7),
            "horizon_dispersion":         round(dispersion, 5),
        })

        done += 1
        if done % 500 == 0:
            print(f"  {done:,}/{total:,}", end="\r", flush=True)

    print(f"  ✓ {total:,} pairs computed          ")

    df = pd.DataFrame(rows)
    df = df.sort_values("rmt_corr_tactical", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  5. FACTOR DECOMPOSITION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def factor_report(matrices: dict, tickers: list[str]) -> pd.DataFrame:
    """
    For each RMT signal factor (across all horizons):
      - eigenvalue (variance explained)
      - top 5 positive / negative stock loadings
      - interpreted factor type (market / sector / idiosyncratic)
    """
    rows = []
    for h in Horizon:
        rmt = matrices[h]["rmt"]
        for k in range(rmt.n_signal):
            ev  = rmt.eigenvalues_signal[k]
            vec = rmt.eigenvectors_signal[:, k]
            # Top positive and negative loaders
            order    = np.argsort(vec)
            top_pos  = [(tickers[i], round(float(vec[i]), 4)) for i in order[-5:][::-1]]
            top_neg  = [(tickers[i], round(float(vec[i]), 4)) for i in order[:5]]
            var_frac  = ev / rmt.eigenvalues_all.sum()
            rows.append({
                "horizon":          h.value,
                "factor":           k + 1,
                "eigenvalue":       round(float(ev), 4),
                "var_explained_pct":round(float(var_frac * 100), 3),
                "top_pos_1":        top_pos[0][0] if len(top_pos) > 0 else "",
                "top_pos_2":        top_pos[1][0] if len(top_pos) > 1 else "",
                "top_pos_3":        top_pos[2][0] if len(top_pos) > 2 else "",
                "top_neg_1":        top_neg[0][0] if len(top_neg) > 0 else "",
                "top_neg_2":        top_neg[1][0] if len(top_neg) > 1 else "",
                "loading_range":    round(float(vec.max() - vec.min()), 4),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  6. REGIME TIMELINE
# ══════════════════════════════════════════════════════════════════════════════

def regime_report(returns: np.ndarray) -> pd.DataFrame:
    print("\n  Computing rolling TDA regime timeline...")
    timeline = rolling_topology(returns, window=60, step=5)
    df = pd.DataFrame(timeline)
    alerts = df["regime_alert"].sum()
    print(f"  ✓ {len(df)} windows · {alerts} regime alerts detected")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  7. TERMINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(
    stock_df:  pd.DataFrame,
    pairs_df:  pd.DataFrame,
    regime_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    matrices:  dict,
    tickers:   list[str],
):
    N = len(tickers)
    C_tac = matrices[Horizon.TACTICAL]["rmt"].filtered_correlation
    ui, uj = np.triu_indices(N, k=1)
    avg_corr = float(C_tac[ui, uj].mean())
    eff_rnk  = effective_rank(C_tac)

    print(f"\n{'═'*60}")
    print("  NIFTY 100 — FULL QUANTITATIVE DEEP DIVE")
    print(f"{'═'*60}")
    print(f"  Universe          : {N} stocks")
    print(f"  Total pairs       : {len(pairs_df):,}")
    print(f"  Avg correlation   : {avg_corr:.4f}")
    print(f"  Effective rank    : {eff_rnk:.2f}  ({eff_rnk/N*100:.1f}% diversified)")
    print(f"  Regime alerts     : {regime_df['regime_alert'].sum()}")

    for h in Horizon:
        rmt = matrices[h]["rmt"]
        print(f"\n  [{h.value.upper()}]")
        print(f"    Signal factors  : {rmt.n_signal}")
        print(f"    λ_max (MP)      : {rmt.lambda_max:.4f}")
        print(f"    Condition #     : {rmt.condition_number:.2f}")

    print(f"\n  TOP 10 MOST CORRELATED PAIRS (Tactical RMT)")
    print(f"  {'Asset A':20s} {'Asset B':20s} {'ρ RMT':>8} {'dCor':>8} {'λ Tail':>8}")
    print(f"  {'-'*68}")
    for _, r in pairs_df.head(10).iterrows():
        print(f"  {r['asset_a']:20s} {r['asset_b']:20s} "
              f"{r['rmt_corr_tactical']:>8.4f} {r['dcor']:>8.4f} {r['tail_dep_tactical']:>8.4f}")

    print(f"\n  TOP 10 HIDDEN NONLINEAR DEPENDENCIES (dCor − |ρ| > 0)")
    hidden = pairs_df.nlargest(10, "nonlinearity_excess")
    print(f"  {'Asset A':20s} {'Asset B':20s} {'ρ Pearson':>10} {'dCor':>8} {'Excess':>8}")
    print(f"  {'-'*70}")
    for _, r in hidden.iterrows():
        print(f"  {r['asset_a']:20s} {r['asset_b']:20s} "
              f"{r['pearson_raw']:>10.4f} {r['dcor']:>8.4f} {r['nonlinearity_excess']:>8.4f}")

    print(f"\n  TOP 10 HIGHEST TAIL RISK PAIRS (Student-t Copula λ)")
    tail_top = pairs_df.nlargest(10, "tail_dep_tactical")
    print(f"  {'Asset A':20s} {'Asset B':20s} {'λ Tactical':>12} {'λ Macro':>10} {'Joint Crash%':>14}")
    print(f"  {'-'*70}")
    for _, r in tail_top.iterrows():
        print(f"  {r['asset_a']:20s} {r['asset_b']:20s} "
              f"{r['tail_dep_tactical']:>12.4f} {r['tail_dep_macro']:>10.4f} "
              f"{r['joint_crash_1pct']*100:>14.5f}")

    print(f"\n  MOST REGIME-UNSTABLE PAIRS (high horizon dispersion)")
    unstable = pairs_df.nlargest(10, "horizon_dispersion")
    print(f"  {'Asset A':20s} {'Asset B':20s} {'Tac ρ':>8} {'Mac ρ':>8} {'Dispersion':>12}")
    print(f"  {'-'*72}")
    for _, r in unstable.iterrows():
        print(f"  {r['asset_a']:20s} {r['asset_b']:20s} "
              f"{r['rmt_corr_tactical']:>8.4f} {r['rmt_corr_macro']:>8.4f} "
              f"{r['horizon_dispersion']:>12.5f}")

    print(f"\n  STOCK RANKING BY SYSTEMIC CONNECTIVITY (avg_corr_tactical)")
    print(f"  {'Ticker':20s} {'Avg Corr':>10} {'Avg Tail':>10} {'Vol (Ann)':>12} {'Cluster':>10}")
    print(f"  {'-'*65}")
    for _, r in stock_df.head(20).iterrows():
        print(f"  {r['ticker']:20s} {r['avg_corr_tactical']:>10.4f} "
              f"{r['avg_tail_dep_tactical']:>10.4f} {r['vol_tactical']:>12.4f} "
              f"{str(r['cluster_id']):>10}")

    print(f"\n  SIGNAL FACTORS (Tactical) — Top 3 Eigenvectors")
    tac_factors = factor_df[factor_df["horizon"] == "tactical"].head(5)
    for _, r in tac_factors.iterrows():
        print(f"  Factor {int(r['factor'])}: {r['var_explained_pct']:.2f}% variance  "
              f"| +loads: {r['top_pos_1']}, {r['top_pos_2']}, {r['top_pos_3']}  "
              f"| -loads: {r['top_neg_1']}, {r['top_neg_2']}")

    print(f"\n{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Fetch ──────────────────────────────────────────────────────────
    prices  = fetch_prices()
    data    = load_from_dataframe(prices)
    returns = data.returns
    tickers = data.tickers
    T, N    = returns.shape

    # ── 2. Build matrices ─────────────────────────────────────────────────
    print("\n  Building EWMA + RMT matrices for all horizons...")
    matrices = build_horizon_matrices(returns, T, N)

    # ── 3. TDA (tactical) ─────────────────────────────────────────────────
    print("\n  Running TDA persistent homology (tactical)...")
    C_tac   = matrices[Horizon.TACTICAL]["rmt"].filtered_correlation
    tda_tac = compute_tda(C_tac)
    print(f"  β₀={tda_tac.betti_0}  β₁={tda_tac.betti_1}  {tda_tac.interpretation}")

    # ── 4. All reports ────────────────────────────────────────────────────
    print("\n  Building stock-level report...")
    stock_df  = stock_level_report(returns, tickers, matrices, tda_tac)

    print("\n  Running all-pairs deep dive...")
    pairs_df  = all_pairs_deep_dive(returns, tickers, matrices)

    print("\n  Building factor decomposition...")
    factor_df = factor_report(matrices, tickers)

    regime_df = regime_report(returns)

    # ── 5. Print terminal summary ─────────────────────────────────────────
    print_summary(stock_df, pairs_df, regime_df, factor_df, matrices, tickers)

    # ── 6. Save CSVs ──────────────────────────────────────────────────────
    print("  Saving outputs...")
    stock_df.to_csv(OUT_DIR  / "nifty100_stock_report.csv",    index=False)
    pairs_df.to_csv(OUT_DIR  / "nifty100_pairs_report.csv",    index=False)
    factor_df.to_csv(OUT_DIR / "nifty100_factors.csv",         index=False)
    regime_df.to_csv(OUT_DIR / "nifty100_regime_timeline.csv", index=False)

    print(f"\n  ✓  Saved 4 CSV files to {OUT_DIR.resolve()}")
    print(f"     nifty100_stock_report.csv    — {len(stock_df)} stocks")
    print(f"     nifty100_pairs_report.csv    — {len(pairs_df):,} pairs")
    print(f"     nifty100_factors.csv         — {len(factor_df)} factors")
    print(f"     nifty100_regime_timeline.csv — {len(regime_df)} windows")
    print()


if __name__ == "__main__":
    main()
    # pair_deep_dive(report, data, "RELIANCE.NS", "TCS.NS")