"""
api/main.py
───────────
FastAPI application exposing the Portfolio Analyzer as a REST service.

Endpoints
─────────
POST /upload          — upload a CSV price file, returns session_id
POST /analyze         — run full analysis on uploaded data
GET  /correlation     — fetch correlation matrix for a horizon
GET  /pairs           — deep-dive on a single asset pair
GET  /regime          — rolling regime-change timeline
GET  /summary         — JSON summary of all horizons

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import io
import uuid
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.data_loader import load_from_csv, load_from_dataframe, PriceData
from core.analyzer    import PortfolioAnalyzer, FullReport
from core.covariance  import Horizon

app = FastAPI(
    title="Portfolio Correlation & Diversity Analyzer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store (replace with Redis/DB in production) ──────────
_sessions: dict[str, PriceData]    = {}
_reports:  dict[str, FullReport]   = {}


# ─────────────────────────────────────────────────────────────────────────────
#  Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    session_id:     str
    horizons:       list[str] = ["tactical", "cyclical", "macro"]
    nu:             float     = 5.0
    include_dcor:   bool      = False
    regime_window:  int       = 60
    regime_step:    int       = 10


class PairRequest(BaseModel):
    session_id: str
    ticker_a:   str
    ticker_b:   str
    horizon:    str = "tactical"


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> PriceData:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Upload data first.")
    return _sessions[session_id]


def _get_report(session_id: str) -> FullReport:
    if session_id not in _reports:
        raise HTTPException(status_code=404, detail=f"No analysis found for session '{session_id}'. Run /analyze first.")
    return _reports[session_id]


def _parse_horizon(h: str) -> Horizon:
    try:
        return Horizon(h.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown horizon '{h}'. Use: tactical, cyclical, macro.")


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_prices(file: UploadFile = File(...)):
    """
    Upload a CSV of closing prices.

    CSV format
    ──────────
    - Row 0:  header with ticker symbols (first column may be a date).
    - Rows 1+: one trading day per row, oldest first.
    - Values:  closing prices (NOT returns — the API converts internally).

    Returns
    -------
    session_id : str  — pass this to /analyze and other endpoints.
    tickers    : list — detected asset names.
    n_obs      : int  — number of return observations.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    try:
        data = load_from_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    sid = str(uuid.uuid4())
    _sessions[sid] = data

    return {
        "session_id": sid,
        "tickers":    data.tickers,
        "n_assets":   data.n_assets,
        "n_obs":      data.n_obs,
        "message":    "Data loaded. Call POST /analyze to run the full pipeline.",
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Run the full correlation and diversity analysis.

    Parameters (JSON body)
    ──────────────────────
    session_id    : from /upload
    horizons      : subset of ["tactical","cyclical","macro"]
    nu            : Student-t copula d.o.f. (default 5)
    include_dcor  : compute distance-correlation matrix (slow for N > 60)
    regime_window : rolling TDA window in days
    regime_step   : step between TDA snapshots
    """
    data      = _get_session(req.session_id)
    horizons  = [_parse_horizon(h) for h in req.horizons]
    analyzer  = PortfolioAnalyzer(data.returns, data.tickers, nu=req.nu)

    try:
        report = analyzer.full_report(
            horizons       = horizons,
            include_dcor   = req.include_dcor,
            regime_window  = req.regime_window,
            regime_step    = req.regime_step,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _reports[req.session_id] = report
    return report.summary()


@app.get("/summary")
def summary(session_id: str = Query(...)):
    """Return the JSON summary of the most recent analysis."""
    return _get_report(session_id).summary()


@app.get("/correlation")
def correlation(
    session_id: str   = Query(...),
    horizon:    str   = Query("tactical"),
    filtered:   bool  = Query(True, description="RMT-filtered (True) or raw EWMA (False)"),
):
    """
    Return the N×N correlation matrix for the requested horizon.

    filtered=true  → RMT noise-filtered (recommended)
    filtered=false → raw EWMA correlation
    """
    data  = _get_session(session_id)
    h     = _parse_horizon(horizon)
    from core.covariance import ewma_covariance, WINDOW_MAP
    from core.rmt_filter import rmt_filter

    T   = data.n_obs
    win = WINDOW_MAP[h]
    r   = data.returns[-min(win, T):]

    ewma = ewma_covariance(r, horizon=h, shrink=True)
    if not filtered:
        mat = ewma.correlation
    else:
        rmt = rmt_filter(ewma.correlation, T=len(r))
        mat = rmt.filtered_correlation

    return {
        "horizon":    horizon,
        "filtered":   filtered,
        "tickers":    data.tickers,
        "matrix":     mat.tolist(),
        "n_assets":   data.n_assets,
    }


@app.post("/pairs")
def pair_analysis(req: PairRequest):
    """Deep-dive dependency analysis for a single asset pair."""
    data     = _get_session(req.session_id)
    analyzer = PortfolioAnalyzer(data.returns, data.tickers)
    h        = _parse_horizon(req.horizon)
    try:
        result = analyzer.pair_analysis(req.ticker_a, req.ticker_b, horizon=h)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/regime")
def regime_timeline(
    session_id: str = Query(...),
    window:     int = Query(60),
    step:       int = Query(10),
):
    """Return the rolling TDA regime-change timeline."""
    data = _get_session(session_id)
    from core.tda import rolling_topology
    timeline = rolling_topology(data.returns, window=window, step=step)
    return {
        "session_id": session_id,
        "n_windows":  len(timeline),
        "timeline":   timeline,
    }


@app.get("/tickers")
def list_tickers(session_id: str = Query(...)):
    """List all tickers in the loaded dataset."""
    data = _get_session(session_id)
    return {"tickers": data.tickers, "n_assets": data.n_assets}
