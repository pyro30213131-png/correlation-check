"""
core/data_loader.py
───────────────────
Data ingestion layer.  NO prices or tickers are hardcoded here.

You supply your own data in ONE of three ways:

  A)  CSV file  (recommended for most users)
      ─────────────────────────────────────
      Format:  rows = trading days (oldest first), columns = assets.
      Row 0:   header row with ticker symbols.
      Column 0 (optional): date string — detected automatically.

      Example layout:
          Date,RELIANCE,TCS,INFY,...
          2019-01-01,1200.50,2100.10,750.30,...
          2019-01-02,1205.20,2095.40,748.90,...
          ...

      Call:
          from core.data_loader import load_from_csv
          returns, tickers, dates = load_from_csv("path/to/prices.csv")

  B)  NumPy array (programmatic)
      ──────────────────────────
      Pass a (T × N) price matrix and a list of N ticker strings.

          from core.data_loader import load_from_array
          returns, tickers = load_from_array(price_array, ticker_list)

  C)  Pandas DataFrame (if you already have one)
      ─────────────────────────────────────────
          from core.data_loader import load_from_dataframe
          returns, tickers, dates = load_from_dataframe(df)
          # df: index = dates, columns = tickers, values = prices

Returns are ALWAYS log-returns:  r_t = log(P_t / P_{t-1})
Missing values are forward-filled then backward-filled before differencing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class PriceData:
    """
    Validated, clean price data ready for analysis.

    Attributes
    ----------
    returns : np.ndarray, shape (T, N)
        Log-returns.  T = trading days, N = assets.
    tickers : list[str]
        Asset identifiers — same order as columns of `returns`.
    dates : Optional[pd.DatetimeIndex]
        Trading dates corresponding to rows of `returns` (may be None
        if the source had no date column).
    prices : Optional[np.ndarray], shape (T+1, N)
        Original price matrix before differencing (retained for reference).
    n_assets : int
    n_obs : int
        Number of return observations (= number of price rows − 1).
    """
    returns:  np.ndarray
    tickers:  list[str]
    dates:    Optional[pd.DatetimeIndex]
    prices:   Optional[np.ndarray]
    n_assets: int
    n_obs:    int

    def __post_init__(self):
        assert self.returns.shape == (self.n_obs, self.n_assets), (
            f"Shape mismatch: returns {self.returns.shape} vs "
            f"({self.n_obs}, {self.n_assets})"
        )

    def window(self, last_n_days: int) -> "PriceData":
        """Return a slice containing only the most recent N trading days."""
        n = min(last_n_days, self.n_obs)
        sliced_ret   = self.returns[-n:]
        sliced_dates = self.dates[-n:] if self.dates is not None else None
        return PriceData(
            returns  = sliced_ret,
            tickers  = self.tickers,
            dates    = sliced_dates,
            prices   = self.prices[-(n + 1):] if self.prices is not None else None,
            n_assets = self.n_assets,
            n_obs    = n,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prices_to_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Convert a (T+1, N) price matrix to a (T, N) log-return matrix.
    Prices must be positive; zeros/NaNs are handled before this step.
    """
    prices = np.asarray(prices, dtype=float)
    if np.any(prices <= 0):
        raise ValueError(
            "Price matrix contains non-positive values. "
            "Ensure prices are clean before ingestion."
        )
    return np.log(prices[1:] / prices[:-1])


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill then backward-fill missing prices within each column.
    Columns that are entirely NaN are dropped with a warning.
    """
    all_nan = df.columns[df.isna().all()]
    if len(all_nan):
        import warnings
        warnings.warn(
            f"Dropping {len(all_nan)} column(s) with no valid prices: "
            f"{list(all_nan)}"
        )
        df = df.drop(columns=all_nan)
    df = df.ffill().bfill()
    return df


def _validate_shape(returns: np.ndarray, tickers: list[str]):
    T, N = returns.shape
    if N != len(tickers):
        raise ValueError(
            f"returns has {N} columns but {len(tickers)} tickers supplied."
        )
    if N > 300:
        raise ValueError(
            f"Universe size {N} exceeds the 300-asset limit. "
            "Subset your data before loading."
        )
    if T < 20:
        raise ValueError(
            f"Only {T} return observations — need at least 20."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Public loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_from_csv(
    filepath: str | Path,
    date_column: Optional[str] = None,
    price_columns: Optional[list[str]] = None,
) -> PriceData:
    """
    Load price data from a CSV file and return log-returns.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    date_column : str, optional
        Name of the date column.  If None, the loader auto-detects it by
        looking for the first non-numeric column.
    price_columns : list[str], optional
        Subset of columns to use as assets.  If None, all numeric columns
        (excluding the date column) are used.

    Returns
    -------
    PriceData

    ── HOW TO PREPARE YOUR CSV ─────────────────────────────────────────────
    1.  Each row is one trading day.  Oldest date must be the first row.
    2.  Each column (after the optional date column) is one asset.
    3.  Values are CLOSING PRICES, NOT returns.
    4.  Missing prices (circuit limits, holidays) will be forward-filled.
    5.  Tickers are taken from the column headers exactly as written.

    Example valid CSV (first 3 rows):
        Date,RELIANCE,TCS,HDFCBANK
        2020-01-01,1500.00,2200.00,1150.00
        2020-01-02,1510.50,2190.00,1162.30
    ────────────────────────────────────────────────────────────────────────
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    df_raw = pd.read_csv(filepath)

    # ── Auto-detect date column ───────────────────────────────────────────
    if date_column is None:
        for col in df_raw.columns:
            if not pd.api.types.is_numeric_dtype(df_raw[col]):
                date_column = col
                break

    dates = None
    if date_column and date_column in df_raw.columns:
        try:
            dates = pd.to_datetime(df_raw[date_column])
            df_raw = df_raw.drop(columns=[date_column])
        except Exception:
            pass   # non-parseable date column — treat as regular column

    # ── Select price columns ──────────────────────────────────────────────
    if price_columns:
        df_raw = df_raw[price_columns]

    # Keep only numeric columns
    df_prices = df_raw.select_dtypes(include=[np.number])
    df_prices = _clean_prices(df_prices)

    tickers   = list(df_prices.columns)
    price_arr = df_prices.to_numpy(dtype=float)
    ret_arr   = _prices_to_log_returns(price_arr)

    _validate_shape(ret_arr, tickers)

    return PriceData(
        returns  = ret_arr,
        tickers  = tickers,
        dates    = pd.DatetimeIndex(dates.values[1:]) if dates is not None else None,
        prices   = price_arr,
        n_assets = ret_arr.shape[1],
        n_obs    = ret_arr.shape[0],
    )


def load_from_array(
    prices: np.ndarray,
    tickers: list[str],
    dates: Optional[list | pd.DatetimeIndex] = None,
) -> PriceData:
    """
    Load price data from a NumPy array.

    Parameters
    ----------
    prices : np.ndarray, shape (T+1, N)
        Closing price matrix.  Rows = days (oldest first), columns = assets.
    tickers : list[str]
        N ticker symbols — must match column order of `prices`.
    dates : list or DatetimeIndex, optional
        T+1 dates corresponding to each price row.

    Returns
    -------
    PriceData

    ── HOW TO USE ───────────────────────────────────────────────────────────
    If you fetch data from yfinance, broker API, or a database, reshape it
    into a (T+1, N) array where T+1 is total price observations:

        import yfinance as yf
        import numpy as np
        from core.data_loader import load_from_array

        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]   # your universe
        raw = yf.download(tickers, start="2019-01-01", end="2024-01-01")
        prices = raw["Close"].to_numpy()                   # (T+1, N)
        data = load_from_array(prices, tickers)
    ────────────────────────────────────────────────────────────────────────
    """
    prices   = np.asarray(prices, dtype=float)
    tickers  = list(tickers)

    if prices.ndim == 1:
        prices = prices[:, None]

    # Clean: forward-fill NaN within columns
    df = pd.DataFrame(prices, columns=tickers)
    df = _clean_prices(df)
    price_arr = df.to_numpy(dtype=float)
    ret_arr   = _prices_to_log_returns(price_arr)

    _validate_shape(ret_arr, tickers)

    if dates is not None:
        dates_idx = pd.DatetimeIndex(dates)
        ret_dates = dates_idx[1:]    # returns correspond to periods between price rows
    else:
        ret_dates = None

    return PriceData(
        returns  = ret_arr,
        tickers  = tickers,
        dates    = ret_dates,
        prices   = price_arr,
        n_assets = ret_arr.shape[1],
        n_obs    = ret_arr.shape[0],
    )


def load_from_dataframe(
    df: pd.DataFrame,
    price_columns: Optional[list[str]] = None,
) -> PriceData:
    """
    Load price data from a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Index = dates (DatetimeIndex preferred), columns = tickers,
        values = closing prices.
    price_columns : list[str], optional
        Subset of columns to use.

    Returns
    -------
    PriceData

    ── HOW TO USE ───────────────────────────────────────────────────────────
        import pandas as pd
        from core.data_loader import load_from_dataframe

        df = pd.read_parquet("prices.parquet")  # your DataFrame
        data = load_from_dataframe(df)
        # or with a subset:
        data = load_from_dataframe(df, price_columns=["RELIANCE","TCS"])
    ────────────────────────────────────────────────────────────────────────
    """
    if price_columns:
        df = df[price_columns]

    df = df.select_dtypes(include=[np.number])
    df = df.sort_index()          # ensure chronological order
    df = _clean_prices(df)

    tickers   = list(df.columns)
    price_arr = df.to_numpy(dtype=float)
    ret_arr   = _prices_to_log_returns(price_arr)

    _validate_shape(ret_arr, tickers)

    dates = df.index[1:] if isinstance(df.index, pd.DatetimeIndex) else None

    return PriceData(
        returns  = ret_arr,
        tickers  = tickers,
        dates    = dates,
        prices   = price_arr,
        n_assets = ret_arr.shape[1],
        n_obs    = ret_arr.shape[0],
    )
