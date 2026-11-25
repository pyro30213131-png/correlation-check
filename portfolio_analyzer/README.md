# Portfolio Correlation & Diversity Analyzer — Backend

## Project structure

```
portfolio_analyzer/
├── core/
│   ├── data_loader.py      ← PUT YOUR DATA IN HERE (see below)
│   ├── covariance.py       ← EWMA engine (3 horizons)
│   ├── rmt_filter.py       ← Marchenko-Pastur noise filtering
│   ├── dependency.py       ← dCor, Student-t copula, effective rank
│   ├── tda.py              ← Persistent homology, regime detection
│   └── analyzer.py         ← Orchestrator (call this)
├── api/
│   └── main.py             ← FastAPI REST layer
└── tests/
    └── test_all.py         ← 39 tests, all passing
```

---

## ══ WHERE TO PUT YOUR DATA ══

You have **three options**.  Pick whichever matches how you store prices.

---

### Option A — CSV file  *(easiest)*

Prepare a CSV where:
- **Row 0**: header with ticker symbols (first column may be a date label)
- **Rows 1+**: one trading day per row, **oldest first**
- **Values**: closing prices (NOT returns — the system converts internally)
- Missing prices are forward-filled automatically

```
Date,RELIANCE,TCS,HDFCBANK,INFY,...
2019-01-01,1500.00,2200.00,1150.00,750.00,...
2019-01-02,1510.50,2190.00,1162.30,748.90,...
2019-01-03,1498.20,2205.00,1158.10,752.40,...
```

**Code:**
```python
from core.data_loader import load_from_csv
from core.analyzer    import PortfolioAnalyzer

data     = load_from_csv("your_prices.csv")          # ← path to your file
analyzer = PortfolioAnalyzer(data.returns, data.tickers)
report   = analyzer.full_report()
print(report.summary())
```

---

### Option B — NumPy array  *(if you fetch from a broker/API)*

```python
import numpy as np
from core.data_loader import load_from_array
from core.analyzer    import PortfolioAnalyzer

# prices: shape (T+1, N) — rows = days oldest first, columns = assets
prices  = np.load("prices.npy")                      # ← your price matrix
tickers = ["RELIANCE", "TCS", "HDFCBANK", ...]       # ← N ticker strings

data     = load_from_array(prices, tickers)
analyzer = PortfolioAnalyzer(data.returns, data.tickers)
report   = analyzer.full_report()
```

**With yfinance:**
```python
import yfinance as yf
from core.data_loader import load_from_array
from core.analyzer    import PortfolioAnalyzer

tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]   # ← your universe
raw     = yf.download(tickers, start="2019-01-01", end="2024-12-31")
prices  = raw["Close"].to_numpy()                     # (T+1, N)

data     = load_from_array(prices, tickers)
analyzer = PortfolioAnalyzer(data.returns, data.tickers)
report   = analyzer.full_report()
```

---

### Option C — Pandas DataFrame  *(if you already have one)*

```python
import pandas as pd
from core.data_loader import load_from_dataframe
from core.analyzer    import PortfolioAnalyzer

df = pd.read_parquet("prices.parquet")   # index=dates, columns=tickers
# OR: df = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

data     = load_from_dataframe(df)
analyzer = PortfolioAnalyzer(data.returns, data.tickers)
report   = analyzer.full_report()
```

---

## Running the REST API

```bash
pip install fastapi uvicorn python-multipart

# start server
uvicorn api.main:app --reload --port 8000
```

Then upload your CSV and run analysis:
```bash
# 1. Upload prices
curl -X POST http://localhost:8000/upload \
  -F "file=@your_prices.csv"
# → returns {"session_id": "abc-123", ...}

# 2. Run full analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "horizons": ["tactical","cyclical","macro"]}'

# 3. Fetch correlation matrix
curl "http://localhost:8000/correlation?session_id=abc-123&horizon=tactical"

# 4. Single-pair deep dive
curl -X POST http://localhost:8000/pairs \
  -H "Content-Type: application/json" \
  -d '{"session_id":"abc-123","ticker_a":"RELIANCE","ticker_b":"TCS"}'

# 5. Regime timeline
curl "http://localhost:8000/regime?session_id=abc-123"
```

---

## Key output fields

| Field | Meaning |
|---|---|
| `avg_correlation` | Mean pairwise correlation (0.35 = moderate, 0.6 = high) |
| `effective_rank` | True diversification count (max = N assets) |
| `diversification_pct` | `effective_rank / N × 100` |
| `signal_factors` | Eigenvalues above Marchenko-Pastur bulk |
| `condition_number` | Matrix conditioning — lower is better |
| `betti_0 / betti_1` | Topological cluster / loop count |
| `tail_dependence_lambda` | Joint crash probability parameter (0–1) |

---

## Data requirements

| Requirement | Value |
|---|---|
| Max assets | 300 |
| Min observations | 20 days |
| Recommended for Tactical | ≥ 60 days |
| Recommended for Cyclical | ≥ 252 days (1 year) |
| Recommended for Macro | ≥ 1260 days (5 years) |
| Price type | Closing prices (adjusted preferred) |
| Missing data | Auto forward-filled |

---

## Running tests

```bash
pip install pytest numpy scipy pandas
python -m pytest tests/test_all.py -v
# 39 passed
```
