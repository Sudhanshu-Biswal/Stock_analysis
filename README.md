# Investment Research System — Phase 1

Pure-Python investment research backend. No LLMs. Fetches financial data, computes fundamental ratios, and runs 2-stage DCF valuations for US and Indian stocks.

## Stack
- `yfinance` — financial data (US + NSE tickers)
- `pandas` / `numpy` — data handling
- `FastAPI` — REST API
- `SQLite` via `SQLAlchemy` — caching
- `pytest` — test suite

## Setup

```bash
cd investment-ai
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the API

```bash
uvicorn api.main:app --reload --reload-exclude ".venv/*"
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

## Example API Calls

```bash
# Fetch company data (uses cache, 24h staleness)
curl http://localhost:8000/company/AAPL

# All fundamental metrics
curl http://localhost:8000/company/AAPL/fundamentals

# Valuation ratios + historical bands
curl http://localhost:8000/company/AAPL/ratios

# DCF valuation
curl -X POST http://localhost:8000/company/AAPL/dcf \
  -H "Content-Type: application/json" \
  -d '{"growth_rate": 0.10, "terminal_growth": 0.03, "years": 5}'

# Reverse DCF — what growth rate is the market pricing in?
curl http://localhost:8000/company/AAPL/reverse-dcf

# Same for Indian ticker (NSE)
curl http://localhost:8000/company/RELIANCE.NS/fundamentals
```

## Run Tests

```bash
# Unit tests only (no network calls, fast)
pytest tests/ -m "not integration" -v

# All tests including live yfinance calls
pytest tests/ -v
```

## Folder Structure

```
investment-ai/
├── data/
│   ├── models.py       # CompanyData Pydantic model (shared contract)
│   ├── fetcher.py      # yfinance wrapper, normalises US vs India tickers
│   └── cache.py        # SQLite read/write with staleness check
├── analytics/
│   ├── fundamentals.py # ROE, ROCE, margins, growth rates, debt ratios
│   └── ratios.py       # P/E, PEG, EV/EBITDA, FCF yield + historical bands
├── valuation/
│   └── dcf.py          # 2-stage DCF + reverse DCF + CAPM WACC
├── api/
│   └── main.py         # FastAPI app
├── tests/
│   ├── test_dcf.py
│   ├── test_fundamentals.py
│   ├── test_ratios.py
│   └── test_integration.py  # @pytest.mark.integration (needs network)
└── requirements.txt
```

## Ticker Format

- **US stocks**: `AAPL`, `MSFT`, `GOOGL`
- **Indian NSE stocks**: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- **Indian BSE stocks**: `RELIANCE.BO`, `TCS.BO`

Bare Indian tickers without exchange suffix (e.g. just `RELIANCE`) will raise a `ValueError` — be explicit.

## Design Principles

1. **No fabricated data** — missing yfinance fields return `null`, never `0`
2. **Provenance on every number** — all metrics carry `period` and `as_of_date`
3. **Cache-first** — yfinance is never called more than once per staleness window
4. **No LLMs** — this layer is purely deterministic math, auditable by hand
