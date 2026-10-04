"""
api/main.py — FastAPI application

All endpoints use the cache layer (never call yfinance directly).
Response shapes follow the project's standard metric envelope pattern:
  {"ticker": ..., "metric": ..., "value": ..., "period": ..., "source": ..., "as_of_date": ...}

Endpoints:
  GET  /health
  GET  /company/{ticker}
  GET  /company/{ticker}/fundamentals
  GET  /company/{ticker}/ratios
  POST /company/{ticker}/dcf
  GET  /company/{ticker}/reverse-dcf
  POST /research/{ticker}
  POST /screen
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

load_dotenv()  # reads investment-ai/.env if present, sets os.environ

from analytics.fundamentals import all_fundamentals
from analytics.ratios import all_ratios
from data.cache import get_or_fetch
from data.models import CompanyData
from screening.criteria import (
    pe_below_historical_median,
    positive_fcf,
    reasonable_debt,
    reverse_dcf_implied_growth_below,
    revenue_growth_positive_recent,
    roe_above,
)
from screening.screen import run_screen
from screening.universes import get_universe
from synthesis.report import PHASE1_API_BASE, Phase1Unavailable, generate_report
from synthesis.schemas import ResearchReport
from valuation.dcf import DCFResult, ReverseResult, estimate_wacc, reverse_dcf, run_dcf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Investment Research API — Phase 1",
    description=(
        "Pure-Python, deterministic investment research API. "
        "No LLM calls. Fetches financial data via yfinance, computes "
        "fundamental ratios, and runs 2-stage DCF valuations. "
        "All numbers carry provenance (period + as_of_date + source)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DCFRequest(BaseModel):
    """Body for POST /company/{ticker}/dcf"""

    growth_rate: float = Field(
        ...,
        description="Stage 1 growth rate (fractional, e.g. 0.15 = 15%)",
        ge=-0.5, le=1.0,
    )
    terminal_growth: float = Field(
        0.03,
        description="Perpetuity growth rate after Stage 1 (fractional, e.g. 0.03 = 3%)",
        ge=-0.05, le=0.10,
    )
    years: int = Field(
        5,
        description="Number of years in the explicit forecast period",
        ge=1, le=20,
    )
    wacc: Optional[float] = Field(
        None,
        description=(
            "Manual WACC override (fractional). If omitted, estimated via CAPM "
            "using the stock's beta and market defaults."
        ),
        ge=0.01, le=0.50,
    )
    net_debt_override: Optional[float] = Field(
        None,
        description=(
            "Manual net-debt override. If omitted, computed from balance sheet "
            "(total_debt - cash). Positive = net debt, negative = net cash."
        ),
    )

    model_config = {"json_schema_extra": {
        "example": {
            "growth_rate": 0.10,
            "terminal_growth": 0.03,
            "years": 5,
        }
    }}


class ReverseDCFRequest(BaseModel):
    """Query params for GET /company/{ticker}/reverse-dcf"""

    terminal_growth: float = Field(0.03, ge=-0.05, le=0.10)
    years: int = Field(5, ge=1, le=20)
    wacc: Optional[float] = Field(None, ge=0.01, le=0.50)


class ResearchRequest(BaseModel):
    """Body for POST /research/{ticker} — optional DCF assumption overrides."""

    growth_rate: Optional[float] = Field(None, ge=-0.5, le=1.0)
    terminal_growth: Optional[float] = Field(None, ge=-0.05, le=0.10)
    years: Optional[int] = Field(None, ge=1, le=20)
    wacc: Optional[float] = Field(None, ge=0.01, le=0.50)


class ScreenRequest(BaseModel):
    """Body for POST /screen — universe + adjustable pass/fail thresholds."""

    universe: str = Field(..., description="'nifty50_sample' | 'us_megacap_sample' | 'custom'")
    custom_tickers: Optional[list[str]] = Field(None, description="Required if universe == 'custom'")
    min_pass_count: int = Field(3, ge=0, description="Shortlist = tickers passing at least this many criteria")
    max_debt_to_equity: float = Field(1.5, gt=0)
    min_roe: float = Field(0.15)
    max_implied_growth: float = Field(0.20, gt=0)
    max_pe_vs_median: bool = Field(True, description="Whether to require P/E below its own historical median")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_data(ticker: str) -> CompanyData:
    """Fetch with cache, raise 404 if ticker is completely unknown."""
    try:
        data = get_or_fetch(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"[{ticker}] Unexpected error fetching data: {exc}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Failed to fetch data for {ticker}: {exc}") from exc

    if data.current_price is None and not data.income_annual:
        raise HTTPException(
            status_code=404,
            detail=f"No data returned for ticker '{ticker}'. Check ticker format (e.g. RELIANCE.NS for NSE).",
        )
    return data


def _net_debt_from_balance(data: CompanyData) -> float:
    """Compute net debt from most-recent balance sheet row."""
    if not data.balance_annual:
        return 0.0
    b = data.balance_annual[0]
    debt = b.total_debt or 0.0
    cash = b.cash_and_equivalents or 0.0
    return debt - cash


def _fcf_base_from_data(data: CompanyData) -> Optional[float]:
    """Return most-recent annual FCF for DCF base."""
    if not data.cashflow_annual:
        return None
    row = data.cashflow_annual[0]
    if row.free_cash_flow is not None:
        return row.free_cash_flow
    if row.operating_cash_flow is not None and row.capital_expenditure is not None:
        return row.operating_cash_flow + row.capital_expenditure
    return None


def _ttm_fcf_from_data(data: CompanyData) -> Optional[tuple[float, str]]:
    """Sum FCF over the last 4 quarters. Returns (ttm_fcf, period_label), or
    None if fewer than 4 quarters are available or any quarter is missing FCF."""
    quarters = data.cashflow_quarterly[:4]
    if len(quarters) < 4:
        return None

    total = 0.0
    for q in quarters:
        fcf = q.free_cash_flow
        if fcf is None and q.operating_cash_flow is not None and q.capital_expenditure is not None:
            fcf = q.operating_cash_flow + q.capital_expenditure
        if fcf is None:
            return None
        total += fcf

    period_label = f"TTM through {quarters[0].as_of_date}"
    return total, period_label


def _fcf_base_for_dcf(data: CompanyData) -> tuple[Optional[float], str, str]:
    """
    Returns (fcf_base, method, period_label) for use as the DCF FCF base.

    Prefers TTM (sum of the last 4 quarters) when fully available -- more
    current than a stale fiscal-year figure. Falls back to the most recent
    annual FCF when quarterly data is incomplete.
    """
    ttm = _ttm_fcf_from_data(data)
    if ttm is not None:
        fcf_base, period_label = ttm
        return fcf_base, "TTM (trailing 4 quarters)", period_label

    fcf_base = _fcf_base_from_data(data)
    period_label = data.cashflow_annual[0].period if data.cashflow_annual else "unknown"
    return fcf_base, "annual (most recent fiscal year)", period_label


def _most_recent_as_of_date(data: CompanyData) -> Optional[str]:
    """Most recent as_of_date across annual and quarterly income data --
    quarterly is usually fresher, so don't just report the fiscal year end."""
    candidates = [row.as_of_date for row in (data.income_quarterly[:1] + data.income_annual[:1])]
    return max(candidates) if candidates else None


def _wacc_inputs_from_data(data: CompanyData) -> dict:
    """Pull the fields estimate_wacc() needs from the latest company financials."""
    total_debt = None
    cash = None
    if data.balance_annual:
        b = data.balance_annual[0]
        total_debt = b.total_debt
        cash = b.cash_and_equivalents

    interest_expense = None
    tax_provision = None
    pretax_income = None
    if data.income_annual:
        i = data.income_annual[0]
        interest_expense = i.interest_expense
        tax_provision = i.tax_provision
        if i.net_income is not None and i.tax_provision is not None:
            pretax_income = i.net_income + i.tax_provision

    return {
        "market_cap": data.market_cap,
        "total_debt": total_debt,
        "cash_and_equivalents": cash,
        "interest_expense": interest_expense,
        "tax_provision": tax_provision,
        "pretax_income": pretax_income,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "version": app.version}


@app.get(
    "/company/{ticker}",
    response_model=CompanyData,
    tags=["Company"],
    summary="Fetch raw company financial data (cached)",
)
def get_company(
    ticker: str = Path(..., description="US ticker (e.g. AAPL) or Indian ticker with exchange suffix (e.g. RELIANCE.NS)"),
) -> CompanyData:
    """
    Returns the full CompanyData object including all financial statement rows.
    Data is cached for 24 hours; use ?force_refresh=true to bypass (not yet implemented).
    """
    return _get_data(ticker)


@app.get(
    "/company/{ticker}/fundamentals",
    tags=["Analytics"],
    summary="Compute all fundamental metrics",
)
def get_fundamentals(
    ticker: str = Path(..., description="Stock ticker"),
) -> dict[str, Any]:
    """
    Returns all fundamental metrics: ROE, ROCE, margins, revenue growth,
    debt ratios, FCF. Each metric includes period and as_of_date provenance.

    revenue_growth_yoy returns a list (one per year).
    All others return a single MetricValue object.
    """
    data = _get_data(ticker)
    result = all_fundamentals(data)

    # Serialise Pydantic models to dicts for consistent JSON output
    serialised: dict[str, Any] = {}
    for key, val in result.items():
        if isinstance(val, list):
            serialised[key] = [v.model_dump() for v in val]
        else:
            serialised[key] = val.model_dump()

    return {
        "ticker": ticker,
        "source": "yfinance",
        "as_of_date": _most_recent_as_of_date(data),
        "metrics": serialised,
    }


@app.get(
    "/company/{ticker}/ratios",
    tags=["Analytics"],
    summary="Compute valuation ratios with historical bands",
)
def get_ratios(
    ticker: str = Path(..., description="Stock ticker"),
) -> dict[str, Any]:
    """
    Returns valuation ratios (P/E, PEG, EV/EBITDA, P/B, FCF yield, dividend yield).
    Ratios with historical data include min/max/median over up to 5 years.
    """
    data = _get_data(ticker)
    result = all_ratios(data)

    serialised: dict[str, Any] = {k: v.model_dump() for k, v in result.items()}

    return {
        "ticker": ticker,
        "current_price": data.current_price,
        "source": "yfinance",
        "as_of_date": _most_recent_as_of_date(data),
        "ratios": serialised,
    }


@app.post(
    "/company/{ticker}/dcf",
    response_model=DCFResult,
    tags=["Valuation"],
    summary="Run a 2-stage DCF valuation",
)
def post_dcf(
    ticker: str = Path(..., description="Stock ticker"),
    body: DCFRequest = ...,
) -> DCFResult:
    """
    Runs a 2-stage DCF valuation using the provided assumptions.

    - Stage 1: Explicit FCF projections for `years` with growth declining
      linearly from `growth_rate` to `terminal_growth`.
    - Stage 2: Gordon Growth Model terminal value.

    WACC is estimated via CAPM if not provided. FCF base prefers trailing-twelve-
    month FCF (summed from the last 4 quarters) over the most recent fiscal
    year, falling back to annual when quarterly data is incomplete -- the
    method actually used is echoed in `.assumptions.fcf_base_method`.

    All input assumptions are echoed back in the response `.assumptions` field.
    """
    data = _get_data(ticker)

    fcf_base, fcf_base_method, fcf_base_period = _fcf_base_for_dcf(data)
    if fcf_base is None:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot run DCF for {ticker}: no FCF data available from yfinance.",
        )

    shares = data.shares_outstanding
    if not shares or shares <= 0:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot run DCF for {ticker}: shares_outstanding is missing or zero.",
        )

    # WACC
    wacc_detail: Optional[dict] = None
    if body.wacc is not None:
        wacc = body.wacc
    else:
        wacc_detail = estimate_wacc(
            beta=data.beta,
            market=data.market,
            **_wacc_inputs_from_data(data),
        )
        wacc = wacc_detail["wacc"]

    # Net debt
    net_debt = body.net_debt_override if body.net_debt_override is not None else _net_debt_from_balance(data)

    try:
        result = run_dcf(
            fcf_base=fcf_base,
            growth_rate_stage1=body.growth_rate,
            years_stage1=body.years,
            terminal_growth_rate=body.terminal_growth,
            wacc=wacc,
            shares_outstanding=shares,
            net_debt=net_debt,
            current_price=data.current_price,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Enrich assumptions with computed fields
    result.assumptions["ticker"] = ticker
    if body.wacc is not None:
        result.assumptions["wacc_source"] = "user_override"
    else:
        result.assumptions["wacc_source"] = "market-value-weighted CAPM+debt"
        result.assumptions["wacc_detail"] = wacc_detail
    result.assumptions["fcf_base_method"] = fcf_base_method
    result.assumptions["fcf_base_period"] = fcf_base_period

    return result


@app.get(
    "/company/{ticker}/reverse-dcf",
    response_model=ReverseResult,
    tags=["Valuation"],
    summary="Implied growth rate at current market price",
)
def get_reverse_dcf(
    ticker: str = Path(..., description="Stock ticker"),
    terminal_growth: float = Query(0.03, ge=-0.05, le=0.10, description="Perpetuity growth rate"),
    years: int = Query(5, ge=1, le=20, description="Stage 1 forecast years"),
    wacc: Optional[float] = Query(None, ge=0.01, le=0.50, description="Manual WACC override"),
) -> ReverseResult:
    """
    Solves for the growth rate implied by the current stock price.

    This answers: "What growth rate is the market pricing in?"
    A very high implied growth rate suggests priced-for-perfection risk.
    """
    data = _get_data(ticker)

    if data.current_price is None:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot run reverse DCF for {ticker}: current_price is unavailable.",
        )

    fcf_base, fcf_base_method, fcf_base_period = _fcf_base_for_dcf(data)
    if fcf_base is None:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot run reverse DCF for {ticker}: no FCF data available.",
        )

    shares = data.shares_outstanding
    if not shares or shares <= 0:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot run reverse DCF for {ticker}: shares_outstanding missing.",
        )

    wacc_detail: Optional[dict] = None
    if wacc is not None:
        effective_wacc = wacc
    else:
        wacc_detail = estimate_wacc(
            beta=data.beta,
            market=data.market,
            **_wacc_inputs_from_data(data),
        )
        effective_wacc = wacc_detail["wacc"]
    net_debt = _net_debt_from_balance(data)

    try:
        result = reverse_dcf(
            current_price=data.current_price,
            fcf_base=fcf_base,
            years_stage1=years,
            terminal_growth_rate=terminal_growth,
            wacc=effective_wacc,
            shares_outstanding=shares,
            net_debt=net_debt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result.assumptions["ticker"] = ticker
    if wacc is not None:
        result.assumptions["wacc_source"] = "user_override"
    else:
        result.assumptions["wacc_source"] = "market-value-weighted CAPM+debt"
        result.assumptions["wacc_detail"] = wacc_detail
    result.assumptions["fcf_base_method"] = fcf_base_method
    result.assumptions["fcf_base_period"] = fcf_base_period

    return result


@app.post(
    "/research/{ticker}",
    response_model=ResearchReport,
    tags=["Synthesis"],
    summary="Generate an LLM-synthesized research report (Phase 2)",
)
def post_research(
    ticker: str = Path(..., description="Stock ticker"),
    body: ResearchRequest = ResearchRequest(),
) -> ResearchReport:
    """
    Combines Phase 1's computed numbers with retrieved news/filing context
    to produce a narrative research report via Claude. Every number in the
    output must trace back to Phase 1 or a retrieved, sourced document.
    """
    data = _get_data(ticker)  # reuse Phase 1 fetch/cache to validate ticker + detect market
    dcf_overrides = body.model_dump(exclude_none=True)

    try:
        report = generate_report(ticker, market=data.market, dcf_overrides=dcf_overrides or None)
    except Phase1Unavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return report


@app.post(
    "/screen",
    tags=["Screening"],
    summary="Screen a universe of tickers against deterministic criteria (Phase 3)",
)
def post_screen(body: ScreenRequest = ...) -> dict[str, Any]:
    """
    Pure deterministic screening -- no LLM calls, same auditable logic as
    Phase 1. Narrows a universe to a shortlist; never recommends a buy.
    Every result shows exactly which criteria passed/failed with real
    values, never a bare score.
    """
    if body.universe == "custom":
        if not body.custom_tickers:
            raise HTTPException(
                status_code=422,
                detail="custom_tickers is required when universe='custom'.",
            )
        universe = body.custom_tickers
    else:
        try:
            universe = get_universe(body.universe)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    criteria_fns: list = [
        lambda f, r, rd: positive_fcf(f),
        lambda f, r, rd: reasonable_debt(f, max_de=body.max_debt_to_equity),
        lambda f, r, rd: roe_above(f, min_roe=body.min_roe),
        lambda f, r, rd: revenue_growth_positive_recent(f),
        lambda f, r, rd: reverse_dcf_implied_growth_below(rd, max_growth=body.max_implied_growth),
    ]
    if body.max_pe_vs_median:
        criteria_fns.append(lambda f, r, rd: pe_below_historical_median(r))

    full_results = run_screen(universe, criteria_fns, PHASE1_API_BASE, max_workers=5)

    shortlist = [r for r in full_results if r.fetch_error is None and r.pass_count >= body.min_pass_count]
    errors = [r.ticker for r in full_results if r.fetch_error is not None]

    return {
        "shortlist": [r.model_dump() for r in shortlist],
        "full_results": [r.model_dump() for r in full_results],
        "universe_size": len(universe),
        "errors": errors,
    }
