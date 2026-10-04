"""
Shared Pydantic data models — the typed contract between all layers.

Nothing in fetcher, analytics, or API should pass raw dicts; everything
goes through CompanyData so downstream code has guaranteed field names.
Missing values are None, never fabricated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Per-period financial statement rows
# ---------------------------------------------------------------------------

class IncomeRow(BaseModel):
    """One fiscal-year (or quarter) slice of the income statement."""

    period: str                          # e.g. "FY2024" or "Q3 FY2024"
    as_of_date: str                      # ISO date string from yfinance index
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    ebit: Optional[float] = None         # Operating income / EBIT
    ebitda: Optional[float] = None
    net_income: Optional[float] = None
    interest_expense: Optional[float] = None
    tax_provision: Optional[float] = None
    basic_eps: Optional[float] = None
    diluted_eps: Optional[float] = None
    shares_outstanding_reported: Optional[float] = None


class BalanceRow(BaseModel):
    """One fiscal-year (or quarter) slice of the balance sheet."""

    period: str
    as_of_date: str
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None   # short-term + long-term debt
    cash_and_equivalents: Optional[float] = None
    book_value_per_share: Optional[float] = None


class CashFlowRow(BaseModel):
    """One fiscal-year (or quarter) slice of the cash flow statement."""

    period: str
    as_of_date: str
    operating_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None   # stored as negative in yfinance; normalised to negative here
    free_cash_flow: Optional[float] = None        # operating_cf + capex (capex is negative, so subtraction)


# ---------------------------------------------------------------------------
# Top-level CompanyData model
# ---------------------------------------------------------------------------

class CompanyData(BaseModel):
    """
    The single typed contract passed between all layers of the system.

    All financial figures are in the company's reporting currency
    (USD for US tickers, INR for .NS tickers).

    missing_fields carries an audit trail of every field yfinance
    failed to return — consumers should check before using a value.
    """

    # Identity
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market: Literal["US", "IN"] = "US"   # detected from ticker suffix
    currency: Optional[str] = None

    # Current market data
    current_price: Optional[float] = None
    shares_outstanding: Optional[float] = None   # in units (not millions)
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    forward_eps: Optional[float] = None
    trailing_eps: Optional[float] = None
    dividend_yield: Optional[float] = None       # fractional (0.015 = 1.5%)
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None

    # Annual financial statements (most-recent first)
    income_annual: list[IncomeRow] = Field(default_factory=list)
    balance_annual: list[BalanceRow] = Field(default_factory=list)
    cashflow_annual: list[CashFlowRow] = Field(default_factory=list)

    # Quarterly financial statements (most-recent first)
    income_quarterly: list[IncomeRow] = Field(default_factory=list)
    balance_quarterly: list[BalanceRow] = Field(default_factory=list)
    cashflow_quarterly: list[CashFlowRow] = Field(default_factory=list)

    # Audit trail
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    missing_fields: list[str] = Field(default_factory=list)
    source: str = "yfinance"


# ---------------------------------------------------------------------------
# Metric wrapper — the standard response envelope for all computed numbers
# ---------------------------------------------------------------------------

class MetricValue(BaseModel):
    """
    Standard envelope for a single computed metric.

    Every analytics function returns this (or a list of these) so the
    API layer always has a consistent structure to serialise.
    """

    ticker: str
    metric: str
    value: Optional[float] = None
    period: Optional[str] = None
    as_of_date: Optional[str] = None
    source: str = "yfinance"
    note: Optional[str] = None          # e.g. "missing data" or "TTM estimate"


class MetricBand(BaseModel):
    """Historical min/max/median band for a ratio."""

    ticker: str
    metric: str
    current_value: Optional[float] = None
    historical_min: Optional[float] = None
    historical_max: Optional[float] = None
    historical_median: Optional[float] = None
    periods_used: int = 0
    as_of_date: Optional[str] = None
    source: str = "yfinance"
