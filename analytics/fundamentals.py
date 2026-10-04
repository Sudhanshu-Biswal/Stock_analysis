"""
analytics/fundamentals.py — Pure fundamental metric calculations

All functions accept a CompanyData object and return either:
  - A single MetricValue dict: {"value": float|None, "period": str, "as_of_date": str, ...}
  - A list of MetricValue dicts (for multi-year metrics like revenue_growth_yoy)

Design rules:
  - NEVER crash on missing data — return None values with a "note" field.
  - NEVER fabricate a 0 or any placeholder — if data is missing, value is None.
  - Every returned number carries period and as_of_date for full provenance.
  - All math is pure Python / numpy — no LLM calls.
"""

from __future__ import annotations

import logging
from typing import Optional

from data.models import (
    BalanceRow,
    CashFlowRow,
    CompanyData,
    IncomeRow,
    MetricValue,
)

logger = logging.getLogger(__name__)

_SOURCE = "yfinance"


# ---------------------------------------------------------------------------
# Helper — safe division
# ---------------------------------------------------------------------------

def _safe_div(
    numerator: Optional[float],
    denominator: Optional[float],
    ticker: str,
    metric: str,
) -> Optional[float]:
    """Returns numerator/denominator, or None if either is None or denominator is 0."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        logger.debug(f"[{ticker}] {metric}: denominator is zero, returning None")
        return None
    return numerator / denominator


def _make_metric(
    ticker: str,
    metric: str,
    value: Optional[float],
    period: Optional[str],
    as_of_date: Optional[str],
    note: Optional[str] = None,
) -> MetricValue:
    return MetricValue(
        ticker=ticker,
        metric=metric,
        value=value,
        period=period,
        as_of_date=as_of_date,
        source=_SOURCE,
        note=note,
    )


# ---------------------------------------------------------------------------
# Revenue growth YoY — returns list, last N years
# ---------------------------------------------------------------------------

def revenue_growth_yoy(data: CompanyData, max_years: int = 5) -> list[MetricValue]:
    """
    Year-over-year revenue growth for up to max_years periods.

    Returns a list ordered from most-recent to oldest. Each entry covers
    the growth from the previous year to the labelled year.

    Requires at least 2 years of revenue data to compute even one growth rate.
    """
    rows = data.income_annual
    if not rows:
        logger.warning(f"[{data.ticker}] revenue_growth_yoy: no annual income data")
        return []

    # rows are already most-recent first from fetcher
    results: list[MetricValue] = []

    # We can compute growth for rows[0..n-2] compared to rows[1..n-1]
    # (rows[0] is newest, rows[1] is one year older)
    for i in range(min(len(rows) - 1, max_years)):
        current = rows[i]
        prior = rows[i + 1]
        if current.revenue is None or prior.revenue is None or prior.revenue == 0:
            results.append(
                _make_metric(
                    data.ticker, "revenue_growth_yoy", None,
                    current.period, current.as_of_date,
                    note="missing revenue data for this or prior year",
                )
            )
        else:
            growth = (current.revenue - prior.revenue) / prior.revenue
            results.append(
                _make_metric(
                    data.ticker, "revenue_growth_yoy",
                    round(growth, 6),
                    current.period, current.as_of_date,
                )
            )

    return results


# ---------------------------------------------------------------------------
# ROE — Return on Equity
# ---------------------------------------------------------------------------

def roe(data: CompanyData) -> MetricValue:
    """
    Return on Equity = Net Income / Average Shareholders' Equity

    Uses the most recent annual income row and the average of the two most
    recent annual balance sheet rows (beginning + ending equity / 2).
    Falls back to ending equity alone if only one balance sheet year available.
    """
    income_rows = data.income_annual
    balance_rows = data.balance_annual

    if not income_rows:
        return _make_metric(data.ticker, "roe", None, None, None, note="no income data")

    latest_income = income_rows[0]
    net_income = latest_income.net_income

    if not balance_rows:
        return _make_metric(data.ticker, "roe", None, latest_income.period, latest_income.as_of_date, note="no balance sheet data")

    equity_end = balance_rows[0].total_equity
    equity_beg = balance_rows[1].total_equity if len(balance_rows) > 1 else None

    if equity_beg is not None and equity_end is not None:
        avg_equity = (equity_beg + equity_end) / 2
    else:
        avg_equity = equity_end

    value = _safe_div(net_income, avg_equity, data.ticker, "roe")
    return _make_metric(
        data.ticker, "roe", round(value, 6) if value is not None else None,
        latest_income.period, latest_income.as_of_date,
    )


# ---------------------------------------------------------------------------
# ROCE — Return on Capital Employed
# ---------------------------------------------------------------------------

def roce(data: CompanyData) -> MetricValue:
    """
    Return on Capital Employed = EBIT / Capital Employed
    Capital Employed = Total Assets − Current Liabilities
    """
    income_rows = data.income_annual
    balance_rows = data.balance_annual

    if not income_rows:
        return _make_metric(data.ticker, "roce", None, None, None, note="no income data")

    latest_income = income_rows[0]
    ebit = latest_income.ebit

    if not balance_rows:
        return _make_metric(data.ticker, "roce", None, latest_income.period, latest_income.as_of_date, note="no balance sheet data")

    latest_balance = balance_rows[0]
    total_assets = latest_balance.total_assets
    current_liabilities = latest_balance.current_liabilities

    if total_assets is None or current_liabilities is None:
        return _make_metric(
            data.ticker, "roce", None,
            latest_income.period, latest_income.as_of_date,
            note="missing total_assets or current_liabilities",
        )

    capital_employed = total_assets - current_liabilities
    value = _safe_div(ebit, capital_employed, data.ticker, "roce")
    return _make_metric(
        data.ticker, "roce", round(value, 6) if value is not None else None,
        latest_income.period, latest_income.as_of_date,
    )


# ---------------------------------------------------------------------------
# Operating Margin
# ---------------------------------------------------------------------------

def operating_margin(data: CompanyData) -> MetricValue:
    """Operating Margin = EBIT / Revenue"""
    income_rows = data.income_annual
    if not income_rows:
        return _make_metric(data.ticker, "operating_margin", None, None, None, note="no income data")

    latest = income_rows[0]
    value = _safe_div(latest.ebit, latest.revenue, data.ticker, "operating_margin")
    return _make_metric(
        data.ticker, "operating_margin",
        round(value, 6) if value is not None else None,
        latest.period, latest.as_of_date,
    )


# ---------------------------------------------------------------------------
# Net Margin
# ---------------------------------------------------------------------------

def net_margin(data: CompanyData) -> MetricValue:
    """Net Margin = Net Income / Revenue"""
    income_rows = data.income_annual
    if not income_rows:
        return _make_metric(data.ticker, "net_margin", None, None, None, note="no income data")

    latest = income_rows[0]
    value = _safe_div(latest.net_income, latest.revenue, data.ticker, "net_margin")
    return _make_metric(
        data.ticker, "net_margin",
        round(value, 6) if value is not None else None,
        latest.period, latest.as_of_date,
    )


# ---------------------------------------------------------------------------
# Debt to Equity
# ---------------------------------------------------------------------------

def debt_to_equity(data: CompanyData) -> MetricValue:
    """Debt / Equity = Total Debt / Total Shareholders' Equity"""
    balance_rows = data.balance_annual
    if not balance_rows:
        return _make_metric(data.ticker, "debt_to_equity", None, None, None, note="no balance sheet data")

    latest = balance_rows[0]
    value = _safe_div(latest.total_debt, latest.total_equity, data.ticker, "debt_to_equity")
    return _make_metric(
        data.ticker, "debt_to_equity",
        round(value, 6) if value is not None else None,
        latest.period, latest.as_of_date,
    )


# ---------------------------------------------------------------------------
# Interest Coverage
# ---------------------------------------------------------------------------

def interest_coverage(data: CompanyData) -> MetricValue:
    """
    Interest Coverage = EBIT / Interest Expense

    Note: yfinance returns interest_expense as a negative number.
    We take abs() so the ratio is positive for a company that can cover interest.
    """
    income_rows = data.income_annual
    if not income_rows:
        return _make_metric(data.ticker, "interest_coverage", None, None, None, note="no income data")

    latest = income_rows[0]
    ebit = latest.ebit
    interest = latest.interest_expense

    if interest is not None:
        interest = abs(interest)   # normalise sign
        if interest == 0:
            return _make_metric(
                data.ticker, "interest_coverage", None,
                latest.period, latest.as_of_date,
                note="interest expense is zero (no debt or not reported)",
            )

    value = _safe_div(ebit, interest, data.ticker, "interest_coverage")
    return _make_metric(
        data.ticker, "interest_coverage",
        round(value, 6) if value is not None else None,
        latest.period, latest.as_of_date,
    )


# ---------------------------------------------------------------------------
# FCF — Free Cash Flow
# ---------------------------------------------------------------------------

def fcf(data: CompanyData) -> MetricValue:
    """
    Free Cash Flow = Operating Cash Flow − Capital Expenditure

    Uses the most recent annual cash flow row. If the FCF was already
    computed and stored in the row (from fetcher), uses that; otherwise
    computes from components.

    Note: capex in yfinance is already negative, so:
        FCF = operating_cf + capex  (both signs already set correctly)
    """
    cashflow_rows = data.cashflow_annual
    if not cashflow_rows:
        return _make_metric(data.ticker, "fcf", None, None, None, note="no cash flow data")

    latest = cashflow_rows[0]

    # Use pre-computed FCF if available
    if latest.free_cash_flow is not None:
        return _make_metric(
            data.ticker, "fcf", latest.free_cash_flow,
            latest.period, latest.as_of_date,
        )

    # Otherwise compute from components
    ocf = latest.operating_cash_flow
    capex = latest.capital_expenditure

    if ocf is None:
        return _make_metric(
            data.ticker, "fcf", None, latest.period, latest.as_of_date,
            note="missing operating_cash_flow",
        )
    if capex is None:
        return _make_metric(
            data.ticker, "fcf", None, latest.period, latest.as_of_date,
            note="missing capital_expenditure",
        )

    # capex is stored as negative in yfinance; adding gives OCF - |capex|
    computed_fcf = ocf + capex
    return _make_metric(
        data.ticker, "fcf", computed_fcf,
        latest.period, latest.as_of_date,
    )


# ---------------------------------------------------------------------------
# Convenience: compute all fundamentals at once
# ---------------------------------------------------------------------------

def all_fundamentals(data: CompanyData) -> dict[str, object]:
    """
    Returns a dict with all fundamental metrics computed in one call.
    The revenue_growth_yoy key maps to a list; all others map to a single MetricValue.
    """
    return {
        "revenue_growth_yoy": revenue_growth_yoy(data),
        "roe": roe(data),
        "roce": roce(data),
        "operating_margin": operating_margin(data),
        "net_margin": net_margin(data),
        "debt_to_equity": debt_to_equity(data),
        "interest_coverage": interest_coverage(data),
        "fcf": fcf(data),
    }
