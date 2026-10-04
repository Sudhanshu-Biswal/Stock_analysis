"""
analytics/ratios.py — Valuation ratio calculations

Each ratio function returns a MetricBand containing:
  - current_value   : the ratio at the current stock price
  - historical_min  : minimum over available annual history (up to 5 years)
  - historical_max  : maximum over available annual history
  - historical_median: median over available annual history

This "currently at X vs historical range Y–Z" pattern is what makes
valuation ratios useful rather than just naked numbers.

Design rules (same as fundamentals.py):
  - No crashes on missing data — return None, never fabricate.
  - No LLM calls.
  - All division is guarded through _safe_div.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from data.models import CompanyData, MetricBand, MetricValue

logger = logging.getLogger(__name__)

_SOURCE = "yfinance"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(
    numerator: Optional[float],
    denominator: Optional[float],
    ticker: str,
    metric: str,
) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        logger.debug(f"[{ticker}] {metric}: denominator is zero, returning None")
        return None
    return numerator / denominator


def _band(
    ticker: str,
    metric: str,
    current: Optional[float],
    historical: list[Optional[float]],
    as_of_date: Optional[str],
) -> MetricBand:
    """Build a MetricBand from a current value and a list of historical values."""
    valid = [v for v in historical if v is not None and not np.isnan(v)]
    return MetricBand(
        ticker=ticker,
        metric=metric,
        current_value=round(current, 4) if current is not None else None,
        historical_min=round(float(np.min(valid)), 4) if valid else None,
        historical_max=round(float(np.max(valid)), 4) if valid else None,
        historical_median=round(float(np.median(valid)), 4) if valid else None,
        periods_used=len(valid),
        as_of_date=as_of_date,
        source=_SOURCE,
    )


def _current_price(data: CompanyData, metric: str) -> Optional[float]:
    if data.current_price is None:
        logger.warning(f"[{data.ticker}] {metric}: current_price is None")
    return data.current_price


def _most_recent_date(data: CompanyData) -> Optional[str]:
    if data.income_annual:
        return data.income_annual[0].as_of_date
    return None


# ---------------------------------------------------------------------------
# P/E Ratio (trailing)
# ---------------------------------------------------------------------------

def pe_ratio(data: CompanyData) -> MetricBand:
    """
    Trailing P/E = Price / Trailing EPS (TTM)

    Historical: computed from annual EPS over up to 5 years,
    paired with the closing price at each year-end (approximated
    by using current price only — year-end prices are not in yfinance
    fundamentals, so we compute the ratio from annual EPS × current_price
    for the historical band, noting this is an approximation).

    Current value uses trailing_eps from yfinance info (TTM).
    """
    price = _current_price(data, "pe_ratio")
    current_pe = _safe_div(price, data.trailing_eps, data.ticker, "pe_ratio")

    # Historical PE: price / annual EPS for each year
    # NOTE: We use the *reported* annual EPS; without historical prices
    # we can only approximate. This is the best we can do without a
    # premium data source.
    historical: list[Optional[float]] = []
    for row in data.income_annual[:5]:
        eps = row.diluted_eps or row.basic_eps
        val = _safe_div(price, eps, data.ticker, "pe_ratio_hist")
        historical.append(val)

    return _band(data.ticker, "pe_ratio", current_pe, historical, _most_recent_date(data))


# ---------------------------------------------------------------------------
# Forward P/E
# ---------------------------------------------------------------------------

def forward_pe(data: CompanyData) -> MetricValue:
    """
    Forward P/E = Current Price / Forward EPS (analyst consensus from yfinance info)

    No historical band — forward estimates aren't stored historically.
    """
    price = _current_price(data, "forward_pe")
    value = _safe_div(price, data.forward_eps, data.ticker, "forward_pe")
    return MetricValue(
        ticker=data.ticker,
        metric="forward_pe",
        value=round(value, 4) if value is not None else None,
        period="NTM",
        as_of_date=_most_recent_date(data),
        source=_SOURCE,
        note="based on forward EPS from yfinance consensus" if value is not None else "forward_eps unavailable",
    )


# ---------------------------------------------------------------------------
# PEG Ratio
# ---------------------------------------------------------------------------

def peg_ratio(data: CompanyData) -> MetricValue:
    """
    PEG = Trailing P/E / 5-year EPS CAGR

    EPS growth rate is computed from the oldest to most-recent annual diluted EPS.
    Returns None if fewer than 2 years of EPS data.
    """
    price = _current_price(data, "peg_ratio")

    # Compute PE
    current_pe = _safe_div(price, data.trailing_eps, data.ticker, "peg_ratio")
    if current_pe is None:
        return MetricValue(
            ticker=data.ticker, metric="peg_ratio", value=None,
            as_of_date=_most_recent_date(data), source=_SOURCE,
            note="trailing P/E unavailable",
        )

    # Compute 5-yr EPS CAGR
    eps_rows = [(r.diluted_eps or r.basic_eps) for r in data.income_annual[:5]]
    valid_eps = [(i, v) for i, v in enumerate(eps_rows) if v is not None and v > 0]

    if len(valid_eps) < 2:
        return MetricValue(
            ticker=data.ticker, metric="peg_ratio", value=None,
            as_of_date=_most_recent_date(data), source=_SOURCE,
            note="insufficient EPS history (need ≥ 2 positive EPS years)",
        )

    newest_idx, newest_eps = valid_eps[0]
    oldest_idx, oldest_eps = valid_eps[-1]
    n_years = oldest_idx - newest_idx  # number of periods between them

    if n_years <= 0 or oldest_eps <= 0:
        return MetricValue(
            ticker=data.ticker, metric="peg_ratio", value=None,
            as_of_date=_most_recent_date(data), source=_SOURCE,
            note="cannot compute EPS CAGR (non-positive base or same period)",
        )

    eps_cagr = (newest_eps / oldest_eps) ** (1 / n_years) - 1
    # PEG is PE divided by growth rate expressed as a percentage
    if eps_cagr <= 0:
        return MetricValue(
            ticker=data.ticker, metric="peg_ratio", value=None,
            as_of_date=_most_recent_date(data), source=_SOURCE,
            note="negative EPS CAGR — PEG not meaningful",
        )

    peg = current_pe / (eps_cagr * 100)
    return MetricValue(
        ticker=data.ticker, metric="peg_ratio",
        value=round(peg, 4),
        period=data.income_annual[0].period if data.income_annual else None,
        as_of_date=_most_recent_date(data),
        source=_SOURCE,
        note=f"EPS CAGR used: {eps_cagr:.2%} over {n_years} years",
    )


# ---------------------------------------------------------------------------
# EV / EBITDA
# ---------------------------------------------------------------------------

def ev_to_ebitda(data: CompanyData) -> MetricBand:
    """
    EV/EBITDA = Enterprise Value / EBITDA
    EV = Market Cap + Total Debt − Cash & Equivalents
    """
    price = _current_price(data, "ev_to_ebitda")
    shares = data.shares_outstanding

    balance_rows = data.balance_annual
    income_rows = data.income_annual

    if not balance_rows or not income_rows:
        return _band(data.ticker, "ev_to_ebitda", None, [], _most_recent_date(data))

    # Current EV
    latest_balance = balance_rows[0]
    market_cap = data.market_cap or (price * shares if price and shares else None)
    debt = latest_balance.total_debt
    cash = latest_balance.cash_and_equivalents
    net_debt = (debt or 0) - (cash or 0)
    ev = (market_cap + net_debt) if market_cap is not None else None

    current_ebitda = income_rows[0].ebitda
    current_ev_ebitda = _safe_div(ev, current_ebitda, data.ticker, "ev_to_ebitda")

    # Historical: approximate EV by using current market cap (no historical prices)
    historical: list[Optional[float]] = []
    for i, income_row in enumerate(income_rows[:5]):
        ebitda = income_row.ebitda
        if i < len(balance_rows):
            b_debt = balance_rows[i].total_debt
            b_cash = balance_rows[i].cash_and_equivalents
            h_net_debt = (b_debt or 0) - (b_cash or 0)
            h_ev = (market_cap + h_net_debt) if market_cap is not None else None
        else:
            h_ev = ev
        val = _safe_div(h_ev, ebitda, data.ticker, "ev_to_ebitda_hist")
        historical.append(val)

    return _band(data.ticker, "ev_to_ebitda", current_ev_ebitda, historical, _most_recent_date(data))


# ---------------------------------------------------------------------------
# Price to Book
# ---------------------------------------------------------------------------

def price_to_book(data: CompanyData) -> MetricBand:
    """Price / Book Value per Share"""
    price = _current_price(data, "price_to_book")
    balance_rows = data.balance_annual

    if not balance_rows:
        return _band(data.ticker, "price_to_book", None, [], None)

    current_pb = _safe_div(price, balance_rows[0].book_value_per_share, data.ticker, "price_to_book")

    historical: list[Optional[float]] = []
    for row in balance_rows[:5]:
        val = _safe_div(price, row.book_value_per_share, data.ticker, "price_to_book_hist")
        historical.append(val)

    return _band(data.ticker, "price_to_book", current_pb, historical, _most_recent_date(data))


# ---------------------------------------------------------------------------
# FCF Yield
# ---------------------------------------------------------------------------

def fcf_yield(data: CompanyData) -> MetricBand:
    """
    FCF Yield = Free Cash Flow / Market Cap

    A higher FCF yield means the company generates more cash relative to
    its price — generally better for value investors.
    """
    market_cap = data.market_cap
    if market_cap is None:
        price = data.current_price
        shares = data.shares_outstanding
        market_cap = (price * shares) if price and shares else None

    cashflow_rows = data.cashflow_annual

    if not cashflow_rows:
        return _band(data.ticker, "fcf_yield", None, [], _most_recent_date(data))

    current_fcf = cashflow_rows[0].free_cash_flow
    if current_fcf is None and cashflow_rows[0].operating_cash_flow is not None and cashflow_rows[0].capital_expenditure is not None:
        current_fcf = cashflow_rows[0].operating_cash_flow + cashflow_rows[0].capital_expenditure

    current_yield = _safe_div(current_fcf, market_cap, data.ticker, "fcf_yield")

    historical: list[Optional[float]] = []
    for row in cashflow_rows[:5]:
        cf = row.free_cash_flow
        if cf is None and row.operating_cash_flow is not None and row.capital_expenditure is not None:
            cf = row.operating_cash_flow + row.capital_expenditure
        val = _safe_div(cf, market_cap, data.ticker, "fcf_yield_hist")
        historical.append(val)

    return _band(data.ticker, "fcf_yield", current_yield, historical, _most_recent_date(data))


# ---------------------------------------------------------------------------
# Dividend Yield
# ---------------------------------------------------------------------------

def dividend_yield(data: CompanyData) -> MetricValue:
    """Dividend Yield from yfinance info (trailing 12-month dividends / price)."""
    value = data.dividend_yield
    return MetricValue(
        ticker=data.ticker,
        metric="dividend_yield",
        value=round(value, 6) if value is not None else None,
        period="TTM",
        as_of_date=_most_recent_date(data),
        source=_SOURCE,
        note=None if value is not None else "dividend_yield not reported by yfinance",
    )


# ---------------------------------------------------------------------------
# Convenience: compute all ratios at once
# ---------------------------------------------------------------------------

def all_ratios(data: CompanyData) -> dict[str, object]:
    """
    Compute all valuation ratios and return as a dict.
    Keys map to either a MetricBand (for ratio with historical bands)
    or a MetricValue (for ratios without bands like forward_pe, peg, dividend_yield).
    """
    return {
        "pe_ratio": pe_ratio(data),
        "forward_pe": forward_pe(data),
        "peg_ratio": peg_ratio(data),
        "ev_to_ebitda": ev_to_ebitda(data),
        "price_to_book": price_to_book(data),
        "fcf_yield": fcf_yield(data),
        "dividend_yield": dividend_yield(data),
    }
