"""
data/fetcher.py — yfinance wrapper

Fetches income statement, balance sheet, and cash flow data for any
US or NSE/BSE ticker and returns a typed CompanyData object.

Key design decisions:
- Never fabricates data: missing fields are None and logged in missing_fields.
- Accepts both AAPL (US) and RELIANCE.NS (India NSE) / RELIANCE.BO (India BSE).
- Bare Indian tickers without .NS or .BO suffix raise ValueError — be explicit.
- Column name normalisation handles yfinance's inconsistent naming across
  ticker types and yfinance versions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import yfinance as yf
import pandas as pd

from data.models import (
    BalanceRow,
    CashFlowRow,
    CompanyData,
    IncomeRow,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name aliases
# yfinance uses slightly different names across ticker regions and versions.
# Each alias list is tried in order; first match wins.
# ---------------------------------------------------------------------------

_INCOME_ALIASES: dict[str, list[str]] = {
    "revenue": [
        "Total Revenue",
        "TotalRevenue",
        "Revenue",
        "Net Revenue",
    ],
    "gross_profit": [
        "Gross Profit",
        "GrossProfit",
    ],
    "ebit": [
        "EBIT",
        "Operating Income",
        "OperatingIncome",
        "Ebit",
    ],
    "ebitda": [
        "EBITDA",
        "Ebitda",
        "Normalized EBITDA",
    ],
    "net_income": [
        "Net Income",
        "NetIncome",
        "Net Income Common Stockholders",
    ],
    "interest_expense": [
        "Interest Expense",
        "InterestExpense",
        "Interest Expense Non Operating",
    ],
    "tax_provision": [
        "Tax Provision",
        "TaxProvision",
        "Income Tax Expense",
    ],
    "basic_eps": [
        "Basic EPS",
        "BasicEPS",
        "Basic Eps",
    ],
    "diluted_eps": [
        "Diluted EPS",
        "DilutedEPS",
        "Diluted Eps",
    ],
    "shares_outstanding_reported": [
        "Basic Average Shares",
        "Diluted Average Shares",
        "Average Dilution Earnings",
    ],
}

_BALANCE_ALIASES: dict[str, list[str]] = {
    "total_assets": [
        "Total Assets",
        "TotalAssets",
    ],
    "current_assets": [
        "Current Assets",
        "CurrentAssets",
        "Total Current Assets",
    ],
    "current_liabilities": [
        "Current Liabilities",
        "CurrentLiabilities",
        "Total Current Liabilities",
    ],
    "total_equity": [
        "Stockholders Equity",
        "StockholdersEquity",
        "Total Equity Gross Minority Interest",
        "Common Stock Equity",
    ],
    "total_debt": [
        "Total Debt",
        "TotalDebt",
        "Long Term Debt",
    ],
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "CashAndCashEquivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash",
    ],
    "book_value_per_share": [
        "Book Value",
        # Computed below if not directly available
    ],
}

_CASHFLOW_ALIASES: dict[str, list[str]] = {
    "operating_cash_flow": [
        "Operating Cash Flow",
        "OperatingCashFlow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
    ],
    "capital_expenditure": [
        "Capital Expenditure",
        "CapitalExpenditure",
        "Capital Expenditures",
        "Purchase Of Plant And Equipment",
    ],
    "free_cash_flow": [
        "Free Cash Flow",
        "FreeCashFlow",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_market(ticker: str) -> tuple[str, str]:
    """
    Returns (market, currency) based on ticker suffix.

    Raises ValueError for bare Indian tickers (no .NS / .BO suffix).
    """
    upper = ticker.upper()
    if upper.endswith(".NS"):
        return "IN", "INR"
    if upper.endswith(".BO"):
        return "IN", "INR"
    # Simple heuristic: US tickers contain only letters and at most one dot
    # that is NOT .NS/.BO. We allow things like BRK-A, BRK.B etc.
    # If the ticker looks like a pure alphabetic Indian company name, raise.
    # For now: warn the caller to be explicit.
    if "." not in upper and "-" not in upper and len(upper) > 5:
        raise ValueError(
            f"Ticker '{ticker}' looks like it might be an Indian stock without "
            "an exchange suffix. Use '{ticker}.NS' for NSE or '{ticker}.BO' for BSE."
        )
    return "US", "USD"


def _resolve_field(
    df: pd.DataFrame,
    field_name: str,
    aliases: list[str],
    col: Any,  # column label (date) in the transposed df
    missing_log: list[str],
    context: str,
) -> Optional[float]:
    """
    Try each alias in order; return first match as float.
    Logs to missing_log if nothing found.
    """
    for alias in aliases:
        if alias in df.index:
            raw = df.loc[alias, col]
            if pd.notna(raw):
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
    missing_log.append(f"{context}.{field_name}")
    return None


def _period_label(col: Any, is_quarterly: bool) -> str:
    """Convert a pandas Timestamp column label to a human-readable period string."""
    if isinstance(col, pd.Timestamp):
        if is_quarterly:
            q = (col.month - 1) // 3 + 1
            return f"Q{q} FY{col.year}"
        return f"FY{col.year}"
    return str(col)


def _as_of_date(col: Any) -> str:
    if isinstance(col, pd.Timestamp):
        return col.strftime("%Y-%m-%d")
    return str(col)


def _parse_income(
    df: pd.DataFrame,
    is_quarterly: bool,
    missing_log: list[str],
) -> list[IncomeRow]:
    rows: list[IncomeRow] = []
    if df is None or df.empty:
        return rows

    for col in df.columns:
        period = _period_label(col, is_quarterly)
        ctx = f"income({'Q' if is_quarterly else 'A'}) {period}"

        def get(field: str) -> Optional[float]:
            return _resolve_field(df, field, _INCOME_ALIASES[field], col, missing_log, ctx)

        rows.append(
            IncomeRow(
                period=period,
                as_of_date=_as_of_date(col),
                revenue=get("revenue"),
                gross_profit=get("gross_profit"),
                ebit=get("ebit"),
                ebitda=get("ebitda"),
                net_income=get("net_income"),
                interest_expense=get("interest_expense"),
                tax_provision=get("tax_provision"),
                basic_eps=get("basic_eps"),
                diluted_eps=get("diluted_eps"),
                shares_outstanding_reported=get("shares_outstanding_reported"),
            )
        )
    return rows


def _parse_balance(
    df: pd.DataFrame,
    is_quarterly: bool,
    missing_log: list[str],
    shares_outstanding: Optional[float],
) -> list[BalanceRow]:
    rows: list[BalanceRow] = []
    if df is None or df.empty:
        return rows

    for col in df.columns:
        period = _period_label(col, is_quarterly)
        ctx = f"balance({'Q' if is_quarterly else 'A'}) {period}"

        def get(field: str) -> Optional[float]:
            return _resolve_field(df, field, _BALANCE_ALIASES[field], col, missing_log, ctx)

        total_equity = get("total_equity")
        # Derive book_value_per_share if not directly available
        bvps = get("book_value_per_share")
        if bvps is None and total_equity is not None and shares_outstanding:
            bvps = total_equity / shares_outstanding

        rows.append(
            BalanceRow(
                period=period,
                as_of_date=_as_of_date(col),
                total_assets=get("total_assets"),
                current_assets=get("current_assets"),
                current_liabilities=get("current_liabilities"),
                total_equity=total_equity,
                total_debt=get("total_debt"),
                cash_and_equivalents=get("cash_and_equivalents"),
                book_value_per_share=bvps,
            )
        )
    return rows


def _parse_cashflow(
    df: pd.DataFrame,
    is_quarterly: bool,
    missing_log: list[str],
) -> list[CashFlowRow]:
    rows: list[CashFlowRow] = []
    if df is None or df.empty:
        return rows

    for col in df.columns:
        period = _period_label(col, is_quarterly)
        ctx = f"cashflow({'Q' if is_quarterly else 'A'}) {period}"

        def get(field: str) -> Optional[float]:
            return _resolve_field(df, field, _CASHFLOW_ALIASES[field], col, missing_log, ctx)

        ocf = get("operating_cash_flow")
        capex = get("capital_expenditure")
        fcf = get("free_cash_flow")

        # Derive FCF if not directly available
        # capex in yfinance is already negative; adding gives OCF - |capex|
        if fcf is None and ocf is not None and capex is not None:
            fcf = ocf + capex  # capex is negative, so this is OCF - |capex|

        rows.append(
            CashFlowRow(
                period=period,
                as_of_date=_as_of_date(col),
                operating_cash_flow=ocf,
                capital_expenditure=capex,
                free_cash_flow=fcf,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_company_data(ticker: str) -> CompanyData:
    """
    Fetch all available financial data for ticker via yfinance and return
    a fully typed CompanyData object.

    Parameters
    ----------
    ticker : str
        US ticker (e.g. "AAPL") or Indian ticker with exchange suffix
        (e.g. "RELIANCE.NS", "TCS.BO"). Bare Indian tickers without suffix
        raise ValueError.

    Returns
    -------
    CompanyData
        Typed data object. Fields that yfinance could not supply are None.
        All missing fields are listed in .missing_fields for audit.
    """
    market, currency = _detect_market(ticker)
    missing: list[str] = []

    logger.info(f"Fetching data for {ticker} (market={market}) from yfinance")

    yf_ticker = yf.Ticker(ticker)

    # ------------------------------------------------------------------
    # Info / market data
    # ------------------------------------------------------------------
    try:
        info: dict = yf_ticker.info or {}
    except Exception as exc:
        logger.warning(f"[{ticker}] Failed to fetch info: {exc}")
        info = {}

    def _info(key: str, fallback: Optional[float] = None) -> Any:
        val = info.get(key, fallback)
        if val is None:
            missing.append(f"info.{key}")
        return val

    current_price: Optional[float] = None
    for price_key in ("currentPrice", "regularMarketPrice", "previousClose"):
        v = info.get(price_key)
        if v is not None:
            try:
                current_price = float(v)
                break
            except (TypeError, ValueError):
                pass
    if current_price is None:
        missing.append("info.currentPrice")

    shares_outstanding: Optional[float] = None
    for so_key in ("sharesOutstanding", "impliedSharesOutstanding"):
        v = info.get(so_key)
        if v is not None:
            try:
                shares_outstanding = float(v)
                break
            except (TypeError, ValueError):
                pass
    if shares_outstanding is None:
        missing.append("info.sharesOutstanding")

    market_cap: Optional[float] = None
    v = info.get("marketCap")
    if v is not None:
        try:
            market_cap = float(v)
        except (TypeError, ValueError):
            pass
    if market_cap is None:
        if current_price and shares_outstanding:
            market_cap = current_price * shares_outstanding
        else:
            missing.append("info.marketCap")

    beta: Optional[float] = None
    v = info.get("beta")
    if v is not None:
        try:
            beta = float(v)
        except (TypeError, ValueError):
            pass
    if beta is None:
        missing.append("info.beta")

    forward_eps: Optional[float] = None
    v = info.get("forwardEps")
    if v is not None:
        try:
            forward_eps = float(v)
        except (TypeError, ValueError):
            pass
    if forward_eps is None:
        missing.append("info.forwardEps")

    trailing_eps: Optional[float] = None
    v = info.get("trailingEps")
    if v is not None:
        try:
            trailing_eps = float(v)
        except (TypeError, ValueError):
            pass
    if trailing_eps is None:
        missing.append("info.trailingEps")

    dividend_yield: Optional[float] = None
    for dy_key in ("dividendYield", "trailingAnnualDividendYield"):
        v = info.get(dy_key)
        if v is not None:
            try:
                dividend_yield = float(v)
                break
            except (TypeError, ValueError):
                pass
    if dividend_yield is None:
        missing.append("info.dividendYield")

    fifty_two_week_high: Optional[float] = None
    v = info.get("fiftyTwoWeekHigh")
    if v is not None:
        try:
            fifty_two_week_high = float(v)
        except (TypeError, ValueError):
            pass

    fifty_two_week_low: Optional[float] = None
    v = info.get("fiftyTwoWeekLow")
    if v is not None:
        try:
            fifty_two_week_low = float(v)
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Financial statements — annual
    # ------------------------------------------------------------------
    try:
        income_annual_df: pd.DataFrame = yf_ticker.financials  # columns = fiscal year dates
    except Exception as exc:
        logger.warning(f"[{ticker}] Annual income statement unavailable: {exc}")
        income_annual_df = pd.DataFrame()

    try:
        balance_annual_df: pd.DataFrame = yf_ticker.balance_sheet
    except Exception as exc:
        logger.warning(f"[{ticker}] Annual balance sheet unavailable: {exc}")
        balance_annual_df = pd.DataFrame()

    try:
        cashflow_annual_df: pd.DataFrame = yf_ticker.cashflow
    except Exception as exc:
        logger.warning(f"[{ticker}] Annual cash flow unavailable: {exc}")
        cashflow_annual_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Financial statements — quarterly
    # ------------------------------------------------------------------
    try:
        income_q_df: pd.DataFrame = yf_ticker.quarterly_financials
    except Exception as exc:
        logger.warning(f"[{ticker}] Quarterly income statement unavailable: {exc}")
        income_q_df = pd.DataFrame()

    try:
        balance_q_df: pd.DataFrame = yf_ticker.quarterly_balance_sheet
    except Exception as exc:
        logger.warning(f"[{ticker}] Quarterly balance sheet unavailable: {exc}")
        balance_q_df = pd.DataFrame()

    try:
        cashflow_q_df: pd.DataFrame = yf_ticker.quarterly_cashflow
    except Exception as exc:
        logger.warning(f"[{ticker}] Quarterly cash flow unavailable: {exc}")
        cashflow_q_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Parse into typed rows
    # ------------------------------------------------------------------
    income_annual = _parse_income(income_annual_df, is_quarterly=False, missing_log=missing)
    balance_annual = _parse_balance(balance_annual_df, is_quarterly=False, missing_log=missing, shares_outstanding=shares_outstanding)
    cashflow_annual = _parse_cashflow(cashflow_annual_df, is_quarterly=False, missing_log=missing)

    income_quarterly = _parse_income(income_q_df, is_quarterly=True, missing_log=missing)
    balance_quarterly = _parse_balance(balance_q_df, is_quarterly=True, missing_log=missing, shares_outstanding=shares_outstanding)
    cashflow_quarterly = _parse_cashflow(cashflow_q_df, is_quarterly=True, missing_log=missing)

    # Warn if significantly less data than expected
    if len(income_annual) < 3:
        logger.warning(
            f"[{ticker}] Only {len(income_annual)} years of annual income data returned "
            "(expected 4–5). Results may be limited."
        )

    if missing:
        logger.warning(f"[{ticker}] Missing fields: {missing}")

    return CompanyData(
        ticker=ticker,
        name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market=market,
        currency=info.get("currency", currency),
        current_price=current_price,
        shares_outstanding=shares_outstanding,
        market_cap=market_cap,
        beta=beta,
        forward_eps=forward_eps,
        trailing_eps=trailing_eps,
        dividend_yield=dividend_yield,
        fifty_two_week_high=fifty_two_week_high,
        fifty_two_week_low=fifty_two_week_low,
        income_annual=income_annual,
        balance_annual=balance_annual,
        cashflow_annual=cashflow_annual,
        income_quarterly=income_quarterly,
        balance_quarterly=balance_quarterly,
        cashflow_quarterly=cashflow_quarterly,
        fetched_at=datetime.utcnow(),
        missing_fields=missing,
        source="yfinance",
    )
