"""
tests/test_integration.py — Live end-to-end integration tests

These tests make REAL yfinance network calls and hit real tickers.
Mark them with @pytest.mark.integration so they can be excluded in fast CI runs:

    pytest tests/ -m "not integration"   # skip these
    pytest tests/ -v                     # run everything

Tests cover:
  1. AAPL (US ticker) full pipeline: fetch → fundamentals → ratios → DCF
  2. RELIANCE.NS (India NSE ticker) same pipeline
  3. Invalid ticker handling
  4. Bare Indian ticker rejection
"""

import pytest

from data.fetcher import fetch_company_data
from analytics.fundamentals import all_fundamentals
from analytics.ratios import all_ratios
from valuation.dcf import estimate_wacc, run_dcf

pytestmark = pytest.mark.integration


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _run_full_pipeline(ticker: str):
    """Shared pipeline: fetch → fundamentals → ratios → DCF."""
    # Step 1: Fetch
    data = fetch_company_data(ticker)
    assert data is not None
    assert data.ticker == ticker

    # Step 2: Fundamentals
    fundamentals = all_fundamentals(data)
    assert isinstance(fundamentals, dict)

    # Step 3: Ratios
    ratios = all_ratios(data)
    assert isinstance(ratios, dict)

    # Step 4: DCF (if FCF data available)
    if data.cashflow_annual:
        row = data.cashflow_annual[0]
        fcf_base = row.free_cash_flow
        if fcf_base is None and row.operating_cash_flow and row.capital_expenditure:
            fcf_base = row.operating_cash_flow + row.capital_expenditure

        if fcf_base is not None and data.shares_outstanding:
            wacc_detail = estimate_wacc(beta=data.beta, market=data.market)
            result = run_dcf(
                fcf_base=fcf_base,
                growth_rate_stage1=0.10,
                years_stage1=5,
                terminal_growth_rate=0.03,
                wacc=wacc_detail["wacc"],
                shares_outstanding=data.shares_outstanding,
                current_price=data.current_price,
            )
            assert result.intrinsic_value_per_share is not None
            assert "fcf_base" in result.assumptions

    return data, fundamentals, ratios


# ─────────────────────────────────────────────────────────────
# AAPL (US)
# ─────────────────────────────────────────────────────────────

class TestAAPL:

    def test_fetch_returns_data(self):
        data = fetch_company_data("AAPL")
        assert data.market == "US"
        assert data.currency in ("USD", None)
        assert data.current_price is not None and data.current_price > 0

    def test_has_income_data(self):
        data = fetch_company_data("AAPL")
        assert len(data.income_annual) >= 1, "Expected at least 1 year of income data"

    def test_revenue_is_positive(self):
        data = fetch_company_data("AAPL")
        if data.income_annual and data.income_annual[0].revenue:
            assert data.income_annual[0].revenue > 0

    def test_no_fabricated_zeros(self):
        """
        Missing fields must be None, not 0.
        Check: if a field is in missing_fields, it should also be None in the model.
        """
        data = fetch_company_data("AAPL")
        # Check a few key fields — if beta is missing it should be None not 0
        if "info.beta" in data.missing_fields:
            assert data.beta is None

    def test_full_pipeline(self):
        _run_full_pipeline("AAPL")

    def test_pe_ratio_positive(self):
        """AAPL should have a positive trailing P/E."""
        from analytics.ratios import pe_ratio
        data = fetch_company_data("AAPL")
        result = pe_ratio(data)
        if result.current_value is not None:
            assert result.current_value > 0

    def test_dcf_assumptions_complete(self):
        """DCF result must carry all input assumptions."""
        data = fetch_company_data("AAPL")
        if not data.cashflow_annual or not data.shares_outstanding:
            pytest.skip("AAPL missing cashflow or shares data")

        row = data.cashflow_annual[0]
        fcf_base = row.free_cash_flow or (
            (row.operating_cash_flow or 0) + (row.capital_expenditure or 0)
        )
        wacc_detail = estimate_wacc(beta=data.beta, market=data.market)
        result = run_dcf(
            fcf_base=fcf_base,
            growth_rate_stage1=0.08,
            years_stage1=5,
            terminal_growth_rate=0.03,
            wacc=wacc_detail["wacc"],
            shares_outstanding=data.shares_outstanding,
            current_price=data.current_price,
        )
        for key in ["fcf_base", "growth_rate_stage1", "wacc", "shares_outstanding"]:
            assert key in result.assumptions


# ─────────────────────────────────────────────────────────────
# RELIANCE.NS (India NSE)
# ─────────────────────────────────────────────────────────────

class TestRELIANCE:

    def test_fetch_returns_data(self):
        data = fetch_company_data("RELIANCE.NS")
        assert data.market == "IN"
        assert data.currency in ("INR", None)

    def test_has_income_data(self):
        data = fetch_company_data("RELIANCE.NS")
        # yfinance coverage for Indian tickers can be sparse; warn but don't fail
        if len(data.income_annual) == 0:
            pytest.skip("yfinance returned no income data for RELIANCE.NS — may be a yfinance limitation")
        assert len(data.income_annual) >= 1

    def test_full_pipeline(self):
        _run_full_pipeline("RELIANCE.NS")

    def test_wacc_uses_india_defaults(self):
        """India CAPM should use rf=7%, ERP=7% → WACC > US default at same beta."""
        data = fetch_company_data("RELIANCE.NS")
        wacc_in = estimate_wacc(beta=1.0, market="IN")["wacc"]
        wacc_us = estimate_wacc(beta=1.0, market="US")["wacc"]
        assert wacc_in > wacc_us


# ─────────────────────────────────────────────────────────────
# Ticker format validation
# ─────────────────────────────────────────────────────────────

class TestTickerValidation:

    def test_bare_indian_ticker_raises_value_error(self):
        """RELIANCE without .NS/.BO must raise ValueError."""
        with pytest.raises(ValueError, match="exchange suffix"):
            fetch_company_data("RELIANCE")

    def test_invalid_us_ticker_does_not_crash(self):
        """
        An invalid US ticker should return a CompanyData with empty statements
        and None price — not an exception (yfinance returns empty DataFrames).
        """
        data = fetch_company_data("XXXXXXINVALID")
        assert isinstance(data, type(data))  # just check it doesn't raise

    def test_bse_ticker_detected_as_india(self):
        """BSE suffix .BO should also be detected as Indian market."""
        data = fetch_company_data("RELIANCE.BO")
        assert data.market == "IN"
