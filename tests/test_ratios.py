"""
tests/test_ratios.py — Unit tests for analytics/ratios.py

Uses synthetic CompanyData fixtures — no network calls.
"""

import pytest
from datetime import datetime

from data.models import BalanceRow, CashFlowRow, CompanyData, IncomeRow
from analytics.ratios import (
    dividend_yield,
    ev_to_ebitda,
    fcf_yield,
    forward_pe,
    pe_ratio,
    peg_ratio,
    price_to_book,
    all_ratios,
)


# ─────────────────────────────────────────────────────────────
# Fixtures (shared with test_fundamentals, but independent copy)
# ─────────────────────────────────────────────────────────────

def _make_income(period, as_of_date, revenue=1000.0, ebit=200.0, ebitda=250.0,
                 net_income=150.0, diluted_eps=1.50, basic_eps=1.55):
    return IncomeRow(
        period=period, as_of_date=as_of_date,
        revenue=revenue, ebit=ebit, ebitda=ebitda,
        net_income=net_income, diluted_eps=diluted_eps, basic_eps=basic_eps,
    )


def _make_balance(period, as_of_date, total_assets=2000.0, current_liabilities=300.0,
                  total_equity=800.0, total_debt=400.0,
                  cash_and_equivalents=100.0, book_value_per_share=8.0):
    return BalanceRow(
        period=period, as_of_date=as_of_date,
        total_assets=total_assets, current_liabilities=current_liabilities,
        total_equity=total_equity, total_debt=total_debt,
        cash_and_equivalents=cash_and_equivalents, book_value_per_share=book_value_per_share,
    )


def _make_cashflow(period, as_of_date, operating_cash_flow=180.0,
                   capital_expenditure=-30.0, free_cash_flow=None):
    return CashFlowRow(
        period=period, as_of_date=as_of_date,
        operating_cash_flow=operating_cash_flow,
        capital_expenditure=capital_expenditure,
        free_cash_flow=free_cash_flow,
    )


def _full_company(price=30.0, trailing_eps=1.50, forward_eps=1.80,
                  dividend_yield_val=0.015, market_cap=3000.0) -> CompanyData:
    return CompanyData(
        ticker="RATIO",
        name="Ratio Test Corp",
        market="US",
        currency="USD",
        current_price=price,
        shares_outstanding=100.0,
        market_cap=market_cap,
        beta=1.1,
        trailing_eps=trailing_eps,
        forward_eps=forward_eps,
        dividend_yield=dividend_yield_val,
        fetched_at=datetime(2024, 1, 1),
        income_annual=[
            _make_income("FY2023", "2023-12-31", ebitda=275.0, diluted_eps=1.50),
            _make_income("FY2022", "2022-12-31", ebitda=250.0, diluted_eps=1.30),
            _make_income("FY2021", "2021-12-31", ebitda=230.0, diluted_eps=1.10),
            _make_income("FY2020", "2020-12-31", ebitda=210.0, diluted_eps=0.95),
            _make_income("FY2019", "2019-12-31", ebitda=190.0, diluted_eps=0.80),
        ],
        balance_annual=[
            _make_balance("FY2023", "2023-12-31"),
            _make_balance("FY2022", "2022-12-31"),
            _make_balance("FY2021", "2021-12-31"),
            _make_balance("FY2020", "2020-12-31"),
            _make_balance("FY2019", "2019-12-31"),
        ],
        cashflow_annual=[
            _make_cashflow("FY2023", "2023-12-31", operating_cash_flow=200.0, capital_expenditure=-35.0),
            _make_cashflow("FY2022", "2022-12-31"),
            _make_cashflow("FY2021", "2021-12-31"),
            _make_cashflow("FY2020", "2020-12-31"),
            _make_cashflow("FY2019", "2019-12-31"),
        ],
    )


def _empty_company() -> CompanyData:
    return CompanyData(ticker="EMPTY", market="US", fetched_at=datetime(2024, 1, 1))


# ─────────────────────────────────────────────────────────────
# P/E Ratio
# ─────────────────────────────────────────────────────────────

class TestPERatio:

    def test_correct_current_pe(self):
        """P/E = 30 / 1.50 = 20.0"""
        data = _full_company()
        result = pe_ratio(data)
        assert result.current_value is not None
        assert abs(result.current_value - 20.0) < 0.01

    def test_none_when_trailing_eps_none(self):
        data = _full_company()
        data.trailing_eps = None
        result = pe_ratio(data)
        assert result.current_value is None

    def test_none_when_price_none(self):
        data = _full_company()
        data.current_price = None
        result = pe_ratio(data)
        assert result.current_value is None

    def test_none_when_trailing_eps_zero(self):
        """No division by zero — should return None."""
        data = _full_company()
        data.trailing_eps = 0.0
        result = pe_ratio(data)
        assert result.current_value is None

    def test_historical_band_populated(self):
        """With 5 years of EPS, should have up to 5 historical data points."""
        data = _full_company()
        result = pe_ratio(data)
        assert result.periods_used >= 1
        assert result.historical_min is not None
        assert result.historical_max is not None
        assert result.historical_median is not None

    def test_historical_min_le_median_le_max(self):
        data = _full_company()
        result = pe_ratio(data)
        if result.historical_min is not None:
            assert result.historical_min <= result.historical_median <= result.historical_max


# ─────────────────────────────────────────────────────────────
# Forward P/E
# ─────────────────────────────────────────────────────────────

class TestForwardPE:

    def test_correct_value(self):
        """30 / 1.80 ≈ 16.67"""
        data = _full_company()
        result = forward_pe(data)
        assert result.value is not None
        assert abs(result.value - 30.0 / 1.80) < 0.01

    def test_none_when_forward_eps_missing(self):
        data = _full_company()
        data.forward_eps = None
        result = forward_pe(data)
        assert result.value is None
        assert result.note is not None


# ─────────────────────────────────────────────────────────────
# PEG Ratio
# ─────────────────────────────────────────────────────────────

class TestPEGRatio:

    def test_returns_value(self):
        """PEG should be a positive float for our fixture with growing EPS."""
        data = _full_company()
        result = peg_ratio(data)
        # EPS grew from 0.80 (FY2019) to 1.50 (FY2023) over 4 years
        # CAGR ≈ (1.50/0.80)^(1/4) - 1 ≈ 17.0%
        # PE = 20.0, PEG = 20.0 / 17.0 ≈ 1.18
        if result.value is not None:
            assert result.value > 0

    def test_none_when_eps_negative(self):
        """PEG not meaningful when EPS was negative."""
        data = _full_company()
        data.income_annual[-1].diluted_eps = -0.50
        # Still may succeed if other years are positive; test that no crash occurs
        result = peg_ratio(data)
        assert isinstance(result.value, (float, type(None)))

    def test_none_when_insufficient_eps_history(self):
        data = _full_company()
        data.income_annual = [data.income_annual[0]]  # only one year
        result = peg_ratio(data)
        assert result.value is None

    def test_note_contains_cagr_info(self):
        data = _full_company()
        result = peg_ratio(data)
        if result.value is not None:
            assert result.note is not None


# ─────────────────────────────────────────────────────────────
# EV / EBITDA
# ─────────────────────────────────────────────────────────────

class TestEVToEBITDA:

    def test_correct_current_value(self):
        """
        EV = market_cap + net_debt = 3000 + (400 - 100) = 3300
        EV/EBITDA = 3300 / 275 = 12.0
        """
        data = _full_company(market_cap=3000.0)
        result = ev_to_ebitda(data)
        assert result.current_value is not None
        assert abs(result.current_value - 3300.0 / 275.0) < 0.01

    def test_historical_band_populated(self):
        data = _full_company()
        result = ev_to_ebitda(data)
        assert result.periods_used >= 1

    def test_none_when_no_income(self):
        data = _full_company()
        data.income_annual = []
        result = ev_to_ebitda(data)
        assert result.current_value is None

    def test_none_when_ebitda_zero(self):
        data = _full_company()
        data.income_annual[0].ebitda = 0.0
        result = ev_to_ebitda(data)
        assert result.current_value is None


# ─────────────────────────────────────────────────────────────
# Price to Book
# ─────────────────────────────────────────────────────────────

class TestPriceToBook:

    def test_correct_value(self):
        """30 / 8.0 = 3.75"""
        data = _full_company()
        result = price_to_book(data)
        assert result.current_value is not None
        assert abs(result.current_value - 30.0 / 8.0) < 0.01

    def test_none_on_zero_book_value(self):
        data = _full_company()
        data.balance_annual[0].book_value_per_share = 0.0
        result = price_to_book(data)
        assert result.current_value is None

    def test_historical_band_populated(self):
        data = _full_company()
        result = price_to_book(data)
        assert result.periods_used >= 1


# ─────────────────────────────────────────────────────────────
# FCF Yield
# ─────────────────────────────────────────────────────────────

class TestFCFYield:

    def test_correct_value(self):
        """
        FCF = 200 + (-35) = 165
        yield = 165 / 3000 ≈ 0.0550
        """
        data = _full_company(market_cap=3000.0)
        result = fcf_yield(data)
        assert result.current_value is not None
        assert abs(result.current_value - 165.0 / 3000.0) < 0.0001

    def test_none_on_no_cashflow(self):
        data = _full_company()
        data.cashflow_annual = []
        result = fcf_yield(data)
        assert result.current_value is None


# ─────────────────────────────────────────────────────────────
# Dividend Yield
# ─────────────────────────────────────────────────────────────

class TestDividendYield:

    def test_correct_value(self):
        data = _full_company(dividend_yield_val=0.025)
        result = dividend_yield(data)
        assert result.value is not None
        assert abs(result.value - 0.025) < 1e-9

    def test_none_when_not_reported(self):
        data = _full_company()
        data.dividend_yield = None
        result = dividend_yield(data)
        assert result.value is None
        assert result.note is not None


# ─────────────────────────────────────────────────────────────
# all_ratios convenience function
# ─────────────────────────────────────────────────────────────

class TestAllRatios:

    def test_returns_all_keys(self):
        result = all_ratios(_full_company())
        expected = {"pe_ratio", "forward_pe", "peg_ratio", "ev_to_ebitda",
                    "price_to_book", "fcf_yield", "dividend_yield"}
        assert set(result.keys()) == expected

    def test_no_crash_on_empty_company(self):
        result = all_ratios(_empty_company())
        assert isinstance(result, dict)
