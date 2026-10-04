"""
tests/test_fundamentals.py — Unit tests for analytics/fundamentals.py

Uses synthetic CompanyData fixtures — no network calls.
Tests:
  1. Correct computation with full data
  2. Graceful None return (not crash, not fabricated 0) on missing data
  3. Edge cases: single year, zero revenue, zero equity
"""

import pytest
from datetime import datetime

from data.models import BalanceRow, CashFlowRow, CompanyData, IncomeRow
from analytics.fundamentals import (
    all_fundamentals,
    debt_to_equity,
    fcf,
    interest_coverage,
    net_margin,
    operating_margin,
    revenue_growth_yoy,
    roe,
    roce,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _make_income(
    period: str,
    as_of_date: str,
    revenue: float | None = 1000.0,
    ebit: float | None = 200.0,
    ebitda: float | None = 250.0,
    net_income: float | None = 150.0,
    interest_expense: float | None = -20.0,  # yfinance stores as negative
) -> IncomeRow:
    return IncomeRow(
        period=period,
        as_of_date=as_of_date,
        revenue=revenue,
        ebit=ebit,
        ebitda=ebitda,
        net_income=net_income,
        interest_expense=interest_expense,
    )


def _make_balance(
    period: str,
    as_of_date: str,
    total_assets: float | None = 2000.0,
    current_assets: float | None = 500.0,
    current_liabilities: float | None = 300.0,
    total_equity: float | None = 800.0,
    total_debt: float | None = 400.0,
    cash_and_equivalents: float | None = 100.0,
    book_value_per_share: float | None = 8.0,
) -> BalanceRow:
    return BalanceRow(
        period=period,
        as_of_date=as_of_date,
        total_assets=total_assets,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        total_equity=total_equity,
        total_debt=total_debt,
        cash_and_equivalents=cash_and_equivalents,
        book_value_per_share=book_value_per_share,
    )


def _make_cashflow(
    period: str,
    as_of_date: str,
    operating_cash_flow: float | None = 180.0,
    capital_expenditure: float | None = -30.0,  # negative in yfinance convention
    free_cash_flow: float | None = None,
) -> CashFlowRow:
    return CashFlowRow(
        period=period,
        as_of_date=as_of_date,
        operating_cash_flow=operating_cash_flow,
        capital_expenditure=capital_expenditure,
        free_cash_flow=free_cash_flow,
    )


def _full_company(ticker: str = "TEST") -> CompanyData:
    """A synthetic CompanyData with 3 full years of data."""
    return CompanyData(
        ticker=ticker,
        name="Test Corp",
        sector="Technology",
        market="US",
        currency="USD",
        current_price=50.0,
        shares_outstanding=100.0,
        market_cap=5000.0,
        beta=1.2,
        fetched_at=datetime(2024, 1, 1),
        income_annual=[
            _make_income("FY2023", "2023-12-31", revenue=1100.0, ebit=230.0, net_income=170.0),
            _make_income("FY2022", "2022-12-31", revenue=1000.0, ebit=200.0, net_income=150.0),
            _make_income("FY2021", "2021-12-31", revenue=900.0,  ebit=180.0, net_income=130.0),
        ],
        balance_annual=[
            _make_balance("FY2023", "2023-12-31", total_equity=850.0),
            _make_balance("FY2022", "2022-12-31", total_equity=800.0),
            _make_balance("FY2021", "2021-12-31", total_equity=750.0),
        ],
        cashflow_annual=[
            _make_cashflow("FY2023", "2023-12-31", operating_cash_flow=200.0, capital_expenditure=-35.0),
            _make_cashflow("FY2022", "2022-12-31", operating_cash_flow=180.0, capital_expenditure=-30.0),
        ],
    )


def _empty_company(ticker: str = "EMPTY") -> CompanyData:
    """A CompanyData with no financial statement data."""
    return CompanyData(
        ticker=ticker,
        market="US",
        fetched_at=datetime(2024, 1, 1),
    )


# ─────────────────────────────────────────────────────────────
# Revenue Growth YoY
# ─────────────────────────────────────────────────────────────

class TestRevenueGrowthYoY:

    def test_correct_growth_rate(self):
        data = _full_company()
        result = revenue_growth_yoy(data)
        # FY2023 vs FY2022: (1100 - 1000) / 1000 = 10%
        assert len(result) >= 1
        assert abs(result[0].value - 0.10) < 1e-6

    def test_growth_for_all_available_periods(self):
        data = _full_company()
        result = revenue_growth_yoy(data)
        assert len(result) == 2   # FY23 vs FY22, FY22 vs FY21

    def test_period_provenance_on_each_entry(self):
        data = _full_company()
        result = revenue_growth_yoy(data)
        for entry in result:
            assert entry.period is not None
            assert entry.as_of_date is not None

    def test_returns_empty_with_single_year(self):
        """Can't compute growth without a prior year."""
        data = _full_company()
        data.income_annual = [data.income_annual[0]]
        result = revenue_growth_yoy(data)
        assert result == []

    def test_returns_empty_on_no_income_data(self):
        result = revenue_growth_yoy(_empty_company())
        assert result == []

    def test_none_value_when_revenue_missing(self):
        """Missing revenue in one year → None (not crash or 0)."""
        data = _full_company()
        data.income_annual[0].revenue = None
        result = revenue_growth_yoy(data)
        assert result[0].value is None
        assert result[0].note is not None

    def test_none_value_when_prior_year_revenue_zero(self):
        data = _full_company()
        data.income_annual[1].revenue = 0.0
        result = revenue_growth_yoy(data)
        assert result[0].value is None


# ─────────────────────────────────────────────────────────────
# ROE
# ─────────────────────────────────────────────────────────────

class TestROE:

    def test_correct_value(self):
        """ROE = net_income / avg_equity = 170 / avg(850, 800) = 170 / 825 ≈ 0.2061"""
        data = _full_company()
        result = roe(data)
        expected = 170.0 / ((850.0 + 800.0) / 2)
        assert result.value is not None
        assert abs(result.value - expected) < 1e-5

    def test_none_on_no_income_data(self):
        result = roe(_empty_company())
        assert result.value is None

    def test_none_on_no_balance_data(self):
        data = _full_company()
        data.balance_annual = []
        result = roe(data)
        assert result.value is None

    def test_none_on_none_equity(self):
        data = _full_company()
        data.balance_annual[0].total_equity = None
        data.balance_annual[1].total_equity = None
        result = roe(data)
        assert result.value is None

    def test_period_in_result(self):
        result = roe(_full_company())
        assert result.period == "FY2023"


# ─────────────────────────────────────────────────────────────
# ROCE
# ─────────────────────────────────────────────────────────────

class TestROCE:

    def test_correct_value(self):
        """ROCE = EBIT / (total_assets - current_liabilities) = 230 / (2000 - 300) = 230/1700 ≈ 0.1353"""
        data = _full_company()
        result = roce(data)
        expected = 230.0 / (2000.0 - 300.0)
        assert result.value is not None
        assert abs(result.value - expected) < 1e-5

    def test_none_on_missing_assets(self):
        data = _full_company()
        data.balance_annual[0].total_assets = None
        result = roce(data)
        assert result.value is None

    def test_none_on_missing_current_liabilities(self):
        data = _full_company()
        data.balance_annual[0].current_liabilities = None
        result = roce(data)
        assert result.value is None


# ─────────────────────────────────────────────────────────────
# Operating Margin
# ─────────────────────────────────────────────────────────────

class TestOperatingMargin:

    def test_correct_value(self):
        """230 / 1100 ≈ 0.2091"""
        data = _full_company()
        result = operating_margin(data)
        assert result.value is not None
        assert abs(result.value - 230.0 / 1100.0) < 1e-6

    def test_none_on_missing_ebit(self):
        data = _full_company()
        data.income_annual[0].ebit = None
        result = operating_margin(data)
        assert result.value is None

    def test_none_on_zero_revenue(self):
        data = _full_company()
        data.income_annual[0].revenue = 0.0
        result = operating_margin(data)
        assert result.value is None

    def test_none_on_no_data(self):
        assert operating_margin(_empty_company()).value is None


# ─────────────────────────────────────────────────────────────
# Net Margin
# ─────────────────────────────────────────────────────────────

class TestNetMargin:

    def test_correct_value(self):
        """170 / 1100 ≈ 0.1545"""
        data = _full_company()
        result = net_margin(data)
        assert result.value is not None
        assert abs(result.value - 170.0 / 1100.0) < 1e-6

    def test_none_on_no_data(self):
        assert net_margin(_empty_company()).value is None


# ─────────────────────────────────────────────────────────────
# Debt to Equity
# ─────────────────────────────────────────────────────────────

class TestDebtToEquity:

    def test_correct_value(self):
        """400 / 850 ≈ 0.4706"""
        data = _full_company()
        result = debt_to_equity(data)
        assert result.value is not None
        assert abs(result.value - 400.0 / 850.0) < 1e-5

    def test_none_on_zero_equity(self):
        data = _full_company()
        data.balance_annual[0].total_equity = 0.0
        result = debt_to_equity(data)
        assert result.value is None   # division by zero → None

    def test_none_on_no_data(self):
        assert debt_to_equity(_empty_company()).value is None


# ─────────────────────────────────────────────────────────────
# Interest Coverage
# ─────────────────────────────────────────────────────────────

class TestInterestCoverage:

    def test_correct_value(self):
        """EBIT / |interest| = 230 / 20 = 11.5"""
        data = _full_company()
        result = interest_coverage(data)
        assert result.value is not None
        assert abs(result.value - 11.5) < 1e-5

    def test_handles_positive_interest_sign(self):
        """Some data sources report interest as positive — abs() normalises."""
        data = _full_company()
        data.income_annual[0].interest_expense = 20.0  # positive
        result = interest_coverage(data)
        assert result.value is not None
        assert abs(result.value - 11.5) < 1e-5

    def test_none_on_zero_interest(self):
        data = _full_company()
        data.income_annual[0].interest_expense = 0.0
        result = interest_coverage(data)
        assert result.value is None
        assert result.note is not None

    def test_none_on_no_data(self):
        assert interest_coverage(_empty_company()).value is None


# ─────────────────────────────────────────────────────────────
# FCF
# ─────────────────────────────────────────────────────────────

class TestFCF:

    def test_correct_computed_fcf(self):
        """200 + (-35) = 165"""
        data = _full_company()
        result = fcf(data)
        assert result.value is not None
        assert abs(result.value - 165.0) < 1e-6

    def test_uses_precomputed_fcf_when_available(self):
        data = _full_company()
        data.cashflow_annual[0].free_cash_flow = 999.0  # explicit override
        result = fcf(data)
        assert result.value == 999.0

    def test_none_on_missing_ocf(self):
        data = _full_company()
        data.cashflow_annual[0].operating_cash_flow = None
        data.cashflow_annual[0].free_cash_flow = None
        result = fcf(data)
        assert result.value is None

    def test_none_on_missing_capex(self):
        data = _full_company()
        data.cashflow_annual[0].capital_expenditure = None
        data.cashflow_annual[0].free_cash_flow = None
        result = fcf(data)
        assert result.value is None

    def test_none_on_no_data(self):
        assert fcf(_empty_company()).value is None


# ─────────────────────────────────────────────────────────────
# all_fundamentals convenience function
# ─────────────────────────────────────────────────────────────

class TestAllFundamentals:

    def test_returns_all_keys(self):
        result = all_fundamentals(_full_company())
        expected_keys = {
            "revenue_growth_yoy", "roe", "roce", "operating_margin",
            "net_margin", "debt_to_equity", "interest_coverage", "fcf",
        }
        assert set(result.keys()) == expected_keys

    def test_no_crash_on_empty_company(self):
        """All functions must handle empty data without raising."""
        result = all_fundamentals(_empty_company())
        assert isinstance(result, dict)
        # Every scalar metric should have value=None, not raise
        for key, val in result.items():
            if key == "revenue_growth_yoy":
                assert isinstance(val, list)
            else:
                assert val.value is None
