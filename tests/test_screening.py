"""
tests/test_screening.py — Phase 3 screening tests

Covers each criterion function individually (including missing-data cases),
run_screen's resilience to a bad/fake ticker, and the "no black box"
requirement that criteria results always carry real value/threshold.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx

from screening.criteria import (
    pe_below_historical_median,
    positive_fcf,
    reasonable_debt,
    reverse_dcf_implied_growth_below,
    revenue_growth_positive_recent,
    roe_above,
)
from screening.screen import run_screen


def _fundamentals(fcf=None, debt_to_equity=None, roe=None, revenue_growth=None) -> dict:
    metrics: dict = {}
    if fcf is not None:
        metrics["fcf"] = {"value": fcf}
    if debt_to_equity is not None:
        metrics["debt_to_equity"] = {"value": debt_to_equity}
    if roe is not None:
        metrics["roe"] = {"value": roe}
    if revenue_growth is not None:
        metrics["revenue_growth_yoy"] = [{"value": revenue_growth}]
    return {"metrics": metrics}


def _ratios(current_pe=None, historical_median=None) -> dict:
    band: dict = {}
    if current_pe is not None:
        band["current_value"] = current_pe
    if historical_median is not None:
        band["historical_median"] = historical_median
    return {"ratios": {"pe_ratio": band}}


class TestPositiveFCF:
    def test_passes_when_positive(self):
        result = positive_fcf(_fundamentals(fcf=1000.0))
        assert result.passed is True
        assert result.value == 1000.0
        assert result.threshold == 0.0

    def test_fails_when_negative(self):
        result = positive_fcf(_fundamentals(fcf=-500.0))
        assert result.passed is False
        assert result.value == -500.0

    def test_insufficient_data(self):
        result = positive_fcf(_fundamentals())
        assert result.passed is False
        assert result.value is None
        assert result.note == "insufficient data"


class TestReasonableDebt:
    def test_passes_below_threshold(self):
        result = reasonable_debt(_fundamentals(debt_to_equity=1.0), max_de=1.5)
        assert result.passed is True
        assert result.threshold == 1.5

    def test_fails_above_threshold(self):
        result = reasonable_debt(_fundamentals(debt_to_equity=2.0), max_de=1.5)
        assert result.passed is False

    def test_insufficient_data(self):
        result = reasonable_debt(_fundamentals())
        assert result.passed is False
        assert result.note == "insufficient data"


class TestROEAbove:
    def test_passes_above_threshold(self):
        result = roe_above(_fundamentals(roe=0.34), min_roe=0.15)
        assert result.passed is True
        assert result.value == 0.34

    def test_fails_below_threshold(self):
        result = roe_above(_fundamentals(roe=0.05), min_roe=0.15)
        assert result.passed is False

    def test_insufficient_data(self):
        result = roe_above(_fundamentals())
        assert result.passed is False
        assert result.note == "insufficient data"


class TestRevenueGrowthPositiveRecent:
    def test_passes_when_positive(self):
        result = revenue_growth_positive_recent(_fundamentals(revenue_growth=0.06))
        assert result.passed is True

    def test_fails_when_negative(self):
        result = revenue_growth_positive_recent(_fundamentals(revenue_growth=-0.03))
        assert result.passed is False

    def test_insufficient_data(self):
        result = revenue_growth_positive_recent(_fundamentals())
        assert result.passed is False
        assert result.note == "insufficient data"


class TestPEBelowHistoricalMedian:
    def test_passes_below_median(self):
        result = pe_below_historical_median(_ratios(current_pe=36.0, historical_median=51.0))
        assert result.passed is True

    def test_fails_above_median(self):
        result = pe_below_historical_median(_ratios(current_pe=60.0, historical_median=51.0))
        assert result.passed is False

    def test_insufficient_data(self):
        result = pe_below_historical_median(_ratios())
        assert result.passed is False
        assert result.note == "insufficient data"


class TestReverseDCFImpliedGrowthBelow:
    def test_passes_below_max(self):
        result = reverse_dcf_implied_growth_below({"implied_growth_rate": 0.10}, max_growth=0.20)
        assert result.passed is True

    def test_fails_above_max(self):
        result = reverse_dcf_implied_growth_below({"implied_growth_rate": 0.30}, max_growth=0.20)
        assert result.passed is False

    def test_fails_when_unavailable(self):
        """AAPL case: reverse-DCF couldn't solve at all -- must fail, not be skipped."""
        result = reverse_dcf_implied_growth_below(None, max_growth=0.20)
        assert result.passed is False
        assert result.value is None


def test_criterion_results_always_include_value_and_threshold():
    """No black box: every criterion result must carry its real value/threshold."""
    result = roe_above(_fundamentals(roe=0.25), min_roe=0.15)
    dumped = result.model_dump()
    assert dumped["value"] == 0.25
    assert dumped["threshold"] == 0.15


def _http_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", "http://testserver")
    return httpx.Response(status_code=status_code, json=json_data, request=request)


class _FakeClient:
    """Stands in for httpx.Client -- branches on the request path's ticker."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, path: str) -> httpx.Response:
        if "NOTAREALTICKER123" in path:
            return _http_response({"detail": "not found"}, status_code=404)
        if path.endswith("/fundamentals"):
            return _http_response(_fundamentals(fcf=1000.0, debt_to_equity=1.0, roe=0.30, revenue_growth=0.05))
        if path.endswith("/ratios"):
            return _http_response(_ratios(current_pe=20.0, historical_median=25.0))
        if path.endswith("/reverse-dcf"):
            return _http_response({"implied_growth_rate": 0.08})
        return _http_response({}, status_code=404)


def test_run_screen_bad_ticker_does_not_crash_batch():
    criteria_fns = [
        lambda f, r, rd: positive_fcf(f),
        lambda f, r, rd: reasonable_debt(f),
        lambda f, r, rd: roe_above(f),
        lambda f, r, rd: revenue_growth_positive_recent(f),
        lambda f, r, rd: reverse_dcf_implied_growth_below(rd),
        lambda f, r, rd: pe_below_historical_median(r),
    ]

    with patch("screening.screen.httpx.Client", return_value=_FakeClient()):
        results = run_screen(
            universe=["AAPL", "NOTAREALTICKER123"],
            criteria_fns=criteria_fns,
            phase1_api_base="http://testserver",
            max_workers=2,
        )

    by_ticker = {r.ticker: r for r in results}

    assert by_ticker["AAPL"].fetch_error is None
    assert by_ticker["AAPL"].pass_count == len(criteria_fns)
    assert by_ticker["AAPL"].total_criteria == len(criteria_fns)
    assert len(by_ticker["AAPL"].criteria_results) == len(criteria_fns)

    assert by_ticker["NOTAREALTICKER123"].fetch_error is not None
    assert by_ticker["NOTAREALTICKER123"].passed_all is False
