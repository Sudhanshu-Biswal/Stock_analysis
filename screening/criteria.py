"""
screening/criteria.py — individual pass/fail + ranking criteria functions

Every criterion is a pure function over already-fetched Phase 1 API output
(a /fundamentals dict, a /ratios dict, or a /reverse-dcf result dict). No
LLM calls, no hidden weighting. Never crash on missing data -- return
passed=False with an explanatory note instead. Every result carries the
actual value and threshold used, never a bare boolean.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CriterionResult(BaseModel):
    criterion: str
    passed: bool
    value: Optional[float]
    threshold: Optional[float]
    note: Optional[str] = None


def _metric_value(fundamentals: dict, key: str) -> Optional[float]:
    """Pull a single metric's .value out of a /fundamentals response dict."""
    metric = fundamentals.get("metrics", {}).get(key)
    if not metric:
        return None
    return metric.get("value")


def positive_fcf(fundamentals: dict) -> CriterionResult:
    """FCF > 0."""
    value = _metric_value(fundamentals, "fcf")
    if value is None:
        return CriterionResult(criterion="positive_fcf", passed=False, value=None, threshold=0.0, note="insufficient data")
    return CriterionResult(criterion="positive_fcf", passed=value > 0, value=value, threshold=0.0)


def reasonable_debt(fundamentals: dict, max_de: float = 1.5) -> CriterionResult:
    """Debt/Equity below threshold."""
    value = _metric_value(fundamentals, "debt_to_equity")
    if value is None:
        return CriterionResult(criterion="reasonable_debt", passed=False, value=None, threshold=max_de, note="insufficient data")
    return CriterionResult(criterion="reasonable_debt", passed=value < max_de, value=value, threshold=max_de)


def roe_above(fundamentals: dict, min_roe: float = 0.15) -> CriterionResult:
    """ROE above threshold."""
    value = _metric_value(fundamentals, "roe")
    if value is None:
        return CriterionResult(criterion="roe_above", passed=False, value=None, threshold=min_roe, note="insufficient data")
    return CriterionResult(criterion="roe_above", passed=value > min_roe, value=value, threshold=min_roe)


def revenue_growth_positive_recent(fundamentals: dict) -> CriterionResult:
    """Most recent YoY revenue growth > 0."""
    growth_list = fundamentals.get("metrics", {}).get("revenue_growth_yoy") or []
    value = growth_list[0].get("value") if growth_list else None
    if value is None:
        return CriterionResult(criterion="revenue_growth_positive_recent", passed=False, value=None, threshold=0.0, note="insufficient data")
    return CriterionResult(criterion="revenue_growth_positive_recent", passed=value > 0, value=value, threshold=0.0)


def pe_below_historical_median(ratios: dict) -> CriterionResult:
    """Current P/E below its own historical median."""
    band = ratios.get("ratios", {}).get("pe_ratio") or {}
    value = band.get("current_value")
    threshold = band.get("historical_median")
    if value is None or threshold is None:
        return CriterionResult(criterion="pe_below_historical_median", passed=False, value=value, threshold=threshold, note="insufficient data")
    return CriterionResult(criterion="pe_below_historical_median", passed=value < threshold, value=value, threshold=threshold)


def reverse_dcf_implied_growth_below(reverse_dcf_result: Optional[dict], max_growth: float = 0.20) -> CriterionResult:
    """
    Flags stocks NOT priced for extreme/unrealistic growth.

    If reverse-DCF couldn't even solve (e.g. no growth rate under the
    solver's cap justifies the current price -- the AAPL case), the market
    is implicitly pricing in more growth than max_growth allows, so this
    fails rather than being silently skipped.
    """
    if not reverse_dcf_result or reverse_dcf_result.get("implied_growth_rate") is None:
        return CriterionResult(
            criterion="reverse_dcf_implied_growth_below", passed=False, value=None,
            threshold=max_growth,
            note="reverse-DCF could not solve -- implied growth exceeds solver's max (priced for perfection)",
        )
    value = reverse_dcf_result["implied_growth_rate"]
    return CriterionResult(criterion="reverse_dcf_implied_growth_below", passed=value < max_growth, value=value, threshold=max_growth)
