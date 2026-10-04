"""
valuation/dcf.py — 2-Stage DCF engine + Reverse DCF + WACC via CAPM

This is the most important file in Phase 1.
Every assumption is surfaced in the output — nothing is a black box.

─────────────────────────────────────────────────────────────────────
  DCF MATH — hand-worked reference example (for test verification)
─────────────────────────────────────────────────────────────────────

  Inputs:
    fcf_base               = 100.0   (base year FCF, e.g. $100 M)
    growth_rate_stage1     = 0.15    (15% in year 1, declining linearly)
    years_stage1           = 5
    terminal_growth_rate   = 0.03    (3%)
    wacc                   = 0.10    (10%)
    shares_outstanding     = 10.0    (10 M shares)
    net_debt               = 0.0     (no debt)

  Stage 1 FCFs (growth linearly declines from 15% → 3% over 5 years):
    Year gaps between growth rates: (0.15 - 0.03) / (5 - 1) = 0.03 per year step
    Year 1 growth = 0.15, Year 2 = 0.12, Year 3 = 0.09, Year 4 = 0.06, Year 5 = 0.03

    FCF_1 = 100 × (1 + 0.15) = 115.000
    FCF_2 = 115 × (1 + 0.12) = 128.800
    FCF_3 = 128.8 × (1 + 0.09) = 140.392
    FCF_4 = 140.392 × (1 + 0.06) = 148.816
    FCF_5 = 148.816 × (1 + 0.03) = 153.280

  PV of Stage 1 FCFs (discounted at WACC = 10%):
    PV_1 = 115.000 / 1.10^1 = 104.545
    PV_2 = 128.800 / 1.10^2 = 106.446
    PV_3 = 140.392 / 1.10^3 = 105.469
    PV_4 = 148.816 / 1.10^4 = 101.589
    PV_5 = 153.280 / 1.10^5 =  95.175
    Sum of Stage 1 PVs = 513.225 (rounded)

  Terminal Value (Gordon Growth, applied at end of year 5):
    TV = FCF_5 × (1 + g) / (WACC - g)
       = 153.280 × 1.03 / (0.10 - 0.03)
       = 157.878 / 0.07
       = 2255.40

    PV of TV = 2255.40 / 1.10^5 = 1400.34

  Enterprise Value = PV_stage1 + PV_TV = 513.225 + 1400.34 = 1913.57
  Equity Value     = EV - net_debt = 1913.57 - 0 = 1913.57
  Intrinsic Value per Share = 1913.57 / 10 = 191.357

─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class DCFResult(BaseModel):
    """
    Full output of a 2-stage DCF valuation.

    Every input assumption is included so the result is self-documenting.
    """

    # Core outputs
    intrinsic_value_per_share: float
    current_price: Optional[float]
    margin_of_safety_pct: Optional[float]  # positive = undervalued, negative = overvalued

    # Build-up details
    stage1_cashflows: list[float]           # FCF projected for each explicit year
    stage1_pv: float                        # sum of PV of stage 1 cashflows
    terminal_value: float                   # raw TV (before discounting)
    terminal_value_pv: float                # PV of TV
    enterprise_value: float                 # stage1_pv + terminal_value_pv
    equity_value: float                     # enterprise_value - net_debt

    # All input assumptions (full transparency)
    assumptions: dict = Field(default_factory=dict)


class ReverseResult(BaseModel):
    """Output of a reverse DCF — implied growth rate at current price."""

    implied_growth_rate: float
    current_price: float

    # Sensitivity: intrinsic value at implied_growth ± 2pp
    sensitivity: dict[str, float] = Field(default_factory=dict)

    assumptions: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# WACC / CAPM
# ---------------------------------------------------------------------------

_WACC_DEFAULTS = {
    "US": {
        "risk_free_rate": 0.045,
        "equity_risk_premium": 0.05,
        "pretax_cost_of_debt_default": 0.05,
        "statutory_tax_rate": 0.21,
    },
    "IN": {
        "risk_free_rate": 0.069,
        "equity_risk_premium": 0.065,
        "pretax_cost_of_debt_default": 0.08,
        "statutory_tax_rate": 0.25,
    },
}


def estimate_cost_of_equity(
    beta: Optional[float],
    market: str = "US",
    risk_free_override: Optional[float] = None,
    erp_override: Optional[float] = None,
    beta_floor: float = 0.5,
) -> float:
    """CAPM cost of equity — unchanged logic, kept as a standalone building block."""
    defaults = _WACC_DEFAULTS.get(market, _WACC_DEFAULTS["US"])
    rf = risk_free_override if risk_free_override is not None else defaults["risk_free_rate"]
    erp = erp_override if erp_override is not None else defaults["equity_risk_premium"]
    b = beta if beta is not None else 1.0
    b = max(b, beta_floor)
    return rf + b * erp


def estimate_cost_of_debt(
    interest_expense: Optional[float],
    total_debt: Optional[float],
    market: str = "US",
    cap_rate: float = 0.15,
) -> float:
    """Pre-tax cost of debt estimated from trailing interest expense / debt."""
    defaults = _WACC_DEFAULTS.get(market, _WACC_DEFAULTS["US"])
    fallback = defaults["pretax_cost_of_debt_default"]

    if not interest_expense or not total_debt or total_debt <= 0:
        return fallback

    rate = interest_expense / total_debt
    if rate <= 0 or rate > cap_rate:
        logger.debug(
            f"Implied cost of debt {rate:.3f} looks unreasonable "
            f"(interest_expense={interest_expense}, total_debt={total_debt}); "
            f"falling back to default {fallback:.3f}"
        )
        return fallback
    return rate


def estimate_tax_rate(
    tax_provision: Optional[float],
    pretax_income: Optional[float],
    market: str = "US",
    min_rate: float = 0.05,
    max_rate: float = 0.40,
) -> float:
    """Effective tax rate from financials, with sane bounds and statutory fallback."""
    defaults = _WACC_DEFAULTS.get(market, _WACC_DEFAULTS["US"])
    fallback = defaults["statutory_tax_rate"]

    if not tax_provision or not pretax_income or pretax_income <= 0:
        return fallback

    rate = tax_provision / pretax_income
    if rate < min_rate or rate > max_rate:
        return fallback
    return rate


def estimate_wacc(
    beta: Optional[float],
    market: str = "US",
    market_cap: Optional[float] = None,
    total_debt: Optional[float] = None,
    cash_and_equivalents: Optional[float] = None,
    interest_expense: Optional[float] = None,
    tax_provision: Optional[float] = None,
    pretax_income: Optional[float] = None,
    risk_free_override: Optional[float] = None,
    erp_override: Optional[float] = None,
    beta_floor: float = 0.5,
) -> dict:
    """
    Full WACC with market-value capital weighting.

    WACC = (E/V × Cost of Equity) + (D/V × Cost of Debt × (1 - tax_rate))
    """
    cost_of_equity = estimate_cost_of_equity(
        beta, market, risk_free_override, erp_override, beta_floor
    )

    if not market_cap or market_cap <= 0:
        logger.debug("No market_cap provided — falling back to equity-only WACC")
        return {
            "wacc": round(cost_of_equity, 6),
            "cost_of_equity": round(cost_of_equity, 6),
            "cost_of_debt_pretax": None,
            "cost_of_debt_after_tax": None,
            "tax_rate_used": None,
            "equity_weight": 1.0,
            "debt_weight": 0.0,
            "method": "equity-only-fallback",
        }

    debt = total_debt if total_debt and total_debt > 0 else 0.0
    equity = market_cap
    total_capital = equity + debt

    equity_weight = equity / total_capital
    debt_weight = debt / total_capital

    cost_of_debt_pretax = estimate_cost_of_debt(interest_expense, debt, market)
    tax_rate = estimate_tax_rate(tax_provision, pretax_income, market)
    cost_of_debt_after_tax = cost_of_debt_pretax * (1 - tax_rate)

    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt_after_tax)

    logger.debug(
        f"WACC: E_weight={equity_weight:.3f} CoE={cost_of_equity:.4f} "
        f"D_weight={debt_weight:.3f} CoD_after_tax={cost_of_debt_after_tax:.4f} "
        f"-> WACC={wacc:.4f}"
    )

    return {
        "wacc": round(wacc, 6),
        "cost_of_equity": round(cost_of_equity, 6),
        "cost_of_debt_pretax": round(cost_of_debt_pretax, 6),
        "cost_of_debt_after_tax": round(cost_of_debt_after_tax, 6),
        "tax_rate_used": round(tax_rate, 4),
        "equity_weight": round(equity_weight, 4),
        "debt_weight": round(debt_weight, 4),
        "method": "market-value-weighted",
    }


# ---------------------------------------------------------------------------
# Stage 1 growth schedule
# ---------------------------------------------------------------------------

def _build_growth_schedule(
    growth_rate_stage1: float,
    terminal_growth_rate: float,
    years: int,
) -> list[float]:
    """
    Returns a list of annual growth rates for years 1..years.

    Growth declines linearly from growth_rate_stage1 (year 1) to
    terminal_growth_rate (year N).

    For years == 1: returns [growth_rate_stage1].
    For years == 2: returns [growth_rate_stage1, terminal_growth_rate].
    """
    if years <= 0:
        return []
    if years == 1:
        return [growth_rate_stage1]

    step = (growth_rate_stage1 - terminal_growth_rate) / (years - 1)
    return [growth_rate_stage1 - step * i for i in range(years)]


# ---------------------------------------------------------------------------
# Core DCF function
# ---------------------------------------------------------------------------

def run_dcf(
    fcf_base: float,
    growth_rate_stage1: float,
    years_stage1: int,
    terminal_growth_rate: float,
    wacc: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
    current_price: Optional[float] = None,
) -> DCFResult:
    """
    Run a 2-stage DCF valuation.

    Stage 1: Explicit FCF projections for years_stage1 years with growth
             linearly declining from growth_rate_stage1 → terminal_growth_rate.
    Stage 2: Gordon Growth Model terminal value applied at the end of Stage 1.

    Parameters
    ----------
    fcf_base : float
        Most recent annual Free Cash Flow (the starting point for projections).
    growth_rate_stage1 : float
        Initial growth rate in Stage 1 (fractional, e.g. 0.15 = 15%).
    years_stage1 : int
        Number of years in the explicit forecast period.
    terminal_growth_rate : float
        Perpetuity growth rate after Stage 1. Must be < wacc.
    wacc : float
        Weighted average cost of capital (fractional, e.g. 0.10 = 10%).
    shares_outstanding : float
        Total shares outstanding (units, not millions).
    net_debt : float
        Net debt = total debt − cash (can be negative if net cash).
    current_price : float | None
        Current stock price. Used to compute margin of safety.

    Returns
    -------
    DCFResult

    Raises
    ------
    ValueError
        If wacc <= terminal_growth_rate (perpetuity would be infinite/negative).
    ValueError
        If years_stage1 < 1.
    ValueError
        If shares_outstanding <= 0.
    """
    # --- Input validation ---
    if wacc <= terminal_growth_rate:
        raise ValueError(
            f"WACC ({wacc:.4f}) must be greater than terminal_growth_rate "
            f"({terminal_growth_rate:.4f}). Otherwise the terminal value is "
            "infinite or negative."
        )
    if years_stage1 < 1:
        raise ValueError(f"years_stage1 must be ≥ 1, got {years_stage1}")
    if shares_outstanding <= 0:
        raise ValueError(f"shares_outstanding must be > 0, got {shares_outstanding}")

    # --- Stage 1: Project FCFs ---
    growth_schedule = _build_growth_schedule(growth_rate_stage1, terminal_growth_rate, years_stage1)
    stage1_cashflows: list[float] = []
    fcf_prev = fcf_base

    for g in growth_schedule:
        fcf_year = fcf_prev * (1 + g)
        stage1_cashflows.append(fcf_year)
        fcf_prev = fcf_year

    # --- PV of Stage 1 FCFs ---
    stage1_pv = sum(
        cf / (1 + wacc) ** (t + 1)
        for t, cf in enumerate(stage1_cashflows)
    )

    # --- Terminal Value (Gordon Growth Model) ---
    fcf_final = stage1_cashflows[-1]
    terminal_value = fcf_final * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
    terminal_value_pv = terminal_value / (1 + wacc) ** years_stage1

    # --- Equity Value ---
    enterprise_value = stage1_pv + terminal_value_pv
    equity_value = enterprise_value - net_debt
    intrinsic_per_share = equity_value / shares_outstanding

    # --- Margin of Safety ---
    margin_of_safety: Optional[float] = None
    if current_price is not None and current_price > 0:
        margin_of_safety = (intrinsic_per_share - current_price) / current_price

    assumptions = {
        "fcf_base": fcf_base,
        "growth_rate_stage1": growth_rate_stage1,
        "years_stage1": years_stage1,
        "terminal_growth_rate": terminal_growth_rate,
        "wacc": wacc,
        "shares_outstanding": shares_outstanding,
        "net_debt": net_debt,
        "growth_schedule": growth_schedule,
        "model": "2-stage DCF with linear growth decay + Gordon Growth terminal value",
    }

    return DCFResult(
        intrinsic_value_per_share=round(intrinsic_per_share, 4),
        current_price=current_price,
        margin_of_safety_pct=round(margin_of_safety * 100, 2) if margin_of_safety is not None else None,
        stage1_cashflows=[round(cf, 4) for cf in stage1_cashflows],
        stage1_pv=round(stage1_pv, 4),
        terminal_value=round(terminal_value, 4),
        terminal_value_pv=round(terminal_value_pv, 4),
        enterprise_value=round(enterprise_value, 4),
        equity_value=round(equity_value, 4),
        assumptions=assumptions,
    )


# ---------------------------------------------------------------------------
# Reverse DCF — implied growth rate
# ---------------------------------------------------------------------------

def reverse_dcf(
    current_price: float,
    fcf_base: float,
    years_stage1: int,
    terminal_growth_rate: float,
    wacc: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
    growth_search_min: float = -0.20,
    growth_search_max: float = 0.50,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> ReverseResult:
    """
    Solve for the implied growth rate that makes DCF intrinsic value == current_price.

    Uses bisection search — guaranteed to converge for any monotone IV(g) function.

    Parameters
    ----------
    current_price : float
        Current stock price.
    fcf_base, years_stage1, terminal_growth_rate, wacc, shares_outstanding, net_debt :
        Same as run_dcf.
    growth_search_min : float
        Lower bound for bisection search (-20% default).
    growth_search_max : float
        Upper bound for bisection search (50% default).
    tolerance : float
        Convergence tolerance on implied growth rate.
    max_iterations : int
        Safety ceiling on bisection iterations.

    Returns
    -------
    ReverseResult

    Raises
    ------
    ValueError
        If current_price <= 0.
    ValueError
        If the implied growth rate falls outside [growth_search_min, growth_search_max].
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be > 0, got {current_price}")

    def iv_at_growth(g: float) -> float:
        result = run_dcf(
            fcf_base=fcf_base,
            growth_rate_stage1=g,
            years_stage1=years_stage1,
            terminal_growth_rate=terminal_growth_rate,
            wacc=wacc,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
            current_price=None,
        )
        return result.intrinsic_value_per_share

    # Check bounds
    iv_low = iv_at_growth(growth_search_min)
    iv_high = iv_at_growth(growth_search_max)

    target = current_price

    # iv_at_growth is monotonically increasing in g
    if target < iv_low:
        raise ValueError(
            f"Current price {target:.2f} is below IV at minimum growth "
            f"{growth_search_min:.0%} ({iv_low:.2f}). "
            "Try a lower growth_search_min."
        )
    if target > iv_high:
        raise ValueError(
            f"Current price {target:.2f} is above IV at maximum growth "
            f"{growth_search_max:.0%} ({iv_high:.2f}). "
            "Try a higher growth_search_max."
        )

    # Bisection
    lo, hi = growth_search_min, growth_search_max
    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        iv_mid = iv_at_growth(mid)
        if abs(iv_mid - target) < tolerance * target:
            break
        if iv_mid < target:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < tolerance:
            break

    implied_growth = (lo + hi) / 2

    # Sensitivity: IV at ±2pp around implied growth
    sensitivity: dict[str, float] = {}
    for delta in [-0.02, -0.01, 0.00, 0.01, 0.02]:
        g_test = implied_growth + delta
        if growth_search_min <= g_test <= growth_search_max:
            try:
                sensitivity[f"g_{round((implied_growth + delta) * 100, 1):.1f}pct"] = round(iv_at_growth(g_test), 4)
            except ValueError:
                pass

    assumptions = {
        "fcf_base": fcf_base,
        "years_stage1": years_stage1,
        "terminal_growth_rate": terminal_growth_rate,
        "wacc": wacc,
        "shares_outstanding": shares_outstanding,
        "net_debt": net_debt,
        "growth_search_range": [growth_search_min, growth_search_max],
        "model": "reverse 2-stage DCF via bisection search",
    }

    return ReverseResult(
        implied_growth_rate=round(implied_growth, 6),
        current_price=current_price,
        sensitivity=sensitivity,
        assumptions=assumptions,
    )
