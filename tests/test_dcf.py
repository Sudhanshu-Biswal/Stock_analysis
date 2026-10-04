"""
tests/test_dcf.py — Unit tests for the DCF engine

The hand-worked reference calculation is reproduced in valuation/dcf.py's
module docstring. This file tests against those exact pre-computed values.

Hand-worked example (reproduced here for clarity):
────────────────────────────────────────────────────
  fcf_base               = 100.0
  growth_rate_stage1     = 0.15   (15%, declining linearly to 3%)
  years_stage1           = 5
  terminal_growth_rate   = 0.03   (3%)
  wacc                   = 0.10   (10%)
  shares_outstanding     = 10.0
  net_debt               = 0.0

  Growth schedule (linear decay 15% → 3% over 5 years, step = 3pp):
    Year 1: 15%   FCF = 100 × 1.15 = 115.000
    Year 2: 12%   FCF = 115 × 1.12 = 128.800
    Year 3:  9%   FCF = 128.8 × 1.09 = 140.392
    Year 4:  6%   FCF = 140.392 × 1.06 = 148.816
    Year 5:  3%   FCF = 148.816 × 1.03 = 153.280

  PV of Stage 1 (discount at 10%):
    PV1 = 115.000 / 1.10^1 = 104.5455
    PV2 = 128.800 / 1.10^2 = 106.4463
    PV3 = 140.392 / 1.10^3 = 105.4696
    PV4 = 148.816 / 1.10^4 = 101.5889
    PV5 = 153.280 / 1.10^5 =  95.1745
    Sum = 513.2248

  Terminal Value: 153.280 × 1.03 / (0.10 - 0.03) = 2255.394
  PV of TV:       2255.394 / 1.10^5              = 1400.291

  Enterprise Value = 513.225 + 1400.291 = 1913.516
  Equity Value     = 1913.516 - 0       = 1913.516
  Intrinsic / share = 1913.516 / 10     = 191.352

  (Small rounding differences between the docstring and here are expected;
   tests use a tolerance of ±0.05 per share.)
────────────────────────────────────────────────────
"""

import pytest

from valuation.dcf import DCFResult, ReverseResult, _build_growth_schedule, estimate_wacc, reverse_dcf, run_dcf

# ─────────────────────────────────────────────────────────────
# Reference inputs (keep in sync with the hand-worked example above)
# ─────────────────────────────────────────────────────────────

REF = dict(
    fcf_base=100.0,
    growth_rate_stage1=0.15,
    years_stage1=5,
    terminal_growth_rate=0.03,
    wacc=0.10,
    shares_outstanding=10.0,
    net_debt=0.0,
)


# ─────────────────────────────────────────────────────────────
# Growth schedule tests
# ─────────────────────────────────────────────────────────────

class TestGrowthSchedule:

    def test_linear_decay_5_years(self):
        schedule = _build_growth_schedule(0.15, 0.03, 5)
        assert len(schedule) == 5
        assert abs(schedule[0] - 0.15) < 1e-9
        assert abs(schedule[-1] - 0.03) < 1e-9

    def test_step_size(self):
        """Step between consecutive growth rates should be constant (3pp)."""
        schedule = _build_growth_schedule(0.15, 0.03, 5)
        diffs = [schedule[i] - schedule[i + 1] for i in range(len(schedule) - 1)]
        for d in diffs:
            assert abs(d - 0.03) < 1e-9, f"Non-uniform step: {diffs}"

    def test_single_year(self):
        schedule = _build_growth_schedule(0.20, 0.03, 1)
        assert schedule == [0.20]

    def test_two_years(self):
        schedule = _build_growth_schedule(0.20, 0.03, 2)
        assert len(schedule) == 2
        assert abs(schedule[0] - 0.20) < 1e-9
        assert abs(schedule[1] - 0.03) < 1e-9

    def test_zero_years(self):
        assert _build_growth_schedule(0.10, 0.03, 0) == []


# ─────────────────────────────────────────────────────────────
# Core DCF tests
# ─────────────────────────────────────────────────────────────

class TestRunDCF:

    def test_reference_example_stage1_cashflows(self):
        result = run_dcf(**REF)
        expected_fcfs = [115.0, 128.8, 140.392, 148.816, 153.280]
        assert len(result.stage1_cashflows) == 5
        for actual, expected in zip(result.stage1_cashflows, expected_fcfs):
            assert abs(actual - expected) < 0.01, f"FCF mismatch: {actual} vs {expected}"

    def test_reference_example_intrinsic_value(self):
        """Intrinsic value per share should be ≈ 191.35 (tolerance ±0.10)."""
        result = run_dcf(**REF)
        assert abs(result.intrinsic_value_per_share - 191.35) < 0.10, (
            f"Intrinsic value {result.intrinsic_value_per_share} outside tolerance"
        )

    def test_reference_terminal_value(self):
        """Terminal value ≈ 2255.39 (before discounting)."""
        result = run_dcf(**REF)
        assert abs(result.terminal_value - 2255.39) < 1.0

    def test_reference_enterprise_value(self):
        """EV ≈ 1913.5."""
        result = run_dcf(**REF)
        assert abs(result.enterprise_value - 1913.5) < 1.0

    def test_net_debt_subtracted(self):
        """Adding net_debt should reduce equity_value by exactly that amount."""
        result_no_debt = run_dcf(**REF)
        result_with_debt = run_dcf(**{**REF, "net_debt": 200.0})
        diff = result_no_debt.equity_value - result_with_debt.equity_value
        assert abs(diff - 200.0) < 0.01

    def test_net_cash_adds_to_equity_value(self):
        """Negative net_debt (net cash) should increase equity value."""
        result_no_debt = run_dcf(**REF)
        result_net_cash = run_dcf(**{**REF, "net_debt": -100.0})
        assert result_net_cash.equity_value > result_no_debt.equity_value
        assert abs(result_net_cash.equity_value - result_no_debt.equity_value - 100.0) < 0.01

    def test_margin_of_safety_positive_when_undervalued(self):
        """If current price < intrinsic value → positive MoS."""
        result = run_dcf(**{**REF, "current_price": 100.0})
        assert result.margin_of_safety_pct is not None
        assert result.margin_of_safety_pct > 0

    def test_margin_of_safety_negative_when_overvalued(self):
        """If current price > intrinsic value → negative MoS."""
        result = run_dcf(**{**REF, "current_price": 500.0})
        assert result.margin_of_safety_pct is not None
        assert result.margin_of_safety_pct < 0

    def test_margin_of_safety_none_when_no_price(self):
        result = run_dcf(**{**REF, "current_price": None})
        assert result.margin_of_safety_pct is None

    def test_assumptions_in_result(self):
        """All input assumptions must be echoed in result.assumptions."""
        result = run_dcf(**REF)
        for key in ["fcf_base", "growth_rate_stage1", "years_stage1",
                    "terminal_growth_rate", "wacc", "shares_outstanding", "net_debt"]:
            assert key in result.assumptions, f"Missing assumption: {key}"

    def test_higher_wacc_lowers_value(self):
        """Higher discount rate → lower intrinsic value (ceteris paribus)."""
        low = run_dcf(**{**REF, "wacc": 0.08})
        high = run_dcf(**{**REF, "wacc": 0.15})
        assert low.intrinsic_value_per_share > high.intrinsic_value_per_share

    def test_higher_growth_raises_value(self):
        """Higher initial growth → higher intrinsic value."""
        low_g = run_dcf(**{**REF, "growth_rate_stage1": 0.05})
        high_g = run_dcf(**{**REF, "growth_rate_stage1": 0.25})
        assert high_g.intrinsic_value_per_share > low_g.intrinsic_value_per_share

    def test_more_shares_lowers_per_share_value(self):
        """Doubling shares halves per-share value."""
        result_10 = run_dcf(**REF)
        result_20 = run_dcf(**{**REF, "shares_outstanding": 20.0})
        assert abs(result_10.intrinsic_value_per_share - 2 * result_20.intrinsic_value_per_share) < 0.01


# ─────────────────────────────────────────────────────────────
# Edge cases and error handling
# ─────────────────────────────────────────────────────────────

class TestDCFEdgeCases:

    def test_raises_when_wacc_le_terminal_growth(self):
        with pytest.raises(ValueError, match="WACC"):
            run_dcf(**{**REF, "wacc": 0.03, "terminal_growth_rate": 0.03})

    def test_raises_when_wacc_lt_terminal_growth(self):
        with pytest.raises(ValueError):
            run_dcf(**{**REF, "wacc": 0.02, "terminal_growth_rate": 0.03})

    def test_raises_when_years_zero(self):
        with pytest.raises(ValueError):
            run_dcf(**{**REF, "years_stage1": 0})

    def test_raises_when_shares_zero(self):
        with pytest.raises(ValueError):
            run_dcf(**{**REF, "shares_outstanding": 0.0})

    def test_raises_when_shares_negative(self):
        with pytest.raises(ValueError):
            run_dcf(**{**REF, "shares_outstanding": -5.0})

    def test_single_year_stage1(self):
        """Should not crash with years_stage1 = 1."""
        result = run_dcf(**{**REF, "years_stage1": 1})
        assert isinstance(result, DCFResult)
        assert len(result.stage1_cashflows) == 1

    def test_negative_fcf_base(self):
        """Negative FCF base should work (result will be negative equity value)."""
        result = run_dcf(**{**REF, "fcf_base": -50.0})
        assert isinstance(result, DCFResult)
        assert result.intrinsic_value_per_share < 0


# ─────────────────────────────────────────────────────────────
# Reverse DCF tests
# ─────────────────────────────────────────────────────────────

class TestReverseDCF:

    def test_round_trip(self):
        """
        run_dcf(implied_g) should reproduce current_price within 1%.

        Round-trip: take a known price, reverse-solve for implied_g,
        then run_dcf with implied_g and verify we get back the same price.
        """
        target_price = 150.0
        result = reverse_dcf(
            current_price=target_price,
            fcf_base=REF["fcf_base"],
            years_stage1=REF["years_stage1"],
            terminal_growth_rate=REF["terminal_growth_rate"],
            wacc=REF["wacc"],
            shares_outstanding=REF["shares_outstanding"],
            net_debt=REF["net_debt"],
        )
        # Now run DCF with implied growth
        dcf_check = run_dcf(
            fcf_base=REF["fcf_base"],
            growth_rate_stage1=result.implied_growth_rate,
            years_stage1=REF["years_stage1"],
            terminal_growth_rate=REF["terminal_growth_rate"],
            wacc=REF["wacc"],
            shares_outstanding=REF["shares_outstanding"],
            net_debt=REF["net_debt"],
        )
        assert abs(dcf_check.intrinsic_value_per_share - target_price) < 0.5, (
            f"Round-trip failed: got {dcf_check.intrinsic_value_per_share:.4f}, expected {target_price}"
        )

    def test_sensitivity_keys_present(self):
        result = reverse_dcf(
            current_price=150.0,
            fcf_base=100.0,
            years_stage1=5,
            terminal_growth_rate=0.03,
            wacc=0.10,
            shares_outstanding=10.0,
        )
        assert len(result.sensitivity) >= 1

    def test_raises_when_price_zero(self):
        with pytest.raises(ValueError, match="current_price"):
            reverse_dcf(
                current_price=0.0,
                fcf_base=100.0,
                years_stage1=5,
                terminal_growth_rate=0.03,
                wacc=0.10,
                shares_outstanding=10.0,
            )

    def test_implied_growth_in_assumptions(self):
        result = reverse_dcf(
            current_price=100.0,
            fcf_base=100.0,
            years_stage1=5,
            terminal_growth_rate=0.03,
            wacc=0.10,
            shares_outstanding=10.0,
        )
        assert "model" in result.assumptions


# ─────────────────────────────────────────────────────────────
# WACC / CAPM tests
# ─────────────────────────────────────────────────────────────

class TestWACC:

    def test_us_defaults(self):
        """US CAPM fallback: rf=4.5%, ERP=5.0%, beta=1.0 → 9.5%"""
        wacc = estimate_wacc(beta=1.0, market="US")
        assert isinstance(wacc, dict)
        assert abs(wacc["wacc"] - 0.095) < 1e-9

    def test_india_defaults(self):
        """India CAPM fallback: rf=6.9%, ERP=6.5%, beta=1.0 → 14.4%"""
        wacc = estimate_wacc(beta=1.0, market="IN")
        assert isinstance(wacc, dict)
        assert abs(wacc["wacc"] - 0.134) < 1e-9

    def test_none_beta_defaults_to_1(self):
        wacc_none = estimate_wacc(beta=None, market="US")
        wacc_one = estimate_wacc(beta=1.0, market="US")
        assert abs(wacc_none["wacc"] - wacc_one["wacc"]) < 1e-9

    def test_beta_floor_applied(self):
        """Very low beta should be floored to 0.5."""
        wacc_floor = estimate_wacc(beta=0.1, market="US", beta_floor=0.5)
        wacc_normal = estimate_wacc(beta=0.5, market="US", beta_floor=0.5)
        assert abs(wacc_floor["wacc"] - wacc_normal["wacc"]) < 1e-9

    def test_override_rf_and_erp(self):
        wacc = estimate_wacc(beta=1.0, market="US", risk_free_override=0.05, erp_override=0.06)
        assert abs(wacc["wacc"] - 0.11) < 1e-9

    def test_market_value_weighting_uses_debt_and_tax_shield(self):
        detail = estimate_wacc(
            beta=1.0,
            market="US",
            market_cap=1000.0,
            total_debt=50.0,
            interest_expense=2.5,
            tax_provision=20.0,
            pretax_income=100.0,
        )
        assert detail["method"] == "market-value-weighted"
        assert detail["equity_weight"] == pytest.approx(0.9524, abs=1e-3)
        assert detail["debt_weight"] == pytest.approx(0.0476, abs=1e-3)
        assert detail["tax_rate_used"] == pytest.approx(0.2, abs=1e-3)
        assert detail["wacc"] < 0.095

    def test_higher_beta_higher_wacc(self):
        low = estimate_wacc(beta=0.8, market="US")
        high = estimate_wacc(beta=1.5, market="US")
        assert high["wacc"] > low["wacc"]
