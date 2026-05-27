"""Unit tests for ``claude_finance.factor_quality_gate``.

Covers the three validators that were adapted from QuantaAlpha's
``factors/regulator/consistency_checker.py`` (rule-based subset only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from claude_finance.factor_quality_gate import (
    ComplexityBreakdown,
    complexity_breakdown,
    complexity_score,
    consistency_check,
    redundancy_check,
)


# --------------------------------------------------------------------------- #
# consistency_check                                                            #
# --------------------------------------------------------------------------- #


class TestConsistencyCheck:
    def test_ibs_hypothesis_expr_and_code_aligned(self) -> None:
        # IBS reversal: negative sign in expression matches "反转" in hypothesis.
        hyp = "高 IBS 反转: 收盘价接近高点后倾向下跌"
        expr = "-(close - low) / (high - low)"
        code = (
            "def ibs(df):\n"
            "    return -(df['close'] - df['low']) / (df['high'] - df['low'])"
        )
        ok, issues = consistency_check(hyp, expr, code)
        assert ok, f"expected consistent, got issues={issues}"
        assert issues == []

    def test_direction_mismatch_detected(self) -> None:
        # Hypothesis says "反转" (negative) but the expression has no leading
        # minus → positive sign. Must fail with R3 issue.
        hyp = "高 IBS 反转: 应当下跌"
        expr = "(close - low) / (high - low)"
        code = "def x(df): return (df['close'] - df['low']) / (df['high'] - df['low'])"
        ok, issues = consistency_check(hyp, expr, code)
        assert ok is False
        assert any("R3" in i for i in issues), issues

    def test_missing_symbol_in_code(self) -> None:
        # Expression references `volume` but code does not.
        hyp = "成交量动量 增长 bullish"
        expr = "volume / close"
        code = "def x(df): return df['close']"
        ok, issues = consistency_check(hyp, expr, code)
        assert ok is False
        assert any("R1" in i for i in issues), issues

    def test_neutral_hypothesis_no_direction_issue(self) -> None:
        # No directional words → R3 should not fire even when expr sign exists.
        hyp = "some descriptive label without direction words"
        expr = "-(close - low) / (high - low)"
        code = "def x(df): return -(df['close'] - df['low']) / (df['high'] - df['low'])"
        ok, issues = consistency_check(hyp, expr, code)
        assert ok, f"unexpected issues={issues}"


# --------------------------------------------------------------------------- #
# complexity_score                                                             #
# --------------------------------------------------------------------------- #


class TestComplexityScore:
    def test_simple_ratio_low_score(self) -> None:
        score = complexity_score("close / low")
        assert 0.0 <= score <= 3.0, score

    def test_medium_ibs_in_middle_band(self) -> None:
        score = complexity_score("(close - low) / (high - low)")
        assert 2.0 <= score <= 6.0, score

    def test_complex_expression_higher_than_simple(self) -> None:
        simple = complexity_score("close / low")
        complex_ = complexity_score(
            "((high - low) - (close - open)) / (high - low + 1e-9)"
        )
        # Strictly: complex > simple (the gate is monotonic in components).
        assert complex_ > simple

    def test_breakdown_components_match_intuition(self) -> None:
        br: ComplexityBreakdown = complexity_breakdown(
            "(close - low) / (high - low)"
        )
        assert br.unique_symbols == 3      # close, low, high
        assert br.nesting_depth == 1
        assert br.operator_count >= 3       # - / -
        assert br.score == pytest.approx(2.5, abs=0.5)

    def test_score_bounded_to_ten(self) -> None:
        # Even a deeply nested huge expression must stay within [0, 10].
        huge = "((((" + " + ".join(["close"] * 50) + "))))"
        score = complexity_score(huge)
        assert 0.0 <= score <= 10.0


# --------------------------------------------------------------------------- #
# redundancy_check                                                             #
# --------------------------------------------------------------------------- #


def _make_panel(n_codes: int = 50, n_days: int = 120, seed: int = 42):
    """Build a synthetic (code, date) MultiIndex panel for redundancy tests."""
    rng = np.random.default_rng(seed)
    codes = [f"SH60000{i}" for i in range(n_codes)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    idx = pd.MultiIndex.from_product([codes, dates], names=["code", "date"])
    n = len(idx)
    base = pd.Series(rng.standard_normal(n), index=idx, name="amp_imb_20d")
    return rng, idx, base


class TestRedundancyCheck:
    def test_independent_factor_passes(self) -> None:
        rng, idx, base = _make_panel()
        indep = pd.Series(rng.standard_normal(len(idx)), index=idx)
        panel = pd.DataFrame({"amp_imb_20d": base})
        ok, rho = redundancy_check(indep, panel)
        assert ok is True
        assert rho["amp_imb_20d"] < 0.30

    def test_correlated_factor_fails(self) -> None:
        rng, idx, base = _make_panel()
        corr = pd.Series(
            0.9 * base.values + 0.1 * rng.standard_normal(len(idx)),
            index=idx,
        )
        panel = pd.DataFrame({"amp_imb_20d": base})
        ok, rho = redundancy_check(corr, panel)
        assert ok is False
        assert rho["amp_imb_20d"] > 0.30

    def test_custom_threshold_respected(self) -> None:
        rng, idx, base = _make_panel()
        # Mildly correlated factor (~0.3 Spearman).
        mild = pd.Series(
            0.3 * base.values + 0.7 * rng.standard_normal(len(idx)),
            index=idx,
        )
        panel = pd.DataFrame({"amp_imb_20d": base})
        ok_strict, _ = redundancy_check(mild, panel, threshold=0.10)
        ok_lenient, _ = redundancy_check(mild, panel, threshold=0.60)
        # Strict threshold should fail, lenient should pass on this synthetic case.
        assert ok_strict is False
        assert ok_lenient is True

    def test_type_errors_are_explicit(self) -> None:
        rng, idx, base = _make_panel()
        with pytest.raises(TypeError):
            redundancy_check(base.values, pd.DataFrame({"x": base}))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            redundancy_check(base, base)  # type: ignore[arg-type]
