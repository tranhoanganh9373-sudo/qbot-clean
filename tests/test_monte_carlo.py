from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from claude_finance.data import load_synthetic_daily
from claude_finance.risk import annualized_volatility, cagr, simulate_paths, value_at_risk


def test_cagr_on_known_doubling():
    """Price doubles in exactly 365 days → CAGR = 100%."""
    idx = pd.date_range("2024-01-01", "2025-01-01", periods=2)
    series = pd.Series([100.0, 200.0], index=idx)
    assert cagr(series) == pytest.approx(1.0, abs=0.01)


def test_annualized_volatility_non_negative():
    df = load_synthetic_daily(n=300, seed=0)
    vol = annualized_volatility(df["close"])
    assert vol > 0


def test_simulate_paths_shape_and_init():
    paths = simulate_paths(init_price=100, mu=0.08, sigma=0.20, horizon_days=30, n_paths=500)
    assert paths.shape == (500, 31)
    assert np.allclose(paths[:, 0], 100.0)


def test_simulate_paths_deterministic_with_seed():
    a = simulate_paths(init_price=50, mu=0.1, sigma=0.3, horizon_days=10, n_paths=10, seed=7)
    b = simulate_paths(init_price=50, mu=0.1, sigma=0.3, horizon_days=10, n_paths=10, seed=7)
    assert np.array_equal(a, b)


def test_value_at_risk_positive_for_volatile_series():
    df = load_synthetic_daily(n=500, seed=42, sigma=0.03)
    var = value_at_risk(df["close"], horizon_days=20, confidence=0.95)
    assert var > 0


def test_value_at_risk_rejects_bad_confidence():
    df = load_synthetic_daily(n=100, seed=0)
    with pytest.raises(ValueError, match="confidence"):
        value_at_risk(df["close"], confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        value_at_risk(df["close"], confidence=0.0)
