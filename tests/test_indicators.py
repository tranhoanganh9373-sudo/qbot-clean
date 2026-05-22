from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from claude_finance.indicators import atr, bias, boll, ema, kdj, macd, rsi, sma


@pytest.fixture
def close() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.02, 100)
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="close")


@pytest.fixture
def hlc(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = close * 1.01
    low = close * 0.99
    return high, low, close


def test_sma_matches_rolling_mean(close: pd.Series):
    s = sma(close, 10)
    assert s.index.equals(close.index)
    assert s.iloc[9] == pytest.approx(close.iloc[:10].mean())


def test_ema_first_value_equals_input(close: pd.Series):
    e = ema(close, 12)
    assert e.iloc[0] == pytest.approx(close.iloc[0])


def test_macd_returns_three_series(close: pd.Series):
    dif, dea, hist = macd(close)
    assert dif.index.equals(close.index)
    assert hist.iloc[-1] == pytest.approx((dif.iloc[-1] - dea.iloc[-1]) * 2)


def test_rsi_in_0_100_range(close: pd.Series):
    r = rsi(close).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_all_up_returns_100():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    up = pd.Series(np.linspace(100, 200, 30), index=idx)
    r = rsi(up, n=14)
    assert r.iloc[-1] == pytest.approx(100.0)


def test_rsi_flat_returns_50():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    flat = pd.Series(np.full(30, 100.0), index=idx)
    r = rsi(flat, n=14)
    assert r.iloc[-1] == pytest.approx(50.0)


def test_kdj_returns_three_series(hlc):
    high, low, close = hlc
    k, d, j = kdj(high, low, close)
    assert k.index.equals(close.index)
    assert j.iloc[-1] == pytest.approx(3 * k.iloc[-1] - 2 * d.iloc[-1])


def test_boll_ordering(close: pd.Series):
    upper, mid, lower = boll(close, 20, 2.0)
    valid = upper.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_atr_non_negative(hlc):
    high, low, close = hlc
    a = atr(high, low, close).dropna()
    assert (a >= 0).all()


def test_bias_centered_on_zero(close: pd.Series):
    b = bias(close, 6).dropna()
    assert abs(b.mean()) < 20
