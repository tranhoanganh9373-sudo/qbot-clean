"""Smoke + invariant tests for all signal-function strategies.

Each strategy is exercised on the same synthetic OHLCV frame; we assert basic
invariants (shape, dtype, no entry+exit on the same bar by default).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from claude_finance.data import load_synthetic_daily
from claude_finance.strategies import (
    adx_trend_signals,
    arbr_signals,
    aroon_signals,
    boll_signals,
    cci_signals,
    factor_momentum_signals,
    kdj_cross_signals,
    macd_cross_signals,
    multi_fusion_signals,
    rsi_divergence_signals,
    rsi_threshold_signals,
    rsrs_timing_signals,
    sma_cross_signals,
    ssa_trend_signals,
    stoch_rsi_signals,
)

_STRATEGIES = [
    ("sma_cross", sma_cross_signals),
    ("boll_reversion", lambda df: boll_signals(df, mode="reversion")),
    ("boll_breakout", lambda df: boll_signals(df, mode="breakout")),
    ("macd_cross", macd_cross_signals),
    ("rsi_threshold", rsi_threshold_signals),
    ("kdj_cross", kdj_cross_signals),
    ("cci", cci_signals),
    ("adx_trend", adx_trend_signals),
    ("aroon", aroon_signals),
    ("arbr", arbr_signals),
    ("stoch_rsi", stoch_rsi_signals),
    ("multi_fusion", multi_fusion_signals),
    ("factor_momentum", factor_momentum_signals),
    ("rsi_divergence", rsi_divergence_signals),
    ("ssa_trend", ssa_trend_signals),
    # rsrs uses smaller-than-default m so it fits in 700-bar synthetic data
    ("rsrs_timing", lambda df: rsrs_timing_signals(df, n=18, m=200)),
]


@pytest.fixture
def df() -> pd.DataFrame:
    # 700 bars so RSRS (n=18, m=200) has at least ~480 valid bars
    return load_synthetic_daily(n=700, seed=2026)


@pytest.mark.parametrize(("name", "fn"), _STRATEGIES)
def test_returns_two_boolean_series_aligned_to_input(df: pd.DataFrame, name: str, fn):
    entries, exits = fn(df)
    assert entries.dtype == bool, f"{name}: entries dtype must be bool"
    assert exits.dtype == bool, f"{name}: exits dtype must be bool"
    assert entries.index.equals(df.index), f"{name}: entries index drift"
    assert exits.index.equals(df.index), f"{name}: exits index drift"


@pytest.mark.parametrize(("name", "fn"), _STRATEGIES)
def test_no_simultaneous_entry_and_exit(df: pd.DataFrame, name: str, fn):
    entries, exits = fn(df)
    overlap = entries & exits
    assert not overlap.any(), f"{name}: same bar emits both entry and exit"


def test_boll_rejects_unknown_mode(df: pd.DataFrame):
    with pytest.raises(ValueError, match="mode must be"):
        boll_signals(df, mode="invalid")


def test_flat_market_emits_no_signals_for_threshold_strategies():
    """On a strictly flat series, RSI / SMA cross should not whipsaw."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    flat = np.full(n, 100.0)
    df = pd.DataFrame(
        {
            "open": flat,
            "high": flat,
            "low": flat,
            "close": flat,
            "volume": np.full(n, 1000, dtype="int64"),
        },
        index=idx,
    )

    e, x = rsi_threshold_signals(df)
    assert e.sum() == 0
    assert x.sum() == 0

    e, x = sma_cross_signals(df)
    assert e.sum() == 0
    assert x.sum() == 0
