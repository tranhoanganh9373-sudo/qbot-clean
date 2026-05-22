from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from claude_finance.data.schema import OHLCV_COLUMNS, validate_ohlcv
from claude_finance.strategies import sma_cross_signals


def _make_ohlcv(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1000, dtype="int64"),
        },
        index=idx,
    )


def test_validate_ohlcv_rejects_missing_columns():
    df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"]))
    with pytest.raises(ValueError, match="missing columns"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_non_datetime_index():
    df = pd.DataFrame({c: [1.0] for c in OHLCV_COLUMNS})
    with pytest.raises(TypeError):
        validate_ohlcv(df)


def test_sma_cross_rejects_invalid_periods():
    df = _make_ohlcv(np.arange(50, dtype=float))
    with pytest.raises(ValueError, match="must be <"):
        sma_cross_signals(df, fast=30, slow=10)


def test_sma_cross_emits_entry_on_upward_cross():
    close = np.concatenate([np.linspace(100, 50, 30), np.linspace(50, 150, 30)])
    df = _make_ohlcv(close)

    entries, exits = sma_cross_signals(df, fast=5, slow=20)

    assert entries.dtype == bool
    assert exits.dtype == bool
    assert entries.index.equals(df.index)
    assert entries.sum() >= 1, "expected at least one upward crossover"


def test_sma_cross_emits_exit_on_downward_cross():
    close = np.concatenate([np.linspace(50, 150, 30), np.linspace(150, 50, 30)])
    df = _make_ohlcv(close)

    _, exits = sma_cross_signals(df, fast=5, slow=20)

    assert exits.sum() >= 1, "expected at least one downward crossover"


def test_sma_cross_no_signals_on_flat_series():
    df = _make_ohlcv(np.full(60, 100.0))
    entries, exits = sma_cross_signals(df, fast=5, slow=20)
    assert entries.sum() == 0
    assert exits.sum() == 0
