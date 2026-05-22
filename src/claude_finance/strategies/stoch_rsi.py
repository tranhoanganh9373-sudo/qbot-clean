"""Stochastic RSI mean-reversion strategy."""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import stoch_rsi


def stoch_rsi_signals(
    df: pd.DataFrame,
    rsi_n: int = 14,
    stoch_n: int = 14,
    lower: float = 20.0,
    upper: float = 80.0,
    price_col: str = "close",
) -> tuple[pd.Series, pd.Series]:
    sr = stoch_rsi(df[price_col], rsi_n, stoch_n)
    is_oversold = sr < lower
    is_overbought = sr > upper
    entries = is_oversold & ~is_oversold.shift(1, fill_value=False)
    exits = is_overbought & ~is_overbought.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
