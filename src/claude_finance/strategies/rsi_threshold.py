"""RSI overbought/oversold reversion strategy."""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import rsi


def rsi_threshold_signals(
    df: pd.DataFrame,
    n: int = 14,
    lower: float = 30.0,
    upper: float = 70.0,
    price_col: str = "close",
) -> tuple[pd.Series, pd.Series]:
    r = rsi(df[price_col], n)
    is_oversold = r < lower
    is_overbought = r > upper
    entries = is_oversold & ~is_oversold.shift(1, fill_value=False)
    exits = is_overbought & ~is_overbought.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
