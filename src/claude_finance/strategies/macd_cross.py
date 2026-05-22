"""MACD golden / death cross strategy."""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import macd


def macd_cross_signals(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close",
) -> tuple[pd.Series, pd.Series]:
    dif, dea, _hist = macd(df[price_col], fast, slow, signal)
    above = dif > dea
    entries = above & ~above.shift(1, fill_value=False)
    exits = ~above & above.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
