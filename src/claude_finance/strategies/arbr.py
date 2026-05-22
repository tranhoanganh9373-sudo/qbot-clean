"""AR/BR sentiment reversion strategy.

Buy when AR and BR are both depressed (sentiment exhausted on the way down);
sell when both are elevated (euphoria).
"""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import arbr


def arbr_signals(
    df: pd.DataFrame,
    n: int = 26,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
) -> tuple[pd.Series, pd.Series]:
    ar, br = arbr(df["open"], df["high"], df["low"], df["close"], n)
    oversold = (ar < low_threshold) & (br < low_threshold)
    overbought = (ar > high_threshold) & (br > high_threshold)
    entries = oversold & ~oversold.shift(1, fill_value=False)
    exits = overbought & ~overbought.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
