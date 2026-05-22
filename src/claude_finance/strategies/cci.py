"""CCI cross strategy.

Buy when CCI crosses up through -100 (recovering from oversold);
sell when CCI crosses down through +100 (rolling over from overbought).
"""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import cci


def cci_signals(
    df: pd.DataFrame,
    n: int = 14,
    lower: float = -100.0,
    upper: float = 100.0,
) -> tuple[pd.Series, pd.Series]:
    c = cci(df["high"], df["low"], df["close"], n)
    cross_up = (c > lower) & (c.shift(1) <= lower)
    cross_dn = (c < upper) & (c.shift(1) >= upper)
    return cross_up.fillna(False), cross_dn.fillna(False)
