"""KDJ K/D crossover strategy with overbought/oversold filter."""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import kdj


def kdj_cross_signals(
    df: pd.DataFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> tuple[pd.Series, pd.Series]:
    k, d, _j = kdj(df["high"], df["low"], df["close"], n, m1, m2)
    k_above = k > d
    cross_up = k_above & ~k_above.shift(1, fill_value=False)
    cross_dn = ~k_above & k_above.shift(1, fill_value=False)
    entries = cross_up & (k < 80)
    exits = cross_dn & (k > 20)
    return entries.fillna(False), exits.fillna(False)
