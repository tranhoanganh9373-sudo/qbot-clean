"""RSRS market-timing strategy.

Buy when the RSRS score crosses above ``buy_threshold``; exit when it crosses
below ``sell_threshold``. Defaults follow the 光大证券 original paper.

Requires ~n + m bars of history before the first signal can fire (typically
~620 bars with defaults).
"""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import rsrs


def rsrs_timing_signals(
    df: pd.DataFrame,
    n: int = 18,
    m: int = 600,
    buy_threshold: float = 0.7,
    sell_threshold: float = -0.7,
) -> tuple[pd.Series, pd.Series]:
    score, _slope = rsrs(df["high"], df["low"], n, m)
    bull = score > buy_threshold
    bear = score < sell_threshold
    entries = bull & ~bull.shift(1, fill_value=False)
    exits = bear & ~bear.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
