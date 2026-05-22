"""Bollinger Bands strategy.

Two modes:
  - "reversion": buy when close crosses below lower band; exit when above mid
  - "breakout":  buy when close breaks above upper band; exit when below mid
"""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import boll


def boll_signals(
    df: pd.DataFrame,
    n: int = 20,
    k: float = 2.0,
    mode: str = "reversion",
    price_col: str = "close",
) -> tuple[pd.Series, pd.Series]:
    if mode not in ("reversion", "breakout"):
        raise ValueError(f"mode must be 'reversion' or 'breakout', got {mode!r}")

    price = df[price_col]
    upper, mid, lower = boll(price, n, k)

    if mode == "reversion":
        below_lower = price < lower
        above_mid = price > mid
        entries = below_lower & ~below_lower.shift(1, fill_value=False)
        exits = above_mid & ~above_mid.shift(1, fill_value=False)
    else:  # breakout
        above_upper = price > upper
        below_mid = price < mid
        entries = above_upper & ~above_upper.shift(1, fill_value=False)
        exits = below_mid & ~below_mid.shift(1, fill_value=False)

    return entries.fillna(False), exits.fillna(False)
