"""SSA-smoothed trend strategy.

Buy when close crosses above the SSA-reconstructed series; exit on cross
below. The SSA series is the rank-1 SVD reconstruction of a trajectory
matrix, so it lags less than an SMA of comparable smoothness.
"""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import ssa


def ssa_trend_signals(
    df: pd.DataFrame,
    window: int = 30,
    price_col: str = "close",
) -> tuple[pd.Series, pd.Series]:
    close = df[price_col].astype(float)
    smooth = ssa(close, window)
    above = close > smooth
    entries = above & ~above.shift(1, fill_value=False)
    exits = ~above & above.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
