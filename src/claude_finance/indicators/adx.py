from __future__ import annotations

import numpy as np
import pandas as pd

from claude_finance.indicators.volatility import atr


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index. Returns (+DI, -DI, ADX).

    +DI > -DI implies bullish directional pressure; ADX > 25 implies trending market.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    atr_n = atr(high, low, close, n)
    plus_di = 100 * plus_dm.rolling(n).sum() / (atr_n * n)
    minus_di = 100 * minus_dm.rolling(n).sum() / (atr_n * n)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.rolling(n).mean()
    return plus_di, minus_di, adx_series
