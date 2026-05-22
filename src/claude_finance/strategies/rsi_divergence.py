"""RSI bullish/bearish divergence strategy.

- Bullish: price prints a lower low while RSI prints a higher low (with RSI < 50)
- Bearish: price prints a higher high while RSI prints a lower high (with RSI > 50)

Peaks are detected via scipy.signal.argrelextrema; ``order`` is the number
of points on each side that must be exceeded for a value to qualify as a
local extremum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from claude_finance.indicators import rsi


def _last_two_indices_of_kind(values: np.ndarray, kind: str, order: int) -> tuple[int, int] | None:
    cmp = np.less if kind == "low" else np.greater
    idx = argrelextrema(values, cmp, order=order)[0]
    if len(idx) < 2:
        return None
    return int(idx[-2]), int(idx[-1])


def rsi_divergence_signals(
    df: pd.DataFrame,
    rsi_n: int = 14,
    order: int = 5,
    price_col: str = "close",
) -> tuple[pd.Series, pd.Series]:
    close = df[price_col].astype(float)
    r = rsi(close, rsi_n)

    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)

    price_arr = close.to_numpy()
    rsi_arr = r.to_numpy()

    for i in range(2 * order + rsi_n, len(close)):
        p_lows = _last_two_indices_of_kind(price_arr[: i + 1], "low", order)
        r_lows = _last_two_indices_of_kind(rsi_arr[: i + 1], "low", order)
        if p_lows and r_lows and not np.isnan(rsi_arr[r_lows[-1]]):
            if (
                price_arr[p_lows[-1]] < price_arr[p_lows[-2]]
                and rsi_arr[r_lows[-1]] > rsi_arr[r_lows[-2]]
                and rsi_arr[i] < 50
            ):
                entries.iloc[i] = True

        p_highs = _last_two_indices_of_kind(price_arr[: i + 1], "high", order)
        r_highs = _last_two_indices_of_kind(rsi_arr[: i + 1], "high", order)
        if p_highs and r_highs and not np.isnan(rsi_arr[r_highs[-1]]):
            if (
                price_arr[p_highs[-1]] > price_arr[p_highs[-2]]
                and rsi_arr[r_highs[-1]] < rsi_arr[r_highs[-2]]
                and rsi_arr[i] > 50
            ):
                exits.iloc[i] = True

    return entries, exits
