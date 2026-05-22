"""ADX-confirmed trend-following strategy.

Buy when ADX is rising AND +DI > -DI AND price above slow EMA;
exit when +DI crosses below -DI.
"""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import adx, ema


def adx_trend_signals(
    df: pd.DataFrame,
    n: int = 14,
    ema_period: int = 55,
    adx_min: float = 20.0,
) -> tuple[pd.Series, pd.Series]:
    plus_di, minus_di, adx_series = adx(df["high"], df["low"], df["close"], n)
    trend_ema = ema(df["close"], ema_period)

    bull = (plus_di > minus_di) & (adx_series > adx_min) & (df["close"] > trend_ema)
    bear = plus_di < minus_di
    entries = bull & ~bull.shift(1, fill_value=False)
    exits = bear & ~bear.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
