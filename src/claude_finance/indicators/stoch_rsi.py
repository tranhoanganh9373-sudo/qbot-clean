from __future__ import annotations

import pandas as pd

from claude_finance.indicators.momentum import rsi


def stoch_rsi(close: pd.Series, rsi_n: int = 14, stoch_n: int = 14) -> pd.Series:
    """Stochastic RSI (0–100). Normalised RSI over a rolling window.

    StochRSI = (RSI - min(RSI, stoch_n)) / (max - min) * 100
    """
    r = rsi(close, rsi_n)
    low = r.rolling(stoch_n).min()
    high = r.rolling(stoch_n).max()
    return ((r - low) / (high - low).replace(0, pd.NA) * 100).astype(float)
