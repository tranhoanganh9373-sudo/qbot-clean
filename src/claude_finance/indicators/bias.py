from __future__ import annotations

import pandas as pd

from claude_finance.indicators.trend import sma


def bias(close: pd.Series, n: int = 6) -> pd.Series:
    """Bias ratio: (close - SMA_n) / SMA_n * 100, in percent."""
    ma = sma(close, n)
    return (close - ma) / ma * 100
