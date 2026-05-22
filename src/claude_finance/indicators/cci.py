from __future__ import annotations

import numpy as np
import pandas as pd


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Commodity Channel Index.

    CCI = (TP - SMA(TP, n)) / (0.015 * MAD(TP, n)),  TP = (H + L + C) / 3.
    Typically: <-100 oversold, >+100 overbought.
    """
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)
