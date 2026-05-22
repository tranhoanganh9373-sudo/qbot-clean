from __future__ import annotations

import pandas as pd


def arbr(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 26,
) -> tuple[pd.Series, pd.Series]:
    """AR (人气) and BR (意愿) sentiment indicators.

    AR = sum(H - O) / sum(O - L) * 100      over n bars
    BR = sum(H - C[-1]) / sum(C[-1] - L) * 100  over n bars
    Typical regions: AR/BR < 50 oversold; AR/BR > 150 overbought.
    """
    ho = (high - open_).clip(lower=0).rolling(n).sum()
    ol = (open_ - low).clip(lower=0).rolling(n).sum()
    pc = close.shift(1)
    hcy = (high - pc).clip(lower=0).rolling(n).sum()
    cyl = (pc - low).clip(lower=0).rolling(n).sum()

    ar = ho / ol.replace(0, pd.NA) * 100
    br = hcy / cyl.replace(0, pd.NA) * 100
    return ar.astype(float), br.astype(float)
