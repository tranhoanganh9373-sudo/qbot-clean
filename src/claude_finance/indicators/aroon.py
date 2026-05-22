from __future__ import annotations

import pandas as pd


def aroon(high: pd.Series, low: pd.Series, n: int = 25) -> tuple[pd.Series, pd.Series]:
    """Aroon Up / Aroon Down (0–100).

    Aroon Up = position-of-max / n * 100 over last n+1 bars.
    """
    high_idx = high.rolling(n + 1).apply(lambda x: x.argmax(), raw=True)
    low_idx = low.rolling(n + 1).apply(lambda x: x.argmin(), raw=True)
    up = high_idx / n * 100
    down = low_idx / n * 100
    return up, down
