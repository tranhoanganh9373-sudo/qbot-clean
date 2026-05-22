"""RSRS (Relative Strength of Relative Strength) timing indicator.

For each window of n bars, regress today's high on today's low; the slope
quantifies how strongly highs lead lows (resistance vs support). The slope is
then z-scored across the trailing m-bar window and multiplied by the
regression R², giving an asymmetry score commonly used as a market-timing
filter for index ETFs.

Reference: 光大证券《基于阻力支撑相对强度的择时模型》
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ols_slope_r2(low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    """Return (slope, r2) of high = a + slope * low."""
    if np.std(low) == 0 or np.std(high) == 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(low, high, 1)
    predicted = slope * low + intercept
    ss_res = np.sum((high - predicted) ** 2)
    ss_tot = np.sum((high - high.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r2)


def rsrs(
    high: pd.Series,
    low: pd.Series,
    n: int = 18,
    m: int = 600,
) -> tuple[pd.Series, pd.Series]:
    """Return (zscore × r2 series, raw slope series).

    n: rolling regression window (slope estimation)
    m: z-score normalisation window (across slopes)
    """
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    length = len(h)

    slope = np.full(length, np.nan)
    r2 = np.full(length, np.nan)
    for i in range(n - 1, length):
        s, r = _ols_slope_r2(lo[i - n + 1 : i + 1], h[i - n + 1 : i + 1])
        slope[i] = s
        r2[i] = r

    slope_s = pd.Series(slope, index=high.index)
    r2_s = pd.Series(r2, index=high.index)
    zscore = (slope_s - slope_s.rolling(m).mean()) / slope_s.rolling(m).std(ddof=1)
    return (zscore * r2_s).rename("rsrs"), slope_s.rename("slope")
