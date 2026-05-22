"""Single-asset price-volume factor momentum strategy.

NOT a cross-sectional multi-factor model — the original qbot multi_factor
strategy ranked across 沪深300 constituents using PE/PB/ROE fundamentals,
which requires a fundamentals data loader we don't ship yet.

Instead this scores each bar by a composite of price-derived factors
(momentum, volatility inverse, volume z-score), then trades on percentile
crossings relative to the trailing ``lookback`` window.
"""
from __future__ import annotations

import pandas as pd


def factor_momentum_signals(
    df: pd.DataFrame,
    mom_n: int = 20,
    vol_n: int = 20,
    lookback: int = 60,
    enter_pct: float = 0.80,
    exit_pct: float = 0.20,
) -> tuple[pd.Series, pd.Series]:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    momentum = close.pct_change(mom_n)
    vol_inv = 1.0 / close.pct_change().rolling(vol_n).std()
    vol_z = (volume - volume.rolling(vol_n).mean()) / volume.rolling(vol_n).std(ddof=1)

    def _zscore(s: pd.Series) -> pd.Series:
        return (s - s.rolling(lookback).mean()) / s.rolling(lookback).std(ddof=1)

    score = (_zscore(momentum) + _zscore(vol_inv) + _zscore(vol_z)) / 3

    enter_thr = score.rolling(lookback).quantile(enter_pct)
    exit_thr = score.rolling(lookback).quantile(exit_pct)

    is_strong = score > enter_thr
    is_weak = score < exit_thr
    entries = is_strong & ~is_strong.shift(1, fill_value=False)
    exits = is_weak & ~is_weak.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
