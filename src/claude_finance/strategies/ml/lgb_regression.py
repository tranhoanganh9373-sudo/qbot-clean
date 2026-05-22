"""LightGBM regression: predict N-bar forward return, trade on percentile.

Refactor of qbot's pytrader/strategies/lgb_strategy.py — original was
cross-sectional top-K selection across 沪深300 constituents with talib
features. Single-asset version uses the same idea but with our own
indicators (no talib dependency).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from claude_finance.indicators import atr, bias, kdj, macd, rsi, sma


def _features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    _dif, _dea, hist = macd(close)
    k, d, j = kdj(high, low, close)

    feats = pd.DataFrame(
        {
            "ret_1": close.pct_change(1),
            "ret_5": close.pct_change(5),
            "ret_20": close.pct_change(20),
            "rsi_14": rsi(close, 14),
            "bias_6": bias(close, 6),
            "macd_hist": hist,
            "kdj_k": k,
            "kdj_d": d,
            "kdj_j": j,
            "atr_14": atr(high, low, close, 14),
            "sma_ratio_5_20": sma(close, 5) / sma(close, 20) - 1,
            "vol_z": (volume - volume.rolling(20).mean()) / volume.rolling(20).std(ddof=1),
        },
        index=close.index,
    )
    return feats


def lgb_regression_signals(
    df: pd.DataFrame,
    horizon: int = 5,
    train_size: float = 0.6,
    enter_pct: float = 0.75,
    exit_pct: float = 0.25,
    num_boost_round: int = 100,
) -> tuple[pd.Series, pd.Series]:
    """Train LGB on first ``train_size`` fraction, predict on the rest."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError(
            "lgb_regression_signals requires lightgbm. Install with: uv pip install -e '.[ml]'"
        ) from e

    feats = _features(df)
    label = df["close"].shift(-horizon) / df["close"] - 1

    valid_mask = feats.notna().all(axis=1) & label.notna()
    feats_v = feats[valid_mask]
    label_v = label[valid_mask]

    n = len(feats_v)
    split = int(n * train_size)
    if split < 50 or n - split < 30:
        raise ValueError(f"Not enough valid rows: {n} (need split>50 and tail>30)")

    X_train, y_train = feats_v.iloc[:split], label_v.iloc[:split]
    X_test = feats_v.iloc[split:]

    model = lgb.train(
        params={
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 10,
            "verbosity": -1,
            "metric": "mse",
        },
        train_set=lgb.Dataset(X_train, label=y_train),
        num_boost_round=num_boost_round,
    )

    preds = pd.Series(model.predict(X_test), index=X_test.index)
    train_preds = pd.Series(model.predict(X_train), index=X_train.index)
    enter_thr = float(train_preds.quantile(enter_pct))
    exit_thr = float(train_preds.quantile(exit_pct))

    full_pred = pd.Series(np.nan, index=df.index)
    full_pred.loc[preds.index] = preds.values

    bull = full_pred > enter_thr
    bear = full_pred < exit_thr
    entries = bull & ~bull.shift(1, fill_value=False)
    exits = bear & ~bear.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
