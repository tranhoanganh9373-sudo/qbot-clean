"""Multi-strategy fusion as a signal function for backtesting.

Wraps decision.fusion.analyze_one's vote logic into rolling (entries, exits)
series so both vectorbt and backtrader adapters can backtest it directly.
"""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import bias, boll, kdj, macd, rsi

_WEIGHTS = {"BIAS": 0.10, "KDJ": 0.20, "RSI": 0.15, "BOLL": 0.25, "MACD": 0.20}
_total = sum(_WEIGHTS.values())
_ACTIVE = {k: v / _total for k, v in _WEIGHTS.items()}
_THRESHOLD = 0.50
_EPS = 1e-9


def multi_fusion_signals(
    df: pd.DataFrame,
    threshold: float = _THRESHOLD,
) -> tuple[pd.Series, pd.Series]:
    """Vectorised rolling votes; emits BUY/SELL on threshold cross.

    Returns:
        entries: True where the bar transitions into a BUY state
        exits:   True where the bar transitions into a SELL state (or out of BUY)
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # BIAS vote
    b5, b10, b20 = (
        (close - close.rolling(n).mean()) / close.rolling(n).mean() * 100
        for n in (5, 10, 20)
    )
    v_bias = pd.Series(0, index=close.index)
    v_bias[(b5 > b10) & (b10 > b20)] = +1
    v_bias[(b5 < b10) & (b10 < b20)] = -1

    # MACD vote
    dif, dea, hist = macd(close)
    above = dif > dea
    cross_up = above & ~above.shift(1, fill_value=False)
    cross_dn = ~above & above.shift(1, fill_value=False)
    hist_up = hist > hist.shift(1)
    hist_dn = hist < hist.shift(1)
    v_macd = pd.Series(0, index=close.index)
    v_macd[cross_up | (above & hist_up)] = +1
    v_macd[cross_dn | (~above & hist_dn)] = -1

    # KDJ vote
    k, d, j = kdj(high, low, close)
    k_above = k > d
    k_cross_up = k_above & ~k_above.shift(1, fill_value=False)
    k_cross_dn = ~k_above & k_above.shift(1, fill_value=False)
    v_kdj = pd.Series(0, index=close.index)
    v_kdj[(k_cross_up & (k < 80)) | ((k < 20) & (j < 0))] = +1
    v_kdj[(k_cross_dn & (k > 20)) | ((k > 80) & (j > 100))] = -1

    # RSI vote
    r = rsi(close)
    v_rsi = pd.Series(0, index=close.index)
    v_rsi[r < 30] = +1
    v_rsi[r > 70] = -1

    # BOLL vote
    upper, _mid, lower = boll(close)
    v_boll = pd.Series(0, index=close.index)
    v_boll[close < lower] = +1
    v_boll[close > upper] = -1

    buy_score = (
        (v_bias == +1).astype(float) * _ACTIVE["BIAS"]
        + (v_macd == +1).astype(float) * _ACTIVE["MACD"]
        + (v_kdj == +1).astype(float) * _ACTIVE["KDJ"]
        + (v_rsi == +1).astype(float) * _ACTIVE["RSI"]
        + (v_boll == +1).astype(float) * _ACTIVE["BOLL"]
    )
    sell_score = (
        (v_bias == -1).astype(float) * _ACTIVE["BIAS"]
        + (v_macd == -1).astype(float) * _ACTIVE["MACD"]
        + (v_kdj == -1).astype(float) * _ACTIVE["KDJ"]
        + (v_rsi == -1).astype(float) * _ACTIVE["RSI"]
        + (v_boll == -1).astype(float) * _ACTIVE["BOLL"]
    )

    is_buy = (buy_score + _EPS >= threshold) & (buy_score > sell_score)
    is_sell = (sell_score + _EPS >= threshold) & (sell_score > buy_score)

    entries = is_buy & ~is_buy.shift(1, fill_value=False)
    exits = is_sell & ~is_sell.shift(1, fill_value=False)

    return entries.fillna(False), exits.fillna(False)
