"""Multi-strategy fusion decision engine (ported from qbot_decision.analyze).

Weighted vote over BIAS / MACD / KDJ / RSI / BOLL produces BUY / SELL / HOLD,
with ATR-based stop/target levels.
"""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import atr, bias, boll, kdj, macd, rsi, sma

# Qbot default weights (LSTM removed; remaining normalised to sum to 1.0)
_RAW_WEIGHTS = {"BIAS": 0.10, "KDJ": 0.20, "RSI": 0.15, "BOLL": 0.25, "MACD": 0.20}
_total = sum(_RAW_WEIGHTS.values())
ACTIVE_WEIGHTS = {k: v / _total for k, v in _RAW_WEIGHTS.items()}

DECISION_THRESHOLD = 0.50
# Float tolerance: prevents 0.111+0.222+0.167 == 0.4999999... being mis-classified.
_EPS = 1e-9


def _vote_bias(close: pd.Series) -> int:
    b1 = bias(close, 5).iloc[-1]
    b2 = bias(close, 10).iloc[-1]
    b3 = bias(close, 20).iloc[-1]
    if b1 > b2 > b3:
        return +1
    if b1 < b2 < b3:
        return -1
    return 0


def _vote_macd(close: pd.Series) -> int:
    dif, dea, hist = macd(close)
    cross_up = dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2]
    cross_dn = dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2]
    if cross_up or (dif.iloc[-1] > dea.iloc[-1] and hist.iloc[-1] > hist.iloc[-2]):
        return +1
    if cross_dn or (dif.iloc[-1] < dea.iloc[-1] and hist.iloc[-1] < hist.iloc[-2]):
        return -1
    return 0


def _vote_kdj(high: pd.Series, low: pd.Series, close: pd.Series) -> int:
    k, d, j = kdj(high, low, close)
    if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] < 80:
        return +1
    if k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2] and k.iloc[-1] > 20:
        return -1
    if k.iloc[-1] < 20 and j.iloc[-1] < 0:
        return +1
    if k.iloc[-1] > 80 and j.iloc[-1] > 100:
        return -1
    return 0


def _vote_rsi(close: pd.Series) -> int:
    r = rsi(close).iloc[-1]
    if r < 30:
        return +1
    if r > 70:
        return -1
    return 0


def _vote_boll(close: pd.Series) -> int:
    upper, _mid, lower = boll(close)
    px = close.iloc[-1]
    if px < lower.iloc[-1]:
        return +1
    if px > upper.iloc[-1]:
        return -1
    return 0


def analyze_one(df: pd.DataFrame, name: str) -> dict:
    """Run multi-strategy fusion on a single OHLCV frame; return a decision dict.

    Expects at least ~90 bars of daily OHLCV so all indicators are stable.
    Returns the same dict shape as the original qbot_decision.analyze.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    votes = {
        "BIAS": _vote_bias(close),
        "MACD": _vote_macd(close),
        "KDJ": _vote_kdj(high, low, close),
        "RSI": _vote_rsi(close),
        "BOLL": _vote_boll(close),
    }
    buy_score = sum(ACTIVE_WEIGHTS[s] for s, v in votes.items() if v == +1)
    sell_score = sum(ACTIVE_WEIGHTS[s] for s, v in votes.items() if v == -1)

    if buy_score + _EPS >= DECISION_THRESHOLD and buy_score > sell_score:
        signal = "BUY"
    elif sell_score + _EPS >= DECISION_THRESHOLD and sell_score > buy_score:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Cached indicator snapshots for the report
    ma5 = sma(close, 5).iloc[-1]
    ma10 = sma(close, 10).iloc[-1]
    ma20 = sma(close, 20).iloc[-1]
    ma60 = sma(close, 60).iloc[-1]
    rsi14 = rsi(close).iloc[-1]
    dif, dea, hist = macd(close)
    k, d, j = kdj(high, low, close)
    upper, mid, lower = boll(close)
    a = float(atr(high, low, close).iloc[-1])
    px = close.iloc[-1]
    px_prev = close.iloc[-2]

    if signal == "BUY":
        entry = [round(min(px, float(ma5)) * 0.995, 2), round(px * 1.005, 2)]
        stop = round(px - 1.8 * a, 2)
        t1, t2 = round(px + 2.0 * a, 2), round(px + 3.5 * a, 2)
    elif signal == "SELL":
        entry = [round(px * 0.995, 2), round(px * 1.005, 2)]
        stop = round(px + 1.8 * a, 2)
        t1, t2 = round(px - 2.0 * a, 2), round(px - 3.5 * a, 2)
    else:
        entry, stop, t1, t2 = None, None, None, None

    trend = "上升" if px > ma60 else "下降"
    above60 = (px / ma60 - 1) * 100

    base_pos = 0
    if signal == "BUY":
        base_pos = 15 if buy_score >= 0.65 else 10
        if trend == "上升":
            base_pos += 5

    return {
        "name": name,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "price": round(px, 2),
        "change_pct": round((px / px_prev - 1) * 100, 2),
        "signal": signal,
        "buy_score": round(buy_score, 3),
        "sell_score": round(sell_score, 3),
        "votes": votes,
        "trend_ma60": trend,
        "above_ma60_pct": round(above60, 2),
        "indicators": {
            "MA5": round(float(ma5), 2),
            "MA10": round(float(ma10), 2),
            "MA20": round(float(ma20), 2),
            "MA60": round(float(ma60), 2),
            "RSI14": round(float(rsi14), 2),
            "K": round(float(k.iloc[-1]), 2),
            "D": round(float(d.iloc[-1]), 2),
            "J": round(float(j.iloc[-1]), 2),
            "DIF": round(float(dif.iloc[-1]), 3),
            "DEA": round(float(dea.iloc[-1]), 3),
            "MACD_hist": round(float(hist.iloc[-1]), 3),
            "BOLL_upper": round(float(upper.iloc[-1]), 2),
            "BOLL_mid": round(float(mid.iloc[-1]), 2),
            "BOLL_lower": round(float(lower.iloc[-1]), 2),
            "ATR14": round(a, 2),
        },
        "entry_range": entry,
        "stop_loss": stop,
        "target_1": t1,
        "target_2": t2,
        "suggested_position_pct": base_pos,
    }
