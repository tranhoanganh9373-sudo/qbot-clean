"""Tabular Q-learning on a discretised RSI state.

Refactor of qbot's pytrader/strategies/q-learning.py — original used
intraday 5-min state buckets we don't have OHLCV for. This single-asset
version discretises the RSI (0–100) into 10 bins as the state, with the
same {Buy, Sell, Wait} action set. Reward = next-bar return signed by action.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from claude_finance.indicators import rsi

_ACTIONS = ("B", "S", "W")
_N_BINS = 10


def _state(rsi_value: float) -> int:
    if np.isnan(rsi_value):
        return -1
    bin_idx = int(rsi_value // (100 / _N_BINS))
    return min(max(bin_idx, 0), _N_BINS - 1)


def _train_q_table(
    rsi_arr: np.ndarray,
    ret_arr: np.ndarray,
    episodes: int = 200,
    lr: float = 0.05,
    gamma: float = 0.9,
    epsilon: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = np.zeros((_N_BINS, len(_ACTIONS)))
    n = len(rsi_arr) - 1

    for _ in range(episodes):
        for t in range(n):
            s = _state(rsi_arr[t])
            s_next = _state(rsi_arr[t + 1])
            if s < 0 or s_next < 0:
                continue

            if rng.uniform() < epsilon:
                a = rng.integers(0, len(_ACTIONS))
            else:
                a = int(q[s].argmax())

            r = ret_arr[t + 1] if _ACTIONS[a] == "B" else (-ret_arr[t + 1] if _ACTIONS[a] == "S" else 0.0)
            q[s, a] += lr * (r + gamma * q[s_next].max() - q[s, a])
    return q


def q_learning_signals(
    df: pd.DataFrame,
    train_size: float = 0.6,
    rsi_n: int = 14,
    episodes: int = 200,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    close = df["close"].astype(float)
    rsi_s = rsi(close, rsi_n)
    ret = close.pct_change().fillna(0)

    n = len(close)
    split = int(n * train_size)

    q = _train_q_table(
        rsi_s.iloc[:split].to_numpy(),
        ret.iloc[:split].to_numpy(),
        episodes=episodes,
        seed=seed,
    )

    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    rsi_test = rsi_s.iloc[split:].to_numpy()
    last_action: str | None = None
    for i, r in enumerate(rsi_test):
        s = _state(r)
        if s < 0:
            continue
        a = _ACTIONS[int(q[s].argmax())]
        bar_idx = split + i
        if a == "B" and last_action != "B":
            entries.iloc[bar_idx] = True
            last_action = "B"
        elif a == "S" and last_action != "S":
            exits.iloc[bar_idx] = True
            last_action = "S"
    return entries, exits
