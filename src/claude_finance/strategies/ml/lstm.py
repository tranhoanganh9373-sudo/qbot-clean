"""LSTM next-bar return prediction → buy/sell on predicted direction.

Refactor of qbot/qbot/strategies/lstm_strategy_bt.py:
- Uses PyTorch (qbot used keras; torch installs cleanly on macOS arm64)
- Fixes the train/test leakage in the original (scaler was fit on full data)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lstm_signals(
    df: pd.DataFrame,
    lookback: int = 20,
    hidden: int = 50,
    epochs: int = 20,
    train_size: float = 0.6,
    direction_threshold: float = 0.0,
    price_col: str = "close",
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Predict next-bar log-return with a 1-layer LSTM; trade on its sign.

    direction_threshold: predicted return must exceed this for a BUY signal
    (and be below -threshold for SELL). 0.0 means trade purely on direction.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError(
            "lstm_signals requires PyTorch. Install with: uv pip install -e '.[ml]'"
        ) from e

    torch.manual_seed(seed)
    np.random.seed(seed)

    close = df[price_col].astype(float).to_numpy()
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))

    n = len(close)
    train_end = int(n * train_size)
    if train_end < lookback + 50:
        raise ValueError(f"Not enough bars: need {lookback + 50}, got {train_end}")

    # Scale on TRAIN only — fix the original leakage bug
    train_mean = log_ret[:train_end].mean()
    train_std = log_ret[:train_end].std() + 1e-9
    scaled = (log_ret - train_mean) / train_std

    def _windows(start: int, end: int):
        xs, ys = [], []
        for i in range(start + lookback, end):
            xs.append(scaled[i - lookback : i])
            ys.append(scaled[i])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    x_train, y_train = _windows(0, train_end)
    x_test, _ = _windows(train_end, n)

    class _LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    model = _LSTMNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x_t = torch.tensor(x_train).unsqueeze(-1)
    y_t = torch.tensor(y_train)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        x_test_t = torch.tensor(x_test).unsqueeze(-1)
        preds = model(x_test_t).numpy()

    preds_unscaled = preds * train_std + train_mean

    signal_arr = np.zeros(n)
    if len(preds_unscaled) > 0:
        offset = train_end + lookback
        signal_arr[offset : offset + len(preds_unscaled)] = preds_unscaled

    pred_s = pd.Series(signal_arr, index=df.index)
    bull = pred_s > direction_threshold
    bear = pred_s < -direction_threshold
    entries = bull & ~bull.shift(1, fill_value=False)
    exits = bear & ~bear.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
