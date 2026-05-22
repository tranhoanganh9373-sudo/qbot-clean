"""ML strategy tests — require the [ml] extra.

Skipped automatically when torch/lightgbm/sklearn aren't installed, so the
suite still passes for the minimal install.
"""
from __future__ import annotations

import pandas as pd
import pytest

from claude_finance.data import load_synthetic_daily

# Conditional import — if any ml dep is missing, skip the whole module.
pytest.importorskip("lightgbm")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from claude_finance.strategies.ml import (  # noqa: E402
    lgb_regression_signals,
    lstm_signals,
    q_learning_signals,
    svm_classification_signals,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return load_synthetic_daily(n=800, seed=2026)


@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("lgb_regression", lgb_regression_signals),
        ("svm_classification", svm_classification_signals),
        ("q_learning", q_learning_signals),
        ("lstm", lambda d: lstm_signals(d, epochs=3)),  # keep epochs low for test speed
    ],
)
def test_ml_signals_shape(df: pd.DataFrame, name: str, fn):
    entries, exits = fn(df)
    assert entries.dtype == bool, f"{name}: entries must be bool"
    assert exits.dtype == bool, f"{name}: exits must be bool"
    assert entries.index.equals(df.index), f"{name}: entries index drift"
    assert exits.index.equals(df.index), f"{name}: exits index drift"


def test_lstm_requires_sufficient_history():
    df = load_synthetic_daily(n=50, seed=0)
    with pytest.raises(ValueError, match="Not enough bars"):
        lstm_signals(df, lookback=20, train_size=0.6, epochs=1)
