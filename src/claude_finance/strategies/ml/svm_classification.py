"""SVM binary classification: predict P(N-bar forward return > 0)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from claude_finance.strategies.ml.lgb_regression import _features


def svm_classification_signals(
    df: pd.DataFrame,
    horizon: int = 5,
    train_size: float = 0.6,
    prob_enter: float = 0.55,
    prob_exit: float = 0.45,
) -> tuple[pd.Series, pd.Series]:
    """Predict direction of ``horizon``-bar return; trade when P(up) > prob_enter."""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as e:
        raise RuntimeError(
            "svm_classification_signals requires scikit-learn. Install with: uv pip install -e '.[ml]'"
        ) from e

    feats = _features(df)
    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1
    label = (fwd_ret > 0).astype(int)

    valid = feats.notna().all(axis=1) & fwd_ret.notna()
    feats_v = feats[valid]
    label_v = label[valid]

    n = len(feats_v)
    split = int(n * train_size)
    if split < 50 or n - split < 30:
        raise ValueError(f"Not enough valid rows: {n}")

    scaler = StandardScaler().fit(feats_v.iloc[:split])
    X_train = scaler.transform(feats_v.iloc[:split])
    X_test = scaler.transform(feats_v.iloc[split:])

    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, label_v.iloc[:split])
    probs_up = pd.Series(model.predict_proba(X_test)[:, 1], index=feats_v.index[split:])

    full = pd.Series(np.nan, index=df.index)
    full.loc[probs_up.index] = probs_up.values

    bull = full > prob_enter
    bear = full < prob_exit
    entries = bull & ~bull.shift(1, fill_value=False)
    exits = bear & ~bear.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
