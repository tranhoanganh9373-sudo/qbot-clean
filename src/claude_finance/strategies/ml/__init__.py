"""ML-backed strategies. Requires the [ml] extra (torch, lightgbm, scikit-learn).

Install with:  uv pip install -e ".[ml]"

These are imported lazily by individual modules so the rest of the package
still imports cleanly when the [ml] extra is absent.
"""

from claude_finance.strategies.ml.lgb_regression import lgb_regression_signals
from claude_finance.strategies.ml.lstm import lstm_signals
from claude_finance.strategies.ml.q_learning import q_learning_signals
from claude_finance.strategies.ml.svm_classification import svm_classification_signals

__all__ = [
    "lgb_regression_signals",
    "lstm_signals",
    "q_learning_signals",
    "svm_classification_signals",
]
