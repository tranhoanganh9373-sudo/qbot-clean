"""P4 demo — runs every ML strategy + Monte Carlo VaR on synthetic data.

Run:  python examples/p4_ml_demo.py

Requires the [ml] extra: uv pip install -e ".[ml]"
"""

from __future__ import annotations

from claude_finance.backtest import run_vectorbt
from claude_finance.data import load_synthetic_daily
from claude_finance.risk import annualized_volatility, cagr, value_at_risk
from claude_finance.strategies.ml import (
    lgb_regression_signals,
    lstm_signals,
    q_learning_signals,
    svm_classification_signals,
)


def main() -> None:
    df = load_synthetic_daily(n=1000, seed=42)
    print(f"Synthetic data: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  CAGR (in-sample)       : {cagr(df['close']) * 100:6.2f}%")
    print(f"  Annualised vol         : {annualized_volatility(df['close']) * 100:6.2f}%")
    print(f"  20-day 95% VaR         : {value_at_risk(df['close'], 20, 0.95) * 100:6.2f}%")
    print()

    strategies = [
        ("lgb_regression", lgb_regression_signals),
        ("svm_classification", svm_classification_signals),
        ("q_learning", q_learning_signals),
        ("lstm (10 epochs)", lambda d: lstm_signals(d, epochs=10)),
    ]
    print(f"{'strategy':<22} {'trades':>7} {'return%':>10}")
    print("-" * 42)
    for name, fn in strategies:
        entries, exits = fn(df)
        n_e = int(entries.sum())
        if n_e == 0:
            print(f"{name:<22} {0:>7} {'  (no trade)':>10}")
            continue
        pf = run_vectorbt(df, entries, exits)
        ret = pf.total_return() * 100
        print(f"{name:<22} {n_e:>7} {ret:>10.2f}")


if __name__ == "__main__":
    main()
