"""End-to-end SMA crossover backtest using vectorbt.

Run:  python examples/backtest_vectorbt.py            # akshare live data
      python examples/backtest_vectorbt.py --offline   # synthetic data, no network
"""

from __future__ import annotations

import argparse

from claude_finance.backtest import run_vectorbt
from claude_finance.data import load_akshare_daily, load_synthetic_daily
from claude_finance.strategies import sma_cross_signals

SYMBOL = "600519"  # 贵州茅台
START = "20230101"
END = "20241231"
FAST, SLOW = 10, 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="use synthetic data instead of akshare")
    args = parser.parse_args()

    if args.offline:
        df = load_synthetic_daily(n=500, seed=42)
        print(f"[offline] synthetic {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    else:
        df = load_akshare_daily(SYMBOL, START, END, adjust="qfq")
        print(f"Loaded {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")

    entries, exits = sma_cross_signals(df, fast=FAST, slow=SLOW)
    print(f"Signals: {int(entries.sum())} entries, {int(exits.sum())} exits")

    pf = run_vectorbt(df, entries, exits)
    stats = pf.stats()

    print("\n--- vectorbt stats ---")
    for key in (
        "Start",
        "End",
        "Total Return [%]",
        "Sharpe Ratio",
        "Max Drawdown [%]",
        "Win Rate [%]",
        "Total Trades",
    ):
        if key in stats.index:
            print(f"{key:<22} {stats[key]}")


if __name__ == "__main__":
    main()
