"""End-to-end SMA crossover backtest using backtrader.

Run:  python examples/backtest_backtrader.py            # akshare live data
      python examples/backtest_backtrader.py --offline   # synthetic data, no network
"""

from __future__ import annotations

import argparse

from claude_finance.backtest.run_backtrader import run_backtrader
from claude_finance.data import load_akshare_daily, load_synthetic_daily

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

    result = run_backtrader(df, fast=FAST, slow=SLOW)

    print("\n--- backtrader stats ---")
    print(f"Final value   {result['final_value']:>15,.2f}")
    print(f"Total return  {result['total_return_pct']:>14.2f} %")
    print(f"Fast / Slow   {result['fast']} / {result['slow']}")


if __name__ == "__main__":
    main()
