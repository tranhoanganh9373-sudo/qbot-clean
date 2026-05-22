from __future__ import annotations

import pandas as pd


def run_backtrader(
    df: pd.DataFrame,
    fast: int = 10,
    slow: int = 30,
    init_cash: float = 100_000.0,
    commission: float = 0.0003,
) -> dict:
    """Run an SMA crossover strategy via backtrader.

    Returns a summary dict with final value and total return.
    Imported lazily so the package imports without backtrader installed.
    """
    import backtrader as bt

    class _SmaCross(bt.Strategy):
        params = dict(fast=fast, slow=slow)

        def __init__(self):
            self.fast_ma = bt.ind.SMA(self.data.close, period=self.p.fast)
            self.slow_ma = bt.ind.SMA(self.data.close, period=self.p.slow)
            self.cross = bt.ind.CrossOver(self.fast_ma, self.slow_ma)

        def next(self):
            if not self.position and self.cross > 0:
                self.buy()
            elif self.position and self.cross < 0:
                self.close()

    cerebro = bt.Cerebro()
    cerebro.addstrategy(_SmaCross)

    feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(feed)

    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    cerebro.run()
    final = cerebro.broker.getvalue()
    return {
        "final_value": final,
        "total_return_pct": (final / init_cash - 1) * 100,
        "fast": fast,
        "slow": slow,
    }
