from __future__ import annotations

import pandas as pd


def run_vectorbt(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float = 100_000.0,
    fees: float = 0.0003,
    slippage: float = 0.0001,
    price_col: str = "close",
):
    """Thin wrapper that runs vectorbt.Portfolio.from_signals.

    Imported lazily so the package imports without vectorbt installed.
    """
    import vectorbt as vbt

    return vbt.Portfolio.from_signals(
        close=df[price_col],
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq="1D",
    )
