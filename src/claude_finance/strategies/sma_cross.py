from __future__ import annotations

import pandas as pd


def sma_cross_signals(
    df: pd.DataFrame,
    fast: int = 10,
    slow: int = 30,
    price_col: str = "close",
) -> tuple[pd.Series, pd.Series]:
    """Generate entry/exit boolean signals from an SMA crossover.

    Pure function — no engine coupling. Both vectorbt and backtrader adapters
    consume the same (entries, exits) pair.

    Returns:
        entries: True on the bar where fast SMA crosses ABOVE slow SMA
        exits:   True on the bar where fast SMA crosses BELOW slow SMA
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")

    price = df[price_col]
    fast_ma = price.rolling(fast, min_periods=fast).mean()
    slow_ma = price.rolling(slow, min_periods=slow).mean()

    above = fast_ma > slow_ma
    prev_above = above.shift(1, fill_value=False)

    entries = above & ~prev_above
    exits = ~above & prev_above

    return entries.fillna(False), exits.fillna(False)
