"""Aroon Up/Down crossover trend strategy (single-asset)."""
from __future__ import annotations

import pandas as pd

from claude_finance.indicators import aroon


def aroon_signals(
    df: pd.DataFrame,
    n: int = 25,
    min_energy: float = 50.0,
) -> tuple[pd.Series, pd.Series]:
    up, down = aroon(df["high"], df["low"], n)
    energy = up + down  # noise filter — both indicators must be active
    above = (up > down) & (energy >= min_energy)
    entries = above & ~above.shift(1, fill_value=False)
    exits = ~above & above.shift(1, fill_value=False)
    return entries.fillna(False), exits.fillna(False)
