from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Assert a DataFrame conforms to the OHLCV schema and return it unchanged.

    Why: Both loaders and tests rely on the same column shape; failing early
    surfaces upstream data drift instead of producing wrong PnL silently.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"OHLCV frame index must be DatetimeIndex, got {type(df.index).__name__}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV frame index must be sorted ascending")
    return df
