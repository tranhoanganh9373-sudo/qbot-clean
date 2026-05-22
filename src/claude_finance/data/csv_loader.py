"""Generic OHLCV CSV loader — pre-loaded multi-symbol market data.

Supports CSVs with at minimum: date, open, high, low, close, volume, code.
Extra columns (amount, outstanding_share, turnover, ...) are tolerated and
dropped. Codes may carry an exchange prefix (sh/sz/bj); use ``strip_prefix=True``
to normalise to bare 6-digit codes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from claude_finance.data.schema import validate_ohlcv

_REQUIRED = ("date", "open", "high", "low", "close", "volume", "code")
_PREFIXES = ("sh", "sz", "bj")


def _strip_prefix(code: str) -> str:
    for p in _PREFIXES:
        if code.startswith(p):
            return code[len(p):]
    return code


def load_market_csv(
    path: str | Path,
    *,
    strip_prefix: bool = False,
    min_bars: int = 90,
) -> dict[str, pd.DataFrame]:
    """Load a multi-symbol OHLCV CSV and return ``{code: DataFrame}``.

    Filters out codes with fewer than ``min_bars`` rows (so downstream indicators
    are stable).
    """
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if strip_prefix:
        df["code"] = df["code"].astype(str).map(_strip_prefix)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])

    out: dict[str, pd.DataFrame] = {}
    for code, g in df.groupby("code", sort=False):
        if len(g) < min_bars:
            continue
        ohlcv = g.set_index("date")[["open", "high", "low", "close", "volume"]].copy()
        ohlcv["volume"] = ohlcv["volume"].astype("int64")
        out[code] = validate_ohlcv(ohlcv)
    return out
