from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from claude_finance.data.schema import validate_ohlcv


def _cache_path(symbol: str, adjust: str) -> Path:
    root = Path(os.getenv("DATA_CACHE_DIR", "./data_cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"akshare_{symbol}_{adjust or 'raw'}.parquet"


def load_akshare_daily(
    symbol: str,
    start: str,
    end: str,
    adjust: str = "qfq",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch A-share daily OHLCV via akshare.

    symbol: e.g. "600519" (no exchange prefix)
    start / end: "YYYYMMDD"
    adjust: "qfq" (前复权) | "hfq" (后复权) | "" (不复权)
    """
    cache = _cache_path(symbol, adjust)
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        import akshare as ak

        raw = ak.stock_zh_a_hist(
            symbol=symbol, period="daily", start_date=start, end_date=end, adjust=adjust
        )
        df = raw.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["volume"] = df["volume"].astype("int64")
        df.to_parquet(cache)

    return validate_ohlcv(df.loc[start:end] if start else df)
