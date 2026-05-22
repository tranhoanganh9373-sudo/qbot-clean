from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from claude_finance.data.schema import validate_ohlcv

load_dotenv()


def _cache_path(ts_code: str, adj: str) -> Path:
    root = Path(os.getenv("DATA_CACHE_DIR", "./data_cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"tushare_{ts_code.replace('.', '_')}_{adj or 'raw'}.parquet"


def load_tushare_daily(
    ts_code: str,
    start: str,
    end: str,
    adj: str = "qfq",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV via tushare pro.

    ts_code: e.g. "600519.SH"
    start / end: "YYYYMMDD"
    """
    cache = _cache_path(ts_code, adj)
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("TUSHARE_TOKEN not set; copy .env.example to .env and fill it in")

        import tushare as ts

        pro = ts.pro_api(token)
        raw = ts.pro_bar(
            ts_code=ts_code, start_date=start, end_date=end, adj=adj, api=pro
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"tushare returned no rows for {ts_code} {start}..{end}")

        df = raw.rename(columns={"vol": "volume"})[
            ["trade_date", "open", "high", "low", "close", "volume"]
        ]
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.set_index("trade_date").sort_index()
        df.index.name = "date"
        df["volume"] = df["volume"].astype("int64")
        df.to_parquet(cache)

    return validate_ohlcv(df.loc[start:end] if start else df)
