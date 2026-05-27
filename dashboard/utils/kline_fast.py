"""单股 kline 快路径 — direct path-read 0.4 ms (vs 主表 view 2.6 ms / kline_hive view 175 ms).

Hive 分区路径: data_cache/baidu_kline_hive/code=XXXXXX/*.parquet
benchmark 来源: docs/duckdb_stage2_benchmark.md

用法:
    from dashboard.utils.kline_fast import get_stock_kline
    df = get_stock_kline("SH600519")  # 茅台全 history
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
HIVE_DIR = ROOT / "data_cache" / "baidu_kline_hive"


def get_stock_kline(sym: str) -> pd.DataFrame:
    """0.4 ms single-stock kline via direct Hive partition path-read.

    Args:
        sym: 'SH600519' / 'SZ300347' format,or raw 6-digit code '600519'.

    Returns:
        DataFrame columns: code, date, open, close, high, low, vol,
        amount, ma5, ma10, ma20, turnoverratio. Sorted by date asc.
        Empty DataFrame if symbol not in Hive partition.
    """
    code = sym[2:].zfill(6) if sym[:2] in ("SH", "SZ") else str(sym).zfill(6)
    partition = HIVE_DIR / f"code={code}"
    if not partition.exists():
        return pd.DataFrame()
    files = sorted(partition.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    out = pd.concat(dfs, ignore_index=True)
    out["code"] = code
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def list_available_codes() -> list[str]:
    """所有 Hive 分区中可用的 6-digit code list."""
    if not HIVE_DIR.exists():
        return []
    return sorted(
        d.name.split("=", 1)[1]
        for d in HIVE_DIR.iterdir()
        if d.is_dir() and d.name.startswith("code=")
    )


if __name__ == "__main__":
    import time
    syms = ["SH600519", "SH600547", "SZ300347", "SH688396"]
    for sym in syms:
        t0 = time.perf_counter()
        df = get_stock_kline(sym)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if df.empty:
            print(f"  {sym}: empty ({elapsed_ms:.2f} ms)")
        else:
            print(f"  {sym}: {len(df)} rows / {df['date'].min().date()} ~ "
                  f"{df['date'].max().date()} ({elapsed_ms:.2f} ms)")
