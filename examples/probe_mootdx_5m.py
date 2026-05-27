"""Phase 0 probe — 实测 mootdx 5m K 线 API.

目标:
1. 单次 call 最大返回多少 bar (offset 上限)
2. 历史最早能到哪一年
3. paginate via start= offset 是否支持
4. mootdx frequency code 是否 0=5m

打印结果到 stdout. 不写任何 parquet/csv.

频率代码 (mootdx/tdx 标准):
  0 = 5m
  1 = 15m
  2 = 30m
  3 = 1h
  4 = day
  5 = week
  6 = month
  7 = year
  8 = 1m
  9 = day (alt)
"""
from __future__ import annotations

import os
import time

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import pandas as pd  # noqa: E402
from mootdx.quotes import Quotes  # noqa: E402

# 大盘股 600519 (茅台), market=1 (sh)
TEST_CODE = "600519"
TEST_MARKET = 1
TDX_SERVERS = [
    ("180.153.18.170", 7709),
    ("123.125.108.14", 7709),
    ("218.6.170.47", 7709),
    ("60.12.136.250", 7709),
    ("115.238.56.198", 7709),
    ("115.238.90.165", 7709),
]


def make_client():
    """Try servers until one works."""
    for srv in TDX_SERVERS:
        try:
            c = Quotes.factory(market="std", server=srv, timeout=15)
            # smoke test
            df = c.bars(symbol=TEST_CODE, frequency=4, start=0, offset=10)
            if df is not None and len(df) > 0:
                print(f"[probe] connected to {srv}")
                return c, srv
        except Exception as e:
            print(f"[probe] {srv} fail: {type(e).__name__}: {e}")
            continue
    raise RuntimeError("no usable TDX server")


def probe_single_call(client):
    """Probe single-call: try common offset sizes."""
    print("\n--- A. 单次 call offset 上限 ---")
    for off in [800, 1000, 2000, 5000, 8000, 10000]:
        try:
            t0 = time.time()
            df = client.bars(symbol=TEST_CODE, frequency=0, start=0, offset=off)
            dt = time.time() - t0
            if df is None:
                print(f"  offset={off:>5d}  None")
                continue
            n = len(df)
            first_col = "datetime" if "datetime" in df.columns else df.columns[0]
            print(f"  offset={off:>5d}  returned={n:>5d}  wall={dt:.2f}s  "
                  f"first={df[first_col].iloc[0]}  last={df[first_col].iloc[-1]}")
        except Exception as e:
            print(f"  offset={off:>5d}  ERROR: {type(e).__name__}: {e}")
            break


def probe_paginate(client):
    """Test paginate via start= offset. Goal: walk back via start increments."""
    print("\n--- B. paginate (start= 偏移) ---")
    df0 = client.bars(symbol=TEST_CODE, frequency=0, start=0, offset=800)
    if df0 is None or len(df0) == 0:
        print("  [skip] start=0 已 empty")
        return
    print(f"  start=0     offset=800  n={len(df0):>5d}  "
          f"first={df0['datetime'].iloc[0]}  last={df0['datetime'].iloc[-1]}")
    df1 = client.bars(symbol=TEST_CODE, frequency=0, start=800, offset=800)
    if df1 is None or len(df1) == 0:
        print("  [paginate fail] start=800 empty")
        return
    print(f"  start=800   offset=800  n={len(df1):>5d}  "
          f"first={df1['datetime'].iloc[0]}  last={df1['datetime'].iloc[-1]}")
    if "datetime" in df0.columns and "datetime" in df1.columns:
        s0 = set(df0["datetime"].astype(str))
        s1 = set(df1["datetime"].astype(str))
        ovlp = len(s0 & s1)
        print(f"  overlap rows={ovlp}  ({len(s1) - ovlp} new in page2)")


def probe_max_history(client):
    """Try start= very large to find historical depth."""
    print("\n--- C. 历史深度: 一直翻页直到 empty ---")
    page = 0
    start = 0
    offset = 800
    earliest = None
    total = 0
    while page < 30:
        try:
            df = client.bars(
                symbol=TEST_CODE, frequency=0, start=start, offset=offset,
            )
        except Exception as e:
            print(f"  page={page} ERROR: {type(e).__name__}: {e}")
            break
        if df is None or len(df) == 0:
            print(f"  page={page} start={start} → empty (end of history)")
            break
        first_dt = df["datetime"].iloc[0] if "datetime" in df.columns else None
        last_dt = df["datetime"].iloc[-1] if "datetime" in df.columns else None
        total += len(df)
        earliest = first_dt
        if page % 5 == 0 or page < 3:
            print(f"  page={page:>2d} start={start:>5d} n={len(df):>4d} "
                  f"first={first_dt} last={last_dt} cum={total}")
        if len(df) < offset:
            print(f"  page={page} short read ({len(df)} < {offset}) → 已到历史尽头")
            break
        start += offset
        page += 1
        time.sleep(0.1)
    print(f"\n  >>> 历史最早 bar: {earliest}")
    print(f"  >>> 累计 5m bar 数: {total}")
    if earliest is not None:
        try:
            earliest_dt = pd.to_datetime(earliest)
            yr_span = (pd.Timestamp.now() - earliest_dt).days / 365.25
            print(f"  >>> 跨度约 {yr_span:.2f} 年")
            est_yr = total / 12096
            print(f"  >>> 按 bar count 估约 {est_yr:.2f} 年 (vs 跨度 {yr_span:.2f})")
        except Exception:
            pass


def main():
    import mootdx
    print(f"=== Phase 0 probe — mootdx 5m for {TEST_CODE} (茅台) ===")
    print(f"mootdx version: {getattr(mootdx, '__version__', 'unknown')}")
    client, srv = make_client()
    print(f"server: {srv}")
    probe_single_call(client)
    probe_paginate(client)
    probe_max_history(client)
    print("\n=== probe done ===")


if __name__ == "__main__":
    main()
