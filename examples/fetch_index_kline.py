"""拉取主流指数日 K 线 (sina json_v2 endpoint, 3000 条上限).

数据源: money.finance.sina.com.cn/quotes_service/api/json_v2.php
        /CN_MarketData.getKLineData?symbol={code}&scale=240&datalen=3000

可拉指数: 上证 sh000001 / 沪深 300 sh000300 / 深成 sz399001 / 创业 sz399006

输出: data_cache/index_kline.parquet
  schema: code(str), date(datetime64), open/high/low/close(float64), volume(int64)

Run:  python examples/fetch_index_kline.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data_cache" / "index_kline.parquet"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
INDEXES = [
    ("sh000001", "上证指数"),
    ("sh000300", "沪深300"),
    ("sz399001", "深证成指"),
    ("sz399006", "创业板指"),
]


def fetch_one(code: str) -> pd.DataFrame:
    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    r = requests.get(url, params={"symbol": code, "scale": "240", "datalen": "3000"},
                      headers={"User-Agent": UA, "Referer": "https://finance.sina.com.cn/"},
                      timeout=20)
    if r.status_code != 200 or not r.text.startswith("["):
        raise RuntimeError(f"{code}: HTTP {r.status_code} preview={r.text[:120]}")
    data = r.json()
    df = pd.DataFrame(data)
    df["code"] = code
    df["date"] = pd.to_datetime(df["day"])
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    return df[["code", "date", "open", "high", "low", "close", "volume"]]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("[1/2] 拉 4 大指数日 K 线 ...")
    parts = []
    for code, name in INDEXES:
        try:
            df = fetch_one(code)
            print(f"  {code} {name}: {len(df)} 天 "
                  f"({df['date'].min().date()} → {df['date'].max().date()})")
            parts.append(df)
        except Exception as e:
            print(f"  {code} {name}: FAIL {e}")

    if not parts:
        print("  全失败")
        return

    big = pd.concat(parts, ignore_index=True).sort_values(["code", "date"])
    big.to_parquet(OUT, index=False)
    print(f"\n[2/2] 保存 {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"  共 {big['code'].nunique()} 指数 × {len(big):,} 行")


if __name__ == "__main__":
    main()
