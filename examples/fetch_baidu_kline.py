"""百度股市通 K 线批量拉取 (全 A 股, 带 MA5/10/20).

输入: data_cache/universe.csv (code, name, market)
输出: data_cache/baidu_kline.parquet (long format: code,date,open,close,high,low,vol,amount,ma5,ma10,ma20)
增量: 已存在的 parquet 会按 code 跳过；checkpoint 每 200 只写一次.

并发: ThreadPoolExecutor MAX_WORKERS = 8 (Baidu 单 IP ~10 req/s 实测可接受).

用法:
    python examples/fetch_baidu_kline.py            # 全量
    python examples/fetch_baidu_kline.py --limit 30 # PoC
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

OUT_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "baidu_kline.parquet"
CHECKPOINT = Path(__file__).resolve().parent.parent / "data_cache" / "baidu_kline_partial.parquet"
UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "universe.csv"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0"
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/vnd.finance-web.v1+json",
    "Origin": "https://gushitong.baidu.com",
    "Referer": "https://gushitong.baidu.com/",
}
MAX_WORKERS = 8
CHECKPOINT_EVERY = 200


def fetch_one(code: str, retries: int = 3) -> pd.DataFrame | None:
    url = "https://finance.pae.baidu.com/selfselect/getstockquotation"
    params = {
        "all": "1", "isIndex": "false", "isBk": "false", "isBlock": "false",
        "isFutures": "false", "isStock": "true", "newFormat": "1",
        "group": "quotation_kline_ab", "finClientType": "pc",
        "code": code, "start_time": "", "ktype": "1",
    }
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                last_err = RuntimeError(f"HTTP {r.status_code}")
                continue
            d = r.json()
            md = (d.get("Result") or {}).get("newMarketData") or {}
            keys = md.get("keys") or []
            rows = (md.get("marketData") or "").split(";")
            if not keys or not rows or not rows[0]:
                return None
            df = pd.DataFrame([r.split(",") for r in rows if r], columns=keys)
            df["code"] = code
            return df
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    return None


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "time": "date",
        "ma5avgprice": "ma5",
        "ma10avgprice": "ma10",
        "ma20avgprice": "ma20",
        "volume": "vol",
    }
    keep = ["code", "date", "open", "close", "high", "low", "vol", "amount",
            "ma5", "ma10", "ma20", "turnoverratio"]
    df = df.rename(columns=rename_map)
    for c in keep:
        if c not in df.columns:
            df[c] = None
    df = df[keep].copy()
    for c in ["open", "close", "high", "low", "vol", "amount", "ma5", "ma10", "ma20",
               "turnoverratio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0=full, >0=PoC limit")
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] 读 universe: {UNIVERSE_PATH}")
    uni = pd.read_csv(UNIVERSE_PATH, dtype={"code": str})
    uni["code"] = uni["code"].astype(str).str.zfill(6)
    if args.limit:
        uni = uni.head(args.limit).copy()
    print(f"  待拉: {len(uni)} 只")

    parts: list[pd.DataFrame] = []
    done_codes: set[str] = set()
    if CHECKPOINT.exists():
        existing = pd.read_parquet(CHECKPOINT)
        parts.append(existing)
        done_codes = set(existing["code"].astype(str).str.zfill(6).unique())
        print(f"  checkpoint 恢复 {len(done_codes)} 只")

    todo = [c for c in uni["code"].tolist() if c not in done_codes]
    print(f"\n[2/4] 并发 {MAX_WORKERS} 路拉 {len(todo)} 只 ...")
    t0 = time.time()
    done = 0
    fail = 0
    last_log = time.time()
    new_parts: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, c): c for c in todo}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                df = fut.result()
            except Exception:
                df = None
            done += 1
            if df is None or df.empty:
                fail += 1
            else:
                new_parts.append(normalize(df))
            if done % 50 == 0 or time.time() - last_log > 15:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{len(todo)}] ok={done - fail} fail={fail} "
                    f"rate={rate:.1f}/s eta={eta/60:.1f}min"
                )
                last_log = time.time()
            if done % CHECKPOINT_EVERY == 0 and new_parts:
                combined = pd.concat(parts + new_parts, ignore_index=True)
                combined.to_parquet(CHECKPOINT, index=False)

    elapsed = time.time() - t0
    print(f"\n  done ({elapsed/60:.1f}min) ok={done - fail} fail={fail}")

    print("\n[3/4] 合并 + 排序 + 写 parquet ...")
    all_parts = parts + new_parts
    big = pd.concat(all_parts, ignore_index=True)
    big["code"] = big["code"].astype(str).str.zfill(6)
    big = big.sort_values(["code", "date"]).reset_index(drop=True)
    big = big.dropna(subset=["date", "close"])
    print(
        f"  total rows: {len(big):,}  unique codes: {big['code'].nunique()}  "
        f"date range: {big['date'].min().date()} → {big['date'].max().date()}"
    )

    big.to_parquet(OUT_PATH, index=False)
    print(f"\n[4/4] saved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    if CHECKPOINT.exists() and not args.limit:
        CHECKPOINT.unlink()
        print("  cleanup checkpoint")


if __name__ == "__main__":
    main()
