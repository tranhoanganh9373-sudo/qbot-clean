"""补 51 个 CSI500 缺失科创板股票 baidu_kline 数据 (全 688*).

复用 fetch_688_missing_csi300.py 模板 (腾讯 hfq).
输出到 data_cache/baidu_kline_csi500_fix.parquet, 后续由调用者 merge 到主表。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_PARQUET = ROOT / "data_cache" / "baidu_kline_csi500_fix.parquet"

# 51 missing CSI500 codes — 全 688* 科创板
MISSING_CODES = [
    "688295", "688297", "688300", "688301", "688318", "688322", "688331",
    "688336", "688343", "688368", "688375", "688385", "688387", "688390",
    "688400", "688409", "688428", "688433", "688449", "688469", "688502",
    "688519", "688523", "688531", "688536", "688538", "688545", "688548",
    "688568", "688578", "688599", "688608", "688615", "688617", "688627",
    "688629", "688630", "688668", "688676", "688692", "688726", "688728",
    "688766", "688777", "688778", "688785", "688796", "688800", "688807",
    "688809", "689009",
]

TENCENT_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Referer": "https://gu.qq.com/",
}

START_YEAR = 2014
END_YEAR = 2026


def _fetch_year_chunk(sym: str, year: int, retries: int = 3) -> Optional[list[list[str]]]:
    last_err = None
    param = f"{sym},day,{year}-01-01,{year}-12-31,300,hfq"
    for attempt in range(retries):
        try:
            r = requests.get(TENCENT_KLINE, params={"param": param},
                             headers=HEADERS, timeout=20)
            r.raise_for_status()
            payload = r.json()
            sec = (payload.get("data") or {}).get(sym) or {}
            return sec.get("hfqday") or sec.get("day") or []
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(3)
    print(f"  ! {sym} year {year} failed: {type(last_err).__name__}: {last_err}", flush=True)
    return None


def fetch_one_hfq(code: str) -> Optional[pd.DataFrame]:
    # 688/689 全是科创板沪市
    sym = f"sh{code}"
    all_rows: list[list[str]] = []
    for year in range(START_YEAR, END_YEAR + 1):
        chunk = _fetch_year_chunk(sym, year)
        if chunk is None:
            return None
        all_rows.extend(chunk)
        time.sleep(0.1)
    if not all_rows:
        return None
    norm_rows = [r[:6] for r in all_rows]
    df = pd.DataFrame(norm_rows, columns=["date", "open", "close", "high", "low", "vol"])
    df["code"] = code
    for c in ("open", "close", "high", "low"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["vol"] = (pd.to_numeric(df["vol"], errors="coerce") * 100).astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["code", "date"]).sort_values("date").reset_index(drop=True)
    df["amount"] = float("nan")
    df["turnoverratio"] = float("nan")
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    return df[["code", "date", "open", "close", "high", "low",
               "vol", "amount", "ma5", "ma10", "ma20", "turnoverratio"]]


def main() -> None:
    t0 = time.time()
    frames: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []
    for i, code in enumerate(MISSING_CODES, 1):
        elapsed = time.time() - t0
        print(f"[{i}/{len(MISSING_CODES)}] {code} (t+{elapsed:.0f}s) ...", end=" ", flush=True)
        df = fetch_one_hfq(code)
        if df is None or df.empty:
            failed.append((code, "empty"))
            print("FAIL (empty)", flush=True)
            continue
        if (df["close"].dropna() <= 0).any():
            failed.append((code, "neg close"))
            print("FAIL (neg close)", flush=True)
            continue
        frames.append(df)
        n = len(df)
        cmin = float(df["close"].min())
        cmax = float(df["close"].max())
        first_d = df["date"].min().date()
        last_d = df["date"].max().date()
        print(f"OK {n} rows {first_d}~{last_d} close=[{cmin:.2f},{cmax:.2f}]", flush=True)
        time.sleep(0.15)

    if not frames:
        print("ABORT: no successful fetches", flush=True)
        return
    all_df = pd.concat(frames, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)

    # Atomic write
    tmp = OUT_PARQUET.with_suffix(".parquet.tmp")
    all_df.to_parquet(tmp, index=False)
    tmp.replace(OUT_PARQUET)

    size_kb = OUT_PARQUET.stat().st_size / 1024
    print(f"\n[done] {len(frames)}/{len(MISSING_CODES)} OK, {len(failed)} fail")
    print(f"  output: {OUT_PARQUET} ({size_kb:.1f} KB, {len(all_df)} rows)")
    print(f"  wall: {time.time() - t0:.1f}s")
    if failed:
        print(f"  failed: {failed}")


if __name__ == "__main__":
    main()
