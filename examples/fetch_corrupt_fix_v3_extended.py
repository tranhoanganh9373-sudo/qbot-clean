"""One-shot extended corrupt-fix fetch (2014-2026) for baidu_kline.parquet v3 swap.

复用 examples/fetch_corrupt_fix_akshare.py 的腾讯 hfq 拉取逻辑, 但:
  - 扫描 corrupt 时不限年份 (覆盖 2017-2024 所有 close<0 或 close<0.5 行)
  - 拉取年范围 2014-2026 (确保 2021+ 的 corruption 也被覆盖)
  - 输出独立文件: data_cache/baidu_kline_corrupt_fix_v3.parquet

不修改任何 strategy/sidecar 脚本, 不动主 parquet (本脚本只写 corrupt_fix_v3.parquet).
后续由调用者 merge 进 baidu_kline.parquet.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
OUT_PARQUET = ROOT / "data_cache" / "baidu_kline_corrupt_fix_v3.parquet"
OUT_LIST = ROOT / "data_cache" / "corrupt_codes_v3.txt"

TENCENT_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Referer": "https://gu.qq.com/",
}

START_YEAR = 2014
END_YEAR = 2026


def market_prefix(code: str) -> str:
    if code.startswith("6") or code.startswith("9"):
        return "sh"
    return "sz"


def scan_corrupt_codes() -> list[str]:
    """全表扫: close<0.5 (含 close<0) → 全部 corrupt 列入."""
    kl = pd.read_parquet(KLINE_PATH, columns=["code", "close"])
    kl["code"] = kl["code"].astype(str).str.zfill(6)
    mask = (kl["close"] < 0.5)
    return sorted(kl.loc[mask, "code"].unique())


def _fetch_year_chunk(
    sym: str, year: int, retries: int = 3, retry_delay: float = 3.0,
) -> Optional[list[list[str]]]:
    last_err: Optional[BaseException] = None
    param = f"{sym},day,{year}-01-01,{year}-12-31,300,hfq"
    for attempt in range(retries):
        try:
            r = requests.get(
                TENCENT_KLINE,
                params={"param": param},
                headers=HEADERS,
                timeout=20,
            )
            r.raise_for_status()
            payload = r.json()
            sec = (payload.get("data") or {}).get(sym) or {}
            rows = sec.get("hfqday") or sec.get("day") or []
            return rows
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(retry_delay)
    print(f"  ! {sym} year {year} failed: {type(last_err).__name__}: {last_err}", flush=True)
    return None


def fetch_one_hfq(code: str) -> Optional[pd.DataFrame]:
    sym = f"{market_prefix(code)}{code}"
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
    df = pd.DataFrame(
        norm_rows,
        columns=["date", "open", "close", "high", "low", "vol"],
    )
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
    return df[[
        "code", "date", "open", "close", "high", "low",
        "vol", "amount", "ma5", "ma10", "ma20", "turnoverratio",
    ]]


def validate(df: Optional[pd.DataFrame]) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, "empty"
    closes = df["close"].dropna()
    if closes.empty:
        return False, "all close NaN"
    if (closes < 0).any():
        return False, f"still has neg close: min={closes.min()}"
    if (closes < 0.5).any():
        bad_dates = df.loc[df["close"] < 0.5, "date"].dt.strftime("%Y-%m-%d").tolist()
        return False, f"close<0.5 still present at {bad_dates[:3]}"
    return True, "ok"


def main() -> None:
    t0 = time.time()
    print(f"[v3-fix] scanning corrupt codes from {KLINE_PATH.name} ...", flush=True)
    codes = scan_corrupt_codes()
    print(f"[v3-fix] found {len(codes)} corrupt codes: {codes}", flush=True)
    OUT_LIST.write_text("\n".join(codes) + "\n", encoding="utf-8")

    frames: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []
    for i, code in enumerate(codes, 1):
        elapsed = time.time() - t0
        print(f"[{i}/{len(codes)}] {code} (t+{elapsed:.0f}s) ...", end=" ", flush=True)
        df = fetch_one_hfq(code)
        ok, msg = validate(df)
        if not ok:
            failed.append((code, msg))
            print(f"FAIL ({msg})", flush=True)
            continue
        assert df is not None
        frames.append(df)
        n = len(df)
        cmin = float(df["close"].min())
        cmax = float(df["close"].max())
        print(f"OK {n} rows close=[{cmin:.2f},{cmax:.2f}]", flush=True)
        time.sleep(0.15)

    if not frames:
        print("[v3-fix] ABORT: no successful fetches", flush=True)
        return

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["code", "date"]).reset_index(drop=True)
    assert (all_df["close"] > 0).all()
    assert (all_df["close"] >= 0.5).all()

    all_df.to_parquet(OUT_PARQUET, index=False)
    size_mb = OUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"\n[v3-fix] DONE in {time.time()-t0:.0f}s", flush=True)
    print(f"[v3-fix] wrote {OUT_PARQUET} ({size_mb:.2f} MB)", flush=True)
    print(f"[v3-fix] rows={len(all_df)}, codes ok={len(frames)}/{len(codes)}", flush=True)
    if failed:
        print(f"[v3-fix] FAILED codes ({len(failed)}):", flush=True)
        for code, msg in failed:
            print(f"    {code}: {msg}", flush=True)


if __name__ == "__main__":
    main()
