"""Fetch 3-year daily history for HS300+CSI500+CSI1000 (~1800 stocks).

Uses sina endpoint (ak.stock_zh_a_daily) which is sandbox-reachable.
Output: data_cache/long_history.parquet (~ 130MB).

Estimated time: 6-12 minutes at 8 parallel workers.

Run:  python examples/fetch_long_history.py
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import pandas as pd

OUT_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "long_history.parquet"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "long_history_partial.parquet"
START_DATE = "2023-04-01"  # 3 years back from latest
MAX_WORKERS = 4  # process pool — mini_racer not thread-safe; 4 procs is the sweet spot
CHECKPOINT_EVERY = 200


def _exchange_prefix(exchange: str) -> str:
    if "上海" in exchange: return "sh"
    if "深圳" in exchange: return "sz"
    if "北京" in exchange: return "bj"
    raise ValueError(exchange)


def _bare_to_sina(bare: str) -> str:
    """6-digit code -> sina prefix; fallback heuristic."""
    if bare.startswith("6"): return "sh" + bare
    if bare.startswith(("0", "3", "1", "2")): return "sz" + bare
    if bare.startswith(("8", "9", "4")): return "bj" + bare
    return bare


def _fetch_one_index(idx_code: str, label: str, retries: int = 3) -> set[str]:
    last_err = None
    # try csindex (with exchange info -> exact prefix)
    for attempt in range(retries):
        try:
            df = ak.index_stock_cons_weight_csindex(symbol=idx_code)
            codes = {_exchange_prefix(r["交易所"]) + str(r["成分券代码"]) for _, r in df.iterrows()}
            print(f"  ✓ {label} (csindex): {len(codes)} 只")
            return codes
        except Exception as e:
            last_err = str(e)[:80]
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    print(f"  ✗ {label} csindex 失败 ({last_err}), 切换 sina 备份 ...")
    # fallback: index_stock_cons returns bare codes, infer prefix
    for attempt in range(retries):
        try:
            df = ak.index_stock_cons(symbol=idx_code)
            codes = {_bare_to_sina(str(r["品种代码"])) for _, r in df.iterrows()}
            print(f"  ✓ {label} (fallback): {len(codes)} 只")
            return codes
        except Exception as e:
            last_err = str(e)[:80]
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{label} 全部 endpoint 失败: {last_err}")


def get_universe() -> set[str]:
    """Return {sina-style code} = HS300 ∪ CSI500 ∪ CSI1000."""
    print("[1/3] 拉 HS300 + CSI500 + CSI1000 成分股清单 ...")
    codes: set[str] = set()
    for idx_code, label in [("000300", "HS300"), ("000905", "CSI500"), ("000852", "CSI1000")]:
        codes |= _fetch_one_index(idx_code, label)
    print(f"  累计去重: {len(codes)}")
    return codes


def fetch_one(code: str) -> pd.DataFrame | None:
    try:
        df = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
        if df is None or len(df) == 0:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= START_DATE].copy()
        if len(df) < 100:
            return None
        df["code"] = code
        return df
    except Exception:
        return None


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    universe = sorted(get_universe())
    print(f"  总计 {len(universe)} 只股票")

    # Resume: skip codes already in checkpoint
    parts: list[pd.DataFrame] = []
    done_codes: set[str] = set()
    if CHECKPOINT_PATH.exists():
        existing = pd.read_parquet(CHECKPOINT_PATH)
        parts.append(existing)
        done_codes = set(existing["code"].unique())
        print(f"  从 checkpoint 恢复 {len(done_codes)} 只已拉取的股票")

    todo = [c for c in universe if c not in done_codes]
    print(f"\n[2/3] 进程池 {MAX_WORKERS} 路拉 {START_DATE} 至今, 待拉 {len(todo)} 只 ...")
    t0 = time.time()
    done = 0
    fail = 0
    last_log = time.time()
    new_parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, c): c for c in todo}
        for fut in as_completed(futures):
            try:
                df = fut.result()
            except Exception:
                df = None
            done += 1
            if df is None:
                fail += 1
            else:
                new_parts.append(df)
            if done % 50 == 0 or time.time() - last_log > 15:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{len(todo)}] ok={done - fail} fail={fail} "
                    f"rate={rate:.1f}/s eta={eta / 60:.1f}min"
                )
                last_log = time.time()
            # checkpoint
            if done % CHECKPOINT_EVERY == 0 and new_parts:
                combined = pd.concat(parts + new_parts, ignore_index=True)
                combined.to_parquet(CHECKPOINT_PATH, index=False)
    elapsed = time.time() - t0
    print(f"  done ({elapsed / 60:.1f}min) — ok={done - fail} fail={fail}")

    print("\n[3/3] 合并 + 写 parquet ...")
    all_parts = parts + new_parts
    big = pd.concat(all_parts, ignore_index=True)
    big = big.sort_values(["code", "date"])
    print(
        f"  total rows: {len(big):,}  unique codes: {big['code'].nunique()}  "
        f"date range: {big['date'].min().date()} → {big['date'].max().date()}"
    )
    big.to_parquet(OUT_PATH, index=False)
    print(f"  saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    # cleanup checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


if __name__ == "__main__":
    main()
