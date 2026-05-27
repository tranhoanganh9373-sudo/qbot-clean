"""Fetch 14-year margin trading history for CSI300 constituents.

读 data_cache/csi300_constituents.csv 的 300 只股票, 调
factor_mining.margin_trading(code) 抓全部历史 (~2010 至今), 算 5d/20d
变化率, 拼大表落盘到 data_cache/csi300_margin_14yr.parquet.

resumable: 已存在的 cache 会按 code 跳过, 中途 Ctrl-C 后重跑 idempotent.

支持中途增量保存: 每 N 只 flush 一次, 避免后段崩了前段白干.

输出 columns: [code, date, rzye, rzmre, rzche, margin_5d_chg, margin_20d_chg]

run:
  python examples/fetch_csi300_margin.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

warnings.filterwarnings("ignore")

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
OUT_PATH = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
FLUSH_EVERY = 25  # 每 25 只 flush 一次到 parquet


def main() -> int:
    from factor_mining import margin_trading

    if not CSI300_PATH.exists():
        print(f"FATAL: {CSI300_PATH} 不存在", file=sys.stderr)
        return 1

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = csi["code"].tolist()
    n_total = len(codes)
    print(f"[init] CSI300: {n_total} 只", flush=True)

    # 已 cache 的跳过 (按 code 集合判定, 不按行数, 避免重算)
    cached_codes: set[str] = set()
    cached_rows: list[pd.DataFrame] = []
    if OUT_PATH.exists():
        old = pd.read_parquet(OUT_PATH)
        cached_codes = set(old["code"].astype(str).str.zfill(6).unique())
        cached_rows.append(old)
        print(f"[resume] {OUT_PATH.name} 已存在, 缓存 {len(cached_codes)} 只", flush=True)

    todo = [c for c in codes if c not in cached_codes]
    if not todo:
        print(f"[done] 全部 {n_total} 只都已 cache, 退出", flush=True)
        return 0

    print(f"[start] 待抓 {len(todo)} 只 (跳过 {len(cached_codes)} 只), "
          f"预计 ~{len(todo)*18/60:.1f} 分钟", flush=True)

    new_rows: list[pd.DataFrame] = []
    failed: list[str] = []
    t_start = time.time()

    for i, code in enumerate(todo, 1):
        t0 = time.time()
        try:
            df = margin_trading(code)
        except Exception as e:
            print(f"  [{i:3d}/{len(todo)}] {code} EXC: {str(e)[:80]}", flush=True)
            failed.append(code)
            continue
        elapsed = time.time() - t0

        if df.empty:
            failed.append(code)
            print(f"  [{i:3d}/{len(todo)}] {code} EMPTY ({elapsed:.1f}s)", flush=True)
            continue

        df = df.sort_values("date").reset_index(drop=True)
        df["margin_5d_chg"] = df["rzye"].pct_change(5)
        df["margin_20d_chg"] = df["rzye"].pct_change(20)
        new_rows.append(df[["code", "date", "rzye", "rzmre", "rzche",
                            "margin_5d_chg", "margin_20d_chg"]])

        if i % 10 == 0 or i == len(todo):
            done_frac = i / len(todo)
            elapsed_total = time.time() - t_start
            eta_min = elapsed_total / done_frac * (1 - done_frac) / 60
            print(f"  [{i:3d}/{len(todo)}] {code}: {len(df)} rows "
                  f"({elapsed:.1f}s) | fail {len(failed)} | "
                  f"elapsed {elapsed_total/60:.1f}m | ETA {eta_min:.1f}m",
                  flush=True)

        # 周期性 flush 防丢
        if i % FLUSH_EVERY == 0:
            combined = cached_rows + new_rows
            big = pd.concat(combined, ignore_index=True)
            big.to_parquet(OUT_PATH, index=False)
            print(f"    [flush] {len(big):,} rows → {OUT_PATH.name}", flush=True)

        time.sleep(0.05)

    # final flush
    if new_rows:
        combined = cached_rows + new_rows
        big = pd.concat(combined, ignore_index=True)
        big.to_parquet(OUT_PATH, index=False)
        print(f"[final-flush] {len(big):,} rows → {OUT_PATH.name}", flush=True)

    n_success = len(todo) - len(failed)
    fail_pct = len(failed) / max(len(todo), 1) * 100
    print(f"\n=== fetch summary ===", flush=True)
    print(f"  todo:    {len(todo)}", flush=True)
    print(f"  success: {n_success}", flush=True)
    print(f"  failed:  {len(failed)} ({fail_pct:.1f}%)", flush=True)
    print(f"  failed codes: {failed[:20]}", flush=True)
    if fail_pct > 10:
        print(f"FATAL: 失败率 {fail_pct:.1f}% > 10%, 数据质量不够, abort backtest",
              file=sys.stderr, flush=True)
        return 2
    print(f"[ok] saved {OUT_PATH.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
