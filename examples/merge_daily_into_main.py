"""Framework: merge daily sidecar parquets into main baidu_kline.parquet.

Purpose: 当出现多 fetcher 并行写 daily sidecar 场景时, 周期性把
data_cache/baidu_kline_daily/*.parquet 合并到主表去重 + atomic swap。

**当前状态**: framework-only. 主 daily fetch 仍走 fetch_baidu_kline.py 直接
更新主表。这个脚本是 future infrastructure 占位 — 等真有 sidecar 写入痛点
再 wire 进 daily_check.sh。

Workflow:
1. 扫 data_cache/baidu_kline_daily/*.parquet
2. read main parquet + concat sidecars
3. dedupe by (code, date) keeping latest row
4. write tmp + atomic swap (backup -> baidu_kline.parquet.pre_merge.bak)
5. rm sidecar 文件

Usage:
  python examples/merge_daily_into_main.py            # dry-run (default)
  python examples/merge_daily_into_main.py --apply    # actually merge + swap
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DC = ROOT / "data_cache"
MAIN = DC / "baidu_kline.parquet"
SIDECAR_DIR = DC / "baidu_kline_daily"
TMP = DC / "baidu_kline.parquet.tmp_merge"
BAK = DC / "baidu_kline.parquet.pre_merge.bak"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    print(f"[merge_daily] sidecar dir: {SIDECAR_DIR}")
    if not SIDECAR_DIR.exists():
        print("  ! sidecar dir does not exist — nothing to merge")
        print("  (framework only; not wired to daily fetch yet)")
        return 0

    sidecars = sorted(SIDECAR_DIR.glob("*.parquet"))
    print(f"  found {len(sidecars)} sidecar files")
    if not sidecars:
        print("[done] nothing to merge")
        return 0

    total_new_rows = 0
    for sc in sidecars:
        sz = sc.stat().st_size
        try:
            n = len(pd.read_parquet(sc, columns=["code"]))
        except Exception as e:
            print(f"  ! {sc.name}: {e}")
            continue
        total_new_rows += n
        print(f"  {sc.name}: {n:,} rows  {sz/1e3:.1f} KB")

    if not args.apply:
        print(f"\n[dry-run] would merge {total_new_rows:,} sidecar rows into main")
        print(f"  would write: {TMP}")
        print(f"  would rename: {MAIN.name} -> {BAK.name}")
        print(f"  would delete: {len(sidecars)} sidecar files")
        print("  re-run with --apply")
        return 0

    t0 = time.perf_counter()
    print(f"[1/4] reading main: {MAIN.name}")
    main_df = pd.read_parquet(MAIN)
    print(f"  main rows: {len(main_df):,}")

    print("[2/4] concatenating sidecars...")
    pieces = [main_df]
    for sc in sidecars:
        pieces.append(pd.read_parquet(sc))
    merged = pd.concat(pieces, ignore_index=True)
    print(f"  pre-dedupe rows: {len(merged):,}")

    print("[3/4] dedupe by (code, date) keep last...")
    merged = merged.drop_duplicates(subset=["code", "date"], keep="last").reset_index(drop=True)
    print(f"  post-dedupe rows: {len(merged):,}  (added {len(merged)-len(main_df):,} new)")

    print(f"[4/4] writing tmp: {TMP}")
    # Match row-group-split layout: sort + per-year row groups
    merged["date"] = pd.to_datetime(merged["date"])
    merged["_year"] = merged["date"].dt.year
    merged = merged.sort_values(["_year", "code", "date"], kind="mergesort").reset_index(drop=True)
    write_df = merged.drop(columns=["_year"])

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.Schema.from_pandas(write_df, preserve_index=False)
    years = sorted(merged["_year"].unique().tolist())
    with pq.ParquetWriter(str(TMP), schema, compression="snappy", use_dictionary=True) as writer:
        for y in years:
            chunk = write_df.loc[merged["_year"] == y]
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            writer.write_table(table, row_group_size=len(chunk) + 1)

    print(f"  tmp size: {TMP.stat().st_size/1e6:.1f} MB")

    print(f"[swap] {MAIN.name} -> {BAK.name}")
    os.rename(MAIN, BAK)
    os.rename(TMP, MAIN)

    print(f"[cleanup] removing {len(sidecars)} sidecar files")
    for sc in sidecars:
        sc.unlink()

    print(f"[done] wall={time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
