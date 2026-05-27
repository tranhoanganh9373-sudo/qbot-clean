"""Re-write baidu_kline.parquet with row groups aligned to year boundaries.

Goal: enable DuckDB / pyarrow predicate pushdown on `date` column to skip
non-relevant row groups. Currently the file has 8 generic row groups of
~1M rows each; after this rewrite each year is its own row group with
date statistics (min/max) for fast pruning.

Strategy:
1. Read full parquet into pandas
2. Sort by (year, code, date)
3. Use pyarrow ParquetWriter to write one row group per year

Atomic swap:
- Write to baidu_kline.parquet.tmp
- mv original -> baidu_kline.parquet.pre_rowgroup.bak
- mv tmp -> baidu_kline.parquet

Usage:
  python examples/parquet_rowgroup_split.py            # dry-run plan
  python examples/parquet_rowgroup_split.py --apply    # actually swap
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data_cache" / "baidu_kline.parquet"
TMP = ROOT / "data_cache" / "baidu_kline.parquet.tmp"
BAK = ROOT / "data_cache" / "baidu_kline.parquet.pre_rowgroup.bak"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually swap (default: dry-run)")
    args = ap.parse_args()

    print(f"[rowgroup-split] src: {SRC}")
    if not SRC.exists():
        print("  ! source not found")
        return 1

    src_size = SRC.stat().st_size
    print(f"  size: {src_size/1e6:.1f} MB")

    pf = pq.ParquetFile(str(SRC))
    print(f"  current row_groups: {pf.num_row_groups}, total_rows: {pf.metadata.num_rows:,}")

    t0 = time.perf_counter()
    print("[1/3] reading full parquet into pandas...")
    df = pd.read_parquet(SRC)
    print(f"  rows={len(df):,} cols={list(df.columns)}")

    # Normalize date and add year
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["_year"] = df["date"].dt.year

    years = sorted(df["_year"].unique().tolist())
    print(f"  years: {years[0]}..{years[-1]}  ({len(years)} groups)")
    by_year_counts = df.groupby("_year").size()
    for y in years:
        print(f"    {y}: {by_year_counts[y]:>10,} rows")

    print("[2/3] sorting by (year, code, date)...")
    df = df.sort_values(["_year", "code", "date"], kind="mergesort").reset_index(drop=True)

    if not args.apply:
        print("\n[dry-run] re-run with --apply to write tmp + atomic swap")
        print(f"  would write: {TMP}")
        print(f"  would rename: {SRC.name} -> {BAK.name}")
        print(f"  would rename: {TMP.name} -> {SRC.name}")
        return 0

    print(f"[3/3] writing per-year row groups -> {TMP}")
    write_df = df.drop(columns=["_year"])
    schema = pa.Schema.from_pandas(write_df, preserve_index=False)

    n_rows_written = 0
    with pq.ParquetWriter(str(TMP), schema, compression="snappy", use_dictionary=True) as writer:
        for y in years:
            mask = df["_year"] == y
            chunk = write_df.loc[mask]
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            # Force one row group per year by setting row_group_size > chunk size.
            writer.write_table(table, row_group_size=len(chunk) + 1)
            n_rows_written += len(chunk)
            print(f"    year {y}: {len(chunk):>10,} rows")

    print(f"  wrote {n_rows_written:,} rows  size={TMP.stat().st_size/1e6:.1f} MB")

    print("[verify] inspecting new parquet...")
    pf_new = pq.ParquetFile(str(TMP))
    print(f"  new row_groups: {pf_new.num_row_groups}")
    date_idx = next(j for j, c in enumerate(write_df.columns) if c == "date")
    for i in range(pf_new.num_row_groups):
        rg = pf_new.metadata.row_group(i)
        stat = rg.column(date_idx).statistics
        print(f"    RG {i}: {rg.num_rows:>10,} rows  date {stat.min}..{stat.max}")

    if pf_new.num_row_groups != len(years):
        print(f"  ! expected {len(years)} row groups, got {pf_new.num_row_groups}")
        print(f"  ! abort swap, leaving tmp at {TMP}")
        return 2
    if pf_new.metadata.num_rows != len(df):
        print(f"  ! row count mismatch: new={pf_new.metadata.num_rows} src={len(df)}")
        print(f"  ! abort swap, leaving tmp at {TMP}")
        return 3

    print(f"[swap] {SRC.name} -> {BAK.name}")
    os.rename(SRC, BAK)
    print(f"[swap] {TMP.name} -> {SRC.name}")
    os.rename(TMP, SRC)

    new_size = SRC.stat().st_size
    print(f"[done] wall={time.perf_counter()-t0:.1f}s  size {src_size/1e6:.1f} -> {new_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
