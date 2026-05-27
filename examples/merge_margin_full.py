"""Stage 2 — merge csi300_margin_14yr.parquet (120 codes, read-only) + margin_180_backfill.parquet
into new csi300_margin_full.parquet (~296 codes).

绝不修改原 14yr 文件 — atomic write 到独立 full 文件, 用户决定 swap.

Schema (与 14yr 完全一致):
  code (str 6位), date (datetime64[ns]), rzye, rzmre, rzche,
  margin_5d_chg, margin_20d_chg (float64)

run:
  python examples/merge_margin_full.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

EXISTING_14YR = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
BACKFILL_PATH = ROOT / "data_cache" / "margin_180_backfill.parquet"
FULL_PATH = ROOT / "data_cache" / "csi300_margin_full.parquet"
CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    if not EXISTING_14YR.exists():
        print(f"FATAL: {EXISTING_14YR} 不存在", file=sys.stderr)
        return 1
    if not BACKFILL_PATH.exists():
        print(f"FATAL: {BACKFILL_PATH} 不存在 — 先跑 fetch_margin_backfill_180.py",
              file=sys.stderr)
        return 1

    base = pd.read_parquet(EXISTING_14YR)
    backfill = pd.read_parquet(BACKFILL_PATH)
    base["code"] = base["code"].astype(str).str.zfill(6)
    backfill["code"] = backfill["code"].astype(str).str.zfill(6)
    base["date"] = pd.to_datetime(base["date"])
    backfill["date"] = pd.to_datetime(backfill["date"])

    base_codes = set(base["code"].unique())
    bf_codes = set(backfill["code"].unique())
    overlap = base_codes & bf_codes
    print(f"[base] {len(base):,} rows × {len(base_codes)} codes  "
          f"({base['date'].min().date()} ~ {base['date'].max().date()})",
          flush=True)
    print(f"[backfill] {len(backfill):,} rows × {len(bf_codes)} codes  "
          f"({backfill['date'].min().date()} ~ {backfill['date'].max().date()})",
          flush=True)
    if overlap:
        print(f"[warn] overlap codes ({len(overlap)}): {sorted(overlap)[:10]}",
              flush=True)
        print(f"[warn] 重复 code 在 merge 时 backfill 优先 (drop_duplicates keep=last)",
              flush=True)

    if list(base.columns) != list(backfill.columns):
        print(f"[warn] schema 不一致: base={list(base.columns)} "
              f"bf={list(backfill.columns)}", flush=True)

    merged = pd.concat([base, backfill], ignore_index=True)
    merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["code", "date"], keep="last")
    after_dedup = len(merged)
    if before_dedup != after_dedup:
        print(f"[dedup] {before_dedup:,} → {after_dedup:,} "
              f"(removed {before_dedup - after_dedup:,} duplicates)", flush=True)

    _atomic_write(merged, FULL_PATH)
    print(f"\n[ok] saved {FULL_PATH.name}", flush=True)
    print(f"  rows: {len(merged):,}", flush=True)
    print(f"  unique codes: {merged['code'].nunique()}", flush=True)
    print(f"  date range: {merged['date'].min().date()} ~ {merged['date'].max().date()}",
          flush=True)

    if CSI300_PATH.exists():
        csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
        csi["code"] = csi["code"].astype(str).str.zfill(6)
        csi_codes = set(csi["code"])
        full_codes = set(merged["code"].unique())
        covered = csi_codes & full_codes
        still_missing = csi_codes - full_codes
        print(f"\n=== CSI300 coverage ===", flush=True)
        print(f"  total CSI300: {len(csi_codes)}", flush=True)
        print(f"  covered:      {len(covered)} ({len(covered)/len(csi_codes)*100:.1f}%)",
              flush=True)
        print(f"  still missing: {len(still_missing)}", flush=True)
        if still_missing:
            print(f"  missing list (first 20): {sorted(still_missing)[:20]}",
                  flush=True)

    latest = merged["date"].max()
    last_30 = merged[merged["date"] > latest - pd.Timedelta(days=40)]
    daily_cov = last_30.groupby("date")["code"].nunique()
    print(f"\n=== recent daily coverage (last 30 trading days) ===",
          flush=True)
    print(f"  median codes/day: {int(daily_cov.median())}", flush=True)
    print(f"  min codes/day:    {int(daily_cov.min())}", flush=True)
    print(f"  max codes/day:    {int(daily_cov.max())}", flush=True)
    print(f"  latest day ({latest.date()}): {daily_cov.get(latest, 0)} codes",
          flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
