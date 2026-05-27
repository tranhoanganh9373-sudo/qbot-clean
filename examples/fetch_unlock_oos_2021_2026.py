"""Fetch 全市场限售股解禁数据 2021-04-01 ~ 2026-06-30, merge 进 cache.

补 v19.9 sidecar OOS 期 (2021-05 ~ 2026-04) 所需数据.
前 cache 仅覆盖 2014 ~ 2021-03-31, 直接跑 OOS 会得到 final == baseline.

策略: 6 个月 chunk + SSL 重试 (复用 unlock.fetch_detail_range).
先备份原 parquet 到 .pre_oos_fetch.bak, 再 merge_into_cache 写回.

输出 (覆盖写):
    data_cache/unlock/unlock_detail_em.parquet           (merge 后全集)
    data_cache/unlock/unlock_detail_em.pre_oos_fetch.bak (备份原 IS-only)

Run:
  .venv/bin/python examples/fetch_unlock_oos_2021_2026.py
"""
from __future__ import annotations

import shutil
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claude_finance import unlock  # noqa: E402

FETCH_START = "20210401"   # 紧接前 cache 末日 2021-03-31
FETCH_END = "20260630"     # OOS 末月 2026-04, +60d forward buffer → 2026-06-30
CHUNK_MONTHS = 6
BACKUP = unlock.CACHE_FILE.parent / "unlock_detail_em.pre_oos_fetch.bak"


def _chunks(start: str, end: str, months: int):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s <= e:
        nxt = (s + pd.DateOffset(months=months)) - pd.Timedelta(days=1)
        if nxt > e:
            nxt = e
        yield s.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")
        s = nxt + pd.Timedelta(days=1)


def main() -> int:
    if not unlock.CACHE_FILE.exists():
        print(f"FATAL: cache {unlock.CACHE_FILE} 缺", file=sys.stderr)
        return 1

    # 备份原 IS-only 数据
    if not BACKUP.exists():
        shutil.copy(unlock.CACHE_FILE, BACKUP)
        print(f"[backup] {unlock.CACHE_FILE.name} → {BACKUP.name}",
              flush=True)
    else:
        print(f"[backup] {BACKUP.name} 已存在, 跳过", flush=True)

    cur = unlock.load_cache()
    print(f"[load] 原 cache: {len(cur):,} rows × "
          f"{cur['code'].nunique():,} stocks; "
          f"range {cur['unlock_date'].min().date()} ~ "
          f"{cur['unlock_date'].max().date()}", flush=True)

    chunks = list(_chunks(FETCH_START, FETCH_END, CHUNK_MONTHS))
    print(f"\n[init] {len(chunks)} 个 {CHUNK_MONTHS} 月 chunks "
          f"({FETCH_START} → {FETCH_END})", flush=True)

    t_start = time.time()
    all_dfs: list[pd.DataFrame] = []
    failed_chunks: list[tuple[str, str, str]] = []

    for i, (s, e) in enumerate(chunks, 1):
        t0 = time.time()
        try:
            df = unlock.fetch_detail_range(
                s, e, max_retries=4, retry_sleep=3.0,
            )
            all_dfs.append(df)
            print(f"  [{i:2d}/{len(chunks)}] {s}-{e}: "
                  f"{len(df):5d} rows ({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed_chunks.append((s, e, str(exc)[:120]))
            print(f"  [{i:2d}/{len(chunks)}] {s}-{e}: FAIL — "
                  f"{str(exc)[:80]}", flush=True)
        time.sleep(0.5)

    n_success = len(chunks) - len(failed_chunks)
    success_pct = n_success / len(chunks) * 100
    elapsed = (time.time() - t_start) / 60
    print(f"\n[fetch summary] success {n_success}/{len(chunks)} "
          f"({success_pct:.1f}%) in {elapsed:.1f}m", flush=True)

    if success_pct < 80:
        print(f"FATAL: 成功率 {success_pct:.1f}% < 80%, abort 写盘",
              file=sys.stderr, flush=True)
        return 2

    if not all_dfs:
        print("FATAL: 0 chunks succeeded", file=sys.stderr)
        return 2

    new_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[fetch] OOS new rows: {len(new_df):,} × "
          f"{new_df['code'].nunique():,} stocks; "
          f"range {new_df['unlock_date'].min().date()} ~ "
          f"{new_df['unlock_date'].max().date()}", flush=True)

    # Merge into existing cache (内部 dedupe + sort + save)
    merged = unlock.merge_into_cache(new_df)
    print(f"\n[save] merged cache: {len(merged):,} rows × "
          f"{merged['code'].nunique():,} stocks; "
          f"range {merged['unlock_date'].min().date()} ~ "
          f"{merged['unlock_date'].max().date()}", flush=True)

    # 验证 OOS 期非空 (2021-05 ~ 2026-04)
    oos_mask = (
        (merged["unlock_date"] >= "2021-05-01")
        & (merged["unlock_date"] <= "2026-04-30")
    )
    n_oos = int(oos_mask.sum())
    print(f"\n[verify] OOS 期 (2021-05 ~ 2026-04) events: {n_oos:,}",
          flush=True)
    if n_oos < 1000:
        print(f"WARN: OOS events {n_oos} < 1000, sparsity 可能过高",
              flush=True)

    # 年度分布
    by_year = merged.groupby(merged["unlock_date"].dt.year).size()
    print("\n[年度分布]")
    print(by_year.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
