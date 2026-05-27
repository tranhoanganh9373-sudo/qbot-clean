"""Fetch 全市场限售股解禁数据 2014-2021Q1, 写 data_cache/unlock/.

策略: akshare.stock_restricted_release_detail_em 一次拉全市场区间,
6 个月 chunk + SSL 重试. 比 per-stock queue_em(300 次循环) 快 ~10x.

【与 spec 偏离】spec 写 "data_cache/unlock/{code}.parquet" per-stock; 实际用
单 parquet 全市场 cache (detail_em 是全市场 endpoint, 拆 per-stock 反而慢且
无 dedupe 优势). 下游 IC 用 unlock.filter_codes(csi300) 过滤.

【为何拉到 2021-Q1】IS=2014-2020, 因子需 forward 60 日窗口, 2020-12-31 边界
样本需要 unlock 数据到 2021-03-01 才能完整覆盖.

输出: data_cache/unlock/unlock_detail_em.parquet

run:
  python examples/fetch_unlock_csi300.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claude_finance import unlock  # noqa: E402

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"

FETCH_START = "20140101"
FETCH_END = "20210331"
CHUNK_MONTHS = 6


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
    if not CSI300_PATH.exists():
        print(f"FATAL: {CSI300_PATH} 缺", file=sys.stderr)
        return 1

    chunks = list(_chunks(FETCH_START, FETCH_END, CHUNK_MONTHS))
    print(f"[init] {len(chunks)} 个 {CHUNK_MONTHS} 月 chunks "
          f"({FETCH_START} → {FETCH_END})", flush=True)

    t_start = time.time()
    all_dfs: list[pd.DataFrame] = []
    failed_chunks: list[tuple[str, str, str]] = []

    for i, (s, e) in enumerate(chunks, 1):
        t0 = time.time()
        try:
            df = unlock.fetch_detail_range(s, e, max_retries=4, retry_sleep=3.0)
            all_dfs.append(df)
            print(f"  [{i:2d}/{len(chunks)}] {s}-{e}: "
                  f"{len(df):4d} rows ({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed_chunks.append((s, e, str(exc)[:120]))
            print(f"  [{i:2d}/{len(chunks)}] {s}-{e}: FAIL — "
                  f"{str(exc)[:80]}", flush=True)
        time.sleep(0.5)

    n_success = len(chunks) - len(failed_chunks)
    success_pct = n_success / len(chunks) * 100
    print(f"\n[fetch summary] success {n_success}/{len(chunks)} "
          f"({success_pct:.1f}%) in {(time.time()-t_start)/60:.1f}m",
          flush=True)

    if success_pct < 80:
        print(f"FATAL: 成功率 {success_pct:.1f}% < 80%, abort",
              file=sys.stderr, flush=True)
        return 2

    big = (pd.concat(all_dfs, ignore_index=True)
           if all_dfs else unlock._normalize(None))
    unlock.save_cache(big)
    final = unlock.load_cache()
    print(f"[save] {len(final):,} rows × "
          f"{final['code'].nunique():,} unique stocks → "
          f"{unlock.CACHE_FILE}", flush=True)

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    csi_codes = set(csi["code"])
    csi_in_cache = csi_codes & set(final["code"].unique())
    print(f"[verify] CSI300 with ≥1 unlock event: "
          f"{len(csi_in_cache)}/{len(csi_codes)} "
          f"({len(csi_in_cache)/len(csi_codes)*100:.1f}%)", flush=True)

    pct = len(csi_in_cache) / len(csi_codes) * 100
    if pct < 50:
        print(f"WARN: CSI300 覆盖 {pct:.1f}% < 50%, 因子 sparsity "
              f"会高 (老股已全流通无解禁事件)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
