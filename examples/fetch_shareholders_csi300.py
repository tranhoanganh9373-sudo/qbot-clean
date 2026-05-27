"""Fetch CSI300 全 history 股东户数明细, 落 data_cache/shareholders/.

策略: per-stock 抓取 akshare.stock_zh_a_gdhs_detail_em(symbol=code).
落 per-stock parquet + 合并 shareholders_csi300.parquet.

输出:
  data_cache/shareholders/per_stock/{code}.parquet
  data_cache/shareholders/shareholders_csi300.parquet

run:
  python examples/fetch_shareholders_csi300.py
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

from claude_finance import shareholders  # noqa: E402

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
RATE_SLEEP = 0.4  # 秒/股, 防 EM 限流


def main() -> int:
    if not CSI300_PATH.exists():
        print(f"FATAL: {CSI300_PATH} 缺", file=sys.stderr)
        return 1

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = csi["code"].tolist()
    print(f"[init] CSI300 universe = {len(codes)} 股", flush=True)

    t_start = time.time()
    n_ok = 0
    n_skip = 0
    failed: list[tuple[str, str]] = []

    shareholders.PER_STOCK_DIR.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(codes, 1):
        # 跳过已存在的 (resumable)
        per_stock_path = shareholders.PER_STOCK_DIR / f"{code}.parquet"
        if per_stock_path.exists():
            n_skip += 1
            if i % 50 == 0:
                print(f"  [{i:3d}/{len(codes)}] skip-existing "
                      f"(ok={n_ok}, skip={n_skip}, fail={len(failed)}) "
                      f"@ {(time.time()-t_start)/60:.1f}m",
                      flush=True)
            continue
        try:
            df = shareholders.fetch_stock(code, max_retries=3, retry_sleep=2.0)
            shareholders.save_per_stock(df, code)
            n_ok += 1
            if i % 25 == 0 or i == len(codes):
                print(f"  [{i:3d}/{len(codes)}] ok={n_ok} skip={n_skip} "
                      f"fail={len(failed)} "
                      f"latest={code} rows={len(df)} "
                      f"@ {(time.time()-t_start)/60:.1f}m",
                      flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append((code, str(exc)[:120]))
            print(f"  [{i:3d}/{len(codes)}] FAIL {code}: "
                  f"{str(exc)[:80]}", flush=True)
        time.sleep(RATE_SLEEP)

    elapsed_min = (time.time() - t_start) / 60
    n_total = len(codes)
    print(f"\n[fetch summary] ok={n_ok} skip={n_skip} fail={len(failed)} "
          f"/ {n_total} in {elapsed_min:.1f}m", flush=True)
    success_rate = (n_ok + n_skip) / n_total * 100
    if success_rate < 80:
        print(f"FATAL: 成功率 {success_rate:.1f}% < 80%, abort",
              file=sys.stderr, flush=True)
        return 2

    # 合并 per-stock → 全集 parquet
    print("[merge] 合并 per-stock → shareholders_csi300.parquet", flush=True)
    big = shareholders.merge_all()
    print(f"[merge] {len(big):,} rows × "
          f"{big['code'].nunique():,} unique stocks → "
          f"{shareholders.MERGED_FILE}", flush=True)

    # CSI300 覆盖核对
    covered = set(big["code"].unique()) & set(codes)
    pct = len(covered) / len(codes) * 100
    print(f"[verify] CSI300 with ≥1 record: "
          f"{len(covered)}/{len(codes)} ({pct:.1f}%)", flush=True)

    # IS 期 (2014-2020) 覆盖
    is_mask = (
        (big["announce_date"] >= "2014-01-01")
        & (big["announce_date"] <= "2020-12-31")
    )
    is_sub = big[is_mask]
    is_covered = set(is_sub["code"].unique()) & set(codes)
    print(f"[verify] IS (2014-2020) CSI300 with ≥1 record: "
          f"{len(is_covered)}/{len(codes)} "
          f"({len(is_covered)/len(codes)*100:.1f}%)", flush=True)
    print(f"[verify] IS rows: {len(is_sub):,}", flush=True)

    if failed:
        print(f"\n[failed] {len(failed)} 股:")
        for code, msg in failed[:20]:
            print(f"  {code}: {msg}")
        if len(failed) > 20:
            print(f"  ... +{len(failed)-20} 更多")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
