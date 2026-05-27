"""抓 CSI300 龙虎榜 2014-2020 IS 期数据 → per-stock parquet cache.

Endpoint: datacenter-web.eastmoney.com RPT_DAILYBILLBOARD_DETAILSNEW
Cache:    data_cache/dragon_tiger/{code}.parquet
Universe: 300 stocks (CSI300 current constituents — survivorship bias 注明)
Wall:     ~4-6 min (实测 ~0.8s/股)

CRITICAL: 严格 IS 2014-2020 only. OOS 期 (2021-2026) 绝不触碰.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claude_finance import dragon_tiger  # noqa: E402

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
IS_START = "2014-01-01"
IS_END = "2020-12-31"


def main() -> int:
    if not CSI300_PATH.exists():
        print(f"FATAL: 缺 {CSI300_PATH}", file=sys.stderr)
        return 1

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = csi["code"].tolist()
    print(f"[fetch] CSI300 {len(codes)} stocks, IS {IS_START} → {IS_END}")

    t_total = time.time()
    n_with_events = 0
    n_empty = 0
    n_failed = 0
    total_rows = 0
    failed_codes: list[str] = []

    for i, code in enumerate(codes):
        t0 = time.time()
        try:
            df = dragon_tiger.fetch_and_cache(
                code,
                IS_START,
                IS_END,
                skip_if_recent=True,
                page_size=500,
                retries=3,
                sleep_between_pages=0.2,
            )
            n_rows = len(df) if df is not None else 0
            total_rows += n_rows
            if n_rows > 0:
                n_with_events += 1
            else:
                n_empty += 1
            if (i + 1) % 30 == 0 or i + 1 == len(codes):
                print(
                    f"[{i+1:3d}/{len(codes)}] {code}: {n_rows} rows "
                    f"({time.time()-t0:.2f}s) "
                    f"events={n_with_events} empty={n_empty} "
                    f"fail={n_failed}",
                    flush=True,
                )
        except Exception as e:
            n_failed += 1
            failed_codes.append(code)
            print(
                f"[{i+1:3d}/{len(codes)}] {code}: FAIL "
                f"{type(e).__name__}: {str(e)[:80]}",
                flush=True,
            )

    elapsed = time.time() - t_total
    print()
    print(f"=== Done in {elapsed/60:.1f} min ===")
    print(f"with_events:  {n_with_events}")
    print(f"empty (no龙虎榜): {n_empty}")
    print(f"failed:       {n_failed}")
    print(f"total rows:   {total_rows}")
    if failed_codes:
        print(f"failed codes: {failed_codes[:20]}")

    return 0 if n_failed < 30 else 1


if __name__ == "__main__":
    raise SystemExit(main())
