"""抓 CSI300 全 300 股基本面 (sina stock_financial_analysis_indicator) 到
per-stock cache (data_cache/fundamentals/{code}.parquet).

设计:
- 读 data_cache/csi300_constituents.csv 拿 300 股 code list.
- 每股: 若 needs_refresh (cache 缺 / 末尾季报 > 90 天 stale) → fetch.
- fetch_fundamentals 内部 retry 2 次, 第 3 次失败记 fail csv.
- 单股预估 9-15 秒, 300 股 ~45-75 min wall.

输出:
- data_cache/fundamentals/{code}.parquet (每股)
- data_cache/fundamentals_fetch_failed.csv (失败列表)
- stdout: 进度 + 最终统计

run:
  python examples/fetch_fundamentals_csi300.py
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

from claude_finance import fundamentals  # noqa: E402

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
FAIL_CSV = ROOT / "data_cache" / "fundamentals_fetch_failed.csv"


def main() -> int:
    if not CSI300_PATH.exists():
        print(f"FATAL: {CSI300_PATH} 不存在", file=sys.stderr)
        return 1

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = csi["code"].tolist()
    n_total = len(codes)
    today = pd.Timestamp.now().normalize()
    print(f"[init] CSI300: {n_total} 只, today={today.date()}", flush=True)

    fails: list[dict] = []
    n_skip = 0
    n_ok = 0
    n_fail = 0
    t_start = time.time()

    for i, code in enumerate(codes, 1):
        if not fundamentals.needs_refresh(code, today):
            n_skip += 1
            if i % 25 == 0:
                print(
                    f"[{i:3d}/{n_total}] skip {code} (cache 新鲜); "
                    f"ok={n_ok} skip={n_skip} fail={n_fail}",
                    flush=True,
                )
            continue
        try:
            df = fundamentals.fetch_fundamentals(code)
            if df.empty:
                raise RuntimeError("fetch returned empty df")
            fundamentals.save_cached(code, df)
            n_ok += 1
            if i % 10 == 0 or i == n_total:
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (n_total - i) / rate if rate > 0 else 0
                print(
                    f"[{i:3d}/{n_total}] ok {code} ({len(df)} q); "
                    f"ok={n_ok} skip={n_skip} fail={n_fail} "
                    f"rate={rate:.2f}/s eta={eta:.0f}s",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            fails.append({
                "code": code,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })
            print(
                f"[{i:3d}/{n_total}] FAIL {code}: {type(e).__name__}",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(
        f"\n[done] total={n_total} ok={n_ok} skip={n_skip} fail={n_fail} "
        f"wall={elapsed:.1f}s ({elapsed/60:.1f}min) "
        f"success_rate={(n_ok + n_skip) / n_total * 100:.1f}%"
    )

    if fails:
        pd.DataFrame(fails).to_csv(FAIL_CSV, index=False)
        print(f"[fails] {len(fails)} 只写到 {FAIL_CSV}")

    return 0 if n_fail < n_total * 0.2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
