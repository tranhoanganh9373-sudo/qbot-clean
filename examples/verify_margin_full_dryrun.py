"""Stage 2 — dry-run 验证 csi300_margin_full.parquet 对 paper_trade_today.py
的 sidecar covered=N/296 影响. 不修改 production paper_trade_today.py.

通过 monkey-patch paper_trade_today.MARGIN_PARQUET → csi300_margin_full.parquet,
然后调用 load_margin_overlay 模拟 banner 数据。

不调用 qlib (训练耗时), 只检查 margin overlay coverage。

run:
  python examples/verify_margin_full_dryrun.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

FULL_PATH = ROOT / "data_cache" / "csi300_margin_full.parquet"
CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"


def main() -> int:
    if not FULL_PATH.exists():
        print(f"FATAL: {FULL_PATH} 不存在 — 先跑 merge_margin_full.py",
              file=sys.stderr)
        return 1

    import paper_trade_today as pt
    pt.MARGIN_PARQUET = FULL_PATH
    # daily sidecar 临时 forced miss → 强制走 long file path (我们要测的 full file)
    pt.MARGIN_DAILY_PARQUET = ROOT / "data_cache" / "__nonexistent__.parquet"

    print(f"[dryrun] MARGIN_PARQUET → {FULL_PATH.name}", flush=True)
    print(f"[dryrun] MARGIN_DAILY_PARQUET → (forced miss to test long file)",
          flush=True)

    today = pd.Timestamp.today().normalize()
    m5_map, m20_map, status = pt.load_margin_overlay(today)
    print(f"\n=== load_margin_overlay(today={today.date()}) ===", flush=True)
    print(f"  status:         {status}", flush=True)
    print(f"  m5 codes:       {len(m5_map)}", flush=True)
    print(f"  m20 codes:      {len(m20_map)}", flush=True)
    print(f"  sample m5 keys: {list(m5_map.keys())[:8]}", flush=True)

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    csi["instrument"] = csi.apply(
        lambda r: ("SH" if str(r["market"]).lower() == "sh" else "SZ") + r["code"],
        axis=1,
    )
    instruments = csi["instrument"].tolist()
    n_total = len(instruments)

    n_with_m5 = sum(1 for inst in instruments if inst in m5_map)
    n_with_m20 = sum(1 for inst in instruments if inst in m20_map)
    n_with_both = sum(1 for inst in instruments
                      if inst in m5_map and inst in m20_map)

    print(f"\n=== simulated banner (universe=CSI300 {n_total} 只) ===",
          flush=True)
    print(f"  Sidecar (v19.4):  λ_m5=0.10  λ_m20=0.10  "
          f"covered={n_with_both}/{n_total}  ({n_with_both/n_total*100:.1f}%)",
          flush=True)
    print(f"  (m5 covered={n_with_m5}, m20 covered={n_with_m20})", flush=True)

    missing_in_overlay = [inst for inst in instruments if inst not in m5_map]
    print(f"\n  still missing in overlay ({len(missing_in_overlay)}):",
          flush=True)
    if missing_in_overlay:
        print(f"    {missing_in_overlay[:20]}"
              f"{'...' if len(missing_in_overlay) > 20 else ''}",
              flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
