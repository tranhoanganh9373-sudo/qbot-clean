"""v19.4 single OOS retest on Phase 2 v3 retrained predictions cache.

Strict OOS protocol: do NOT re-sweep λ. Use locked λ=(m5=0.10, m20=0.10, dt=0.00)
identified during the original 2026-05-25 v19.4 lock.

Inputs:
  data_cache/v17_dens_train24_predictions.parquet  (Phase 2 v3 retrained, 4625-code)
  data_cache/csi300_margin_14yr.parquet            (Stage 2 swap, 300 codes)

Output:
  examples/v19_4_oos_phase2_v3_stats.csv    (per-month abs_ret_%)
  examples/v19_4_oos_phase2_v3_summary.txt  (Calmar/Sharpe/ann/MDD)

Does NOT touch production / does NOT re-run IS sweep.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

# Reuse v19.4 helpers
from strategy_v19_4_sidecar_technical import (  # noqa: E402
    build_technical_panel,
    build_adjusted_predictions,
    run_walkforward,
    OOS_FIRST,
    OOS_LAST,
)

# Locked λ from 2026-05-25 v19.4 IS sweep — do NOT change
LOCKED_LAM_M5 = 0.10
LOCKED_LAM_M20 = 0.10
LOCKED_LAM_DT = 0.00
LOCKED_LABEL = "c4_m5m20_l020"

OUT_STATS_CSV = ROOT / "examples" / "v19_4_oos_phase2_v3_stats.csv"
OUT_SUMMARY = ROOT / "examples" / "v19_4_oos_phase2_v3_summary.txt"


def main() -> int:
    print("=" * 70)
    print("v19.4 OOS-only retest on Phase 2 v3 retrained cache")
    print("=" * 70)
    print(f"OOS    : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"locked : {LOCKED_LABEL} "
          f"lam=(m5={LOCKED_LAM_M5}, m20={LOCKED_LAM_M20}, dt={LOCKED_LAM_DT})")
    print("strict OOS — no IS re-sweep, no lambda re-optimization")
    print()

    print("[step 1] build technical PIT panel (margin + dragon_tiger)...")
    panel = build_technical_panel()

    print("\n[step 2] build adjusted predictions with locked lambda ...")
    build_adjusted_predictions(panel, LOCKED_LAM_M5, LOCKED_LAM_M20,
                               LOCKED_LAM_DT, LOCKED_LABEL)

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}) ...")
    stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_phase2_v3")

    months_df = stats.pop("months_df")
    months_df.to_csv(OUT_STATS_CSV, index=False)
    print(f"\n[saved] {OUT_STATS_CSV}")

    summary_lines = [
        "=" * 70,
        "v19.4 OOS Phase 2 v3 retest summary",
        "=" * 70,
        f"locked config: {LOCKED_LABEL} "
        f"lam=(m5={LOCKED_LAM_M5}, m20={LOCKED_LAM_M20}, dt={LOCKED_LAM_DT})",
        f"OOS window  : {OOS_FIRST} ~ {OOS_LAST} ({stats['n']} months)",
        "",
        f"  cum_%      : {stats['cum_%']}",
        f"  ann_%      : {stats['ann_%']}",
        f"  sharpe     : {stats['sharpe']}",
        f"  mdd_%      : {stats['mdd_%']}",
        f"  calmar     : {stats['calmar']}",
        f"  win_%      : {stats['win_%']}",
        f"  avg_picks  : {stats['avg_picks']}",
        "",
        "Reference baselines (v2 clean cache, 60m OOS 2021-05~2026-04):",
        "  baseline train24   : Calmar=0.86  Sharpe=0.81  ann=29.12%  MDD=-33.74%",
        "  v19.4 (m5+m20 0.1) : Calmar=1.28  Sharpe=0.94  ann=40.19%  MDD=-31.32%",
        "  v19.6 (amp 0.30)   : Calmar=0.58  Sharpe=0.67  ann=21.78%  MDD=-37.24%",
        "",
    ]
    summary = "\n".join(summary_lines)
    OUT_SUMMARY.write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print(f"\n[saved] {OUT_SUMMARY}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
