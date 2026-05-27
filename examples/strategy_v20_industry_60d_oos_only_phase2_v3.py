"""v20 industry_adj_ret_60d single OOS retest on Phase 2 v3 retrained cache.

Strict OOS protocol: do NOT re-sweep λ. Use locked λ=0.10 sign=+1 from
Phase A (factor_ic_industry_adj_ret_is.csv, subagent a2f9adefb996c3eeb).

Formula:
    final = z(train24_pred) + 0.10 · z(industry_adj_ret_60d)

Reference (v2 cache):
    v20 industry_60d (λ=0.10) : Calmar=0.84

This task tests the same locked combo on v3 (Phase 2 retrained) cache.

Inputs:
  data_cache/v17_dens_train24_predictions.parquet  (Phase 2 v3 retrained)
  data_cache/baidu_kline.parquet                   (v3 hfq cache, for ret_60d)
  data_cache/industry/industry_membership.parquet  (SW level-1, Phase A)

Output:
  examples/v20_industry_60d_oos_phase2_v3_stats.csv    (per-month abs_ret_%)
  examples/v20_industry_60d_oos_phase2_v3_summary.txt  (Calmar/Sharpe/...)

Does NOT touch production / does NOT run IS sweep.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

# Reuse helpers from the original v20 script
from strategy_v20_industry_60d import (  # noqa: E402
    build_industry_60d_panel,
    build_adjusted_predictions,
    run_walkforward,
    OOS_FIRST,
    OOS_LAST,
)

# Locked from Phase A — do NOT change
LOCKED_LAM = 0.10
LOCKED_SIGN = +1
LOCKED_LABEL = "v20_ind60d_l010"

OUT_STATS_CSV = ROOT / "examples" / "v20_industry_60d_oos_phase2_v3_stats.csv"
OUT_SUMMARY = ROOT / "examples" / "v20_industry_60d_oos_phase2_v3_summary.txt"


def main() -> int:
    print("=" * 70)
    print("v20 industry_adj_ret_60d OOS-only retest on Phase 2 v3 cache")
    print("=" * 70)
    print(f"OOS    : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"locked : λ={LOCKED_LAM} sign={LOCKED_SIGN:+d} ({LOCKED_LABEL})")
    print("strict OOS — no IS sweep, no λ re-optimization")
    print()

    print("[step 1] build industry_adj_ret_60d PIT panel...")
    panel = build_industry_60d_panel()

    print("\n[step 2] build adjusted predictions with locked λ ...")
    build_adjusted_predictions(panel, LOCKED_LAM, LOCKED_LABEL)

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}) ...")
    stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_phase2_v3")

    months_df = stats.pop("months_df")
    months_df.to_csv(OUT_STATS_CSV, index=False)
    print(f"\n[saved] {OUT_STATS_CSV}")

    summary_lines = [
        "=" * 70,
        "v20 industry_adj_ret_60d OOS Phase 2 v3 retest summary",
        "=" * 70,
        f"locked config: λ={LOCKED_LAM} sign={LOCKED_SIGN:+d} "
        f"({LOCKED_LABEL})",
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
        "  baseline train24      : Calmar=0.86  Sharpe=0.81  ann=29.12%  MDD=-33.74%",
        "  v19.4 (m5+m20 0.10)   : Calmar=1.28  Sharpe=0.94  ann=40.19%  MDD=-31.32%",
        "  v19.6 (amp 0.30)      : Calmar=0.58  Sharpe=0.67  ann=21.78%  MDD=-37.24%",
        "  v20  (ind60d 0.10)    : Calmar=0.84",
        "",
        "Phase 2 v3 retrained cache (60m OOS):",
        "  baseline train24      : Calmar=0.77",
        "  v19.4 (m5+m20 0.10)   : Calmar=0.62",
        "  v19.6 (amp 0.30) PROD : Calmar=1.29",
        f"  v20  (ind60d 0.10)    : Calmar={stats['calmar']}  <-- this task",
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
