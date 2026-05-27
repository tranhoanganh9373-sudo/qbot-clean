"""v19.7 stacked (amp_imb_20d + margin_5d_chg) OOS retest on Phase 2 v3 cache.

Strict OOS protocol: do NOT re-sweep λ. Use locked λ=(a20=0.20, m5=0.20)
identified during the original 2026-05-25 v19.7 IS lock (combo c6_a020_m020,
recorded in MEMORY `project_phase4_sidecar_v7_stacked.md`).

Formula:
    final = z(train24_pred) - 0.20 · z(amp_imb_20d) - 0.20 · z(margin_5d_chg)

Reference (v2 clean cache, locked combo c6_a020_m020):
    v19.7 stacked       : OOS Calmar=0.65 ann=12.23% Sharpe=0.69 MDD=-18.72%

This task tests the same locked combo on v3 (Phase 2 retrained) cache to verify
whether the v3 baseline upgrade changes the stacking verdict relative to
v19.6 (single a20) which scored Calmar 1.29 on v3.

Inputs:
  data_cache/v17_dens_train24_predictions.parquet  (Phase 2 v3 retrained)
  data_cache/csi300_margin_14yr.parquet            (Stage 2 swap, 300 codes)
  data_cache/baidu_kline.parquet                   (v3 hfq cache, for amp_imb_20d)

Output:
  examples/v19_7_oos_phase2_v3_stats.csv    (per-month abs_ret_%)
  examples/v19_7_oos_phase2_v3_summary.txt  (Calmar/Sharpe/ann/MDD)

Does NOT touch production / does NOT run IS sweep.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

# Reuse v19.7 helpers from the original sidecar script.
from strategy_v19_7_stacked import (  # noqa: E402
    build_margin_panel,
    build_adjusted_predictions,
    run_walkforward,
    OOS_FIRST,
    OOS_LAST,
    ORIG_PRED,
)
from _factor_kline_panel import build_pit_panel_on_pred_axis  # noqa: E402

# Locked combo from 2026-05-25 v19.7 IS sweep — do NOT change
LOCKED_LAM_A20 = 0.20
LOCKED_LAM_M5 = 0.20
LOCKED_LABEL = "c6_a020_m020"

OUT_STATS_CSV = ROOT / "examples" / "v19_7_oos_phase2_v3_stats.csv"
OUT_SUMMARY = ROOT / "examples" / "v19_7_oos_phase2_v3_summary.txt"


def main() -> int:
    print("=" * 70)
    print("v19.7 stacked (a20+m5) OOS-only retest on Phase 2 v3 cache")
    print("=" * 70)
    print(f"OOS    : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"locked : {LOCKED_LABEL} "
          f"λ=(a20={LOCKED_LAM_A20}, m5={LOCKED_LAM_M5})")
    print("strict OOS — no IS re-sweep, no λ re-optimization")
    print()

    print("[step 1] build kline PIT panel (amp_imb_20d)...")
    amp_panel = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["amp_imb_20d"],
    )

    print("\n[step 1b] build margin PIT panel (margin_5d_chg)...")
    margin_panel = build_margin_panel()

    print("\n[step 2] build adjusted predictions with locked λ ...")
    build_adjusted_predictions(
        amp_panel, margin_panel,
        LOCKED_LAM_A20, LOCKED_LAM_M5,
        LOCKED_LABEL,
    )

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}) ...")
    stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_phase2_v3")

    months_df = stats.pop("months_df")
    months_df.to_csv(OUT_STATS_CSV, index=False)
    print(f"\n[saved] {OUT_STATS_CSV}")

    summary_lines = [
        "=" * 70,
        "v19.7 stacked OOS Phase 2 v3 retest summary",
        "=" * 70,
        f"locked config: {LOCKED_LABEL} "
        f"λ=(a20={LOCKED_LAM_A20}, m5={LOCKED_LAM_M5})",
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
        "  baseline train24      : Calmar=0.86  Sharpe=0.81  ann=29.12%",
        "  v19.4 (m5+m20 0.10)   : Calmar=1.28  Sharpe=0.94  ann=40.19%",
        "  v19.6 (amp 0.30)      : Calmar=0.58  Sharpe=0.67  ann=21.78%",
        "  v19.7 (a20+m5 0.20)   : Calmar=0.65  Sharpe=0.69  ann=12.23%",
        "",
        "Phase 2 v3 retrained cache (60m OOS):",
        "  baseline train24      : Calmar=0.77",
        "  v19.4 (m5+m20 0.10)   : Calmar=0.62",
        "  v19.6 (amp 0.30) PROD : Calmar=1.29",
        f"  v19.7 (a20+m5 0.20)   : Calmar={stats['calmar']}  <-- this task",
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
