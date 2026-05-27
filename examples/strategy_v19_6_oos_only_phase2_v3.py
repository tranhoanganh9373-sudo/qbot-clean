"""v19.6 single OOS retest on Phase 2 v3 retrained predictions cache.

Strict OOS protocol: do NOT re-sweep λ. Use locked λ_a20=0.30 sign=-1
identified during the original 2026-05-25 v19.6 lock (c6_a20_l030).

Inputs:
  data_cache/v17_dens_train24_predictions.parquet  (Phase 2 v3 retrained, 4625-code)
  data_cache/csi300_margin_14yr.parquet            (unused at locked lambda but kept
                                                    for API parity)
  data_cache/baidu_kline.parquet                   (v3 hfq cache, for amp_imb_20d)

Output:
  examples/v19_6_oos_phase2_v3_stats.csv    (per-month abs_ret_%)
  examples/v19_6_oos_phase2_v3_summary.txt  (Calmar/Sharpe/ann/MDD)

Does NOT touch production / does NOT re-run IS sweep.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

# Reuse v19.6 helpers from the original sidecar script.
from strategy_v19_6_amplitude import (  # noqa: E402
    build_margin_panel,
    build_adjusted_predictions,
    run_walkforward,
    OOS_FIRST,
    OOS_LAST,
    ORIG_PRED,
)
from _factor_kline_panel import build_pit_panel_on_pred_axis  # noqa: E402

# Locked combo from 2026-05-25 v19.6 IS sweep — do NOT change
LOCKED_LAM_A5 = 0.00
LOCKED_LAM_A20 = 0.30
LOCKED_LAM_MARGIN = 0.00
LOCKED_LABEL = "c6_a20_l030"

OUT_STATS_CSV = ROOT / "examples" / "v19_6_oos_phase2_v3_stats.csv"
OUT_SUMMARY = ROOT / "examples" / "v19_6_oos_phase2_v3_summary.txt"


def main() -> int:
    print("=" * 70)
    print("v19.6 OOS-only retest on Phase 2 v3 retrained cache")
    print("=" * 70)
    print(f"OOS    : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"locked : {LOCKED_LABEL} "
          f"lam=(a5={LOCKED_LAM_A5}, a20={LOCKED_LAM_A20}, "
          f"margin={LOCKED_LAM_MARGIN})")
    print("strict OOS — no IS re-sweep, no lambda re-optimization")
    print()

    print("[step 1] build kline PIT panel (amp_imb_5d/20d)...")
    amp_panel = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["amp_imb_5d", "amp_imb_20d"],
    )
    # locked lam_margin=0 → margin_panel not used, but pass empty/None-safe
    margin_panel = build_margin_panel()

    print("\n[step 2] build adjusted predictions with locked lambda ...")
    build_adjusted_predictions(
        amp_panel, margin_panel,
        LOCKED_LAM_A5, LOCKED_LAM_A20, LOCKED_LAM_MARGIN,
        LOCKED_LABEL,
    )

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}) ...")
    stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_phase2_v3")

    months_df = stats.pop("months_df")
    months_df.to_csv(OUT_STATS_CSV, index=False)
    print(f"\n[saved] {OUT_STATS_CSV}")

    summary_lines = [
        "=" * 70,
        "v19.6 OOS Phase 2 v3 retest summary",
        "=" * 70,
        f"locked config: {LOCKED_LABEL} "
        f"lam=(a5={LOCKED_LAM_A5}, a20={LOCKED_LAM_A20}, "
        f"margin={LOCKED_LAM_MARGIN})",
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
        "Phase 2 v3 retrained cache (task #79):",
        "  v19.4 (m5+m20 0.1) : Calmar=0.62 (re-tested, see "
        "v19_4_oos_phase2_v3_summary.txt)",
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
