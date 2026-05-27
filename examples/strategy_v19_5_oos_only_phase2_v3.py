"""v19.5 team_coin_20d OOS retest on Phase 2 v3 cache.

Strict OOS protocol: do NOT re-sweep λ. Use locked combo c5_tc20_l020
(λ_tc5=0.00, λ_tc20=0.20, λ_margin=0.00, sign=-1) identified during the
original 2026-05-25 v19.5 IS lock (MEMORY `project_phase4_sidecar_v5_v6.md`).

Formula:
    final = z(train24_pred) - 0.20 · z(team_coin_20d)

team_coin = intraday_ret × overnight_ret (per code, daily). Rolling 20d sum
captures persistent agreement (positive) or disagreement (negative) between
the day session and the overnight gap. Empirical sign is -1 (mean reversion).

Reference (v2 clean cache, locked c5_tc20_l020):
    v19.5 team_coin     : OOS Calmar=0.56 (abort vs v19.6 0.79)

This task tests the same locked combo on v3 (Phase 2 retrained) cache to verify
whether the v3 baseline upgrade changes the verdict relative to
v19.6 (single a20) which scored Calmar 1.29 on v3.

Inputs:
  data_cache/v17_dens_train24_predictions.parquet  (Phase 2 v3 retrained)
  data_cache/csi300_margin_14yr.parquet            (unused at lam_margin=0)
  data_cache/baidu_kline.parquet                   (v3 hfq cache, team_coin)

Output:
  examples/v19_5_oos_phase2_v3_stats.csv    (per-month abs_ret_%)
  examples/v19_5_oos_phase2_v3_summary.txt  (Calmar/Sharpe/ann/MDD)

Does NOT touch production / does NOT run IS sweep.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

# Reuse v19.5 helpers from the original sidecar script.
from strategy_v19_5_team_coin import (  # noqa: E402
    build_margin_panel,
    build_adjusted_predictions,
    run_walkforward,
    OOS_FIRST,
    OOS_LAST,
    ORIG_PRED,
)
from _factor_kline_panel import build_pit_panel_on_pred_axis  # noqa: E402

# Locked combo from 2026-05-25 v19.5 IS sweep — do NOT change
# c5_tc20_l020 = (lam_tc5=0.00, lam_tc20=0.20, lam_margin=0.00)
LOCKED_LAM_TC5 = 0.00
LOCKED_LAM_TC20 = 0.20
LOCKED_LAM_MARGIN = 0.00
LOCKED_LABEL = "c5_tc20_l020"

OUT_STATS_CSV = ROOT / "examples" / "v19_5_oos_phase2_v3_stats.csv"
OUT_SUMMARY = ROOT / "examples" / "v19_5_oos_phase2_v3_summary.txt"


def main() -> int:
    print("=" * 70)
    print("v19.5 team_coin OOS-only retest on Phase 2 v3 cache")
    print("=" * 70)
    print(f"OOS    : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"locked : {LOCKED_LABEL} "
          f"λ=(tc5={LOCKED_LAM_TC5}, tc20={LOCKED_LAM_TC20}, "
          f"margin={LOCKED_LAM_MARGIN})")
    print("strict OOS — no IS re-sweep, no λ re-optimization")
    print()

    print("[step 1] build kline PIT panel (team_coin_5d/20d)...")
    tc_panel = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["team_coin_5d", "team_coin_20d"],
    )

    # lam_margin=0 → margin merge branch is skipped inside build_adjusted_predictions
    # (see strategy_v19_5_team_coin.py lines 121-127), but the helper signature
    # requires the panel object. Build it once — small cost, keeps API parity.
    print("\n[step 1b] build margin PIT panel (unused at locked lam_margin=0)...")
    margin_panel = build_margin_panel()

    print("\n[step 2] build adjusted predictions with locked λ ...")
    build_adjusted_predictions(
        tc_panel, margin_panel,
        LOCKED_LAM_TC5, LOCKED_LAM_TC20, LOCKED_LAM_MARGIN,
        LOCKED_LABEL,
    )

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}) ...")
    stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_phase2_v3")

    months_df = stats.pop("months_df")
    months_df.to_csv(OUT_STATS_CSV, index=False)
    print(f"\n[saved] {OUT_STATS_CSV}")

    summary_lines = [
        "=" * 70,
        "v19.5 team_coin OOS Phase 2 v3 retest summary",
        "=" * 70,
        f"locked config: {LOCKED_LABEL} "
        f"λ=(tc5={LOCKED_LAM_TC5}, tc20={LOCKED_LAM_TC20}, "
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
        "  baseline train24      : Calmar=0.86  Sharpe=0.81  ann=29.12%",
        "  v19.4 (m5+m20 0.10)   : Calmar=1.28  Sharpe=0.94  ann=40.19%",
        "  v19.5 (tc20 0.20)     : Calmar=0.56  (abort)",
        "  v19.6 (amp 0.30)      : Calmar=0.58  Sharpe=0.67  ann=21.78%",
        "",
        "Phase 2 v3 retrained cache (60m OOS):",
        "  baseline train24      : Calmar=0.77",
        "  v19.4 (m5+m20 0.10)   : Calmar=0.62",
        "  v19.6 (amp 0.30) PROD : Calmar=1.29",
        f"  v19.5 (tc20 0.20)     : Calmar={stats['calmar']}  <-- this task",
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
