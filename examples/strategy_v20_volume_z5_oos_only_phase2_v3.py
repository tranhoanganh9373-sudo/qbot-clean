"""v20 vol_z_5d OOS retest on Phase 2 v3 cache.

Strict OOS protocol: do NOT re-sweep λ. Use locked λ=0.30 sign=+1 from
Phase A (factor_ic_volume_zscore_is.csv / v20_volume_z5_is_grid.csv).

Formula:
    final = z(train24_pred) + 0.30 · z(vol_z_5d)

vol_z_5d is point-in-time backward-only:
    z5(code, T) = (vol[T] - mean(vol, T-5..T-1)) / std(vol, T-5..T-1)
clipped at 0.5%/99.5% then cross-sectional z per datetime.

Reference (v2 clean cache, locked λ=0.30):
    v20 vol_z_5d        : OOS Calmar=0.54 (abort)

This task tests the same locked λ on v3 (Phase 2 retrained) cache to verify
whether the v3 baseline upgrade changes the verdict relative to
v19.6 (single a20) which scored Calmar 1.29 on v3.

Inputs:
  data_cache/v17_dens_train24_predictions.parquet  (Phase 2 v3 retrained)
  data_cache/baidu_kline.parquet                   (v3 hfq cache, for vol_z_5d)

Output:
  examples/v20_volume_z5_oos_phase2_v3_stats.csv    (per-month abs_ret_%)
  examples/v20_volume_z5_oos_phase2_v3_summary.txt  (Calmar/Sharpe/ann/MDD)

Does NOT touch production / does NOT run IS sweep.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

# Reuse v20 vol_z_5d helpers from the original sidecar script.
from strategy_v20_volume_z5 import (  # noqa: E402
    build_vol_z5_panel,
    build_adjusted_predictions,
    run_walkforward,
    OOS_FIRST,
    OOS_LAST,
)

# Locked from Phase A — do NOT change
LOCKED_LAM = 0.30
LOCKED_SIGN = +1
LOCKED_LABEL = "v20_volz5_l030"

OUT_STATS_CSV = ROOT / "examples" / "v20_volume_z5_oos_phase2_v3_stats.csv"
OUT_SUMMARY = ROOT / "examples" / "v20_volume_z5_oos_phase2_v3_summary.txt"


def main() -> int:
    print("=" * 70)
    print("v20 vol_z_5d OOS-only retest on Phase 2 v3 cache")
    print("=" * 70)
    print(f"OOS    : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"locked : λ={LOCKED_LAM} sign={LOCKED_SIGN:+d} ({LOCKED_LABEL})")
    print("strict OOS — no IS re-sweep, no λ re-optimization")
    print()

    print("[step 1] build vol_z_5d PIT panel...")
    panel = build_vol_z5_panel()

    print("\n[step 2] build adjusted predictions with locked λ ...")
    build_adjusted_predictions(panel, LOCKED_LAM, LOCKED_LABEL)

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}) ...")
    stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_phase2_v3")

    months_df = stats.pop("months_df")
    months_df.to_csv(OUT_STATS_CSV, index=False)
    print(f"\n[saved] {OUT_STATS_CSV}")

    summary_lines = [
        "=" * 70,
        "v20 vol_z_5d OOS Phase 2 v3 retest summary",
        "=" * 70,
        f"locked config: λ={LOCKED_LAM} sign={LOCKED_SIGN:+d} ({LOCKED_LABEL})",
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
        "  v19.4 (m5+m20 0.10)   : Calmar=1.28",
        "  v19.6 (amp 0.30)      : Calmar=0.58",
        "  v20  (vol_z_5d 0.30)  : Calmar=0.54  (abort)",
        "",
        "Phase 2 v3 retrained cache (60m OOS):",
        "  baseline train24      : Calmar=0.77",
        "  v19.4 (m5+m20 0.10)   : Calmar=0.62",
        "  v19.6 (amp 0.30) PROD : Calmar=1.29",
        f"  v20  (vol_z_5d 0.30)  : Calmar={stats['calmar']}  <-- this task",
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
