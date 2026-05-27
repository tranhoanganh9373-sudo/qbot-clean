"""Resume MACD Phase B run that was killed mid-variant-3.

State at kill:
  - Variant 1 (macd_hist) OOS saved → macd_hist_oos_equity.csv (60 months)
    Locked: sign=-1, λ=0.10, IS Calmar 6.11
  - Variant 2 (macd_cross_binary) OOS saved → macd_cross_binary_oos_equity.csv
    Locked: sign=+1, λ=0.10, IS Calmar 5.45
  - Variant 3 (macd_triple_cross) IS combo 1 done (Calmar 5.77), combos 2+3
    incomplete, OOS not run.

This script:
  1. Loads existing variant 1/2 OOS equity CSVs and re-derives stats
  2. Recomputes MACD factors
  3. Re-runs the 3 IS combos for variant 3 (idempotent, full sweep)
  4. Locks best IS, runs OOS
  5. Writes final summary CSV + markdown
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
sys.path.insert(0, str(ROOT / "examples"))

from macd_phase_b_oos_60m import (  # noqa: E402
    compute_macd_factors,
    run_variant_binary,
    _annualize,
    OOS_FIRST, OOS_LAST,
    BASELINE_CALMAR, V196_CALMAR, V194_CALMAR,
    OUT_DIR, OUT_SUMMARY_CSV, OUT_SUMMARY_MD,
    _verdict,
)


# Recovered from /Volumes/SSD/finance/claude_finance/logs/macd_phase_b.log
RECOVERED = {
    "macd_hist": {
        "locked_lambda": 0.10,
        "locked_sign": -1,
        "is_calmar": 6.11,
        "equity_csv": OUT_DIR / "macd_hist_oos_equity.csv",
    },
    "macd_cross_binary": {
        "locked_lambda": 0.10,
        "locked_sign": +1,
        "is_calmar": 5.45,
        "equity_csv": OUT_DIR / "macd_cross_binary_oos_equity.csv",
    },
}


def stats_from_equity_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    stats = _annualize(df["abs_ret_%"])
    stats["avg_picks"] = round(df["avg_picks"].mean(), 2)
    return stats


def main() -> int:
    t_wall = time.time()
    print("=" * 70)
    print("活法波段 MACD Phase B RESUME — finish variant 3 + summary")
    print("=" * 70)
    print()

    # ----- recovered variant 1 + 2 -----
    results = []
    for v, info in RECOVERED.items():
        s = stats_from_equity_csv(info["equity_csv"])
        print(f"[recovered] {v}  λ={info['locked_lambda']} "
              f"sign={info['locked_sign']:+d}  IS={info['is_calmar']}  "
              f"OOS Calmar={s['calmar']} Sharpe={s['sharpe']} "
              f"ann={s['ann_%']}% MDD={s['mdd_%']}%")
        results.append({
            "variant": v,
            "locked_lambda": info["locked_lambda"],
            "locked_sign": info["locked_sign"],
            "is_calmar": info["is_calmar"],
            "oos_calmar": s["calmar"],
            "oos_sharpe": s["sharpe"],
            "oos_mdd": s["mdd_%"],
            "oos_ann": s["ann_%"],
            "oos_cum": s["cum_%"],
            "oos_n": s["n"],
        })

    # ----- variant 3: re-run IS sweep + OOS -----
    print("\n[step] recomputing MACD factors for variant 3 ...")
    factors = compute_macd_factors(min_date="2014-01-01")

    print("\n[step] running variant 3 (macd_triple_cross) IS+OOS ...")
    r3 = run_variant_binary(factors, "macd_triple_cross", "3/3")
    results.append(r3)

    # ----- build final summary -----
    rows = []
    for r in results:
        rows.append({
            "variant": r["variant"],
            "locked_lambda": r["locked_lambda"],
            "locked_sign": r["locked_sign"],
            "is_calmar": r["is_calmar"],
            "oos_calmar": r["oos_calmar"],
            "oos_sharpe": r["oos_sharpe"],
            "oos_mdd": r["oos_mdd"],
            "oos_ann": r["oos_ann"],
            "oos_cum": r["oos_cum"],
            "vs_baseline_pct": round(
                (r["oos_calmar"] - BASELINE_CALMAR) / BASELINE_CALMAR * 100, 1
            ),
            "vs_v196_pct": round(
                (r["oos_calmar"] - V196_CALMAR) / V196_CALMAR * 100, 1
            ),
            "vs_v194_pct": round(
                (r["oos_calmar"] - V194_CALMAR) / V194_CALMAR * 100, 1
            ),
            "verdict": _verdict(r["oos_calmar"]),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\n[saved] {OUT_SUMMARY_CSV}")

    wall_min = (time.time() - t_wall) / 60
    md = ["# 活法波段 MACD Phase B 严格 OOS 60 月回测",
          "",
          f"- IS  : 2017-01 ~ 2020-12 (48 months, pred cache 起点对齐)",
          f"- OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)",
          f"- λ sweep: [0.10, 0.20, 0.30] (lock 后不允许 fine-tune)",
          f"- Top K=8 daily rebalance, v17_dens_train24 + Alpha158 + DEnsemble",
          f"- Wall time (resume run): **{wall_min:.1f} min**",
          "",
          "## Reference baselines (Phase 2 v3 cache, 60m OOS)",
          "",
          f"- baseline train24 : Calmar={BASELINE_CALMAR}",
          f"- v19.6 production (amp_imb_20d λ=0.30) : Calmar={V196_CALMAR}",
          f"- v19.4 shadow (margin_5d+20d λ=0.10)   : Calmar={V194_CALMAR}",
          "",
          "## Results",
          "",
          "| variant | sign | λ | IS Calmar | OOS Calmar | OOS Sharpe | "
          "OOS MDD% | OOS ann% | OOS cum% | vs baseline | vs v19.6 | vs v19.4 | "
          "verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['locked_sign']:+d} | "
            f"{r['locked_lambda']} | {r['is_calmar']} | "
            f"**{r['oos_calmar']}** | {r['oos_sharpe']} | "
            f"{r['oos_mdd']} | {r['oos_ann']} | {r['oos_cum']} | "
            f"{r['vs_baseline_pct']:+.1f}% | {r['vs_v196_pct']:+.1f}% | "
            f"{r['vs_v194_pct']:+.1f}% | {r['verdict']} |"
        )

    md.append("")
    md.append("## Verdict")
    md.append("")
    beat_196 = [r for r in rows if r["oos_calmar"] > V196_CALMAR]
    if beat_196:
        md.append("**有 variant 击败 v19.6 production**:")
        for r in beat_196:
            md.append(f"- `{r['variant']}` λ={r['locked_lambda']} "
                      f"sign={r['locked_sign']:+d}: OOS Calmar="
                      f"{r['oos_calmar']} vs v19.6={V196_CALMAR}")
    else:
        md.append("**所有 variant 均未击败 v19.6 production**.")
        md.append("维持 v19.6 (amp_imb_20d λ=0.30) 为 production.")
    md.append("")

    OUT_SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {OUT_SUMMARY_MD}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nWall time (resume): {wall_min:.1f} min")

    return 0


if __name__ == "__main__":
    sys.exit(main())
