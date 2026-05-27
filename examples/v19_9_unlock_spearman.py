"""Spearman corr: v19.9 unlock factor vs v19.6 amp_imb_20d (mechanism check).

期望: 两因子机制不同 (unlock = 事件型 forward 解禁压力; amp_imb_20d = 价格振幅
不对称), 横截面 rho 应低.

Pairs:
    unlock_pct_next_60 / unlock_pct_next_20 / combo_neg_pct / unlock_imminent_20
    vs
    amp_imb_20d (v19.6 best)

Run:
  .venv/bin/python examples/v19_9_unlock_spearman.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from _factor_kline_panel import build_pit_panel_on_pred_axis  # noqa: E402
from strategy_v19_9_unlock import build_unlock_panel  # noqa: E402

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
OUT_CSV = ROOT / "examples" / "v19_9_unlock_amp_spearman.csv"


def main() -> int:
    print("Building unlock panel...", flush=True)
    unlock_panel = build_unlock_panel(windows_days=(20, 60))

    print("\nBuilding amp_imb panel (kline)...", flush=True)
    z_kline = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["amp_imb_20d"],
    )

    pred = pd.read_parquet(
        ORIG_PRED, columns=["datetime", "instrument", "month"],
    )
    df = pred.merge(
        unlock_panel, on=["datetime", "instrument"], how="left",
    ).merge(
        z_kline, on=["datetime", "instrument"], how="left",
    )
    df = df.rename(columns={
        "z_unlock_pct_next_20": "u20",
        "z_unlock_pct_next_60": "u60",
        "z_combo_neg_pct": "ucombo",
        "z_unlock_imminent_20": "uimm20",
        "z_amp_imb_20d": "a20",
    })

    pairs = [
        ("u20",    "a20"),
        ("u60",    "a20"),
        ("ucombo", "a20"),
        ("uimm20", "a20"),
        ("u20",    "u60"),
    ]

    print("\nComputing monthly cross-section Spearman...", flush=True)
    print(f"month count: {df['month'].nunique()}", flush=True)
    rows = []
    for mo, sub in df.groupby("month", sort=True):
        rec = {"month": mo, "n_stocks": sub["instrument"].nunique()}
        for a, b in pairs:
            s = sub[[a, b]].dropna()
            if len(s) < 30 or s[a].nunique() < 2 or s[b].nunique() < 2:
                rec[f"rho_{a}_{b}"] = np.nan
            else:
                rec[f"rho_{a}_{b}"] = s[a].corr(s[b], method="spearman")
        rows.append(rec)
    mdf = pd.DataFrame(rows)
    mdf.to_csv(OUT_CSV, index=False)
    print(f"saved → {OUT_CSV}", flush=True)

    print("\n=== Monthly Spearman rho — IS (2017-01 ~ 2020-12) ===")
    is_mask = (mdf["month"] >= "2017-01") & (mdf["month"] <= "2020-12")
    is_df = mdf[is_mask]
    for a, b in pairs:
        col = f"rho_{a}_{b}"
        v = is_df[col].dropna()
        if len(v) == 0:
            print(f"  {a:7s} vs {b:5s}: no data")
            continue
        print(f"  {a:7s} vs {b:5s}: mean={v.mean():+.3f}  "
              f"std={v.std():.3f}  median={v.median():+.3f}  n={len(v)}")

    print("\n=== Monthly Spearman rho — OOS (2021-05 ~ 2026-04) ===")
    oos_mask = (mdf["month"] >= "2021-05") & (mdf["month"] <= "2026-04")
    oos_df = mdf[oos_mask]
    for a, b in pairs:
        col = f"rho_{a}_{b}"
        v = oos_df[col].dropna()
        if len(v) == 0:
            print(f"  {a:7s} vs {b:5s}: no data")
            continue
        print(f"  {a:7s} vs {b:5s}: mean={v.mean():+.3f}  "
              f"std={v.std():.3f}  median={v.median():+.3f}  n={len(v)}")

    print("\n=== Verdict ===")
    for a, b in pairs:
        col = f"rho_{a}_{b}"
        rho = mdf[col].dropna().mean()
        if abs(rho) > 0.7:
            note = "HIGHLY REDUNDANT"
        elif abs(rho) > 0.4:
            note = "moderately correlated"
        elif abs(rho) > 0.2:
            note = "weakly correlated"
        else:
            note = "INDEPENDENT (low corr, 机制不同)"
        print(f"  {a:7s} vs {b:5s}: rho={rho:+.3f} → {note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
