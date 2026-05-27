"""3x3 Spearman corr matrix on CSI300 universe:
    team_coin_5d  (v19.5)  vs  amp_imb_20d  (v19.6)  vs  margin_5d_chg  (v19.4)

每月做横截面 Spearman, 然后报告 mean / std / median of monthly rho per pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

from _factor_kline_panel import (  # noqa: E402
    build_pit_panel_on_pred_axis, _instrument_to_code6,
)

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
MARGIN_PARQUET = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
OUT_CSV = ROOT / "examples" / "v19_signals_spearman_corr.csv"


def _build_margin_pit() -> pd.DataFrame:
    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))
    m = pd.read_parquet(MARGIN_PARQUET)
    m["code"] = m["code"].astype(str).str.zfill(6)
    m["date"] = pd.to_datetime(m["date"])
    m = m[["code", "date", "margin_5d_chg"]]
    parts = []
    for code, sub in m.groupby("code", sort=False):
        sub = sub.sort_values("date")
        d = sub["date"].values
        v = sub["margin_5d_chg"].values
        idx = np.searchsorted(d, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe = np.clip(idx, 0, len(sub) - 1)
        parts.append(pd.DataFrame({
            "datetime": pred_dt, "code": code,
            "margin_5d_chg": np.where(valid, v[safe], np.nan),
        }))
    mp = pd.concat(parts, ignore_index=True)
    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    out = pred_axis.merge(mp, on=["datetime", "code"], how="left")
    return out[["datetime", "instrument", "margin_5d_chg"]]


def main() -> int:
    print("Building factor panels...", flush=True)
    z_kline = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["team_coin_5d", "team_coin_20d",
                                "amp_imb_5d", "amp_imb_20d"],
    )
    margin = _build_margin_pit()

    print("Merging...", flush=True)
    pred = pd.read_parquet(ORIG_PRED)
    df = pred[["datetime", "instrument", "month"]].merge(
        z_kline, on=["datetime", "instrument"], how="left",
    ).merge(
        margin, on=["datetime", "instrument"], how="left",
    )
    # Spearman is rank-based — monotonic transforms (incl per-date z-score)
    # don't change rho, so z_kline values are fine.
    df = df.rename(columns={
        "z_team_coin_5d": "tc5",
        "z_team_coin_20d": "tc20",
        "z_amp_imb_5d": "a5",
        "z_amp_imb_20d": "a20",
        "margin_5d_chg": "m5",
    })
    pairs = [
        ("tc5",  "a20"),
        ("tc5",  "m5"),
        ("a20",  "m5"),
        ("tc20", "a20"),
        ("tc5",  "tc20"),
        ("a5",   "a20"),
        ("tc20", "m5"),
        ("a5",   "m5"),
    ]

    print("\nComputing monthly cross-section Spearman...", flush=True)
    print(f"month count: {df['month'].nunique()}")
    rows = []
    for mo, sub in df.groupby("month", sort=True):
        rec = {"month": mo, "n_stocks": sub["instrument"].nunique()}
        for a, b in pairs:
            s = sub[[a, b]].dropna()
            if len(s) < 30:
                rec[f"rho_{a}_{b}"] = np.nan
            else:
                rec[f"rho_{a}_{b}"] = s[a].corr(s[b], method="spearman")
        rows.append(rec)
    mdf = pd.DataFrame(rows)
    mdf.to_csv(OUT_CSV, index=False)
    print(f"saved → {OUT_CSV}")

    print("\n=== Monthly Spearman rho — IS (2017-01 ~ 2020-12) ===")
    is_mask = (mdf["month"] >= "2017-01") & (mdf["month"] <= "2020-12")
    is_df = mdf[is_mask]
    for a, b in pairs:
        col = f"rho_{a}_{b}"
        v = is_df[col].dropna()
        print(f"  {a:5s} vs {b:5s}: mean={v.mean():+.3f}  "
              f"std={v.std():.3f}  median={v.median():+.3f}  n={len(v)}")

    print("\n=== Monthly Spearman rho — OOS (2021-05 ~ 2026-04) ===")
    oos_mask = (mdf["month"] >= "2021-05") & (mdf["month"] <= "2026-04")
    oos_df = mdf[oos_mask]
    for a, b in pairs:
        col = f"rho_{a}_{b}"
        v = oos_df[col].dropna()
        print(f"  {a:5s} vs {b:5s}: mean={v.mean():+.3f}  "
              f"std={v.std():.3f}  median={v.median():+.3f}  n={len(v)}")

    print("\n=== Mean rho 3x3 matrix — primary signals "
          "(tc20=v19.5 best, a20=v19.6 best, m5=v19.4 best) ===")
    full_labels = ["tc20", "a20", "m5"]
    M = pd.DataFrame(index=full_labels, columns=full_labels, dtype=float)
    for a in full_labels:
        for b in full_labels:
            if a == b:
                M.loc[a, b] = 1.0
            else:
                col1 = f"rho_{a}_{b}"
                col2 = f"rho_{b}_{a}"
                if col1 in mdf.columns:
                    M.loc[a, b] = mdf[col1].mean()
                elif col2 in mdf.columns:
                    M.loc[a, b] = mdf[col2].mean()
                else:
                    M.loc[a, b] = np.nan
    print(M.round(3).to_string())

    print("\n=== Redundancy verdict (primary signals: tc20 / a20 / m5) ===")
    for a, b in [("tc20", "a20"), ("tc20", "m5"), ("a20", "m5")]:
        col1 = f"rho_{a}_{b}"
        col2 = f"rho_{b}_{a}"
        if col1 in mdf.columns:
            rho = mdf[col1].mean()
        else:
            rho = mdf[col2].mean()
        if abs(rho) > 0.7:
            note = "HIGHLY REDUNDANT — keep one only"
        elif abs(rho) > 0.4:
            note = "moderately correlated — caution stacking"
        else:
            note = "OK to stack (low corr)"
        print(f"  {a} vs {b}: rho={rho:+.3f} → {note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
