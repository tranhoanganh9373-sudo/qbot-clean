"""Phase 2 v2 retest — v19.6 (a20) + v19.4 (m5+m20) sidecar OOS only.

严格 OOS 协议:
  - 锁定原 IS-winning λ (不重 IS sweep, CLAUDE.md rule 5)
  - 用 v3 baidu_kline + Phase 2 v2 retrain 后的 train24 predictions
  - 只跑 OOS 60 month (2021-05 ~ 2026-04)

v19.6 locked: c6_a20_l030  λ=(a5=0.0, a20=0.30, m=0.0)
v19.4 locked: c4_m5m20_l020 λ=(m5=0.10, m20=0.10, dt=0.0)

输出:
  examples/v19_6_phase2_v2_oos_stats.csv  — 60m monthly OOS returns (v19.6)
  examples/v19_4_phase2_v2_oos_stats.csv  — 60m monthly OOS returns (v19.4)
  Final comparison table printed to stdout

Run:
  .venv/bin/python examples/v19_phase2_v2_oos_retest.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from _factor_kline_panel import (  # noqa: E402
    build_pit_panel_on_pred_axis, _zscore_cs, _instrument_to_code6,
)

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
MARGIN_PARQUET = ROOT / "data_cache" / "csi300_margin_14yr.parquet"

V196_ADJ_PRED = ROOT / "data_cache" / "v19_6_phase2_v2_predictions.parquet"
V194_ADJ_PRED = ROOT / "data_cache" / "v19_4_phase2_v2_predictions.parquet"

V196_OUT = ROOT / "examples" / "v19_6_phase2_v2_oos_stats.csv"
V194_OUT = ROOT / "examples" / "v19_4_phase2_v2_oos_stats.csv"
BASELINE_OUT = ROOT / "examples" / "baseline_phase2_v2_oos_stats.csv"

OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# Locked λ from IS sweep (DO NOT modify — strict OOS protocol)
V196_LAM_5 = 0.0
V196_LAM_20 = 0.30
V196_LAM_M = 0.0

V194_LAM_M5 = 0.10
V194_LAM_M20 = 0.10
V194_LAM_DT = 0.0  # dragon_tiger not used in c4_m5m20


def _annualize(returns: pd.Series) -> dict:
    cum = (1 + returns / 100).prod() - 1
    n = len(returns)
    years = n / 12
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else 0
    mean = (returns / 100).mean()
    std = (returns / 100).std()
    sharpe = mean / std * np.sqrt(12) if std > 0 else 0
    cs = (1 + returns / 100).cumprod()
    peak = cs.cummax()
    mdd = ((cs - peak) / peak).min()
    calmar = (ann * 100) / abs(mdd * 100) if mdd < 0 else 0.0
    return {
        "cum_%": round(cum * 100, 2),
        "ann_%": round(ann * 100, 2),
        "sharpe": round(sharpe, 2),
        "mdd_%": round(mdd * 100, 2),
        "win_%": round((returns > 0).mean() * 100, 2),
        "calmar": round(calmar, 2),
        "n": n,
    }


def build_margin_panel_m5m20() -> pd.DataFrame:
    """Margin panel with BOTH margin_5d_chg and margin_20d_chg (PIT on pred axis)."""
    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))

    print("[margin] loading margin parquet...", flush=True)
    m = pd.read_parquet(MARGIN_PARQUET)
    m["code"] = m["code"].astype(str).str.zfill(6)
    m["date"] = pd.to_datetime(m["date"])
    m = m[["code", "date", "margin_5d_chg", "margin_20d_chg"]]

    parts = []
    for code, sub in m.groupby("code", sort=False):
        sub = sub.sort_values("date")
        dates_arr = sub["date"].values
        m5 = sub["margin_5d_chg"].values
        m20 = sub["margin_20d_chg"].values
        idx = np.searchsorted(dates_arr, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe_idx = np.clip(idx, 0, len(sub) - 1)
        m5_v = np.where(valid, m5[safe_idx], np.nan)
        m20_v = np.where(valid, m20[safe_idx], np.nan)
        parts.append(pd.DataFrame({
            "datetime": pred_dt,
            "code": code,
            "margin_5d_chg": m5_v,
            "margin_20d_chg": m20_v,
        }))
    margin_panel = pd.concat(parts, ignore_index=True) if parts else (
        pd.DataFrame(columns=["datetime", "code",
                              "margin_5d_chg", "margin_20d_chg"])
    )
    print(f"[margin] panel rows: {len(margin_panel):,}", flush=True)

    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    out = pred_axis.merge(margin_panel, on=["datetime", "code"], how="left")
    out["z_margin_5d_chg"] = out.groupby("datetime")[
        "margin_5d_chg"
    ].transform(_zscore_cs).fillna(0.0)
    out["z_margin_20d_chg"] = out.groupby("datetime")[
        "margin_20d_chg"
    ].transform(_zscore_cs).fillna(0.0)

    n_m5 = (out["margin_5d_chg"].notna()).sum()
    n_m20 = (out["margin_20d_chg"].notna()).sum()
    print(f"[margin] coverage: m5={n_m5/len(out)*100:.1f}% "
          f"m20={n_m20/len(out)*100:.1f}%", flush=True)
    return out[["datetime", "instrument",
                "z_margin_5d_chg", "z_margin_20d_chg"]]


def build_v196_adjusted(amp_panel: pd.DataFrame) -> Path:
    """v19.6: final = z(pred) - 0.30 * z(amp_imb_20d)"""
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(amp_panel, on=["datetime", "instrument"], how="left")
    merged[["z_amp_imb_5d", "z_amp_imb_20d"]] = merged[
        ["z_amp_imb_5d", "z_amp_imb_20d"]
    ].fillna(0.0)
    merged["final_score"] = (
        merged["z_pred"]
        - V196_LAM_5 * merged["z_amp_imb_5d"]
        - V196_LAM_20 * merged["z_amp_imb_20d"]
    )
    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(V196_ADJ_PRED, index=False)
    cov = (merged["z_amp_imb_20d"] != 0).mean() * 100
    print(f"[v19.6] adj_pred rows={len(out):,}  "
          f"z_amp_imb_20d active={cov:.1f}% (was ~3.4% pre-v3)",
          flush=True)
    return V196_ADJ_PRED


def build_v194_adjusted(margin_panel: pd.DataFrame) -> Path:
    """v19.4: final = z(pred) - 0.10 * z(margin_5d_chg) - 0.10 * z(margin_20d_chg)"""
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(margin_panel, on=["datetime", "instrument"], how="left")
    merged[["z_margin_5d_chg", "z_margin_20d_chg"]] = merged[
        ["z_margin_5d_chg", "z_margin_20d_chg"]
    ].fillna(0.0)
    merged["final_score"] = (
        merged["z_pred"]
        - V194_LAM_M5 * merged["z_margin_5d_chg"]
        - V194_LAM_M20 * merged["z_margin_20d_chg"]
    )
    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(V194_ADJ_PRED, index=False)
    print(f"[v19.4] adj_pred rows={len(out):,}", flush=True)
    return V194_ADJ_PRED


def run_oos(adj_path: Path, tag: str) -> tuple[dict, pd.DataFrame]:
    """Run OOS walk-forward [OOS_FIRST, OOS_LAST] using adj_path as PRED_CACHE."""
    import qlib  # noqa: F401
    from qlib.constant import REG_CN

    import strategy_v17_dens_grid as v17

    QLIB_DIR = str(ROOT / "data_cache" / "qlib_baidu")
    if not getattr(run_oos, "_qlib_initialized", False):
        qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
        run_oos._qlib_initialized = True
        run_oos._proxy = v17.build_market_proxy()

    v17.PRED_CACHE = adj_path
    v17._pred_disk_df = None
    v17._pred_cache.clear()
    v17.MARKET = "csi300"
    v17.TRAIN_MONTHS = 24
    v17.K_NORMAL = 8
    v17.DROP_NORMAL = 2
    v17.PORTFOLIO_VALUE = 5e4
    v17.STOP_LOSS_PCT = 0.0
    v17.VOL_TARGET_ANN = 0.0

    first = datetime.strptime(OOS_FIRST + "-01", "%Y-%m-%d")
    last = datetime.strptime(OOS_LAST + "-01", "%Y-%m-%d")
    months = []
    cur = first
    while cur <= last:
        months.append(cur)
        cur += relativedelta(months=1)

    rows = []
    for i, m in enumerate(months, 1):
        try:
            res = v17.realistic_window(
                m, run_oos._proxy, with_regime=False,
            )
            res["month"] = m.strftime("%Y-%m")
            rows.append(res)
            if i % 12 == 0 or i == len(months) or i == 1:
                print(f"    [{tag}] {i:3d}/{len(months)} {res['month']}: "
                      f"abs_ret={res['abs_ret_%']:+6.2f}%  "
                      f"picks={res['avg_picks']:.1f}", flush=True)
        except Exception as e:
            print(f"    [{tag}] {i:3d}/{len(months)} {m.strftime('%Y-%m')} "
                  f"FAIL: {str(e)[:120]}", flush=True)
            rows.append({"month": m.strftime("%Y-%m"),
                         "abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                         "regime_days": "", "n_skipped_limit": 0,
                         "n_stop_loss": 0})

    df = pd.DataFrame(rows)
    stats = _annualize(df["abs_ret_%"])
    stats["avg_picks"] = round(df["avg_picks"].mean(), 2)
    return stats, df


def main() -> int:
    print("=" * 72)
    print("Phase 2 v2 retest — v19.6 (a20) + v19.4 (m5+m20) OOS only")
    print("=" * 72)
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"v19.6 locked: a20 λ={V196_LAM_20}")
    print(f"v19.4 locked: m5 λ={V194_LAM_M5} + m20 λ={V194_LAM_M20}")
    print()

    print("[step 1] build amplitude PIT panel from v3 baidu_kline...")
    amp_panel = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["amp_imb_5d", "amp_imb_20d"],
    )

    print("\n[step 2] build margin PIT panel...")
    margin_panel = build_margin_panel_m5m20()

    print("\n[step 3] baseline OOS (just train24 predictions, no sidecar)")
    base_stats, base_df = run_oos(ORIG_PRED, "baseline")
    base_df.to_csv(BASELINE_OUT, index=False)
    print(f"[saved] {BASELINE_OUT}")
    print(f"  baseline OOS: Calmar={base_stats['calmar']} "
          f"Sharpe={base_stats['sharpe']} ann={base_stats['ann_%']}% "
          f"MDD={base_stats['mdd_%']}% cum={base_stats['cum_%']}%")

    print("\n[step 4] v19.6 build adjusted predictions (a20 λ=0.30)")
    build_v196_adjusted(amp_panel)

    print("\n[step 5] v19.6 OOS single run")
    v196_stats, v196_df = run_oos(V196_ADJ_PRED, "v19.6")
    v196_df.to_csv(V196_OUT, index=False)
    print(f"[saved] {V196_OUT}")
    print(f"  v19.6 OOS: Calmar={v196_stats['calmar']} "
          f"Sharpe={v196_stats['sharpe']} ann={v196_stats['ann_%']}% "
          f"MDD={v196_stats['mdd_%']}% cum={v196_stats['cum_%']}%")

    print("\n[step 6] v19.4 build adjusted predictions (m5 λ=0.10 + m20 λ=0.10)")
    build_v194_adjusted(margin_panel)

    print("\n[step 7] v19.4 OOS single run")
    v194_stats, v194_df = run_oos(V194_ADJ_PRED, "v19.4")
    v194_df.to_csv(V194_OUT, index=False)
    print(f"[saved] {V194_OUT}")
    print(f"  v19.4 OOS: Calmar={v194_stats['calmar']} "
          f"Sharpe={v194_stats['sharpe']} ann={v194_stats['ann_%']}% "
          f"MDD={v194_stats['mdd_%']}% cum={v194_stats['cum_%']}%")

    print("\n" + "=" * 72)
    print("=== FINAL COMPARISON (Phase 2 v2, v3 clean baidu_kline) ===")
    print("=" * 72)
    print(f"{'config':<25} {'prev OOS':>12} {'v3 OOS':>12} {'delta':>10}")
    print("-" * 72)

    def fmt_row(name, prev, cur):
        delta = cur - prev
        return f"{name:<25} {prev:>12.2f} {cur:>12.2f} {delta:>+10.2f}"

    print(fmt_row("baseline Calmar", 0.42, base_stats["calmar"]))
    print(fmt_row("v19.4 (m5+m20) Calmar", 0.61, v194_stats["calmar"]))
    print(fmt_row("v19.6 (a20) Calmar", 0.79, v196_stats["calmar"]))
    print()
    print(fmt_row("baseline Sharpe", 0.68, base_stats["sharpe"]))
    print(fmt_row("v19.4 Sharpe", 0.76, v194_stats["sharpe"]))
    print(fmt_row("v19.6 Sharpe", 0.85, v196_stats["sharpe"]))
    print()
    print(fmt_row("baseline ann%", 12.45, base_stats["ann_%"]))
    print(fmt_row("v19.4 ann%", 12.86, v194_stats["ann_%"]))
    print(fmt_row("v19.6 ann%", 16.00, v196_stats["ann_%"]))
    print()
    print(fmt_row("baseline MDD%", -29.86, base_stats["mdd_%"]))
    print(fmt_row("v19.4 MDD%", -21.23, v194_stats["mdd_%"]))
    print(fmt_row("v19.6 MDD%", -20.21, v196_stats["mdd_%"]))

    print("\n=== VERDICT ===")
    c = v196_stats["calmar"]
    if c >= 0.70:
        print(f"v19.6 Calmar={c} >= 0.70 -> production v19.6 ROBUST [OK]")
    elif c >= 0.50:
        print(f"v19.6 Calmar={c} in [0.50, 0.70) -> 边际 accept v19.6 with "
              "lowered expectations")
    else:
        print(f"v19.6 Calmar={c} < 0.50 -> RECONSIDER production")

    return 0


if __name__ == "__main__":
    sys.exit(main())
