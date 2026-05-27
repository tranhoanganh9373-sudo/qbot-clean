"""v19.7 Sidecar — train24 + amp_imb_20d + margin_5d_chg STACKED sidecar.

背景:
    v19.4 margin (m5_chg λ=0.10 + m20_chg λ=0.10):   OOS Calmar 0.61
    v19.6 amplitude (a20 λ=0.30):                     OOS Calmar 0.79
    Spearman 3x3 |rho| < 0.10 → 三因子正交; 理论可叠加.

试 stacked v19.7 = amp + margin 双因子 sidecar.

公式:
    final_score = z(train24_pred)
                  - λ_a20 × z(amp_imb_20d)     # 振幅反向
                  - λ_m5  × z(margin_5d_chg)   # margin 反向

严格 OOS 协议 (CLAUDE.md rule 5 — CRITICAL):
    - IS 期 2017-01~2020-12 (48 月) 重新 sweep 9 combos
    - 锁定 IS 最佳 Calmar (λ_a20, λ_m5) → OOS 2021-05~2026-04 跑一次
    - 不允许复用 v19.4 / v19.6 单因子已锁 λ (那是 OOS-leaked 选优结果)
    - 不允许看 OOS 结果回头改 λ
    - 不允许看 OOS 后跑别的 combo

Run:
  .venv/bin/python examples/strategy_v19_7_stacked.py
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
ADJ_PRED = ROOT / "data_cache" / "v19_7_stacked_predictions.parquet"
OUT_IS_GRID = ROOT / "examples" / "v19_7_stacked_is_grid.csv"
OUT_OOS_STATS = ROOT / "examples" / "v19_7_stacked_oos_stats.csv"

IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# 9 IS combos: (label, lam_a20, lam_m5)
# 3x3 grid: λ_a20 ∈ {0.10, 0.20, 0.30} × λ_m5 ∈ {0.05, 0.10, 0.20}
IS_COMBOS = [
    ("c1_a010_m005", 0.10, 0.05),
    ("c2_a010_m010", 0.10, 0.10),
    ("c3_a010_m020", 0.10, 0.20),
    ("c4_a020_m005", 0.20, 0.05),
    ("c5_a020_m010", 0.20, 0.10),
    ("c6_a020_m020", 0.20, 0.20),
    ("c7_a030_m005", 0.30, 0.05),
    ("c8_a030_m010", 0.30, 0.10),
    ("c9_a030_m020", 0.30, 0.20),
]


def build_margin_panel() -> pd.DataFrame:
    """复用 v19.4/v19.6 的 margin PIT panel 构造 (margin_5d_chg ICIR -0.97)."""
    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))

    print("[margin] loading margin parquet...", flush=True)
    m = pd.read_parquet(MARGIN_PARQUET)
    m["code"] = m["code"].astype(str).str.zfill(6)
    m["date"] = pd.to_datetime(m["date"])
    m = m[["code", "date", "margin_5d_chg"]]

    parts = []
    for code, sub in m.groupby("code", sort=False):
        sub = sub.sort_values("date")
        dates_arr = sub["date"].values
        m5 = sub["margin_5d_chg"].values
        idx = np.searchsorted(dates_arr, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe_idx = np.clip(idx, 0, len(sub) - 1)
        m5_v = np.where(valid, m5[safe_idx], np.nan)
        parts.append(pd.DataFrame({
            "datetime": pred_dt,
            "code": code,
            "margin_5d_chg": m5_v,
        }))
    margin_panel = pd.concat(parts, ignore_index=True) if parts else (
        pd.DataFrame(columns=["datetime", "code", "margin_5d_chg"])
    )
    print(f"[margin] panel rows: {len(margin_panel):,}", flush=True)

    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    out = pred_axis.merge(margin_panel, on=["datetime", "code"], how="left")
    out["z_margin_5d_chg"] = out.groupby("datetime")[
        "margin_5d_chg"
    ].transform(_zscore_cs).fillna(0.0)
    return out[["datetime", "instrument", "z_margin_5d_chg"]]


def build_adjusted_predictions(amp_panel: pd.DataFrame,
                               margin_panel: pd.DataFrame,
                               lam_a20: float, lam_m5: float,
                               label: str) -> Path:
    """final = z(pred) - lam_a20 * z(amp_imb_20d) - lam_m5 * z(margin_5d_chg)."""
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(amp_panel, on=["datetime", "instrument"], how="left")
    merged = merged.merge(margin_panel, on=["datetime", "instrument"], how="left")
    merged["z_amp_imb_20d"] = merged["z_amp_imb_20d"].fillna(0.0)
    merged["z_margin_5d_chg"] = merged["z_margin_5d_chg"].fillna(0.0)
    merged["final_score"] = (
        merged["z_pred"]
        - lam_a20 * merged["z_amp_imb_20d"]
        - lam_m5  * merged["z_margin_5d_chg"]
    )
    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
    print(f"  [adj] {label} λ=(a20={lam_a20}, m5={lam_m5}) "
          f"rows={len(out):,}", flush=True)
    return ADJ_PRED


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


def run_walkforward(first_month: str, last_month: str, tag: str) -> dict:
    """跑指定 [first, last] 区间的 walk-forward backtest."""
    import qlib  # noqa: F401
    from qlib.constant import REG_CN

    import strategy_v17_dens_grid as v17

    QLIB_DIR = str(ROOT / "data_cache" / "qlib_baidu")
    if not getattr(run_walkforward, "_qlib_initialized", False):
        qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
        run_walkforward._qlib_initialized = True
        run_walkforward._proxy = v17.build_market_proxy()

    v17.PRED_CACHE = ADJ_PRED
    v17._pred_disk_df = None
    v17._pred_cache.clear()
    v17.MARKET = "csi300"
    v17.TRAIN_MONTHS = 24
    v17.K_NORMAL = 8
    v17.DROP_NORMAL = 2
    v17.PORTFOLIO_VALUE = 5e4
    v17.STOP_LOSS_PCT = 0.0
    v17.VOL_TARGET_ANN = 0.0

    first = datetime.strptime(first_month + "-01", "%Y-%m-%d")
    last = datetime.strptime(last_month + "-01", "%Y-%m-%d")
    months = []
    cur = first
    while cur <= last:
        months.append(cur)
        cur += relativedelta(months=1)

    rows = []
    for i, m in enumerate(months, 1):
        try:
            res = v17.realistic_window(
                m, run_walkforward._proxy, with_regime=False,
            )
            res["month"] = m.strftime("%Y-%m")
            rows.append(res)
            if i % 12 == 0 or i == len(months) or i == 1:
                print(f"    {i:3d}/{len(months)} {res['month']}: "
                      f"abs_ret={res['abs_ret_%']:+6.2f}%  "
                      f"picks={res['avg_picks']:.1f}", flush=True)
        except Exception as e:
            print(f"    {i:3d}/{len(months)} {m.strftime('%Y-%m')} FAIL: "
                  f"{str(e)[:120]}", flush=True)
            rows.append({"month": m.strftime("%Y-%m"),
                         "abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                         "regime_days": "", "n_skipped_limit": 0,
                         "n_stop_loss": 0})

    df = pd.DataFrame(rows)
    stats = _annualize(df["abs_ret_%"])
    stats["avg_picks"] = round(df["avg_picks"].mean(), 2)
    stats["months_df"] = df
    return stats


def main() -> int:
    print("=" * 70)
    print("Phase 4 Sidecar v19.7 — train24 + amp_imb_20d + margin_5d_chg STACKED")
    print("=" * 70)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST} (48 months)")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"combos: {len(IS_COMBOS)}")
    print()

    print("[step 1] build PIT panels (amp_imb_20d + margin_5d_chg)...")
    amp_panel = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["amp_imb_20d"],
    )
    margin_panel = build_margin_panel()

    print("\n[step 2] IS sweep 9 combos (2017-01 ~ 2020-12, 48 months)")
    is_rows = []
    for label, lam_a20, lam_m5 in IS_COMBOS:
        print(f"\n  --- {label}  λ=(a20={lam_a20}, m5={lam_m5}) ---")
        build_adjusted_predictions(
            amp_panel, margin_panel, lam_a20, lam_m5, label,
        )
        stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{label}")
        row = {
            "combo": label,
            "lam_a20": lam_a20,
            "lam_m5": lam_m5,
            "cum_%": stats["cum_%"],
            "ann_%": stats["ann_%"],
            "sharpe": stats["sharpe"],
            "mdd_%": stats["mdd_%"],
            "calmar": stats["calmar"],
            "win_%": stats["win_%"],
            "avg_picks": stats["avg_picks"],
            "n_months": stats["n"],
        }
        is_rows.append(row)
        print(f"    >> IS Calmar={stats['calmar']} Sharpe={stats['sharpe']} "
              f"ann={stats['ann_%']}% MDD={stats['mdd_%']}%")

    is_df = pd.DataFrame(is_rows)
    is_df = is_df.sort_values("calmar", ascending=False).reset_index(drop=True)
    is_df.to_csv(OUT_IS_GRID, index=False)
    print(f"\n[saved] {OUT_IS_GRID}")
    print("\n=== IS 9-combo Calmar table (sorted desc) ===")
    print(is_df.to_string(index=False))

    best = is_df.iloc[0]
    best_label = best["combo"]
    best_a20 = float(best["lam_a20"])
    best_m5 = float(best["lam_m5"])
    is_best_calmar = float(best["calmar"])
    print(f"\n[lock] best IS: {best_label}  "
          f"λ=(a20={best_a20}, m5={best_m5})  "
          f"IS Calmar={is_best_calmar}")

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    print(f"  locked config: {best_label} λ=(a20={best_a20}, m5={best_m5})")
    build_adjusted_predictions(
        amp_panel, margin_panel, best_a20, best_m5, best_label,
    )
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_locked")
    oos_df = oos_stats.pop("months_df")
    oos_df.to_csv(OUT_OOS_STATS, index=False)
    print(f"\n[saved] {OUT_OOS_STATS}")

    print("\n" + "=" * 70)
    print("=== FINAL COMPARISON v19.7 STACKED ===")
    print("=" * 70)
    print("baseline train24 (Phase 2 clean_phase2 OOS, 60m):")
    print("   Calmar=0.42  Sharpe=0.68  ann=12.45%  MDD=-29.86%")
    print("\nv19.4 (margin m5+m20):")
    print("   Calmar=0.61  Sharpe=0.76  ann=12.86%  MDD=-21.23%")
    print("\nv19.6 (amp_imb_20d):")
    print("   Calmar=0.79  Sharpe=0.71  ann=14.54%  MDD=-18.51%")
    print(f"\nv19.7 (a20 + m5 STACKED) [{best_label}]:")
    print(f"   Calmar={oos_stats['calmar']}  Sharpe={oos_stats['sharpe']}  "
          f"ann={oos_stats['ann_%']}%  MDD={oos_stats['mdd_%']}%  "
          f"cum={oos_stats['cum_%']}%  win%={oos_stats['win_%']}")

    v196_calmar = 0.79
    rel_imp = (oos_stats["calmar"] - v196_calmar) / v196_calmar * 100
    print(f"\nrelative Calmar Δ vs v19.6: {rel_imp:+.1f}%")

    # stacked overfit detection
    is_oos_gap = is_best_calmar - oos_stats["calmar"]
    print(f"\n[overfit check] IS Calmar={is_best_calmar:.2f}  "
          f"OOS Calmar={oos_stats['calmar']:.2f}  gap={is_oos_gap:.2f}")
    if is_best_calmar > 1.2 and oos_stats["calmar"] < v196_calmar:
        print("  ⚠ STACKED OVERFIT detected: IS 远高但 OOS 比 v19.6 (单因子) 低.")
        print("    双因子在 IS 上找到了不可复制的 sweet spot.")

    if oos_stats["calmar"] > v196_calmar:
        margin_imp = (oos_stats["calmar"] - v196_calmar) / v196_calmar
        if margin_imp < 0.05:
            print(f"\n[verdict] OOS Calmar 仅微高 v19.6 ({margin_imp*100:.1f}%) "
                  "→ 边际增益不值得 production 复杂度, 推荐 v19.6")
        else:
            print(f"\n[verdict] OOS Calmar > v19.6 ({margin_imp*100:+.1f}%) "
                  "→ 推荐升级 v19.7 production")
    else:
        print("\n[verdict] OOS Calmar <= v19.6 → 接受 v19.6, 放弃 stacked sidecar")

    return 0


if __name__ == "__main__":
    sys.exit(main())
