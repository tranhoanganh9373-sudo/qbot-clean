"""v19.6 Sidecar — train24 + 振幅隐藏结构因子 (amplitude imbalance) overlay.

参考: hugo2046/QuantsPlaybook 仓库 "振幅隐藏结构" 笔记本.

因子逻辑:
    amp     = (high - low) / prev_close
    amp_up  = max(0, close - prev_close) / prev_close * amp   # 涨势贡献部分
    amp_dn  = max(0, prev_close - close) / prev_close * amp   # 跌势贡献部分
    amp_imb_Nd = (sum(amp_up, Nd) - sum(amp_dn, Nd)) / sum(amp, Nd)

逻辑: 涨势振幅 > 跌势振幅 → 主动买盘强;反之亦然.
Mean reversion 期望: 涨势振幅过强 → 后续反转.

文献 ICIR -2.97 (反向使用 sign -1):

    final_score = z(train24_pred) - λ_5  * z(amp_imb_5d)
                                   - λ_20 * z(amp_imb_20d)
                                   - λ_m  * z(margin_5d_chg)   # combo 9 多因子叠加

9 IS combo sweep (2017-01 ~ 2020-12, 48 月). 锁定 best IS Calmar 后 OOS 跑一次
(2021-05 ~ 2026-04, 60 月). **看 OOS 不允许回头改 λ** (CLAUDE.md rule 5 严格 OOS 协议).

Run:
  .venv/bin/python examples/strategy_v19_6_amplitude.py
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
ADJ_PRED = ROOT / "data_cache" / "v19_6_amplitude_predictions.parquet"
OUT_IS_GRID = ROOT / "examples" / "v19_6_amplitude_is_grid.csv"
OUT_OOS_STATS = ROOT / "examples" / "v19_6_amplitude_oos_stats.csv"

IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# 9 IS combos: (label, lam_5, lam_20, lam_margin)
IS_COMBOS = [
    ("c1_a5_l010",        0.10, 0.00, 0.00),
    ("c2_a5_l020",        0.20, 0.00, 0.00),
    ("c3_a5_l030",        0.30, 0.00, 0.00),
    ("c4_a20_l010",       0.00, 0.10, 0.00),
    ("c5_a20_l020",       0.00, 0.20, 0.00),
    ("c6_a20_l030",       0.00, 0.30, 0.00),
    ("c7_a5a20_l020",     0.10, 0.10, 0.00),
    ("c8_a5a20_l030",     0.15, 0.15, 0.00),
    ("c9_a5margin_l020",  0.10, 0.00, 0.10),
]


def build_margin_panel() -> pd.DataFrame:
    """复用 v19.4 的 margin PIT panel 构造 (margin_5d_chg ICIR -0.97)."""
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
                               lam_5: float, lam_20: float, lam_m: float,
                               label: str) -> Path:
    """final = z(pred) - lam_5 * z(amp_imb_5d) - lam_20 * z(amp_imb_20d) - lam_m * z(margin)."""
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(amp_panel, on=["datetime", "instrument"], how="left")
    if lam_m > 0:
        merged = merged.merge(
            margin_panel, on=["datetime", "instrument"], how="left"
        )
        merged["z_margin_5d_chg"] = merged["z_margin_5d_chg"].fillna(0.0)
    else:
        merged["z_margin_5d_chg"] = 0.0

    merged[["z_amp_imb_5d", "z_amp_imb_20d"]] = merged[
        ["z_amp_imb_5d", "z_amp_imb_20d"]
    ].fillna(0.0)
    merged["final_score"] = (
        merged["z_pred"]
        - lam_5  * merged["z_amp_imb_5d"]
        - lam_20 * merged["z_amp_imb_20d"]
        - lam_m  * merged["z_margin_5d_chg"]
    )
    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
    print(f"  [adj] {label} λ=(a5={lam_5},a20={lam_20},m={lam_m}) "
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
    print("Phase 4 Sidecar v19.6 — train24 + 振幅隐藏结构因子")
    print("=" * 70)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST} (48 months)")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"combos: {len(IS_COMBOS)}")
    print()

    print("[step 1] build kline PIT panel (amp_imb_5d/20d)...")
    amp_panel = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["amp_imb_5d", "amp_imb_20d"],
    )
    margin_panel = build_margin_panel()

    print("\n[step 2] IS sweep 9 combos (2017-01 ~ 2020-12, 48 months)")
    is_rows = []
    for label, lam_5, lam_20, lam_m in IS_COMBOS:
        print(f"\n  --- {label}  λ=(a5={lam_5}, a20={lam_20}, m={lam_m}) ---")
        build_adjusted_predictions(
            amp_panel, margin_panel, lam_5, lam_20, lam_m, label,
        )
        stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{label}")
        row = {
            "combo": label,
            "lam_a5": lam_5,
            "lam_a20": lam_20,
            "lam_margin": lam_m,
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
    best_5 = float(best["lam_a5"])
    best_20 = float(best["lam_a20"])
    best_m = float(best["lam_margin"])
    print(f"\n[lock] best IS: {best_label}  "
          f"λ=(a5={best_5}, a20={best_20}, m={best_m})  "
          f"IS Calmar={best['calmar']}")

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    print(f"  locked config: {best_label} "
          f"λ=(a5={best_5}, a20={best_20}, m={best_m})")
    build_adjusted_predictions(
        amp_panel, margin_panel, best_5, best_20, best_m, best_label,
    )
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_locked")
    oos_df = oos_stats.pop("months_df")
    oos_df.to_csv(OUT_OOS_STATS, index=False)
    print(f"\n[saved] {OUT_OOS_STATS}")

    print("\n" + "=" * 70)
    print("=== FINAL COMPARISON v19.6 ===")
    print("=" * 70)
    print("baseline train24 (Phase 2 clean_phase2 OOS, 60m):")
    print("   Calmar=0.42  Sharpe=0.68  ann=12.45%  MDD=-29.86%  cum=79.84%")
    print("\nv19.4 (margin sidecar) OOS:")
    print("   Calmar=0.61  Sharpe=0.76  ann=12.86%  MDD=-21.23%  cum=83.07%")
    print(f"\nv19.6 (amplitude sidecar) OOS [{best_label}]:")
    print(f"   Calmar={oos_stats['calmar']}  Sharpe={oos_stats['sharpe']}  "
          f"ann={oos_stats['ann_%']}%  MDD={oos_stats['mdd_%']}%  "
          f"cum={oos_stats['cum_%']}%  win%={oos_stats['win_%']}")

    v194_calmar = 0.61
    rel_imp = (oos_stats["calmar"] - v194_calmar) / v194_calmar * 100
    print(f"\nrelative Calmar Δ vs v19.4: {rel_imp:+.1f}%")

    if oos_stats["calmar"] > v194_calmar:
        print("\n[verdict] OOS Calmar > v19.4 → 推荐升级 production v19.6")
    else:
        print("\n[verdict] OOS Calmar <= v19.4 → abort, amplitude sidecar 不优于 margin")

    return 0


if __name__ == "__main__":
    sys.exit(main())
