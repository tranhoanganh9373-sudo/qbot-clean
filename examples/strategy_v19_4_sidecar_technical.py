"""v19.4 Sidecar — train24 + 技术因子 overlay (margin_5d_chg + margin_20d_chg + net_buy_pct_evt).

技术因子探索发现的强 ICIR(IS 2014-2020, 84 月):
    margin_5d_chg       ICIR = -0.972 (reverse)
    margin_20d_chg      ICIR = -0.739 (reverse)
    net_buy_pct_evt     ICIR = +0.641 (+ sign)

Sidecar 公式 (符号已按因子方向归并):
    z_tech = - λ_m5  * z(margin_5d_chg)
             - λ_m20 * z(margin_20d_chg)
             + λ_dt  * z(net_buy_pct_evt)
    final_score = z(train24_pred) + z_tech

9 IS combo sweep (2017-01 ~ 2020-12, 48 月):
    #1 margin_5d only          λ_m5=0.10
    #2 margin_5d only          λ_m5=0.20
    #3 margin_5d only          λ_m5=0.30
    #4 margin_5d+margin_20d 等权 λ=0.20
    #5 margin_5d+margin_20d 等权 λ=0.30
    #6 net_buy_pct_evt only    λ_dt=0.20
    #7 net_buy_pct_evt only    λ_dt=0.30
    #8 margin_5d+net_buy_pct_evt 等权 λ=0.20
    #9 margin_5d+margin_20d+net_buy_pct_evt 三因子等权 λ=0.30

锁定 best (combo, λ) 后 OOS 跑一次 (2021-05 ~ 2026-04, 60 月).
看 OOS 不允许回头改 λ.

数据缺口披露:
    dragon_tiger cache 仅含 IS 2014-2020, OOS 2021-2026 无龙虎榜事件.
    若 IS 选中含 net_buy_pct_evt 的 combo, OOS 上 z_dt = 0 退化为 baseline + margin-only.
    此为 sandbox 数据限制, 不是 leak.

不动任何 production 文件 / pred cache / margin cache / DT cache / v19_3 文件.

Run:
  .venv/bin/python examples/strategy_v19_4_sidecar_technical.py
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

from claude_finance import dragon_tiger as dt_mod  # noqa: E402

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
CSI300_CSV = ROOT / "data_cache" / "csi300_constituents.csv"
MARGIN_PARQUET = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
DT_DIR = ROOT / "data_cache" / "dragon_tiger"
ADJ_PRED = ROOT / "data_cache" / "v19_4_sidecar_technical_predictions.parquet"
OUT_IS_GRID = ROOT / "examples" / "v19_4_sidecar_technical_is_grid.csv"
OUT_OOS_STATS = ROOT / "examples" / "v19_4_sidecar_technical_oos_stats.csv"

IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# 9 IS combo: each entry is (label, lam_m5, lam_m20, lam_dt)
# 符号: margin_* ICIR 负 → sidecar 中减; net_buy_pct_evt ICIR 正 → 加.
# 等权 λ 表示总 λ; 二因子等权 → 每因子 λ/2; 三因子等权 → 每因子 λ/3.
IS_COMBOS = [
    ("c1_m5_l010",       0.10, 0.00, 0.00),
    ("c2_m5_l020",       0.20, 0.00, 0.00),
    ("c3_m5_l030",       0.30, 0.00, 0.00),
    ("c4_m5m20_l020",    0.10, 0.10, 0.00),
    ("c5_m5m20_l030",    0.15, 0.15, 0.00),
    ("c6_dt_l020",       0.00, 0.00, 0.20),
    ("c7_dt_l030",       0.00, 0.00, 0.30),
    ("c8_m5dt_l020",     0.10, 0.00, 0.10),
    ("c9_m5m20dt_l030",  0.10, 0.10, 0.10),
]


def _zscore_cs(s: pd.Series) -> pd.Series:
    """Cross-section z-score within a group, NaN-safe."""
    mu = s.mean()
    sd = s.std()
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _code6_to_instrument(c: str) -> str:
    c = str(c).zfill(6)
    if c.startswith("6"):
        return f"SH{c}"
    return f"SZ{c}"


def _instrument_to_code6(inst: str) -> str:
    s = str(inst)
    return s[-6:].zfill(6)


def build_technical_panel() -> pd.DataFrame:
    """构造 (datetime, instrument) × [z_m5, z_m20, z_dt] 全期 panel.

    覆盖 IS+OOS 2017-01-01 ~ 2026-04-30. 在 pred 的 (datetime, instrument)
    上 point-in-time 取值.

    margin_*_chg:    daily cache, 取 pred date <= date 的最新值
    net_buy_pct_evt: event-driven; 当日上榜的 net_buy_pct, 非上榜日 0
    """
    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))
    print(f"[panel] pred dates: {len(pred_dt)}, "
          f"unique instruments: {pred['instrument'].nunique()}", flush=True)

    # === margin panel ===
    print("[panel] loading margin parquet...", flush=True)
    m = pd.read_parquet(MARGIN_PARQUET)
    m["code"] = m["code"].astype(str).str.zfill(6)
    m["date"] = pd.to_datetime(m["date"])
    m = m[["code", "date", "margin_5d_chg", "margin_20d_chg"]]
    m_codes = sorted(m["code"].unique())
    print(f"[panel] margin coverage: {len(m_codes)} codes total "
          "(IS subset 99, OOS extends to 120)", flush=True)

    m_parts = []
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
        m_parts.append(pd.DataFrame({
            "datetime": pred_dt,
            "code": code,
            "margin_5d_chg": m5_v,
            "margin_20d_chg": m20_v,
        }))
    margin_panel = pd.concat(m_parts, ignore_index=True) if m_parts else (
        pd.DataFrame(columns=["datetime", "code", "margin_5d_chg",
                              "margin_20d_chg"])
    )
    print(f"[panel] margin panel rows: {len(margin_panel):,}", flush=True)

    # === dragon_tiger panel ===
    print("[panel] loading dragon_tiger cache...", flush=True)
    csi = pd.read_csv(CSI300_CSV, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    dt_parts = []
    n_with = 0
    n_no_cache = 0
    n_empty = 0
    for code in csi["code"]:
        p = DT_DIR / f"{code}.parquet"
        if not p.exists():
            n_no_cache += 1
            continue
        df = pd.read_parquet(p)
        if df.empty:
            n_empty += 1
            continue
        dt_parts.append(df)
        n_with += 1
    print(f"[panel] DT cache: {n_with} 有 events, "
          f"{n_empty} 空, {n_no_cache} 缺", flush=True)

    if dt_parts:
        dt_all = pd.concat(dt_parts, ignore_index=True)
        dt_all["code"] = dt_all["code"].astype(str).str.zfill(6)
        dt_all["date"] = pd.to_datetime(dt_all["date"]).dt.normalize()
        dt_daily = dt_mod.daily_features(dt_all)
        dt_daily = dt_daily.rename(columns={"date": "datetime"})
        dt_min = dt_daily["datetime"].min()
        dt_max = dt_daily["datetime"].max()
        oos_dt_events = (
            (dt_daily["datetime"] >= "2021-05-01")
            & (dt_daily["datetime"] <= "2026-04-30")
        ).sum()
        print(f"[panel] DT daily events: {len(dt_daily):,}, "
              f"range {dt_min.date()} ~ {dt_max.date()}", flush=True)
        print(f"[panel] DT OOS events 2021-05~2026-04: {oos_dt_events:,} "
              "(0 means OOS will degenerate to baseline + margin-only on z_dt)",
              flush=True)
    else:
        dt_daily = pd.DataFrame(columns=["datetime", "code", "net_buy_pct"])

    # === Join all into pred axis ===
    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()

    out = pred_axis.merge(
        margin_panel, on=["datetime", "code"], how="left"
    )
    if not dt_daily.empty:
        dt_pick = dt_daily[["datetime", "code", "net_buy_pct"]].rename(
            columns={"net_buy_pct": "net_buy_pct_evt"}
        )
        out = out.merge(dt_pick, on=["datetime", "code"], how="left")
    else:
        out["net_buy_pct_evt"] = np.nan
    # event-driven: NaN on non-event day → 0
    out["net_buy_pct_evt"] = out["net_buy_pct_evt"].fillna(0.0)

    n_m5 = out["margin_5d_chg"].notna().sum()
    n_m20 = out["margin_20d_chg"].notna().sum()
    n_dt_nonzero = (out["net_buy_pct_evt"] != 0).sum()
    n_total = len(out)
    print(f"[panel] coverage out of {n_total:,} pred rows:")
    print(f"        margin_5d_chg     : {n_m5:,} ({n_m5/n_total*100:.1f}%)")
    print(f"        margin_20d_chg    : {n_m20:,} ({n_m20/n_total*100:.1f}%)")
    print(f"        net_buy_pct_evt≠0 : {n_dt_nonzero:,} "
          f"({n_dt_nonzero/n_total*100:.2f}%)")

    print("[panel] computing cross-section z-scores...", flush=True)
    out["z_m5"] = out.groupby("datetime")["margin_5d_chg"].transform(_zscore_cs)
    out["z_m20"] = out.groupby("datetime")["margin_20d_chg"].transform(_zscore_cs)
    out["z_dt"] = out.groupby("datetime")["net_buy_pct_evt"].transform(_zscore_cs)
    out[["z_m5", "z_m20", "z_dt"]] = out[["z_m5", "z_m20", "z_dt"]].fillna(0.0)
    return out[["datetime", "instrument", "z_m5", "z_m20", "z_dt"]]


def build_adjusted_predictions(tech_panel: pd.DataFrame,
                               lam_m5: float, lam_m20: float, lam_dt: float,
                               label: str) -> Path:
    """构造 sidecar parquet.

        final = z(pred) - lam_m5 * z_m5 - lam_m20 * z_m20 + lam_dt * z_dt
    """
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(tech_panel, on=["datetime", "instrument"], how="left")
    merged[["z_m5", "z_m20", "z_dt"]] = merged[
        ["z_m5", "z_m20", "z_dt"]
    ].fillna(0.0)
    merged["final_score"] = (
        merged["z_pred"]
        - lam_m5  * merged["z_m5"]
        - lam_m20 * merged["z_m20"]
        + lam_dt  * merged["z_dt"]
    )
    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
    print(f"  [adj] {label} λ=(m5={lam_m5},m20={lam_m20},dt={lam_dt}) "
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
    """跑指定 [first, last] 区间的 walk-forward backtest, 返回 stats."""
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
    print("Phase 4 Sidecar v2 — train24 + 技术因子 overlay")
    print("=" * 70)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST} (48 months)")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"combos: {len(IS_COMBOS)}")
    print()

    print("[step 1] build technical PIT panel (margin + dragon_tiger)...")
    panel = build_technical_panel()

    print("\n[step 2] IS sweep 9 combos (2017-01 ~ 2020-12, 48 months)")
    is_rows = []
    for label, lam_m5, lam_m20, lam_dt in IS_COMBOS:
        print(f"\n  --- {label}  λ=(m5={lam_m5}, m20={lam_m20}, dt={lam_dt}) ---")
        build_adjusted_predictions(panel, lam_m5, lam_m20, lam_dt, label)
        stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{label}")
        row = {
            "combo": label,
            "lam_m5": lam_m5,
            "lam_m20": lam_m20,
            "lam_dt": lam_dt,
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
    best_m5 = float(best["lam_m5"])
    best_m20 = float(best["lam_m20"])
    best_dt = float(best["lam_dt"])
    print(f"\n[lock] best IS: {best_label}  "
          f"λ=(m5={best_m5}, m20={best_m20}, dt={best_dt})  "
          f"IS Calmar={best['calmar']}")

    print(f"\n[step 4] OOS single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    print(f"  locked config: {best_label} "
          f"λ=(m5={best_m5}, m20={best_m20}, dt={best_dt})")
    build_adjusted_predictions(panel, best_m5, best_m20, best_dt, best_label)
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_locked")
    oos_df = oos_stats.pop("months_df")
    oos_df.to_csv(OUT_OOS_STATS, index=False)
    print(f"\n[saved] {OUT_OOS_STATS}")

    print("\n" + "=" * 70)
    print("=== FINAL COMPARISON ===")
    print("=" * 70)
    print("train24 baseline (Phase 2 clean_phase2 OOS, 60m):")
    print("   Calmar=0.42  Sharpe=0.68  ann=12.45%  MDD=-29.86%  cum=79.84%")
    print("\nv19.3 sidecar v1 (revenue_yoy): pending (v1 in-flight, "
          "results not yet saved to disk)")
    print(f"\nv19.4 sidecar v2 OOS ({best_label}, "
          f"λ=(m5={best_m5}, m20={best_m20}, dt={best_dt})):")
    print(f"   Calmar={oos_stats['calmar']}  Sharpe={oos_stats['sharpe']}  "
          f"ann={oos_stats['ann_%']}%  MDD={oos_stats['mdd_%']}%  "
          f"cum={oos_stats['cum_%']}%  win%={oos_stats['win_%']}")

    baseline_calmar = 0.42
    rel_imp = (oos_stats["calmar"] - baseline_calmar) / baseline_calmar * 100
    print(f"\nrelative Calmar Δ vs baseline: {rel_imp:+.1f}%")

    if oos_stats["calmar"] > baseline_calmar * 1.10:
        print("\n[verdict] OOS Calmar > baseline × 1.10 → 推荐升级 production v19.4")
    elif oos_stats["calmar"] > baseline_calmar:
        print("\n[verdict] OOS Calmar > baseline but <10% imp → marginal, NOT recommend")
    else:
        print("\n[verdict] OOS Calmar <= baseline → abort, technical sidecar 无效 / IS bias 暴露")

    return 0


if __name__ == "__main__":
    sys.exit(main())
