"""v19.3 Sidecar — train24 + fundamental overlay (revenue_yoy / combo_growth).

Sidecar 公式:
    final_score = z(train24_pred) + λ × z(fundamental_signal)

候选 fundamental signal:
    F1 "revenue_yoy"    : z(revenue_yoy)
    F2 "combo_growth"   : (z(revenue_yoy) + z(net_profit_yoy)) / 2
    F3 "rev_growth_mix" : (z(revenue_yoy) + z(combo_growth)) / 2
                        = (1.5*z(revenue_yoy) + 0.5*z(net_profit_yoy)) / 2

λ ∈ {0.05, 0.10, 0.20}

严格 OOS 协议:
  IS  期 2017-01 ~ 2020-12 (48 月) sweep 9 组合, 选最佳 Calmar
  OOS 期 2021-05 ~ 2026-04 (60 月) 仅锁定 (factor, λ) 后跑 1 次

不动任何 production 文件 / pred cache / fundamentals cache.

Run:
  python examples/strategy_v19_3_sidecar.py
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

from claude_finance import fundamentals as fund  # noqa: E402

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
CSI300_CSV = ROOT / "data_cache" / "csi300_constituents.csv"
ADJ_PRED = ROOT / "data_cache" / "v19_3_sidecar_predictions.parquet"
OUT_IS_GRID = ROOT / "examples" / "v19_3_sidecar_grid_is.csv"
OUT_OOS_STATS = ROOT / "examples" / "v19_3_sidecar_oos_stats.csv"

# IS / OOS 严格边界 (与 Phase 2 clean_phase2 baseline 对齐)
IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

FACTOR_CANDIDATES = ["revenue_yoy", "combo_growth", "rev_growth_mix"]
LAMBDA_CANDIDATES = [0.05, 0.10, 0.20]


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


def build_fundamental_panel() -> pd.DataFrame:
    """对 pred 的每个 (datetime, instrument) 取 PIT fundamentals.

    返回 long df: datetime / instrument / revenue_yoy / net_profit_yoy.
    PIT cutoff: report_date + ANNOUNCE_LAG_DAYS (60) <= datetime.

    优化: fundamentals per stock 是 quarterly; pred dates 是 daily. 用 binary
    search 把每股的 quarterly 报告 forward-fill 到 trading-day axis.
    """
    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))
    inst_set = sorted(pred["instrument"].unique())
    print(f"[panel] pred dates: {len(pred_dt)}, instruments: {len(inst_set)}",
          flush=True)

    csi = pd.read_csv(CSI300_CSV, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    csi["instrument"] = csi["code"].apply(_code6_to_instrument)
    inst_to_code = dict(zip(csi["instrument"], csi["code"]))

    rows = []
    n_skipped = 0
    for inst in inst_set:
        code = inst_to_code.get(inst)
        if code is None:
            n_skipped += 1
            continue
        df = fund.load_cached(code)
        if df is None or df.empty:
            n_skipped += 1
            continue
        df = df.sort_values("report_date").reset_index(drop=True)
        visible_from = (
            pd.to_datetime(df["report_date"])
            + pd.Timedelta(days=fund.ANNOUNCE_LAG_DAYS)
        ).values
        # 二分查每个 pred date 对应的"最新可见 report" 索引
        idx = np.searchsorted(visible_from, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe_idx = np.clip(idx, 0, len(df) - 1)
        rev = np.where(valid, df["revenue_yoy"].values[safe_idx], np.nan)
        npp = np.where(valid, df["net_profit_yoy"].values[safe_idx], np.nan)
        sub = pd.DataFrame({
            "datetime": pred_dt,
            "instrument": inst,
            "revenue_yoy": rev,
            "net_profit_yoy": npp,
        })
        rows.append(sub)

    if n_skipped:
        print(f"[panel] {n_skipped} instruments without cache (skipped)",
              flush=True)
    out = pd.concat(rows, ignore_index=True)
    n_with_rev = out["revenue_yoy"].notna().sum()
    print(f"[panel] built {len(out):,} rows, {n_with_rev:,} have revenue_yoy "
          f"({n_with_rev / len(out) * 100:.1f}%)", flush=True)
    return out


def build_adjusted_predictions(panel: pd.DataFrame, factor: str, lam: float) -> Path:
    """构造 sidecar parquet: score = z(pred) + λ × z(factor).

    覆盖 IS + OOS 全期 (2017-01 ~ 2026-04). Cross-section z 按 datetime.
    """
    pred = pd.read_parquet(ORIG_PRED)
    # z(pred) cross-section by datetime
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)

    # z(factor) cross-section by datetime
    p = panel.copy()
    p["z_rev"] = p.groupby("datetime")["revenue_yoy"].transform(_zscore_cs)
    p["z_np"] = p.groupby("datetime")["net_profit_yoy"].transform(_zscore_cs)
    p["z_combo_growth"] = (p["z_rev"] + p["z_np"]) / 2

    if factor == "revenue_yoy":
        sig = p["z_rev"]
    elif factor == "combo_growth":
        sig = p["z_combo_growth"]
    elif factor == "rev_growth_mix":
        sig = (p["z_rev"] + p["z_combo_growth"]) / 2
    else:
        raise ValueError(f"unknown factor: {factor}")

    p["z_fund"] = sig
    # 再做一次 cross-section z 以 normalize scale 与 z_pred 同等级
    p["z_fund"] = p.groupby("datetime")["z_fund"].transform(_zscore_cs)

    merged = pred.merge(
        p[["datetime", "instrument", "z_fund"]],
        on=["datetime", "instrument"], how="left",
    )
    n_miss = merged["z_fund"].isna().sum()
    merged["z_fund"] = merged["z_fund"].fillna(0.0)
    merged["final_score"] = merged["z_pred"] + lam * merged["z_fund"]

    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
    print(f"  [adj] factor={factor} λ={lam} rows={len(out):,} "
          f"missing_fund={n_miss:,} ({n_miss / len(out) * 100:.1f}%)",
          flush=True)
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

    # Override v17 config + reset disk cache (force re-read adjusted pred)
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
    print("Phase 4 Sidecar — train24 + (revenue_yoy / combo_growth) overlay")
    print("=" * 70)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST} (48 months)")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"factors: {FACTOR_CANDIDATES}")
    print(f"lambdas: {LAMBDA_CANDIDATES}")
    print()

    # Step 1: build PIT panel once
    print("[step 1] build fundamentals PIT panel...")
    panel = build_fundamental_panel()

    # Step 2: IS sweep 9 combos
    print("\n[step 2] IS sweep 9 combos (2017-01 ~ 2020-12, 48 months)")
    is_rows = []
    for factor in FACTOR_CANDIDATES:
        for lam in LAMBDA_CANDIDATES:
            print(f"\n  --- factor={factor}  λ={lam} ---")
            build_adjusted_predictions(panel, factor, lam)
            stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{factor}_l{lam}")
            row = {
                "factor": factor,
                "lambda": lam,
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

    # Step 3: lock best (factor, λ)
    best = is_df.iloc[0]
    best_factor = best["factor"]
    best_lam = float(best["lambda"])
    print(f"\n[lock] best IS: factor={best_factor}  λ={best_lam}  "
          f"IS Calmar={best['calmar']}")

    # Step 4: OOS single run (no peeking allowed after this)
    print(f"\n[step 4] OOS single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    print(f"  locked config: factor={best_factor}  λ={best_lam}")
    build_adjusted_predictions(panel, best_factor, best_lam)
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_locked")
    oos_df = oos_stats.pop("months_df")
    oos_df.to_csv(OUT_OOS_STATS, index=False)
    print(f"\n[saved] {OUT_OOS_STATS}")

    print("\n" + "=" * 70)
    print("=== FINAL COMPARISON ===")
    print("=" * 70)
    print("train24 baseline (Phase 2 clean_phase2 OOS, 60m):")
    print("   Calmar=0.42  Sharpe=0.68  ann=12.45%  MDD=-29.86%  cum=79.84%")
    print(f"\nv19.3 sidecar OOS (factor={best_factor}, λ={best_lam}):")
    print(f"   Calmar={oos_stats['calmar']}  Sharpe={oos_stats['sharpe']}  "
          f"ann={oos_stats['ann_%']}%  MDD={oos_stats['mdd_%']}%  "
          f"cum={oos_stats['cum_%']}%")

    baseline_calmar = 0.42
    rel_imp = (oos_stats["calmar"] - baseline_calmar) / baseline_calmar * 100
    print(f"\nrelative Calmar Δ: {rel_imp:+.1f}%")

    if oos_stats["calmar"] > baseline_calmar * 1.05:
        print("\n[verdict] OOS Calmar > baseline × 1.05 → 推荐升级 production v19.3")
    elif oos_stats["calmar"] > baseline_calmar:
        print("\n[verdict] OOS Calmar > baseline but <5% imp → marginal, NOT recommend")
    else:
        print("\n[verdict] OOS Calmar <= baseline → abort, sidecar 无效")

    return 0


if __name__ == "__main__":
    sys.exit(main())
