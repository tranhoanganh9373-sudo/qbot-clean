"""v20 shareholders Sidecar — train24 + count_change_12m (股东户数 12 月累积变化).

Phase B (锁定 Phase A 结果, 不可改):
    factor:   count_change_12m  (12 月累积股东户数变化%, PIT announce_date)
    sign:     -1                (户数减少 → 集中度上升 → bullish)
    horizon:  5d forward
    λ ∈ {0.10, 0.20, 0.30}

公式:
    final_score = z(train24_pred) - λ · z(count_change_12m)

阶段:
  Step 1: build PIT panel
    - shareholders cache: data_cache/shareholders/shareholders_csi300.parquet
      (23341 rows × 289 unique stocks, PIT announce_date 2013-01 ~ 2026-05)
    - 在 pred axis 上每个月初 sampling point T 取 announce_date ≤ T 的 latest
      season 数据, 用 max_lookback_days=365, 累积 12m 区间内 count_change_pct.
    - 横截面 z-score per month_anchor, fillna(0) 给无 events 股
    - merge_asof forward-fill 到 daily pred axis

  Step 2: IS sweep
    IS 2017-01 ~ 2020-12 (48 月), 跑 λ ∈ {0.10, 0.20, 0.30}, 看 IS Calmar

  Step 3: lock best λ → OOS single run
    OOS 2021-05 ~ 2026-04 (60 月), 跑一次, 不可重 sweep / 不可改 λ / sign.

严格 OOS 协议:
  - sign 不可改
  - λ candidates 不可加
  - IS 期不含 2021-01+
  - OOS 跑一次, 即使数字差也不能回去改

Prediction cache:
  data_cache/v17_dens_train24_predictions.parquet (Phase 2 v3 主表).
  CSI300 instrument 子集 = 300, 月跨 2017-01 ~ 2026-04 (112 月).

警告 — 高风险:
  super_big_net Phase B (n=36) IS Calmar 2.01 → OOS -0.07 catastrophic.
  shareholders n=49 月 比 super 36 多 36%, 但仍属低样本.
  预期 OOS Calmar 可能 -0.3 ~ +0.3 区间.

Run:
  .venv/bin/python examples/strategy_v20_shareholders.py
"""
from __future__ import annotations

import argparse
import json
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

from claude_finance import shareholders  # noqa: E402

# === Prediction cache 选择 ===
PRED_CACHE_SRC = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
ADJ_PRED = ROOT / "data_cache" / "v20_shareholders_predictions.parquet"

OUT_IS_GRID = ROOT / "examples" / "v20_shareholders_phase_b_is_grid.csv"
OUT_LOCKED_LAM = ROOT / "examples" / "v20_shareholders_phase_b_locked_lam.json"
OUT_OOS_STATS = ROOT / "examples" / "v20_shareholders_phase_b_oos_stats.csv"
OUT_SUMMARY = ROOT / "examples" / "v20_shareholders_phase_b_summary.md"

IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# Phase A 锁定 (不可改)
FACTOR_SIGN = -1
LAMBDA_CANDIDATES = [0.10, 0.20, 0.30]
FACTOR_COL = "count_change_12m"


def _zscore_cs(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std()
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _instrument_to_code6(inst: str) -> str:
    return str(inst)[-6:].zfill(6)


def load_pred() -> pd.DataFrame:
    if not PRED_CACHE_SRC.exists():
        raise FileNotFoundError(f"Pred cache 缺: {PRED_CACHE_SRC}")
    pred = pd.read_parquet(PRED_CACHE_SRC)
    print(f"[pred] cache = {PRED_CACHE_SRC.name}; rows={len(pred):,}, "
          f"dates {pred['datetime'].min().date()} ~ "
          f"{pred['datetime'].max().date()}, "
          f"instruments={pred['instrument'].nunique()}",
          flush=True)
    return pred


def build_pit_panel(pred: pd.DataFrame) -> pd.DataFrame:
    """构造 PIT (datetime, instrument) × z_factor."""
    share = shareholders.load_cache()
    print(f"[share] cache rows={len(share):,}, "
          f"codes={share['code'].nunique()}, "
          f"announce {share['announce_date'].min().date()} ~ "
          f"{share['announce_date'].max().date()}", flush=True)

    pred_axis = pred[["datetime", "instrument"]].drop_duplicates().copy()
    pred_axis["code"] = pred_axis["instrument"].apply(_instrument_to_code6)
    codes = sorted(pred_axis["code"].unique())
    print(f"[panel] pred axis codes (6-digit): {len(codes)}", flush=True)

    # 月初 sampling (与 Phase A IC 同口径)
    pred_dt_ser = pd.Series(sorted(pd.DatetimeIndex(
        pred["datetime"].unique()
    ).tolist()))
    months = pred_dt_ser.dt.to_period("M")
    month_first = pred_dt_ser.groupby(months).first().tolist()
    print(f"[panel] month-first sampling points: {len(month_first)}",
          flush=True)

    factor_panel = shareholders.build_factor_panel(
        share, codes, month_first, max_lookback_days=365,
    )
    factor_panel = factor_panel[[
        "asof_date", "code", FACTOR_COL,
    ]].rename(columns={"asof_date": "month_anchor"})
    n_total = len(factor_panel)
    n_ok = factor_panel[FACTOR_COL].notna().sum()
    print(f"[panel] monthly factor cells: {n_total:,} "
          f"({n_ok:,} non-NaN = {n_ok/n_total*100:.1f}%)",
          flush=True)

    # 横截面 z-score per month_anchor, fillna(0) 给无 events 股
    factor_panel["z_factor"] = factor_panel.groupby("month_anchor")[
        FACTOR_COL
    ].transform(_zscore_cs).fillna(0.0)

    # merge_asof forward-fill: 每个 (code, daily T) 找 ≤ T 的 latest month_anchor
    parts = []
    pred_axis_sorted = pred_axis.sort_values(["code", "datetime"])
    for code, sub_axis in pred_axis_sorted.groupby("code", sort=False):
        sub_fac = factor_panel[factor_panel["code"] == code]
        if sub_fac.empty:
            tmp = sub_axis.copy()
            tmp["z_factor"] = 0.0
            parts.append(tmp[["datetime", "instrument", "z_factor"]])
            continue
        merged = pd.merge_asof(
            sub_axis.sort_values("datetime"),
            sub_fac[["month_anchor", "z_factor"]].sort_values("month_anchor"),
            left_on="datetime", right_on="month_anchor",
            direction="backward",
        )
        merged["z_factor"] = merged["z_factor"].fillna(0.0)
        parts.append(merged[["datetime", "instrument", "z_factor"]])
    out = pd.concat(parts, ignore_index=True)

    print(f"[panel] pred axis × z_factor rows={len(out):,} "
          f"(non-zero={int((out['z_factor']!=0).sum()):,})",
          flush=True)
    return out


def write_adjusted_pred(pred: pd.DataFrame, panel: pd.DataFrame,
                        lam: float, label: str) -> Path:
    """final = z(pred) + sign * lam * z(factor) = z(pred) - lam * z(factor)."""
    p = pred.copy()
    p["z_pred"] = p.groupby("datetime")["score"].transform(_zscore_cs)
    merged = p.merge(panel, on=["datetime", "instrument"], how="left")
    merged["z_factor"] = merged["z_factor"].fillna(0.0)
    merged["final_score"] = (
        merged["z_pred"]
        + FACTOR_SIGN * lam * merged["z_factor"]
    )
    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
    print(f"  [adj] {label} λ={lam} sign={FACTOR_SIGN} rows={len(out):,}",
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
    """跑指定 [first, last] 区间 walk-forward backtest 使用 ADJ_PRED."""
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


def run_is_sweep(pred: pd.DataFrame, panel: pd.DataFrame
                 ) -> tuple[float, float, pd.DataFrame]:
    """跑 IS sweep, lock 最高 Calmar 的 λ."""
    print(f"\n[IS] sweep λ ∈ {LAMBDA_CANDIDATES} "
          f"({IS_FIRST} ~ {IS_LAST}, 48 months)")
    is_rows = []
    for lam in LAMBDA_CANDIDATES:
        label = f"is_lam{lam:.2f}"
        print(f"\n  --- {label}  λ={lam} ---")
        write_adjusted_pred(pred, panel, lam, label)
        stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{label}")
        is_rows.append({
            "phase": "IS",
            "lam": lam,
            "cum_%": stats["cum_%"],
            "ann_%": stats["ann_%"],
            "sharpe": stats["sharpe"],
            "mdd_%": stats["mdd_%"],
            "calmar": stats["calmar"],
            "win_%": stats["win_%"],
            "avg_picks": stats["avg_picks"],
            "n_months": stats["n"],
        })
        print(f"    >> IS Calmar={stats['calmar']} Sharpe={stats['sharpe']} "
              f"ann={stats['ann_%']}% MDD={stats['mdd_%']}%")

    is_df = pd.DataFrame(is_rows)
    is_df_sorted = is_df.sort_values(
        "calmar", ascending=False
    ).reset_index(drop=True)
    is_df_sorted.to_csv(OUT_IS_GRID, index=False)
    print(f"\n[saved] {OUT_IS_GRID}")
    print("\n=== IS λ sweep (sorted desc by Calmar) ===")
    print(is_df_sorted.to_string(index=False))

    best = is_df_sorted.iloc[0]
    best_lam = float(best["lam"])
    best_calmar_is = float(best["calmar"])
    print(f"\n[lock] best IS: λ={best_lam}  IS Calmar={best_calmar_is}")

    OUT_LOCKED_LAM.write_text(json.dumps({
        "locked_lam": best_lam,
        "is_calmar": best_calmar_is,
        "factor_sign": FACTOR_SIGN,
        "factor_col": FACTOR_COL,
        "lam_candidates": LAMBDA_CANDIDATES,
        "is_first": IS_FIRST,
        "is_last": IS_LAST,
        "pred_cache": PRED_CACHE_SRC.name,
        "locked_at": datetime.now().isoformat(),
    }, indent=2))
    print(f"[saved] {OUT_LOCKED_LAM}")
    return best_lam, best_calmar_is, is_df_sorted


def run_oos_single(pred: pd.DataFrame, panel: pd.DataFrame,
                   best_lam: float) -> dict:
    """跑 OOS 单次."""
    print(f"\n[OOS] single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    print(f"  locked: λ={best_lam}  sign={FACTOR_SIGN}")
    write_adjusted_pred(pred, panel, best_lam, f"oos_lam{best_lam:.2f}")
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_locked")
    oos_df = oos_stats.pop("months_df")
    oos_df.to_csv(OUT_OOS_STATS, index=False)
    print(f"\n[saved] {OUT_OOS_STATS}")
    return oos_stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-only", action="store_true",
                        help="Run only IS sweep, lock λ. Skip OOS.")
    parser.add_argument("--oos-only", action="store_true",
                        help="Read locked λ from json, run only OOS.")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase B Sidecar v20 — train24 + count_change_12m (shareholders)")
    print("=" * 72)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST} (48 months)")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"λ candidates: {LAMBDA_CANDIDATES}")
    print(f"sign: {FACTOR_SIGN} (locked Phase A, immutable)")
    print(f"factor: {FACTOR_COL} (locked Phase A)")
    if args.is_only:
        print("mode: IS-ONLY (skip OOS)")
    elif args.oos_only:
        print("mode: OOS-ONLY (read locked λ from json)")
    else:
        print("mode: FULL (IS sweep → lock → OOS)")
    print()

    print("[step 1] load prediction cache")
    pred = load_pred()
    print()

    print("[step 2] build PIT factor panel (announce_date ≤ T)")
    panel = build_pit_panel(pred)
    print()

    if args.oos_only:
        if not OUT_LOCKED_LAM.exists():
            print(f"FATAL: {OUT_LOCKED_LAM} 缺. 先跑 --is-only.",
                  file=sys.stderr)
            return 1
        locked = json.loads(OUT_LOCKED_LAM.read_text())
        best_lam = float(locked["locked_lam"])
        best_calmar_is = float(locked["is_calmar"])
        is_df_sorted = (pd.read_csv(OUT_IS_GRID)
                        if OUT_IS_GRID.exists() else None)
        print(f"[oos-only] reading locked λ={best_lam} (IS Calmar="
              f"{best_calmar_is}) from {OUT_LOCKED_LAM.name}")
    else:
        best_lam, best_calmar_is, is_df_sorted = run_is_sweep(pred, panel)

    if args.is_only:
        print("\n[is-only] skipping OOS run.")
        return 0

    oos_stats = run_oos_single(pred, panel, best_lam)

    print("\n" + "=" * 72)
    print("=== FINAL COMPARISON v20 shareholders ===")
    print("=" * 72)
    # v2-era baselines (per task brief)
    baselines = {
        "Phase2_v2_baseline":      0.86,
        "v19.4_margin_v2":         1.28,
        "v19.6_amplitude_v2":      0.58,
        "v20_industry_60d_v2":     0.84,
        "v20_vol_z_5d_v2":         0.54,
        "v20_super_big_net_v2":    -0.07,
    }
    for k, v in baselines.items():
        print(f"  {k:30s}: Calmar {v:+.2f}")
    print()
    print(f"v20 shareholders OOS [λ={best_lam}]:")
    print(f"   Calmar={oos_stats['calmar']}  Sharpe={oos_stats['sharpe']}  "
          f"ann={oos_stats['ann_%']}%  MDD={oos_stats['mdd_%']}%  "
          f"cum={oos_stats['cum_%']}%  win%={oos_stats['win_%']}")

    c = oos_stats["calmar"]
    if c >= 0.80:
        verdict = "推荐升级 (production candidate, OOS Calmar ≥ 0.80)"
    elif c >= 0.50:
        verdict = "shadow 跟踪 (OOS Calmar 0.50-0.80)"
    else:
        verdict = "abort (OOS Calmar < 0.50)"
    print(f"\n[verdict] {verdict}")

    # Save summary md
    summary = []
    summary.append("# Phase B v20 shareholders Sidecar — 结果")
    summary.append("")
    summary.append(f"**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    summary.append(f"**Base pred cache:** {PRED_CACHE_SRC.name} (Phase 2 v3)")
    summary.append(f"**factor:** {FACTOR_COL}  sign={FACTOR_SIGN}  horizon=5d")
    summary.append(f"**λ candidates:** {LAMBDA_CANDIDATES}")
    summary.append(f"**IS:** {IS_FIRST} ~ {IS_LAST} (48 months)")
    summary.append(f"**OOS:** {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    summary.append("")
    summary.append("## IS sweep (sorted desc by Calmar)")
    summary.append("")
    if is_df_sorted is not None:
        summary.append(is_df_sorted.to_markdown(index=False))
    summary.append("")
    summary.append(
        f"**Locked λ = {best_lam}**  (best IS Calmar = {best_calmar_is})"
    )
    summary.append("")
    summary.append("## OOS single run")
    summary.append("")
    summary.append("| metric | value |")
    summary.append("|---|---|")
    summary.append(f"| Calmar | {oos_stats['calmar']} |")
    summary.append(f"| Sharpe | {oos_stats['sharpe']} |")
    summary.append(f"| ann %  | {oos_stats['ann_%']} |")
    summary.append(f"| MDD %  | {oos_stats['mdd_%']} |")
    summary.append(f"| cum %  | {oos_stats['cum_%']} |")
    summary.append(f"| win %  | {oos_stats['win_%']} |")
    summary.append(f"| n_months | {oos_stats['n']} |")
    summary.append(f"| avg_picks | {oos_stats['avg_picks']} |")
    summary.append("")
    summary.append("## Comparison (vs v2-era runs)")
    summary.append("")
    summary.append("| version | OOS Calmar |")
    summary.append("|---|---|")
    for k, v in baselines.items():
        summary.append(f"| {k} | {v} |")
    summary.append(
        f"| **v20_shareholders** | **{oos_stats['calmar']}** |"
    )
    summary.append("")
    summary.append(f"**Verdict:** {verdict}")
    OUT_SUMMARY.write_text("\n".join(summary))
    print(f"\n[saved] {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
