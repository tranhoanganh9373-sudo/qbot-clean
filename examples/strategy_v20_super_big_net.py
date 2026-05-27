"""v20 super_big_net Sidecar — train24 + Sina fund-flow super-big-net 5d change.

Phase B (锁定 Phase A 结果, 不可改):
    factor: net_super_big_5d_chg  (= (nsb_t - nsb_{t-5}) / |nsb_{t-5}|)
    sign:   -1
    horizon: 5d
    λ ∈ {0.10, 0.20, 0.30}

公式:
    final_score = z(train24_pred) - λ · z(net_super_big_5d_chg)

阶段:
  Step 1: build PIT panel
    - IS 期 (≤ 2020-12-31): 从 data_cache/fund_flow/fund_flow_csi300.parquet (Stage 1)
    - OOS 期 (≥ 2021-01-01): 从 data_cache/fund_flow_oos/fund_flow_csi300_oos.parquet (Stage 2)
    - rolling 5d pct_change of net_super_big per code
    - PIT 投影到 pred axis (datetime, instrument), 横截面 z-score, fillna(0)

  Step 2: IS sweep
    IS 2017-01 ~ 2020-12 (48 月), 跑 λ ∈ {0.10, 0.20, 0.30}, 看 IS Calmar

  Step 3: lock best λ → OOS single run
    OOS 2021-05 ~ 2026-04 (60 月), 跑一次, 不可重 sweep

严格 OOS 协议 (CLAUDE.md rule 5):
  - sign 不可改
  - λ candidates 不可加
  - IS 期不含 2021-01+
  - OOS 跑一次, 即使数字差也不能回去改

Prediction cache 选择:
  使用 v17_dens_train24_predictions.pre_phase2_v2.bak (Phase 2 v2 clean cache)
  原因: 用户明确指示 "用 v2 保证 reproducibility — log 用了哪个 cache".
  Phase 2 v3 重训仍在 in-flight (subagent a17e61bb8b3261cbb), 主 parquet 已被截断
  到 2022-06.

Run:
  .venv/bin/python examples/strategy_v20_super_big_net.py [--is-only|--oos-only]

  --is-only: 仅跑 IS sweep, lock λ, 写 is_grid.csv 与 locked_lam.json (skip OOS).
  --oos-only: 读 locked_lam.json + 跑 OOS, 写 oos_stats.csv + summary.md.
  默认: 完整流水 (IS sweep → lock → OOS run).

注意 --is-only 模式下 OOS fund_flow 不需要存在; --oos-only 模式需要 OOS fund_flow
和 locked_lam.json 都已存在.
"""
from __future__ import annotations

import argparse
import json
import shutil
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

# === Prediction cache 选择 ===
# v2 clean cache (Phase 2 v2 backup, untouched by in-flight v3 rerun).
PRED_CACHE_SRC = ROOT / "data_cache" / "v17_dens_train24_predictions.pre_phase2_v2.bak"
# 强 v17_dens_grid 内部 PRED_CACHE 指向独立 working copy (避免污染原 backup)
ORIG_PRED = ROOT / "data_cache" / "v20_super_big_net_base_pred.parquet"

FUND_FLOW_IS = ROOT / "data_cache" / "fund_flow" / "fund_flow_csi300.parquet"
FUND_FLOW_OOS = ROOT / "data_cache" / "fund_flow_oos" / "fund_flow_csi300_oos.parquet"
ADJ_PRED = ROOT / "data_cache" / "v20_super_big_net_predictions.parquet"
OUT_IS_GRID = ROOT / "examples" / "v20_super_big_net_is_grid.csv"
OUT_LOCKED_LAM = ROOT / "examples" / "v20_super_big_net_locked_lam.json"
OUT_OOS_STATS = ROOT / "examples" / "v20_super_big_net_oos_stats.csv"
OUT_SUMMARY = ROOT / "examples" / "v20_super_big_net_summary.md"

IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# Phase A 锁定 (不可改)
FACTOR_SIGN = -1
LAMBDA_CANDIDATES = [0.10, 0.20, 0.30]


def _zscore_cs(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std()
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _instrument_to_code6(inst: str) -> str:
    return str(inst)[-6:].zfill(6)


def prepare_base_pred() -> Path:
    """Copy Phase 2 v2 cache to working file (不污染原 backup)."""
    if not PRED_CACHE_SRC.exists():
        raise FileNotFoundError(
            f"Phase 2 v2 cache 缺: {PRED_CACHE_SRC}"
        )
    if not ORIG_PRED.exists():
        print(f"[pred] copying {PRED_CACHE_SRC.name} -> {ORIG_PRED.name}",
              flush=True)
        shutil.copyfile(PRED_CACHE_SRC, ORIG_PRED)
    pred = pd.read_parquet(ORIG_PRED)
    print(f"[pred] cache = {ORIG_PRED.name}; rows={len(pred):,}, "
          f"dates {pred['datetime'].min().date()} ~ "
          f"{pred['datetime'].max().date()}, "
          f"instruments={pred['instrument'].nunique()}",
          flush=True)
    return ORIG_PRED


def load_fund_flow() -> pd.DataFrame:
    """合并 IS + OOS fund flow; 算 net_super_big_5d_chg per code."""
    parts = []
    if FUND_FLOW_IS.exists():
        df_is = pd.read_parquet(FUND_FLOW_IS, columns=[
            "code", "date", "net_super_big"
        ])
        parts.append(df_is)
        print(f"[ff] IS rows={len(df_is):,}, codes={df_is['code'].nunique()}, "
              f"{df_is['date'].min().date()} ~ {df_is['date'].max().date()}",
              flush=True)
    else:
        raise FileNotFoundError(f"IS fund_flow 缺: {FUND_FLOW_IS}")

    if FUND_FLOW_OOS.exists():
        df_oos = pd.read_parquet(FUND_FLOW_OOS, columns=[
            "code", "date", "net_super_big"
        ])
        parts.append(df_oos)
        print(f"[ff] OOS rows={len(df_oos):,}, codes={df_oos['code'].nunique()}, "
              f"{df_oos['date'].min().date()} ~ {df_oos['date'].max().date()}",
              flush=True)
    else:
        print(f"[ff] WARN: OOS fund_flow 缺 {FUND_FLOW_OOS}, "
              f"OOS 期 factor 全 NaN (退化为 baseline)", flush=True)

    ff = pd.concat(parts, ignore_index=True)
    ff["code"] = ff["code"].astype(str).str.zfill(6)
    ff["date"] = pd.to_datetime(ff["date"])
    ff = ff.drop_duplicates(["code", "date"]).sort_values(
        ["code", "date"]
    ).reset_index(drop=True)
    print(f"[ff] merged rows={len(ff):,}, codes={ff['code'].nunique()}, "
          f"{ff['date'].min().date()} ~ {ff['date'].max().date()}",
          flush=True)

    # Compute net_super_big_5d_chg per code (Phase A definition)
    grp = ff.groupby("code", sort=False)
    nsb = ff["net_super_big"]
    nsb_lag5 = grp["net_super_big"].shift(5)
    denom = nsb_lag5.abs()
    ff["net_super_big_5d_chg"] = np.where(
        denom > 0, (nsb - nsb_lag5) / denom, np.nan
    )
    n_valid = ff["net_super_big_5d_chg"].notna().sum()
    print(f"[ff] factor net_super_big_5d_chg valid={n_valid:,} "
          f"({n_valid/len(ff)*100:.1f}%)", flush=True)
    return ff[["code", "date", "net_super_big_5d_chg"]]


def build_pit_panel(pred_path: Path, ff: pd.DataFrame) -> pd.DataFrame:
    """构造 (datetime, instrument) × z_net_super_big_5d_chg, PIT on pred axis."""
    pred = pd.read_parquet(pred_path, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))
    print(f"[panel] pred dates: {len(pred_dt)}, "
          f"unique instruments: {pred['instrument'].nunique()}", flush=True)

    ff_sorted = ff.sort_values(["code", "date"])
    parts = []
    for code, sub in ff_sorted.groupby("code", sort=False):
        dates_arr = sub["date"].values
        fac = sub["net_super_big_5d_chg"].values
        idx = np.searchsorted(dates_arr, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe_idx = np.clip(idx, 0, len(sub) - 1)
        v = np.where(valid, fac[safe_idx], np.nan)
        parts.append(pd.DataFrame({
            "datetime": pred_dt,
            "code": code,
            "net_super_big_5d_chg": v,
        }))
    if not parts:
        raise RuntimeError("[panel] no PIT factor rows built")
    panel = pd.concat(parts, ignore_index=True)

    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    out = pred_axis.merge(panel, on=["datetime", "code"], how="left")

    n_total = len(out)
    n_ok = out["net_super_big_5d_chg"].notna().sum()
    print(f"[panel] factor coverage on pred axis: "
          f"{n_ok:,}/{n_total:,} ({n_ok/n_total*100:.1f}%)", flush=True)

    # Cross-section z per datetime
    out["z_net_super_big_5d_chg"] = out.groupby("datetime")[
        "net_super_big_5d_chg"
    ].transform(_zscore_cs).fillna(0.0)
    return out[["datetime", "instrument", "z_net_super_big_5d_chg"]]


def write_adjusted_pred(panel: pd.DataFrame, lam: float, label: str) -> Path:
    """final = z(pred) + sign * lam * z(factor) = z(pred) - lam * z(factor)."""
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(panel, on=["datetime", "instrument"], how="left")
    merged["z_net_super_big_5d_chg"] = merged[
        "z_net_super_big_5d_chg"
    ].fillna(0.0)
    merged["final_score"] = (
        merged["z_pred"]
        + FACTOR_SIGN * lam * merged["z_net_super_big_5d_chg"]
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


def run_is_sweep(panel: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """跑 IS sweep, lock 最高 Calmar 的 λ, 写 locked_lam.json + is_grid.csv."""
    print(f"\n[IS] sweep λ ∈ {LAMBDA_CANDIDATES} "
          f"({IS_FIRST} ~ {IS_LAST}, 48 months)")
    is_rows = []
    for lam in LAMBDA_CANDIDATES:
        label = f"is_lam{lam:.2f}"
        print(f"\n  --- {label}  λ={lam} ---")
        write_adjusted_pred(panel, lam, label)
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
        "lam_candidates": LAMBDA_CANDIDATES,
        "is_first": IS_FIRST,
        "is_last": IS_LAST,
        "locked_at": datetime.now().isoformat(),
    }, indent=2))
    print(f"[saved] {OUT_LOCKED_LAM}")
    return best_lam, best_calmar_is, is_df_sorted


def run_oos_single(panel: pd.DataFrame, best_lam: float,
                   best_calmar_is: float, is_df_sorted: pd.DataFrame | None
                   ) -> dict:
    """跑 OOS 单次, 写 oos_stats.csv + summary.md."""
    print(f"\n[OOS] single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    print(f"  locked: λ={best_lam}  sign={FACTOR_SIGN}")
    write_adjusted_pred(panel, best_lam, f"oos_lam{best_lam:.2f}")
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_locked")
    oos_df = oos_stats.pop("months_df")
    oos_df.to_csv(OUT_OOS_STATS, index=False)
    print(f"\n[saved] {OUT_OOS_STATS}")
    return oos_stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is-only", action="store_true",
        help="Run only IS sweep, lock λ. Skip OOS.",
    )
    parser.add_argument(
        "--oos-only", action="store_true",
        help="Read locked λ from json, run only OOS.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Phase B Sidecar v20 — train24 + net_super_big_5d_chg")
    print("=" * 72)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST} (48 months)")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"λ candidates: {LAMBDA_CANDIDATES}")
    print(f"sign: {FACTOR_SIGN} (locked Phase A, immutable)")
    if args.is_only:
        print("mode: IS-ONLY (skip OOS)")
    elif args.oos_only:
        print("mode: OOS-ONLY (read locked λ from json)")
    else:
        print("mode: FULL (IS sweep → lock → OOS)")
    print()

    print("[step 0] prepare base prediction cache (Phase 2 v2)")
    pred_path = prepare_base_pred()
    print()

    print("[step 1] load fund_flow (IS + OOS) and compute factor")
    ff = load_fund_flow()
    print()

    print("[step 2] build PIT panel on pred axis")
    panel = build_pit_panel(pred_path, ff)
    print()

    if args.oos_only:
        if not OUT_LOCKED_LAM.exists():
            print(f"FATAL: {OUT_LOCKED_LAM} 缺. 先跑 --is-only.",
                  file=sys.stderr)
            return 1
        locked = json.loads(OUT_LOCKED_LAM.read_text())
        best_lam = float(locked["locked_lam"])
        best_calmar_is = float(locked["is_calmar"])
        # Read existing IS grid if present (for summary)
        is_df_sorted = (pd.read_csv(OUT_IS_GRID)
                        if OUT_IS_GRID.exists() else None)
        print(f"[oos-only] reading locked λ={best_lam} (IS Calmar="
              f"{best_calmar_is}) from {OUT_LOCKED_LAM.name}")
    else:
        best_lam, best_calmar_is, is_df_sorted = run_is_sweep(panel)

    if args.is_only:
        print("\n[is-only] skipping OOS run. Re-run with --oos-only after "
              "OOS fund_flow ready.")
        return 0

    oos_stats = run_oos_single(panel, best_lam, best_calmar_is, is_df_sorted)

    print("\n" + "=" * 72)
    print("=== FINAL COMPARISON v20 super_big_net ===")
    print("=" * 72)
    # Baselines (per project memory):
    # Phase 2 clean v2 baseline: Calmar 0.42 Sharpe 0.68 ann 12.45% MDD -29.86%
    # v19.4 (margin sidecar):   Calmar 0.61 Sharpe 0.76 ann 12.86% MDD -21.23%
    # v19.6 (amplitude):        Calmar 0.79 Sharpe -    ann 14.54% MDD -18.51%
    baselines = {
        "Phase2_v2_baseline": {
            "calmar": 0.42, "sharpe": 0.68, "ann": 12.45, "mdd": -29.86,
            "cum": 79.84,
        },
        "v19.4_margin_sidecar": {
            "calmar": 0.61, "sharpe": 0.76, "ann": 12.86, "mdd": -21.23,
            "cum": 83.07,
        },
        "v19.6_amplitude": {
            "calmar": 0.79, "sharpe": None, "ann": 14.54, "mdd": -18.51,
            "cum": None,
        },
    }
    print(f"\nbaseline Phase2_v2: Calmar 0.42 Sharpe 0.68 ann 12.45% "
          f"MDD -29.86% cum 79.84%")
    print(f"v19.4 margin:       Calmar 0.61 Sharpe 0.76 ann 12.86% "
          f"MDD -21.23% cum 83.07%")
    print(f"v19.6 amplitude:    Calmar 0.79 ann 14.54% MDD -18.51%")
    print()
    print(f"v20 super_big_net OOS [λ={best_lam}]:")
    print(f"   Calmar={oos_stats['calmar']}  Sharpe={oos_stats['sharpe']}  "
          f"ann={oos_stats['ann_%']}%  MDD={oos_stats['mdd_%']}%  "
          f"cum={oos_stats['cum_%']}%  win%={oos_stats['win_%']}")

    rel_v194 = (oos_stats["calmar"] - 0.61) / 0.61 * 100
    rel_v196 = (oos_stats["calmar"] - 0.79) / 0.79 * 100
    rel_base = (oos_stats["calmar"] - 0.42) / 0.42 * 100
    print(f"\nrelative Calmar Δ vs baseline (0.42): {rel_base:+.1f}%")
    print(f"relative Calmar Δ vs v19.4    (0.61): {rel_v194:+.1f}%")
    print(f"relative Calmar Δ vs v19.6    (0.79): {rel_v196:+.1f}%")

    # Verdict per user threshold:
    #   ≥ 0.80 → 推荐升级
    #   0.50-0.80 → shadow
    #   < 0.50 → abort
    c = oos_stats["calmar"]
    if c >= 0.80:
        verdict = "推荐升级 (production candidate, OOS Calmar ≥ 0.80)"
    elif c >= 0.50:
        verdict = "shadow 跟踪 (OOS Calmar 0.50-0.80, 不优于 v19.6)"
    else:
        verdict = "abort (OOS Calmar < 0.50, sidecar 不生效或负贡献)"
    print(f"\n[verdict] {verdict}")

    # Save summary md
    summary = []
    summary.append("# Phase B v20 super_big_net Sidecar — 结果")
    summary.append("")
    summary.append(f"**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    summary.append(f"**Base pred cache:** {PRED_CACHE_SRC.name} (Phase 2 v2)")
    summary.append(f"**factor:** net_super_big_5d_chg  sign={FACTOR_SIGN}  "
                   f"horizon=5d")
    summary.append(f"**λ candidates:** {LAMBDA_CANDIDATES}")
    summary.append(f"**IS:** {IS_FIRST} ~ {IS_LAST} (48 months)")
    summary.append(f"**OOS:** {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    summary.append("")
    summary.append("## IS sweep (sorted desc by Calmar)")
    summary.append("")
    summary.append(is_df_sorted.to_markdown(index=False))
    summary.append("")
    summary.append(f"**Locked λ = {best_lam}**  (best IS Calmar = {best_calmar_is})")
    summary.append("")
    summary.append("## OOS single run")
    summary.append("")
    summary.append(f"| metric | value |")
    summary.append(f"|---|---|")
    summary.append(f"| Calmar | {oos_stats['calmar']} |")
    summary.append(f"| Sharpe | {oos_stats['sharpe']} |")
    summary.append(f"| ann %  | {oos_stats['ann_%']} |")
    summary.append(f"| MDD %  | {oos_stats['mdd_%']} |")
    summary.append(f"| cum %  | {oos_stats['cum_%']} |")
    summary.append(f"| win %  | {oos_stats['win_%']} |")
    summary.append(f"| n_months | {oos_stats['n']} |")
    summary.append(f"| avg_picks | {oos_stats['avg_picks']} |")
    summary.append("")
    summary.append("## Comparison")
    summary.append("")
    summary.append("| version | Calmar | Sharpe | ann % | MDD % | cum % |")
    summary.append("|---|---|---|---|---|---|")
    for k, v in baselines.items():
        summary.append(
            f"| {k} | {v['calmar']} | "
            f"{v['sharpe'] if v['sharpe'] is not None else '—'} | "
            f"{v['ann']} | {v['mdd']} | "
            f"{v['cum'] if v['cum'] is not None else '—'} |"
        )
    summary.append(
        f"| **v20** | **{oos_stats['calmar']}** | "
        f"{oos_stats['sharpe']} | {oos_stats['ann_%']} | "
        f"{oos_stats['mdd_%']} | {oos_stats['cum_%']} |"
    )
    summary.append("")
    summary.append(f"**Verdict:** {verdict}")
    OUT_SUMMARY.write_text("\n".join(summary))
    print(f"\n[saved] {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
