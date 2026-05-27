"""v19.10 stacked sidecar — 严格 60 月 OOS single run.

Stacked composition (both factors IS-locked, NOT re-swept):

    final_score = z(pred) - 0.30 × z(amp_imb_20d) + 0.10 × z(JZF)
                            ^^^                    ^^^
                            sign=-1 λ=0.30         sign=+1 λ=0.10
                            (v19.6 production)     (竞价勾魂翻 IS-locked)

Background (per task brief & MEMORY):
  - v19.6 production: amp_imb_20d sign=-1 λ=0.30 → OOS Calmar 1.29 (Phase 2 v3)
  - 竞价勾魂翻 single sidecar: JZF sign=+1 λ=0.10 → OOS Calmar 1.27
  - Spearman |ρ|(JZF, amp_imb_20d) IS 60 月 = 0.048 (极度独立)
  - Historical v19.7 stacked abort: weak m5 (OOS 0.61) 拖累 strong a20

This run: BOTH constituents 单跑都强, hypothesis 是 0.048 独立性允许 stack.

严格协议:
  - NOT re-sweeping λ — λ_amp=0.30, λ_JZF=0.10 each IS-locked, fixed
  - NOT re-sweeping sign — sign_amp=-1, sign_JZF=+1 each IS-decided, fixed
  - Single OOS run: 2021-05 ~ 2026-04 (60 months)
  - production untouched

JZF factor: cross-section z-score of (open - prev_close) / prev_close (qfq)
  ≡ TDX formula `JZF:=100*(O-REF(C,1))/REF(C,1)` from indicator
    b9c99f5c-6e8d-4a8b-818f-0b48a023effe (竞价勾魂翻), output col 开盘涨幅.
  Scaling by 100 doesn't affect z-score.

amp_imb_20d factor: 复用 v19.6 / phase_b helper (`_factor_kline_panel.py`).

Inputs (read-only):
  data_cache/v17_dens_train24_predictions.parquet  (Phase 2 v3 retrained)
  data_cache/baidu_kline.parquet                   (qfq, production-consistent)
  data_cache/qlib_baidu/                            (qlib provider)

Outputs (under data_cache/factors/):
  v19_10_stacked_oos.csv         — IS + OOS metrics summary
  v19_10_stacked_oos_equity.csv  — 60 month equity curve

约束:
  - 不动 production paper_trade_today.py / paper_trade_v19_4.py /
    forward_oos_monitor.py / strategy_v19_*.py / portfolio_excel
  - 不动 main predictions parquet (只读, scratch copy 写 ADJ_PRED)
  - 不 commit
"""
from __future__ import annotations

import sys
import time
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
    build_pit_panel_on_pred_axis,
    _zscore_cs,
    _instrument_to_code6,
)

# ---------------------------------------------------------------------------
# Config (locked, NOT swept)
# ---------------------------------------------------------------------------
ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
KLINE_QFQ = ROOT / "data_cache" / "baidu_kline.parquet"
QLIB_DIR = ROOT / "data_cache" / "qlib_baidu"
ADJ_PRED = ROOT / "data_cache" / "v19_10_stacked_predictions.parquet"  # scratch

OUT_DIR = ROOT / "data_cache" / "factors"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SUMMARY = OUT_DIR / "v19_10_stacked_oos.csv"
OUT_EQUITY = OUT_DIR / "v19_10_stacked_oos_equity.csv"

# Locked sidecar coefficients (IS-locked elsewhere; NOT swept here)
LAMBDA_AMP = 0.30      # v19.6 production locked
SIGN_AMP = -1
LAMBDA_JZF = 0.10      # 竞价勾魂翻 IS-locked
SIGN_JZF = +1

# IS / OOS windows
IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# Reference (per MEMORY / phase_b_60m_oos_results.csv)
REF_V19_6 = {"calmar": 1.29, "sharpe": 0.92, "ann": 34.36, "mdd": -26.66,
             "cum": 337.80, "label": "v19.6 amp_imb_20d λ=0.30 (PRODUCTION)"}
REF_JZF = {"calmar": 1.27, "sharpe": 0.89, "ann": 35.75, "mdd": -28.22,
           "cum": 360.98, "label": "竞价勾魂翻 JZF λ=0.10 (single sidecar)"}


# ---------------------------------------------------------------------------
# JZF factor computation (qfq, daily, cross-section z-scored later)
# ---------------------------------------------------------------------------
def compute_jzf_panel(min_date: str = "2014-01-01") -> pd.DataFrame:
    """JZF (开盘涨幅) = (open - prev_close) / prev_close, per (code, date).

    Returns DataFrame [code, date, jzf]. NaN where prev_close invalid.
    """
    print(f"[jzf] loading qfq baidu_kline.parquet for JZF...", flush=True)
    k = pd.read_parquet(
        KLINE_QFQ, columns=["code", "date", "open", "close"]
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k["date"] = pd.to_datetime(k["date"])
    k = k[k["date"] >= pd.Timestamp(min_date)].copy()
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"[jzf] rows={len(k):,} codes={k['code'].nunique()} "
          f"{k['date'].min().date()} ~ {k['date'].max().date()}", flush=True)

    k["prev_close"] = k.groupby("code", sort=False)["close"].shift(1)
    valid = k["prev_close"].notna() & (k["prev_close"] > 0)
    k["jzf"] = np.where(valid, (k["open"] - k["prev_close"]) / k["prev_close"],
                         np.nan)
    k.loc[~valid, "jzf"] = np.nan
    out = k[["code", "date", "jzf"]].copy()
    print(f"[jzf] non-null={out['jzf'].notna().sum():,} "
          f"({out['jzf'].notna().mean()*100:.1f}%)", flush=True)
    return out


def build_jzf_pit_panel(pred_dt: pd.DatetimeIndex) -> pd.DataFrame:
    """PIT join JZF onto pred axis, cross-section z-score per date.

    Returns DataFrame [datetime, instrument, z_jzf].
    """
    jzf_panel = compute_jzf_panel()

    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)

    # PIT join per code
    print(f"[jzf] PIT join JZF → pred axis ({len(pred_dt)} dates)...", flush=True)
    jzf_sorted = jzf_panel.sort_values(["code", "date"])
    parts: list[pd.DataFrame] = []
    for code, sub in jzf_sorted.groupby("code", sort=False):
        dates_arr = sub["date"].values
        idx = np.searchsorted(dates_arr, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe_idx = np.clip(idx, 0, len(sub) - 1)
        v = sub["jzf"].values
        f_arr = np.where(valid, v[safe_idx], np.nan)
        parts.append(pd.DataFrame({
            "datetime": pred_dt, "code": code, "jzf": f_arr,
        }))
    if not parts:
        raise RuntimeError("[jzf] no PIT panel rows built")
    panel = pd.concat(parts, ignore_index=True)

    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    out = pred_axis.merge(panel, on=["datetime", "code"], how="left")

    n_total = len(out)
    n_ok = out["jzf"].notna().sum()
    print(f"[jzf] PIT rows={n_total:,} non-null={n_ok:,} "
          f"({n_ok/n_total*100:.1f}%)", flush=True)

    out["z_jzf"] = out.groupby("datetime")["jzf"].transform(_zscore_cs)
    out["z_jzf"] = out["z_jzf"].fillna(0.0)
    return out[["datetime", "instrument", "z_jzf"]]


# ---------------------------------------------------------------------------
# Stacked sidecar: write adjusted predictions
# ---------------------------------------------------------------------------
def write_stacked_predictions(z_amp_panel: pd.DataFrame,
                               z_jzf_panel: pd.DataFrame) -> Path:
    """final = z(pred) + sign_amp × λ_amp × z(amp_imb_20d) + sign_jzf × λ_jzf × z(JZF)

    Equivalent to: final = z(pred) - 0.30 × z(amp_imb_20d) + 0.10 × z(JZF)
    """
    print(f"[stack] composing final_score = z(pred) "
          f"{SIGN_AMP:+d}×{LAMBDA_AMP}×z(amp_imb_20d) "
          f"{SIGN_JZF:+d}×{LAMBDA_JZF}×z(jzf)...", flush=True)
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(z_amp_panel, on=["datetime", "instrument"], how="left")
    merged = merged.merge(z_jzf_panel, on=["datetime", "instrument"], how="left")
    merged["z_amp_imb_20d"] = merged["z_amp_imb_20d"].fillna(0.0)
    merged["z_jzf"] = merged["z_jzf"].fillna(0.0)

    merged["final_score"] = (
        merged["z_pred"]
        + SIGN_AMP * LAMBDA_AMP * merged["z_amp_imb_20d"]
        + SIGN_JZF * LAMBDA_JZF * merged["z_jzf"]
    )

    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)

    n_amp_zero = (merged["z_amp_imb_20d"] == 0.0).sum()
    n_jzf_zero = (merged["z_jzf"] == 0.0).sum()
    print(f"[stack] adjusted predictions written: rows={len(out):,}, "
          f"z_amp=0 share={n_amp_zero/len(merged)*100:.1f}%, "
          f"z_jzf=0 share={n_jzf_zero/len(merged)*100:.1f}%", flush=True)
    return ADJ_PRED


# ---------------------------------------------------------------------------
# Spearman ρ sanity check (only on IS window, for verification)
# ---------------------------------------------------------------------------
def spearman_corr_amp_jzf(z_amp_panel: pd.DataFrame,
                           z_jzf_panel: pd.DataFrame) -> dict:
    """Cross-section Spearman ρ (raw factor, not z) — monthly mean & overall.

    Sanity check: should be ~ 0.048 per task brief.
    Use IS window for replication.
    """
    print(f"\n[sanity] Spearman ρ (amp_imb_20d vs JZF) IS window {IS_FIRST}~{IS_LAST}...",
          flush=True)
    merged = z_amp_panel.merge(z_jzf_panel, on=["datetime", "instrument"], how="inner")
    is_start = pd.Timestamp(IS_FIRST + "-01")
    is_end = pd.Timestamp(IS_LAST + "-01") + pd.offsets.MonthEnd(1)
    is_slice = merged[(merged["datetime"] >= is_start) &
                       (merged["datetime"] <= is_end)].copy()
    if is_slice.empty:
        print("[sanity] empty IS slice — skipping ρ.", flush=True)
        return {"is_rho_mean_abs": None, "is_rho_mean": None,
                "is_rho_min": None, "is_rho_max": None}

    rhos = []
    for dt, sub in is_slice.groupby("datetime"):
        if len(sub) < 30:
            continue
        x = sub["z_amp_imb_20d"].rank()
        y = sub["z_jzf"].rank()
        if x.std() == 0 or y.std() == 0:
            continue
        rho = x.corr(y)
        if pd.notna(rho):
            rhos.append(rho)
    if not rhos:
        print("[sanity] no cross-section ρ computed.", flush=True)
        return {"is_rho_mean_abs": None, "is_rho_mean": None,
                "is_rho_min": None, "is_rho_max": None}
    arr = np.array(rhos)
    res = {
        "is_rho_n_dates": len(rhos),
        "is_rho_mean": round(float(arr.mean()), 4),
        "is_rho_mean_abs": round(float(np.mean(np.abs(arr))), 4),
        "is_rho_min": round(float(arr.min()), 4),
        "is_rho_max": round(float(arr.max()), 4),
        "is_rho_std": round(float(arr.std()), 4),
    }
    print(f"[sanity] IS Spearman: n_dates={res['is_rho_n_dates']}, "
          f"mean={res['is_rho_mean']:+.4f}, mean|ρ|={res['is_rho_mean_abs']}, "
          f"range=[{res['is_rho_min']:+.4f}, {res['is_rho_max']:+.4f}]",
          flush=True)
    return res


# ---------------------------------------------------------------------------
# Walkforward helpers (mirror phase_b_oos_60m_candidates.py)
# ---------------------------------------------------------------------------
_v17 = None
_proxy = None
_qlib_inited = False


def _ensure_qlib():
    global _v17, _proxy, _qlib_inited
    if _qlib_inited:
        return
    import qlib  # noqa: F401
    from qlib.constant import REG_CN
    import strategy_v17_dens_grid as v17

    qlib.init(provider_uri=str(QLIB_DIR), region=REG_CN)
    _proxy = v17.build_market_proxy()
    v17.MARKET = "csi300"
    v17.TRAIN_MONTHS = 24
    v17.K_NORMAL = 8
    v17.DROP_NORMAL = 2
    v17.PORTFOLIO_VALUE = 5e4
    v17.STOP_LOSS_PCT = 0.0
    v17.VOL_TARGET_ANN = 0.0
    _v17 = v17
    _qlib_inited = True


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
    _ensure_qlib()
    _v17.PRED_CACHE = ADJ_PRED
    _v17._pred_disk_df = None
    _v17._pred_cache.clear()

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
            res = _v17.realistic_window(m, _proxy, with_regime=False)
            res["month"] = m.strftime("%Y-%m")
            rows.append(res)
        except Exception as e:
            print(f"    [{tag}] {i}/{len(months)} {m.strftime('%Y-%m')} FAIL: "
                  f"{str(e)[:120]}", flush=True)
            rows.append({
                "month": m.strftime("%Y-%m"),
                "abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                "regime_days": "", "n_skipped_limit": 0, "n_stop_loss": 0,
            })
        if i == 1 or i % 12 == 0 or i == len(months):
            print(f"    [{tag}] {i:3d}/{len(months)} {rows[-1]['month']}: "
                  f"abs_ret={rows[-1]['abs_ret_%']:+6.2f}%  "
                  f"picks={rows[-1]['avg_picks']:.1f}", flush=True)

    df = pd.DataFrame(rows)
    stats = _annualize(df["abs_ret_%"])
    stats["avg_picks"] = round(df["avg_picks"].mean(), 2)
    stats["months_df"] = df
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _verdict(oos_calmar: float, v19_6: float = 1.29) -> str:
    if oos_calmar > v19_6:
        return "BEAT v19.6 — recommend production upgrade"
    if oos_calmar >= 1.20:
        return "marginal — not clearly above v19.6, hold"
    if oos_calmar > 0.77:
        return "below v19.6 + below singles — weak stack"
    return "ABORT — stacked drag like v19.7"


def main() -> int:
    t0 = time.time()

    print("=" * 72)
    print("v19.10 stacked sidecar — 严格 60 月 OOS single run")
    print("=" * 72)
    print(f"Formula : final = z(pred) "
          f"{SIGN_AMP:+d}×{LAMBDA_AMP}×z(amp_imb_20d) "
          f"{SIGN_JZF:+d}×{LAMBDA_JZF}×z(JZF)")
    print(f"IS      : {IS_FIRST} ~ {IS_LAST}  (sanity Spearman check)")
    print(f"OOS     : {OOS_FIRST} ~ {OOS_LAST}  (60 months, single run, no sweep)")
    print(f"Pred    : {ORIG_PRED.name} (Phase 2 v3 retrained, read-only)")
    print(f"Kline   : {KLINE_QFQ.name} (qfq, production-consistent)")
    print()

    # Step 1: build z_amp_imb_20d on pred axis (use existing helper)
    print("[step 1] build amp_imb_20d PIT panel (re-use helper)...", flush=True)
    z_amp = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["amp_imb_20d"], min_date="2014-01-01"
    )
    print(f"[step 1] z_amp rows={len(z_amp):,}", flush=True)

    # Step 2: build z_jzf on pred axis
    print("\n[step 2] build JZF PIT panel...", flush=True)
    pred_dt = pd.DatetimeIndex(
        sorted(pd.read_parquet(ORIG_PRED, columns=["datetime"])["datetime"].unique())
    )
    z_jzf = build_jzf_pit_panel(pred_dt)
    print(f"[step 2] z_jzf rows={len(z_jzf):,}", flush=True)

    # Step 3: sanity Spearman ρ check (IS window)
    rho = spearman_corr_amp_jzf(z_amp, z_jzf)

    # Step 4: write stacked adjusted predictions
    print("\n[step 4] write stacked predictions...", flush=True)
    write_stacked_predictions(z_amp, z_jzf)

    # Step 5: walkforward OOS single run
    print(f"\n[step 5] walkforward OOS single run "
          f"{OOS_FIRST}~{OOS_LAST} (60 months)...", flush=True)
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "v19_10_OOS")
    oos_df = oos_stats.pop("months_df")
    oos_df.to_csv(OUT_EQUITY, index=False)
    print(f"[saved equity] {OUT_EQUITY}")

    # Step 6: summary csv + report
    summary_row = {
        "strategy": "v19.10_stacked",
        "formula": f"z(pred) {SIGN_AMP:+d}*{LAMBDA_AMP}*z(amp_imb_20d) "
                   f"{SIGN_JZF:+d}*{LAMBDA_JZF}*z(jzf)",
        "lambda_amp": LAMBDA_AMP, "sign_amp": SIGN_AMP,
        "lambda_jzf": LAMBDA_JZF, "sign_jzf": SIGN_JZF,
        "oos_first": OOS_FIRST, "oos_last": OOS_LAST,
        "oos_n_months": oos_stats["n"],
        "oos_calmar": oos_stats["calmar"],
        "oos_sharpe": oos_stats["sharpe"],
        "oos_ann_%": oos_stats["ann_%"],
        "oos_mdd_%": oos_stats["mdd_%"],
        "oos_cum_%": oos_stats["cum_%"],
        "oos_win_%": oos_stats["win_%"],
        "oos_avg_picks": oos_stats["avg_picks"],
        "vs_v19_6_calmar_pct": round(
            (oos_stats["calmar"] - REF_V19_6["calmar"]) / REF_V19_6["calmar"] * 100, 2
        ),
        "vs_jzf_calmar_pct": round(
            (oos_stats["calmar"] - REF_JZF["calmar"]) / REF_JZF["calmar"] * 100, 2
        ),
        "is_spearman_mean_abs_rho": rho.get("is_rho_mean_abs"),
        "verdict": _verdict(oos_stats["calmar"]),
    }
    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(OUT_SUMMARY, index=False)
    print(f"[saved summary] {OUT_SUMMARY}")

    wall = time.time() - t0

    # Markdown report
    print("\n" + "=" * 72)
    print(f"DONE.  wall time = {wall/60:.1f} min")
    print("=" * 72)
    print()
    print("| Strategy | Calmar | Sharpe | ann% | MDD% | cum% |")
    print("|---|---|---|---|---|---|")
    print(f"| v19.6 (主因子 amp_imb_20d) | "
          f"{REF_V19_6['calmar']} | {REF_V19_6['sharpe']} | "
          f"{REF_V19_6['ann']} | {REF_V19_6['mdd']} | {REF_V19_6['cum']} |")
    print(f"| 竞价勾魂翻 (单 JZF) | "
          f"{REF_JZF['calmar']} | {REF_JZF['sharpe']} | "
          f"{REF_JZF['ann']} | {REF_JZF['mdd']} | {REF_JZF['cum']} |")
    print(f"| **v19.10 stacked** | "
          f"**{oos_stats['calmar']}** | **{oos_stats['sharpe']}** | "
          f"**{oos_stats['ann_%']}** | **{oos_stats['mdd_%']}** | "
          f"**{oos_stats['cum_%']}** |")
    print()
    print(f"Verdict: **{summary_row['verdict']}**")
    print(f"  vs v19.6 ({REF_V19_6['calmar']}): {summary_row['vs_v19_6_calmar_pct']:+.2f}%")
    print(f"  vs JZF single ({REF_JZF['calmar']}): "
          f"{summary_row['vs_jzf_calmar_pct']:+.2f}%")
    if rho.get("is_rho_mean_abs") is not None:
        print(f"  IS Spearman |ρ| (amp vs JZF): {rho['is_rho_mean_abs']} "
              f"(brief expects ~0.048)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
