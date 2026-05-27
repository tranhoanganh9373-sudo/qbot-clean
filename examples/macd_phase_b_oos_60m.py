"""Phase B 严格 OOS 回测 — 活法波段 MACD 金叉策略 3 variants.

mt180 销量第一公式 (52,536). 之前 IC 评估的是 first output col `MA1` (实际是 MA5),
不是真 MACD 信号. 本任务测真 MACD histogram 信号.

因子定义:
    DIF = EMA(CLOSE, 12) - EMA(CLOSE, 26)
    DEA = EMA(DIF, 9)
    MACD_hist = (DIF - DEA) * 2

3 个变体 (各独立 IS sweep + OOS 单跑):
  1. macd_hist        (continuous) — DIF-DEA 连续值, sign 由 IS sweep 选 ±1
  2. macd_cross_binary (binary)    — CROSS(DIF, DEA) 1/0, sign=+1
  3. macd_triple_cross (三重共振)  — CROSS(DIF, DEA) AND CROSS(K, D)
                                     AND CROSS(MA5, MA10), 1/0, sign=+1

严格 Phase B 协议:
    IS  : 2017-01 ~ 2020-12 (48 months, 跟 v19.6/v19.4 一致)
    OOS : 2021-05 ~ 2026-04 (60 months)
    λ sweep ∈ {0.10, 0.20, 0.30} (锁后不允许 fine-tune)
    final = z(pred) + λ * sign * z(factor)
    Top 8 daily rebalance via v17_dens_train24 + Alpha158 + DEnsemble

Run:
  .venv/bin/python examples/macd_phase_b_oos_60m.py
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

from _factor_kline_panel import _instrument_to_code6, _zscore_cs  # noqa: E402

# -------- Paths --------
ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
KLINE_PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"
ADJ_PRED = ROOT / "data_cache" / "macd_phase_b_adj_predictions.parquet"

OUT_DIR = ROOT / "data_cache" / "factors"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SUMMARY_CSV = OUT_DIR / "macd_huofa_phase_b_60m.csv"
OUT_SUMMARY_MD = OUT_DIR / "macd_huofa_summary.md"

# -------- Window --------
IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# -------- Sweep --------
LAMBDAS = [0.10, 0.20, 0.30]


# ============================================================
# Factor computation
# ============================================================

def _ema(s: np.ndarray, span: int) -> np.ndarray:
    """Standard pandas EMA with adjust=False (recursive)."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(s, dtype=float)
    out[:] = np.nan
    first_valid = -1
    for i, v in enumerate(s):
        if np.isfinite(v):
            first_valid = i
            out[i] = v
            break
    if first_valid < 0:
        return out
    prev = out[first_valid]
    for i in range(first_valid + 1, len(s)):
        v = s[i]
        if np.isfinite(v):
            prev = alpha * v + (1 - alpha) * prev
        out[i] = prev
    return out


def _tdx_sma(s: np.ndarray, n: int, m: int) -> np.ndarray:
    """通达信 SMA(X, N, M) — recursive: Y = (m*X + (n-m)*prev) / n"""
    out = np.empty_like(s, dtype=float)
    out[:] = np.nan
    first_valid = -1
    for i, v in enumerate(s):
        if np.isfinite(v):
            first_valid = i
            out[i] = v
            break
    if first_valid < 0:
        return out
    prev = out[first_valid]
    for i in range(first_valid + 1, len(s)):
        v = s[i]
        if np.isfinite(v):
            prev = (m * v + (n - m) * prev) / n
        out[i] = prev
    return out


def _cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CROSS(a, b): a 上穿 b. Today: a >= b AND prev: a < b. Returns 1/0/NaN."""
    out = np.zeros_like(a, dtype=float)
    out[0] = np.nan
    for i in range(1, len(a)):
        if not (np.isfinite(a[i]) and np.isfinite(b[i])
                and np.isfinite(a[i-1]) and np.isfinite(b[i-1])):
            out[i] = np.nan
        elif a[i] >= b[i] and a[i-1] < b[i-1]:
            out[i] = 1.0
        else:
            out[i] = 0.0
    return out


def compute_macd_factors(min_date: str = "2014-01-01") -> pd.DataFrame:
    """Compute the 3 MACD variant factors per (code, date).

    Returns columns:
        code, date, macd_hist, macd_cross_binary, macd_triple_cross
    """
    print(f"[kline] loading {KLINE_PARQUET.name}...", flush=True)
    k = pd.read_parquet(
        KLINE_PARQUET,
        columns=["code", "date", "high", "low", "close"],
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k["date"] = pd.to_datetime(k["date"])
    k = k[k["date"] >= pd.Timestamp(min_date)].copy()
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"[kline] rows={len(k):,}  codes={k['code'].nunique()}  "
          f"date range {k['date'].min().date()} ~ {k['date'].max().date()}",
          flush=True)

    print("[macd] computing per-code factors (EMA12/26, DIF/DEA, KDJ, MA5/10)...",
          flush=True)
    parts = []
    n_codes = k["code"].nunique()
    t0 = time.time()
    last_log = t0
    done = 0
    for code, sub in k.groupby("code", sort=False):
        sub = sub.sort_values("date").reset_index(drop=True)
        close = sub["close"].to_numpy(dtype=float)
        high = sub["high"].to_numpy(dtype=float)
        low = sub["low"].to_numpy(dtype=float)

        # Guard NaN/0 prices
        close = np.where((np.isfinite(close)) & (close > 0), close, np.nan)
        high = np.where((np.isfinite(high)) & (high > 0), high, np.nan)
        low = np.where((np.isfinite(low)) & (low > 0), low, np.nan)

        # MACD
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        dif = ema12 - ema26
        dea = _ema(dif, 9)
        macd_hist = (dif - dea) * 2.0

        # MACD cross
        cross_macd = _cross_up(dif, dea)

        # KDJ 9: RSV = (close - LLV(low,9)) / (HHV(high,9) - LLV(low,9)) * 100
        s_low = pd.Series(low)
        s_high = pd.Series(high)
        llv = s_low.rolling(9, min_periods=9).min().to_numpy()
        hhv = s_high.rolling(9, min_periods=9).max().to_numpy()
        denom = hhv - llv
        with np.errstate(divide="ignore", invalid="ignore"):
            rsv = np.where(denom > 0, (close - llv) / denom * 100.0, np.nan)
        kdj_k = _tdx_sma(rsv, 9, 1)
        kdj_d = _tdx_sma(kdj_k, 9, 1)
        cross_kd = _cross_up(kdj_k, kdj_d)

        # MA5 / MA10
        s_close = pd.Series(close)
        ma5 = s_close.rolling(5, min_periods=5).mean().to_numpy()
        ma10 = s_close.rolling(10, min_periods=10).mean().to_numpy()
        cross_ma = _cross_up(ma5, ma10)

        # Triple cross: all three on same day
        valid_mask = (np.isfinite(cross_macd) & np.isfinite(cross_kd)
                      & np.isfinite(cross_ma))
        triple_f = np.where(
            valid_mask,
            ((cross_macd == 1.0) & (cross_kd == 1.0)
             & (cross_ma == 1.0)).astype(float),
            np.nan,
        )

        parts.append(pd.DataFrame({
            "code": code,
            "date": sub["date"].values,
            "macd_hist": macd_hist,
            "macd_cross_binary": cross_macd,
            "macd_triple_cross": triple_f,
        }))
        done += 1
        now = time.time()
        if now - last_log > 5.0:
            print(f"  [macd] {done}/{n_codes} codes  "
                  f"elapsed={now - t0:.0f}s", flush=True)
            last_log = now

    out = pd.concat(parts, ignore_index=True)
    print(f"[macd] computed rows={len(out):,}  elapsed={time.time() - t0:.0f}s",
          flush=True)
    # Coverage / signal density
    for col in ("macd_hist", "macd_cross_binary", "macd_triple_cross"):
        cov = out[col].notna().mean() * 100
        if col != "macd_hist":
            n1 = (out[col] == 1.0).sum()
            print(f"  {col}: coverage={cov:.1f}%, ones={n1:,}", flush=True)
        else:
            print(f"  {col}: coverage={cov:.1f}%", flush=True)
    return out


def build_pit_panel(factors_df: pd.DataFrame,
                    pred_path: Path,
                    factor_col: str) -> pd.DataFrame:
    """PIT panel: for each (pred_datetime, instrument), latest factor at date <= datetime.

    Returns DataFrame: [datetime, instrument, z_<factor_col>] cross-section z-scored.
    """
    pred = pd.read_parquet(pred_path, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))
    print(f"[panel] {factor_col}: pred dates={len(pred_dt)}, "
          f"unique insts={pred['instrument'].nunique()}", flush=True)

    sub_factors = factors_df[["code", "date", factor_col]].copy()
    sub_factors = sub_factors.sort_values(["code", "date"])

    parts = []
    for code, sub in sub_factors.groupby("code", sort=False):
        sub = sub.sort_values("date")
        dates_arr = sub["date"].values
        vals = sub[factor_col].values
        idx = np.searchsorted(dates_arr, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe_idx = np.clip(idx, 0, len(sub) - 1)
        v = np.where(valid, vals[safe_idx], np.nan)
        parts.append(pd.DataFrame({
            "datetime": pred_dt,
            "code": code,
            factor_col: v,
        }))
    panel = pd.concat(parts, ignore_index=True)

    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    out = pred_axis.merge(panel, on=["datetime", "code"], how="left")
    n_ok = out[factor_col].notna().sum()
    print(f"[panel] {factor_col}: pred-axis rows={len(out):,}, "
          f"non-nan={n_ok:,} ({n_ok/len(out)*100:.1f}%)", flush=True)

    zcol = f"z_{factor_col}"
    out[zcol] = out.groupby("datetime")[factor_col].transform(_zscore_cs)
    out[zcol] = out[zcol].fillna(0.0)
    return out[["datetime", "instrument", zcol]]


# ============================================================
# Adjusted predictions + walkforward
# ============================================================

def build_adjusted_predictions(panel: pd.DataFrame,
                               factor_col: str,
                               lam: float, sign: int,
                               label: str) -> Path:
    """final = z(pred) + sign * lam * z(factor). Writes ADJ_PRED parquet."""
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(panel, on=["datetime", "instrument"], how="left")
    zcol = f"z_{factor_col}"
    merged[zcol] = merged[zcol].fillna(0.0)
    merged["final_score"] = merged["z_pred"] + sign * lam * merged[zcol]
    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
    print(f"    [adj] {label} {factor_col} λ={lam} sign={sign:+d} "
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
                print(f"      {i:3d}/{len(months)} {res['month']}: "
                      f"abs_ret={res['abs_ret_%']:+6.2f}%  "
                      f"picks={res['avg_picks']:.1f}", flush=True)
        except Exception as e:
            print(f"      {i:3d}/{len(months)} {m.strftime('%Y-%m')} FAIL: "
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


# ============================================================
# Per-variant IS sweep + OOS single run
# ============================================================

def run_variant_macd_hist(factors: pd.DataFrame) -> dict:
    """macd_hist (continuous) — sign ±1 sweep × λ ∈ {0.10,0.20,0.30}.

    6 IS runs + 1 OOS run.
    """
    print("\n" + "=" * 70)
    print("VARIANT 1/3: macd_hist (continuous DIF-DEA*2)")
    print("=" * 70)

    panel = build_pit_panel(factors, ORIG_PRED, "macd_hist")

    is_rows = []
    for sign in (+1, -1):
        for lam in LAMBDAS:
            label = f"hist_s{'p' if sign > 0 else 'n'}_l{int(lam*100):03d}"
            print(f"\n  --- IS {label}  sign={sign:+d} λ={lam} ---")
            build_adjusted_predictions(panel, "macd_hist", lam, sign, label)
            stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{label}")
            is_rows.append({
                "label": label, "sign": sign, "lam": lam,
                "is_calmar": stats["calmar"], "is_sharpe": stats["sharpe"],
                "is_ann": stats["ann_%"], "is_mdd": stats["mdd_%"],
            })
            print(f"    >> IS Calmar={stats['calmar']} "
                  f"Sharpe={stats['sharpe']} ann={stats['ann_%']}% "
                  f"MDD={stats['mdd_%']}%")

    is_df = pd.DataFrame(is_rows).sort_values("is_calmar", ascending=False)
    print("\n  === IS 6-combo Calmar (sorted desc) ===")
    print(is_df.to_string(index=False))
    best = is_df.iloc[0]
    locked_lam = float(best["lam"])
    locked_sign = int(best["sign"])
    locked_label = best["label"]
    print(f"\n  [lock] {locked_label}  sign={locked_sign:+d}  "
          f"λ={locked_lam}  IS Calmar={best['is_calmar']}")

    print(f"\n  [OOS] single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    build_adjusted_predictions(panel, "macd_hist", locked_lam, locked_sign,
                               locked_label)
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_macd_hist")
    eq_df = oos_stats.pop("months_df")
    eq_path = OUT_DIR / "macd_hist_oos_equity.csv"
    eq_df.to_csv(eq_path, index=False)
    print(f"  [saved] {eq_path}")

    return {
        "variant": "macd_hist",
        "locked_lambda": locked_lam,
        "locked_sign": locked_sign,
        "is_calmar": float(best["is_calmar"]),
        "oos_calmar": oos_stats["calmar"],
        "oos_sharpe": oos_stats["sharpe"],
        "oos_mdd": oos_stats["mdd_%"],
        "oos_ann": oos_stats["ann_%"],
        "oos_cum": oos_stats["cum_%"],
        "oos_n": oos_stats["n"],
    }


def run_variant_binary(factors: pd.DataFrame,
                       factor_col: str,
                       title: str) -> dict:
    """macd_cross_binary or macd_triple_cross — sign fixed +1, λ sweep only."""
    print("\n" + "=" * 70)
    print(f"VARIANT {title}: {factor_col} (binary, sign=+1)")
    print("=" * 70)

    panel = build_pit_panel(factors, ORIG_PRED, factor_col)

    is_rows = []
    sign = +1
    for lam in LAMBDAS:
        label = f"{factor_col}_sp_l{int(lam*100):03d}"
        print(f"\n  --- IS {label}  sign={sign:+d} λ={lam} ---")
        build_adjusted_predictions(panel, factor_col, lam, sign, label)
        stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{label}")
        is_rows.append({
            "label": label, "sign": sign, "lam": lam,
            "is_calmar": stats["calmar"], "is_sharpe": stats["sharpe"],
            "is_ann": stats["ann_%"], "is_mdd": stats["mdd_%"],
        })
        print(f"    >> IS Calmar={stats['calmar']} Sharpe={stats['sharpe']} "
              f"ann={stats['ann_%']}% MDD={stats['mdd_%']}%")

    is_df = pd.DataFrame(is_rows).sort_values("is_calmar", ascending=False)
    print("\n  === IS 3-combo Calmar (sorted desc) ===")
    print(is_df.to_string(index=False))
    best = is_df.iloc[0]
    locked_lam = float(best["lam"])
    locked_sign = int(best["sign"])
    locked_label = best["label"]
    print(f"\n  [lock] {locked_label}  sign={locked_sign:+d}  "
          f"λ={locked_lam}  IS Calmar={best['is_calmar']}")

    print(f"\n  [OOS] single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    build_adjusted_predictions(panel, factor_col, locked_lam, locked_sign,
                               locked_label)
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, f"OOS_{factor_col}")
    eq_df = oos_stats.pop("months_df")
    eq_path = OUT_DIR / f"{factor_col}_oos_equity.csv"
    eq_df.to_csv(eq_path, index=False)
    print(f"  [saved] {eq_path}")

    return {
        "variant": factor_col,
        "locked_lambda": locked_lam,
        "locked_sign": locked_sign,
        "is_calmar": float(best["is_calmar"]),
        "oos_calmar": oos_stats["calmar"],
        "oos_sharpe": oos_stats["sharpe"],
        "oos_mdd": oos_stats["mdd_%"],
        "oos_ann": oos_stats["ann_%"],
        "oos_cum": oos_stats["cum_%"],
        "oos_n": oos_stats["n"],
    }


# ============================================================
# Main
# ============================================================

# Reference baselines (Phase 2 v3 retrained cache, 60m OOS 2021-05~2026-04)
BASELINE_CALMAR = 0.77
V196_CALMAR = 1.29       # v19.6 prod amp_imb_20d λ=0.30
V194_CALMAR = 0.62       # v19.4 shadow margin_5d+20d λ=0.10


def _verdict(oos_calmar: float) -> str:
    if oos_calmar > V196_CALMAR:
        return "BEAT_v19.6"
    if oos_calmar > V194_CALMAR:
        return "BEAT_v19.4_only"
    if oos_calmar > BASELINE_CALMAR:
        return "BEAT_baseline_only"
    return "ABORT"


def main() -> int:
    t_wall = time.time()
    print("=" * 70)
    print("活法波段 MACD Phase B 严格 OOS 60 月回测")
    print("=" * 70)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST} (48 months)")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"λ   : {LAMBDAS}  (no fine-tune)")
    print("variants: macd_hist (sign ±1 sweep), "
          "macd_cross_binary (sign=+1), macd_triple_cross (sign=+1)")
    print()

    print("[step 0] computing MACD factors on baidu_kline.parquet ...")
    factors = compute_macd_factors(min_date="2014-01-01")

    results = []
    results.append(run_variant_macd_hist(factors))
    results.append(run_variant_binary(factors, "macd_cross_binary", "2/3"))
    results.append(run_variant_binary(factors, "macd_triple_cross", "3/3"))

    # Build summary table
    rows = []
    for r in results:
        rows.append({
            "variant": r["variant"],
            "locked_lambda": r["locked_lambda"],
            "locked_sign": r["locked_sign"],
            "is_calmar": r["is_calmar"],
            "oos_calmar": r["oos_calmar"],
            "oos_sharpe": r["oos_sharpe"],
            "oos_mdd": r["oos_mdd"],
            "oos_ann": r["oos_ann"],
            "oos_cum": r["oos_cum"],
            "vs_baseline_pct": round(
                (r["oos_calmar"] - BASELINE_CALMAR) / BASELINE_CALMAR * 100, 1
            ),
            "vs_v196_pct": round(
                (r["oos_calmar"] - V196_CALMAR) / V196_CALMAR * 100, 1
            ),
            "vs_v194_pct": round(
                (r["oos_calmar"] - V194_CALMAR) / V194_CALMAR * 100, 1
            ),
            "verdict": _verdict(r["oos_calmar"]),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\n[saved] {OUT_SUMMARY_CSV}")

    # Markdown report
    wall_min = (time.time() - t_wall) / 60
    md = ["# 活法波段 MACD Phase B 严格 OOS 60 月回测",
          "",
          f"- IS  : {IS_FIRST} ~ {IS_LAST} (48 months)",
          f"- OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)",
          f"- λ sweep: {LAMBDAS} (lock 后不允许 fine-tune)",
          f"- Top K=8 daily rebalance, v17_dens_train24 + Alpha158 + DEnsemble",
          f"- Wall time: **{wall_min:.1f} min**",
          "",
          "## Reference baselines (Phase 2 v3 cache, 60m OOS)",
          "",
          f"- baseline train24 : Calmar={BASELINE_CALMAR}",
          f"- v19.6 production (amp_imb_20d λ=0.30) : Calmar={V196_CALMAR}",
          f"- v19.4 shadow (margin_5d+20d λ=0.10)   : Calmar={V194_CALMAR}",
          "",
          "## Results",
          "",
          "| variant | sign | λ | IS Calmar | OOS Calmar | OOS Sharpe | "
          "OOS MDD% | OOS ann% | OOS cum% | vs baseline | vs v19.6 | vs v19.4 | "
          "verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['locked_sign']:+d} | "
            f"{r['locked_lambda']} | {r['is_calmar']} | "
            f"**{r['oos_calmar']}** | {r['oos_sharpe']} | "
            f"{r['oos_mdd']} | {r['oos_ann']} | {r['oos_cum']} | "
            f"{r['vs_baseline_pct']:+.1f}% | {r['vs_v196_pct']:+.1f}% | "
            f"{r['vs_v194_pct']:+.1f}% | {r['verdict']} |"
        )

    md.append("")
    md.append("## Verdict")
    md.append("")
    beat_196 = [r for r in rows if r["oos_calmar"] > V196_CALMAR]
    if beat_196:
        md.append("**有 variant 击败 v19.6 production**:")
        for r in beat_196:
            md.append(f"- `{r['variant']}` λ={r['locked_lambda']} "
                      f"sign={r['locked_sign']:+d}: OOS Calmar="
                      f"{r['oos_calmar']} vs v19.6={V196_CALMAR}")
    else:
        md.append("**所有 variant 均未击败 v19.6 production**.")
        md.append("维持 v19.6 (amp_imb_20d λ=0.30) 为 production.")
    md.append("")

    OUT_SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {OUT_SUMMARY_MD}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nWall time: {wall_min:.1f} min")

    return 0


if __name__ == "__main__":
    sys.exit(main())
