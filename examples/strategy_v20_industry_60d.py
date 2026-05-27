"""v20 Sidecar (Phase B) — train24 + industry_adj_ret_60d overlay.

Phase A 锁定 (factor_ic_industry_adj_ret_is.csv):
    factor = industry_adj_ret_60d
    sign   = +1 (momentum)
    IS ICIR = +1.112 (32 月, IS 2014-2020)
    λ candidate ∈ {0.05, 0.10, 0.20, 0.30}

公式:
    industry_adj_ret_60d(code, T) =
        close(T)/close(T-60) - 1
        - mean_{j in same SW industry}(close(T)/close(T-60) - 1)
    final = z(train24_pred) + λ * z(industry_adj_ret_60d)

严格 OOS 协议:
- IS 期: 2017-01 ~ 2020-12 (48 月), 仅用于 λ sweep
- OOS 期: 2021-05 ~ 2026-04 (60 月), 锁定 λ 后单次跑
- 锁定后 OOS 不重 sweep, 不改因子定义

Warning: Spearman vs amp_imb_20d 0.29 (Phase A) — 中等相关, sidecar 信息可能
       与 v19.6 部分重叠.  本任务独立锁定 λ; stacking 留 follow-up.

Run:
  .venv/bin/python examples/strategy_v20_industry_60d.py
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

from _factor_kline_panel import _zscore_cs, _instrument_to_code6  # noqa: E402

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
KLINE_PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"
INDUSTRY_PATH = ROOT / "data_cache" / "industry" / "industry_membership.parquet"
ADJ_PRED = ROOT / "data_cache" / "v20_industry_60d_predictions.parquet"
OUT_IS_GRID = ROOT / "examples" / "v20_industry_60d_is_grid.csv"
OUT_OOS_STATS = ROOT / "examples" / "v20_industry_60d_oos_stats.csv"

IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

LAM_CANDIDATES = [0.05, 0.10, 0.20, 0.30]
LOOKBACK_TD = 60  # 60 trading days


def build_industry_60d_panel() -> pd.DataFrame:
    """每个 (code, date) 的 industry_adj_ret_60d, 然后 PIT join 到 pred axis.

    Step 1: 对每只 code, 按交易日索引算 ret_60 = close/close.shift(60) - 1.
    Step 2: 在每个 date 上做横截面 SW level-1 industry demean.
    Step 3: PIT join 到 pred (datetime, instrument) — 取最近 ≤ datetime 的 ret_60d.
    Step 4: cross-section z-score per datetime.

    Returns: DataFrame[datetime, instrument, z_industry_adj_ret_60d]
    """
    print("[panel] loading industry membership...", flush=True)
    ind = pd.read_parquet(INDUSTRY_PATH)
    ind["code"] = ind["code"].astype(str).str.zfill(6)
    ind = ind[["code", "industry_code"]].drop_duplicates(subset=["code"])
    code_to_ind = dict(zip(ind["code"], ind["industry_code"]))
    print(f"[panel] industry membership: {len(ind):,} codes, "
          f"{ind['industry_code'].nunique()} industries", flush=True)

    print("[panel] loading kline (close only) ...", flush=True)
    k = pd.read_parquet(
        KLINE_PARQUET, columns=["code", "date", "close"],
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k["date"] = pd.to_datetime(k["date"])
    # buffer: ample for 60d rolling before 2017-01 anchor (~120 calendar days)
    lo = pd.Timestamp("2016-09-01")
    k = k[k["date"] >= lo].copy()
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"[panel] kline rows={len(k):,}, "
          f"codes={k['code'].nunique()}, "
          f"date range {k['date'].min().date()} ~ {k['date'].max().date()}",
          flush=True)

    # ret_60d per code (lookback 60 trading days)
    grp = k.groupby("code", sort=False)
    k["close_lag60"] = grp["close"].shift(LOOKBACK_TD)
    k["ret_60d"] = np.where(
        (k["close_lag60"] > 0) & k["close_lag60"].notna(),
        k["close"] / k["close_lag60"] - 1,
        np.nan,
    )
    k["industry_code"] = k["code"].map(code_to_ind)
    k = k.dropna(subset=["ret_60d", "industry_code"]).copy()
    print(f"[panel] valid ret_60d rows={len(k):,}", flush=True)

    # cross-section demean per (date, industry_code)
    print("[panel] computing industry demean per date...", flush=True)
    ind_mean = k.groupby(["date", "industry_code"])["ret_60d"].transform("mean")
    k["industry_adj_ret_60d"] = k["ret_60d"] - ind_mean
    factor = k[["code", "date", "industry_adj_ret_60d"]].copy()

    # PIT join to pred axis
    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))
    print(f"[panel] pred dates={len(pred_dt)}, "
          f"unique instruments={pred['instrument'].nunique()}", flush=True)

    factor_sorted = factor.sort_values(["code", "date"])
    parts = []
    for code, sub in factor_sorted.groupby("code", sort=False):
        dates_arr = sub["date"].values
        vals = sub["industry_adj_ret_60d"].values
        idx = np.searchsorted(dates_arr, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe_idx = np.clip(idx, 0, len(sub) - 1)
        v = np.where(valid, vals[safe_idx], np.nan)
        parts.append(pd.DataFrame({
            "datetime": pred_dt,
            "code": code,
            "industry_adj_ret_60d": v,
        }))
    if not parts:
        raise RuntimeError("[panel] PIT join produced 0 rows")
    panel = pd.concat(parts, ignore_index=True)
    print(f"[panel] PIT panel rows={len(panel):,}", flush=True)

    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    out = pred_axis.merge(panel, on=["datetime", "code"], how="left")

    n_ok = out["industry_adj_ret_60d"].notna().sum()
    print(f"[panel] coverage: {n_ok:,}/{len(out):,} "
          f"({n_ok/len(out)*100:.1f}%)", flush=True)

    print("[panel] cross-section z-score per datetime...", flush=True)
    out["z_industry_adj_ret_60d"] = out.groupby("datetime")[
        "industry_adj_ret_60d"
    ].transform(_zscore_cs).fillna(0.0)
    return out[["datetime", "instrument", "z_industry_adj_ret_60d"]]


def build_adjusted_predictions(panel: pd.DataFrame, lam: float,
                               label: str) -> Path:
    """final = z(pred) + lam * z(industry_adj_ret_60d), sign=+1."""
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(panel, on=["datetime", "instrument"], how="left")
    merged["z_industry_adj_ret_60d"] = merged[
        "z_industry_adj_ret_60d"
    ].fillna(0.0)
    merged["final_score"] = (
        merged["z_pred"] + lam * merged["z_industry_adj_ret_60d"]
    )
    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
    print(f"  [adj] {label} λ={lam} rows={len(out):,}", flush=True)
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
    print("Phase B Sidecar v20 — train24 + industry_adj_ret_60d")
    print("=" * 70)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST} (48 months)")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"λ candidates: {LAM_CANDIDATES}")
    print()

    print("[step 1] build industry_adj_ret_60d PIT panel...")
    panel = build_industry_60d_panel()

    print(f"\n[step 2] IS sweep {len(LAM_CANDIDATES)} λ "
          f"({IS_FIRST} ~ {IS_LAST}, 48 months)")
    is_rows = []
    for lam in LAM_CANDIDATES:
        label = f"l{int(lam*100):03d}"
        print(f"\n  --- λ={lam} ---")
        build_adjusted_predictions(panel, lam, label)
        stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{label}")
        row = {
            "lam": lam,
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
    print("\n=== IS λ-sweep Calmar table (sorted desc) ===")
    print(is_df.to_string(index=False))

    best = is_df.iloc[0]
    best_lam = float(best["lam"])
    print(f"\n[lock] best IS λ={best_lam}  IS Calmar={best['calmar']}")

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    print(f"  locked λ={best_lam}")
    build_adjusted_predictions(panel, best_lam, "OOS_locked")
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_locked")
    oos_df = oos_stats.pop("months_df")
    oos_df.to_csv(OUT_OOS_STATS, index=False)
    print(f"\n[saved] {OUT_OOS_STATS}")

    print("\n" + "=" * 70)
    print("=== FINAL v20 industry_60d ===")
    print("=" * 70)
    print(f"locked λ={best_lam}")
    print(f"   Calmar={oos_stats['calmar']}  Sharpe={oos_stats['sharpe']}  "
          f"ann={oos_stats['ann_%']}%  MDD={oos_stats['mdd_%']}%  "
          f"cum={oos_stats['cum_%']}%  win%={oos_stats['win_%']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
