"""v19.2 Step 2 — 严格 OOS 应用 (2021-05 ~ 2026-04, 60 个月).

读 Step 1 选定的 (best_combo, best_lambda), build adjusted pred parquet:
  final_score = z_pred + λ * z_margin_combined_signal  (cross-section by datetime)

然后复用 strategy_v17_dens_grid.realistic_window() 完整 backtest engine 跑 OOS,
output 与 baseline (v17_dens_train24_60m_stats.csv) 直接可比.

输入:
  data_cache/v17_dens_train24_predictions.parquet  (cached DEns OOS preds)
  data_cache/csi300_margin_14yr.parquet            (Step 0)
  data_cache/csi300_constituents.csv

CLI:
  python examples/strategy_v19_2_strict_oos.py --combo A --lambda 1.5

输出:
  data_cache/v19_2_predictions.parquet   (adjusted preds; v17 grid 兼容 schema)
  examples/v19_2_strict_oos_stats.csv    (60 月 monthly returns)
  STDOUT: 与 baseline 1.96 的对比表
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
MARGIN_PARQ = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
CSI300_CSV = ROOT / "data_cache" / "csi300_constituents.csv"
ADJ_PRED = ROOT / "data_cache" / "v19_2_predictions.parquet"
OUT_STATS = ROOT / "examples" / "v19_2_strict_oos_stats.csv"

OOS_START = "2021-05-01"
OOS_END = "2026-04-30"
FIRST_TEST = "2021-05"
LAST_TEST = "2026-04"


def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std()
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _build_factor_panel(margin: pd.DataFrame) -> pd.DataFrame:
    df = margin.sort_values(["code", "date"]).copy()
    df["rzmre_5d_chg"] = df.groupby("code")["rzmre"].pct_change(5)
    df["repay_ratio"] = df["rzche"] / df["rzmre"].replace(0, np.nan)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _combine_signal(panel_day: pd.DataFrame, combo: str) -> pd.Series:
    neg_m5 = -panel_day["margin_5d_chg"]
    neg_m20 = -panel_day["margin_20d_chg"]
    neg_rzmre = -panel_day["rzmre_5d_chg"]
    neg_repay = -panel_day["repay_ratio"]

    if combo == "A":
        sig = _zscore(neg_m5)
    elif combo == "B":
        sig = (_zscore(neg_m5) + _zscore(neg_m20)) / 2
    elif combo == "C":
        sig = (_zscore(neg_m5) + _zscore(neg_rzmre)) / 2
    elif combo == "D":
        sig = (_zscore(neg_m5) + _zscore(neg_repay)) / 2
    elif combo == "E":
        sig = (_zscore(neg_m5) + _zscore(neg_m20)
               + _zscore(neg_rzmre) + _zscore(neg_repay)) / 4
    else:
        raise ValueError(combo)
    return sig


def _code6_to_instrument(c: str) -> str:
    c = str(c).zfill(6)
    if c.startswith("6"):
        return f"SH{c}"
    return f"SZ{c}"


def build_adjusted_predictions(combo: str, lam: float) -> int:
    print(f"[build] combo={combo} λ={lam}", flush=True)

    pred = pd.read_parquet(ORIG_PRED)
    oos_start_ts = pd.to_datetime(OOS_START)
    oos_end_ts = pd.to_datetime(OOS_END)
    pred = pred[(pred["datetime"] >= oos_start_ts) &
                 (pred["datetime"] <= oos_end_ts)].copy()
    print(f"  pred OOS rows: {len(pred):,}", flush=True)

    margin = pd.read_parquet(MARGIN_PARQ)
    margin["code"] = margin["code"].astype(str).str.zfill(6)
    margin["date"] = pd.to_datetime(margin["date"])
    margin = margin[(margin["date"] >= oos_start_ts - pd.Timedelta(days=40)) &
                     (margin["date"] <= oos_end_ts)].copy()
    margin = _build_factor_panel(margin)
    print(f"  margin OOS rows: {len(margin):,}", flush=True)

    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore)

    margin_signals = []
    for d, sub in margin.groupby("date"):
        sub_idx = sub.set_index("code")
        sig = _combine_signal(sub_idx, combo)
        for code, val in sig.items():
            margin_signals.append({"datetime": d,
                                    "instrument": _code6_to_instrument(code),
                                    "raw_margin_sig": val})
    margin_sig_df = pd.DataFrame(margin_signals)
    if margin_sig_df.empty:
        print("FATAL: empty margin signals", file=sys.stderr)
        return 2
    margin_sig_df["z_margin"] = margin_sig_df.groupby("datetime")[
        "raw_margin_sig"].transform(_zscore)

    merged = pred.merge(margin_sig_df[["datetime", "instrument", "z_margin"]],
                         on=["datetime", "instrument"], how="left")
    n_missing = merged["z_margin"].isna().sum()
    merged["z_margin"] = merged["z_margin"].fillna(0.0)
    merged["final_score"] = merged["z_pred"] + lam * merged["z_margin"]
    print(f"  merged rows: {len(merged):,}; missing margin: {n_missing:,} "
          f"({n_missing/len(merged)*100:.1f}%)", flush=True)

    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
    print(f"[saved] {ADJ_PRED}", flush=True)
    return 0


def run_backtest():
    import qlib
    from qlib.constant import REG_CN

    import strategy_v17_dens_grid as v17

    QLIB_DIR = str(ROOT / "data_cache" / "qlib_baidu")
    qlib.init(provider_uri=QLIB_DIR, region=REG_CN)

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

    print(f"[init] PRED_CACHE override → {v17.PRED_CACHE.name}", flush=True)
    print(f"[init] K={v17.K_NORMAL} D={v17.DROP_NORMAL} "
          f"train_months={v17.TRAIN_MONTHS}", flush=True)

    proxy = v17.build_market_proxy()

    first = datetime.strptime(FIRST_TEST + "-01", "%Y-%m-%d")
    last = datetime.strptime(LAST_TEST + "-01", "%Y-%m-%d")
    months = []
    cur = first
    while cur <= last:
        months.append(cur)
        cur += relativedelta(months=1)
    print(f"[run] {len(months)} 月 walk-forward (v19.2 strict OOS)", flush=True)

    rows = []
    for i, m in enumerate(months, 1):
        try:
            res = v17.realistic_window(m, proxy, with_regime=False)
            res["month"] = m.strftime("%Y-%m")
            res["config"] = "v19_2 K=8 D=2"
            rows.append(res)
            print(f"  {i:2d}/{len(months)} {res['month']}: "
                  f"abs_ret={res['abs_ret_%']:+6.2f}%  picks={res['avg_picks']:.1f}",
                  flush=True)
        except Exception as e:
            print(f"  {i:2d}/{len(months)} {m.strftime('%Y-%m')} FAIL: "
                  f"{str(e)[:120]}", flush=True)
            rows.append({"month": m.strftime("%Y-%m"),
                          "config": "v19_2 K=8 D=2",
                          "abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                          "regime_days": "", "n_skipped_limit": 0,
                          "n_stop_loss": 0})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_STATS, index=False)
    print(f"[saved] {OUT_STATS}", flush=True)

    mm = v17.annualize_metrics(df["abs_ret_%"])
    ann_pct = mm["ann_%"] / 100
    mdd_pct = mm["mdd_%"] / 100
    calmar = ann_pct / abs(mdd_pct) if mdd_pct < 0 else 0
    mm["calmar"] = round(calmar, 2)
    mm["avg_picks"] = round(df["avg_picks"].mean(), 1)
    print(f"\n=== v19.2 SUMMARY (strict OOS 60 月) ===", flush=True)
    print(pd.Series(mm).to_string(), flush=True)
    return mm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", required=True,
                         choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    args = parser.parse_args()

    rc = build_adjusted_predictions(args.combo, args.lam)
    if rc != 0:
        return rc

    metrics = run_backtest()

    print(f"\n=== 对比 baseline (v19.1 train24 no overlay) ===", flush=True)
    print(f"  baseline: Calmar=1.96 Sharpe=1.09 ann=51.4% MDD=-26.3% "
          f"cum=694.9% win=61.7%", flush=True)
    print(f"  v19.2   : Calmar={metrics['calmar']} Sharpe={metrics['sharpe']} "
          f"ann={metrics['ann_%']}% MDD={metrics['mdd_%']}% "
          f"cum={metrics['cum_%']}% win={metrics['win_%']}%", flush=True)
    if metrics["calmar"] > 1.96:
        print(f"\n[verdict] v19.2 Calmar > 1.96 — 推荐升级 production",
              flush=True)
    else:
        print(f"\n[verdict] v19.2 Calmar <= 1.96 — margin overlay 在 OOS 上无效, 放弃",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
