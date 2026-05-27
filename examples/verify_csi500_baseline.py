"""Step 8: verify CSI500 v3 clean baseline OOS stats and compare with CSI300 v3.

读 examples/v17_dens_clean_phase2_v3_csi500_stats.csv (Step 7 输出 112 月),
计算 OOS (2021-05~2026-04 = 60 月) baseline 统计, 跟 CSI300 v3 baseline 对比.

不动 production. 不写 production cache. (只打印, 不写文件.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSI500_STATS = ROOT / "examples" / "v17_dens_clean_phase2_v3_csi500_stats.csv"
CSI500_PRED = ROOT / "data_cache" / "v17_dens_csi500_train24_predictions.parquet"

# CSI300 v3 Phase 2 baseline (from memory): OOS Calmar 0.42; brief 提到 0.77 (可能是不同 train window)
CSI300_V3_OOS_CALMAR_MEASURED = 0.42  # Phase 2 clean retrain measured (memory)
CSI300_V3_OOS_CALMAR_BRIEF = 0.77     # brief 引用值


def _annualize(rets_pct: pd.Series) -> dict:
    """rets_pct = monthly abs_ret in % (e.g. 8.5 means +8.5%)."""
    r = rets_pct / 100.0
    n_months = len(r)
    if n_months == 0:
        return {"ann_ret": 0, "ann_vol": 0, "sharpe": 0, "mdd": 0, "calmar": 0,
                "cum_ret_pct": 0, "n_months": 0}
    cum = (1 + r).cumprod()
    if cum.iloc[-1] <= 0:
        return {"ann_ret": -1.0, "ann_vol": float(r.std() * np.sqrt(12)),
                "sharpe": 0, "mdd": -1.0, "calmar": 0,
                "cum_ret_pct": float((cum.iloc[-1] - 1) * 100),
                "n_months": n_months}
    ann_ret = cum.iloc[-1] ** (12.0 / n_months) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else 0
    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "calmar": float(calmar),
        "cum_ret_pct": float((cum.iloc[-1] - 1) * 100),
        "n_months": n_months,
    }


def main() -> int:
    if not CSI500_STATS.exists():
        print(f"FATAL: {CSI500_STATS} 不存在 — Step 7 未完成", file=sys.stderr)
        return 1

    df = pd.read_csv(CSI500_STATS)
    df = df.dropna(subset=["month"])
    df["month_dt"] = pd.to_datetime(df["month"], format="%Y-%m")
    df = df.sort_values("month_dt").reset_index(drop=True)
    print(f"[stats] {len(df)} 月 ({df['month'].iloc[0]} ~ {df['month'].iloc[-1]})",
          flush=True)

    full = _annualize(df["abs_ret_%"])
    print(f"\n=== CSI500 v3 baseline (full {full['n_months']} 月 IS+OOS) ===")
    for k, v in full.items():
        if isinstance(v, float):
            print(f"  {k:14s} = {v:.4f}")
        else:
            print(f"  {k:14s} = {v}")

    oos_start = pd.Timestamp("2021-05-01")
    oos_end = pd.Timestamp("2026-04-30")
    oos = df[(df["month_dt"] >= oos_start) & (df["month_dt"] <= oos_end)]
    oos_stats = _annualize(oos["abs_ret_%"])
    print(f"\n=== CSI500 v3 baseline (OOS {oos_stats['n_months']} 月 "
          f"2021-05~2026-04) ===")
    for k, v in oos_stats.items():
        if isinstance(v, float):
            print(f"  {k:14s} = {v:.4f}")
        else:
            print(f"  {k:14s} = {v}")

    is_end = pd.Timestamp("2020-12-31")
    is_period = df[df["month_dt"] <= is_end]
    is_stats = _annualize(is_period["abs_ret_%"])
    print(f"\n=== CSI500 v3 baseline (IS 部分 ≤2020-12 = "
          f"{is_stats['n_months']} 月) ===")
    for k, v in is_stats.items():
        if isinstance(v, float):
            print(f"  {k:14s} = {v:.4f}")
        else:
            print(f"  {k:14s} = {v}")

    print(f"\n=== 跟 CSI300 v3 baseline 对比 ===")
    print(f"  CSI300 v3 OOS Calmar (memory measured): {CSI300_V3_OOS_CALMAR_MEASURED:.2f}")
    print(f"  CSI300 v3 OOS Calmar (brief reference): {CSI300_V3_OOS_CALMAR_BRIEF:.2f}")
    print(f"  CSI500 v3 OOS Calmar (this run)       : {oos_stats['calmar']:.4f}")
    delta_measured = (oos_stats['calmar'] / CSI300_V3_OOS_CALMAR_MEASURED - 1) * 100 \
        if CSI300_V3_OOS_CALMAR_MEASURED > 0 else 0
    delta_brief = (oos_stats['calmar'] / CSI300_V3_OOS_CALMAR_BRIEF - 1) * 100 \
        if CSI300_V3_OOS_CALMAR_BRIEF > 0 else 0
    print(f"  Δ vs measured: {delta_measured:+.1f}%")
    print(f"  Δ vs brief   : {delta_brief:+.1f}%")

    if CSI500_PRED.exists():
        pred = pd.read_parquet(CSI500_PRED)
        pred["datetime"] = pd.to_datetime(pred["datetime"])
        pred["month"] = pred["datetime"].dt.to_period("M")
        last_month = pred["month"].max()
        last_cover = pred[pred["month"] == last_month]["instrument"].nunique()
        avg_cover = int(pred.groupby("month")["instrument"].nunique().mean())
        print(f"\n=== predictions cache verify ===")
        print(f"  file: {CSI500_PRED.name}")
        print(f"  size: {CSI500_PRED.stat().st_size/1e6:.2f} MB")
        print(f"  rows: {len(pred):,}")
        print(f"  months: {pred['month'].nunique()}")
        print(f"  date range: {pred['datetime'].min().date()} ~ "
              f"{pred['datetime'].max().date()}")
        print(f"  avg instruments/month: {avg_cover}")
        print(f"  last month ({last_month}) instrument cover: "
              f"{last_cover}/494")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
