"""Shared helper: 从 baidu_kline (hfq) 计算 IBS 多 horizon 因子.

IBS (Internal Bar Strength) 因子,Phase A 候选.

参考: Connors / Larsson 短期 mean-reversion literature, US ETF/股票市场
ICIR -0.4~-0.8 magnitude (high IBS → 近期反转).

因子定义:
    IBS_day_t = (close_t - low_t) / (high_t - low_t)   # range [0,1]
    IBS_1d         = IBS_day at T-1 (单日)
    IBS_5d_mean    = mean(IBS_day, T-5..T-1)
    IBS_10d_mean   = mean(IBS_day, T-10..T-1)
    IBS_20d_mean   = mean(IBS_day, T-20..T-1)
    IBS_60d_mean   = mean(IBS_day, T-60..T-1)

涨跌停 / 一字板 mask: high == low → IBS 未定义 → NaN.

预期 sign: -1 (反转); high IBS → 下月跑输.

hfq vs qfq: 同一股 ratio (close-low)/(high-low) 不受复权 factor 影响.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KLINE_PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"

IBS_HORIZONS = [1, 5, 10, 20, 60]
IBS_COLS = ["IBS_1d", "IBS_5d_mean", "IBS_10d_mean",
            "IBS_20d_mean", "IBS_60d_mean"]


def compute_ibs_daily(k: pd.DataFrame) -> pd.DataFrame:
    """计算 daily IBS = (close - low) / (high - low).

    Returns columns: code, date, ibs_day.
    high == low 或 数据异常 → NaN.
    """
    k = k.copy()
    span = k["high"] - k["low"]
    valid = (span > 0) & k["close"].notna() & k["low"].notna()
    ibs = (k["close"] - k["low"]) / span
    k["ibs_day"] = np.where(valid, ibs, np.nan)
    return k[["code", "date", "ibs_day"]]


def build_ibs_factors(
    codes: set[str] | None = None,
    min_date: str = "2013-08-01",
    max_date: str | None = None,
) -> pd.DataFrame:
    """计算多 horizon IBS panel.

    Returns columns: code, date, IBS_1d, IBS_5d_mean, IBS_10d_mean,
    IBS_20d_mean, IBS_60d_mean

    backward-only: rolling 在 ibs_day.shift(1) 上做 → T 日因子只看 T-1..T-N.
    """
    print(f"[ibs] loading {KLINE_PARQUET.name}...", flush=True)
    k = pd.read_parquet(
        KLINE_PARQUET,
        columns=["code", "date", "high", "low", "close"],
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k["date"] = pd.to_datetime(k["date"])
    if codes is not None:
        k = k[k["code"].isin(codes)]
    k = k[k["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        k = k[k["date"] <= pd.Timestamp(max_date)]
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"[ibs] rows={len(k):,} codes={k['code'].nunique()} "
          f"date {k['date'].min().date()} ~ {k['date'].max().date()}",
          flush=True)

    ibs = compute_ibs_daily(k)
    n_nan = ibs["ibs_day"].isna().sum()
    print(f"[ibs] daily NaN (high==low / data 异常) = {n_nan:,} "
          f"({n_nan / max(len(ibs), 1) * 100:.2f}%)", flush=True)

    # backward-only: shift(1) 再 rolling → T 上只用 T-N..T-1
    ibs = ibs.sort_values(["code", "date"]).reset_index(drop=True)
    grp = ibs.groupby("code", sort=False)
    ibs["_ibs_lag1"] = grp["ibs_day"].shift(1)
    grp2 = ibs.groupby("code", sort=False)

    print("[ibs] computing rolling 1/5/10/20/60d means...", flush=True)
    ibs["IBS_1d"] = ibs["_ibs_lag1"]
    for N in (5, 10, 20, 60):
        ibs[f"IBS_{N}d_mean"] = grp2["_ibs_lag1"].transform(
            lambda s, n=N: s.rolling(n, min_periods=n).mean()
        )

    out = ibs[["code", "date", *IBS_COLS]].copy()
    for c in IBS_COLS:
        cov = out[c].notna().mean() * 100
        print(f"[ibs] {c:18s} coverage={cov:.1f}%", flush=True)
    return out


if __name__ == "__main__":
    # smoke test
    f = build_ibs_factors(min_date="2014-01-01", max_date="2014-12-31")
    print(f.head(10))
    print(f.tail(5))
    print("IBS_1d quantiles:")
    print(f["IBS_1d"].dropna().quantile([0.0, 0.25, 0.5, 0.75, 1.0]))
