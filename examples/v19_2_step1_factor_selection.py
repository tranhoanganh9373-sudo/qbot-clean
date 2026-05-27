"""v19.2 Step 1 — 严格 OOS 因子组合选择 (训练期 2014-01 ~ 2020-12).

只用 in-sample data (2014-2020), 不碰 OOS (2021-05 ~ 2026-04).

输入:
  data_cache/csi300_margin_14yr.parquet  (Step 0 拉取)
  data_cache/baidu_kline.parquet         (kline)
  data_cache/csi300_constituents.csv     (300 codes, 简化 universe)

候选因子组合 (5 个, 反向 z-score, "融资余额变化越大 → 后续收益越差" 假设):
  A: reverse_m5_only        单 margin_5d_chg 反向 z
  B: reverse_m5 + reverse_m20 等权
  C: reverse_m5 + reverse_rzmre_chg 等权     (rzmre_5d_chg = rzmre 滚动 5d 增长率)
  D: reverse_m5 + 偿还比 (rzche/rzmre) 等权
  E: 4 因子全 z 等权: reverse m5/m20/rzmre_chg + 偿还比

λ sweep: {1.0, 1.5, 2.0, 2.5, 3.0}.

评分逻辑:
  每月初: 用横截面 z(combined_signal) 排序
    long  = top 30% 等权
    short = bottom 30% 等权
  long-short 月度收益 = mean(long fwd_5d) - mean(short fwd_5d)
  注: 月度采样, 不每日; fwd_5d 用 close[t+5]/close[t+1] - 1.
  Sharpe = monthly_mean / monthly_std * sqrt(12)
  Calmar = ann_ret / |MDD|

输出:
  examples/v19_2_step1_grid.csv (5 × 5 = 25 行)
  STDOUT: 选定 (best_combo, best_lambda) by Calmar

run:
  python examples/v19_2_step1_factor_selection.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MARGIN_PARQ = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
KLINE_PARQ = ROOT / "data_cache" / "baidu_kline.parquet"
CSI300_CSV = ROOT / "data_cache" / "csi300_constituents.csv"
OUT_CSV = ROOT / "examples" / "v19_2_step1_grid.csv"

# 严格训练期 (永远不碰 OOS).
# 注: 任务原 spec 是 2014-01-01 起, 但 baidu_kline.parquet 实际只从 2018-02
# 才覆盖 CSI300 大部分股 (2014-2017: <1% 股有数据, 2018-01: 47/300 股).
# 截取 2018-02 ~ 2020-12 (35 月) 作为有效 IS — 仍在 OOS (2021-05+) 之前, 无 leak.
IS_START = "2018-02-01"
IS_END = "2020-12-31"
LONG_Q = 0.30
SHORT_Q = 0.30
LAMBDAS = [1.0, 1.5, 2.0, 2.5, 3.0]
COMBOS = ["A", "B", "C", "D", "E"]


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
    """Returns z-scored combined signal (正向: 越大 = 越看多)."""
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


def _monthly_long_short(panel: pd.DataFrame, fwd: pd.DataFrame,
                         combo: str, lam: float) -> pd.Series:
    panel["ym"] = panel["date"].dt.to_period("M")
    fwd["ym"] = fwd["date"].dt.to_period("M")

    monthly_returns = []
    months = sorted(panel["ym"].unique())
    for ym in months:
        candidate_dates = sorted(set(panel[panel["ym"] == ym]["date"]) &
                                  set(fwd[fwd["ym"] == ym]["date"]))
        if not candidate_dates:
            continue
        anchor = candidate_dates[0]
        p_day = panel[panel["date"] == anchor].set_index("code")
        f_day = fwd[fwd["date"] == anchor].set_index("code")
        common = p_day.index.intersection(f_day.index)
        if len(common) < 20:
            continue
        sig = _combine_signal(p_day.loc[common], combo)
        final = lam * sig  # λ 对 quantile 排序无影响, 留接口
        fwd_r = f_day.loc[common, "fwd_ret"]
        valid = (~final.isna()) & (~fwd_r.isna())
        if valid.sum() < 20:
            continue
        final = final[valid]
        fwd_r = fwd_r[valid]
        n_long = max(int(len(final) * LONG_Q), 1)
        n_short = max(int(len(final) * SHORT_Q), 1)
        order = final.sort_values(ascending=False)
        long_idx = order.head(n_long).index
        short_idx = order.tail(n_short).index
        ret = fwd_r.loc[long_idx].mean() - fwd_r.loc[short_idx].mean()
        monthly_returns.append((ym, ret))
    if not monthly_returns:
        return pd.Series(dtype=float)
    s = pd.Series(
        [r for _, r in monthly_returns],
        index=[pd.Period(ym, freq="M") for ym, _ in monthly_returns],
    )
    return s.sort_index()


def _metrics(returns: pd.Series) -> dict:
    n = len(returns)
    if n < 6:
        return {"n": n, "sharpe": 0.0, "calmar": 0.0, "ann": 0.0,
                "mdd": 0.0, "cum": 0.0}
    cum = (1 + returns).prod() - 1
    years = n / 12
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else 0
    mu = returns.mean()
    sd = returns.std()
    sharpe = mu / sd * np.sqrt(12) if sd > 0 else 0
    cs = (1 + returns).cumprod()
    peak = cs.cummax()
    mdd = ((cs - peak) / peak).min()
    calmar = ann / abs(mdd) if mdd < 0 else 0
    return {
        "n": n, "sharpe": round(float(sharpe), 3),
        "calmar": round(float(calmar), 3),
        "ann": round(float(ann) * 100, 2),
        "mdd": round(float(mdd) * 100, 2),
        "cum": round(float(cum) * 100, 2),
    }


def main() -> int:
    if not MARGIN_PARQ.exists():
        print(f"FATAL: {MARGIN_PARQ} 缺失 (跑 fetch_csi300_margin.py 先)",
              file=sys.stderr)
        return 2

    print(f"[load] margin parquet", flush=True)
    margin = pd.read_parquet(MARGIN_PARQ)
    margin["code"] = margin["code"].astype(str).str.zfill(6)
    margin["date"] = pd.to_datetime(margin["date"])
    print(f"  margin: {len(margin):,} rows, {margin['code'].nunique()} stocks, "
          f"{margin['date'].min().date()} ~ {margin['date'].max().date()}",
          flush=True)

    print(f"[load] kline parquet", flush=True)
    kl = pd.read_parquet(KLINE_PARQ, columns=["code", "date", "close"])
    kl["code"] = kl["code"].astype(str).str.zfill(6)
    kl["date"] = pd.to_datetime(kl["date"])

    csi = pd.read_csv(CSI300_CSV, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    universe = set(csi["code"])
    margin = margin[margin["code"].isin(universe)]
    kl = kl[kl["code"].isin(universe)]
    print(f"  kline filtered to CSI300: {len(kl):,} rows", flush=True)

    is_end_ts = pd.to_datetime(IS_END)
    is_start_ts = pd.to_datetime(IS_START)
    kl_is = kl[(kl["date"] >= is_start_ts - pd.Timedelta(days=10)) &
                (kl["date"] <= is_end_ts + pd.Timedelta(days=14))].copy()
    kl_is = kl_is.sort_values(["code", "date"]).reset_index(drop=True)
    kl_is["close_t1"] = kl_is.groupby("code")["close"].shift(-1)
    kl_is["close_t5"] = kl_is.groupby("code")["close"].shift(-5)
    # 防御: close 可能 0 / NaN, fwd_ret 产生 inf
    kl_is.loc[kl_is["close_t1"] <= 0, "close_t1"] = np.nan
    kl_is["fwd_ret"] = kl_is["close_t5"] / kl_is["close_t1"] - 1
    kl_is["fwd_ret"] = kl_is["fwd_ret"].replace([np.inf, -np.inf], np.nan)
    # 极端 outlier 截 (split / 数据错): ±50% 5d 收益
    # (真实 5d 收益 > 50% 通常是 split 未复权或脏数据)
    kl_is.loc[kl_is["fwd_ret"].abs() > 0.5, "fwd_ret"] = np.nan
    fwd = kl_is.dropna(subset=["fwd_ret"])[["code", "date", "fwd_ret"]].copy()
    fwd = fwd[(fwd["date"] >= is_start_ts) & (fwd["date"] <= is_end_ts)]

    margin_is = margin[(margin["date"] >= is_start_ts) &
                        (margin["date"] <= is_end_ts)].copy()
    margin_is = _build_factor_panel(margin_is)
    print(f"  margin IS: {len(margin_is):,} rows, fwd IS: {len(fwd):,} rows",
          flush=True)

    rows = []
    for combo in COMBOS:
        for lam in LAMBDAS:
            mr = _monthly_long_short(margin_is.copy(), fwd.copy(), combo, lam)
            m = _metrics(mr)
            row = {"combo": combo, "lambda": lam, **m}
            rows.append(row)
            print(f"  {combo} λ={lam}: n={m['n']} sharpe={m['sharpe']} "
                  f"calmar={m['calmar']} ann={m['ann']}% mdd={m['mdd']}% "
                  f"cum={m['cum']}%", flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_CSV, index=False)
    print(f"\n[saved] {OUT_CSV}", flush=True)

    best = grid.sort_values("calmar", ascending=False).iloc[0]
    print(f"\n=== best (in-sample, IS 2014-01 ~ 2020-12) ===", flush=True)
    print(f"  combo={best['combo']} lambda={best['lambda']}", flush=True)
    print(f"  sharpe={best['sharpe']} calmar={best['calmar']} "
          f"ann={best['ann']}% mdd={best['mdd']}%", flush=True)

    if grid["calmar"].max() < 0.5:
        print(f"\nFATAL: all 25 combos Calmar < 0.5 "
              f"(training-set max = {grid['calmar'].max():.3f}). "
              f"Margin overlay 全废, ABORT OOS.", file=sys.stderr, flush=True)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
