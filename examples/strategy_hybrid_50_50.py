"""Hybrid 50/50 — 25k train24 主动 + 25k CSI300 ETF buy-hold, 60 月回测.

策略:
  本金 50,000 元
    25,000 → train24 主动 (K=8 D=2, 跟 paper_trade_today 同 picks 逻辑)
    25,000 → 沪深300 ETF (用 sh000300 close 做近似), 一次性买入持有

数据源 (纯 cache, 不抓网络):
  - examples/v17_dens_clean_phase2_stats.csv     主动半月度收益 (train24 Phase 2 clean)
  - data_cache/index_kline.parquet (code=sh000300) ETF 月度 proxy

输出:
  - examples/strategy_hybrid_60m_stats.csv  (60 月 × 4 配置明细)
  - stdout: 4 配置对比表 + 跨 regime 分析 + 决策

两种 hybrid:
  - rebalance:   每月把仓位拉回 50/50 (constant mix), 经典学术 baseline
  - drift:       一次性 50/50, 之后让权重随收益漂移 (任务 spec: ETF 持有不动)

Run: python examples/strategy_hybrid_50_50.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN24_STATS = ROOT / "examples" / "v17_dens_clean_phase2_stats.csv"
INDEX_PARQUET = ROOT / "data_cache" / "index_kline.parquet"
INDEX_CODE = "sh000300"
OUT_CSV = ROOT / "examples" / "strategy_hybrid_60m_stats.csv"

WINDOW_START = "2021-05"
WINDOW_END = "2026-04"
N_PERIODS_PER_YEAR = 12
INITIAL_CAPITAL = 50_000.0
ACTIVE_HALF = 25_000.0
PASSIVE_HALF = 25_000.0


def load_train24_monthly() -> pd.Series:
    """读 phase2 clean 60m stats, 返回 monthly abs_ret_% 序列 (index=YYYY-MM)."""
    df = pd.read_csv(TRAIN24_STATS)
    df = df.sort_values("month").reset_index(drop=True)
    df = df[(df["month"] >= WINDOW_START) & (df["month"] <= WINDOW_END)].copy()
    return df.set_index("month")["abs_ret_%"]


def _load_csi300_full() -> pd.Series:
    """读全量 CSI300 月度收益 (用于扩展 regime 检查, 不限 60m 窗口)."""
    idx = pd.read_parquet(INDEX_PARQUET)
    idx = idx[idx["code"] == INDEX_CODE].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["ym"] = idx["date"].dt.to_period("M").astype(str)
    monthly_close = idx.groupby("ym")["close"].last().sort_index()
    rets = monthly_close.pct_change().dropna() * 100
    return rets


def load_csi300_monthly() -> pd.Series:
    """读 index_kline.parquet sh000300, 返回月度 abs_ret_% 序列."""
    idx = pd.read_parquet(INDEX_PARQUET)
    idx = idx[idx["code"] == INDEX_CODE].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["ym"] = idx["date"].dt.to_period("M").astype(str)
    monthly_close = idx.groupby("ym")["close"].last()
    months = pd.period_range(WINDOW_START, WINDOW_END, freq="M").astype(str)
    anchor_month = (pd.Period(WINDOW_START, freq="M") - 1).strftime("%Y-%m")
    if anchor_month not in monthly_close.index:
        raise RuntimeError(f"missing anchor month {anchor_month} in index_kline")
    rets = []
    prev = monthly_close[anchor_month]
    for m in months:
        cur = monthly_close[m]
        rets.append((cur / prev - 1) * 100)
        prev = cur
    return pd.Series(rets, index=months)


def annualize_metrics(returns_pct: pd.Series) -> dict:
    """input: monthly abs_ret in %. Output: cum/ann/sharpe/mdd/calmar/win + n."""
    ret = returns_pct.dropna()
    if len(ret) == 0:
        return dict(cum_pct=0, ann_pct=0, sharpe=0, mdd_pct=0, calmar=0, win_pct=0, n=0)
    cum = (1 + ret / 100).prod() - 1
    n = len(ret)
    years = n / N_PERIODS_PER_YEAR
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else 0
    mean = (ret / 100).mean()
    std = (ret / 100).std()
    sharpe = mean / std * np.sqrt(N_PERIODS_PER_YEAR) if std > 0 else 0
    cum_s = (1 + ret / 100).cumprod()
    peak = cum_s.cummax()
    mdd = ((cum_s - peak) / peak).min()
    calmar = ann / abs(mdd) if mdd < 0 else 0
    win = (ret > 0).sum() / n
    return dict(
        cum_pct=round(cum * 100, 2),
        ann_pct=round(ann * 100, 2),
        sharpe=round(sharpe, 2),
        mdd_pct=round(mdd * 100, 2),
        calmar=round(calmar, 2),
        win_pct=round(win * 100, 1),
        n=n,
    )


def simulate_hybrid_rebalance(active: pd.Series, passive: pd.Series) -> pd.Series:
    """每月 rebalance 回 50/50 (constant mix). 月度 ret = 0.5*r_a + 0.5*r_p."""
    return 0.5 * active + 0.5 * passive


def simulate_hybrid_drift(active: pd.Series, passive: pd.Series) -> pd.DataFrame:
    """No rebalance: active + passive 独立复利, 月末 portfolio_value = a_val + p_val.
    返回 DataFrame with columns [a_val, p_val, total, total_ret_pct, w_active]."""
    rows = []
    a_val = ACTIVE_HALF
    p_val = PASSIVE_HALF
    prev_total = a_val + p_val
    for m in active.index:
        r_a = active[m] / 100
        r_p = passive[m] / 100
        a_val *= (1 + r_a)
        p_val *= (1 + r_p)
        total = a_val + p_val
        ret_pct = (total / prev_total - 1) * 100
        w_a = a_val / total if total > 0 else 0
        rows.append(dict(month=m, a_val=a_val, p_val=p_val, total=total,
                         total_ret_pct=ret_pct, w_active=w_a))
        prev_total = total
    return pd.DataFrame(rows).set_index("month")


def regime_breakdown(monthly_pct: pd.Series, regimes: dict) -> pd.DataFrame:
    """按 regime 区间 (start_ym, end_ym) 切片, 计算每段 cum + n_months + mdd + win + avg."""
    rows = []
    for label, (start, end) in regimes.items():
        seg = monthly_pct[(monthly_pct.index >= start) & (monthly_pct.index <= end)]
        if len(seg) == 0:
            continue
        cum = (1 + seg / 100).prod() - 1
        cum_s = (1 + seg / 100).cumprod()
        peak = cum_s.cummax()
        mdd = ((cum_s - peak) / peak).min()
        win = (seg > 0).sum() / len(seg)
        rows.append(dict(
            regime=label, start=start, end=end,
            n=len(seg),
            cum_pct=round(cum * 100, 2),
            mdd_pct=round(mdd * 100, 2),
            win_pct=round(win * 100, 1),
            avg_pct=round(seg.mean(), 2),
        ))
    return pd.DataFrame(rows)


def main():
    print("=== Hybrid 50/50 Simulator (60 月, 2021-05 → 2026-04) ===\n")
    active = load_train24_monthly()
    passive = load_csi300_monthly()
    print(f"[load] train24 active half: {len(active)} months "
          f"({active.index[0]} → {active.index[-1]})")
    print(f"[load] CSI300 passive half: {len(passive)} months "
          f"({passive.index[0]} → {passive.index[-1]})")
    assert (active.index == passive.index).all(), "month index mismatch"

    # Correlation (sanity)
    corr = active.corr(passive)
    print(f"[corr] train24 vs CSI300 monthly returns: {corr:+.3f}")

    hybrid_rebal = simulate_hybrid_rebalance(active, passive)
    drift_df = simulate_hybrid_drift(active, passive)
    hybrid_drift = drift_df["total_ret_pct"]

    m_train24 = annualize_metrics(active)
    m_csi300 = annualize_metrics(passive)
    m_hyb_reb = annualize_metrics(hybrid_rebal)
    m_hyb_drf = annualize_metrics(hybrid_drift)

    print("\n=== 60 月对比表 ===")
    header = f"{'配置':<34} {'cum%':>8} {'ann%':>7} {'Sharpe':>7} {'MDD%':>8} {'Calmar':>7} {'Win%':>6}"
    print(header)
    print("-" * len(header))
    for name, m in [
        ("train24 100% (Phase 2 clean)", m_train24),
        ("CSI300 100% ETF (buy-hold)", m_csi300),
        ("Hybrid 50/50 (monthly rebal)", m_hyb_reb),
        ("Hybrid 50/50 (drift, no rebal)", m_hyb_drf),
    ]:
        print(f"{name:<34} {m['cum_pct']:>+8.2f} {m['ann_pct']:>+7.2f} "
              f"{m['sharpe']:>7.2f} {m['mdd_pct']:>+8.2f} "
              f"{m['calmar']:>7.2f} {m['win_pct']:>6.1f}")

    out = pd.DataFrame({
        "month": active.index,
        "train24_ret_pct": active.values,
        "csi300_ret_pct": passive.values,
        "hybrid_rebal_ret_pct": hybrid_rebal.values,
        "hybrid_drift_ret_pct": hybrid_drift.values,
        "hybrid_drift_total_val": drift_df["total"].values,
        "hybrid_drift_w_active": drift_df["w_active"].values,
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[saved] {OUT_CSV}")

    # Extended pre-2021 regime check (任务 Step 4 提到 2019 / 2020-Q1)
    print("\n=== 扩展 regime: 2019 + 2020 COVID (60月窗口外) ===")
    full_train24 = pd.read_csv(TRAIN24_STATS).sort_values("month").set_index("month")["abs_ret_%"]
    full_csi = _load_csi300_full()
    common_idx = full_train24.index.intersection(full_csi.index)
    ft = full_train24.loc[common_idx]
    fc = full_csi.loc[common_idx]
    ext_regimes = {
        "2019 全年 (train24 -13%)": ("2019-01", "2019-12"),
        "2020-Q1 COVID":            ("2020-01", "2020-03"),
        "2020-Q2 反弹":              ("2020-04", "2020-06"),
    }
    for label, series in [("train24", ft), ("CSI300", fc),
                          ("Hybrid rebal", 0.5 * ft + 0.5 * fc)]:
        print(f"  --- {label} ---")
        rdf = regime_breakdown(series, ext_regimes)
        for _, r in rdf.iterrows():
            print(f"    {r['regime']:<26} n={r['n']:>2}  cum={r['cum_pct']:>+7.2f}%  "
                  f"mdd={r['mdd_pct']:>+6.2f}%")

    print("\n=== 跨 regime 分析 (60m 窗口内) ===")
    regimes = {
        "2021-H2 抱团崩":   ("2021-05", "2021-12"),
        "2022 全面熊":      ("2022-01", "2022-12"),
        "2023 横盘微跌":    ("2023-01", "2023-12"),
        "2024-H1 小盘踩踏": ("2024-01", "2024-06"),
        "2024-H2 924 行情": ("2024-07", "2024-12"),
        "2025-26 当下":     ("2025-01", "2026-04"),
    }
    for label, series in [
        ("train24",       active),
        ("CSI300",        passive),
        ("Hybrid rebal",  hybrid_rebal),
        ("Hybrid drift",  hybrid_drift),
    ]:
        print(f"\n  --- {label} ---")
        rdf = regime_breakdown(series, regimes)
        for _, r in rdf.iterrows():
            print(f"    {r['regime']:<18} n={r['n']:>2}  cum={r['cum_pct']:>+7.2f}%  "
                  f"mdd={r['mdd_pct']:>+6.2f}%  win={r['win_pct']:>4.1f}%  "
                  f"avg={r['avg_pct']:>+5.2f}%/mo")

    print("\n=== 决策依据 ===")
    print(f"  train24 单用:           Sharpe={m_train24['sharpe']:.2f}  Calmar={m_train24['calmar']:.2f}  ann={m_train24['ann_pct']:.2f}%")
    print(f"  Hybrid (monthly rebal): Sharpe={m_hyb_reb['sharpe']:.2f}  Calmar={m_hyb_reb['calmar']:.2f}  ann={m_hyb_reb['ann_pct']:.2f}%")
    print(f"  Hybrid (drift):         Sharpe={m_hyb_drf['sharpe']:.2f}  Calmar={m_hyb_drf['calmar']:.2f}  ann={m_hyb_drf['ann_pct']:.2f}%")

    best_hyb = m_hyb_reb if m_hyb_reb["sharpe"] >= m_hyb_drf["sharpe"] else m_hyb_drf
    best_name = "rebal" if m_hyb_reb["sharpe"] >= m_hyb_drf["sharpe"] else "drift"

    if best_hyb["sharpe"] > m_train24["sharpe"]:
        print(f"\n  → Hybrid ({best_name}) Sharpe {best_hyb['sharpe']:.2f} > "
              f"train24 {m_train24['sharpe']:.2f}: **推荐 Hybrid** (降风险)")
    else:
        print(f"\n  → Hybrid ({best_name}) Sharpe {best_hyb['sharpe']:.2f} <= "
              f"train24 {m_train24['sharpe']:.2f}: **不推荐 Hybrid** (ETF 拖累 alpha)")

    if best_hyb["calmar"] > m_train24["calmar"]:
        print(f"     Calmar {best_hyb['calmar']:.2f} > {m_train24['calmar']:.2f}: 风险调整后回报更好")
    else:
        print(f"     Calmar {best_hyb['calmar']:.2f} <= {m_train24['calmar']:.2f}: 风险调整后回报未改善")


if __name__ == "__main__":
    main()
