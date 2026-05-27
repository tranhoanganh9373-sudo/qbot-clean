"""v17 业绩归因 (pyfolio-reloaded + empyrical-reloaded).

输入:
  data_cache/v17_daily_returns.csv      # (date, daily_ret, n_holdings, month)
  data_cache/index_kline.parquet        # sh000300 close 作 benchmark

输出:
  examples/v17_perf_tearsheet.md
  examples/v17_perf_metrics.json

run:
  python examples/perf_attribution_v17.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data_cache" / "v17_daily_returns.csv"
INDEX_PATH = ROOT / "data_cache" / "index_kline.parquet"
INDEX_CODE = "sh000300"
OUT_DIR = Path(__file__).resolve().parent
OUT_MD = OUT_DIR / "v17_perf_tearsheet.md"
OUT_JSON = OUT_DIR / "v17_perf_metrics.json"


def load_strategy_returns() -> pd.Series:
    df = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    s = df.set_index("date")["daily_ret"].astype(float)
    s.index = s.index.tz_localize("UTC")
    s.name = "v17_strategy"
    return s


def load_benchmark_returns(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    idx = pd.read_parquet(INDEX_PATH)
    idx = idx[idx["code"] == INDEX_CODE].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").set_index("date")["close"]
    idx = idx.loc[start - pd.Timedelta(days=5): end + pd.Timedelta(days=5)]
    bench = idx.pct_change().dropna()
    bench.index = bench.index.tz_localize("UTC")
    bench.name = "csi300"
    return bench


def main() -> None:
    if not DAILY_PATH.exists():
        raise SystemExit(f"missing artifact: {DAILY_PATH} — run strategy_v17 first")
    import empyrical as ep

    strat = load_strategy_returns()
    print(f"[1/3] 策略 daily returns: {len(strat)} 天  "
          f"{strat.index.min().date()} → {strat.index.max().date()}")

    bench = load_benchmark_returns(strat.index.min().tz_localize(None),
                                    strat.index.max().tz_localize(None))
    aligned = pd.concat([strat, bench], axis=1, join="inner").dropna()
    strat_a = aligned.iloc[:, 0]
    bench_a = aligned.iloc[:, 1]
    print(f"  benchmark sh000300 aligned: {len(bench_a)} 天")

    print("[2/3] empyrical metrics ...")
    metrics = {
        "n_days": int(len(strat_a)),
        "start": str(strat_a.index.min().date()),
        "end": str(strat_a.index.max().date()),
        "cum_return": float(ep.cum_returns_final(strat_a)),
        "annual_return": float(ep.annual_return(strat_a)),
        "annual_volatility": float(ep.annual_volatility(strat_a)),
        "sharpe_ratio": float(ep.sharpe_ratio(strat_a, risk_free=0)),
        "sortino_ratio": float(ep.sortino_ratio(strat_a)),
        "calmar_ratio": float(ep.calmar_ratio(strat_a)),
        "omega_ratio": float(ep.omega_ratio(strat_a)),
        "max_drawdown": float(ep.max_drawdown(strat_a)),
        "tail_ratio": float(ep.tail_ratio(strat_a)),
        "stability": float(ep.stability_of_timeseries(strat_a)),
        "value_at_risk_95": float(ep.value_at_risk(strat_a, cutoff=0.05)),
        "cvar_95": float(ep.conditional_value_at_risk(strat_a, cutoff=0.05)),
        "alpha_vs_csi300": float(ep.alpha(strat_a, bench_a, risk_free=0)),
        "beta_vs_csi300": float(ep.beta(strat_a, bench_a, risk_free=0)),
        "downside_risk": float(ep.downside_risk(strat_a)),
        "excess_sharpe": float(ep.excess_sharpe(strat_a, bench_a)),
        "csi300_cum_return": float(ep.cum_returns_final(bench_a)),
        "csi300_annual_return": float(ep.annual_return(bench_a)),
        "csi300_sharpe": float(ep.sharpe_ratio(bench_a)),
        "csi300_max_drawdown": float(ep.max_drawdown(bench_a)),
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  metrics json → {OUT_JSON.name}")

    print("[3/3] markdown tearsheet ...")

    def fmt(x: float, pct: bool = False) -> str:
        return f"{x * 100:+.2f}%" if pct else f"{x:.3f}"

    md = [
        "# v17 业绩归因 (empyrical)",
        "",
        f"**期间**: {metrics['start']} → {metrics['end']} ({metrics['n_days']} 交易日)",
        "",
        "## 策略 vs CSI300 benchmark",
        "",
        "| 指标 | v17 | CSI300 |",
        "|------|---:|---:|",
        f"| 累计收益 | {fmt(metrics['cum_return'], True)} "
        f"| {fmt(metrics['csi300_cum_return'], True)} |",
        f"| 年化收益 | {fmt(metrics['annual_return'], True)} "
        f"| {fmt(metrics['csi300_annual_return'], True)} |",
        f"| Sharpe | {fmt(metrics['sharpe_ratio'])} "
        f"| {fmt(metrics['csi300_sharpe'])} |",
        f"| Max Drawdown | {fmt(metrics['max_drawdown'], True)} "
        f"| {fmt(metrics['csi300_max_drawdown'], True)} |",
        "",
        "## 风险/收益指标",
        "",
        "| 指标 | 值 | 含义 |",
        "|---|---:|---|",
        f"| Annual Vol | {fmt(metrics['annual_volatility'], True)} | 年化波动率 |",
        f"| Sortino | {fmt(metrics['sortino_ratio'])} | 只算下行 σ 的 Sharpe |",
        f"| Calmar | {fmt(metrics['calmar_ratio'])} | 年化 / |MDD| |",
        f"| Omega | {fmt(metrics['omega_ratio'])} | gain/loss probability ratio |",
        f"| Tail Ratio | {fmt(metrics['tail_ratio'])} | 右尾/左尾, >1 好 |",
        f"| Stability (R²) | {fmt(metrics['stability'])} | 累计曲线对时间的 R² |",
        f"| VaR 95% | {fmt(metrics['value_at_risk_95'], True)} | 5% 概率单日最大亏损 |",
        f"| CVaR 95% | {fmt(metrics['cvar_95'], True)} | VaR 之外的平均亏损 |",
        f"| Downside Risk | {fmt(metrics['downside_risk'], True)} | 仅下行 σ (年化) |",
        "",
        "## 相对 CSI300",
        "",
        "| 指标 | 值 | 含义 |",
        "|---|---:|---|",
        f"| α (年化) | {fmt(metrics['alpha_vs_csi300'], True)} | 扣 β·benchmark 后的超额 |",
        f"| β | {fmt(metrics['beta_vs_csi300'])} | 跟 CSI300 协动性 |",
        f"| Excess Sharpe | {fmt(metrics['excess_sharpe'])} | (R_s − R_b) / σ(R_s − R_b) |",
        "",
        "## 读法",
        "",
        "- **Sharpe > 1.0** 好, **> 2.0** 很好",
        "- **Sortino > Sharpe** 表示下行风险小于整体波动",
        "- **Calmar > 1.0** 表示年化收益 > MDD, 资金曲线舒服",
        "- **α > 0** 且 **β < 1** 最理想 (跟大盘弱相关但跑赢)",
        "- **Stability < 0.5** 说明锯齿严重, 业绩不可持续",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"  tearsheet md → {OUT_MD.name}")
    print("\n--- 关键指标 ---")
    for k in ("cum_return", "annual_return", "sharpe_ratio", "max_drawdown",
              "alpha_vs_csi300", "beta_vs_csi300", "calmar_ratio"):
        print(f"  {k:<22} {metrics[k]:+.4f}")


if __name__ == "__main__":
    main()
