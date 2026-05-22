"""Overnight momentum strategy backtest on data/deepseek_trading.csv.

Strategy (一晚持有 T -> T+1 隔夜):
- T 日收盘 (14:30 代理) 对每只 A 股打分
- 取 Top N 等权买入收盘价
- T+1 开盘全平
- 重复每个交易日

Signal 评分（六项滤网都过才入候选，最后按分数 Top N）:
  1. 红盘:        close > open
  2. 未涨停:      0 < change_pct < 9.0%
  3. 量能放大:    today_vol > 1.3 x avg_vol(5)
  4. 趋势向上:    close > SMA20
  5. 收盘强:      (close - low) / (high - low)  in [0.5, 0.95]
  6. 流动性:      turnover > 0.5%
Score = (vol / avg_vol) * change_pct * position_in_range

Cost: 0.25% 单次往返 (印花 0.05 + 佣金 0.05 + 滑点 0.15)

Caveats:
- 14:30 用 close[T] 代理 = ~30min look-ahead bias (尾盘集合竞价信息泄露)
- T+1 开盘成交价直接用 open[T+1] = 忽略开盘集合竞价滑点
- 30 日 warmup 排除新股；涨停板过滤防追高
- 无幸存者偏差（CSV 含全市场）

Run:  python examples/overnight_momentum_backtest.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "deepseek_trading.csv"
OUT_DIR = Path(__file__).resolve().parent

PICKS_PER_DAY = 20
WARMUP_BARS = 30
COST_RATE = 0.0025                  # 0.25% round-trip
MAX_CHG_PCT = 9.0
MIN_TURNOVER = 0.005                # 0.5% 换手率下限
SCORE_RANGE_LOW = 0.50
SCORE_RANGE_HIGH = 0.95
VOL_SURGE_MIN = 1.3
TRADING_DAYS_PER_YEAR = 252


def load_market(path: Path) -> dict[str, pd.DataFrame]:
    """Load CSV preserving turnover for the signal filter."""
    print(f"[1/4] loading {path.name}...")
    t = time.time()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    market: dict[str, pd.DataFrame] = {}
    for code, g in df.groupby("code", sort=False):
        if len(g) < WARMUP_BARS + 5:
            continue
        g = g.set_index("date")[["open", "high", "low", "close", "volume", "turnover"]].copy()
        g["volume"] = g["volume"].astype("int64")
        market[code] = g
    print(f"  ok ({time.time() - t:.1f}s) — {len(market)} stocks with >= {WARMUP_BARS + 5} bars")
    return market


def score_series(df: pd.DataFrame) -> pd.Series:
    """Per-bar score; NaN where any filter fails."""
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"].astype(float)
    turn = df["turnover"]

    chg = close.pct_change() * 100
    avg_vol = vol.shift(1).rolling(5).mean()
    sma20 = close.shift(1).rolling(20).mean()
    pos = (close - low) / (high - low).replace(0, np.nan)
    vol_ratio = vol / avg_vol

    valid = (
        (close > open_)
        & chg.between(0.01, MAX_CHG_PCT)
        & vol_ratio.gt(VOL_SURGE_MIN)
        & close.gt(sma20)
        & pos.between(SCORE_RANGE_LOW, SCORE_RANGE_HIGH)
        & turn.gt(MIN_TURNOVER)
    )
    score = vol_ratio * chg * pos
    return score.where(valid)


def main() -> None:
    market = load_market(CSV_PATH)

    print(f"\n[2/4] pre-computing signal scores for {len(market)} stocks...")
    t = time.time()
    scores_df = pd.concat({c: score_series(d) for c, d in market.items()}, axis=1)
    next_open_df = pd.concat({c: d["open"].shift(-1) for c, d in market.items()}, axis=1)
    close_df = pd.concat({c: d["close"] for c, d in market.items()}, axis=1)
    print(f"  ok ({time.time() - t:.1f}s) — score matrix {scores_df.shape}")

    all_dates = sorted(scores_df.index)
    backtest_dates = all_dates[WARMUP_BARS:-1]
    print(f"  backtest period: {backtest_dates[0].date()} -> {backtest_dates[-1].date()} ({len(backtest_dates)} days)")

    print(f"\n[3/4] running daily picker (top {PICKS_PER_DAY} per day, cost {COST_RATE * 100:.2f}%)...")
    t = time.time()
    daily_rows = []
    picks_log: dict[str, list] = {}
    no_pick_days = 0
    total_trades = 0

    for date in backtest_dates:
        scores_row = scores_df.loc[date].dropna()
        if scores_row.empty:
            daily_rows.append({"date": date, "n_picks": 0, "gross_ret": 0.0, "net_ret": 0.0})
            no_pick_days += 1
            continue

        top_codes = scores_row.sort_values(ascending=False).head(PICKS_PER_DAY).index
        closes = close_df.loc[date, top_codes]
        next_opens = next_open_df.loc[date, top_codes]
        valid_mask = next_opens.notna() & closes.notna() & (closes > 0)

        if not valid_mask.any():
            daily_rows.append({"date": date, "n_picks": 0, "gross_ret": 0.0, "net_ret": 0.0})
            no_pick_days += 1
            continue

        rets = (next_opens[valid_mask] / closes[valid_mask] - 1)
        gross = rets.mean()
        net = gross - COST_RATE

        daily_rows.append(
            {"date": date, "n_picks": int(valid_mask.sum()),
             "gross_ret": float(gross), "net_ret": float(net)}
        )
        total_trades += int(valid_mask.sum())
        picks_log[date.strftime("%Y-%m-%d")] = [
            {"code": str(c), "score": round(float(scores_row[c]), 3),
             "close": round(float(closes[c]), 2), "next_open": round(float(next_opens[c]), 2),
             "ret_pct": round(float((next_opens[c] / closes[c] - 1) * 100), 2)}
            for c in top_codes[valid_mask]
        ]
    elapsed = time.time() - t
    print(f"  ok ({elapsed:.1f}s) — {total_trades:,} total trades, {no_pick_days} 无信号空仓日")

    eq_df = pd.DataFrame(daily_rows).set_index("date")
    eq_df["equity_gross"] = (1 + eq_df["gross_ret"]).cumprod()
    eq_df["equity_net"] = (1 + eq_df["net_ret"]).cumprod()

    ew_price = close_df.mean(axis=1).reindex(eq_df.index)
    ew_ret = ew_price.pct_change().fillna(0)
    eq_df["benchmark_ret"] = ew_ret
    eq_df["benchmark_equity"] = (1 + ew_ret).cumprod()

    n_days = len(eq_df)
    n_active = int((eq_df["n_picks"] > 0).sum())
    years = n_days / TRADING_DAYS_PER_YEAR

    def _stats(returns: pd.Series, label: str) -> dict:
        cum = (1 + returns).cumprod()
        tot = cum.iloc[-1] - 1
        ann = (1 + tot) ** (1 / years) - 1 if years > 0 else 0
        vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = ann / vol if vol > 0 else 0
        peak = cum.cummax()
        dd = (cum - peak) / peak
        mdd = dd.min()
        win_rate = (returns > 0).sum() / max(1, (returns != 0).sum())
        return {
            "label": label,
            "total_return_pct": tot * 100,
            "annualized_pct": ann * 100,
            "annualized_vol_pct": vol * 100,
            "sharpe": sharpe,
            "max_drawdown_pct": mdd * 100,
            "win_rate_pct": win_rate * 100,
            "best_day_pct": returns.max() * 100,
            "worst_day_pct": returns.min() * 100,
        }

    print(f"\n[4/4] 统计 ({n_days} 交易日, {n_active} 持仓日)...\n")
    strat_gross = _stats(eq_df["gross_ret"], "策略 (gross, 不算成本)")
    strat_net = _stats(eq_df["net_ret"], "策略 (net, 扣 0.25% 成本)")
    bench = _stats(eq_df["benchmark_ret"], "Benchmark (等权全A close-close)")

    rows = [strat_gross, strat_net, bench]
    summary_df = pd.DataFrame(rows).set_index("label").round(2)
    print(summary_df.to_string())

    eq_csv = OUT_DIR / "overnight_equity_curve.csv"
    eq_df.to_csv(eq_csv)

    picks_json = OUT_DIR / "overnight_picks_by_date.json"
    picks_json.write_text(json.dumps(picks_log, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = OUT_DIR / "overnight_backtest_report.md"
    md = ["# 隔夜动量策略回测报告", ""]
    md.append(f"**回测期间**: {eq_df.index[0].date()} → {eq_df.index[-1].date()} ({n_days} 交易日)")
    md.append(f"**股票池**: {len(market):,} 只 A 股 (含主板/中小板/创业板/科创板/北交所)")
    md.append(f"**Top {PICKS_PER_DAY} 等权日选, 持仓日: {n_active}, 总交易笔数: {total_trades:,}**")
    md.append(f"**单次往返成本: {COST_RATE * 100:.2f}%** (印花 0.05 + 佣金 0.05 + 滑点 0.15)")
    md.append("")
    md.append("## 业绩对比")
    md.append("")
    md.append("| 指标 | 策略 (gross) | 策略 (net) | Benchmark 等权全A |")
    md.append("|---|---|---|---|")
    for key, name in [
        ("total_return_pct", "总收益 %"),
        ("annualized_pct", "年化 %"),
        ("annualized_vol_pct", "年化波动 %"),
        ("sharpe", "Sharpe"),
        ("max_drawdown_pct", "最大回撤 %"),
        ("win_rate_pct", "胜率 %"),
        ("best_day_pct", "最佳日 %"),
        ("worst_day_pct", "最差日 %"),
    ]:
        md.append(f"| {name} | {strat_gross[key]:.2f} | {strat_net[key]:.2f} | {bench[key]:.2f} |")
    md.append("")
    md.append("## 假设与免责")
    md.append("- **14:30 选股**用 close[T] 代理，存在尾盘 30 分钟 look-ahead bias")
    md.append("- **T+1 开盘卖出**用 open[T+1]，忽略开盘集合竞价滑点（实际可能多 0.1-0.2%）")
    md.append("- 涨停板 (>9%) 过滤，避免追高；漏掉真正强势股")
    md.append("- 30 日 warmup 排除新股")
    md.append("- 无幸存者偏差（CSV 含全市场，含退市）")
    md.append("- **不构成投资建议**；过往业绩不预示未来")
    report_md.write_text("\n".join(md), encoding="utf-8")

    print(f"\n输出:")
    print(f"  {eq_csv.name}                 # 净值曲线 + 每日明细")
    print(f"  {picks_json.name}            # 每日具体持仓清单")
    print(f"  {report_md.name}             # 业绩报告 Markdown")


if __name__ == "__main__":
    main()
