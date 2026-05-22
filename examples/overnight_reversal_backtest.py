"""Overnight REVERSAL strategy backtest — the inverse of momentum.

Hypothesis: in A 股, today's biggest losers (sold by panicked retail) often
gap up at next-day open. Standard "overnight reversal" effect.

Strategy:
- T 日收盘 (14:30 代理) 对每只 A 股打分
- 取 Top N (按超跌强度) 等权买入收盘价
- T+1 开盘全平
- 重复每个交易日

Signal 过滤（六项）:
  1. 绿盘:        close < open
  2. 深跌未跌停:   -9.5% < change_pct < -2.0%
  3. 放量恐慌:    today_vol > 1.3 x avg_vol(5)
  4. 趋势完好:    close > 0.85 x SMA60   (不是死亡螺旋)
  5. 收盘偏弱不极端: (close - low) / (high - low) in [0.05, 0.50]
  6. 流动性:      turnover > 0.5%
Score = vol_ratio * |chg| * (1 - position_in_range)

同 v1 momentum: 0.25% 单次往返成本、20 只 Top

Run:  python examples/overnight_reversal_backtest.py
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
WARMUP_BARS = 60                    # SMA60 要 60 bars
COST_RATE = 0.0025
DROP_MIN = -2.0                     # 跌幅必须 < -2%
DROP_MAX = -9.5                     # 但不能跌停
TRADING_DAYS_PER_YEAR = 252
SCORE_RANGE_LOW = 0.05
SCORE_RANGE_HIGH = 0.50
VOL_SURGE_MIN = 1.3
TREND_FLOOR = 0.85                  # close 至少 SMA60 的 85% (不是趋势瓦解)
MIN_TURNOVER = 0.005


def load_market(path: Path) -> dict[str, pd.DataFrame]:
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
    """Per-bar reversal score; NaN where any filter fails."""
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"].astype(float)
    turn = df["turnover"]

    chg = close.pct_change() * 100
    avg_vol = vol.shift(1).rolling(5).mean()
    sma60 = close.shift(1).rolling(60).mean()
    pos = (close - low) / (high - low).replace(0, np.nan)
    vol_ratio = vol / avg_vol

    valid = (
        (close < open_)
        & chg.between(DROP_MAX, DROP_MIN)
        & vol_ratio.gt(VOL_SURGE_MIN)
        & close.gt(sma60 * TREND_FLOOR)
        & pos.between(SCORE_RANGE_LOW, SCORE_RANGE_HIGH)
        & turn.gt(MIN_TURNOVER)
    )
    score = vol_ratio * chg.abs() * (1 - pos)
    return score.where(valid)


def main() -> None:
    market = load_market(CSV_PATH)

    print(f"\n[2/4] pre-computing reversal signal scores for {len(market)} stocks...")
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
    strat_gross = _stats(eq_df["gross_ret"], "反转策略 (gross)")
    strat_net = _stats(eq_df["net_ret"], "反转策略 (net, 扣 0.25%)")
    bench = _stats(eq_df["benchmark_ret"], "Benchmark 等权全A")

    summary_df = pd.DataFrame([strat_gross, strat_net, bench]).set_index("label").round(2)
    print(summary_df.to_string())

    (OUT_DIR / "overnight_reversal_equity_curve.csv").write_text(eq_df.to_csv())
    (OUT_DIR / "overnight_reversal_picks_by_date.json").write_text(
        json.dumps(picks_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_md = OUT_DIR / "overnight_reversal_backtest_report.md"
    md = ["# 隔夜反转策略回测报告", ""]
    md.append(f"**回测期间**: {eq_df.index[0].date()} -> {eq_df.index[-1].date()} ({n_days} 交易日)")
    md.append(f"**股票池**: {len(market):,} 只 A 股")
    md.append(f"**Top {PICKS_PER_DAY} 等权日选, 持仓日: {n_active}, 总交易笔数: {total_trades:,}**")
    md.append(f"**单次往返成本: {COST_RATE * 100:.2f}%**")
    md.append("")
    md.append("## 业绩对比")
    md.append("")
    md.append("| 指标 | 反转策略 (gross) | 反转策略 (net) | Benchmark 等权全A |")
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
    md.append("## 假设")
    md.append("- 14:30 用 close[T] 代理（look-ahead 30min）")
    md.append("- T+1 开盘成交价 = open[T+1] (忽略开盘集合竞价滑点)")
    md.append("- 60 日 warmup (用了 SMA60 趋势过滤)")
    md.append("- 趋势底线 close > 0.85 * SMA60 防死亡螺旋")
    md.append("- 跌幅 -9.5% ~ -2% (避免跌停 + 噪音)")
    md.append("- **不构成投资建议**")
    report_md.write_text("\n".join(md), encoding="utf-8")

    print(f"\n输出: overnight_reversal_*.{{csv,json,md}}")


if __name__ == "__main__":
    main()
