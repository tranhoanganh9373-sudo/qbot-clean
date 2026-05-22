"""Multi-variant strategy comparison on data/deepseek_trading.csv.

Variants tested:
  v1 隔夜动量:       pick close[T]  -> sell open[T+1]   (signal: 强势)
  v2 隔夜反转:       pick close[T]  -> sell open[T+1]   (signal: 超跌反弹)
  v3b 跨日 24h 动量: pick close[T]  -> buy open[T+1]   -> sell open[T+2]  (T+1 合规)
  v3c 多日 + 止损:   pick close[T]  -> buy open[T+1]   -> hold N=3 days OR -5% stop
  Baselines (no signal, 等权全市场):
    overnight: close[T] -> next_open[T+1]
    intraday : open[T]  -> close[T]
    24h      : open[T]  -> open[T+1]

Run:  python examples/strategy_comparison.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "deepseek_trading.csv"
OUT_DIR = Path(__file__).resolve().parent

PICKS_PER_DAY = 20
WARMUP = 60
COST_RATE = 0.0025
TRADING_DAYS_PER_YEAR = 252
HOLD_DAYS = 3
STOP_LOSS = -0.05

MOM_VOL_SURGE = 1.3
MOM_RANGE_LOW = 0.5
MOM_RANGE_HIGH = 0.95
MIN_TURNOVER = 0.005

REV_DROP_MIN = -2.0
REV_DROP_MAX = -9.5
REV_RANGE_LOW = 0.05
REV_RANGE_HIGH = 0.50
REV_TREND_FLOOR = 0.85


def load_market(path: Path) -> dict[str, pd.DataFrame]:
    print(f"loading {path.name}...")
    t = time.time()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    market: dict[str, pd.DataFrame] = {}
    for code, g in df.groupby("code", sort=False):
        if len(g) < WARMUP + 10:
            continue
        g = g.set_index("date")[["open", "high", "low", "close", "volume", "turnover"]].copy()
        g["volume"] = g["volume"].astype("int64")
        market[code] = g
    print(f"  ok ({time.time() - t:.1f}s) — {len(market)} stocks")
    return market


def score_momentum(df: pd.DataFrame) -> pd.Series:
    close, open_, high, low = df["close"], df["open"], df["high"], df["low"]
    vol, turn = df["volume"].astype(float), df["turnover"]
    chg = close.pct_change() * 100
    avg_vol = vol.shift(1).rolling(5).mean()
    sma20 = close.shift(1).rolling(20).mean()
    pos = (close - low) / (high - low).replace(0, np.nan)
    vol_ratio = vol / avg_vol
    valid = (
        (close > open_)
        & chg.between(0.01, 9.0)
        & vol_ratio.gt(MOM_VOL_SURGE)
        & close.gt(sma20)
        & pos.between(MOM_RANGE_LOW, MOM_RANGE_HIGH)
        & turn.gt(MIN_TURNOVER)
    )
    return (vol_ratio * chg * pos).where(valid)


def score_reversal(df: pd.DataFrame) -> pd.Series:
    close, open_, high, low = df["close"], df["open"], df["high"], df["low"]
    vol, turn = df["volume"].astype(float), df["turnover"]
    chg = close.pct_change() * 100
    avg_vol = vol.shift(1).rolling(5).mean()
    sma60 = close.shift(1).rolling(60).mean()
    pos = (close - low) / (high - low).replace(0, np.nan)
    vol_ratio = vol / avg_vol
    valid = (
        (close < open_)
        & chg.between(REV_DROP_MAX, REV_DROP_MIN)
        & vol_ratio.gt(MOM_VOL_SURGE)
        & close.gt(sma60 * REV_TREND_FLOOR)
        & pos.between(REV_RANGE_LOW, REV_RANGE_HIGH)
        & turn.gt(MIN_TURNOVER)
    )
    return (vol_ratio * chg.abs() * (1 - pos)).where(valid)


def run_stateless(score_fn, market, *, entry_kind: str, exit_kind: str) -> pd.DataFrame:
    """Daily picker.

    entry_kind: "close_T"  -> entry = close[T]
                "open_T1"  -> entry = open[T+1]
    exit_kind:  "open_T1"  -> exit = open[T+1]
                "open_T2"  -> exit = open[T+2]
    """
    scores_df = pd.concat({c: score_fn(d) for c, d in market.items()}, axis=1)

    if entry_kind == "close_T":
        entry_df = pd.concat({c: d["close"] for c, d in market.items()}, axis=1)
    elif entry_kind == "open_T1":
        entry_df = pd.concat({c: d["open"].shift(-1) for c, d in market.items()}, axis=1)
    else:
        raise ValueError(entry_kind)

    if exit_kind == "open_T1":
        exit_df = pd.concat({c: d["open"].shift(-1) for c, d in market.items()}, axis=1)
        tail_skip = 1
    elif exit_kind == "open_T2":
        exit_df = pd.concat({c: d["open"].shift(-2) for c, d in market.items()}, axis=1)
        tail_skip = 2
    else:
        raise ValueError(exit_kind)

    all_dates = sorted(scores_df.index)
    backtest_dates = all_dates[WARMUP:-tail_skip]

    daily_rets = []
    for date in backtest_dates:
        scores_row = scores_df.loc[date].dropna()
        if scores_row.empty:
            daily_rets.append((date, 0, 0.0))
            continue
        top_codes = scores_row.sort_values(ascending=False).head(PICKS_PER_DAY).index
        e = entry_df.loc[date, top_codes]
        x = exit_df.loc[date, top_codes]
        valid = e.notna() & x.notna() & (e > 0)
        if not valid.any():
            daily_rets.append((date, 0, 0.0))
            continue
        rets = x[valid] / e[valid] - 1
        daily_rets.append((date, int(valid.sum()), float(rets.mean())))

    df = pd.DataFrame(daily_rets, columns=["date", "n_picks", "gross_ret"]).set_index("date")
    df["net_ret"] = df["gross_ret"] - COST_RATE * (df["n_picks"] > 0).astype(int)
    return df


def run_multi_day(market) -> pd.DataFrame:
    """v3c: pick T close, buy open[T+1], sell at min(open[T+1+N], stop trigger)."""
    scores_df = pd.concat({c: score_momentum(d) for c, d in market.items()}, axis=1)
    open_df = pd.concat({c: d["open"] for c, d in market.items()}, axis=1)
    low_df = pd.concat({c: d["low"] for c, d in market.items()}, axis=1)

    all_dates = sorted(scores_df.index)
    backtest_dates = all_dates[WARMUP:-(HOLD_DAYS + 2)]

    daily_rets = []
    for date in backtest_dates:
        scores_row = scores_df.loc[date].dropna()
        if scores_row.empty:
            daily_rets.append((date, 0, 0.0))
            continue
        top_codes = scores_row.sort_values(ascending=False).head(PICKS_PER_DAY).index
        entry_idx = all_dates.index(date) + 1
        entry_date = all_dates[entry_idx]
        entries = open_df.loc[entry_date, top_codes]
        exit_idx = entry_idx + HOLD_DAYS
        if exit_idx >= len(all_dates):
            daily_rets.append((date, 0, 0.0))
            continue
        exit_date = all_dates[exit_idx]
        exits = open_df.loc[exit_date, top_codes]

        # Stop loss check: any bar in (entry_date, exit_date] where low < entry * (1 + stop)
        hold_lows = low_df.loc[entry_date:exit_date, top_codes].iloc[1:]  # exclude entry day's low

        rets_per_stock = []
        for c in top_codes:
            e_p = entries[c]
            if pd.isna(e_p) or e_p <= 0:
                continue
            stop_price = e_p * (1 + STOP_LOSS)
            col_lows = hold_lows[c]
            stop_hits = col_lows[col_lows < stop_price]
            if len(stop_hits) > 0:
                rets_per_stock.append(STOP_LOSS)
            else:
                x_p = exits[c]
                if pd.notna(x_p):
                    rets_per_stock.append(x_p / e_p - 1)
        if not rets_per_stock:
            daily_rets.append((date, 0, 0.0))
        else:
            daily_rets.append((date, len(rets_per_stock), float(np.mean(rets_per_stock))))

    df = pd.DataFrame(daily_rets, columns=["date", "n_picks", "gross_ret"]).set_index("date")
    df["net_ret"] = df["gross_ret"] - COST_RATE * (df["n_picks"] > 0).astype(int)
    return df


def stats(returns: pd.Series, label: str, n_days: int) -> dict:
    years = n_days / TRADING_DAYS_PER_YEAR
    cum = (1 + returns).cumprod()
    tot = cum.iloc[-1] - 1
    ann = (1 + tot) ** (1 / years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann / vol if vol > 0 else 0
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    win_rate = (returns > 0).sum() / max(1, (returns != 0).sum())
    return {
        "label": label,
        "total_%": round(tot * 100, 2),
        "ann_%": round(ann * 100, 2),
        "vol_%": round(vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "mdd_%": round(mdd * 100, 2),
        "win_%": round(win_rate * 100, 2),
    }


def main() -> None:
    market = load_market(CSV_PATH)

    print("\n[baselines] computing market-wide reference returns...")
    close_df = pd.concat({c: d["close"] for c, d in market.items()}, axis=1)
    open_df = pd.concat({c: d["open"] for c, d in market.items()}, axis=1)
    ew_overnight = (open_df.shift(-1) / close_df - 1).mean(axis=1)
    ew_intraday = (close_df / open_df - 1).mean(axis=1)
    ew_24h = (open_df.shift(-1) / open_df - 1).mean(axis=1)

    print("\n[v1] 隔夜动量 (强势追涨, close[T] -> open[T+1])...")
    t = time.time()
    df_v1 = run_stateless(score_momentum, market, entry_kind="close_T", exit_kind="open_T1")
    print(f"  ok ({time.time() - t:.1f}s)")

    print("\n[v2] 隔夜反转 (超跌反弹, close[T] -> open[T+1])...")
    t = time.time()
    df_v2 = run_stateless(score_reversal, market, entry_kind="close_T", exit_kind="open_T1")
    print(f"  ok ({time.time() - t:.1f}s)")

    print("\n[v3b] 跨日 24h 动量 (open[T+1] -> open[T+2], T+1合规)...")
    t = time.time()
    df_v3b = run_stateless(score_momentum, market, entry_kind="open_T1", exit_kind="open_T2")
    print(f"  ok ({time.time() - t:.1f}s)")

    print(f"\n[v3c] 多日动量 + {int(STOP_LOSS*100)}% 止损 ({HOLD_DAYS} 日)...")
    t = time.time()
    df_v3c = run_multi_day(market)
    print(f"  ok ({time.time() - t:.1f}s)")

    common_dates = (
        df_v1.index.intersection(df_v2.index).intersection(df_v3b.index).intersection(df_v3c.index)
    )
    n = len(common_dates)
    print(f"\n[stats] {n} 共同交易日 ({common_dates[0].date()} → {common_dates[-1].date()})")

    rows = []
    rows.append(stats(ew_overnight.reindex(common_dates).fillna(0), "Baseline 隔夜 close→next_open", n))
    rows.append(stats(ew_intraday.reindex(common_dates).fillna(0), "Baseline 日内 open→close", n))
    rows.append(stats(ew_24h.reindex(common_dates).fillna(0), "Baseline 24h open→next_open", n))
    rows.append(stats(df_v1.loc[common_dates, "gross_ret"], "v1 隔夜动量 (gross)", n))
    rows.append(stats(df_v1.loc[common_dates, "net_ret"], "v1 隔夜动量 (net)", n))
    rows.append(stats(df_v2.loc[common_dates, "gross_ret"], "v2 隔夜反转 (gross)", n))
    rows.append(stats(df_v2.loc[common_dates, "net_ret"], "v2 隔夜反转 (net)", n))
    rows.append(stats(df_v3b.loc[common_dates, "gross_ret"], "v3b 跨日24h动量 (gross)", n))
    rows.append(stats(df_v3b.loc[common_dates, "net_ret"], "v3b 跨日24h动量 (net)", n))
    rows.append(stats(df_v3c.loc[common_dates, "gross_ret"], "v3c 多日+止损 (gross)", n))
    rows.append(stats(df_v3c.loc[common_dates, "net_ret"], "v3c 多日+止损 (net)", n))

    summary = pd.DataFrame(rows).set_index("label")
    print("\n=== 综合对照表 ===")
    print(summary.to_string())

    eq = pd.DataFrame({
        "v1_net": (1 + df_v1.loc[common_dates, "net_ret"]).cumprod(),
        "v2_net": (1 + df_v2.loc[common_dates, "net_ret"]).cumprod(),
        "v3b_net": (1 + df_v3b.loc[common_dates, "net_ret"]).cumprod(),
        "v3c_net": (1 + df_v3c.loc[common_dates, "net_ret"]).cumprod(),
        "baseline_overnight": (1 + ew_overnight.reindex(common_dates).fillna(0)).cumprod(),
        "baseline_intraday": (1 + ew_intraday.reindex(common_dates).fillna(0)).cumprod(),
        "baseline_24h": (1 + ew_24h.reindex(common_dates).fillna(0)).cumprod(),
    })
    eq.to_csv(OUT_DIR / "strategy_comparison_equity.csv")
    (OUT_DIR / "strategy_comparison_summary.md").write_text(
        "# 策略对比汇总\n\n" + summary.to_markdown(), encoding="utf-8"
    )

    print("\n输出:")
    print("  strategy_comparison_equity.csv     # 7 条净值曲线")
    print("  strategy_comparison_summary.md     # 业绩对照表")


if __name__ == "__main__":
    main()
