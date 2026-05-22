"""v4 strategy = v3c 多日动量 + 两个优化:
   B. 周 rebalance (5 trading days per cycle) — 大幅降换手成本
   C. Top 5 集中 (vs Top 20) — alpha 集中

Workflow per cycle (5 trading days = 1 周):
  T 日收盘: 跑 score_momentum, 选 Top 5
  T+1 日开盘: 等权买入
  持仓直至 T+1+N 日开盘
  期间任意 bar low < entry * 0.95 即 -5% 止损 (按 stop_price 平仓)

为了对比, 同时跑 4 个配置:
  v4-3d/Top5   每 3 个交易日 rebalance
  v4-5d/Top5   每 5 个交易日 (周) rebalance
  v4-10d/Top5  每 10 个交易日 (半月) rebalance
  v4-5d/Top3   每 5 个 + 更集中

Run:  python examples/strategy_v4_weekly_concentrated.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "deepseek_trading.csv"
OUT_DIR = Path(__file__).resolve().parent

WARMUP = 60
COST_RATE = 0.0025
STOP_LOSS = -0.05
TRADING_DAYS_PER_YEAR = 252

MOM_VOL_SURGE = 1.3
MIN_TURNOVER = 0.005


def load_market(path: Path) -> dict[str, pd.DataFrame]:
    print(f"loading {path.name}...")
    t = time.time()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    market: dict[str, pd.DataFrame] = {}
    for code, g in df.groupby("code", sort=False):
        if len(g) < WARMUP + 20:
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
        & pos.between(0.5, 0.95)
        & turn.gt(MIN_TURNOVER)
    )
    return (vol_ratio * chg * pos).where(valid)


def run_periodic(market, *, picks: int, rebal_freq: int, label: str) -> pd.DataFrame:
    """Pick every rebal_freq trading days, hold for rebal_freq days (no overlap)."""
    scores_df = pd.concat({c: score_momentum(d) for c, d in market.items()}, axis=1)
    open_df = pd.concat({c: d["open"] for c, d in market.items()}, axis=1)
    low_df = pd.concat({c: d["low"] for c, d in market.items()}, axis=1)

    all_dates = sorted(scores_df.index)
    pick_dates = []
    i = WARMUP
    while i + rebal_freq + 1 < len(all_dates):
        pick_dates.append(all_dates[i])
        i += rebal_freq

    rows = []
    for pd_date in pick_dates:
        scores_row = scores_df.loc[pd_date].dropna()
        if scores_row.empty:
            rows.append({"date": pd_date, "n_picks": 0, "period_ret": 0.0})
            continue
        top_codes = scores_row.sort_values(ascending=False).head(picks).index
        entry_idx = all_dates.index(pd_date) + 1
        entry_date = all_dates[entry_idx]
        exit_idx = entry_idx + rebal_freq
        if exit_idx >= len(all_dates):
            rows.append({"date": pd_date, "n_picks": 0, "period_ret": 0.0})
            continue
        exit_date = all_dates[exit_idx]
        entries = open_df.loc[entry_date, top_codes]
        exits = open_df.loc[exit_date, top_codes]
        hold_lows = low_df.loc[entry_date:exit_date, top_codes].iloc[1:]

        rets_per_stock = []
        for c in top_codes:
            e_p = entries[c]
            if pd.isna(e_p) or e_p <= 0:
                continue
            stop_price = e_p * (1 + STOP_LOSS)
            col_lows = hold_lows[c]
            if (col_lows < stop_price).any():
                rets_per_stock.append(STOP_LOSS)
            else:
                x_p = exits[c]
                if pd.notna(x_p):
                    rets_per_stock.append(x_p / e_p - 1)
        period_ret = float(np.mean(rets_per_stock)) if rets_per_stock else 0.0
        rows.append({"date": pd_date, "n_picks": len(rets_per_stock), "period_ret": period_ret})

    df = pd.DataFrame(rows).set_index("date")
    df["net_ret"] = df["period_ret"] - COST_RATE * (df["n_picks"] > 0).astype(int)
    df.attrs["label"] = label
    df.attrs["rebal_freq"] = rebal_freq
    return df


def stats_one(df: pd.DataFrame, label: str) -> list:
    rebal_freq = df.attrs["rebal_freq"]
    periods_per_year = TRADING_DAYS_PER_YEAR / rebal_freq

    def _calc(returns: pd.Series, name: str) -> dict:
        cum = (1 + returns).cumprod()
        tot = cum.iloc[-1] - 1
        n_periods = len(returns)
        years = n_periods / periods_per_year
        ann = (1 + tot) ** (1 / years) - 1 if years > 0 else 0
        vol_per_period = returns.std()
        ann_vol = vol_per_period * np.sqrt(periods_per_year)
        sharpe = ann / ann_vol if ann_vol > 0 else 0
        peak = cum.cummax()
        mdd = ((cum - peak) / peak).min()
        win_rate = (returns > 0).sum() / max(1, (returns != 0).sum())
        return {
            "label": f"{label} ({name})",
            "n_periods": n_periods,
            "total_%": round(tot * 100, 2),
            "ann_%": round(ann * 100, 2),
            "ann_vol_%": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 2),
            "mdd_%": round(mdd * 100, 2),
            "win_%": round(win_rate * 100, 2),
            "trades": int(df["n_picks"].sum()),
        }

    return [_calc(df["period_ret"], "gross"), _calc(df["net_ret"], "net")]


def main() -> None:
    market = load_market(CSV_PATH)

    variants = [
        ("v4-3d  每3日, Top5", 5, 3),
        ("v4-5d  每5日, Top5", 5, 5),
        ("v4-10d 每10日, Top5", 5, 10),
        ("v4-5d  每5日, Top3", 3, 5),
    ]

    rows = []
    eq_dict = {}
    for label, picks, freq in variants:
        print(f"\n[{label}] rebal_freq={freq}, picks={picks}...")
        t = time.time()
        df = run_periodic(market, picks=picks, rebal_freq=freq, label=label)
        print(f"  ok ({time.time() - t:.1f}s) — {len(df)} 周期, "
              f"avg_picks_per_period={df['n_picks'].mean():.1f}, "
              f"net_total={(1 + df['net_ret']).prod() - 1:+.2%}")
        rows.extend(stats_one(df, label))
        eq_dict[f"{label}_gross"] = (1 + df["period_ret"]).cumprod()
        eq_dict[f"{label}_net"] = (1 + df["net_ret"]).cumprod()

    summary = pd.DataFrame(rows).set_index("label")
    print("\n=== v4 多频次/集中度对照 ===")
    print(summary.to_string())

    eq_df = pd.DataFrame(eq_dict)
    eq_df.to_csv(OUT_DIR / "strategy_v4_equity.csv")
    (OUT_DIR / "strategy_v4_report.md").write_text(
        "# v4 周 rebalance + Top 5 集中策略\n\n" + summary.to_markdown(), encoding="utf-8"
    )

    print("\n输出: strategy_v4_{equity.csv, report.md}")


if __name__ == "__main__":
    main()
