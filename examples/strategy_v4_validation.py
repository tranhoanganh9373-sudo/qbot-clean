"""v4 strategy validation — three robustness checks on data/deepseek_trading.csv:

1. 时段稳定性: 把 40 周期切成 5 个 ~8 周期桶 (~2 个月一桶), 看 alpha 是否
   每个时段都有, 还是集中在 1-2 段牛市行情里
2. 参数 grid: 4 rebal_freq x 4 picks = 16 配置, 看 5d/Top5 是局部最优
   (大多数邻近配置也 +) 还是孤立 cherry-pick
3. 滑点敏感性: cost 从 0.1% 到 1.0% 扫一遍, 找盈亏平衡点

Run:  python examples/strategy_v4_validation.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "deepseek_trading.csv"
OUT_DIR = Path(__file__).resolve().parent

WARMUP = 60
DEFAULT_COST = 0.0025
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


def run_periodic(
    market, *, picks: int, rebal_freq: int, cost_rate: float = DEFAULT_COST
) -> pd.DataFrame:
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
    df["net_ret"] = df["period_ret"] - cost_rate * (df["n_picks"] > 0).astype(int)
    return df


def compute_stats(returns: pd.Series, rebal_freq: int) -> dict:
    periods_per_year = TRADING_DAYS_PER_YEAR / rebal_freq
    cum = (1 + returns).cumprod()
    tot = cum.iloc[-1] - 1
    years = len(returns) / periods_per_year
    ann = (1 + tot) ** (1 / years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann / vol if vol > 0 else 0
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    win_rate = (returns > 0).sum() / max(1, (returns != 0).sum())
    return {
        "total_%": round(tot * 100, 2),
        "ann_%": round(ann * 100, 2),
        "ann_vol_%": round(vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "mdd_%": round(mdd * 100, 2),
        "win_%": round(win_rate * 100, 2),
    }


def stability_analysis(market) -> pd.DataFrame:
    print("\n[1/3] 时段稳定性: 5 桶 x ~8 周期 (~2 月 / 桶) ...")
    df = run_periodic(market, picks=5, rebal_freq=5)
    n = len(df)
    bucket_size = n // 5
    rows = []
    for i in range(5):
        start = i * bucket_size
        end = (i + 1) * bucket_size if i < 4 else n
        chunk = df.iloc[start:end]
        s = compute_stats(chunk["net_ret"], 5)
        rows.append({
            "bucket": f"#{i + 1} 期 {start + 1}-{end}",
            "start_date": str(chunk.index[0].date()),
            "end_date": str(chunk.index[-1].date()),
            "n_periods": len(chunk),
            **s,
        })
    return pd.DataFrame(rows)


def grid_analysis(market) -> pd.DataFrame:
    print("\n[2/3] 参数 grid: 4 rebal x 4 picks = 16 configs ...")
    rebal_grid = [3, 5, 7, 10]
    picks_grid = [3, 5, 10, 20]
    rows = []
    for r in rebal_grid:
        for p in picks_grid:
            t = time.time()
            df = run_periodic(market, picks=p, rebal_freq=r)
            s = compute_stats(df["net_ret"], r)
            print(
                f"  rebal={r:2d} picks={p:2d}: net={s['total_%']:+7.2f}% "
                f"sharpe={s['sharpe']:+5.2f} mdd={s['mdd_%']:.2f}%  ({time.time() - t:.1f}s)"
            )
            rows.append({"rebal_freq": r, "picks": p, **s})
    return pd.DataFrame(rows).sort_values("total_%", ascending=False).reset_index(drop=True)


def slippage_analysis(market) -> pd.DataFrame:
    print("\n[3/3] 滑点敏感性: cost 0.10% -> 1.00% (v4-5d/Top5) ...")
    costs = [0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050, 0.0075, 0.0100]
    df_gross = run_periodic(market, picks=5, rebal_freq=5, cost_rate=0.0)
    rows = []
    for cost in costs:
        net = df_gross["period_ret"] - cost * (df_gross["n_picks"] > 0).astype(int)
        s = compute_stats(net, 5)
        rows.append({"cost_per_round_trip_%": round(cost * 100, 3), **s})
        print(f"  cost={cost * 100:.2f}%: net_total={s['total_%']:+.2f}% sharpe={s['sharpe']:+.2f}")
    return pd.DataFrame(rows)


def main() -> None:
    market = load_market(CSV_PATH)

    stab = stability_analysis(market)
    print("\n=== 时段稳定性 ===")
    print(stab.to_string(index=False))

    grid = grid_analysis(market)
    print("\n=== 参数 grid (按 net 总收益降序) ===")
    print(grid.to_string(index=False))

    slip = slippage_analysis(market)
    print("\n=== 滑点敏感性 ===")
    print(slip.to_string(index=False))

    stab.to_csv(OUT_DIR / "v4_validation_stability.csv", index=False)
    grid.to_csv(OUT_DIR / "v4_validation_grid.csv", index=False)
    slip.to_csv(OUT_DIR / "v4_validation_slippage.csv", index=False)

    md_parts = ["# v4 策略验证报告", ""]
    md_parts += ["## 1. 时段稳定性 (v4-5d/Top5 每 ~2 月一桶)", "", stab.to_markdown(index=False), ""]
    md_parts += ["## 2. 参数 grid", "", grid.to_markdown(index=False), ""]
    md_parts += ["## 3. 滑点敏感性 (v4-5d/Top5)", "", slip.to_markdown(index=False), ""]
    (OUT_DIR / "v4_validation_report.md").write_text("\n".join(md_parts), encoding="utf-8")

    print("\n输出:")
    print("  v4_validation_stability.csv  v4_validation_grid.csv  v4_validation_slippage.csv")
    print("  v4_validation_report.md")


if __name__ == "__main__":
    main()
