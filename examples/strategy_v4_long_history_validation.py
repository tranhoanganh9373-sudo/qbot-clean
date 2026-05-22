"""Re-run v4-5d/Top5 strategy on 3-year history with stability + grid checks.

Requires fetch_long_history.py to have been run first (writes
data_cache/long_history.parquet).

Run:  python examples/strategy_v4_long_history_validation.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "long_history.parquet"
OUT_DIR = Path(__file__).resolve().parent

WARMUP = 60
COST_RATE = 0.0025
STOP_LOSS = -0.05
TRADING_DAYS_PER_YEAR = 252
MOM_VOL_SURGE = 1.3
MIN_TURNOVER = 0.005
N_BUCKETS = 6


def load_market(path: Path) -> dict[str, pd.DataFrame]:
    print(f"loading {path.name}...")
    t = time.time()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    market: dict[str, pd.DataFrame] = {}
    for code, g in df.groupby("code", sort=False):
        if len(g) < WARMUP + 20:
            continue
        g = g.set_index("date")[["open", "high", "low", "close", "volume", "turnover"]].copy()
        g["volume"] = g["volume"].astype("int64")
        market[code] = g
    print(
        f"  ok ({time.time() - t:.1f}s) — {len(market)} stocks  "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )
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
    market, *, picks: int, rebal_freq: int, cost_rate: float = COST_RATE
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


def stability_analysis(market):
    print(f"\n[1/2] 时段稳定性 ({N_BUCKETS} 桶) ...")
    df = run_periodic(market, picks=5, rebal_freq=5)
    n = len(df)
    bucket_size = n // N_BUCKETS
    rows = []
    for i in range(N_BUCKETS):
        start = i * bucket_size
        end = (i + 1) * bucket_size if i < N_BUCKETS - 1 else n
        chunk = df.iloc[start:end]
        s = compute_stats(chunk["net_ret"], 5)
        rows.append({
            "bucket": f"#{i + 1}",
            "start_date": str(chunk.index[0].date()),
            "end_date": str(chunk.index[-1].date()),
            "n_periods": len(chunk),
            **s,
        })
    return pd.DataFrame(rows), df


def grid_analysis(market) -> pd.DataFrame:
    print("\n[2/2] 参数 grid (4 rebal x 4 picks) ...")
    rows = []
    for r in [3, 5, 7, 10]:
        for p in [3, 5, 10, 20]:
            t = time.time()
            df = run_periodic(market, picks=p, rebal_freq=r)
            s = compute_stats(df["net_ret"], r)
            print(
                f"  rebal={r:2d} picks={p:2d}: net={s['total_%']:+8.2f}% "
                f"sharpe={s['sharpe']:+5.2f} mdd={s['mdd_%']:.2f}%  ({time.time() - t:.1f}s)"
            )
            rows.append({"rebal_freq": r, "picks": p, **s})
    return pd.DataFrame(rows).sort_values("total_%", ascending=False).reset_index(drop=True)


def main() -> None:
    if not DATA_PATH.exists():
        print(f"❌ {DATA_PATH} 不存在")
        print("先运行: python examples/fetch_long_history.py")
        return

    market = load_market(DATA_PATH)

    stab, full_df = stability_analysis(market)
    print("\n=== 时段稳定性 ===")
    print(stab.to_string(index=False))

    grid = grid_analysis(market)
    print("\n=== 参数 grid (按 net 总收益降序) ===")
    print(grid.to_string(index=False))

    stab.to_csv(OUT_DIR / "long_history_v4_stability.csv", index=False)
    grid.to_csv(OUT_DIR / "long_history_v4_grid.csv", index=False)

    full = compute_stats(full_df["net_ret"], 5)
    md_parts = [
        "# v4-5d/Top5 长历史验证报告 (HS300+CSI500+CSI1000, 3 年)",
        "",
        "## 完整期 v4-5d/Top5 (net)",
        "",
        f"- **总收益**: {full['total_%']:.2f}%",
        f"- **年化**: {full['ann_%']:.2f}%",
        f"- **年化波动**: {full['ann_vol_%']:.2f}%",
        f"- **Sharpe**: {full['sharpe']:.2f}",
        f"- **最大回撤**: {full['mdd_%']:.2f}%",
        f"- **胜率**: {full['win_%']:.2f}%",
        f"- **总周期数**: {len(full_df)}",
        "",
        "## 时段稳定性",
        "",
        stab.to_markdown(index=False),
        "",
        "## 参数 grid",
        "",
        grid.to_markdown(index=False),
    ]
    (OUT_DIR / "long_history_v4_report.md").write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\n输出: long_history_v4_{{stability.csv, grid.csv, report.md}}")


if __name__ == "__main__":
    main()
