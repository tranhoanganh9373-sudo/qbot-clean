"""v5 LightGBM (clean rewrite) vs v4 momentum.

Fixes vs /Volumes/SSD/finance/deepseek_finance/scripts/backtest_lgb.py:
- ❌ board MA5 used `target` (future return) → ✅ uses ret_1d rolling().shift(1)
- ❌ pick at T close, sell at T+1 close (look-ahead + impossible execution)
   → ✅ pick at T close, buy at T+1 open, sell at T+1+N open (T+1 compliant)
- ❌ zero transaction cost → ✅ 0.25% per round-trip
- ❌ cumulative = arithmetic sum → ✅ cumulative = compound product
- ✅ time-based train/test split (前 50% train, 后 50% test)

Two variants:
  v5-daily : daily rebalance, 24h hold (matches original v5 cadence)
  v5-5d    : 5-day rebalance, 5-day hold (matches v4-5d/Top5 for fair comparison)

Run:  python examples/strategy_v5_lgb_clean.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "deepseek_trading.csv"
OUT_DIR = Path(__file__).resolve().parent

PICKS_PER_DAY = 5
COST_RATE = 0.0025
MIN_MARKET_CAP = 10
MIN_TURNOVER = 0.005
TRADING_DAYS_PER_YEAR = 252
WARMUP = 60

LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1,
)


def get_board(code: str) -> str:
    if code.startswith("sh60"): return "沪主板"
    if code.startswith("sh68"): return "科创板"
    if code.startswith("sz00"): return "深主板"
    if code.startswith("sz30"): return "创业板"
    if code.startswith("bj"): return "北交所"
    return "其他"


def build_features(market: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for code, grp in market.items():
        c = grp["close"]; o = grp["open"]; v = grp["volume"]
        os_ = grp["outstanding_share"]
        if len(c) < WARMUP: continue
        f = pd.DataFrame(index=grp.index)
        f["ret_1d"] = c.pct_change(1)
        f["ret_3d"] = c.pct_change(3)
        f["ret_5d"] = c.pct_change(5)
        f["ret_10d"] = c.pct_change(10)
        f["ret_20d"] = c.pct_change(20)
        f["intraday"] = (c - o) / o
        ma5 = c.rolling(5).mean(); ma10 = c.rolling(10).mean()
        ma20 = c.rolling(20).mean(); ma60 = c.rolling(60).mean()
        f["ma5_dist"] = (c - ma5) / ma5
        f["ma20_dist"] = (c - ma20) / ma20
        f["ma60_dist"] = (c - ma60) / ma60
        f["ma5_ma10"] = (ma5 - ma10) / ma10
        f["vol10"] = c.pct_change().rolling(10).std() * np.sqrt(10)
        f["vol20"] = c.pct_change().rolling(20).std() * np.sqrt(20)
        vm = v.rolling(20).mean().replace(0, np.nan)
        f["vol_ratio"] = v / vm
        d = c.diff(); g = d.clip(lower=0); ls = (-d).clip(lower=0)
        rs = g.rolling(14).mean() / ls.rolling(14).mean().replace(0, np.nan)
        f["rsi"] = 100 - 100 / (1 + rs)
        bm = c.rolling(20).mean(); bs = c.rolling(20).std()
        f["bb_pos"] = (c - (bm - 2 * bs)) / (4 * bs.replace(0, np.nan))
        e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
        macd_ = e12 - e26; sig = macd_.ewm(span=9).mean()
        f["macd"] = macd_ - sig
        f["turnover"] = v / os_
        f["mkt_cap"] = os_ * c / 1e8
        f["target_bin"] = ((c.shift(-1) / c - 1) > 0).astype(int)
        f["code"] = code
        f["board"] = get_board(code)
        parts.append(f)
    return pd.concat(parts).dropna()


def board_rotation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """5-day rolling avg of ret_1d per board, SHIFTED so day T uses [T-5..T-1] only."""
    daily_board = features.groupby([features.index, "board"])["ret_1d"].mean()
    pivoted = daily_board.unstack("board")
    return pivoted.rolling(5).mean().shift(1)


def run_v5(
    test: pd.DataFrame,
    *,
    open_df: pd.DataFrame,
    board_5d_avg: pd.DataFrame,
    all_dates: list,
    rebal_freq: int,
    picks: int = PICKS_PER_DAY,
    cost_rate: float = COST_RATE,
) -> pd.DataFrame:
    test_dates = sorted(test.index.unique())
    if rebal_freq == 1:
        pick_indices = list(range(len(test_dates)))
    else:
        pick_indices = list(range(0, len(test_dates), rebal_freq))

    daily_rows = []
    for pi in pick_indices:
        pick_day = test_dates[pi]
        global_idx = all_dates.index(pick_day)
        if global_idx + 1 + rebal_freq >= len(all_dates):
            break
        buy_date = all_dates[global_idx + 1]
        sell_date = all_dates[global_idx + 1 + rebal_freq]

        day_data = test[test.index == pick_day].copy()
        if day_data.empty:
            daily_rows.append({"date": pick_day, "n_picks": 0, "gross_ret": 0.0, "net_ret": 0.0})
            continue

        day_data = day_data[(day_data["mkt_cap"] >= MIN_MARKET_CAP) &
                            (day_data["turnover"] >= MIN_TURNOVER)]
        day_data = day_data[~day_data["code"].str.contains("900|200", na=False)]

        if pick_day in board_5d_avg.index:
            scores = board_5d_avg.loc[pick_day].dropna()
            top3 = scores.nlargest(3).index.tolist() if len(scores) else \
                   ["沪主板", "深主板", "创业板"]
            day_data = day_data[day_data["board"].isin(top3)]

        if len(day_data) < picks:
            daily_rows.append({"date": pick_day, "n_picks": 0, "gross_ret": 0.0, "net_ret": 0.0})
            continue

        top = day_data.nlargest(picks, "prob")
        codes = top["code"].tolist()
        try:
            buy_p = open_df.loc[buy_date, codes]
            sell_p = open_df.loc[sell_date, codes]
        except KeyError:
            daily_rows.append({"date": pick_day, "n_picks": 0, "gross_ret": 0.0, "net_ret": 0.0})
            continue

        valid = buy_p.notna() & sell_p.notna() & (buy_p > 0)
        if not valid.any():
            daily_rows.append({"date": pick_day, "n_picks": 0, "gross_ret": 0.0, "net_ret": 0.0})
            continue

        rets = (sell_p[valid] / buy_p[valid] - 1)
        gross = float(rets.mean())
        net = gross - cost_rate
        daily_rows.append({
            "date": pick_day, "n_picks": int(valid.sum()),
            "gross_ret": gross, "net_ret": net,
        })

    return pd.DataFrame(daily_rows).set_index("date")


def stats(returns: pd.Series, periods_per_year: float, label: str) -> dict:
    cum = (1 + returns).cumprod()
    tot = cum.iloc[-1] - 1
    n = len(returns)
    years = n / periods_per_year
    ann = (1 + tot) ** (1 / years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann / vol if vol > 0 else 0
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    win = (returns > 0).sum() / max(1, (returns != 0).sum())
    return {
        "label": label, "n_periods": n,
        "total_%": round(tot * 100, 2),
        "ann_%": round(ann * 100, 2),
        "vol_%": round(vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "mdd_%": round(mdd * 100, 2),
        "win_%": round(win * 100, 2),
    }


def main() -> None:
    print(f"[1/5] loading {CSV_PATH.name}...")
    t = time.time()
    df = pd.read_csv(CSV_PATH, dtype={"code": str}, parse_dates=["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = df.set_index("date")
    market = {code: g for code, g in df.groupby("code") if len(g) >= WARMUP + 5}
    print(f"  ok ({time.time() - t:.1f}s) — {len(market)} stocks")

    print(f"\n[2/5] features ...")
    t = time.time()
    features = build_features(market)
    fcols = [c for c in features.columns if c not in ["code", "target_bin", "board"]]
    print(f"  ok ({time.time() - t:.1f}s) — {len(features):,} rows, {len(fcols)} features")

    print(f"\n[3/5] board rotation matrix (shifted, no leak) ...")
    board_5d_avg = board_rotation_matrix(features)
    open_df = pd.concat({c: g["open"] for c, g in market.items()}, axis=1)

    all_dates = sorted(features.index.unique())
    split_idx = len(all_dates) // 2
    train_end = all_dates[split_idx - 1]
    test_start = all_dates[split_idx]
    train = features[features.index <= train_end]
    test = features[features.index >= test_start].copy()
    print(f"\n[4/5] train+predict ...")
    print(f"  train: {train.index.min().date()} → {train.index.max().date()} ({len(train):,})")
    print(f"  test:  {test.index.min().date()} → {test.index.max().date()} ({len(test):,})")
    t = time.time()
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(train[fcols].values, train["target_bin"].values)
    test["prob"] = model.predict_proba(test[fcols].values)[:, 1]
    print(f"  trained ({time.time() - t:.1f}s)")

    print(f"\n[5/5] backtest 2 variants ...")
    df_daily = run_v5(test, open_df=open_df, board_5d_avg=board_5d_avg,
                      all_dates=all_dates, rebal_freq=1)
    df_5d = run_v5(test, open_df=open_df, board_5d_avg=board_5d_avg,
                   all_dates=all_dates, rebal_freq=5)

    print(f"\n  v5-daily: {len(df_daily)} pick days, {df_daily['n_picks'].sum()} trades")
    print(f"  v5-5d:    {len(df_5d)} pick periods, {df_5d['n_picks'].sum()} trades")

    rows = []
    rows.append(stats(df_daily["gross_ret"], TRADING_DAYS_PER_YEAR, "v5-daily LGB (gross, 24h)"))
    rows.append(stats(df_daily["net_ret"], TRADING_DAYS_PER_YEAR, "v5-daily LGB (net, 24h)"))
    rows.append(stats(df_5d["gross_ret"], TRADING_DAYS_PER_YEAR / 5, "v5-5d LGB (gross, 5d)"))
    rows.append(stats(df_5d["net_ret"], TRADING_DAYS_PER_YEAR / 5, "v5-5d LGB (net, 5d)"))

    summary = pd.DataFrame(rows).set_index("label").round(2)
    print(f"\n=== v5 clean 业绩 ===")
    print(summary.to_string())

    print(f"\n=== 对照 v4-5d/Top5 (同 CSV 同期) ===")
    print("  net total +61.19%, ann +82.50%, Sharpe 1.58, MDD -19.04%, win 55.00%, 40 周期")

    df_daily.to_csv(OUT_DIR / "strategy_v5_daily_equity.csv")
    df_5d.to_csv(OUT_DIR / "strategy_v5_5d_equity.csv")
    md = ["# v5 LightGBM (clean) vs v4 momentum", ""]
    md.append(f"**回测期**: test set = {test.index.min().date()} → {test.index.max().date()}")
    md.append(f"**股票池**: {len(market):,} 只 A 股")
    md.append(f"**单次往返成本**: {COST_RATE * 100:.2f}%")
    md.append(f"**Train/test 切分**: 50/50 时序")
    md.append(f"**修复点**: board 用 shifted ret_1d (无 leak); buy T+1 open, sell T+1+N open (T+1 合规)")
    md.append("")
    md.append("## v5 clean 两变体业绩")
    md.append("")
    md.append(summary.to_markdown())
    md.append("")
    md.append("## v4-5d/Top5 业绩 (同 CSV 同期对照)")
    md.append("")
    md.append("- net +61.19%, ann +82.50%, Sharpe 1.58, MDD -19.04%, win 55.00%, 40 周期")
    (OUT_DIR / "strategy_v5_clean_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出: strategy_v5_{{daily,5d}}_equity.csv  strategy_v5_clean_report.md")


if __name__ == "__main__":
    main()
