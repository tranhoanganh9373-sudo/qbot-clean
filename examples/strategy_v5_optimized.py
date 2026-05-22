"""v5 optimization — two variants targeting v5-clean's weaknesses:

v5a: 5d-label LGB + 5d rebal + -5% stop
  v5-clean trains on (close[T+1]/close[T] > 0) — 1-day horizon.
  Fix: train on (close[T+5]/close[T] > 0), align signal w/ 5d hold.
  + Add -5% intra-period stop loss (matches v4-5d/Top5).

v5c: ensemble = LGB_prob(5d) × v4 momentum score, 5d rebal
  Try stacking orthogonal alphas.

Cost / execution: same as v5-clean (0.25% round-trip, T+1 open buy, T+6 open sell).

Run:  python examples/strategy_v5_optimized.py
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

PICKS = 5
COST_RATE = 0.0025
MIN_MARKET_CAP = 10
MIN_TURNOVER = 0.005
TRADING_DAYS_PER_YEAR = 252
WARMUP = 60
HOLD_DAYS = 5
STOP_LOSS = -0.05

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


def build_features(market: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    """Build features + H-day forward label + v4 momentum score column."""
    parts = []
    for code, grp in market.items():
        c = grp["close"]; o = grp["open"]; v = grp["volume"]
        h = grp["high"]; l = grp["low"]
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
        # H-day forward label
        f["target_bin"] = ((c.shift(-horizon) / c - 1) > 0).astype(int)
        # v4 momentum (for ensemble; NaN where v4 filter fails)
        chg_pct = c.pct_change() * 100
        avg_vol5 = v.shift(1).rolling(5).mean()
        sma20_v4 = c.shift(1).rolling(20).mean()
        pos = (c - l) / (h - l).replace(0, np.nan)
        v4_valid = (
            (c > o)
            & chg_pct.between(0.01, 9.0)
            & ((v / avg_vol5) > 1.3)
            & (c > sma20_v4)
            & pos.between(0.5, 0.95)
            & (v / os_ > MIN_TURNOVER)
        )
        f["v4_score"] = ((v / avg_vol5) * chg_pct * pos).where(v4_valid)
        f["code"] = code
        f["board"] = get_board(code)
        parts.append(f)
    return pd.concat(parts).dropna(
        subset=["ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "intraday",
                "ma5_dist", "ma20_dist", "ma60_dist", "ma5_ma10",
                "vol10", "vol20", "vol_ratio", "rsi", "bb_pos", "macd",
                "turnover", "mkt_cap", "target_bin"]
    )


def board_rotation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    daily_board = features.groupby([features.index, "board"])["ret_1d"].mean()
    pivoted = daily_board.unstack("board")
    return pivoted.rolling(5).mean().shift(1)


def run_picker(
    test: pd.DataFrame,
    *,
    rank_col: str,
    open_df: pd.DataFrame,
    low_df: pd.DataFrame,
    board_5d_avg: pd.DataFrame,
    all_dates: list,
    hold_days: int,
    picks: int = PICKS,
    cost_rate: float = COST_RATE,
    use_stop: bool = True,
) -> pd.DataFrame:
    test_dates = sorted(test.index.unique())
    pick_indices = list(range(0, len(test_dates), hold_days))

    daily_rows = []
    for pi in pick_indices:
        pick_day = test_dates[pi]
        global_idx = all_dates.index(pick_day)
        if global_idx + 1 + hold_days >= len(all_dates):
            break
        buy_date = all_dates[global_idx + 1]
        sell_date = all_dates[global_idx + 1 + hold_days]

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

        day_data = day_data.dropna(subset=[rank_col])
        if len(day_data) < picks:
            daily_rows.append({"date": pick_day, "n_picks": 0, "gross_ret": 0.0, "net_ret": 0.0})
            continue

        top = day_data.nlargest(picks, rank_col)
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

        codes_v = [c for c, ok in zip(codes, valid) if ok]
        rets_per_stock = []
        if use_stop:
            hold_lows = low_df.loc[buy_date:sell_date, codes_v].iloc[1:]
        for c in codes_v:
            ep = buy_p[c]
            stop_price = ep * (1 + STOP_LOSS)
            if use_stop:
                col_lows = hold_lows[c]
                if (col_lows < stop_price).any():
                    rets_per_stock.append(STOP_LOSS)
                    continue
            xp = sell_p[c]
            rets_per_stock.append(xp / ep - 1)
        gross = float(np.mean(rets_per_stock)) if rets_per_stock else 0.0
        daily_rows.append({"date": pick_day, "n_picks": len(rets_per_stock),
                            "gross_ret": gross, "net_ret": gross - cost_rate})

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
    print(f"[1/4] loading {CSV_PATH.name}...")
    t = time.time()
    df = pd.read_csv(CSV_PATH, dtype={"code": str}, parse_dates=["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = df.set_index("date")
    market = {code: g for code, g in df.groupby("code") if len(g) >= WARMUP + 5}
    print(f"  ok ({time.time() - t:.1f}s) — {len(market)} stocks")

    print(f"\n[2/4] features (label = {HOLD_DAYS}d forward direction) ...")
    t = time.time()
    features = build_features(market, horizon=HOLD_DAYS)
    fcols = [c for c in features.columns
             if c not in ("code", "target_bin", "board", "v4_score")]
    print(f"  ok ({time.time() - t:.1f}s) — {len(features):,} rows, {len(fcols)} features")

    print(f"\n[3/4] board rotation + price matrices ...")
    board_5d_avg = board_rotation_matrix(features)
    open_df = pd.concat({c: g["open"] for c, g in market.items()}, axis=1)
    low_df = pd.concat({c: g["low"] for c, g in market.items()}, axis=1)

    all_dates = sorted(features.index.unique())
    split_idx = len(all_dates) // 2
    train_end = all_dates[split_idx - 1]
    test_start = all_dates[split_idx]
    train = features[features.index <= train_end]
    test = features[features.index >= test_start].copy()
    print(f"  train: {train.index.min().date()} → {train.index.max().date()} ({len(train):,})")
    print(f"  test:  {test.index.min().date()} → {test.index.max().date()} ({len(test):,})")

    print(f"\n[4/4] training LGB + 3 variants ...")
    t = time.time()
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(train[fcols].values, train["target_bin"].values)
    test["lgb_prob"] = model.predict_proba(test[fcols].values)[:, 1]
    print(f"  trained ({time.time() - t:.1f}s)")

    test["ensemble_score"] = test["lgb_prob"] * test["v4_score"]

    print(f"\nrunning 3 variants ...")
    df_v5a = run_picker(test, rank_col="lgb_prob", open_df=open_df, low_df=low_df,
                         board_5d_avg=board_5d_avg, all_dates=all_dates,
                         hold_days=HOLD_DAYS, use_stop=True)
    df_v5a_nostop = run_picker(test, rank_col="lgb_prob", open_df=open_df, low_df=low_df,
                                board_5d_avg=board_5d_avg, all_dates=all_dates,
                                hold_days=HOLD_DAYS, use_stop=False)
    df_v5c = run_picker(test, rank_col="ensemble_score", open_df=open_df, low_df=low_df,
                         board_5d_avg=board_5d_avg, all_dates=all_dates,
                         hold_days=HOLD_DAYS, use_stop=True)

    print(f"\n  v5a (LGB 5d-label + stop):     {len(df_v5a)} 周期, avg_picks={df_v5a['n_picks'].mean():.1f}")
    print(f"  v5a-nostop:                    {len(df_v5a_nostop)} 周期")
    print(f"  v5c (LGB × v4 ensemble+stop):  {len(df_v5c)} 周期, avg_picks={df_v5c['n_picks'].mean():.1f}")

    periods_per_year = TRADING_DAYS_PER_YEAR / HOLD_DAYS
    rows = []
    rows.append(stats(df_v5a["gross_ret"], periods_per_year, "v5a LGB-5dlabel+stop (gross)"))
    rows.append(stats(df_v5a["net_ret"], periods_per_year, "v5a LGB-5dlabel+stop (net)"))
    rows.append(stats(df_v5a_nostop["gross_ret"], periods_per_year, "v5a-nostop (gross)"))
    rows.append(stats(df_v5a_nostop["net_ret"], periods_per_year, "v5a-nostop (net)"))
    rows.append(stats(df_v5c["gross_ret"], periods_per_year, "v5c ensemble+stop (gross)"))
    rows.append(stats(df_v5c["net_ret"], periods_per_year, "v5c ensemble+stop (net)"))

    summary = pd.DataFrame(rows).set_index("label")
    print(f"\n=== v5 optimization 业绩 ===")
    print(summary.to_string())

    print(f"\n=== 参考 ===")
    print(f"  v4-5d/Top5 net (10月):   +61.19%, ann +82.50%, Sharpe 1.58, MDD -19%, win 55%, 40 周期")
    print(f"  v5-clean-daily net (5月): +18.86%, ann +53.89%, Sharpe 1.04, MDD -24%, win 51%, 101 周期")
    print(f"  v5-clean-5d net (5月):   -7.86% (失败: 1d signal + 5d hold)")

    df_v5a.to_csv(OUT_DIR / "strategy_v5a_equity.csv")
    df_v5c.to_csv(OUT_DIR / "strategy_v5c_equity.csv")
    md = ["# v5 optimization", ""]
    md.append("## 修复点")
    md.append("- v5a 修复: LGB 训练 5d label, 信号 horizon 跟 5d 持仓对齐")
    md.append("- v5c 集成: LGB_prob(5d) × v4 动量分数")
    md.append("- 加 -5% 止损 (跟 v4-5d/Top5 对齐)")
    md.append("")
    md.append("## 业绩")
    md.append("")
    md.append(summary.to_markdown())
    md.append("")
    md.append("## 参考基准")
    md.append("- v4-5d/Top5 net (10月): total +61.19%, ann +82.50%, Sharpe 1.58, MDD -19%, win 55%")
    md.append("- v5-clean-daily net (5月): total +18.86%, ann +53.89%, Sharpe 1.04")
    md.append("- v5-clean-5d net (5月): total -7.86% (失败)")
    (OUT_DIR / "strategy_v5_opt_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出: strategy_v5_opt_report.md  strategy_v5{{a,c}}_equity.csv")


if __name__ == "__main__":
    main()
