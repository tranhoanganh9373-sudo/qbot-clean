"""v5 LightGBM 长历史 (3 年) 对比 — 在 long_history.parquet 上跑.

Runs:
  v5-daily LGB:        1d label, daily rebal, 24h hold
  v5-5d (1d label):    1d label, 5d hold (alpha decay test)
  v5a 5d-label:        5d label, 5d rebal, 5d hold (horizon aligned)

Run: python examples/strategy_v5_long_history.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "long_history.parquet"
OUT_DIR = Path(__file__).resolve().parent

PICKS = 5
COST_RATE = 0.0025
MIN_MARKET_CAP = 10
MIN_TURNOVER = 0.005
TRADING_DAYS_PER_YEAR = 252
WARMUP = 60
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


def build_features(market, horizon):
    parts = []
    base_cols = ["ret_1d","ret_3d","ret_5d","ret_10d","ret_20d","intraday",
                 "ma5_dist","ma20_dist","ma60_dist","ma5_ma10",
                 "vol10","vol20","vol_ratio","rsi","bb_pos","macd",
                 "turnover","mkt_cap","target_bin"]
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
        f["target_bin"] = ((c.shift(-horizon) / c - 1) > 0).astype(int)
        f["code"] = code
        f["board"] = get_board(code)
        parts.append(f)
    return pd.concat(parts).dropna(subset=base_cols)


def board_rotation_matrix(features):
    daily_board = features.groupby([features.index, "board"])["ret_1d"].mean()
    pivoted = daily_board.unstack("board")
    return pivoted.rolling(5).mean().shift(1)


def run_picker(test, *, rank_col, open_df, low_df, board_5d_avg, all_dates,
               hold_days, picks=PICKS, cost_rate=COST_RATE, use_stop=True):
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
        rets = []
        if use_stop and hold_days > 1:
            hold_lows = low_df.loc[buy_date:sell_date, codes_v].iloc[1:]
        for c in codes_v:
            ep = buy_p[c]
            stop_price = ep * (1 + STOP_LOSS)
            if use_stop and hold_days > 1:
                if (hold_lows[c] < stop_price).any():
                    rets.append(STOP_LOSS)
                    continue
            rets.append(sell_p[c] / ep - 1)
        gross = float(np.mean(rets)) if rets else 0.0
        daily_rows.append({"date": pick_day, "n_picks": len(rets),
                            "gross_ret": gross, "net_ret": gross - cost_rate})
    return pd.DataFrame(daily_rows).set_index("date")


def stats(returns, periods_per_year, label):
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
    return {"label": label, "n": n, "total_%": round(tot*100,2), "ann_%": round(ann*100,2),
            "vol_%": round(vol*100,2), "sharpe": round(sharpe,2),
            "mdd_%": round(mdd*100,2), "win_%": round(win*100,2)}


def main() -> None:
    if not DATA_PATH.exists():
        print(f"❌ {DATA_PATH} 不存在; 先 fetch")
        return

    print(f"[1/4] loading {DATA_PATH.name}...")
    t = time.time()
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code","date"]).reset_index(drop=True).set_index("date")
    market = {code: g for code, g in df.groupby("code") if len(g) >= WARMUP + 20}
    print(f"  ok ({time.time()-t:.1f}s) — {len(market)} stocks  "
          f"{df.index.min().date()} → {df.index.max().date()}")

    print(f"\n[2/4] 训练 1d-label LGB + 5d-label LGB ...")
    t = time.time()
    feats_1d = build_features(market, horizon=1)
    feats_5d = build_features(market, horizon=5)
    fcols = [c for c in feats_1d.columns if c not in ("code","target_bin","board")]
    print(f"  features: {len(feats_1d):,} rows, {len(fcols)} cols  ({time.time()-t:.1f}s)")

    all_dates = sorted(feats_1d.index.unique())
    split_idx = len(all_dates) // 2
    train_end = all_dates[split_idx - 1]
    test_start = all_dates[split_idx]
    print(f"  train: → {train_end.date()}, test: {test_start.date()} → {all_dates[-1].date()}")

    open_df = pd.concat({c: g["open"] for c, g in market.items()}, axis=1)
    low_df = pd.concat({c: g["low"] for c, g in market.items()}, axis=1)

    t = time.time()
    train_1d = feats_1d[feats_1d.index <= train_end]
    test_1d = feats_1d[feats_1d.index >= test_start].copy()
    board_5d_avg_1d = board_rotation_matrix(feats_1d)
    m1 = lgb.LGBMClassifier(**LGB_PARAMS).fit(train_1d[fcols].values, train_1d["target_bin"].values)
    test_1d["prob"] = m1.predict_proba(test_1d[fcols].values)[:, 1]
    print(f"  1d LGB trained ({time.time()-t:.1f}s)")

    t = time.time()
    train_5d = feats_5d[feats_5d.index <= train_end]
    test_5d = feats_5d[feats_5d.index >= test_start].copy()
    board_5d_avg_5d = board_rotation_matrix(feats_5d)
    m5 = lgb.LGBMClassifier(**LGB_PARAMS).fit(train_5d[fcols].values, train_5d["target_bin"].values)
    test_5d["prob"] = m5.predict_proba(test_5d[fcols].values)[:, 1]
    print(f"  5d LGB trained ({time.time()-t:.1f}s)")

    print(f"\n[3/4] 跑 3 个 v5 变体 ...")
    df_v5_daily = run_picker(test_1d, rank_col="prob", open_df=open_df, low_df=low_df,
                             board_5d_avg=board_5d_avg_1d, all_dates=all_dates,
                             hold_days=1, use_stop=False)
    df_v5_5d_1dlab = run_picker(test_1d, rank_col="prob", open_df=open_df, low_df=low_df,
                                 board_5d_avg=board_5d_avg_1d, all_dates=all_dates,
                                 hold_days=5, use_stop=True)
    df_v5a = run_picker(test_5d, rank_col="prob", open_df=open_df, low_df=low_df,
                        board_5d_avg=board_5d_avg_5d, all_dates=all_dates,
                        hold_days=5, use_stop=True)

    print(f"  v5-daily (1d label, 24h hold):    {len(df_v5_daily)} 周期")
    print(f"  v5-5d (1d label, 5d hold):        {len(df_v5_5d_1dlab)} 周期")
    print(f"  v5a (5d label, 5d hold):          {len(df_v5a)} 周期")

    print(f"\n[4/4] 统计 ...")
    rows = []
    rows.append(stats(df_v5_daily["gross_ret"], TRADING_DAYS_PER_YEAR, "v5-daily (gross)"))
    rows.append(stats(df_v5_daily["net_ret"], TRADING_DAYS_PER_YEAR, "v5-daily (net)"))
    rows.append(stats(df_v5_5d_1dlab["gross_ret"], TRADING_DAYS_PER_YEAR/5, "v5-5d (1d label, gross)"))
    rows.append(stats(df_v5_5d_1dlab["net_ret"], TRADING_DAYS_PER_YEAR/5, "v5-5d (1d label, net)"))
    rows.append(stats(df_v5a["gross_ret"], TRADING_DAYS_PER_YEAR/5, "v5a (5d label, gross)"))
    rows.append(stats(df_v5a["net_ret"], TRADING_DAYS_PER_YEAR/5, "v5a (5d label, net)"))

    summary = pd.DataFrame(rows).set_index("label")
    print(f"\n=== v5 在 3 年长历史上的业绩 ===")
    print(summary.to_string())

    print(f"\n=== 参考 v4 同期 (3 年, 1757 stocks) ===")
    print(f"  v4-5d/Top5 net:    -33.19%, Sharpe -0.50, MDD -44%, win 44.6%")
    print(f"  v4-7d/Top3 net:    +67.39%, Sharpe +0.52, MDD -30%, win 41.4% (best in grid)")
    print(f"  v4-3d/Top3 net:    +61.64%, Sharpe +0.46, MDD -49%, win 45.0%")

    df_v5_daily.to_csv(OUT_DIR / "long_history_v5_daily_equity.csv")
    df_v5a.to_csv(OUT_DIR / "long_history_v5a_equity.csv")
    md = ["# v5 LightGBM 在 3 年长历史 (1,757 stocks) 上的业绩", ""]
    md.append(f"**回测期间**: {test_start.date()} → {all_dates[-1].date()}")
    md.append(f"**训练**: → {train_end.date()} (50/50 时序切分)")
    md.append(f"**成本**: {COST_RATE*100}% per round-trip")
    md.append("")
    md.append("## 业绩")
    md.append("")
    md.append(summary.to_markdown())
    md.append("")
    md.append("## v4 同期参考")
    md.append("- v4-5d/Top5 net: -33.19%, Sharpe -0.50, MDD -44%, win 44.6%")
    md.append("- v4-7d/Top3 net: +67.39%, Sharpe +0.52, MDD -30%, win 41.4%")
    md.append("- v4-3d/Top3 net: +61.64%, Sharpe +0.46, MDD -49%, win 45.0%")
    (OUT_DIR / "long_history_v5_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出: long_history_v5_{{daily,a}}_equity.csv  long_history_v5_report.md")


if __name__ == "__main__":
    main()
