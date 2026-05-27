"""推荐策略 (生产可交易版).

= v4-7d/Top3 价量动量 + 3 项保守增强:
  1. WARMUP = 252 (严格排除上市 < 1 年的次新股)
  2. ST 启发式过滤: 历史 60 天 max|chg| > 5.2% 才算非 ST
  3. PE/PB 大盘风控:
     - PE > 34 OR PB > 3.0  -> 空仓
     - PE > 30 OR PB > 2.6  -> 轻仓 (Top 2, 60% 仓位)
     - 否则正常 (Top 3, 100% 仓位)
  4. (故意不加板块轮换 - 经 v6 验证有害)
  5. (故意不加 LightGBM ensemble - 经 v7 验证有害)
  6. 流通市值 >= 10 亿, 换手率 >= 0.5%

执行规则:
  - 每周一收盘后跑该脚本选 Top 3
  - 周二开盘等权买入 (T+1 合规)
  - 持 7 个交易日, 期间任意 bar low < entry * 0.95 -> -5% 止损
  - 第二个周三开盘卖出剩余持仓

历史业绩 (3 年长样本, 1,757 stocks):
  - 总收益 +63% / 年化 +19% / Sharpe 0.52 / MDD -30%

警告: 普通策略, 真实可执行 Sharpe 期望 0.5 ± 0.3 跨市场风格.

Run:  python examples/strategy_recommended.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

from claude_finance.scan_cache import cache_or_fetch

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "long_history.parquet"
OUT_DIR = Path(__file__).resolve().parent

PICKS = 3
LIGHT_PICKS = 2
HOLD_DAYS = 7
COST_RATE = 0.0025
TRADING_DAYS_PER_YEAR = 252
STOP_LOSS = -0.05
MOM_VOL_SURGE = 1.3
MIN_TURNOVER = 0.005
MIN_MARKET_CAP = 10
WARMUP = 252
PE_EMPTY = 34.0
PE_LIGHT = 30.0
PB_EMPTY = 3.0
PB_LIGHT = 2.6
ST_HEURISTIC_PCT = 5.2


def get_board(code: str) -> str:
    if code.startswith("sh60"): return "沪主板"
    if code.startswith("sh68"): return "科创板"
    if code.startswith("sz00"): return "深主板"
    if code.startswith("sz30"): return "创业板"
    if code.startswith("bj"): return "北交所"
    return "其他"


def build_features(market):
    parts = []
    for code, grp in market.items():
        c = grp["close"]; o = grp["open"]; v = grp["volume"]
        h = grp["high"]; low = grp["low"]
        os_ = grp["outstanding_share"]
        if len(c) < WARMUP: continue
        f = pd.DataFrame(index=grp.index)
        chg = c.pct_change() * 100
        avg_vol5 = v.shift(1).rolling(5).mean()
        sma20 = c.shift(1).rolling(20).mean()
        pos = (c - low) / (h - low).replace(0, np.nan)
        vol_ratio = v / avg_vol5
        f["v4_score"] = (vol_ratio * chg * pos).where(
            (c > o)
            & chg.between(0.01, 9.0)
            & vol_ratio.gt(MOM_VOL_SURGE)
            & c.gt(sma20)
            & pos.between(0.5, 0.95)
            & ((v / os_) > MIN_TURNOVER)
        )
        f["turnover"] = v / os_
        f["mkt_cap"] = os_ * c / 1e8
        f["max_chg_60d"] = chg.abs().rolling(60).max()
        f["code"] = code
        f["board"] = get_board(code)
        parts.append(f)
    return pd.concat(parts).dropna(subset=["max_chg_60d"])


def fetch_pe_pb_aligned(dates: pd.DatetimeIndex):
    """Cache PE/PB raw 到 ``data_cache/scan_cache/ak_stock_market_{pe,pb}_lg.parquet``,
    TTL 24h. 大盘 PE/PB 每日更新一次, 1 天足够.
    """
    pe_df = cache_or_fetch(
        key="ak_stock_market_pe_lg",
        fetcher=lambda: ak.stock_market_pe_lg(),
        ttl_hours=24.0,
    )
    pb_df = cache_or_fetch(
        key="ak_stock_market_pb_lg",
        fetcher=lambda: ak.stock_market_pb_lg(),
        ttl_hours=24.0,
    )
    pe_df = pe_df.copy()
    pb_df = pb_df.copy()
    pe_df["日期"] = pd.to_datetime(pe_df["日期"])
    pb_df["日期"] = pd.to_datetime(pb_df["日期"])
    pe = pe_df.set_index("日期")["平均市盈率"].sort_index()
    pb = pb_df.set_index("日期")["市净率中位数"].sort_index()
    return pe.reindex(dates, method="ffill"), pb.reindex(dates, method="ffill")


def regime(pe, pb):
    if (pe is not None and pe > PE_EMPTY) or (pb is not None and pb > PB_EMPTY):
        return 0, 0.0
    if (pe is not None and pe > PE_LIGHT) or (pb is not None and pb > PB_LIGHT):
        return LIGHT_PICKS, 0.6
    return PICKS, 1.0


def run(features, *, open_df, low_df, all_dates, pe_series, pb_series):
    pick_indices = list(range(0, len(all_dates), HOLD_DAYS))
    rows = []
    for pi in pick_indices:
        pick_day = all_dates[pi]
        if pi + 1 + HOLD_DAYS >= len(all_dates):
            break
        buy_date = all_dates[pi + 1]
        sell_date = all_dates[pi + 1 + HOLD_DAYS]

        pe_val = pe_series.get(pick_day) if pick_day in pe_series.index else None
        pb_val = pb_series.get(pick_day) if pick_day in pb_series.index else None
        n_picks, budget = regime(pe_val, pb_val)
        if n_picks == 0:
            rows.append({"date": pick_day, "n_picks": 0, "gross_ret": 0.0,
                         "net_ret": 0.0, "regime": "empty",
                         "pe": pe_val, "pb": pb_val})
            continue

        day_data = features[features.index == pick_day].copy()
        if day_data.empty:
            rows.append({"date": pick_day, "n_picks": 0, "gross_ret": 0.0,
                         "net_ret": 0.0, "regime": "no_data",
                         "pe": pe_val, "pb": pb_val})
            continue

        day_data = day_data[(day_data["mkt_cap"] >= MIN_MARKET_CAP) &
                            (day_data["turnover"] >= MIN_TURNOVER)]
        day_data = day_data[~day_data["code"].str.contains("900|200", na=False)]
        day_data = day_data[day_data["max_chg_60d"] > ST_HEURISTIC_PCT]
        day_data = day_data.dropna(subset=["v4_score"])
        if len(day_data) < n_picks:
            rows.append({"date": pick_day, "n_picks": 0, "gross_ret": 0.0,
                         "net_ret": 0.0, "regime": "insufficient",
                         "pe": pe_val, "pb": pb_val})
            continue

        top = day_data.nlargest(n_picks, "v4_score")
        codes = top["code"].tolist()
        try:
            buy_p = open_df.loc[buy_date, codes]
            sell_p = open_df.loc[sell_date, codes]
        except KeyError:
            rows.append({"date": pick_day, "n_picks": 0, "gross_ret": 0.0,
                         "net_ret": 0.0, "regime": "no_price",
                         "pe": pe_val, "pb": pb_val})
            continue
        valid = buy_p.notna() & sell_p.notna() & (buy_p > 0)
        if not valid.any():
            rows.append({"date": pick_day, "n_picks": 0, "gross_ret": 0.0,
                         "net_ret": 0.0, "regime": "no_valid",
                         "pe": pe_val, "pb": pb_val})
            continue

        codes_v = [c for c, ok in zip(codes, valid) if ok]
        hold_lows = low_df.loc[buy_date:sell_date, codes_v].iloc[1:]
        rets = []
        for c in codes_v:
            ep = buy_p[c]
            stop_price = ep * (1 + STOP_LOSS)
            if (hold_lows[c] < stop_price).any():
                rets.append(STOP_LOSS)
                continue
            rets.append(sell_p[c] / ep - 1)
        gross_pick = float(np.mean(rets)) if rets else 0.0
        gross = gross_pick * budget
        cost = COST_RATE * budget if rets else 0.0
        rg = "normal" if budget >= 0.99 else "light"
        rows.append({"date": pick_day, "n_picks": len(rets), "gross_ret": gross,
                     "net_ret": gross - cost, "regime": rg,
                     "pe": pe_val, "pb": pb_val,
                     "picks": ",".join(codes_v)})
    return pd.DataFrame(rows).set_index("date")


def stats(returns, periods_per_year):
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
        "n_periods": n,
        "total_%": round(tot * 100, 2),
        "ann_%": round(ann * 100, 2),
        "vol_%": round(vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "mdd_%": round(mdd * 100, 2),
        "win_%": round(win * 100, 2),
    }


def main() -> None:
    if not DATA_PATH.exists():
        print(f"❌ {DATA_PATH} 不存在; 先跑 fetch_long_history.py")
        return

    print(f"[1/3] loading + features ...")
    t = time.time()
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code","date"]).reset_index(drop=True).set_index("date")
    market = {code: g for code, g in df.groupby("code") if len(g) >= WARMUP + 20}
    print(f"  ok ({time.time()-t:.1f}s) — {len(market)} stocks >= 1 年历史")
    features = build_features(market)
    print(f"  features: {len(features):,} rows")

    open_df = pd.concat({c: g["open"] for c, g in market.items()}, axis=1)
    low_df = pd.concat({c: g["low"] for c, g in market.items()}, axis=1)

    print(f"\n[2/3] fetch PE/PB ...")
    t = time.time()
    all_dates = sorted(features.index.unique())
    pe_series, pb_series = fetch_pe_pb_aligned(pd.DatetimeIndex(all_dates))
    print(f"  PE range: {pe_series.min():.1f} ~ {pe_series.max():.1f} (now {pe_series.iloc[-1]:.1f})")
    print(f"  PB range: {pb_series.min():.2f} ~ {pb_series.max():.2f} (now {pb_series.iloc[-1]:.2f})")

    print(f"\n[3/3] backtest 全期 ...")
    t = time.time()
    df_res = run(features, open_df=open_df, low_df=low_df,
                 all_dates=all_dates, pe_series=pe_series, pb_series=pb_series)
    print(f"  done ({time.time()-t:.1f}s) — {len(df_res)} 周期")
    n_active = (df_res["n_picks"] > 0).sum()
    n_light = (df_res["regime"] == "light").sum()
    n_empty = (df_res["regime"] == "empty").sum()
    print(f"  持仓 {n_active} (轻仓 {n_light}) / 空仓 {n_empty}")

    s = stats(df_res["net_ret"], TRADING_DAYS_PER_YEAR / HOLD_DAYS)
    print(f"\n=== 推荐策略业绩 (3 年长样本) ===")
    for k, v in s.items():
        print(f"  {k:>15s}: {v}")

    df_res.to_csv(OUT_DIR / "recommended_equity.csv")
    md = ["# 推荐策略 (生产可交易版)", ""]
    md.append("## 配置")
    md.append("- 基础: v4-7d/Top3 价量动量 (7 日 rebal, 等权)")
    md.append("- 增强: WARMUP=252 + ST 启发式 + PE/PB 风控")
    md.append("- 故意不加: 板块轮换 (经 v6 验证有害), LGB ensemble (经 v7 验证有害)")
    md.append("- 成本: 0.25% 单次往返 (印花 0.05 + 佣金 0.05 + 滑点 0.15)")
    md.append("- 止损: -5% intra-period")
    md.append("")
    md.append("## 3 年长样本业绩")
    for k, v in s.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## 警告")
    md.append("- Sharpe 0.5 是**普通策略**, 不是机构级 alpha")
    md.append("- 真实可执行 Sharpe 期望 0.5 ± 0.3 跨市场风格")
    md.append("- 强烈依赖中国 A 股结构 (T+1, 涨跌停, 散户主导)")
    md.append("- 不构成投资建议; 过往业绩不预示未来")
    (OUT_DIR / "recommended_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n输出: recommended_{{equity.csv, report.md}}")


if __name__ == "__main__":
    main()
