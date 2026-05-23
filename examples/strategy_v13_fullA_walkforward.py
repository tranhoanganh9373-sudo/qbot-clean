"""v13 K=8 drop=2 在 Baidu 全A 数据上的 walk-forward 验证.

跟 v13 K sweep / v10 实质同一个引擎, 但:
  - data_cache/qlib_baidu (4942 只全 A, 2014-2026)
  - instruments="all"
  - K=8 drop=2 单配置
  - 含涨跌停过滤 (跟 paper_trade_today.py 一致)
  - 2022-01 → 2026-04 (52 月 OOS)
  - 真实摩擦 (impact + commission + slippage + 信号 T+1 延)
  - 5万本金

输出:
  examples/v13_fullA_stats.csv     - 每月明细
  examples/v13_fullA_report.md     - 汇总报告

Run:  python examples/strategy_v13_fullA_walkforward.py
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from dateutil.relativedelta import relativedelta
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.data import D
from qlib.data.dataset import DatasetH

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = str(ROOT / "data_cache" / "qlib_baidu")
OUT_DIR = Path(__file__).resolve().parent

MARKET = "all"
TRAIN_MONTHS = 12
PORTFOLIO_VALUE = 5e4

K = 8
N_DROP = 2

IMPACT_COEF = 0.5
MAX_POSITION_PCT_OF_VOL = 0.05
PRICE_LIMIT_UP = 0.090
SIGNAL_DELAY_FACTOR = 0.5
MIN_LOT = 100

CANDIDATE_POOL_MULTIPLIER = 4
LIMIT_UP_THRESHOLD = 0.095
LIMIT_DOWN_THRESHOLD = -0.095
LIMIT_UP_THRESHOLD_HIGH = 0.195
LIMIT_DOWN_THRESHOLD_HIGH = -0.195


def is_limit_up(sym, chg):
    thresh = LIMIT_UP_THRESHOLD_HIGH if sym.startswith(("SH688", "SZ300")) else LIMIT_UP_THRESHOLD
    return chg >= thresh


def is_limit_down(sym, chg):
    thresh = LIMIT_DOWN_THRESHOLD_HIGH if sym.startswith(("SH688", "SZ300")) else LIMIT_DOWN_THRESHOLD
    return chg <= thresh


LGB_PARAMS = dict(
    loss="mse", colsample_bytree=0.8879, learning_rate=0.0421,
    subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768,
    max_depth=8, num_leaves=210, num_threads=1,
)


def month_start(d):
    return d.strftime("%Y-%m-01")


def month_end(d):
    nm = (d.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)
    return nm.strftime("%Y-%m-%d")


def get_price_data(start, end):
    df = D.features(
        D.instruments(market=MARKET),
        ["$open", "$close", "$volume"],
        start_time=start, end_time=end, freq="day",
    ).reset_index()
    df.columns = ["instrument", "date", "open", "close", "volume"]
    return df


def realistic_window(test_month_start):
    test_start = month_start(test_month_start)
    test_end = month_end(test_month_start)
    train_start = month_start(test_month_start - relativedelta(months=TRAIN_MONTHS))
    valid_start = month_start(test_month_start - relativedelta(months=1))
    train_end = month_end(test_month_start - relativedelta(months=2))
    valid_end = month_end(test_month_start - relativedelta(months=1))

    handler = Alpha158(
        start_time=train_start, end_time=test_end,
        fit_start_time=train_start, fit_end_time=train_end,
        instruments=MARKET,
    )
    dataset = DatasetH(handler=handler, segments={
        "train": (train_start, train_end),
        "valid": (valid_start, valid_end),
        "test": (test_start, test_end),
    })
    model = LGBModel(**LGB_PARAMS)
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")

    price_end = (pd.to_datetime(test_end) + timedelta(days=10)).strftime("%Y-%m-%d")
    price_df = get_price_data(test_start, price_end)
    open_pv = price_df.pivot(index="date", columns="instrument", values="open")
    close_pv = price_df.pivot(index="date", columns="instrument", values="close")
    vol_pv = price_df.pivot(index="date", columns="instrument", values="volume")

    pred_unstacked = pred.unstack(level="instrument")
    test_dates = sorted(pred_unstacked.index)
    if len(test_dates) < 2:
        return {"abs_ret_%": 0, "avg_picks": 0, "n_days": 0, "n_skipped_limit": 0}

    current_holdings = {}
    cash = PORTFOLIO_VALUE
    daily_ret = []
    n_picks_realized = []
    last_known_price = {}
    n_skipped_limit = 0

    def mark_to_market(td_):
        pv = cash
        for c, sh in current_holdings.items():
            p_use = None
            if c in close_pv.columns and td_ in close_pv.index:
                p_candidate = close_pv.loc[td_, c]
                if pd.notna(p_candidate) and p_candidate > 0:
                    if c in last_known_price and last_known_price[c] > 0:
                        chg = abs(p_candidate / last_known_price[c] - 1)
                        if chg <= 0.15:
                            p_use = p_candidate
                            last_known_price[c] = p_candidate
                        else:
                            p_use = last_known_price[c]
                    else:
                        p_use = p_candidate
                        last_known_price[c] = p_candidate
            if p_use is None:
                p_use = last_known_price.get(c, 0)
            pv += sh * p_use
        return pv

    for di, td in enumerate(test_dates):
        if di + 1 >= len(test_dates):
            break
        next_td = test_dates[di + 1]
        if next_td not in open_pv.index:
            continue

        scores_all = pred_unstacked.loc[td].dropna().sort_values(ascending=False)
        pool = scores_all.head(K * CANDIDATE_POOL_MULTIPLIER)

        filtered = []
        for sym, score in pool.items():
            if sym not in close_pv.columns or di == 0:
                filtered.append((sym, score))
                if len(filtered) >= K:
                    break
                continue
            prev_td = test_dates[di - 1] if di > 0 else None
            curr_p = close_pv.loc[td, sym] if td in close_pv.index else None
            prev_p = close_pv.loc[prev_td, sym] if prev_td and prev_td in close_pv.index else None
            chg = (curr_p / prev_p - 1) if (pd.notna(curr_p) and pd.notna(prev_p)
                                              and prev_p > 0) else 0
            if is_limit_up(sym, chg) or is_limit_down(sym, chg):
                n_skipped_limit += 1
                continue
            filtered.append((sym, score))
            if len(filtered) >= K:
                break

        target_topk = [s for s, _ in filtered]
        port_val = mark_to_market(td)

        scores_for_drop = pred_unstacked.loc[td]
        to_drop_candidates = sorted(
            [c for c in current_holdings if c not in target_topk],
            key=lambda c: scores_for_drop.get(c, -np.inf),
        )
        to_drop = to_drop_candidates[:N_DROP]
        buy_candidates = [c for c in target_topk if c not in current_holdings][:N_DROP]

        for c in to_drop:
            shares = current_holdings[c]
            if c not in open_pv.columns or next_td not in open_pv.index:
                continue
            no_p = open_pv.loc[next_td, c]
            nc_p = close_pv.loc[next_td, c] if next_td in close_pv.index else no_p
            if pd.isna(no_p) or pd.isna(nc_p):
                continue
            exec_p = no_p * (1 - SIGNAL_DELAY_FACTOR) + nc_p * SIGNAL_DELAY_FACTOR
            if exec_p <= 0:
                continue
            order_amount = shares * exec_p
            daily_amount = (vol_pv.loc[next_td, c] * exec_p
                            if next_td in vol_pv.index and pd.notna(vol_pv.loc[next_td, c])
                            else order_amount * 100)
            impact = IMPACT_COEF * np.sqrt(min(1.0, order_amount / max(daily_amount, 1e3))) * 0.01
            cash += shares * exec_p * (1 - impact - 0.0005)
            del current_holdings[c]

        cash_per_pick = cash / max(len(buy_candidates), 1)
        for c in buy_candidates:
            if c not in open_pv.columns or next_td not in open_pv.index:
                continue
            prev_close = close_pv.loc[td, c] if td in close_pv.index else None
            next_open = open_pv.loc[next_td, c]
            if pd.notna(prev_close) and pd.notna(next_open):
                chg = next_open / prev_close - 1
                if chg >= PRICE_LIMIT_UP:
                    continue
            nc_p = close_pv.loc[next_td, c] if next_td in close_pv.index else next_open
            if pd.isna(next_open) or pd.isna(nc_p):
                continue
            exec_p = next_open * (1 - SIGNAL_DELAY_FACTOR) + nc_p * SIGNAL_DELAY_FACTOR
            if exec_p <= 0:
                continue
            daily_amount = (vol_pv.loc[next_td, c] * exec_p
                            if next_td in vol_pv.index and pd.notna(vol_pv.loc[next_td, c])
                            else 1e9)
            max_amount = daily_amount * MAX_POSITION_PCT_OF_VOL
            target_amount = min(cash_per_pick, max_amount)
            if target_amount < exec_p * MIN_LOT:
                continue
            shares = (target_amount // (exec_p * MIN_LOT)) * MIN_LOT
            order_amount = shares * exec_p
            impact = IMPACT_COEF * np.sqrt(min(1.0, order_amount / max(daily_amount, 1e3))) * 0.01
            cash -= shares * exec_p * (1 + impact + 0.0003)
            current_holdings[c] = shares
        n_picks_realized.append(len(current_holdings))

        new_port_val = mark_to_market(next_td)
        if port_val > 0:
            daily_ret.append(new_port_val / port_val - 1)

    if not daily_ret:
        return {"abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                "n_skipped_limit": n_skipped_limit}
    abs_ret = (1 + pd.Series(daily_ret)).prod() - 1
    return {
        "abs_ret_%": round(abs_ret * 100, 2),
        "avg_picks": round(np.mean(n_picks_realized), 1) if n_picks_realized else 0,
        "n_days": len(daily_ret),
        "n_skipped_limit": n_skipped_limit,
    }


def annualize_metrics(returns, n_periods_per_year=12):
    cum = (1 + returns / 100).prod() - 1
    n = len(returns)
    years = n / n_periods_per_year
    ann_ret = (1 + cum) ** (1 / years) - 1 if years > 0 else 0
    mean = (returns / 100).mean()
    std = (returns / 100).std()
    sharpe = mean / std * np.sqrt(n_periods_per_year) if std > 0 else 0
    cum_series = (1 + returns / 100).cumprod()
    peak = cum_series.cummax()
    mdd = ((cum_series - peak) / peak).min()
    return {
        "cum_%": round(cum * 100, 2),
        "ann_%": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 2),
        "mdd_%": round(mdd * 100, 2),
        "win_%": round((returns > 0).sum() / len(returns) * 100, 2),
        "n": n,
    }


def main():
    qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
    print(f"[1/3] qlib initialized — {QLIB_DIR}")
    print(f"     全 A 股 universe, K={K} drop={N_DROP}, 5万本金, 真实摩擦 + 涨跌停过滤")

    first_test = datetime(2022, 1, 1)
    last_test = datetime(2026, 4, 1)
    months = []
    cur = first_test
    while cur <= last_test:
        months.append(cur)
        cur += relativedelta(months=1)
    print(f"[2/3] {len(months)} 月 walk-forward, 每月独立 train+test")

    all_rows = []
    for i, m in enumerate(months, 1):
        try:
            res = realistic_window(m)
            res["month"] = m.strftime("%Y-%m")
            all_rows.append(res)
            print(f"  {i:2d}/{len(months)} {res['month']}: abs_ret={res['abs_ret_%']:+6.2f}%  "
                  f"picks={res['avg_picks']:.1f}  skip_limit={res['n_skipped_limit']}", flush=True)
        except Exception as e:
            print(f"  {i:2d}/{len(months)} {m.strftime('%Y-%m')} FAIL: {str(e)[:80]}")
            all_rows.append({"month": m.strftime("%Y-%m"), "abs_ret_%": 0,
                              "avg_picks": 0, "n_days": 0, "n_skipped_limit": 0})

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "v13_fullA_stats.csv", index=False)

    print("\n[3/3] === 汇总 ===\n")
    m = annualize_metrics(df["abs_ret_%"])
    m["avg_picks"] = round(df["avg_picks"].mean(), 1)
    m["total_skip_limit"] = int(df["n_skipped_limit"].sum())
    print(pd.Series(m).to_string())

    md = [
        f"# v13 K={K} drop={N_DROP} 在 Baidu 全A数据上的 walk-forward",
        "",
        "**Universe**: 全A股 4942 只 (剔除 ST)",
        f"**Period**: {months[0].strftime('%Y-%m')} → {months[-1].strftime('%Y-%m')} "
        f"({len(months)} 月)",
        f"**Capital**: {PORTFOLIO_VALUE:.0f} 元",
        "**Friction**: 0.25% 单次往返 + Almgren-Chriss 冲击 + 信号 T+1 + ±10% 涨停过滤",
        "**Filter**: 候选池扩 4×, 过滤涨停/跌停 (主板±10%, 科创/创业±20%)",
        "",
        "## 业绩汇总",
        "",
        pd.Series(m).to_markdown(),
        "",
        "## 跟 v13 CSI300 基线对比",
        "",
        "| 项 | CSI300 (v13 backtest) | 全A (本次) |",
        "|----|---------------------:|----------:|",
        f"| 月数 | 44 | {len(months)} |",
        f"| cum %  | +207.22 | {m['cum_%']:+.2f} |",
        f"| ann %  | +35.81  | {m['ann_%']:+.2f} |",
        f"| sharpe | 1.38    | {m['sharpe']:.2f} |",
        f"| mdd %  | -16.16  | {m['mdd_%']:.2f} |",
        f"| win %  | 65.9    | {m['win_%']:.2f} |",
    ]
    (OUT_DIR / "v13_fullA_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出: v13_fullA_{{stats.csv, report.md}}")


if __name__ == "__main__":
    main()
