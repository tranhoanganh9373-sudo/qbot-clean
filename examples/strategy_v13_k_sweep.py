"""v13 = K 参数扫描 找 5万账户最优持仓数.

Tests 6 K values on v10 framework (CSI300 / 2017-2020 / 5万 / 真实摩擦):
  K=3, N_DROP=1   - 高集中
  K=5, N_DROP=1   - 集中
  K=8, N_DROP=2   - 偏集中
  K=10, N_DROP=2  - 平衡 (5万账户最自然)
  K=15, N_DROP=3  - 偏分散
  K=20, N_DROP=4  - 分散

关键优化: predictions 缓存, 每月只训练 LGB 一次, 6 个 K 共享.

Run: python examples/strategy_v13_k_sweep.py
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

OUT_DIR = Path(__file__).resolve().parent
QLIB_DIR = "~/.qlib/qlib_data/cn_data"
MARKET = "csi300"
BENCHMARK = "SH000300"
TRAIN_MONTHS = 12
PORTFOLIO_VALUE = 5e4

IMPACT_COEF = 0.5
MAX_POSITION_PCT_OF_VOL = 0.05
PRICE_LIMIT_UP = 0.090
SIGNAL_DELAY_FACTOR = 0.5
MIN_LOT = 100

K_SWEEP = [(3, 1), (5, 1), (8, 2), (10, 2), (15, 3), (20, 4)]

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


_pred_cache = {}


def get_pred_for_month(test_month_start):
    key = test_month_start.strftime("%Y-%m")
    if key in _pred_cache:
        return _pred_cache[key]

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
    _pred_cache[key] = pred
    return pred


def realistic_window(test_month_start, topk, n_drop):
    test_start = month_start(test_month_start)
    test_end = month_end(test_month_start)

    pred = get_pred_for_month(test_month_start)

    price_end = (pd.to_datetime(test_end) + timedelta(days=10)).strftime("%Y-%m-%d")
    price_df = get_price_data(test_start, price_end)
    open_pv = price_df.pivot(index="date", columns="instrument", values="open")
    close_pv = price_df.pivot(index="date", columns="instrument", values="close")
    vol_pv = price_df.pivot(index="date", columns="instrument", values="volume")

    pred_unstacked = pred.unstack(level="instrument")
    test_dates = sorted(pred_unstacked.index)
    if len(test_dates) < 2:
        return {"abs_ret_%": 0, "avg_picks": 0, "bench_ret_%": 0, "n_days": 0}

    bench_data = D.features([BENCHMARK], ["$close"], start_time=test_start,
                             end_time=test_end, freq="day")
    bench_close = bench_data.reset_index().set_index("datetime")["$close"]
    bench_ret = bench_close.iloc[-1] / bench_close.iloc[0] - 1 if len(bench_close) > 1 else 0

    current_holdings = {}
    cash = PORTFOLIO_VALUE
    daily_ret = []
    n_picks_realized = []
    last_known_price = {}

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

        scores = pred_unstacked.loc[td].dropna().sort_values(ascending=False)
        target_topk = scores.head(topk).index.tolist()
        port_val = mark_to_market(td)

        to_drop_candidates = sorted(
            [c for c in current_holdings if c not in target_topk],
            key=lambda c: scores.get(c, -np.inf),
        )
        to_drop = to_drop_candidates[:n_drop]
        buy_candidates = [c for c in target_topk if c not in current_holdings][:n_drop]

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
        return {"abs_ret_%": 0, "avg_picks": 0,
                "bench_ret_%": round(bench_ret * 100, 2), "n_days": 0}

    abs_ret = (1 + pd.Series(daily_ret)).prod() - 1
    return {
        "abs_ret_%": round(abs_ret * 100, 2),
        "bench_ret_%": round(bench_ret * 100, 2),
        "avg_picks": round(np.mean(n_picks_realized), 1) if n_picks_realized else 0,
        "n_days": len(daily_ret),
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
    print(f"[1/3] qlib initialized — K sweep on CSI300 2017-2020, 5万账户, 真实摩擦")

    first_test = datetime(2017, 1, 1)
    last_test = datetime(2020, 8, 1)
    months = []
    cur = first_test
    while cur <= last_test:
        months.append(cur)
        cur += relativedelta(months=1)
    print(f"[2/3] {len(months)} 月 × {len(K_SWEEP)} K 值 ({len(months) * len(K_SWEEP)} 个回测)")
    print(f"     predictions cached, 每月只训练一次 LGB")

    all_rows = []
    for ki, (k, drop) in enumerate(K_SWEEP, 1):
        print(f"\n  [K={k:2d} drop={drop}]  ({ki}/{len(K_SWEEP)})", flush=True)
        month_results = []
        for i, m in enumerate(months, 1):
            try:
                res = realistic_window(m, topk=k, n_drop=drop)
                res["month"] = m.strftime("%Y-%m")
                res["k"] = k
                month_results.append(res)
                if i % 12 == 0:
                    print(f"    {i}/{len(months)} done")
            except Exception as e:
                print(f"    {m.strftime('%Y-%m')} FAIL: {str(e)[:60]}")
                month_results.append({"month": m.strftime("%Y-%m"), "k": k,
                                       "abs_ret_%": 0, "bench_ret_%": 0,
                                       "avg_picks": 0, "n_days": 0})
        all_rows.extend(month_results)
        df_k = pd.DataFrame(month_results)
        m = annualize_metrics(df_k["abs_ret_%"])
        print(f"    => cum={m['cum_%']:+.2f}%  ann={m['ann_%']:+.2f}%  "
              f"sharpe={m['sharpe']:+.2f}  mdd={m['mdd_%']:.2f}%  win={m['win_%']:.1f}%  "
              f"avg_picks={df_k['avg_picks'].mean():.1f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "v13_k_sweep_stats.csv", index=False)

    print(f"\n[3/3] === K 扫描汇总 ===\n")
    summary = []
    for k, drop in K_SWEEP:
        sub = df[df["k"] == k]
        m = annualize_metrics(sub["abs_ret_%"])
        m["k"] = k
        m["drop"] = drop
        m["avg_picks"] = round(sub["avg_picks"].mean(), 1)
        summary.append(m)

    summary_df = pd.DataFrame(summary).set_index("k")
    print(summary_df.to_string())

    bench_sub = df[df["k"] == K_SWEEP[0][0]]
    bench_m = annualize_metrics(bench_sub["bench_ret_%"])
    print(f"\n  CSI300 benchmark: cum={bench_m['cum_%']:+.2f}% ann={bench_m['ann_%']:+.2f}% "
          f"sharpe={bench_m['sharpe']:+.2f}")

    md = ["# v13 K 参数扫描 (5万账户, CSI300, 2017-2020, 真实摩擦)", ""]
    md.append("## 业绩对照")
    md.append("")
    md.append(summary_df.to_markdown())
    md.append("")
    md.append(f"## CSI300 benchmark: cum {bench_m['cum_%']:+.2f}% / sharpe {bench_m['sharpe']:+.2f}")
    (OUT_DIR / "v13_k_sweep_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出: v13_k_sweep_{{stats.csv, report.md}}")


if __name__ == "__main__":
    main()
