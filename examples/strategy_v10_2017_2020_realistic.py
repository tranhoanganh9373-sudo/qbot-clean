"""v10 = v8 walk-forward + v9 真实摩擦 + 5 万账户 + 2017-2020 跨牛熊数据.

Cross-regime stability test using qlib's bundled CN data (2008-2020):
  - 2017 蓝筹牛市
  - 2018 股灾 (贸易战 + 去杠杆) → CSI300 跌 25%
  - 2019 反弹
  - 2020 疫情冲击 + 复苏

Realism layers (same as v9):
  1. 冲击成本: 0.5 × √(order / daily_amount)
  2. 严格涨停过滤: T+1 open ≥ +9% → 跳过
  3. 信号延迟: 成交价 = (open + close) / 2
  4. 整手约束 (100 股) + 单股仓位 ≤ 当日成交额 5%

Portfolio: 5 万元 (散户级别)
Universe: CSI300 (qlib 自带数据)
Walk-forward: 12-mo train + 1-mo test, monthly roll
Test range: 2017-01 → 2020-08 = 44 个月 OOS

Run: python examples/strategy_v10_2017_2020_realistic.py
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
TOPK = 30
N_DROP = 5
TRAIN_MONTHS = 12
PORTFOLIO_VALUE = 5e4

IMPACT_COEF = 0.5
MAX_POSITION_PCT_OF_VOL = 0.05
PRICE_LIMIT_UP = 0.090
SIGNAL_DELAY_FACTOR = 0.5
MIN_LOT = 100

LGB_PARAMS = dict(
    loss="mse",
    colsample_bytree=0.8879,
    learning_rate=0.0421,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=1,
)


def month_start(d):
    return d.strftime("%Y-%m-01")


def month_end(d):
    nm = (d.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)
    return nm.strftime("%Y-%m-%d")


def regime_label(month):
    if month < datetime(2018, 1, 1): return "2017 蓝筹牛"
    if month < datetime(2019, 1, 1): return "2018 股灾"
    if month < datetime(2020, 1, 1): return "2019 反弹"
    return "2020 疫情+复苏"


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
        return {"n_days": 0, "abs_ret_%": 0, "n_skipped_limit_up": 0,
                "avg_impact_%": 0, "avg_picks_realized": 0, "bench_ret_%": 0,
                "excess_%": 0}

    bench_data = D.features([BENCHMARK], ["$close"], start_time=test_start,
                             end_time=test_end, freq="day")
    bench_close = bench_data.reset_index().set_index("datetime")["$close"]
    bench_ret = bench_close.iloc[-1] / bench_close.iloc[0] - 1 if len(bench_close) > 1 else 0

    current_holdings = {}
    cash = PORTFOLIO_VALUE
    daily_ret = []
    n_skipped_limit_up = 0
    impact_costs = []
    n_picks_realized = []
    last_known_price = {}  # c -> 上次有效 close (用于 停牌 / NaN 处理)
    n_anomaly_skipped = 0  # 单日 chg > ±15% 的异常数据被拒绝

    def mark_to_market(td_):
        """Compute portfolio value at date td_ using last_known_price fallback + sanity check."""
        nonlocal n_anomaly_skipped
        pv = cash
        for c, sh in current_holdings.items():
            p_use = None
            if c in close_pv.columns and td_ in close_pv.index:
                p_candidate = close_pv.loc[td_, c]
                if pd.notna(p_candidate) and p_candidate > 0:
                    # Sanity check: reject single-day move > ±15% (exceeds 涨跌停)
                    if c in last_known_price and last_known_price[c] > 0:
                        chg = abs(p_candidate / last_known_price[c] - 1)
                        if chg > 0.15:
                            n_anomaly_skipped += 1
                            p_use = last_known_price[c]
                        else:
                            p_use = p_candidate
                            last_known_price[c] = p_candidate
                    else:
                        p_use = p_candidate
                        last_known_price[c] = p_candidate
            if p_use is None:
                p_use = last_known_price.get(c, 0)  # 停牌时用上次价
            pv += sh * p_use
        return pv

    for di, td in enumerate(test_dates):
        if di + 1 >= len(test_dates):
            break
        next_td = test_dates[di + 1]
        if next_td not in open_pv.index:
            continue

        scores = pred_unstacked.loc[td].dropna().sort_values(ascending=False)
        target_topk = scores.head(TOPK).index.tolist()

        port_val = mark_to_market(td)

        to_drop_candidates = sorted(
            [c for c in current_holdings if c not in target_topk],
            key=lambda c: scores.get(c, -np.inf),
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
            impact_costs.append(impact)
            cash += shares * exec_p * (1 - impact - 0.0005)
            del current_holdings[c]

        # 注: 对于 5 万账户 + TOPK=30 (每股 1667 元), 多数 A 股一手 > 1667 元 → 跳过
        # 故 cash/N_DROP 反而是 5 万账户的合理预算 (每股 ~1 万, 能买大多数股一手)
        # 实际 picks 自然停在 ~10 而非 30, 这是 5 万本金的物理上限, 非 bug
        cash_per_pick = cash / max(len(buy_candidates), 1)
        for c in buy_candidates:
            if c not in open_pv.columns or next_td not in open_pv.index:
                continue
            prev_close = close_pv.loc[td, c] if td in close_pv.index else None
            next_open = open_pv.loc[next_td, c]
            if pd.notna(prev_close) and pd.notna(next_open):
                chg = next_open / prev_close - 1
                if chg >= PRICE_LIMIT_UP:
                    n_skipped_limit_up += 1
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
            impact_costs.append(impact)
            cash -= shares * exec_p * (1 + impact + 0.0003)
            current_holdings[c] = shares
        n_picks_realized.append(len(current_holdings))

        new_port_val = mark_to_market(next_td)
        if port_val > 0:
            daily_ret.append(new_port_val / port_val - 1)

    if not daily_ret:
        return {"n_days": 0, "abs_ret_%": 0, "n_skipped_limit_up": 0,
                "avg_impact_%": 0, "avg_picks_realized": 0, "bench_ret_%": 0,
                "excess_%": 0, "n_anomaly": 0}

    abs_ret = (1 + pd.Series(daily_ret)).prod() - 1
    avg_impact = np.mean(impact_costs) * 100 if impact_costs else 0
    excess = abs_ret - bench_ret
    return {
        "n_days": len(daily_ret),
        "abs_ret_%": round(abs_ret * 100, 2),
        "bench_ret_%": round(bench_ret * 100, 2),
        "excess_%": round(excess * 100, 2),
        "n_skipped_limit_up": int(n_skipped_limit_up),
        "avg_impact_%": round(avg_impact, 3),
        "avg_picks_realized": round(np.mean(n_picks_realized), 1) if n_picks_realized else 0,
        "n_anomaly": int(n_anomaly_skipped),
    }


def main():
    qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
    print(f"[1/3] qlib initialized — CSI300, 5万账户, 真实摩擦")

    first_test = datetime(2017, 1, 1)
    last_test = datetime(2020, 8, 1)
    months = []
    cur = first_test
    while cur <= last_test:
        months.append(cur)
        cur += relativedelta(months=1)
    print(f"[2/3] {len(months)} 月 OOS: {months[0].strftime('%Y-%m')} → {months[-1].strftime('%Y-%m')}")
    print(f"  跨牛熊: 2017牛 / 2018灾 / 2019反弹 / 2020疫情+复苏")

    rows = []
    for i, m in enumerate(months, 1):
        regime = regime_label(m)
        print(f"  [{i:2d}/{len(months)}] {m.strftime('%Y-%m')} [{regime}] ",
              end="", flush=True)
        try:
            res = realistic_window(m)
            res["month"] = m.strftime("%Y-%m")
            res["regime"] = regime
            rows.append(res)
            print(f"abs={res['abs_ret_%']:+7.2f}% bench={res['bench_ret_%']:+6.2f}% "
                  f"excess={res['excess_%']:+7.2f}% picks={res['avg_picks_realized']:.0f} "
                  f"anom={res.get('n_anomaly', 0)}")
        except Exception as e:
            print(f"FAIL: {str(e)[:60]}")
            rows.append({"month": m.strftime("%Y-%m"), "regime": regime,
                         "n_days": 0, "abs_ret_%": 0, "bench_ret_%": 0,
                         "excess_%": 0, "n_skipped_limit_up": 0,
                         "avg_impact_%": 0, "avg_picks_realized": 0})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "v10_2017_2020_stats.csv", index=False)

    print(f"\n[3/3] === 跨 regime 聚合 ===\n")
    by_regime = df.groupby("regime").agg(
        n_months=("abs_ret_%", "count"),
        avg_abs=("abs_ret_%", "mean"),
        cum_abs=("abs_ret_%", lambda x: ((1 + x / 100).prod() - 1) * 100),
        cum_bench=("bench_ret_%", lambda x: ((1 + x / 100).prod() - 1) * 100),
        cum_excess=("excess_%", lambda x: ((1 + x / 100).prod() - 1) * 100),
        n_pos=("excess_%", lambda x: (x > 0).sum()),
    ).round(2)
    print(by_regime.to_string())

    cum_abs = ((1 + df["abs_ret_%"] / 100).prod() - 1) * 100
    cum_bench = ((1 + df["bench_ret_%"] / 100).prod() - 1) * 100
    cum_excess = ((1 + df["excess_%"] / 100).prod() - 1) * 100
    n_pos = (df["excess_%"] > 0).sum()
    avg_picks = df["avg_picks_realized"].mean()
    avg_impact = df["avg_impact_%"].mean()
    total_skip = df["n_skipped_limit_up"].sum()

    print(f"\n=== 全期 ({len(df)} 月) ===")
    print(f"  累计 abs (5 万账户): {cum_abs:+.2f}%")
    print(f"  累计 benchmark CSI300: {cum_bench:+.2f}%")
    print(f"  累计 excess: {cum_excess:+.2f}%")
    print(f"  正超额月: {n_pos}/{len(df)} ({n_pos/len(df)*100:.1f}%)")
    print(f"  平均持仓数: {avg_picks:.1f}")
    print(f"  平均冲击成本: {avg_impact:.3f}%")
    print(f"  涨停跳过总数: {total_skip}")

    print(f"\n=== 对比基准 ===")
    print(f"  v8 (2023-2026, 无摩擦):           累计 +152%, IR 2.01, 16/20 正")
    print(f"  v9 (2023-2026, 5万摩擦):           累计 +23%")

    md = [f"# v10 跨牛熊真实回测 (5万账户, CSI300, 2017-01→2020-08)", ""]
    md.append("## 关键指标")
    md.append(f"- 累计 abs (5 万账户): **{cum_abs:+.2f}%**")
    md.append(f"- 累计 benchmark CSI300: {cum_bench:+.2f}%")
    md.append(f"- 累计 excess: **{cum_excess:+.2f}%**")
    md.append(f"- 正超额月: {n_pos}/{len(df)}")
    md.append(f"- 平均冲击成本: {avg_impact:.3f}%")
    md.append("")
    md.append("## 跨 regime 分解")
    md.append("")
    md.append(by_regime.to_markdown())
    md.append("")
    md.append("## 月度明细")
    md.append("")
    md.append(df.to_markdown(index=False))
    (OUT_DIR / "v10_2017_2020_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出: v10_2017_2020_{{stats.csv, report.md}}")


if __name__ == "__main__":
    main()
