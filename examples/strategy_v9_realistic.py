"""v9 真实摩擦回测 = v8 walk-forward + 现实成本.

Realism layers added on top of v8:
  1. 冲击成本: impact = 0.5 × √(order_amount / daily_amount)
  2. 严格涨停过滤: T+1 open chg ≥ +9% → 跳过 (买不到)
  3. 信号延迟: 用 (open + close) / 2 替代 open 作为实际成交价
  4. 单股仓位 ≤ 当日成交额 5%, 不够整手则跳过

Compared across 4 portfolio sizes:
  - 5万 / 500万 / 5,000万 / 5亿

Run: python examples/strategy_v9_realistic.py
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
QLIB_DIR = ROOT / "data_cache" / "qlib_bin"
OUT_DIR = Path(__file__).resolve().parent
MARKET = "all"
TOPK = 30
N_DROP = 5
TRAIN_MONTHS = 12

PORTFOLIO_SIZES = [
    (5e4, "5万"),
    (5e6, "500万"),
    (5e7, "5,000万"),
    (5e8, "5亿"),
]
IMPACT_COEF = 0.5
MAX_POSITION_PCT_OF_VOL = 0.05
PRICE_LIMIT_UP = 0.090
SIGNAL_DELAY_FACTOR = 0.5

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


def get_price_data(start, end):
    df = D.features(
        D.instruments(market=MARKET),
        ["$open", "$close", "$volume"],
        start_time=start, end_time=end, freq="day",
    ).reset_index()
    df.columns = ["instrument", "date", "open", "close", "volume"]
    return df


def realistic_window(test_month_start, portfolio_value):
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
                "avg_impact_%": 0, "avg_picks_realized": 0}

    current_holdings = {}
    cash = portfolio_value
    daily_ret = []
    n_skipped_limit_up = 0
    impact_costs = []
    n_picks_realized = []

    for di, td in enumerate(test_dates):
        if di + 1 >= len(test_dates):
            break
        next_td = test_dates[di + 1]
        if next_td not in open_pv.index:
            continue

        scores = pred_unstacked.loc[td].dropna().sort_values(ascending=False)
        target_topk = scores.head(TOPK).index.tolist()

        port_val = cash
        for c, sh in current_holdings.items():
            if c in close_pv.columns and td in close_pv.index:
                p = close_pv.loc[td, c]
                if pd.notna(p):
                    port_val += sh * p

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
            if target_amount < exec_p * 100:
                continue
            shares = (target_amount // (exec_p * 100)) * 100
            order_amount = shares * exec_p
            impact = IMPACT_COEF * np.sqrt(min(1.0, order_amount / max(daily_amount, 1e3))) * 0.01
            impact_costs.append(impact)
            cash -= shares * exec_p * (1 + impact + 0.0003)
            current_holdings[c] = shares
        n_picks_realized.append(len(current_holdings))

        new_port_val = cash
        for c, sh in current_holdings.items():
            if c in close_pv.columns and next_td in close_pv.index:
                p = close_pv.loc[next_td, c]
                if pd.notna(p):
                    new_port_val += sh * p
        if port_val > 0:
            daily_ret.append(new_port_val / port_val - 1)

    if not daily_ret:
        return {"n_days": 0, "abs_ret_%": 0, "n_skipped_limit_up": 0,
                "avg_impact_%": 0, "avg_picks_realized": 0}

    abs_ret = (1 + pd.Series(daily_ret)).prod() - 1
    avg_impact = np.mean(impact_costs) * 100 if impact_costs else 0
    return {
        "n_days": len(daily_ret),
        "abs_ret_%": round(abs_ret * 100, 2),
        "n_skipped_limit_up": int(n_skipped_limit_up),
        "avg_impact_%": round(avg_impact, 3),
        "avg_picks_realized": round(np.mean(n_picks_realized), 1) if n_picks_realized else 0,
    }


def main():
    qlib.init(provider_uri=str(QLIB_DIR), region=REG_CN)
    print("[1/3] qlib initialized — v9 realistic walk-forward")

    first_test = datetime(2024, 9, 1)
    last_test = datetime(2026, 4, 1)
    months = []
    cur = first_test
    while cur <= last_test:
        months.append(cur)
        cur += relativedelta(months=1)
    print(f"[2/3] {len(months)} 月 × {len(PORTFOLIO_SIZES)} 资金 = "
          f"{len(months) * len(PORTFOLIO_SIZES)} 次回测")

    all_rows = []
    for pv, pv_label in PORTFOLIO_SIZES:
        print(f"\n--- portfolio = {pv_label} ({pv:.0e}) ---")
        month_rows = []
        for i, m in enumerate(months, 1):
            print(f"  [{i:2d}/{len(months)}] {m.strftime('%Y-%m')} ", end="", flush=True)
            try:
                res = realistic_window(m, pv)
                res["month"] = m.strftime("%Y-%m")
                res["pv"] = pv_label
                month_rows.append(res)
                print(f"abs={res['abs_ret_%']:+6.2f}% picks={res['avg_picks_realized']:.0f} "
                      f"limit_skip={res['n_skipped_limit_up']:2d} "
                      f"impact={res['avg_impact_%']:.3f}%")
            except Exception as e:
                print(f"FAIL: {str(e)[:60]}")
                month_rows.append({"month": m.strftime("%Y-%m"), "pv": pv_label,
                                    "n_days": 0, "abs_ret_%": 0, "n_skipped_limit_up": 0,
                                    "avg_impact_%": 0, "avg_picks_realized": 0})
        all_rows.extend(month_rows)
        df_pv = pd.DataFrame(month_rows)
        cum = ((1 + df_pv["abs_ret_%"] / 100).prod() - 1) * 100
        avg_pos = (df_pv["abs_ret_%"] > 0).sum()
        print(f"  → 累计 abs (复利): {cum:+.2f}% | 正月: {avg_pos}/{len(df_pv)}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "v9_realistic_stats.csv", index=False)

    print(f"\n[3/3] === 4 资金规模对照 (20 月 OOS) ===\n")
    pivot = df.pivot_table(index="month", columns="pv", values="abs_ret_%")
    print(pivot.to_string())

    print(f"\n=== 累计收益 (复利) ===")
    for _, pv_label in PORTFOLIO_SIZES:
        sub = df[df["pv"] == pv_label]
        cum = ((1 + sub["abs_ret_%"] / 100).prod() - 1) * 100
        n_pos = (sub["abs_ret_%"] > 0).sum()
        avg_impact = sub["avg_impact_%"].mean()
        total_skip = sub["n_skipped_limit_up"].sum()
        print(f"  {pv_label:>10s}: 累计 {cum:+7.2f}%  正月 {n_pos}/{len(sub)}  "
              f"平均冲击 {avg_impact:.3f}%  涨停跳过 {total_skip}")

    print(f"\n=== 对比 v8 理想回测 ===")
    print(f"  v8 (无摩擦): 累计 +152.18%, IR 2.01, 16/20 正")

    md = ["# v9 真实摩擦回测 (4 资金规模 × 20 月 OOS)", ""]
    md.append("## 现实层")
    md.append(f"- 冲击成本: 0.5 × √(order/daily_amount)")
    md.append(f"- 涨停过滤: T+1 open chg ≥ 9% 跳过")
    md.append(f"- 信号延迟: 成交价 = (open + close) / 2")
    md.append(f"- 单股仓位 ≤ 当日成交额 5%")
    md.append("")
    md.append("## 累计 (复利) 收益对照")
    md.append("")
    md.append("| 资金 | 累计 abs% | 正月/总 | 平均冲击% | 涨停跳过 |")
    md.append("|---|---|---|---|---|")
    for _, pv_label in PORTFOLIO_SIZES:
        sub = df[df["pv"] == pv_label]
        cum = ((1 + sub["abs_ret_%"] / 100).prod() - 1) * 100
        n_pos = (sub["abs_ret_%"] > 0).sum()
        avg_impact = sub["avg_impact_%"].mean()
        total_skip = sub["n_skipped_limit_up"].sum()
        md.append(f"| {pv_label} | {cum:+.2f}% | {n_pos}/{len(sub)} | "
                  f"{avg_impact:.3f}% | {total_skip} |")
    md.append("")
    md.append("## 对比 v8 理想回测")
    md.append("- v8 (无摩擦): 累计 **+152.18%**, IR 2.01")
    (OUT_DIR / "v9_realistic_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出: v9_realistic_{{stats.csv, report.md}}")


if __name__ == "__main__":
    main()
