"""v8 walk-forward 验证：每月 retrain + 测下月.

For each test month from 2024-09 onwards:
  1. Train: past TRAIN_MONTHS months (default 12)
  2. Validate: last 1 month of train period
  3. Test:  the test month with TopkDropout (k=30, drop=5)

Aggregate per-month abs/excess return, IR, MDD.

Purpose: see if v8's IR 1.75 in 2025-07~2026-05 test is repeatable
or a cherry-picked window.

Run: python examples/strategy_v8_walkforward.py
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from dateutil.relativedelta import relativedelta
from qlib.backtest import backtest
from qlib.backtest import executor as exec_module
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data.dataset import DatasetH

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = ROOT / "data_cache" / "qlib_bin"
OUT_DIR = Path(__file__).resolve().parent
MARKET = "all"
TOPK = 30
N_DROP = 5
TRAIN_MONTHS = 12
BENCHMARK = "sh600519"

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


def one_window(test_month_start):
    test_start = month_start(test_month_start)
    test_end = month_end(test_month_start)
    train_start = month_start(test_month_start - relativedelta(months=TRAIN_MONTHS))
    valid_start = month_start(test_month_start - relativedelta(months=1))
    train_end = month_end(test_month_start - relativedelta(months=2))
    valid_end = month_end(test_month_start - relativedelta(months=1))

    data_handler_config = {
        "start_time": train_start,
        "end_time": test_end,
        "fit_start_time": train_start,
        "fit_end_time": train_end,
        "instruments": MARKET,
    }
    handler = Alpha158(**data_handler_config)
    dataset = DatasetH(handler=handler, segments={
        "train": (train_start, train_end),
        "valid": (valid_start, valid_end),
        "test": (test_start, test_end),
    })

    model = LGBModel(**LGB_PARAMS)
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")

    strategy = TopkDropoutStrategy(signal=pred, topk=TOPK, n_drop=N_DROP)
    executor_obj = exec_module.SimulatorExecutor(
        time_per_step="day", generate_portfolio_metrics=True
    )
    portfolio_metric_dict, _indicator_dict = backtest(
        start_time=test_start,
        end_time=test_end,
        strategy=strategy,
        executor=executor_obj,
        benchmark=BENCHMARK,
        account=100_000_000,
        exchange_kwargs=dict(
            freq="day", limit_threshold=0.095, deal_price="close",
            open_cost=0.0005, close_cost=0.0015, min_cost=5,
        ),
    )
    pm_tuple = portfolio_metric_dict["1day"]
    if isinstance(pm_tuple, tuple):
        pm = pm_tuple[0]
    else:
        pm = pm_tuple

    daily_ret = pm["return"]
    daily_bench = pm["bench"]
    excess_no_cost = daily_ret - daily_bench

    if len(daily_ret) == 0:
        return {"month": test_month_start.strftime("%Y-%m"), "n_days": 0,
                "abs_ret_%": 0, "excess_%": 0, "ir": 0, "mdd_%": 0}

    abs_ret = (1 + daily_ret).prod() - 1
    excess_ret = (1 + excess_no_cost).prod() - 1
    ir = (excess_no_cost.mean() / excess_no_cost.std() * np.sqrt(252)
          if excess_no_cost.std() > 0 else 0)
    cum = (1 + excess_no_cost).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()

    return {
        "month": test_month_start.strftime("%Y-%m"),
        "n_days": len(daily_ret),
        "abs_ret_%": round(abs_ret * 100, 2),
        "excess_%": round(excess_ret * 100, 2),
        "ir": round(ir, 2),
        "mdd_%": round(mdd * 100, 2),
    }


def main() -> None:
    qlib.init(provider_uri=str(QLIB_DIR), region=REG_CN)
    print(f"[1/3] qlib initialized — walk-forward: train {TRAIN_MONTHS} mo + test 1 mo")

    first_test = datetime(2024, 9, 1)
    last_test = datetime(2026, 4, 1)
    months = []
    cur = first_test
    while cur <= last_test:
        months.append(cur)
        cur += relativedelta(months=1)
    print(f"[2/3] rolling {len(months)} test months: "
          f"{months[0].strftime('%Y-%m')} → {months[-1].strftime('%Y-%m')}")

    rows = []
    for i, m in enumerate(months, 1):
        print(f"  [{i:2d}/{len(months)}] {m.strftime('%Y-%m')} ", end="", flush=True)
        try:
            res = one_window(m)
            rows.append(res)
            print(f"abs={res['abs_ret_%']:+6.2f}% excess={res['excess_%']:+6.2f}% ir={res['ir']:+5.2f}")
        except Exception as e:
            print(f"FAIL: {str(e)[:80]}")
            rows.append({"month": m.strftime("%Y-%m"), "n_days": 0,
                         "abs_ret_%": 0, "excess_%": 0, "ir": 0, "mdd_%": 0})

    df = pd.DataFrame(rows)
    print(f"\n[3/3] === walk-forward 总览 ===")
    print(df.to_string(index=False))

    n_pos_excess = (df["excess_%"] > 0).sum()
    avg_excess = df["excess_%"].mean()
    cum_excess = ((1 + df["excess_%"] / 100).prod() - 1) * 100
    cum_abs = ((1 + df["abs_ret_%"] / 100).prod() - 1) * 100
    avg_ir = df["ir"].mean()
    median_ir = df["ir"].median()

    print(f"\n=== 聚合 ({len(df)} 月 OOS) ===")
    print(f"  正超额月: {n_pos_excess}/{len(df)} ({n_pos_excess/len(df)*100:.1f}%)")
    print(f"  平均月超额: {avg_excess:+.2f}%")
    print(f"  累计超额 (复利): {cum_excess:+.2f}%")
    print(f"  累计 abs (复利): {cum_abs:+.2f}%")
    print(f"  平均月 IR: {avg_ir:+.2f}  中位 {median_ir:+.2f}")

    df.to_csv(OUT_DIR / "v8_walkforward_stats.csv", index=False)
    md = [f"# v8 walk-forward 验证 ({len(df)} 月 OOS)", ""]
    md.append(f"- 训练窗: {TRAIN_MONTHS} 月")
    md.append(f"- 测试: 每月独立, 滚动")
    md.append(f"- Topk={TOPK}, n_drop={N_DROP}")
    md.append(f"- 期间: {months[0].strftime('%Y-%m')} → {months[-1].strftime('%Y-%m')}")
    md.append("")
    md.append("## 月度结果")
    md.append(df.to_markdown(index=False))
    md.append("")
    md.append("## 聚合")
    md.append(f"- 正超额月: {n_pos_excess}/{len(df)} ({n_pos_excess/len(df)*100:.1f}%)")
    md.append(f"- 平均月超额: {avg_excess:+.2f}%")
    md.append(f"- 累计超额: {cum_excess:+.2f}%")
    md.append(f"- 累计 abs: {cum_abs:+.2f}%")
    md.append(f"- 平均月 IR: {avg_ir:+.2f}")
    (OUT_DIR / "v8_walkforward_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出: v8_walkforward_{{stats.csv, report.md}}")


if __name__ == "__main__":
    main()
