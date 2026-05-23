"""v8 qlib Alpha158 + TopkDropout 在 3 年长历史 (2023-2026) 上跑.

Same architecture as strategy_v8_qlib_alpha158.py but:
  - data: data_cache/qlib_bin (1,757 全 A 股, 2023-04~2026-05)
  - universe: 'all' 替代 'csi300'
  - benchmark: 茅台 (qlib backtest 需要 benchmark 字段)

For fair vs qbot-clean recommended (v4-7d/Top3 + 增强).

Run: python examples/strategy_v8_long_history.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = ROOT / "data_cache" / "qlib_bin"
MARKET = "all"

TRAIN_START, TRAIN_END = "2023-08-01", "2025-01-31"
VALID_START, VALID_END = "2025-02-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2026-05-20"
TOPK = 30
N_DROP = 5
BENCHMARK = "sh600519"


def build_task():
    data_handler_config = {
        "start_time": TRAIN_START,
        "end_time": TEST_END,
        "fit_start_time": TRAIN_START,
        "fit_end_time": TRAIN_END,
        "instruments": MARKET,
    }
    return {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 1,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": (TRAIN_START, TRAIN_END),
                    "valid": (VALID_START, VALID_END),
                    "test": (TEST_START, TEST_END),
                },
            },
        },
    }


def main() -> None:
    qlib.init(provider_uri=str(QLIB_DIR), region=REG_CN)
    print(f"[1/4] qlib initialized — 1757 全市场 A 股, Alpha158")
    print(f"  Train: {TRAIN_START} → {TRAIN_END}")
    print(f"  Valid: {VALID_START} → {VALID_END}")
    print(f"  Test:  {TEST_START} → {TEST_END}")

    task = build_task()
    print(f"\n[2/4] building model + Alpha158 dataset (158 features) ...")
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": TOPK,
                "n_drop": N_DROP,
            },
        },
        "backtest": {
            "start_time": TEST_START,
            "end_time": TEST_END,
            "account": 100_000_000,
            "benchmark": BENCHMARK,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    print(f"\n[3/4] training + backtesting ...")
    with R.start(experiment_name="v8_long_history"):
        R.log_params(model_class="LGBModel", topk=TOPK, n_drop=N_DROP,
                     features="Alpha158", benchmark=BENCHMARK,
                     universe="all_1757_stocks",
                     period=f"{TEST_START}_{TEST_END}")
        model.fit(dataset)
        R.save_objects(trained_model=model)

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        sar = SigAnaRecord(recorder)
        sar.generate()
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

    print(f"\n[4/4] done\n")
    print(f"=== 同时段对照参考 ===")
    print(f"  qbot-clean recommended (v4-7d + 增强):")
    print(f"     2023-04 → 2026-05, Sharpe 0.52, ann +19%, MDD -30%")
    print(f"  v8 qlib Alpha158 + TopkDropout (上面 ↑):")
    print(f"     {TEST_START} → {TEST_END}, see IR / excess return above")


if __name__ == "__main__":
    main()
