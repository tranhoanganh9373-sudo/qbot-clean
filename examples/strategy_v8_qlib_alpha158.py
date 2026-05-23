"""v8 = qlib Alpha158 (158 因子) + LightGBM + TopkDropout 调仓 + qlib data layer.

Uses qlib's built-in modules directly (no custom port needed):
  - data layer:  qlib.data + qlib bin format (~/.qlib/qlib_data/cn_data)
  - features:    Alpha158 handler (158 个 factor)
  - model:       LGBModel (qlib 封装的 LightGBM)
  - strategy:    TopkDropoutStrategy (k=30, dropout=5 per day)

Time range: qlib bundled cn_data only goes to 2020-09. Strategy runs 2017-2020.

Run:  python examples/strategy_v8_qlib_alpha158.py
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

OUT_DIR = Path(__file__).resolve().parent
MARKET = "csi300"
BENCHMARK = "SH000300"

TRAIN_START, TRAIN_END = "2008-01-01", "2014-12-31"
VALID_START, VALID_END = "2015-01-01", "2016-12-31"
TEST_START, TEST_END = "2017-01-01", "2020-08-01"
TOPK = 30
N_DROP = 5


def build_task():
    data_handler_config = {
        "start_time": TRAIN_START,
        "end_time": TEST_END,
        "fit_start_time": TRAIN_START,
        "fit_end_time": TRAIN_END,
        "instruments": MARKET,
    }

    task = {
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
    return task


def main() -> None:
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    print(f"[1/4] qlib initialized — CSI300 universe, Alpha158 features")
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

    print(f"\n[3/4] training + backtesting (this may take ~5 min) ...")
    with R.start(experiment_name="v8_alpha158_topkdropout"):
        R.log_params(model_class="LGBModel", topk=TOPK, n_drop=N_DROP,
                     features="Alpha158", benchmark=BENCHMARK)
        model.fit(dataset)
        R.save_objects(trained_model=model)

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        sar = SigAnaRecord(recorder)
        sar.generate()
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

    print(f"\n[4/4] done — see business metrics above (excess return + IR)")
    print(f"\n参考对比:")
    print(f"  qbot-clean recommended (v4 + 增强): 2023-04 → 2026-05, Sharpe 0.52, ann +19%")
    print(f"  v8 qlib Alpha158 + TopkDropout:    2017-01 → 2020-08, ann +24.7%, IR 1.50 (excess)")
    print(f"  注意: 时段不同 + universe 不同 (csi300 vs 全市场)")


if __name__ == "__main__":
    main()
