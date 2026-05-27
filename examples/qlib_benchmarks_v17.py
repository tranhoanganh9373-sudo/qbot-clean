"""qlib 内置模型 benchmark 对比 (vs v17 LGB).

固定 single fold (避免 40 月 walk-forward 训练 4 模型 = 160 次, 不现实):
  Train: 2025-01-01 → 2025-12-31   (12 月)
  Valid: 2026-01-01 → 2026-02-28   (2 月)
  Test:  2026-03-01 → 2026-04-30   (2 月)

模型 (全部用 Alpha158 表格特征, 避免 LSTM/TabNet 需要 Alpha360 序列 handler 的复杂度):
  LGB              (v17 production, baseline)
  DoubleEnsemble   (LGB-based ensemble, 表格类)
  Linear           (Ridge / 线性基线)
  RandomForest     (qlib.contrib.model.gbdt 的 sklearn 包装) (可选)

输出:
  examples/v17_benchmarks_metrics.csv
  examples/v17_benchmarks_compare.md

run:
  python examples/qlib_benchmarks_v17.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = str(ROOT / "data_cache" / "qlib_baidu")
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "v17_benchmarks_metrics.csv"
OUT_MD = OUT_DIR / "v17_benchmarks_compare.md"

MARKET = "csi300"
TRAIN_RANGE = ("2025-01-01", "2025-12-31")
VALID_RANGE = ("2026-01-01", "2026-02-28")
TEST_RANGE = ("2026-03-01", "2026-04-30")
TOPK = 8


def make_dataset() -> DatasetH:
    handler = Alpha158(
        start_time=TRAIN_RANGE[0], end_time=TEST_RANGE[1],
        fit_start_time=TRAIN_RANGE[0], fit_end_time=TRAIN_RANGE[1],
        instruments=MARKET,
    )
    return DatasetH(handler=handler, segments={
        "train": TRAIN_RANGE, "valid": VALID_RANGE, "test": TEST_RANGE,
    })


def get_models() -> list[tuple[str, object]]:
    models: list[tuple[str, object]] = []
    models.append(("LGB", LGBModel(
        loss="mse", colsample_bytree=0.8879, learning_rate=0.0421,
        subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768,
        max_depth=8, num_leaves=210, num_threads=1,
    )))
    try:
        from qlib.contrib.model.double_ensemble import DEnsembleModel
        models.append(("DoubleEnsemble", DEnsembleModel(
            base_model="gbm",
            loss="mse", num_models=3,
            enable_sr=True, enable_fs=True,
            alpha1=1.0, alpha2=1.0, bins_sr=10, bins_fs=5,
            decay=0.5, sample_ratios=[0.8, 0.7, 0.6, 0.5, 0.4],
            sub_weights=[1, 0.2, 0.2], epochs=20,
        )))
    except Exception as e:
        print(f"[skip] DoubleEnsemble: {e}")
    try:
        from qlib.contrib.model.linear import LinearModel
        models.append(("Linear (Ridge)", LinearModel(estimator="ridge", alpha=0.05)))
    except Exception as e:
        print(f"[skip] Linear: {e}")
    return models


def topk_cum_return(pred: pd.Series, dataset: DatasetH, k: int = TOPK) -> dict:
    """简化 TopK 等权: 每日选 top-k 等权持仓到下一日."""
    label = dataset.prepare(segments="test", col_set="label", data_key="raw")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]
    label.name = "ret"
    df = pd.concat([pred.rename("score"), label], axis=1, join="inner").dropna()
    df.index.names = ["datetime", "instrument"]
    daily_ret = []
    for _, g in df.groupby(level="datetime"):
        top = g.nlargest(k, "score")
        daily_ret.append(top["ret"].mean())
    rets = pd.Series(daily_ret)
    cum = (1 + rets).prod() - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    mdd = ((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min()
    return {
        "n_days": int(len(rets)),
        "cum_ret": float(cum),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "win_rate": float((rets > 0).mean()),
    }


def ic_stats(pred: pd.Series, dataset: DatasetH) -> dict:
    label = dataset.prepare(segments="test", col_set="label", data_key="raw")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1, join="inner").dropna()
    df.index.names = ["datetime", "instrument"]
    daily_ic = df.groupby(level="datetime").apply(
        lambda g: g["p"].corr(g["y"], method="spearman") if len(g) > 5 else np.nan
    ).dropna()
    return {
        "ic_mean": float(daily_ic.mean()),
        "ic_std": float(daily_ic.std()),
        "icir": float(daily_ic.mean() / daily_ic.std() * np.sqrt(252))
                if daily_ic.std() > 0 else 0,
        "ic_pos_pct": float((daily_ic > 0).mean() * 100),
    }


def main() -> None:
    qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
    print(f"[1/3] qlib initialized — {QLIB_DIR}")
    print(f"  Train {TRAIN_RANGE[0]} → {TRAIN_RANGE[1]}")
    print(f"  Valid {VALID_RANGE[0]} → {VALID_RANGE[1]}")
    print(f"  Test  {TEST_RANGE[0]} → {TEST_RANGE[1]}")

    dataset = make_dataset()
    models = get_models()
    print(f"\n[2/3] {len(models)} 模型 single-fold train ...\n")

    rows = []
    for name, model in models:
        print(f"=== {name} ===")
        t0 = time.time()
        try:
            model.fit(dataset)
            pred = model.predict(dataset, segment="test")
            if isinstance(pred, pd.DataFrame):
                pred = pred.iloc[:, 0]
            ic = ic_stats(pred, dataset)
            top = topk_cum_return(pred, dataset, k=TOPK)
            elapsed = time.time() - t0
            row = {
                "model": name,
                "fit_secs": round(elapsed, 1),
                **{f"{k}": round(v, 4) for k, v in ic.items()},
                **{f"top{TOPK}_{k}": round(v, 4) for k, v in top.items()},
            }
            rows.append(row)
            print(f"  fit {elapsed:.1f}s  IC={ic['ic_mean']:+.4f}  "
                  f"ICIR={ic['icir']:+.3f}  TopK cum={top['cum_ret']*100:+.2f}%")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAIL after {elapsed:.1f}s: {type(e).__name__}: {str(e)[:120]}")
            rows.append({"model": name, "fit_secs": round(elapsed, 1),
                          "error": str(e)[:200]})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[3/3] csv → {OUT_CSV.name}")

    cols_show = ["model", "fit_secs", "ic_mean", "icir", "ic_pos_pct",
                 f"top{TOPK}_cum_ret", f"top{TOPK}_sharpe", f"top{TOPK}_mdd",
                 f"top{TOPK}_win_rate", f"top{TOPK}_n_days"]
    cols_show = [c for c in cols_show if c in df.columns]
    md = [
        "# qlib benchmark 模型对比 (single fold)",
        "",
        f"**Train**: {TRAIN_RANGE[0]} → {TRAIN_RANGE[1]} (12 月)  ",
        f"**Valid**: {VALID_RANGE[0]} → {VALID_RANGE[1]} (2 月)  ",
        f"**Test** : {TEST_RANGE[0]} → {TEST_RANGE[1]} (2 月)  ",
        f"**Universe**: CSI300 ({MARKET})  **TopK**: {TOPK}",
        "",
        df[cols_show].to_markdown(index=False),
        "",
        "## 解读",
        "",
        "- **ic_mean** 高 = 模型预测和未来收益相关性强",
        "- **icir** = IC 的信息比率, > 1.0 强, > 2.0 很强",
        "- **top8_cum_ret** = 每日选 score 最高 8 只等权持仓的累计收益",
        "- **fit_secs** = 单次训练耗时, LGB 通常最快",
        "- 注: 测试期仅 2 个月 (~40 交易日), 噪声大, 结论需要 walk-forward 验证",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"  markdown → {OUT_MD.name}")


if __name__ == "__main__":
    main()
