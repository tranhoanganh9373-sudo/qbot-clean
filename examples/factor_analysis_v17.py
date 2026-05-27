"""v17 LGB score 单因子分析 (alphalens-reloaded).

输入:
  data_cache/v17_predictions.parquet   # (datetime, instrument, score, month)
  data_cache/baidu_kline.parquet       # 日 K (含 close, 算 forward returns)

输出:
  examples/v17_factor_alphalens_report.md
  examples/v17_factor_ic_by_period.csv

run:
  python examples/factor_analysis_v17.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PRED_PATH = ROOT / "data_cache" / "v17_predictions.parquet"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
OUT_DIR = Path(__file__).resolve().parent
OUT_MD = OUT_DIR / "v17_factor_alphalens_report.md"
OUT_IC = OUT_DIR / "v17_factor_ic_by_period.csv"

PERIODS = (1, 5, 10, 21)
QUANTILES = 5


def _instrument_to_baidu(sym: str) -> str:
    """SH600000 -> 600000  (baidu_kline 用 6 位 code)."""
    return sym[2:] if sym.startswith(("SH", "SZ", "BJ")) else sym


def load_factor_data() -> pd.Series:
    pred = pd.read_parquet(PRED_PATH)
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    pred["asset"] = pred["instrument"]
    return pred.set_index(["datetime", "asset"])["score"]


def load_prices(start: pd.Timestamp, end: pd.Timestamp, instruments: list[str]) -> pd.DataFrame:
    print(f"  读 baidu_kline: {KLINE_PATH.name}")
    kl = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
    kl["date"] = pd.to_datetime(kl["date"])
    code_set = {_instrument_to_baidu(s) for s in instruments}
    kl = kl[kl["code"].isin(code_set)]
    kl = kl[(kl["date"] >= start - pd.Timedelta(days=5)) &
            (kl["date"] <= end + pd.Timedelta(days=40))]
    sym_map = {_instrument_to_baidu(s): s for s in instruments}
    kl["asset"] = kl["code"].map(sym_map)
    kl = kl.dropna(subset=["asset"])
    return kl.pivot(index="date", columns="asset", values="close").sort_index()


def main() -> None:
    if not PRED_PATH.exists():
        raise SystemExit(f"missing artifact: {PRED_PATH} — run strategy_v17 first")
    from alphalens.utils import get_clean_factor_and_forward_returns
    from alphalens.performance import (
        factor_information_coefficient,
        mean_return_by_quantile,
    )

    factor = load_factor_data()
    dates = factor.index.get_level_values(0)
    print(f"[1/4] factor shape: {factor.shape}  "
          f"dates {dates.min().date()} → {dates.max().date()}  "
          f"({dates.nunique()} days)")

    instruments = sorted(set(factor.index.get_level_values("asset").unique()))
    prices = load_prices(dates.min(), dates.max(), instruments)
    print(f"  prices shape: {prices.shape}")

    print(f"[2/4] get_clean_factor_and_forward_returns "
          f"(periods={PERIODS}, quantiles={QUANTILES}) ...")
    factor_data = get_clean_factor_and_forward_returns(
        factor=factor, prices=prices,
        quantiles=QUANTILES, periods=PERIODS, max_loss=0.50,
    )
    print(f"  factor_data rows: {len(factor_data):,}")

    print("[3/4] IC / mean returns by quantile ...")
    ic = factor_information_coefficient(factor_data)
    mean_ret_q, _ = mean_return_by_quantile(factor_data)
    mean_ret_q_bps = mean_ret_q * 10000

    ic.to_csv(OUT_IC)
    print(f"  IC by period → {OUT_IC.name}")

    ann_factor = 252 ** 0.5
    ic_summary = pd.DataFrame({
        "IC mean": ic.mean(),
        "IC std": ic.std(),
        "IR (annualized)": ic.mean() / ic.std() * ann_factor,
        "% IC > 0": (ic > 0).mean() * 100,
    })

    print("\n[IC summary]\n", ic_summary.round(4).to_string())
    print("\n[mean returns by quantile, bps]\n",
          mean_ret_q_bps.round(2).to_string())

    md = [
        "# v17 LGB score — alphalens 单因子分析",
        "",
        f"**输入**: `{PRED_PATH.name}` (LGB predictions, "
        f"{dates.nunique()} 交易日, {len(instruments)} 只 CSI300)",
        f"**Periods**: {PERIODS}  **Quantiles**: {QUANTILES}",
        "",
        "## IC summary",
        "",
        ic_summary.round(4).to_markdown(),
        "",
        "## Mean returns by quantile (bps)",
        "",
        "Q1 = 模型评分最低 20%, Q5 = 评分最高 20%",
        "",
        mean_ret_q_bps.round(2).to_markdown(),
        "",
        "## 解读",
        "",
        "- **IC mean** = LGB score vs 未来 period 收益的 Spearman 相关",
        "- **IR > 0.5** 视为可用因子, **IR > 1.0** 为强因子",
        "- **Q5 − Q1**: 高 score 减低 score 的多空组合 bps spread",
        "- 1日/5日 IC 正向但 21日 转负 → 信号衰减快, 周以内换手最佳",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[4/4] markdown report → {OUT_MD.name}")


if __name__ == "__main__":
    main()
