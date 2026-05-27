"""K/D sweep on Phase 2 clean predictions (v17_dens_train24).

目的: 在不重训 model 的前提下, 量化集中度 (K) × 月度替换数 (D) 对
风险/收益 trade-off 的影响.

策略:
  - 复用 strategy_v17_dens_grid.realistic_window 保证跟 baseline 同口径
  - 读 data_cache/v17_dens_train24_predictions.parquet (Phase 2 clean)
  - 不重训 model, 60m OOS walk-forward (2021-05 ~ 2026-04)
  - 12 combos: K ∈ {3, 4, 5, 8} × D ∈ {1, 2, 3}
  - 默认 NO 价格 filter (保持跟 baseline v17_dens_clean_phase2 同口径,
    K=8 D=2 应该 == Calmar 0.42 Sharpe 0.68 ann 12.45%)
  - 可选 --with-price-filter 加 125-元上限 (paper_trade production 行为)

性能优化:
  - 外层 month loop, 内层 K/D loop -> qlib 每月仅 load 一次 price
  - LRU cache on get_price_data: 60 unique month vs 720 backtest

输出:
  - examples/k_d_sweep_clean_oos_stats.csv  (per-month per-combo)
  - 终端打印 12-combo summary 表 (sorted by Calmar desc)

绝不修改:
  - examples/paper_trade_today.py
  - examples/strategy_v17_dens_grid.py  (只 import + monkey-patch)
  - data_cache/v17_dens_train24_predictions.parquet  (只读)
  - data_cache/baidu_kline.parquet  (只读)

Run:
  .venv/bin/python examples/k_d_sweep_clean_oos.py
  .venv/bin/python examples/k_d_sweep_clean_oos.py --with-price-filter
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "src"))

# ---- OOS window (matches baseline v17_dens_clean_phase2 60m) ----
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# ---- sweep grid ----
K_VALUES = [3, 4, 5, 8]
D_VALUES = [1, 2, 3]
COMBOS = [(k, d) for k in K_VALUES for d in D_VALUES]

# ---- artifacts ----
PRED_CACHE = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
OUT_STATS_CSV = ROOT / "examples" / "k_d_sweep_clean_oos_stats.csv"

# ---- price filter (production paper_trade behavior) ----
# 125 元上限固定; 不随 K/D 动态 — 跟 baseline 60m K=8 D=2 可比性
MAX_AFFORDABLE_PRICE = 125.0


def _annualize(returns: pd.Series, n_periods_per_year: int = 12) -> dict:
    cum = (1 + returns / 100).prod() - 1
    n = len(returns)
    years = n / n_periods_per_year
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else 0
    mean = (returns / 100).mean()
    std = (returns / 100).std()
    sharpe = mean / std * np.sqrt(n_periods_per_year) if std > 0 else 0
    cs = (1 + returns / 100).cumprod()
    peak = cs.cummax()
    mdd = ((cs - peak) / peak).min()
    calmar = (ann * 100) / abs(mdd * 100) if mdd < 0 else 0.0
    return {
        "cum_%": round(cum * 100, 2),
        "ann_%": round(ann * 100, 2),
        "sharpe": round(sharpe, 2),
        "mdd_%": round(mdd * 100, 2),
        "win_%": round((returns > 0).mean() * 100, 2),
        "calmar": round(calmar, 2),
        "n": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-price-filter", action="store_true",
                        help="apply 125-元 close price upper limit (production filter)")
    args = parser.parse_args()

    # qlib init must happen before importing v17 in a way that triggers data load
    import qlib  # noqa: WPS433
    from qlib.constant import REG_CN

    QLIB_DIR = str(ROOT / "data_cache" / "qlib_baidu")
    qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
    print(f"[init] qlib OK (provider={QLIB_DIR})")
    print(f"[init] price filter (125-元): {'ON' if args.with_price_filter else 'OFF'}")

    import strategy_v17_dens_grid as v17  # noqa: WPS433

    # Point v17 at Phase 2 clean cache
    v17.PRED_CACHE = PRED_CACHE
    v17._pred_disk_df = None
    v17._pred_cache.clear()
    v17.TRAIN_MONTHS = 24
    v17.PORTFOLIO_VALUE = 5e4
    v17.STOP_LOSS_PCT = 0.0
    v17.VOL_TARGET_ANN = 0.0
    v17.MARKET = "csi300"

    if not PRED_CACHE.exists():
        print(f"[FATAL] missing pred cache: {PRED_CACHE}")
        return 1
    sz = PRED_CACHE.stat().st_size / 1e6
    print(f"[init] pred cache: {PRED_CACHE.name} ({sz:.1f} MB)")

    # ---- LRU cache on get_price_data so 12 combos share same-month price ----
    _orig_get_price_data = v17.get_price_data

    @lru_cache(maxsize=128)
    def _cached_get_price_data(start: str, end: str) -> pd.DataFrame:
        return _orig_get_price_data(start, end)

    v17.get_price_data = _cached_get_price_data
    print("[init] get_price_data wrapped with LRU cache")

    # ---- optional: build 125-元 filtered pred view ----
    first_dt = datetime.strptime(OOS_FIRST + "-01", "%Y-%m-%d")
    last_dt = datetime.strptime(OOS_LAST + "-01", "%Y-%m-%d")

    if args.with_price_filter:
        print("[init] building 125-元 price filter view ...", flush=True)
        pred_df = pd.read_parquet(PRED_CACHE)
        pred_start = (first_dt - relativedelta(months=1)).strftime("%Y-%m")
        pred_end = (last_dt + relativedelta(months=1)).strftime("%Y-%m")
        pred_df = pred_df[
            (pred_df["month"] >= pred_start) & (pred_df["month"] <= pred_end)
        ].copy()

        span_start = first_dt.strftime("%Y-%m-01")
        span_end = (last_dt + relativedelta(months=1, days=10)).strftime("%Y-%m-%d")
        print(f"[init] loading qlib prices {span_start} ~ {span_end} ...", flush=True)
        t0 = time.time()
        price_full = _orig_get_price_data(span_start, span_end)
        print(f"[init] price panel loaded: shape={price_full.shape} "
              f"({time.time()-t0:.1f}s)")

        price_full["date"] = pd.to_datetime(price_full["date"])
        expensive = price_full[price_full["close"] > MAX_AFFORDABLE_PRICE][
            ["date", "instrument"]
        ].copy()
        expensive = expensive.rename(columns={"date": "datetime"})
        expensive["expensive"] = True

        pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])
        pred_df = pred_df.merge(expensive, on=["datetime", "instrument"], how="left")
        n_before = len(pred_df)
        pred_df = pred_df[pred_df["expensive"].isna()].drop(columns=["expensive"])
        n_after = len(pred_df)
        print(f"[init] 125-元 filter: dropped {n_before - n_after:,} "
              f"of {n_before:,} pred rows ({(n_before-n_after)/n_before*100:.1f}%)")

        # v17._load_pred_from_disk reads disk only when _pred_disk_df is None.
        # By setting it here we feed it our in-memory filtered pred.
        v17._pred_disk_df = pred_df
    else:
        print("[init] no price filter — direct disk read by v17._load_pred_from_disk")

    months: list[datetime] = []
    cur = first_dt
    while cur <= last_dt:
        months.append(cur)
        cur += relativedelta(months=1)
    print(f"[run] OOS {len(months)} months × {len(COMBOS)} combos "
          f"= {len(months) * len(COMBOS)} backtests")

    proxy = v17.build_market_proxy()

    # ---- outer: month loop; inner: combo loop ----
    all_rows: list[dict] = []
    t_start = time.time()
    for mi, m in enumerate(months, 1):
        for k, d in COMBOS:
            v17.K_NORMAL = k
            v17.DROP_NORMAL = d
            try:
                res = v17.realistic_window(m, proxy, with_regime=False)
                res["month"] = m.strftime("%Y-%m")
                res["k"] = k
                res["drop"] = d
                res["config"] = f"K={k}D={d}"
                all_rows.append(res)
            except Exception as e:
                print(f"  FAIL {m.strftime('%Y-%m')} K={k}D={d}: {str(e)[:80]}",
                      flush=True)
                all_rows.append({
                    "month": m.strftime("%Y-%m"),
                    "k": k, "drop": d, "config": f"K={k}D={d}",
                    "abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                    "regime_days": "", "n_skipped_limit": 0, "n_stop_loss": 0,
                })
        elapsed = time.time() - t_start
        eta = elapsed / mi * (len(months) - mi)
        print(f"  [{mi:2d}/{len(months)}] {m.strftime('%Y-%m')} "
              f"({len(COMBOS)} combos)  cum={elapsed:.0f}s  ETA={eta:.0f}s",
              flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n[done] total {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_STATS_CSV, index=False)
    print(f"[saved] {OUT_STATS_CSV.relative_to(ROOT)}")

    # ---- summary per combo ----
    summary_rows = []
    for k, d in COMBOS:
        sub = df[(df["k"] == k) & (df["drop"] == d)].sort_values("month")
        stats = _annualize(sub["abs_ret_%"])
        stats["k"] = k
        stats["d"] = d
        stats["avg_picks"] = round(sub["avg_picks"].mean(), 2)
        stats["config"] = f"K={k}D={d}"
        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("calmar", ascending=False).reset_index(drop=True)
    summary_df = summary_df[["config", "k", "d", "calmar", "sharpe", "ann_%",
                              "mdd_%", "cum_%", "win_%", "avg_picks", "n"]]

    print("\n" + "=" * 78)
    print("=== 12-combo K/D trade-off (sorted by Calmar desc) ===")
    print("=" * 78)
    print(summary_df.to_string(index=False))

    baseline_row = summary_df[summary_df["config"] == "K=8D=2"]
    if not baseline_row.empty:
        b = baseline_row.iloc[0]
        print(f"\n[baseline check] K=8 D=2: Calmar={b['calmar']} Sharpe={b['sharpe']} "
              f"ann={b['ann_%']}% MDD={b['mdd_%']}%")
        print("  reference (baseline v17_dens_clean_phase2 no-filter): "
              "Calmar=0.42 Sharpe=0.68 ann=12.45% MDD=-29.86%")
        print("  [note] 本脚本加了 125-元 价格 filter, baseline 没加 -> 数字会有差异")

    # ---- risk profile recommendations ----
    print("\n" + "=" * 78)
    print("=== risk profile recommendations ===")
    print("=" * 78)
    best_calmar = summary_df.iloc[0]
    moderate_pool = summary_df[summary_df["mdd_%"] > -40].sort_values("sharpe", ascending=False)
    moderate = moderate_pool.iloc[0] if not moderate_pool.empty else summary_df.iloc[0]
    aggressive = summary_df.sort_values("ann_%", ascending=False).iloc[0]

    print(f"  保守 (最佳 Calmar): {best_calmar['config']}  "
          f"Calmar={best_calmar['calmar']}  Sharpe={best_calmar['sharpe']}  "
          f"ann={best_calmar['ann_%']}%  MDD={best_calmar['mdd_%']}%")
    print(f"  中风险 (最佳 Sharpe, MDD>-40%): {moderate['config']}  "
          f"Calmar={moderate['calmar']}  Sharpe={moderate['sharpe']}  "
          f"ann={moderate['ann_%']}%  MDD={moderate['mdd_%']}%")
    print(f"  激进 (最高 ann): {aggressive['config']}  "
          f"Calmar={aggressive['calmar']}  Sharpe={aggressive['sharpe']}  "
          f"ann={aggressive['ann_%']}%  MDD={aggressive['mdd_%']}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
