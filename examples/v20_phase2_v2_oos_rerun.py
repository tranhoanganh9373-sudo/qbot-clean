"""Task 2+3: Re-run v19.6 (amplitude) + v19.4 (margin) **OOS only**
on the new Phase 2 v2 pred cache.

Both λ already 锁定 by user (Phase A decisions, no re-sweep):
    v19.6: λ_a20=0.30,  factor=amp_imb_20d,    sign=-1
    v19.4: λ_m5=0.10, λ_m20=0.10, factor=margin_5d_chg + margin_20d_chg, sign=-1

不改 strategy_v19_*.py (production-touch 禁区). 复用它们的 panel 构造函数
(build_pit_panel_on_pred_axis / build_margin_panel / build_technical_panel).

输出:
    examples/v19_6_amplitude_oos_stats_v2.csv
    examples/v19_4_sidecar_technical_oos_stats_v2.csv

不写入新的 IS grid (本任务不重 sweep).

Run:
    .venv/bin/python examples/v20_phase2_v2_oos_rerun.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

# Reuse panel builders from existing strategy scripts (read-only import)
import strategy_v19_6_amplitude as v196  # noqa: E402
import strategy_v19_4_sidecar_technical as v194  # noqa: E402
from _factor_kline_panel import build_pit_panel_on_pred_axis  # noqa: E402

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"

OUT_V196_OOS = ROOT / "examples" / "v19_6_amplitude_oos_stats_v2.csv"
OUT_V194_OOS = ROOT / "examples" / "v19_4_sidecar_technical_oos_stats_v2.csv"

OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# Locked λ (Phase A user decision, no re-sweep on v2 cache)
V196_LAM_5 = 0.0
V196_LAM_20 = 0.30
V196_LAM_M = 0.0

V194_LAM_M5 = 0.10
V194_LAM_M20 = 0.10
V194_LAM_DT = 0.0


def _annualize(returns: pd.Series) -> dict:
    cum = (1 + returns / 100).prod() - 1
    n = len(returns)
    years = n / 12
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else 0
    mean = (returns / 100).mean()
    std = (returns / 100).std()
    sharpe = mean / std * np.sqrt(12) if std > 0 else 0
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


def run_oos(adj_pred_path: Path, tag: str) -> tuple[pd.DataFrame, dict]:
    """Run OOS walk-forward using the adj pred parquet."""
    import qlib  # noqa: F401
    from qlib.constant import REG_CN

    import strategy_v17_dens_grid as v17

    QLIB_DIR = str(ROOT / "data_cache" / "qlib_baidu")
    if not getattr(run_oos, "_qlib_init", False):
        qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
        run_oos._qlib_init = True
        run_oos._proxy = v17.build_market_proxy()

    v17.PRED_CACHE = adj_pred_path
    v17._pred_disk_df = None
    v17._pred_cache.clear()
    v17.MARKET = "csi300"
    v17.TRAIN_MONTHS = 24
    v17.K_NORMAL = 8
    v17.DROP_NORMAL = 2
    v17.PORTFOLIO_VALUE = 5e4
    v17.STOP_LOSS_PCT = 0.0
    v17.VOL_TARGET_ANN = 0.0

    first = datetime.strptime(OOS_FIRST + "-01", "%Y-%m-%d")
    last = datetime.strptime(OOS_LAST + "-01", "%Y-%m-%d")
    months = []
    cur = first
    while cur <= last:
        months.append(cur)
        cur += relativedelta(months=1)

    rows = []
    for i, m in enumerate(months, 1):
        try:
            res = v17.realistic_window(
                m, run_oos._proxy, with_regime=False,
            )
            res["month"] = m.strftime("%Y-%m")
            rows.append(res)
            if i % 12 == 0 or i == len(months) or i == 1:
                print(f"  [{tag}] {i:3d}/{len(months)} {res['month']}: "
                      f"abs_ret={res['abs_ret_%']:+6.2f}%  "
                      f"picks={res['avg_picks']:.1f}", flush=True)
        except Exception as e:
            print(f"  [{tag}] {i:3d}/{len(months)} "
                  f"{m.strftime('%Y-%m')} FAIL: {str(e)[:120]}", flush=True)
            rows.append({"month": m.strftime("%Y-%m"),
                         "abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                         "regime_days": "", "n_skipped_limit": 0,
                         "n_stop_loss": 0})

    df = pd.DataFrame(rows)
    stats = _annualize(df["abs_ret_%"])
    stats["avg_picks"] = round(df["avg_picks"].mean(), 2)
    return df, stats


def task2_v196() -> dict:
    print("=" * 70)
    print("Task 2: v19.6 amplitude OOS rerun on v2 pred cache "
          "(locked λ_a20=0.30)")
    print("=" * 70)
    print(f"OOS: {OOS_FIRST} ~ {OOS_LAST} (60 months)")

    print("[panel] building amp_imb_5d/20d via _factor_kline_panel ...")
    amp_panel = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["amp_imb_5d", "amp_imb_20d"],
    )
    margin_panel = v196.build_margin_panel()  # required by builder signature

    print(f"[adj] building adjusted pred with locked λ=(a5={V196_LAM_5}, "
          f"a20={V196_LAM_20}, m={V196_LAM_M})")
    adj_path = v196.build_adjusted_predictions(
        amp_panel, margin_panel,
        V196_LAM_5, V196_LAM_20, V196_LAM_M,
        "OOS_locked_v2",
    )

    print(f"[oos] walk-forward {OOS_FIRST} ~ {OOS_LAST}")
    df, stats = run_oos(adj_path, tag="v19.6")
    df.to_csv(OUT_V196_OOS, index=False)
    print(f"[saved] {OUT_V196_OOS}")
    print(f"v19.6 OOS: Calmar={stats['calmar']} Sharpe={stats['sharpe']} "
          f"ann={stats['ann_%']}% MDD={stats['mdd_%']}% "
          f"cum={stats['cum_%']}% win%={stats['win_%']}")
    return stats


def task3_v194() -> dict:
    print("\n" + "=" * 70)
    print("Task 3: v19.4 margin OOS rerun on v2 pred cache "
          "(locked λ_m5=0.10, λ_m20=0.10)")
    print("=" * 70)
    print(f"OOS: {OOS_FIRST} ~ {OOS_LAST} (60 months)")

    print("[panel] building margin + DT panel via v194.build_technical_panel...")
    tech_panel = v194.build_technical_panel()

    print(f"[adj] building adjusted pred with locked λ=(m5={V194_LAM_M5}, "
          f"m20={V194_LAM_M20}, dt={V194_LAM_DT})")
    adj_path = v194.build_adjusted_predictions(
        tech_panel,
        V194_LAM_M5, V194_LAM_M20, V194_LAM_DT,
        "OOS_locked_v2",
    )

    print(f"[oos] walk-forward {OOS_FIRST} ~ {OOS_LAST}")
    df, stats = run_oos(adj_path, tag="v19.4")
    df.to_csv(OUT_V194_OOS, index=False)
    print(f"[saved] {OUT_V194_OOS}")
    print(f"v19.4 OOS: Calmar={stats['calmar']} Sharpe={stats['sharpe']} "
          f"ann={stats['ann_%']}% MDD={stats['mdd_%']}% "
          f"cum={stats['cum_%']}% win%={stats['win_%']}")
    return stats


def main() -> int:
    s196 = task2_v196()
    s194 = task3_v194()

    print("\n" + "=" * 70)
    print("=== FINAL OOS Phase 2 v2 ===")
    print("=" * 70)
    print(f"v19.6 (amp_imb_20d λ=0.30): Calmar={s196['calmar']} "
          f"Sharpe={s196['sharpe']} ann={s196['ann_%']}% "
          f"MDD={s196['mdd_%']}% cum={s196['cum_%']}%")
    print(f"v19.4 (m5+m20 λ=0.10+0.10): Calmar={s194['calmar']} "
          f"Sharpe={s194['sharpe']} ann={s194['ann_%']}% "
          f"MDD={s194['mdd_%']}% cum={s194['cum_%']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
