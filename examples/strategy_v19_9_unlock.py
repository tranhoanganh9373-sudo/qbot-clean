"""v19.9 Sidecar — train24 + 限售股解禁因子 (unlock) overlay.

参考: 自建因子 (claude_finance.unlock 模块) IC IS 2014-2020.

因子逻辑 (本质 mean reversion):
    unlock_pct_next_N = sum over (T, T+N 日历日] of (akshare 占解禁前流通市值比例)
    含义: 该股 T 之后 N 天内有多少 % 的流通市值即将解禁?
    解禁压力 大 → 后续走弱.
    IS ICIR (84 月, fwd 5d): -1.144 强反向; combo_neg_pct 1.140 镜像.

Sidecar 公式:
    final_score = z(train24_pred) - λ * z(unlock_factor)
  (unlock 因子 IC 负 → 减号; combo_neg_pct 因子 IC 正 → 已是 +sign)

9 IS combo sweep (2017-01 ~ 2020-12, 48 月) → 锁定 best IS Calmar → OOS 60 月.

| # | factor                | λ     |
|---|-----------------------|-------|
| 1 | unlock_pct_next_60    | 0.10  |
| 2 | unlock_pct_next_60    | 0.20  |
| 3 | unlock_pct_next_60    | 0.30  |
| 4 | unlock_pct_next_20    | 0.10  |
| 5 | unlock_pct_next_20    | 0.20  |
| 6 | combo_neg_pct (z20+z60 反向等权) | 0.10 |
| 7 | combo_neg_pct                   | 0.20 |
| 8 | combo_neg_pct                   | 0.30 |
| 9 | unlock_imminent_20    | 0.20  |

OOS 严格协议: 看 OOS 不允许回头改 λ.

不动任何 production 文件 / pred cache / kline / qlib_baidu / 历代 sidecar.

Run:
  .venv/bin/python examples/strategy_v19_9_unlock.py
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

from claude_finance import unlock as unlock_mod  # noqa: E402

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
ADJ_PRED = ROOT / "data_cache" / "v19_9_unlock_predictions.parquet"
OUT_IS_GRID = ROOT / "examples" / "v19_9_unlock_is_grid.csv"
OUT_OOS_STATS = ROOT / "examples" / "v19_9_unlock_oos_stats.csv"

IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# 9 IS combos: (label, factor_col, lambda, sign)
# 实际公式: final = z(pred) - sign * λ * z(factor)
# sign=+1: factor IC 负 (压力大→走弱), 直接减 → buy weight 反向.
# sign=-1: combo_neg_pct IC 正 (已镜像), 用 - 号变 + 号.
IS_COMBOS = [
    ("c1_p60_l010",        "unlock_pct_next_60", 0.10, +1),
    ("c2_p60_l020",        "unlock_pct_next_60", 0.20, +1),
    ("c3_p60_l030",        "unlock_pct_next_60", 0.30, +1),
    ("c4_p20_l010",        "unlock_pct_next_20", 0.10, +1),
    ("c5_p20_l020",        "unlock_pct_next_20", 0.20, +1),
    ("c6_combo_l010",      "combo_neg_pct",      0.10, -1),
    ("c7_combo_l020",      "combo_neg_pct",      0.20, -1),
    ("c8_combo_l030",      "combo_neg_pct",      0.30, -1),
    ("c9_imm20_l020",      "unlock_imminent_20", 0.20, +1),
]


def _zscore_cs(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std()
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _instrument_to_code6(inst: str) -> str:
    return str(inst)[-6:].zfill(6)


def build_unlock_panel(
    windows_days: tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    """构造 (datetime, instrument) × unlock factor panel, PIT, 全期.

    对每个 (pred_dt T, code), 因子 = sum over (T, T+N] of unlock_ratio_cap.
    Lookahead-safe: 只看 future 解禁 calendar (T 时点已知的 schedule).

    返回字段:
        datetime, instrument,
        z_unlock_pct_next_20, z_unlock_pct_next_60,
        z_unlock_imminent_20, z_combo_neg_pct
    """
    print("[unlock] loading cache...", flush=True)
    cache = unlock_mod.load_cache()
    print(f"[unlock] cache: {len(cache):,} rows × "
          f"{cache['code'].nunique():,} stocks; "
          f"range {cache['unlock_date'].min().date()} ~ "
          f"{cache['unlock_date'].max().date()}", flush=True)

    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))
    pred_codes = sorted(pred["code"].unique())
    print(f"[unlock] pred axis: {len(pred_dt)} dates × "
          f"{len(pred_codes)} codes", flush=True)

    # 仅保留 pred 涉及的 code
    cache = cache[cache["code"].isin(set(pred_codes))].copy()
    cache = cache.sort_values(["code", "unlock_date"]).reset_index(drop=True)
    print(f"[unlock] cache filtered to pred codes: {len(cache):,} rows × "
          f"{cache['code'].nunique()} stocks", flush=True)

    pred_dt_ns = pred_dt.values.astype("datetime64[ns]")

    rows = []
    n_codes = len(pred_codes)
    for i, code in enumerate(pred_codes, 1):
        sub = cache[cache["code"] == code]
        row = {"datetime": pred_dt, "code": code}
        if sub.empty:
            for n in windows_days:
                row[f"unlock_pct_next_{n}"] = np.zeros(len(pred_dt))
                row[f"unlock_imminent_{n}"] = np.zeros(len(pred_dt))
            rows.append(pd.DataFrame(row))
            continue

        ev_dates = sub["unlock_date"].values.astype("datetime64[ns]")
        ev_pct = sub["unlock_ratio_cap"].fillna(0.0).values.astype(np.float64)

        # 对每个 pred date T, 找 (T, T+N] 内事件 idx 范围
        # left bound: strictly > T → searchsorted with side='right' on T
        lo = np.searchsorted(ev_dates, pred_dt_ns, side="right")
        cs = np.concatenate([[0.0], np.cumsum(ev_pct)])
        for n in windows_days:
            upper = pred_dt_ns + np.timedelta64(n, "D")
            hi = np.searchsorted(ev_dates, upper, side="right")
            pct = cs[hi] - cs[lo]
            imm = (hi > lo).astype(np.float64)
            row[f"unlock_pct_next_{n}"] = pct
            row[f"unlock_imminent_{n}"] = imm
        rows.append(pd.DataFrame(row))
        if i % 50 == 0 or i == n_codes:
            print(f"  [unlock] {i}/{n_codes} codes", flush=True)

    panel = pd.concat(rows, ignore_index=True)
    print(f"[unlock] panel rows: {len(panel):,}", flush=True)

    # sparsity 报告
    for n in windows_days:
        col = f"unlock_pct_next_{n}"
        nz = (panel[col] > 0).mean() * 100
        print(f"[unlock] {col} nonzero%: {nz:.2f}", flush=True)

    # cross-section z-scores per datetime
    print("[unlock] cross-section z-scores...", flush=True)
    panel["z_unlock_pct_next_20"] = panel.groupby("datetime")[
        "unlock_pct_next_20"
    ].transform(_zscore_cs).fillna(0.0)
    panel["z_unlock_pct_next_60"] = panel.groupby("datetime")[
        "unlock_pct_next_60"
    ].transform(_zscore_cs).fillna(0.0)
    panel["z_unlock_imminent_20"] = panel.groupby("datetime")[
        "unlock_imminent_20"
    ].transform(_zscore_cs).fillna(0.0)

    # combo_neg_pct = -(z20 + z60) / 2  (raw, 然后 z 一次)
    panel["combo_neg_pct_raw"] = -(
        panel["z_unlock_pct_next_20"] + panel["z_unlock_pct_next_60"]
    ) / 2
    panel["z_combo_neg_pct"] = panel.groupby("datetime")[
        "combo_neg_pct_raw"
    ].transform(_zscore_cs).fillna(0.0)

    # Join into instrument axis
    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    panel_pick = panel[[
        "datetime", "code",
        "z_unlock_pct_next_20", "z_unlock_pct_next_60",
        "z_unlock_imminent_20", "z_combo_neg_pct",
    ]]
    out = pred_axis.merge(panel_pick, on=["datetime", "code"], how="left")
    z_cols = [
        "z_unlock_pct_next_20", "z_unlock_pct_next_60",
        "z_unlock_imminent_20", "z_combo_neg_pct",
    ]
    out[z_cols] = out[z_cols].fillna(0.0)

    return out[["datetime", "instrument"] + z_cols]


# 映射: factor name → z 列名
FACTOR_TO_Z = {
    "unlock_pct_next_20":  "z_unlock_pct_next_20",
    "unlock_pct_next_60":  "z_unlock_pct_next_60",
    "unlock_imminent_20":  "z_unlock_imminent_20",
    "combo_neg_pct":       "z_combo_neg_pct",
}


def build_adjusted_predictions(panel: pd.DataFrame,
                               factor: str, lam: float, sign: int,
                               label: str) -> Path:
    """final = z(pred) - sign * lam * z(factor)."""
    zcol = FACTOR_TO_Z[factor]
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(
        panel[["datetime", "instrument", zcol]],
        on=["datetime", "instrument"], how="left",
    )
    merged[zcol] = merged[zcol].fillna(0.0)
    merged["final_score"] = (
        merged["z_pred"] - sign * lam * merged[zcol]
    )
    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
    print(f"  [adj] {label} factor={factor} λ={lam} sign={sign:+d} "
          f"rows={len(out):,}", flush=True)
    return ADJ_PRED


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


def run_walkforward(first_month: str, last_month: str, tag: str) -> dict:
    import qlib  # noqa: F401
    from qlib.constant import REG_CN

    import strategy_v17_dens_grid as v17

    QLIB_DIR = str(ROOT / "data_cache" / "qlib_baidu")
    if not getattr(run_walkforward, "_qlib_initialized", False):
        qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
        run_walkforward._qlib_initialized = True
        run_walkforward._proxy = v17.build_market_proxy()

    v17.PRED_CACHE = ADJ_PRED
    v17._pred_disk_df = None
    v17._pred_cache.clear()
    v17.MARKET = "csi300"
    v17.TRAIN_MONTHS = 24
    v17.K_NORMAL = 8
    v17.DROP_NORMAL = 2
    v17.PORTFOLIO_VALUE = 5e4
    v17.STOP_LOSS_PCT = 0.0
    v17.VOL_TARGET_ANN = 0.0

    first = datetime.strptime(first_month + "-01", "%Y-%m-%d")
    last = datetime.strptime(last_month + "-01", "%Y-%m-%d")
    months = []
    cur = first
    while cur <= last:
        months.append(cur)
        cur += relativedelta(months=1)

    rows = []
    for i, m in enumerate(months, 1):
        try:
            res = v17.realistic_window(
                m, run_walkforward._proxy, with_regime=False,
            )
            res["month"] = m.strftime("%Y-%m")
            rows.append(res)
            if i % 12 == 0 or i == len(months) or i == 1:
                print(f"    {i:3d}/{len(months)} {res['month']}: "
                      f"abs_ret={res['abs_ret_%']:+6.2f}%  "
                      f"picks={res['avg_picks']:.1f}", flush=True)
        except Exception as e:
            print(f"    {i:3d}/{len(months)} {m.strftime('%Y-%m')} FAIL: "
                  f"{str(e)[:120]}", flush=True)
            rows.append({"month": m.strftime("%Y-%m"),
                         "abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                         "regime_days": "", "n_skipped_limit": 0,
                         "n_stop_loss": 0})

    df = pd.DataFrame(rows)
    stats = _annualize(df["abs_ret_%"])
    stats["avg_picks"] = round(df["avg_picks"].mean(), 2)
    stats["months_df"] = df
    return stats


def main() -> int:
    print("=" * 70)
    print("Phase 4 Sidecar v19.9 — train24 + 限售股解禁因子")
    print("=" * 70)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST} (48 months)")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"combos: {len(IS_COMBOS)}")
    print()

    print("[step 1] build unlock PIT panel...")
    panel = build_unlock_panel(windows_days=(20, 60))

    # 验证 OOS 期非零率
    print("\n[verify] OOS 期 unlock z 非零率:")
    oos_panel = panel[
        (panel["datetime"] >= "2021-05-01")
        & (panel["datetime"] <= "2026-04-30")
    ]
    for zc in ["z_unlock_pct_next_20", "z_unlock_pct_next_60",
               "z_unlock_imminent_20", "z_combo_neg_pct"]:
        nz = (oos_panel[zc] != 0).mean() * 100
        print(f"  {zc}: {nz:.2f}% nonzero (OOS)")

    print(f"\n[step 2] IS sweep {len(IS_COMBOS)} combos "
          f"({IS_FIRST} ~ {IS_LAST}, 48 months)")
    is_rows = []
    for label, factor, lam, sign in IS_COMBOS:
        print(f"\n  --- {label}  factor={factor} λ={lam} sign={sign:+d} ---")
        build_adjusted_predictions(panel, factor, lam, sign, label)
        stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{label}")
        row = {
            "combo": label,
            "factor": factor,
            "lam": lam,
            "sign": sign,
            "cum_%": stats["cum_%"],
            "ann_%": stats["ann_%"],
            "sharpe": stats["sharpe"],
            "mdd_%": stats["mdd_%"],
            "calmar": stats["calmar"],
            "win_%": stats["win_%"],
            "avg_picks": stats["avg_picks"],
            "n_months": stats["n"],
        }
        is_rows.append(row)
        print(f"    >> IS Calmar={stats['calmar']} Sharpe={stats['sharpe']} "
              f"ann={stats['ann_%']}% MDD={stats['mdd_%']}%")

    is_df = pd.DataFrame(is_rows)
    is_df = is_df.sort_values("calmar", ascending=False).reset_index(drop=True)
    is_df.to_csv(OUT_IS_GRID, index=False)
    print(f"\n[saved] {OUT_IS_GRID}")
    print("\n=== IS 9-combo Calmar table (sorted desc) ===")
    print(is_df.to_string(index=False))

    best = is_df.iloc[0]
    best_label = best["combo"]
    best_factor = best["factor"]
    best_lam = float(best["lam"])
    best_sign = int(best["sign"])
    print(f"\n[lock] best IS: {best_label}  "
          f"factor={best_factor} λ={best_lam} sign={best_sign:+d}  "
          f"IS Calmar={best['calmar']}")

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    print(f"  locked: {best_label} factor={best_factor} "
          f"λ={best_lam} sign={best_sign:+d}")
    build_adjusted_predictions(
        panel, best_factor, best_lam, best_sign, best_label,
    )
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_locked")
    oos_df = oos_stats.pop("months_df")
    oos_df.to_csv(OUT_OOS_STATS, index=False)
    print(f"\n[saved] {OUT_OOS_STATS}")

    print("\n" + "=" * 70)
    print("=== FINAL COMPARISON v19.9 ===")
    print("=" * 70)
    print("baseline train24 (Phase 2 clean_phase2 OOS, 60m):")
    print("   Calmar=0.42  Sharpe=0.68  ann=12.45%  MDD=-29.86%  cum=79.84%")
    print("\nv19.4 (margin sidecar) OOS:")
    print("   Calmar=0.61  Sharpe=0.76  ann=12.86%  MDD=-21.23%  cum=83.07%")
    print("\nv19.6 (amplitude sidecar) OOS:")
    print("   Calmar=0.79  Sharpe=0.71  ann=14.54%  MDD=-18.51%  cum=97.15%")
    print(f"\nv19.9 (unlock sidecar) OOS [{best_label}]:")
    print(f"   Calmar={oos_stats['calmar']}  Sharpe={oos_stats['sharpe']}  "
          f"ann={oos_stats['ann_%']}%  MDD={oos_stats['mdd_%']}%  "
          f"cum={oos_stats['cum_%']}%  win%={oos_stats['win_%']}")

    v196_calmar = 0.79
    rel_imp = (oos_stats["calmar"] - v196_calmar) / v196_calmar * 100
    print(f"\nrelative Calmar Δ vs v19.6: {rel_imp:+.1f}%")

    if oos_stats["calmar"] > v196_calmar:
        print("\n[verdict] OOS Calmar > v19.6 → 推荐升级 production v19.9")
    elif oos_stats["calmar"] > 0.42:
        print("\n[verdict] OOS Calmar < v19.6 但 > baseline → candidate, "
              "不建议替代 v19.6")
    else:
        print("\n[verdict] OOS Calmar < baseline → abort, unlock sidecar 无用")

    return 0


if __name__ == "__main__":
    sys.exit(main())
