"""v19.8 Sidecar — train24 + 龙虎榜 net_buy_pct_evt 单因子严格 OOS.

背景:
    技术因子探索发现 net_buy_pct_evt (龙虎榜净买占成交额比) IS ICIR +0.64
    (中等强, sign +1 正向). 数据已抓 data_cache/dragon_tiger/{code}.parquet
    (256/300 有 events, 3,465 events, 2014-2020 IS 期).

    本任务做独立 sidecar OOS 验证, 不与其它因子混叠.

数据缺口披露 (与 v19.4 一致, 不是 leak):
    dragon_tiger cache 仅含 2014-01~2020-12, OOS 2021-05~2026-04 有 0 事件.
    sidecar 中 net_buy_pct_evt 在 OOS 全为 0 → z 分量 0 → final = z(pred)
    退化为 baseline train24. 此为 sandbox 数据限制 (东财数据已抓到 IS 末尾).
    严格 OOS 协议依然执行: IS sweep λ → 锁定 → OOS 单跑, 若 OOS = baseline
    说明该因子的"OOS 提升能力"是 0, 这本身就是真实结论.

因子定义:
    net_buy_pct_evt          = net_amt / accum_amount * 100  (per event day)
                               非上榜日 fillna(0)
    net_buy_pct_evt_30d_avg  = rolling 30 trading-day mean of net_buy_pct_evt
                               (非上榜日参与平均, 等价 net_amt/accum sum 平摊)
    evt + 30d_avg 等权        = 0.5 * z(evt) + 0.5 * z(30d_avg)

Sidecar 公式 (因子 sign **正向**, ICIR +0.64):
    final_score = z(train24_pred) + λ × z(factor)         # 注意 + 不是 -

9 IS combo sweep (2017-01 ~ 2020-12, 48 月). 锁定 best IS Calmar 后 OOS
跑一次 (2021-05 ~ 2026-04, 60 月). **看 OOS 不允许回头改 λ** (CLAUDE.md
rule 5).

不动任何 production / pred cache / margin cache / DT cache / v19_3-7 文件.

Run:
  .venv/bin/python examples/strategy_v19_8_dragontiger.py
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

from claude_finance import dragon_tiger as dt_mod  # noqa: E402

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
CSI300_CSV = ROOT / "data_cache" / "csi300_constituents.csv"
DT_DIR = ROOT / "data_cache" / "dragon_tiger"
MARGIN_PARQUET = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
ADJ_PRED = ROOT / "data_cache" / "v19_8_dragontiger_predictions.parquet"
OUT_IS_GRID = ROOT / "examples" / "v19_8_dragontiger_is_grid.csv"
OUT_OOS_STATS = ROOT / "examples" / "v19_8_dragontiger_oos_stats.csv"

IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

# 9 IS combos: (label, factor_kind, lam)
# factor_kind:
#   'evt'  → z(net_buy_pct_evt)
#   '30d'  → z(net_buy_pct_evt_30d_avg)
#   'both' → 0.5 * z(evt) + 0.5 * z(30d)
IS_COMBOS = [
    ("c1_evt_l005",   "evt",  0.05),
    ("c2_evt_l010",   "evt",  0.10),
    ("c3_evt_l020",   "evt",  0.20),
    ("c4_evt_l030",   "evt",  0.30),
    ("c5_30d_l010",   "30d",  0.10),
    ("c6_30d_l020",   "30d",  0.20),
    ("c7_30d_l030",   "30d",  0.30),
    ("c8_both_l020",  "both", 0.20),
    ("c9_both_l030",  "both", 0.30),
]


def _zscore_cs(s: pd.Series) -> pd.Series:
    """Cross-section z-score within a group, NaN-safe."""
    mu = s.mean()
    sd = s.std()
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _instrument_to_code6(inst: str) -> str:
    return str(inst)[-6:].zfill(6)


def build_dragon_tiger_panel() -> pd.DataFrame:
    """构造 (datetime, instrument) × [z_evt, z_30d] 全期 panel.

    覆盖 IS+OOS 2017-01 ~ 2026-04. 在 pred 的 (datetime, instrument) axis
    上 join. 非上榜日 net_buy_pct_evt = 0; 30d_avg = 滚动 30 trading-day 平均.
    """
    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))
    print(f"[panel] pred dates: {len(pred_dt)}, "
          f"unique instruments: {pred['instrument'].nunique()}", flush=True)

    # === Load DT cache ===
    print("[panel] loading dragon_tiger cache...", flush=True)
    csi = pd.read_csv(CSI300_CSV, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    dt_parts = []
    n_with = n_empty = n_no_cache = 0
    for code in csi["code"]:
        p = DT_DIR / f"{code}.parquet"
        if not p.exists():
            n_no_cache += 1
            continue
        df = pd.read_parquet(p)
        if df.empty:
            n_empty += 1
            continue
        dt_parts.append(df)
        n_with += 1
    print(f"[panel] DT cache: {n_with} 有 events, "
          f"{n_empty} 空, {n_no_cache} 缺", flush=True)

    if dt_parts:
        dt_all = pd.concat(dt_parts, ignore_index=True)
        dt_all["code"] = dt_all["code"].astype(str).str.zfill(6)
        dt_all["date"] = pd.to_datetime(dt_all["date"]).dt.normalize()
        dt_daily = dt_mod.daily_features(dt_all)
        dt_min = dt_daily["date"].min()
        dt_max = dt_daily["date"].max()
        is_events = (
            (dt_daily["date"] >= "2017-01-01")
            & (dt_daily["date"] <= "2020-12-31")
        ).sum()
        oos_events = (
            (dt_daily["date"] >= "2021-05-01")
            & (dt_daily["date"] <= "2026-04-30")
        ).sum()
        print(f"[panel] DT daily events: {len(dt_daily):,}, "
              f"range {dt_min.date()} ~ {dt_max.date()}", flush=True)
        print(f"[panel] DT IS events 2017-01~2020-12: {is_events:,}",
              flush=True)
        print(f"[panel] DT OOS events 2021-05~2026-04: {oos_events:,} "
              "(0 → OOS degenerates to baseline train24)", flush=True)
    else:
        dt_daily = pd.DataFrame(columns=["code", "date", "net_buy_pct"])

    # === Build full (date × code) panel on pred axis ===
    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()

    # join event-day net_buy_pct
    if not dt_daily.empty:
        dt_evt = dt_daily[["date", "code", "net_buy_pct"]].rename(
            columns={"date": "datetime", "net_buy_pct": "net_buy_pct_evt"}
        )
        merged = pred_axis.merge(
            dt_evt, on=["datetime", "code"], how="left"
        )
    else:
        merged = pred_axis.copy()
        merged["net_buy_pct_evt"] = np.nan

    # event-driven: NaN on non-event day → 0
    merged["net_buy_pct_evt"] = merged["net_buy_pct_evt"].fillna(0.0)

    # rolling 30 trading-day avg per code on pred axis
    merged = merged.sort_values(["code", "datetime"]).reset_index(drop=True)
    grp = merged.groupby("code", sort=False)
    merged["net_buy_pct_evt_30d_avg"] = grp["net_buy_pct_evt"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )

    # coverage 报告
    n_total = len(merged)
    n_evt_nonzero = (merged["net_buy_pct_evt"] != 0).sum()
    n_30d_nonzero = (merged["net_buy_pct_evt_30d_avg"] != 0).sum()
    print(f"[panel] coverage out of {n_total:,} pred rows:")
    print(f"        net_buy_pct_evt≠0       : {n_evt_nonzero:,} "
          f"({n_evt_nonzero/n_total*100:.2f}%)")
    print(f"        net_buy_pct_evt_30d≠0   : {n_30d_nonzero:,} "
          f"({n_30d_nonzero/n_total*100:.2f}%)")

    # cross-section z-score per date
    print("[panel] computing cross-section z-scores...", flush=True)
    merged["z_evt"] = merged.groupby("datetime")[
        "net_buy_pct_evt"
    ].transform(_zscore_cs)
    merged["z_30d"] = merged.groupby("datetime")[
        "net_buy_pct_evt_30d_avg"
    ].transform(_zscore_cs)
    merged[["z_evt", "z_30d"]] = merged[["z_evt", "z_30d"]].fillna(0.0)

    return merged[["datetime", "instrument", "z_evt", "z_30d"]]


def build_adjusted_predictions(panel: pd.DataFrame, factor_kind: str,
                               lam: float, label: str) -> Path:
    """构造 sidecar parquet.

        evt  → final = z(pred) + λ × z_evt
        30d  → final = z(pred) + λ × z_30d
        both → final = z(pred) + λ × (0.5 z_evt + 0.5 z_30d)
    """
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(panel, on=["datetime", "instrument"], how="left")
    merged[["z_evt", "z_30d"]] = merged[["z_evt", "z_30d"]].fillna(0.0)

    if factor_kind == "evt":
        z_combo = merged["z_evt"]
    elif factor_kind == "30d":
        z_combo = merged["z_30d"]
    elif factor_kind == "both":
        z_combo = 0.5 * merged["z_evt"] + 0.5 * merged["z_30d"]
    else:
        raise ValueError(f"unknown factor_kind={factor_kind!r}")

    # sign +1 正向: + λ × z(factor)
    merged["final_score"] = merged["z_pred"] + lam * z_combo

    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
    print(f"  [adj] {label} kind={factor_kind} λ={lam} "
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
    """跑指定 [first, last] 区间的 walk-forward backtest, 返回 stats."""
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


def compute_spearman_corr(panel: pd.DataFrame) -> pd.DataFrame:
    """IS-period Spearman corr between net_buy_pct_evt and amp_imb_20d /
    margin_5d_chg. 防 future stacked overfit 检查 (本任务不 stack)."""
    from _factor_kline_panel import build_pit_panel_on_pred_axis

    print("\n[spearman] computing IS-period Spearman corr...", flush=True)
    amp_panel = build_pit_panel_on_pred_axis(
        ORIG_PRED, factor_cols=["amp_imb_20d"],
    )

    margin = pd.read_parquet(MARGIN_PARQUET)
    margin["code"] = margin["code"].astype(str).str.zfill(6)
    margin["date"] = pd.to_datetime(margin["date"])
    margin = margin[["code", "date", "margin_5d_chg"]]

    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred = pred[(pred["datetime"] >= "2017-01-01") &
                (pred["datetime"] <= "2020-12-31")]
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))

    m_parts = []
    for code, sub in margin.groupby("code", sort=False):
        sub = sub.sort_values("date")
        dates_arr = sub["date"].values
        m5 = sub["margin_5d_chg"].values
        idx = np.searchsorted(dates_arr, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe_idx = np.clip(idx, 0, len(sub) - 1)
        m5_v = np.where(valid, m5[safe_idx], np.nan)
        m_parts.append(pd.DataFrame({
            "datetime": pred_dt, "code": code, "margin_5d_chg": m5_v,
        }))
    margin_panel = pd.concat(m_parts, ignore_index=True)

    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    df = pred_axis.merge(panel, on=["datetime", "instrument"], how="left")
    df = df.merge(amp_panel, on=["datetime", "instrument"], how="left")
    df = df.merge(margin_panel, on=["datetime", "code"], how="left")

    cols = ["z_evt", "z_30d", "amp_imb_20d", "margin_5d_chg"]
    df = df[cols].dropna(how="all")
    corr = df.corr(method="spearman")
    print(corr.round(3).to_string())
    return corr


def main() -> int:
    print("=" * 70)
    print("Phase 4 Sidecar v19.8 — train24 + 龙虎榜 net_buy_pct_evt 单因子")
    print("=" * 70)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST} (48 months)")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST} (60 months)")
    print(f"combos: {len(IS_COMBOS)}")
    print()

    print("[step 1] build dragon_tiger PIT panel...")
    panel = build_dragon_tiger_panel()

    print("\n[step 2] IS sweep 9 combos (2017-01 ~ 2020-12, 48 months)")
    is_rows = []
    for label, factor_kind, lam in IS_COMBOS:
        print(f"\n  --- {label}  kind={factor_kind}  λ={lam} ---")
        build_adjusted_predictions(panel, factor_kind, lam, label)
        stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{label}")
        row = {
            "combo": label,
            "factor_kind": factor_kind,
            "lam": lam,
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
    best_kind = best["factor_kind"]
    best_lam = float(best["lam"])
    print(f"\n[lock] best IS: {best_label}  kind={best_kind} λ={best_lam}  "
          f"IS Calmar={best['calmar']}")

    print(f"\n[step 3] OOS single run ({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    print(f"  locked config: {best_label} kind={best_kind} λ={best_lam}")
    print("  WARN: DT cache 仅到 2020-12, OOS 期 0 events → degenerate "
          "to baseline train24", flush=True)
    build_adjusted_predictions(panel, best_kind, best_lam, best_label)
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, "OOS_locked")
    oos_df = oos_stats.pop("months_df")
    oos_df.to_csv(OUT_OOS_STATS, index=False)
    print(f"\n[saved] {OUT_OOS_STATS}")

    # === Spearman 相关性 (跟 v19.6 amp & v19.4 margin) ===
    try:
        compute_spearman_corr(panel)
    except Exception as e:
        print(f"[spearman] failed: {e}")

    print("\n" + "=" * 70)
    print("=== FINAL COMPARISON v19.8 ===")
    print("=" * 70)
    print("baseline train24 (Phase 2 clean_phase2 OOS, 60m):")
    print("   Calmar=0.42  Sharpe=0.68  ann=12.45%  MDD=-29.86%  cum=79.84%")
    print("\nv19.4 (margin sidecar) OOS:")
    print("   Calmar=0.61  Sharpe=0.76  ann=12.86%  MDD=-21.23%  cum=83.07%")
    print("\nv19.6 (amplitude sidecar) OOS:")
    print("   Calmar=0.79  Sharpe=0.71  ann=14.54%  MDD=-18.51%")
    print("\nv19.7 (a20+m5 stacked) OOS:")
    print("   Calmar=0.65  Sharpe=0.69  ann=12.23%  MDD=-18.72%")
    print(f"\nv19.8 (net_buy_pct_evt sidecar) OOS [{best_label}]:")
    print(f"   Calmar={oos_stats['calmar']}  Sharpe={oos_stats['sharpe']}  "
          f"ann={oos_stats['ann_%']}%  MDD={oos_stats['mdd_%']}%  "
          f"cum={oos_stats['cum_%']}%  win%={oos_stats['win_%']}")

    v196_calmar = 0.79
    baseline_calmar = 0.42
    rel_v196 = (oos_stats["calmar"] - v196_calmar) / v196_calmar * 100
    rel_baseline = (
        (oos_stats["calmar"] - baseline_calmar) / baseline_calmar * 100
    )
    print(f"\nrelative Calmar Δ vs v19.6: {rel_v196:+.1f}%")
    print(f"relative Calmar Δ vs baseline: {rel_baseline:+.1f}%")

    if oos_stats["calmar"] > v196_calmar:
        print("\n[verdict] OOS Calmar > v19.6 (0.79) → 推荐升级 production "
              "v19.8")
    elif oos_stats["calmar"] > baseline_calmar:
        print("\n[verdict] OOS Calmar > baseline (0.42) 但 ≤ v19.6 → 作 "
              "candidate sidecar, 不升级")
    else:
        print("\n[verdict] OOS Calmar ≤ baseline → abort, "
              "net_buy_pct_evt sidecar OOS 失效 (DT 数据缺口 + 因子无 OOS 力)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
