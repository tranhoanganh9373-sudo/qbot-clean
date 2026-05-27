"""Phase B 严格 60 月 OOS 测试 — 4 candidates 各独立跑.

跨 3 base IC 评估后通过 robust 筛选的 4 个 candidates:

  candidate          sign  source            IS hfq ICIR  n_months
  obv_slope_60d      -1    classic alpha     +1.53        55
  idio_vol_20d       -1    classic alpha     +3.58        58
  竞价勾魂翻         +1    mt180 TDX         +1.71        59
  金蛇量能F          -1    mt180 TDX         -1.93        59

严格协议 (per `feedback_strict_oos_backtest`):
  - IS 锁 λ: 2017-01 ~ 2020-12 (48 months, predictions cache 起点 2017-01,
    严格 pre-OOS) sweep λ ∈ {0.10, 0.20, 0.30} → pick best IS Calmar lock
  - OOS 单跑: 2021-05 ~ 2026-04 (60 months, 用 IS-locked λ)
  - sidecar 公式: final_score = z(pred) + sign × λ × z(factor)
    sign 约定: factor IC ICIR > 0 → sign=+1 (高 factor → 高 fwd ret, 同向叠加);
              factor IC ICIR < 0 → sign=-1 (高 factor → 低 fwd ret, 反向叠加).
  - 各 candidate **独立** 运行, 不 batch、不混 λ.

与 v19.6 历史习惯对齐:
  v19.6 公式: final = z(pred) - 0.30 × z(amp_imb_20d), amp sign=-1
  本任务统一: final = z(pred) + sign × λ × z(factor)
  ⇒ sign=-1 等价: final = z(pred) - λ × z(factor)  (matches v19.6 / v19.4 习惯)

Inputs:
  data_cache/v17_dens_train24_predictions.parquet  (Phase 2 v3 retrained 主表只读)
  data_cache/baidu_kline.parquet                   (qfq, 跟 production 一致)
  data_cache/mt180/indicators_detail.jsonl         (TDX formulas)
  data_cache/qlib_baidu/                            (qlib provider)

Outputs (under data_cache/factors/):
  phase_b_60m_oos_results.csv          (per-candidate summary row)
  phase_b_60m_oos_{candidate}_is.csv   (IS sweep grid, per candidate)
  phase_b_60m_oos_{candidate}_oos.csv  (per-month OOS equity, per candidate)
  phase_b_60m_oos_summary.md           (markdown report)

约束:
  - 不动 production paper_trade/forward_oos_monitor/strategy_v19*.py/portfolio*.py
  - 不动 v17_dens_train24_predictions.parquet 主表 (只读)
  - λ ∈ {0.10, 0.20, 0.30} 三选一 lock 后不再调
  - 严格 single OOS run
  - 不 commit
"""
from __future__ import annotations

import json
import sys
import time
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

from claude_finance.tdx_parser import compile_tdx  # noqa: E402
from _factor_kline_panel import _zscore_cs, _instrument_to_code6  # noqa: E402

ORIG_PRED = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
KLINE_QFQ = ROOT / "data_cache" / "baidu_kline.parquet"
INDICATORS_DETAIL = ROOT / "data_cache" / "mt180" / "indicators_detail.jsonl"
QLIB_DIR = ROOT / "data_cache" / "qlib_baidu"
ADJ_PRED = ROOT / "data_cache" / "phase_b_60m_adj_predictions.parquet"  # scratch

OUT_DIR = ROOT / "data_cache" / "factors"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SUMMARY_CSV = OUT_DIR / "phase_b_60m_oos_results.csv"
OUT_SUMMARY_MD = OUT_DIR / "phase_b_60m_oos_summary.md"

# IS / OOS windows (aligned with all v19.x / v20.x history)
IS_FIRST = "2017-01"
IS_LAST = "2020-12"
OOS_FIRST = "2021-05"
OOS_LAST = "2026-04"

LAMBDAS = [0.10, 0.20, 0.30]

# Per-candidate spec
CANDIDATES = [
    {
        "factor_id": "obv_slope_60d",
        "name": "obv_slope_60d",
        "sign": -1,
        "source": "classic",
    },
    {
        "factor_id": "idio_vol_20d",
        "name": "idio_vol_20d",
        "sign": -1,
        "source": "classic",
    },
    {
        "factor_id": "jingjia_gouhun_fan",
        "name": "竞价勾魂翻",
        "sign": +1,
        "source": "tdx",
        "tdx_id": "b9c99f5c-6e8d-4a8b-818f-0b48a023effe",
        "tdx_output_col": "开盘涨幅",
    },
    {
        "factor_id": "jinshe_liangneng_F",
        "name": "金蛇量能F",
        "sign": -1,
        "source": "tdx",
        "tdx_id": "eec80502-62b6-411c-b5f9-12a19c40cce5",
        "tdx_output_col": "量能",
    },
]


# ==========================================================================
# Factor computation
# ==========================================================================

def _load_pred_universe() -> tuple[pd.DatetimeIndex, list[str]]:
    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))
    codes = sorted(pred["instrument"].apply(_instrument_to_code6).unique())
    return pred_dt, codes


def _load_qfq_kline(codes: set[str], min_date: str = "2014-01-01",
                     max_date: str = "2026-12-31") -> pd.DataFrame:
    print(f"[kline] loading qfq baidu_kline.parquet (codes ∩ pred={len(codes)})...",
          flush=True)
    k = pd.read_parquet(
        KLINE_QFQ,
        columns=["code", "date", "open", "high", "low", "close", "vol", "amount"],
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    k["date"] = pd.to_datetime(k["date"])
    k = k[(k["date"] >= pd.Timestamp(min_date)) & (k["date"] <= pd.Timestamp(max_date))]
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"[kline] rows={len(k):,}  codes={k['code'].nunique()}  "
          f"{k['date'].min().date()} ~ {k['date'].max().date()}", flush=True)
    return k


def _obv_slope_60d(sub: pd.DataFrame) -> pd.Series:
    diff = sub["close"].diff()
    sign = np.sign(diff.fillna(0))
    obv = (sign * sub["vol"]).cumsum().to_numpy()
    n = 60
    L = len(obv)
    if L < n:
        return pd.Series(np.full(L, np.nan), index=sub.index)
    x = np.arange(n, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = n * sum_x2 - sum_x * sum_x
    sum_y_roll = np.convolve(obv, np.ones(n)[::-1], mode="valid")
    sum_xy_roll = np.convolve(obv, x[::-1], mode="valid")
    slope_valid = (n * sum_xy_roll - sum_x * sum_y_roll) / denom
    out = np.full(L, np.nan)
    out[n - 1:] = slope_valid
    return pd.Series(out, index=sub.index)


def compute_classic_factor(kline: pd.DataFrame, factor_id: str) -> pd.DataFrame:
    """Return DataFrame [code, date, factor]."""
    k = kline.copy()
    grp = k.groupby("code", sort=False, group_keys=False)

    if factor_id == "obv_slope_60d":
        k["factor"] = grp.apply(_obv_slope_60d)
    elif factor_id == "idio_vol_20d":
        k["_ret"] = grp["close"].pct_change()
        xs_mean = k.groupby("date")["_ret"].transform("mean")
        k["_ret_resid"] = k["_ret"] - xs_mean
        k["factor"] = k.groupby("code", sort=False, group_keys=False)[
            "_ret_resid"
        ].transform(lambda s: s.rolling(20, min_periods=20).std())
        k = k.drop(columns=["_ret", "_ret_resid"])
    else:
        raise ValueError(f"unknown classic factor: {factor_id}")

    k["factor"] = k["factor"].replace([np.inf, -np.inf], np.nan)
    return k[["code", "date", "factor"]]


def compute_tdx_factor(kline: pd.DataFrame, tdx_id: str,
                        output_col: str) -> pd.DataFrame:
    """Compile + eval TDX formula per code. Return [code, date, factor]."""
    formula = None
    with INDICATORS_DETAIL.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"] == tdx_id:
                formula = rec["formula"]
                break
    if formula is None:
        raise RuntimeError(f"tdx_id {tdx_id} not found in indicators_detail.jsonl")

    cf = compile_tdx(formula)
    if cf.status != "ok":
        raise RuntimeError(f"compile_tdx status={cf.status} err={cf.error}")
    if output_col not in cf.output_cols:
        raise RuntimeError(
            f"output_col {output_col!r} not in {cf.output_cols!r}"
        )

    parts: list[pd.DataFrame] = []
    n_codes = kline["code"].nunique()
    print(f"[tdx] evaluating {tdx_id[:8]}/{output_col} across {n_codes} codes...",
          flush=True)
    t0 = time.time()
    for i, (code, sub) in enumerate(kline.groupby("code", sort=False), 1):
        sub_idx = sub.set_index("date")
        try:
            results = cf(sub_idx)
            if not results or output_col not in results:
                continue
            fact = results[output_col]
            df = pd.DataFrame({
                "code": code,
                "date": sub_idx.index,
                "factor": fact.values,
            })
            parts.append(df)
        except Exception:
            continue
        if i % 100 == 0:
            print(f"  [tdx] {i}/{n_codes}  elapsed={time.time()-t0:.1f}s",
                  flush=True)
    if not parts:
        return pd.DataFrame(columns=["code", "date", "factor"])
    out = pd.concat(parts, ignore_index=True)
    out["factor"] = out["factor"].replace([np.inf, -np.inf], np.nan)
    print(f"[tdx] {output_col}: panel rows={len(out):,}, "
          f"non-null={out['factor'].notna().sum():,} "
          f"({out['factor'].notna().mean()*100:.1f}%)  "
          f"elapsed={time.time()-t0:.1f}s", flush=True)
    return out


def build_pit_panel(factor_panel: pd.DataFrame,
                    pred_dt: pd.DatetimeIndex) -> pd.DataFrame:
    """Convert (code, date, factor) → (datetime ∈ pred_dt, instrument, z_factor).

    PIT join: for each pred date, take latest factor value with factor.date <= pred.dt.
    Cross-section z-score per pred date. fillna(0).
    """
    factor_panel = factor_panel.sort_values(["code", "date"])
    parts = []
    for code, sub in factor_panel.groupby("code", sort=False):
        dates_arr = sub["date"].values
        idx = np.searchsorted(dates_arr, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe_idx = np.clip(idx, 0, len(sub) - 1)
        v = sub["factor"].values
        f_arr = np.where(valid, v[safe_idx], np.nan)
        parts.append(pd.DataFrame({
            "datetime": pred_dt,
            "code": code,
            "factor": f_arr,
        }))
    if not parts:
        raise RuntimeError("[panel] no PIT panel rows built")
    panel = pd.concat(parts, ignore_index=True)

    pred = pd.read_parquet(ORIG_PRED, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    out = pred_axis.merge(panel, on=["datetime", "code"], how="left")

    n_total = len(out)
    n_ok = out["factor"].notna().sum()
    print(f"[panel] PIT panel rows={n_total:,}, non-null={n_ok:,} "
          f"({n_ok/n_total*100:.1f}%)", flush=True)

    out["z_factor"] = out.groupby("datetime")["factor"].transform(_zscore_cs)
    out["z_factor"] = out["z_factor"].fillna(0.0)
    return out[["datetime", "instrument", "z_factor"]]


# ==========================================================================
# Sidecar + walkforward
# ==========================================================================

def write_adjusted_pred(z_panel: pd.DataFrame, lam: float, sign: int) -> Path:
    """final_score = z(pred) + sign × λ × z(factor)."""
    pred = pd.read_parquet(ORIG_PRED)
    pred["z_pred"] = pred.groupby("datetime")["score"].transform(_zscore_cs)
    merged = pred.merge(z_panel, on=["datetime", "instrument"], how="left")
    merged["z_factor"] = merged["z_factor"].fillna(0.0)
    merged["final_score"] = merged["z_pred"] + sign * lam * merged["z_factor"]

    out = merged[["datetime", "instrument", "month"]].copy()
    out["score"] = merged["final_score"]
    out = out[["datetime", "instrument", "score", "month"]]
    out.to_parquet(ADJ_PRED, index=False)
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


_v17 = None
_proxy = None
_qlib_inited = False


def _ensure_qlib():
    global _v17, _proxy, _qlib_inited
    if _qlib_inited:
        return
    import qlib  # noqa: F401
    from qlib.constant import REG_CN
    import strategy_v17_dens_grid as v17

    qlib.init(provider_uri=str(QLIB_DIR), region=REG_CN)
    _proxy = v17.build_market_proxy()
    v17.MARKET = "csi300"
    v17.TRAIN_MONTHS = 24
    v17.K_NORMAL = 8
    v17.DROP_NORMAL = 2
    v17.PORTFOLIO_VALUE = 5e4
    v17.STOP_LOSS_PCT = 0.0
    v17.VOL_TARGET_ANN = 0.0
    _v17 = v17
    _qlib_inited = True


def run_walkforward(first_month: str, last_month: str, tag: str) -> dict:
    _ensure_qlib()
    _v17.PRED_CACHE = ADJ_PRED
    _v17._pred_disk_df = None
    _v17._pred_cache.clear()

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
            res = _v17.realistic_window(m, _proxy, with_regime=False)
            res["month"] = m.strftime("%Y-%m")
            rows.append(res)
        except Exception as e:
            print(f"    [{tag}] {i}/{len(months)} {m.strftime('%Y-%m')} FAIL: "
                  f"{str(e)[:120]}", flush=True)
            rows.append({"month": m.strftime("%Y-%m"),
                         "abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                         "regime_days": "", "n_skipped_limit": 0,
                         "n_stop_loss": 0})
        if i == 1 or i % 12 == 0 or i == len(months):
            print(f"    [{tag}] {i:3d}/{len(months)} {rows[-1]['month']}: "
                  f"abs_ret={rows[-1]['abs_ret_%']:+6.2f}%  "
                  f"picks={rows[-1]['avg_picks']:.1f}", flush=True)

    df = pd.DataFrame(rows)
    stats = _annualize(df["abs_ret_%"])
    stats["avg_picks"] = round(df["avg_picks"].mean(), 2)
    stats["months_df"] = df
    return stats


# ==========================================================================
# Per-candidate pipeline
# ==========================================================================

def run_candidate(cand: dict, kline_qfq: pd.DataFrame,
                   pred_dt: pd.DatetimeIndex) -> dict:
    fid = cand["factor_id"]
    name = cand["name"]
    sign = cand["sign"]
    source = cand["source"]

    print("\n" + "=" * 72)
    print(f"[candidate] {fid}  ({name})  sign={sign:+d}  source={source}")
    print("=" * 72)

    t0 = time.time()
    if source == "classic":
        factor_panel = compute_classic_factor(kline_qfq, fid)
    elif source == "tdx":
        factor_panel = compute_tdx_factor(
            kline_qfq, cand["tdx_id"], cand["tdx_output_col"]
        )
    else:
        raise ValueError(f"unknown source: {source}")
    print(f"[factor] computed in {time.time()-t0:.1f}s. "
          f"non-null={factor_panel['factor'].notna().sum():,}", flush=True)

    z_panel = build_pit_panel(factor_panel, pred_dt)

    # IS sweep
    print(f"\n[step IS] sweep λ ∈ {LAMBDAS}  ({IS_FIRST} ~ {IS_LAST}, 48 months)")
    is_rows = []
    for lam in LAMBDAS:
        print(f"\n  --- λ={lam} sign={sign:+d} ---")
        write_adjusted_pred(z_panel, lam, sign)
        stats = run_walkforward(IS_FIRST, IS_LAST, f"IS_{fid}_l{lam:.2f}")
        stats.pop("months_df", None)
        row = {"lambda": lam, "sign": sign, **stats}
        is_rows.append(row)
        print(f"    >> IS  Calmar={stats['calmar']} Sharpe={stats['sharpe']} "
              f"ann={stats['ann_%']}% MDD={stats['mdd_%']}%", flush=True)

    is_df = pd.DataFrame(is_rows).sort_values("calmar", ascending=False)
    is_csv = OUT_DIR / f"phase_b_60m_oos_{fid}_is.csv"
    is_df.to_csv(is_csv, index=False)
    print(f"\n[saved IS] {is_csv}")

    best = is_df.iloc[0]
    locked_lambda = float(best["lambda"])
    is_calmar = float(best["calmar"])
    is_sharpe = float(best["sharpe"])
    print(f"\n[lock] best IS λ={locked_lambda}  Calmar={is_calmar}  Sharpe={is_sharpe}")

    # OOS single run
    print(f"\n[step OOS] single run λ={locked_lambda} "
          f"({OOS_FIRST} ~ {OOS_LAST}, 60 months)")
    write_adjusted_pred(z_panel, locked_lambda, sign)
    oos_stats = run_walkforward(OOS_FIRST, OOS_LAST, f"OOS_{fid}")
    oos_df = oos_stats.pop("months_df")
    oos_csv = OUT_DIR / f"phase_b_60m_oos_{fid}_oos.csv"
    oos_df.to_csv(oos_csv, index=False)
    print(f"[saved OOS] {oos_csv}")

    print(f"\n[result] {fid}  λ={locked_lambda}  sign={sign:+d}")
    print(f"  IS  Calmar={is_calmar}  Sharpe={is_sharpe}")
    print(f"  OOS Calmar={oos_stats['calmar']}  Sharpe={oos_stats['sharpe']}  "
          f"ann={oos_stats['ann_%']}%  MDD={oos_stats['mdd_%']}%  "
          f"cum={oos_stats['cum_%']}%")

    return {
        "candidate": fid,
        "name": name,
        "sign": sign,
        "source": source,
        "locked_lambda": locked_lambda,
        "is_calmar": is_calmar,
        "is_sharpe": is_sharpe,
        "is_ann_%": float(best["ann_%"]),
        "is_mdd_%": float(best["mdd_%"]),
        "oos_calmar": oos_stats["calmar"],
        "oos_sharpe": oos_stats["sharpe"],
        "oos_ann_%": oos_stats["ann_%"],
        "oos_mdd_%": oos_stats["mdd_%"],
        "oos_cum_%": oos_stats["cum_%"],
        "oos_win_%": oos_stats["win_%"],
        "oos_n_months": oos_stats["n"],
        "oos_avg_picks": oos_stats["avg_picks"],
    }


# ==========================================================================
# Reference benchmarks (from MEMORY, used for comparison)
# ==========================================================================

REFERENCE_BENCHMARKS = {
    "baseline_v3": {"calmar": 0.77, "sharpe": None, "ann": 23.02, "mdd": -30.01,
                    "cum": None, "label": "baseline train24 (no sidecar)"},
    "v19_4_v3":   {"calmar": 0.62, "sharpe": None, "ann": 19.35, "mdd": -31.21,
                    "cum": None, "label": "v19.4 m5+m20 λ=0.10 (shadow)"},
    "v19_6_v3":   {"calmar": 1.29, "sharpe": 0.92, "ann": 34.36, "mdd": -26.66,
                    "cum": 337.80, "label": "v19.6 amp_imb_20d λ=0.30 (PRODUCTION)"},
}


def _verdict(oos_calmar: float, v19_6: float = 1.29) -> str:
    if oos_calmar > v19_6:
        return "BEAT v19.6 — consider production upgrade"
    if oos_calmar > 0.77:
        return "above v3 baseline but below v19.6"
    if oos_calmar > 0.50:
        return "marginal — below baseline, abort"
    if oos_calmar > 0.0:
        return "weak — clear abort"
    return "negative — disastrous abort"


def build_markdown_report(df: pd.DataFrame, wall_sec: float) -> str:
    lines: list[str] = []
    lines.append("# Phase B 严格 60 月 OOS — 4 candidates 综合报告")
    lines.append("")
    lines.append(f"- wall time: **{wall_sec/60:.1f} min**")
    lines.append(f"- IS window : {IS_FIRST} ~ {IS_LAST} (48 months, "
                 f"λ sweep ∈ {LAMBDAS})")
    lines.append(f"- OOS window: {OOS_FIRST} ~ {OOS_LAST} (60 months, "
                 f"locked λ, single run)")
    lines.append(f"- pred cache: `{ORIG_PRED.name}` (Phase 2 v3 retrained, 主表只读)")
    lines.append(f"- kline    : `{KLINE_QFQ.name}` (qfq, production-consistent)")
    lines.append(f"- sidecar  : `final = z(pred) + sign × λ × z(factor)`")
    lines.append("")
    lines.append("## Reference benchmarks (Phase 2 v3 cache)")
    lines.append("")
    lines.append("| ref | Calmar | Sharpe | ann | MDD |")
    lines.append("|---|---|---|---|---|")
    for _, v in REFERENCE_BENCHMARKS.items():
        cal = v["calmar"]
        sh = v["sharpe"] if v["sharpe"] is not None else "—"
        lines.append(f"| {v['label']} | {cal} | {sh} | {v['ann']}% | {v['mdd']}% |")
    lines.append("")
    lines.append("## Candidates OOS 60m single-run results")
    lines.append("")
    lines.append("| candidate | sign | λ | IS Calmar | OOS Calmar | OOS Sharpe | "
                 "OOS ann% | OOS MDD% | OOS cum% | Δ vs baseline | Δ vs v19.6 | "
                 "verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    if "oos_calmar" not in df.columns:
        lines.append("| (no successful runs) | | | | | | | | | | | |")
    else:
        for _, r in df.iterrows():
            if "error" in r and pd.notna(r.get("error")):
                lines.append(f"| {r['name']} | {r['sign']} | — | — | "
                             f"ERROR: {r['error'][:60]} | | | | | | | |")
                continue
            lines.append(
                f"| {r['name']} | {int(r['sign']):+d} | {r['locked_lambda']:.2f} | "
                f"{r['is_calmar']} | **{r['oos_calmar']}** | {r['oos_sharpe']} | "
                f"{r['oos_ann_%']}% | {r['oos_mdd_%']}% | {r['oos_cum_%']}% | "
                f"{r['delta_vs_baseline_%']:+.1f}% | {r['delta_vs_v19_6_%']:+.1f}% | "
                f"{r['verdict']} |"
            )
    lines.append("")
    lines.append("## 历史 Phase B 5 次 abort 比较 (per MEMORY)")
    lines.append("")
    lines.append("| run | n_months | IS Calmar | OOS Calmar | IS→OOS Δ% |")
    lines.append("|---|---|---|---|---|")
    lines.append("| v19.7 stacked (a20+m5)          | 36 | 0.76 |  0.65 | -14.5%  |")
    lines.append("| v19.9 unlock (combo_neg λ=0.30) | 36 | 2.70 |  0.09 | -96.7%  |")
    lines.append("| v20 super_big_net λ=0.30        | 36 | 2.01 | -0.07 | -103.5% |")
    lines.append("| v20 shareholders λ=0.30         | 49 | 6.09 |  0.39 | -93.6%  |")
    lines.append("| v20 volume_z_5d λ=0.30 (v3)     | 36 | 3.39 |  0.32 | -90.6%  |")
    lines.append("")
    if "oos_calmar" in df.columns:
        lines.append("### This batch (n_months 55-59):")
        for _, r in df.iterrows():
            if "error" in r and pd.notna(r.get("error")):
                continue
            decay = (r["oos_calmar"] - r["is_calmar"]) / r["is_calmar"] * 100 \
                if r["is_calmar"] > 0 else 0
            lines.append(
                f"- **{r['name']}** (n_months IC≈55-59): IS={r['is_calmar']} → "
                f"OOS={r['oos_calmar']} (Δ {decay:+.1f}%)"
            )
        lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    if "oos_calmar" not in df.columns:
        lines.append("- 所有候选 OOS 失败,维持 v19.6 production")
    else:
        winners = df[df["oos_calmar"] > 1.29]
        above_baseline = df[(df["oos_calmar"] > 0.77) & (df["oos_calmar"] <= 1.29)]
        lines.append(f"- 击败 v19.6 (Calmar > 1.29): **{len(winners)}** 个候选")
        for _, r in winners.iterrows():
            lines.append(f"  - {r['name']}: OOS Calmar={r['oos_calmar']}, "
                         f"{r['delta_vs_v19_6_%']:+.1f}% vs v19.6")
        lines.append(f"- 高于 v3 baseline (0.77) 但低于 v19.6: **{len(above_baseline)}** 个")
        for _, r in above_baseline.iterrows():
            lines.append(f"  - {r['name']}: OOS Calmar={r['oos_calmar']}")
        lines.append("")
        if len(winners) > 0:
            top = winners.sort_values("oos_calmar", ascending=False).iloc[0]
            lines.append(f"**推荐**: 评估 {top['name']} (OOS Calmar={top['oos_calmar']}) "
                         f"作为 production v19.6 升级候选 (**用户决定**, 本任务 "
                         f"production 全 untouched).")
        else:
            lines.append("**推荐**: 维持 v19.6 production (Calmar=1.29), "
                         "所有候选未能击败 v19.6.")
    lines.append("")
    return "\n".join(lines)


# ==========================================================================
# Main
# ==========================================================================

def main() -> int:
    overall_t0 = time.time()

    print("=" * 72)
    print("Phase B 严格 60 月 OOS — 4 candidates (单跑, 不 batch)")
    print("=" * 72)
    print(f"IS  : {IS_FIRST} ~ {IS_LAST}  (48 months, λ sweep ∈ {LAMBDAS})")
    print(f"OOS : {OOS_FIRST} ~ {OOS_LAST}  (60 months, λ locked, single run)")
    print(f"Predictions: {ORIG_PRED.name}  (Phase 2 v3 retrained, read-only)")
    print(f"Kline      : {KLINE_QFQ.name}  (qfq base, production-consistent)")
    print(f"Sidecar    : final = z(pred) + sign × λ × z(factor)")
    print(f"Candidates : {[c['name'] for c in CANDIDATES]}")
    print()

    pred_dt, pred_codes = _load_pred_universe()
    kline_qfq = _load_qfq_kline(set(pred_codes), min_date="2014-01-01")

    summary_rows: list[dict] = []
    for cand in CANDIDATES:
        try:
            res = run_candidate(cand, kline_qfq, pred_dt)
            summary_rows.append(res)
        except Exception as exc:
            print(f"\n[FAIL] candidate {cand['factor_id']}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            summary_rows.append({
                "candidate": cand["factor_id"],
                "name": cand["name"],
                "sign": cand["sign"],
                "source": cand["source"],
                "error": str(exc)[:200],
            })

    summary_df = pd.DataFrame(summary_rows)
    baseline = REFERENCE_BENCHMARKS["baseline_v3"]["calmar"]
    v196 = REFERENCE_BENCHMARKS["v19_6_v3"]["calmar"]
    if "oos_calmar" in summary_df.columns:
        summary_df["delta_vs_baseline_%"] = (
            (summary_df["oos_calmar"] - baseline) / baseline * 100
        ).round(1)
        summary_df["delta_vs_v19_6_%"] = (
            (summary_df["oos_calmar"] - v196) / v196 * 100
        ).round(1)
        summary_df["verdict"] = summary_df["oos_calmar"].apply(_verdict)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\n[saved summary CSV] {OUT_SUMMARY_CSV}")

    wall = time.time() - overall_t0
    md = build_markdown_report(summary_df, wall)
    OUT_SUMMARY_MD.write_text(md, encoding="utf-8")
    print(f"[saved summary MD]  {OUT_SUMMARY_MD}")

    print()
    print("=" * 72)
    print(f"DONE. wall time = {wall/60:.1f} min")
    print("=" * 72)
    print(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
