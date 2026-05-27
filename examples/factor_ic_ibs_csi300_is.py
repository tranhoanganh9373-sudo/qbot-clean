"""CSI300 IBS (Internal Bar Strength) 多 horizon 因子 IS IC 分析 — 严格 OOS 协议.

任务: 在 IS 期 2014-2020 (84 月) 评估 IBS 各 horizon 的 alpha,选 Phase B
候选,严格不读 OOS 期 (2021-01+)。

候选因子 (5 horizon × 2 sign = 10 项):
  IBS_1d         单日 IBS (T-1)
  IBS_5d_mean    5d backward rolling mean
  IBS_10d_mean   10d backward rolling mean
  IBS_20d_mean   20d backward rolling mean
  IBS_60d_mean   60d backward rolling mean

数据源:
  data_cache/baidu_kline.parquet         (high/low/close, hfq)
  data_cache/csi300_constituents.csv     (300 codes)
  data_cache/csi300_margin_14yr.parquet  (Spearman vs m5/m20)

方法 (point-in-time, no lookahead):
  - 月度采样: 每月最后 1 个 IS 交易日 T
  - factor at T (backward-only: 用 T-1, T-2, ..., T-N 的 ibs_day)
  - fwd_ret = close(T+22d) / close(T) - 1
  - 横截面 Spearman corr per month → 月度 IC 序列
  - ICIR = mean(IC) / std(IC) × sqrt(12)
  - min cross-section obs per month = 30

Spearman 正交性 (vs 已部署/已测因子):
  - amp_imb_20d        (v19.6 production)
  - margin_5d_chg      (v19.4 shadow)
  - margin_20d_chg     (v19.4 shadow)
  - vol_z_5d           (已测 abort, 验证机制差异)

判定 (Phase A protocol):
  - |ICIR| >= 0.40 + clean sign + Spearman vs production <0.30: 推荐 Phase B
  - |ICIR| in [0.20, 0.40): 边际, 不推荐, 记录
  - |ICIR| < 0.20: abort
  - 若 |ICIR| > 1.5: Phase B OOS overfit 风险高 → λ candidates 收紧
    到 {0.05, 0.10, 0.20} (v19.7/v19.9/super_big_net/shareholders 教训)

严格 OOS 约束: 只读 2014-01-01 ~ 2020-12-31 + 35 日 forward buffer.

输出:
  examples/factor_ic_ibs_csi300_is_grid.csv      10 项 IC 表 (|ICIR| desc)
  examples/factor_ic_ibs_csi300_is_monthly.csv   月度 IC 长表
  examples/factor_ic_ibs_csi300_is_spearman.csv  vs 4 因子正交性
  examples/factor_ic_ibs_csi300_is_report.md     人读 verdict

运行:
  .venv/bin/python examples/factor_ic_ibs_csi300_is.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

from _factor_ibs_panel import IBS_COLS, build_ibs_factors  # noqa: E402

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
MARGIN_PATH = ROOT / "data_cache" / "csi300_margin_14yr.parquet"

OUT_GRID = ROOT / "examples" / "factor_ic_ibs_csi300_is_grid.csv"
OUT_MONTHLY = ROOT / "examples" / "factor_ic_ibs_csi300_is_monthly.csv"
OUT_SPEARMAN = ROOT / "examples" / "factor_ic_ibs_csi300_is_spearman.csv"
OUT_REPORT = ROOT / "examples" / "factor_ic_ibs_csi300_is_report.md"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
PRE_BUFFER_DAYS = 120  # 60d rolling needs ~3 months pre-buffer
POST_BUFFER_DAYS = 45  # ~22 trading day forward window
FORWARD_DAYS = 22  # ~ 1 month
MIN_MONTHLY_OBS = 30

FACTOR_COLS = IBS_COLS  # 5 horizons


def load_csi300_codes() -> set[str]:
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    return set(csi["code"].tolist())


def load_kline_is(codes: set[str]) -> pd.DataFrame:
    """读 baidu_kline IS 期 + forward buffer, columns: code, date, close, vol."""
    pre_start = IS_START - pd.Timedelta(days=PRE_BUFFER_DAYS)
    post_end = IS_END + pd.Timedelta(days=POST_BUFFER_DAYS)
    k = pd.read_parquet(
        KLINE_PATH, columns=["code", "date", "close", "vol"]
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    k = k[(k["date"] >= pre_start) & (k["date"] <= post_end)]
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    print(
        f"[load] kline shape={k.shape}, codes={k['code'].nunique()}, "
        f"date range {k['date'].min().date()} ~ {k['date'].max().date()}",
        flush=True,
    )
    return k


def load_margin_is(codes: set[str]) -> pd.DataFrame | None:
    if not MARGIN_PATH.exists():
        return None
    m = pd.read_parquet(MARGIN_PATH)
    m["code"] = m["code"].astype(str).str.zfill(6)
    m["date"] = pd.to_datetime(m["date"])
    m = m[m["code"].isin(codes)]
    m = m[(m["date"] >= IS_START) & (m["date"] <= IS_END)]
    return m.sort_values(["code", "date"]).reset_index(drop=True)


def compute_amp_imb_20d_is(codes: set[str]) -> pd.DataFrame:
    """Compute amp_imb_20d on IS period for Spearman comparison (v19.6 ref)."""
    k = pd.read_parquet(
        KLINE_PATH, columns=["code", "date", "open", "high", "low", "close"]
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k["date"] = pd.to_datetime(k["date"])
    k = k[k["code"].isin(codes)]
    pre_start = IS_START - pd.Timedelta(days=PRE_BUFFER_DAYS)
    k = k[(k["date"] >= pre_start) & (k["date"] <= IS_END)].copy()
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    k["prev_close"] = k.groupby("code", sort=False)["close"].shift(1)
    valid = k["prev_close"].notna() & (k["prev_close"] > 0)

    amp = (k["high"] - k["low"]) / k["prev_close"]
    delta = k["close"] - k["prev_close"]
    amp_up = np.where(delta > 0, delta, 0.0) / k["prev_close"] * amp
    amp_dn = np.where(delta < 0, -delta, 0.0) / k["prev_close"] * amp
    k["amp"] = amp.where(valid)
    k["amp_up"] = pd.Series(amp_up, index=k.index).where(valid)
    k["amp_dn"] = pd.Series(amp_dn, index=k.index).where(valid)

    grp = k.groupby("code", sort=False)
    amp_sum_20 = grp["amp"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    amp_up_sum_20 = grp["amp_up"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    amp_dn_sum_20 = grp["amp_dn"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["amp_imb_20d"] = np.where(
        amp_sum_20 > 0,
        (amp_up_sum_20 - amp_dn_sum_20) / amp_sum_20,
        np.nan,
    )
    out = k[["code", "date", "amp_imb_20d"]].copy()
    out = out[(out["date"] >= IS_START) & (out["date"] <= IS_END)]
    return out


def compute_vol_z_5d_is(kline: pd.DataFrame) -> pd.DataFrame:
    """Compute vol_z_5d on IS period, backward-only.

    z(T) = (vol(T) - mean(vol, T-5..T-1)) / std(vol, T-5..T-1).
    """
    k = kline[["code", "date", "vol"]].copy()
    k["vol"] = k["vol"].astype(float)
    grp = k.groupby("code", sort=False)
    k["_vol_lag1"] = grp["vol"].shift(1)
    grp2 = k.groupby("code", sort=False)
    rmean = grp2["_vol_lag1"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )
    rstd = grp2["_vol_lag1"].transform(
        lambda s: s.rolling(5, min_periods=5).std()
    )
    k["vol_z_5d"] = (k["vol"] - rmean) / rstd.replace(0, np.nan)
    # mild clip to suppress outliers
    s = k["vol_z_5d"]
    if s.notna().any():
        lo, hi = s.quantile(0.005), s.quantile(0.995)
        k["vol_z_5d"] = s.clip(lo, hi)
    out = k[["code", "date", "vol_z_5d"]]
    out = out[(out["date"] >= IS_START) & (out["date"] <= IS_END)]
    return out


def build_panel(
    factor_df: pd.DataFrame,
    kline: pd.DataFrame,
    margin: pd.DataFrame | None,
    amp_df: pd.DataFrame,
    volz_df: pd.DataFrame,
    factor_cols: list[str],
) -> pd.DataFrame:
    """Build month-end PIT panel.

    For each month-end T (last IS trading day in month):
      - factor at T (backward-only, no look-ahead)
      - fwd_ret = close(T + 22 trading days) / close(T) - 1
      - margin / amp / vol_z snapshots at T (PIT, fall back to most recent <=T)
    """
    kline_dates = pd.DatetimeIndex(sorted(kline["date"].unique()))
    is_dates = kline_dates[
        (kline_dates >= IS_START) & (kline_dates <= IS_END)
    ]
    months = pd.Series(is_dates).dt.to_period("M")
    month_end_dates = (
        pd.Series(is_dates).groupby(months).last().reset_index(drop=True)
    )
    print(
        f"[panel] {len(month_end_dates)} 月度采样点 "
        f"{month_end_dates.iloc[0].date()} → "
        f"{month_end_dates.iloc[-1].date()}",
        flush=True,
    )

    wide_close = (
        kline.pivot_table(
            index="date", columns="code", values="close", aggfunc="first",
        )
        .sort_index()
    )

    f_wide_by_col = {
        c: factor_df.pivot_table(
            index="date", columns="code", values=c, aggfunc="first"
        ).sort_index()
        for c in factor_cols
    }

    m_wide_5 = m_wide_20 = None
    if margin is not None and not margin.empty:
        m_wide_5 = margin.pivot_table(
            index="date", columns="code", values="margin_5d_chg",
            aggfunc="first",
        ).sort_index()
        m_wide_20 = margin.pivot_table(
            index="date", columns="code", values="margin_20d_chg",
            aggfunc="first",
        ).sort_index()

    a_wide = amp_df.pivot_table(
        index="date", columns="code", values="amp_imb_20d", aggfunc="first",
    ).sort_index()

    v_wide = volz_df.pivot_table(
        index="date", columns="code", values="vol_z_5d", aggfunc="first",
    ).sort_index()

    rows: list[dict] = []
    for T in month_end_dates:
        idx_T = wide_close.index.get_indexer([T])[0]
        if idx_T < 0 or idx_T + FORWARD_DAYS >= len(wide_close.index):
            continue
        T_close = wide_close.iloc[idx_T]
        T_plus = wide_close.iloc[idx_T + FORWARD_DAYS]

        snap_factor = {}
        for c in factor_cols:
            fw = f_wide_by_col[c]
            i = fw.index.get_indexer([T])[0]
            snap_factor[c] = (
                fw.iloc[i] if i >= 0 else pd.Series(dtype=float)
            )

        snap_m5 = snap_m20 = None
        if m_wide_5 is not None:
            i5 = m_wide_5.index.get_indexer([T])[0]
            if i5 < 0:
                mask = m_wide_5.index <= T
                if mask.any():
                    snap_m5 = m_wide_5[mask].iloc[-1]
                    snap_m20 = m_wide_20[mask].iloc[-1]
            else:
                snap_m5 = m_wide_5.iloc[i5]
                snap_m20 = m_wide_20.iloc[i5]

        ia = a_wide.index.get_indexer([T])[0]
        snap_amp = a_wide.iloc[ia] if ia >= 0 else pd.Series(dtype=float)
        iv = v_wide.index.get_indexer([T])[0]
        snap_vz = v_wide.iloc[iv] if iv >= 0 else pd.Series(dtype=float)

        for code in wide_close.columns:
            c_now = T_close.get(code)
            c_fut = T_plus.get(code)
            if pd.isna(c_now) or pd.isna(c_fut) or c_now <= 0:
                continue
            row = {
                "month_end": T,
                "code": code,
                "fwd_ret": c_fut / c_now - 1,
            }
            for c in factor_cols:
                row[c] = snap_factor[c].get(code, np.nan)
            row["margin_5d_chg"] = (
                snap_m5.get(code, np.nan) if snap_m5 is not None else np.nan
            )
            row["margin_20d_chg"] = (
                snap_m20.get(code, np.nan) if snap_m20 is not None else np.nan
            )
            row["amp_imb_20d"] = snap_amp.get(code, np.nan)
            row["vol_z_5d"] = snap_vz.get(code, np.nan)
            rows.append(row)

    panel = pd.DataFrame(rows)
    print(
        f"[panel] built {len(panel)} rows × "
        f"{panel['code'].nunique()} codes × "
        f"{panel['month_end'].nunique()} months",
        flush=True,
    )
    return panel


def monthly_ic(
    panel: pd.DataFrame, factor_col: str, sign: int
) -> tuple[pd.Series, dict]:
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    if df.empty:
        return pd.Series(dtype=float), {
            "factor": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
            "ic_pos_pct": 0.0, "avg_obs": 0.0,
        }
    df["signed"] = sign * df[factor_col]

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < MIN_MONTHLY_OBS:
            return np.nan
        if g["signed"].std() <= 0:
            return np.nan
        return g["signed"].corr(g["fwd_ret"], method="spearman")

    obs = df.groupby("month_end").size()
    monthly = df.groupby("month_end").apply(_corr).dropna()
    if len(monthly) == 0:
        return monthly, {
            "factor": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
            "ic_pos_pct": 0.0,
            "avg_obs": float(obs.mean()) if len(obs) else 0.0,
        }
    mean = float(monthly.mean())
    std = float(monthly.std())
    icir = mean / std * np.sqrt(12) if std > 0 else 0.0
    return monthly, {
        "factor": factor_col,
        "sign": sign,
        "n_months": int(len(monthly)),
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "ic_pos_pct": float((monthly > 0).mean() * 100),
        "avg_obs": float(obs.mean()),
    }


def spearman_orth(
    panel: pd.DataFrame, factor_col: str, ref_col: str
) -> float:
    """Monthly cross-sectional Spearman rho between factor and ref.

    Returns mean |rho| across months (min_periods=30 obs).
    """
    df = panel.dropna(subset=[factor_col, ref_col])
    if df.empty:
        return float("nan")

    def _r(g: pd.DataFrame) -> float:
        if len(g) < MIN_MONTHLY_OBS:
            return np.nan
        if g[factor_col].std() <= 0 or g[ref_col].std() <= 0:
            return np.nan
        return g[factor_col].corr(g[ref_col], method="spearman")

    rho = df.groupby("month_end").apply(_r).dropna()
    if rho.empty:
        return float("nan")
    return float(rho.abs().mean())


def main() -> int:
    if not KLINE_PATH.exists():
        print(f"FATAL: {KLINE_PATH} 缺", file=sys.stderr)
        return 1
    if not CSI300_PATH.exists():
        print(f"FATAL: {CSI300_PATH} 缺", file=sys.stderr)
        return 1

    print("=" * 70)
    print("CSI300 IBS (Internal Bar Strength) Phase A IS IC 分析")
    print(f"IS 期: {IS_START.date()} → {IS_END.date()}")
    print(f"Forward return: {FORWARD_DAYS} 日")
    print("=" * 70)

    codes = load_csi300_codes()
    print(f"[load] CSI300 codes: {len(codes)}")

    # 1. Build IBS factor panel (raw daily values, all horizons)
    print("\n[1] Build IBS factor panel...")
    factor_pre_start = (IS_START - pd.Timedelta(days=PRE_BUFFER_DAYS)).strftime(
        "%Y-%m-%d"
    )
    ibs = build_ibs_factors(
        codes=codes,
        min_date=factor_pre_start,
        max_date=IS_END.strftime("%Y-%m-%d"),
    )
    # restrict factor rows to IS for joins (panel only samples at month-end ≤ IS_END)
    ibs_is = ibs[(ibs["date"] >= IS_START) & (ibs["date"] <= IS_END)]
    print(f"[1] IBS IS rows={len(ibs_is):,} codes={ibs_is['code'].nunique()}")

    # 2. Load kline IS (close + vol for fwd_ret + vol_z_5d)
    print("\n[2] Load kline IS + buffer...")
    kline = load_kline_is(codes)

    # 3. Load margin / compute amp_imb_20d / vol_z_5d for Spearman refs
    print("\n[3] Load margin + compute amp_imb_20d + vol_z_5d ...")
    margin = load_margin_is(codes)
    if margin is not None:
        print(f"[3] margin rows={len(margin):,} codes={margin['code'].nunique()}")
    else:
        print("[3] margin: cache missing — Spearman vs m5/m20 skipped")

    amp_df = compute_amp_imb_20d_is(codes)
    cov_amp = amp_df["amp_imb_20d"].notna().mean() * 100
    print(f"[3] amp_imb_20d rows={len(amp_df):,} coverage={cov_amp:.1f}%")

    volz_df = compute_vol_z_5d_is(kline)
    cov_volz = volz_df["vol_z_5d"].notna().mean() * 100
    print(f"[3] vol_z_5d rows={len(volz_df):,} coverage={cov_volz:.1f}%")

    # 4. Build month-end PIT panel
    print("\n[4] Build month-end PIT panel...")
    panel = build_panel(
        ibs_is, kline, margin, amp_df, volz_df, FACTOR_COLS
    )
    if panel.empty:
        print("FATAL: empty panel", file=sys.stderr)
        return 1

    # 5. Compute monthly IC × 10 (5 horizons × 2 signs)
    print("\n[5] Monthly IC × 10 (5 horizons × 2 signs)...")
    summaries: list[dict] = []
    monthly_series: dict[str, pd.Series] = {}
    for f in FACTOR_COLS:
        for sign in (+1, -1):
            m, s = monthly_ic(panel, f, sign=sign)
            s["factor_signed"] = f if sign == +1 else f + "__neg"
            summaries.append(s)
            monthly_series[s["factor_signed"]] = m

    summary_df = pd.DataFrame(summaries).round(4)
    summary_df = summary_df.reindex(
        summary_df["icir"].abs().sort_values(
            ascending=False, na_position="last"
        ).index
    )
    summary_df = summary_df[[
        "factor_signed", "factor", "sign", "n_months",
        "ic_mean", "ic_std", "icir", "ic_pos_pct", "avg_obs",
    ]]
    summary_df.to_csv(OUT_GRID, index=False)
    print(f"[output] grid → {OUT_GRID}")
    print(summary_df.to_string(index=False))

    # 6. Spearman vs production / shadow / aborted factors
    print("\n[6] Spearman 正交性 vs amp_imb_20d / m5 / m20 / vol_z_5d...")
    spr_rows = []
    for f in FACTOR_COLS:
        spr_rows.append({
            "factor": f,
            "vs_amp_imb_20d": spearman_orth(panel, f, "amp_imb_20d"),
            "vs_margin_5d_chg": spearman_orth(panel, f, "margin_5d_chg"),
            "vs_margin_20d_chg": spearman_orth(panel, f, "margin_20d_chg"),
            "vs_vol_z_5d": spearman_orth(panel, f, "vol_z_5d"),
        })
    spr_df = pd.DataFrame(spr_rows).round(4)
    spr_df.to_csv(OUT_SPEARMAN, index=False)
    print(f"[output] spearman → {OUT_SPEARMAN}")
    print(spr_df.to_string(index=False))

    # 7. Monthly long-format output
    monthly_rows = []
    for k, ser in monthly_series.items():
        for m, v in ser.items():
            monthly_rows.append(
                {"factor_signed": k, "month_end": m, "ic": v}
            )
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly → {OUT_MONTHLY}")

    # 8. Verdict — pick row with largest |ICIR|, then deploy sign (ic_mean>0)
    best_abs_idx = summary_df["icir"].abs().idxmax()
    best_factor = summary_df.loc[best_abs_idx, "factor"]
    mirror_pair = summary_df[summary_df["factor"] == best_factor]
    pos_rows = mirror_pair[mirror_pair["ic_mean"] > 0]
    if len(pos_rows):
        best = pos_rows.iloc[0]
    else:
        best = mirror_pair.iloc[0]
    best_abs = abs(best["icir"]) if pd.notna(best["icir"]) else 0.0
    best_sign = int(best["sign"])
    spr_row = spr_df[spr_df["factor"] == best_factor].iloc[0]

    def _safe(x):
        return float(x) if pd.notna(x) else 0.0

    orth_vs_amp = _safe(spr_row["vs_amp_imb_20d"])
    orth_vs_m5 = _safe(spr_row["vs_margin_5d_chg"])
    orth_vs_m20 = _safe(spr_row["vs_margin_20d_chg"])
    orth_vs_volz = _safe(spr_row["vs_vol_z_5d"])
    max_orth_prod = max(orth_vs_amp, orth_vs_m5, orth_vs_m20)

    # λ candidate selection: tighten if IS |ICIR| > 1.5 (overfit risk)
    if best_abs > 1.5:
        lambda_candidates = "{0.05, 0.10, 0.20}  (IS |ICIR|>1.5 收紧, overfit 防御)"
    else:
        lambda_candidates = "{0.10, 0.20, 0.30}"

    if best_abs >= 0.40 and max_orth_prod < 0.30:
        verdict = (
            f"STRONG — 推荐进 Phase B sidecar OOS. "
            f"Lock (factor={best_factor}, sign={best_sign}, "
            f"λ candidates={lambda_candidates})"
        )
    elif best_abs >= 0.40:
        verdict = (
            f"MARGINAL — |ICIR|={best_abs:.3f} 够但 Spearman vs production "
            f"max={max_orth_prod:.3f} ≥ 0.30, 信息冗余风险高."
        )
    elif best_abs >= 0.20:
        verdict = f"WEAK — |ICIR|={best_abs:.3f}, abort (< 0.40 阈值)."
    else:
        verdict = f"VERY WEAK — |ICIR|={best_abs:.3f}, abort."

    # 9. Markdown report
    n_codes = panel["code"].nunique()
    n_months = panel["month_end"].nunique()
    with open(OUT_REPORT, "w") as out:
        out.write("# CSI300 IBS (Internal Bar Strength) IS IC 分析 (2014-2020)\n\n")
        out.write(f"**Universe:** CSI300 ({n_codes} 实际覆盖 / 300)\n\n")
        out.write(f"**IS 期:** {IS_START.date()} → {IS_END.date()} ")
        out.write(f"({n_months} 月)\n\n")
        out.write(f"**Forward return:** {FORWARD_DAYS} 日 (~ 1 月)\n\n")
        out.write(
            f"**Panel:** {len(panel):,} rows × {n_codes} codes × {n_months} months\n\n"
        )
        out.write(
            "**数据源:** baidu_kline.parquet (hfq close/high/low)\n\n"
        )
        out.write("## 因子定义\n\n")
        out.write(
            "`IBS_day_t = (close_t - low_t) / (high_t - low_t)` ∈ [0, 1]\n\n"
        )
        out.write("| factor | 含义 |\n|---|---|\n")
        out.write("| IBS_1d        | IBS_day at T-1 (单日) |\n")
        out.write("| IBS_5d_mean   | mean(IBS_day, T-5..T-1) |\n")
        out.write("| IBS_10d_mean  | mean(IBS_day, T-10..T-1) |\n")
        out.write("| IBS_20d_mean  | mean(IBS_day, T-20..T-1) |\n")
        out.write("| IBS_60d_mean  | mean(IBS_day, T-60..T-1) |\n\n")
        out.write(
            "**机制 (Connors literature):** high IBS → 当日强势 close at high "
            "→ 下月反转 (mean reversion). 预期 sign = **-1**.\n\n"
        )
        out.write("**涨跌停 mask:** high == low → IBS 未定义 (NaN) → 月度 IC 自动剔除.\n\n")
        out.write("## IC 表 (按 |ICIR| 降序, 10 项)\n\n")
        out.write(summary_df.to_markdown(index=False))
        out.write("\n\n")
        out.write("## Spearman 正交性 (mean |rho| per month)\n\n")
        out.write(spr_df.to_markdown(index=False))
        out.write("\n\n")
        out.write(
            "**阈值:** <0.10 独立, <0.30 可接受, >0.30 强相关.\n\n"
        )
        out.write("## 判定规则\n\n")
        out.write("- |ICIR| ≥ 0.40 + Spearman vs production <0.30: 进 Phase B\n")
        out.write("- |ICIR| ≥ 0.40 但 Spearman ≥0.30: MARGINAL\n")
        out.write("- |ICIR| ∈ [0.20, 0.40): WEAK abort\n")
        out.write("- |ICIR| < 0.20: VERY WEAK abort\n")
        out.write(
            "- |ICIR| > 1.5: λ candidates 收紧 (v19.7/v19.9 OOS 衰减教训)\n\n"
        )
        out.write("## 结论\n\n")
        out.write(
            f"**Best factor:** `{best_factor}` "
            f"(deploy sign={best_sign}, ICIR={best['icir']:.3f}, "
            f"IC mean={best['ic_mean']:.4f}, n_months={best['n_months']}, "
            f"avg_obs={best['avg_obs']:.0f})\n\n"
        )
        out.write(
            f"**Spearman:** vs amp_imb_20d={orth_vs_amp:.3f}, "
            f"vs margin_5d_chg={orth_vs_m5:.3f}, "
            f"vs margin_20d_chg={orth_vs_m20:.3f}, "
            f"vs vol_z_5d={orth_vs_volz:.3f}\n\n"
        )
        out.write(f"**Verdict:** {verdict}\n\n")
        if orth_vs_amp >= 0.30:
            out.write(
                f"**Note:** Spearman vs amp_imb_20d (v19.6 production) "
                f"= {orth_vs_amp:.3f} ≥ 0.30. IBS 与 amp_imb_20d "
                "机制重叠 (都是日内 K 线结构衍生), 边际增益受限.\n\n"
            )
        if best_abs > 1.5:
            out.write(
                "**Warning:** IS |ICIR| > 1.5 极强. 历史教训 (v19.7 / v19.9 / "
                "super_big_net / shareholders) 显示 IS IC 越强 OOS overfit "
                "概率越大. Phase B 务必严格 OOS 协议, λ 收紧到 "
                "{0.05, 0.10, 0.20} (不可用 0.30).\n"
            )
    print(f"[output] report → {OUT_REPORT}")

    print(f"\n=== VERDICT ===\n{verdict}")
    print(
        f"Best: {best_factor} | deploy sign={best_sign} | "
        f"ICIR={best['icir']:.3f} | IC mean={best['ic_mean']:.4f} | "
        f"n_months={best['n_months']} | "
        f"Spearman vs amp={orth_vs_amp:.3f} m5={orth_vs_m5:.3f} "
        f"m20={orth_vs_m20:.3f} volz={orth_vs_volz:.3f}"
    )
    print(f"λ candidates: {lambda_candidates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
