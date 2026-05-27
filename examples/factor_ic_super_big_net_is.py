"""CSI300 super_big_net (资金流分层) 因子 IC 分析 — 严格 IS 2014-2020.

数据源:
  data_cache/fund_flow/fund_flow_csi300.parquet  (Sina money flow, 14yr depth)
  data_cache/baidu_kline.parquet                  (forward return, close)
  data_cache/csi300_margin_14yr.parquet           (Spearman vs margin_5d/20d)

候选因子 (5 个, 各 sign +/- 评估, 共 10 项):
  1. pct_super_big_5d      = 5d rolling mean of pct_super_big (≡ r0_net/total)
  2. pct_super_big_10d
  3. pct_super_big_20d
  4. net_super_big_5d_chg  = 5d pct_change of net_super_big
  5. main_inflow_ratio_5d  = (net_super_big + net_big) / total_amount, 5d mean
                            ≡ pct_main_5d

方法 (point-in-time, no lookahead):
  - 月度采样: 每月最后 1 个 IS 交易日 T (用 T 上的 backward-looking 因子)
  - factor 值: 用 ≤ T 的 fund_flow daily, rolling 5d/10d/20d backward-only
  - fwd_ret = close(T+22d) / close(T) - 1  (~ 1 月前向收益)
  - 横截面 Spearman corr per month → 月度 IC 序列
  - ICIR = mean / std * sqrt(12)

Spearman 正交性 (vs 现有 production 因子):
  - margin_5d_chg, margin_20d_chg  (v19.4 production)
  - amp_imb_20d                    (v19.6 production candidate)

判定:
  |ICIR| ≥ 0.40 + clean sign + Spearman vs m5/m20 < 0.30 → 进 Phase B
  否则 abort

输出:
  examples/super_big_net_is_grid.csv         按 |ICIR| 排序的 10 项 grid
  examples/super_big_net_is_monthly.csv      月度 IC 长表
  examples/super_big_net_is_report.md        人读 (verdict + 推荐)
  examples/super_big_net_spearman.csv        正交性表
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
FUND_FLOW_PATH = ROOT / "data_cache" / "fund_flow" / "fund_flow_csi300.parquet"
MARGIN_PATH = ROOT / "data_cache" / "csi300_margin_14yr.parquet"

OUT_GRID = ROOT / "examples" / "super_big_net_is_grid.csv"
OUT_MONTHLY = ROOT / "examples" / "super_big_net_is_monthly.csv"
OUT_REPORT = ROOT / "examples" / "super_big_net_is_report.md"
OUT_SPEARMAN = ROOT / "examples" / "super_big_net_spearman.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 22  # ~ 1 month
MIN_MONTHLY_OBS = 30


def build_fund_flow_factors(ff: pd.DataFrame) -> pd.DataFrame:
    """Compute backward-only rolling factors per (code, date)."""
    ff = ff.sort_values(["code", "date"]).reset_index(drop=True)
    grp = ff.groupby("code", sort=False)

    ff["pct_super_big_5d"] = grp["pct_super_big"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )
    ff["pct_super_big_10d"] = grp["pct_super_big"].transform(
        lambda s: s.rolling(10, min_periods=10).mean()
    )
    ff["pct_super_big_20d"] = grp["pct_super_big"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )

    nsb = ff["net_super_big"]
    nsb_lag5 = grp["net_super_big"].shift(5)
    denom = nsb_lag5.abs()
    ff["net_super_big_5d_chg"] = np.where(
        denom > 0, (nsb - nsb_lag5) / denom, np.nan
    )

    ff["main_inflow_ratio_5d"] = grp["pct_main"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )

    factor_cols = [
        "pct_super_big_5d",
        "pct_super_big_10d",
        "pct_super_big_20d",
        "net_super_big_5d_chg",
        "main_inflow_ratio_5d",
    ]
    return ff[["code", "date", *factor_cols]].copy()


def load_kline_is() -> pd.DataFrame:
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = set(csi["code"].tolist())
    k = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
    k["code"] = k["code"].astype(str).str.zfill(6)
    k["date"] = pd.to_datetime(k["date"])
    k = k[k["code"].isin(codes)]
    k = k[
        (k["date"] >= IS_START)
        & (k["date"] <= IS_END + pd.Timedelta(days=45))
    ]
    return k.sort_values(["code", "date"]).reset_index(drop=True)


def load_margin_is() -> pd.DataFrame | None:
    if not MARGIN_PATH.exists():
        return None
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    m = pd.read_parquet(MARGIN_PATH)
    m["code"] = m["code"].astype(str).str.zfill(6)
    m = m[m["code"].isin(csi["code"])]
    m["date"] = pd.to_datetime(m["date"])
    m = m[(m["date"] >= IS_START) & (m["date"] <= IS_END)]
    return m.sort_values(["code", "date"]).reset_index(drop=True)


def compute_amp_imb_20d_is() -> pd.DataFrame:
    """Compute amp_imb_20d for IS period (open/high/low/close required)."""
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = set(csi["code"].tolist())
    k = pd.read_parquet(
        KLINE_PATH, columns=["code", "date", "open", "high", "low", "close"]
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k["date"] = pd.to_datetime(k["date"])
    k = k[k["code"].isin(codes)]
    k = k[(k["date"] >= IS_START) & (k["date"] <= IS_END)].copy()
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
    return k[["code", "date", "amp_imb_20d"]]


def build_panel(
    factor_df: pd.DataFrame,
    kline: pd.DataFrame,
    margin: pd.DataFrame | None,
    amp_df: pd.DataFrame,
    factor_cols: list[str],
) -> pd.DataFrame:
    """Build month-end sampled PIT panel.

    For each month-end T (last IS trading day in month):
      - factor at T (backward-only rolling, no look-ahead)
      - fwd_ret = close(T+22d) / close(T) - 1
      - margin / amp snapshots at T (PIT, fall back to most recent <=T)
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
            "factor": factor_col, "sign": sign,
            "n_months": 0, "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
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
            "factor": factor_col, "sign": sign,
            "n_months": 0, "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
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

    Returns mean |rho| across months.
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


FACTOR_COLS = [
    "pct_super_big_5d",
    "pct_super_big_10d",
    "pct_super_big_20d",
    "net_super_big_5d_chg",
    "main_inflow_ratio_5d",
]


def main() -> int:
    if not FUND_FLOW_PATH.exists():
        print(
            f"FATAL: {FUND_FLOW_PATH} 缺, run fetch_fund_flow_csi300.py",
            file=sys.stderr,
        )
        return 1
    if not KLINE_PATH.exists():
        print(f"FATAL: {KLINE_PATH} 缺", file=sys.stderr)
        return 1

    print(f"[load] fund_flow {FUND_FLOW_PATH.name}...")
    ff = pd.read_parquet(FUND_FLOW_PATH)
    ff["code"] = ff["code"].astype(str).str.zfill(6)
    ff["date"] = pd.to_datetime(ff["date"])
    print(
        f"[load] fund_flow rows={len(ff):,} codes={ff['code'].nunique()} "
        f"date {ff['date'].min().date()} ~ {ff['date'].max().date()}"
    )

    print("[compute] backward rolling factors...")
    factors = build_fund_flow_factors(ff)
    for c in FACTOR_COLS:
        cov = factors[c].notna().mean() * 100
        print(f"  {c:32s} coverage={cov:.1f}%")

    print("[load] kline (close, IS+buffer)...")
    kline = load_kline_is()
    print(
        f"[load] kline rows={len(kline):,} codes={kline['code'].nunique()}"
    )

    print("[load] margin (for Spearman vs m5/m20)...")
    margin = load_margin_is()
    if margin is not None:
        print(
            f"[load] margin rows={len(margin):,} "
            f"codes={margin['code'].nunique()}"
        )
    else:
        print("[load] margin: cache missing — Spearman vs m5/m20 skipped")

    print("[compute] amp_imb_20d (IS only) for Spearman ref...")
    amp_df = compute_amp_imb_20d_is()
    cov_amp = amp_df["amp_imb_20d"].notna().mean() * 100
    print(f"[compute] amp_imb_20d coverage={cov_amp:.1f}%")

    print("[panel] building month-end panel...")
    panel = build_panel(factors, kline, margin, amp_df, FACTOR_COLS)
    if panel.empty:
        print("FATAL: empty panel", file=sys.stderr)
        return 1

    summaries: list[dict] = []
    monthly_series: dict[str, pd.Series] = {}
    print("\n[ic] computing monthly IC × 10 (5 factors × 2 signs)...")
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

    print("\n[spearman] vs margin_5d_chg / margin_20d_chg / amp_imb_20d...")
    spr_rows = []
    for f in FACTOR_COLS:
        spr_rows.append({
            "factor": f,
            "vs_margin_5d_chg": spearman_orth(panel, f, "margin_5d_chg"),
            "vs_margin_20d_chg": spearman_orth(panel, f, "margin_20d_chg"),
            "vs_amp_imb_20d": spearman_orth(panel, f, "amp_imb_20d"),
        })
    spr_df = pd.DataFrame(spr_rows).round(4)
    spr_df.to_csv(OUT_SPEARMAN, index=False)
    print(f"[output] spearman → {OUT_SPEARMAN}")
    print(spr_df.to_string(index=False))

    monthly_rows = []
    for k, ser in monthly_series.items():
        for m, v in ser.items():
            monthly_rows.append(
                {"factor_signed": k, "month_end": m, "ic": v}
            )
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly → {OUT_MONTHLY}")

    # Best factor: row with largest |ICIR|, then among the (+1, -1) mirror
    # pair pick the sign that gives positive ICIR (the one to deploy as
    # `final = z(pred) + λ·z(signed_factor)`, where positive ICIR ⇒ IC>0).
    best_abs_idx = summary_df["icir"].abs().idxmax()
    best_factor = summary_df.loc[best_abs_idx, "factor"]
    mirror_pair = summary_df[summary_df["factor"] == best_factor]
    # row with ic_mean > 0 is the deployable sign
    pos_rows = mirror_pair[mirror_pair["ic_mean"] > 0]
    if len(pos_rows):
        best = pos_rows.iloc[0]
    else:
        best = mirror_pair.iloc[0]  # fallback
    best_abs = abs(best["icir"]) if pd.notna(best["icir"]) else 0.0
    best_sign = int(best["sign"])
    spr_row = spr_df[spr_df["factor"] == best_factor].iloc[0]
    max_orth_vs_margin = float(
        max(
            spr_row["vs_margin_5d_chg"]
            if pd.notna(spr_row["vs_margin_5d_chg"]) else 0,
            spr_row["vs_margin_20d_chg"]
            if pd.notna(spr_row["vs_margin_20d_chg"]) else 0,
        )
    )
    orth_vs_amp = float(
        spr_row["vs_amp_imb_20d"]
        if pd.notna(spr_row["vs_amp_imb_20d"]) else 0.0
    )

    if best_abs >= 0.40 and max_orth_vs_margin < 0.30:
        verdict = (
            f"STRONG — 推荐进 Phase B sidecar OOS. "
            f"Lock (factor={best_factor}, sign={best_sign}, "
            "λ candidates={0.10, 0.20, 0.30})"
        )
    elif best_abs >= 0.40:
        verdict = (
            f"MARGINAL — |ICIR|={best_abs:.3f} 够但 Spearman vs margin "
            f"max={max_orth_vs_margin:.3f} ≥ 0.30, 信息冗余风险高."
        )
    elif best_abs >= 0.25:
        verdict = f"WEAK — |ICIR|={best_abs:.3f}, abort (< 0.40 阈值)."
    else:
        verdict = f"VERY WEAK — |ICIR|={best_abs:.3f}, abort."

    with open(OUT_REPORT, "w") as out:
        out.write("# CSI300 super_big_net 因子 IS IC 分析 (2014-2020)\n\n")
        out.write(
            f"**Universe:** CSI300 ({panel['code'].nunique()} 实际覆盖)\n\n"
        )
        out.write(f"**IS 期:** {IS_START.date()} → {IS_END.date()} ")
        out.write(f"({panel['month_end'].nunique()} 月)\n\n")
        out.write(f"**Forward return:** {FORWARD_DAYS} 日 (~ 1 月)\n\n")
        out.write(
            f"**Panel:** {len(panel)} rows × "
            f"{panel['code'].nunique()} codes × "
            f"{panel['month_end'].nunique()} months\n\n"
        )
        out.write(
            "**数据源:** Sina money.finance "
            "(`MoneyFlow.ssl_qsfx_lscjfb`)\n\n"
        )
        out.write("## 因子定义\n\n")
        out.write("| factor | 含义 |\n|---|---|\n")
        out.write("| pct_super_big_5d  | 5d mean of r0_net/total_amount |\n")
        out.write("| pct_super_big_10d | 10d mean (= user spec 主因子) |\n")
        out.write("| pct_super_big_20d | 20d mean |\n")
        out.write("| net_super_big_5d_chg | 5d pct change of r0_net |\n")
        out.write(
            "| main_inflow_ratio_5d | 5d mean of (r0+r1)_net/total |\n\n"
        )
        out.write("## IC 表 (按 |ICIR| 降序, 10 项)\n\n")
        out.write(summary_df.to_markdown(index=False))
        out.write("\n\n")
        out.write("## Spearman 正交性 (mean |rho| per month)\n\n")
        out.write(spr_df.to_markdown(index=False))
        out.write("\n\n")
        out.write("**阈值:** <0.10 独立, <0.30 可接受, >0.30 强相关.\n\n")
        out.write("## 判定规则\n\n")
        out.write("- |ICIR| ≥ 0.40 + Spearman vs margin <0.30: 进 Phase B\n")
        out.write("- |ICIR| ≥ 0.40 但 Spearman ≥0.30: MARGINAL\n")
        out.write("- |ICIR| < 0.40: abort\n\n")
        out.write("## 结论\n\n")
        out.write(
            f"**Best factor:** `{best_factor}` "
            f"(deploy sign={best_sign}, ICIR={best['icir']:.3f}, "
            f"IC mean={best['ic_mean']:.4f}, n_months={best['n_months']}, "
            f"Spearman max vs margin={max_orth_vs_margin:.3f}, "
            f"vs amp_imb_20d={orth_vs_amp:.3f})\n\n"
        )
        out.write(f"**Verdict:** {verdict}\n\n")
        if orth_vs_amp >= 0.30:
            out.write(
                f"**Note:** Spearman vs amp_imb_20d = {orth_vs_amp:.3f} ≥ 0.30. "
                "因子与 v19.6 amp_imb_20d 相关性偏高 (但与 v19.4 margin "
                "正交). 若 v19.6 已 production, 边际增益受限; 若 v19.4 "
                "production, super_big_net 提供独立信息.\n"
            )
    print(f"[output] report → {OUT_REPORT}")
    print(f"\n=== VERDICT ===\n{verdict}")
    print(
        f"Best: {best_factor} | deploy sign={best_sign} | "
        f"ICIR={best['icir']:.3f} | IC mean={best['ic_mean']:.4f} | "
        f"Spearman vs margin max={max_orth_vs_margin:.3f} | "
        f"vs amp_imb_20d={orth_vs_amp:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
