"""CSI300 股东户数变化因子 IC 分析 — 严格 IS 2014-2020.

方法 (point-in-time, no lookahead):
- 每月第 1 个交易日 T:
  - 因子值 = build_factor_panel(...) PIT, 用 announce_date ≤ T 的 latest 季度变化
  - 前向收益 = close(T+5) / close(T) - 1
- 横截面 Spearman corr(factor_z, fwd_z) → monthly IC
- 月度 IC 序列 → mean / std / ICIR = mean / std * sqrt(12) / %>0

候选因子 (raw + sign 评估):
  机制: 户数减少 (count_change < 0) → 集中度上升 → bullish → sign=-1
  1. count_change_3m       (sign=-1)
  2. count_change_6m       (sign=-1)
  3. count_change_12m      (sign=-1)
  4. lvl_concentration     (sign=+1; 横截面 = 1 - avg/median, 越大越分散, 反向相关)
  5. combo_neg_chg = -(z3 + z6 + z12)/3  (反向: 期望 IC > 0)

判定:
  best |ICIR| > 0.5 → 进 Phase B
  0.3 < |ICIR| < 0.5 → 边际
  |ICIR| < 0.3 → abort
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

from claude_finance import shareholders  # noqa: E402

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
MARGIN_PATH = ROOT / "data_cache" / "csi300_margin_14yr.parquet"

OUT_CSV = ROOT / "examples" / "shareholders_is_grid.csv"
OUT_MONTHLY = ROOT / "examples" / "factor_ic_shareholders_is_monthly.csv"
OUT_REPORT = ROOT / "examples" / "factor_ic_shareholders_is_report.md"
OUT_SPEARMAN = ROOT / "examples" / "factor_ic_shareholders_is_spearman.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 5

# (factor_col, sign) — sign 用于 IC 翻号
SINGLE_FACTORS = [
    ("count_change_3m", -1),
    ("count_change_6m", -1),
    ("count_change_12m", -1),
    ("lvl_concentration", +1),
]


def _strip_prefix(s: pd.Series) -> pd.Series:
    """剥离 'SH'/'SZ'/'BJ' 前缀, 留 6 位数字 (兼容多种表)."""
    s = s.astype(str)
    extracted = s.str.extract(r"(\d{6})")[0]
    return extracted


def load_kline_is() -> pd.DataFrame:
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = set(csi["code"].tolist())

    k = pd.read_parquet(KLINE_PATH,
                         columns=["code", "date", "close", "high", "low"])
    k["code"] = _strip_prefix(k["code"])
    k = k.dropna(subset=["code"])
    k = k[k["code"].isin(codes)]
    k["date"] = pd.to_datetime(k["date"])
    k = k[(k["date"] >= IS_START - pd.Timedelta(days=60))
          & (k["date"] <= IS_END + pd.Timedelta(days=15))]
    return k.sort_values(["code", "date"]).reset_index(drop=True)


def build_panel(kline: pd.DataFrame,
                share_df: pd.DataFrame) -> pd.DataFrame:
    """构造 panel: (asof_date, code, factor cols, fwd_ret)."""
    kline_dates = pd.DatetimeIndex(sorted(kline["date"].unique()))
    is_dates = kline_dates[
        (kline_dates >= IS_START) & (kline_dates <= IS_END)
    ]
    months = pd.Series(is_dates).dt.to_period("M")
    month_first_day = (
        pd.Series(is_dates).groupby(months).first().reset_index(drop=True)
    )
    print(f"[panel] {len(month_first_day)} 月度采样点 "
          f"({month_first_day.iloc[0].date()} → "
          f"{month_first_day.iloc[-1].date()})", flush=True)

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = csi["code"].tolist()

    factor_panel = shareholders.build_factor_panel(
        share_df, codes, list(month_first_day),
        max_lookback_days=365,
    )

    wide = kline.pivot_table(
        index="date", columns="code", values="close", aggfunc="first",
    ).sort_index()
    fwd_rows = []
    for T in month_first_day:
        idx = wide.index.get_indexer([T])[0]
        if idx < 0 or idx + FORWARD_DAYS >= len(wide.index):
            continue
        T_close = wide.iloc[idx]
        T_plus = wide.iloc[idx + FORWARD_DAYS]
        for code in codes:
            c_now = T_close.get(code)
            c_fut = T_plus.get(code)
            if pd.isna(c_now) or pd.isna(c_fut) or c_now <= 0:
                continue
            fwd_rows.append({
                "asof_date": T, "code": code,
                "fwd_ret": c_fut / c_now - 1,
            })
    fwd = pd.DataFrame(fwd_rows)
    panel = factor_panel.merge(fwd, on=["asof_date", "code"], how="inner")
    print(f"[panel] {len(panel)} rows × "
          f"{panel['code'].nunique()} codes × "
          f"{panel['asof_date'].nunique()} months", flush=True)
    return panel


def monthly_ic(panel: pd.DataFrame, factor_col: str, sign: int = 1):
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    if df.empty:
        return pd.Series(dtype=float), {
            "factor": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
            "ic_pos_pct": np.nan,
        }
    df["signed_factor"] = sign * df[factor_col]

    def _corr(g):
        if len(g) < 10:
            return np.nan
        if g["signed_factor"].nunique() < 2:
            return np.nan
        return g["signed_factor"].corr(g["fwd_ret"], method="spearman")

    monthly = df.groupby("asof_date").apply(_corr).dropna()
    if monthly.empty:
        return monthly, {
            "factor": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
            "ic_pos_pct": np.nan,
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
    }


def _zscore(g: pd.Series) -> pd.Series:
    std = g.std(ddof=0)
    if std <= 0 or pd.isna(std):
        return pd.Series(0.0, index=g.index)
    return (g - g.mean()) / std


def compute_spearman_vs_existing(
    panel: pd.DataFrame, kline: pd.DataFrame
) -> pd.DataFrame:
    """月度横截面 Spearman: 候选 factor vs (margin_5d_chg / margin_20d_chg / amp_imb_20d).

    margin: 直接读 csi300_margin_14yr.parquet 的预算列.
    amp_imb_20d: 用 kline (high-low)/close 20 日 rolling std/mean.
    """
    out_rows = []
    asof_dates = sorted(panel["asof_date"].unique())

    # margin
    if not MARGIN_PATH.exists():
        print(f"[spearman] WARN: {MARGIN_PATH} 缺")
        margin_panel = pd.DataFrame()
    else:
        margin = pd.read_parquet(MARGIN_PATH)
        margin["code"] = _strip_prefix(margin["code"])
        margin = margin.dropna(subset=["code"])
        margin["date"] = pd.to_datetime(margin["date"])
        # asof_date 用 ≤ T 的 latest 月度 margin
        margin = margin.sort_values(["code", "date"])
        rows = []
        for asof in asof_dates:
            slab = margin[margin["date"] <= asof]
            if slab.empty:
                continue
            latest = slab.groupby("code").tail(1)
            for _, r in latest.iterrows():
                rows.append({
                    "asof_date": asof, "code": r["code"],
                    "margin_5d_chg": r.get("margin_5d_chg", np.nan),
                    "margin_20d_chg": r.get("margin_20d_chg", np.nan),
                })
        margin_panel = pd.DataFrame(rows)

    # amp_imb_20d: 用 canonical 公式 (与 production v19.6 一致)
    # amp_imb_20d = (sum(amp_up,20) - sum(amp_dn,20)) / sum(amp,20)
    # amp = (high - low) / prev_close
    # amp_up = max(0, close - prev_close)/prev_close * amp
    # amp_dn = max(0, prev_close - close)/prev_close * amp
    sys.path.insert(0, str(ROOT / "examples"))
    from _factor_kline_panel import build_kline_factors  # noqa: E402
    kf = build_kline_factors(
        min_date=(IS_START - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
    )
    kf["code"] = kf["code"].astype(str).str.zfill(6)
    kf = kf[kf["date"] <= IS_END][["code", "date", "amp_imb_20d"]]
    kf = kf.sort_values(["code", "date"])
    rows = []
    for asof in asof_dates:
        slab = kf[kf["date"] <= asof]
        if slab.empty:
            continue
        latest = slab.groupby("code").tail(1)
        for _, r in latest.iterrows():
            rows.append({
                "asof_date": asof, "code": r["code"],
                "amp_imb_20d": r["amp_imb_20d"],
            })
    amp_panel = pd.DataFrame(rows)

    merged = panel.copy()
    if not margin_panel.empty:
        merged = merged.merge(margin_panel,
                              on=["asof_date", "code"], how="left")
    if not amp_panel.empty:
        merged = merged.merge(amp_panel,
                              on=["asof_date", "code"], how="left")

    factor_cols = [
        "count_change_3m", "count_change_6m", "count_change_12m",
        "lvl_concentration", "combo_neg_chg",
    ]
    existing_cols = [c for c in
                     ["margin_5d_chg", "margin_20d_chg", "amp_imb_20d"]
                     if c in merged.columns]
    for fc in factor_cols:
        if fc not in merged.columns:
            continue
        for ec in existing_cols:
            corrs = []
            for asof, g in merged.groupby("asof_date"):
                sub = g[[fc, ec]].dropna()
                if len(sub) < 20:
                    continue
                if sub[fc].nunique() < 2 or sub[ec].nunique() < 2:
                    continue
                rho = sub[fc].corr(sub[ec], method="spearman")
                if pd.notna(rho):
                    corrs.append(rho)
            if corrs:
                out_rows.append({
                    "factor": fc, "vs": ec,
                    "n_months": len(corrs),
                    "mean_rho": float(np.mean(corrs)),
                    "mean_abs_rho": float(np.mean(np.abs(corrs))),
                })
    return pd.DataFrame(out_rows)


def main() -> int:
    if not KLINE_PATH.exists():
        print(f"FATAL: kline {KLINE_PATH} 缺", file=sys.stderr)
        return 1
    cache = shareholders.load_cache()
    if cache.empty:
        print(f"FATAL: shareholders cache 空, 先跑 fetch_shareholders_csi300.py",
              file=sys.stderr)
        return 1
    print(f"[load] shareholders cache: {len(cache):,} rows × "
          f"{cache['code'].nunique():,} stocks")
    kline = load_kline_is()
    print(f"[load] kline IS shape={kline.shape}")

    panel = build_panel(kline, cache)
    if panel.empty:
        print("FATAL: panel 空", file=sys.stderr)
        return 1

    print("\n[sparsity] 因子非 NaN 率:")
    for f, _ in SINGLE_FACTORS:
        n_nz = panel[f].notna().mean() * 100
        print(f"  {f}: {n_nz:.1f}% non-NaN")

    summaries: list[dict] = []
    monthly_series: dict[str, pd.Series] = {}

    for f, s in SINGLE_FACTORS:
        m, summary = monthly_ic(panel, f, sign=s)
        summaries.append(summary)
        monthly_series[f] = m

    # combo: -(z3 + z6 + z12)/3
    panel = panel.copy()
    z3 = panel.groupby("asof_date")["count_change_3m"].transform(_zscore)
    z6 = panel.groupby("asof_date")["count_change_6m"].transform(_zscore)
    z12 = panel.groupby("asof_date")["count_change_12m"].transform(_zscore)
    panel["combo_neg_chg"] = -(z3 + z6 + z12) / 3.0
    m, summary = monthly_ic(panel, "combo_neg_chg", sign=+1)
    summaries.append(summary)
    monthly_series["combo_neg_chg"] = m

    summary_df = pd.DataFrame(summaries).round(4)
    summary_df = summary_df.reindex(
        summary_df["icir"].abs().sort_values(
            ascending=False, na_position="last",
        ).index
    )
    summary_df.to_csv(OUT_CSV, index=False)
    print(f"\n[output] summary → {OUT_CSV}")
    print(summary_df.to_string(index=False))

    monthly_rows = []
    for f, ser in monthly_series.items():
        for d, v in ser.items():
            monthly_rows.append({"factor": f, "asof_date": d, "ic": v})
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly → {OUT_MONTHLY}")

    print("\n[spearman] vs (margin_5d_chg, margin_20d_chg, amp_imb_20d)...")
    spearman_df = compute_spearman_vs_existing(panel, kline)
    if not spearman_df.empty:
        spearman_df.to_csv(OUT_SPEARMAN, index=False)
        print(spearman_df.to_string(index=False))
        print(f"[output] spearman → {OUT_SPEARMAN}")

    best = summary_df.iloc[0]
    best_icir_abs = abs(best["icir"]) if pd.notna(best["icir"]) else 0.0
    if best_icir_abs > 0.5:
        verdict = "STRONG — 推荐进 Phase B sidecar OOS"
    elif best_icir_abs > 0.3:
        verdict = "MARGINAL — 可选, 风险中等"
    else:
        verdict = "WEAK — abort, 不做 Phase B"

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    csi_with_events = len(set(csi["code"]) & set(cache["code"].unique()))

    with open(OUT_REPORT, "w") as out:
        out.write("# CSI300 股东户数变化因子 IC 分析 (IS 2014-2020)\n\n")
        out.write("**Universe:** CSI300 (300 只)\n\n")
        out.write(f"**IS 期:** {IS_START.date()} → {IS_END.date()} (84 月)\n\n")
        out.write(f"**Forward return:** {FORWARD_DAYS} 日\n\n")
        out.write(f"**Shareholders cache:** {len(cache):,} 行 × "
                  f"{cache['code'].nunique():,} unique stocks; "
                  f"CSI300 中 {csi_with_events}/300 有 ≥1 记录 "
                  f"({csi_with_events/300*100:.1f}%)\n\n")
        out.write(f"**Panel:** {len(panel)} rows × "
                  f"{panel['code'].nunique()} codes × "
                  f"{panel['asof_date'].nunique()} months\n\n")
        out.write("**PIT 约束:** "
                  "T 月信号 = announce_date ≤ T 的 latest 季度变化\n\n")
        out.write("## 因子非 NaN 率\n\n")
        out.write("| factor | non-NaN% |\n|---|---|\n")
        for f, _ in SINGLE_FACTORS:
            out.write(f"| {f} | {panel[f].notna().mean()*100:.1f}% |\n")
        out.write(f"| combo_neg_chg | "
                  f"{panel['combo_neg_chg'].notna().mean()*100:.1f}% |\n")
        out.write("\n## IC 表 (按 |ICIR| 排序)\n\n")
        out.write(summary_df.to_markdown(index=False))
        if not spearman_df.empty:
            out.write("\n\n## Spearman vs 现有因子\n\n")
            out.write(spearman_df.round(4).to_markdown(index=False))
        out.write("\n\n## 判定规则\n\n")
        out.write("- |ICIR| > 0.5: 强, 进 Phase B sidecar OOS\n")
        out.write("- 0.3 < |ICIR| < 0.5: 边际\n")
        out.write("- |ICIR| < 0.3: 弱, abort\n\n")
        out.write(f"## 结论\n\n**Best factor:** `{best['factor']}` "
                  f"(sign={best['sign']}, ICIR={best['icir']:.3f}, "
                  f"IC mean={best['ic_mean']:.4f}, "
                  f"n_months={best['n_months']})\n\n")
        out.write(f"**Verdict:** {verdict}\n")
    print(f"[output] report → {OUT_REPORT}")
    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"Best factor: {best['factor']} | sign={best['sign']} | "
          f"ICIR={best['icir']:.3f} | IC mean={best['ic_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
