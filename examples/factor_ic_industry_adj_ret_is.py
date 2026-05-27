"""CSI300 industry_adj_ret 因子 IC 分析 - 严格 IS 2014-2020.

CLAUDE.md 严格 OOS 协议:
- IS (因子选择期): 2014-01-01 ~ 2020-12-31 (84 月)
- OOS (留作 backtest): 2021-01-01 ~ 2026-04-30 → 本脚本绝不触碰

因子定义 (point-in-time, no lookahead):
- 每月末日 T_prev (= asof_date, IS 期月末):
  - 用 T_prev 之前 (含 T_prev) 的 close 算 lookback return:
    ret_h(code, T_prev) = close(T_prev) / close(T_prev - h trading_days) - 1
  - 横截面 (same date T_prev) 在 SW 一级行业内 demean:
    industry_adj_ret_h(code, T_prev) = ret_h(code, T_prev)
                                     - mean_{j in industry(code)} ret_h(j, T_prev)
  - 单股票 industry_adj_ret_h 即剥离 sector beta, 保留 intra-industry alpha.
- 前向收益: close(T_next_month_close) / close(T_prev) - 1, ~22 trading days.

多 horizon sweep (IS 仅, 锁定后不再调):
  h ∈ {5, 20, 60} trading days
  sign ∈ {+1 (momentum), -1 (reversion)}
共 3 * 2 = 6 个 (factor, sign) 组合.

Universe: CSI300 (300 只), 100% SW 行业覆盖.

输出:
  examples/factor_ic_industry_adj_ret_is.csv         summary
  examples/factor_ic_industry_adj_ret_is_monthly.csv long-format monthly
  examples/factor_ic_industry_adj_ret_is_report.md   人读

Spearman vs amp_imb_20d 正交性单算 (vs production sidecar 因子).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
INDUSTRY_PATH = ROOT / "data_cache" / "industry" / "industry_membership.parquet"
OUT_CSV = ROOT / "examples" / "factor_ic_industry_adj_ret_is.csv"
OUT_MONTHLY = ROOT / "examples" / "factor_ic_industry_adj_ret_is_monthly.csv"
OUT_REPORT = ROOT / "examples" / "factor_ic_industry_adj_ret_is_report.md"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_TRADE_DAYS = 22  # 1 month
MIN_MONTHLY_OBS = 30
HORIZONS = [5, 20, 60]


def load_kline_is() -> pd.DataFrame:
    """Load baidu_kline (CSI300, IS window + 90d back-buffer for 60d lookback,
    + 35d forward buffer for 22-day fwd return)."""
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = set(csi["code"].tolist())

    k = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    lo = IS_START - pd.Timedelta(days=90)
    hi = IS_END + pd.Timedelta(days=35)
    k = k[(k["date"] >= lo) & (k["date"] <= hi)]
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    return k


def load_industry() -> pd.DataFrame:
    """SW level-1 industry membership; CSI300 only."""
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)

    ind = pd.read_parquet(INDUSTRY_PATH)
    ind["code"] = ind["code"].astype(str).str.zfill(6)
    ind = ind[ind["code"].isin(csi["code"])]
    ind = ind[["code", "industry_code", "industry_name"]].drop_duplicates(
        subset=["code"]
    )
    return ind


def build_panel(kline: pd.DataFrame,
                industry: pd.DataFrame) -> pd.DataFrame:
    """Returns panel with factors + fwd_ret.

    Columns: asof_date (T_prev = 月末日 in IS), code, industry_code,
             ret_5d/20d/60d (raw), industry_adj_ret_5d/20d/60d, fwd_ret.
    """
    wide = (
        kline.pivot_table(index="date", columns="code",
                          values="close", aggfunc="first")
        .sort_index()
    )
    print(f"[wide] {wide.shape[0]} dates × {wide.shape[1]} codes",
          flush=True)

    all_dates = pd.DatetimeIndex(wide.index)
    is_dates = all_dates[(all_dates >= IS_START) & (all_dates <= IS_END)]
    months = pd.Series(is_dates).dt.to_period("M")
    month_last = (
        pd.Series(is_dates).groupby(months).max().reset_index(drop=True)
    )
    print(f"[panel] {len(month_last)} 月末采样点 "
          f"({month_last.iloc[0].date()} → "
          f"{month_last.iloc[-1].date()})", flush=True)

    code_to_ind = dict(zip(industry["code"], industry["industry_code"]))

    rows = []
    skipped_no_forward = 0
    for T in month_last:
        idx_T = wide.index.get_indexer([T])[0]
        if idx_T < 0:
            continue
        idx_fwd = idx_T + FORWARD_TRADE_DAYS
        if idx_fwd >= len(wide.index):
            skipped_no_forward += 1
            continue
        T_close = wide.iloc[idx_T]
        T_fwd_close = wide.iloc[idx_fwd]

        lookbacks = {}
        for h in HORIZONS:
            idx_lb = idx_T - h
            lookbacks[h] = wide.iloc[idx_lb] if idx_lb >= 0 else None

        records = []
        for code in wide.columns:
            c_now = T_close.get(code)
            c_fut = T_fwd_close.get(code)
            ind_code = code_to_ind.get(code)
            if (pd.isna(c_now) or pd.isna(c_fut)
                    or c_now <= 0 or ind_code is None):
                continue
            entry = {
                "code": code,
                "industry_code": ind_code,
                "fwd_ret": c_fut / c_now - 1,
            }
            for h in HORIZONS:
                lb = lookbacks[h]
                if lb is None:
                    continue
                c_back = lb.get(code)
                if pd.isna(c_back) or c_back <= 0:
                    continue
                entry[f"ret_{h}d"] = c_now / c_back - 1
            records.append(entry)
        if not records:
            continue
        df_T = pd.DataFrame.from_records(records)

        for h in HORIZONS:
            col = f"ret_{h}d"
            if col not in df_T.columns:
                continue
            ind_mean = df_T.groupby("industry_code")[col].transform("mean")
            df_T[f"industry_adj_ret_{h}d"] = df_T[col] - ind_mean

        df_T["asof_date"] = T
        rows.append(df_T)

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)
    print(f"[panel] {len(panel):,} rows × "
          f"{panel['code'].nunique()} codes × "
          f"{panel['asof_date'].nunique()} months", flush=True)
    if skipped_no_forward:
        print(f"  skipped {skipped_no_forward} months (no T+22 fwd data)",
              flush=True)

    for h in HORIZONS:
        col = f"industry_adj_ret_{h}d"
        if col in panel.columns:
            disp = panel.groupby(["asof_date", "industry_code"])[col].std()
            print(f"  intra-industry std ({col}): "
                  f"mean={disp.mean():.4f}, "
                  f"median={disp.median():.4f}", flush=True)

    return panel


def monthly_ic(panel: pd.DataFrame, factor_col: str,
               sign: int = 1) -> tuple[pd.Series, dict]:
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    if df.empty:
        return pd.Series(dtype=float), {
            "factor": factor_col, "sign": sign,
            "n_months": 0, "ic_mean": np.nan, "ic_std": np.nan,
            "icir": np.nan, "ic_pos_pct": np.nan,
            "avg_obs_per_month": 0.0,
        }
    df["signed_factor"] = sign * df[factor_col]

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < MIN_MONTHLY_OBS:
            return np.nan
        if g["signed_factor"].std() <= 0:
            return np.nan
        return g["signed_factor"].corr(g["fwd_ret"], method="spearman")

    obs = df.groupby("asof_date").size()
    monthly = df.groupby("asof_date").apply(_corr).dropna()
    if monthly.empty:
        return monthly, {
            "factor": factor_col, "sign": sign,
            "n_months": 0, "ic_mean": np.nan, "ic_std": np.nan,
            "icir": np.nan, "ic_pos_pct": np.nan,
            "avg_obs_per_month": float(obs.mean()) if len(obs) else 0.0,
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
        "avg_obs_per_month": float(obs.mean()),
    }


def orthogonality_vs_amp_imb_20d(panel: pd.DataFrame,
                                 kline_full: pd.DataFrame) -> dict:
    """Compute month-by-month Spearman rho between each industry_adj_ret_*
    factor and amp_imb_20d, then average |rho|."""
    cols_needed = {"code", "date", "high", "low", "close"}
    if not cols_needed.issubset(kline_full.columns):
        return {}
    k = kline_full[list(cols_needed)].copy()
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k.sort_values(["code", "date"]).reset_index(drop=True)

    k["hl"] = k["high"] - k["low"]
    k["hc"] = k["high"] - k["close"]
    k["cl"] = k["close"] - k["low"]
    g = k.groupby("code")
    k["sum_hl_20"] = g["hl"].transform(lambda x: x.rolling(20).sum())
    k["sum_hc_20"] = g["hc"].transform(lambda x: x.rolling(20).sum())
    k["sum_cl_20"] = g["cl"].transform(lambda x: x.rolling(20).sum())
    k["amp_imb_20d"] = (
        (k["sum_hc_20"] - k["sum_cl_20"])
        / k["sum_hl_20"].replace(0, np.nan)
    )
    amp = k[["code", "date", "amp_imb_20d"]].rename(
        columns={"date": "asof_date"}
    )

    merged = panel.merge(amp, on=["code", "asof_date"], how="left")
    n_nonnull = merged["amp_imb_20d"].notna().sum()
    print(f"[orthogonality] amp_imb_20d non-null merge: "
          f"{n_nonnull:,}/{len(panel):,} rows", flush=True)

    results: dict[str, float] = {}
    for h in HORIZONS:
        col = f"industry_adj_ret_{h}d"
        if col not in merged.columns:
            continue
        sub = merged.dropna(subset=[col, "amp_imb_20d"])
        if sub.empty:
            results[col] = np.nan
            continue

        def _rho(g: pd.DataFrame) -> float:
            if len(g) < MIN_MONTHLY_OBS:
                return np.nan
            if g[col].std() <= 0 or g["amp_imb_20d"].std() <= 0:
                return np.nan
            return g[col].corr(g["amp_imb_20d"], method="spearman")

        monthly_rho = sub.groupby("asof_date").apply(_rho).dropna()
        if monthly_rho.empty:
            results[col] = np.nan
            continue
        results[col] = float(monthly_rho.abs().mean())
        print(f"  vs amp_imb_20d / {col}: "
              f"mean |rho|={results[col]:.4f}, "
              f"min={monthly_rho.min():.4f}, "
              f"max={monthly_rho.max():.4f}, "
              f"n_months={len(monthly_rho)}", flush=True)
    return results


def main() -> int:
    if not KLINE_PATH.exists():
        print(f"FATAL: 缺 {KLINE_PATH}", file=sys.stderr)
        return 1
    if not INDUSTRY_PATH.exists():
        print(f"FATAL: 缺 {INDUSTRY_PATH}, 先跑 fetch_industry_em.py",
              file=sys.stderr)
        return 1

    print("[load] industry membership ...")
    ind = load_industry()
    print(f"  CSI300 with industry: {len(ind)}/300 "
          f"({len(ind)/300*100:.1f}%)")
    print(f"  industries: {ind['industry_code'].nunique()}")

    print("\n[load] kline (CSI300 IS+buffer) ...")
    kline = load_kline_is()
    print(f"  kline shape={kline.shape}, "
          f"date range {kline['date'].min().date()} → "
          f"{kline['date'].max().date()}")

    print("\n[load] kline_full (OHLC for amp_imb_20d) ...")
    kline_full = pd.read_parquet(
        KLINE_PATH,
        columns=["code", "date", "high", "low", "close"],
    )
    kline_full["code"] = kline_full["code"].astype(str).str.zfill(6)
    csi_codes = set(ind["code"].tolist())
    kline_full = kline_full[
        (kline_full["code"].isin(csi_codes))
        & (kline_full["date"] >= IS_START - pd.Timedelta(days=90))
        & (kline_full["date"] <= IS_END + pd.Timedelta(days=35))
    ].sort_values(["code", "date"]).reset_index(drop=True)
    print(f"  shape={kline_full.shape}")

    panel = build_panel(kline, ind)
    if panel.empty:
        print("FATAL: panel empty", file=sys.stderr)
        return 1

    summaries: list[dict] = []
    monthly_series: dict[str, pd.Series] = {}

    for h in HORIZONS:
        col = f"industry_adj_ret_{h}d"
        if col not in panel.columns:
            continue
        for sign in (+1, -1):
            m, s = monthly_ic(panel, col, sign=sign)
            label = f"{col}_sign{'+' if sign > 0 else '-'}"
            s_out = dict(s)
            s_out["factor"] = label
            summaries.append(s_out)
            monthly_series[label] = m

    summary_df = pd.DataFrame(summaries).round(4)
    summary_df = summary_df.reindex(
        summary_df["icir"].abs().sort_values(
            ascending=False, na_position="last",
        ).index
    ).reset_index(drop=True)
    summary_df.to_csv(OUT_CSV, index=False)
    print(f"\n[output] summary → {OUT_CSV}")
    print(summary_df.to_string(index=False))

    monthly_rows = []
    for f, ser in monthly_series.items():
        for m, v in ser.items():
            monthly_rows.append({"factor": f, "asof_date": m, "ic": v})
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly → {OUT_MONTHLY}")

    print("\n[orthogonality] vs production sidecar amp_imb_20d ...")
    ortho = orthogonality_vs_amp_imb_20d(panel, kline_full)

    best = summary_df.iloc[0]
    best_abs = abs(best["icir"]) if pd.notna(best["icir"]) else 0.0
    best_factor = str(best["factor"])
    best_monthly = monthly_series.get(best_factor, pd.Series(dtype=float))
    if not best_monthly.empty:
        same_sign_pct = float(
            (np.sign(best_monthly) == np.sign(best["ic_mean"])).mean() * 100
        )
    else:
        same_sign_pct = np.nan

    if best_abs > 0.40 and same_sign_pct >= 60:
        verdict = "STRONG — 推荐进 Phase B sidecar OOS"
    elif best_abs > 0.40:
        verdict = (f"BORDERLINE — ICIR 强但月度同号率 "
                   f"{same_sign_pct:.0f}% < 60%, 风险中等")
    elif best_abs > 0.25:
        verdict = "MARGINAL — 可选, 风险中等"
    else:
        verdict = "WEAK — abort, 不做 Phase B"

    def _ic_turning_points(monthly: pd.Series, top_k: int = 3) -> str:
        if monthly.empty:
            return "—"
        ranks = monthly.abs().sort_values(ascending=False).head(top_k)
        return ", ".join(
            f"{d.strftime('%Y-%m')}({v:+.3f})" for d, v in ranks.items()
        )

    with open(OUT_REPORT, "w") as out:
        out.write("# CSI300 industry_adj_ret 因子 IC 分析 (IS 2014-2020)\n\n")
        out.write("**Universe:** CSI300 (300 只)\n\n")
        out.write(f"**IS 期:** {IS_START.date()} → {IS_END.date()} "
                  f"(theoretical 84 月, actual "
                  f"{panel['asof_date'].nunique()} 月)\n\n")
        out.write(f"**Forward return:** {FORWARD_TRADE_DAYS} trading days "
                  f"(~1 月)\n\n")
        out.write("**Industry source:** SW level-1 (申万一级行业, 31 sectors)\n\n")
        out.write(f"**Industry coverage (CSI300):** {len(ind)}/300 "
                  f"({len(ind)/300*100:.1f}%)\n\n")
        out.write(f"**Panel:** {len(panel):,} rows × "
                  f"{panel['code'].nunique()} codes × "
                  f"{panel['asof_date'].nunique()} months\n\n")
        out.write("## Caveat: 行业归属 snapshot\n\n")
        out.write("akshare `sw_index_first_info` + `index_component_sw` "
                  "给的是**当前 (2021-12 SW 最新调整) 归属**.\n")
        out.write("2014-2020 早期 SW 行业可能略有不同, 但 SW level-1 "
                  "(31 个一级行业) 长期稳定, "
                  "对 sector-relative momentum 影响有限.\n\n")
        out.write("## IC 表 (按 |ICIR| 排序)\n\n")
        out.write(summary_df.to_markdown(index=False))
        out.write("\n\n## IC Time Series 转折点 (top |IC| 月份)\n\n")
        for f in summary_df["factor"].tolist():
            tp = _ic_turning_points(monthly_series.get(f, pd.Series()))
            out.write(f"- `{f}`: {tp}\n")
        out.write("\n## Orthogonality vs production sidecar (amp_imb_20d)\n\n")
        out.write("月度横截面 Spearman |rho| 平均 "
                  "(独立性必要不充分; OOS 单测才知是否可叠加).\n\n")
        if ortho:
            for k_, v_ in ortho.items():
                if pd.isna(v_):
                    continue
                tag = ("INDEPENDENT (|rho|<0.10)" if v_ < 0.10
                       else "MODERATE (0.10~0.30)" if v_ < 0.30
                       else "CORRELATED (|rho|>0.30)")
                out.write(f"- `{k_}` vs `amp_imb_20d`: mean |rho| = "
                          f"{v_:.4f}  [{tag}]\n")
        else:
            out.write("(amp_imb_20d 计算失败 — 缺 high/low 列?)\n")
        out.write("\n## 判定规则\n\n")
        out.write("- |ICIR| > 0.40 且月度 IC 同号率 ≥ 60%: STRONG → Phase B\n")
        out.write("- |ICIR| > 0.40 但同号率 < 60%: BORDERLINE\n")
        out.write("- 0.25 < |ICIR| < 0.40: MARGINAL\n")
        out.write("- |ICIR| < 0.25: WEAK → abort\n\n")
        out.write(f"## 结论\n\n**Best factor:** `{best['factor']}` "
                  f"(ICIR = {best['icir']:.3f}, "
                  f"IC mean = {best['ic_mean']:.4f}, "
                  f"同号率 = {same_sign_pct:.0f}%, "
                  f"n_months = {best['n_months']})\n\n")
        out.write(f"**Verdict:** {verdict}\n\n")
        if best_abs > 0.40:
            best_h = int(best['factor'].split('_')[3].rstrip('d'))
            best_sign = +1 if best['factor'].endswith('+') else -1
            out.write("## Phase B 锁定 (不可改)\n\n")
            out.write(f"- factor: `industry_adj_ret_{best_h}d`\n")
            out.write(f"- sign: "
                      f"{'+1 (momentum)' if best_sign > 0 else '-1 (reversion)'}\n")
            out.write(f"- horizon: {best_h} trading days\n")
            out.write("- λ candidate set (Phase B sweep): "
                      "{0.05, 0.10, 0.20, 0.30}\n")
    print(f"[output] report → {OUT_REPORT}")
    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"Best factor: {best['factor']} | ICIR={best['icir']:.3f} | "
          f"IC mean={best['ic_mean']:.4f} | "
          f"same-sign pct={same_sign_pct:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
