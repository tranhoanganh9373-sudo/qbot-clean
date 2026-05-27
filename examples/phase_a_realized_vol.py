"""CSI300 Realized Volatility short-term 因子 Phase A IS IC 分析.

Goal: 测 RV 3 horizons (5/20/60d) × 2 signs (±1) = 6 combos 的 IS IC, 重点
verify **RV 与 amp_imb_20d / JZF 的 Spearman 重叠风险** — RV 与 amp_imb_20d 都基
于 daily-ret based, 可能高度相关.

经典 low-vol premium sign = -1 (低波动股长期跑赢). 期望 ICIR sign=-1 为正.

Data merge:
  - backfill (2014-2017 CSI300 235 codes, kline_2014_2017_csi300_backfill.parquet)
  - main baidu_kline.parquet (全期, 2018+ 用作 IS 主体)
  - dedup by (code, date), backfill 优先
  - CSI300 universe via data_cache/csi300_constituents.csv

RV 公式 (per task spec):
    daily_ret = close / prev_close - 1
    RV_5d  = sqrt( sum(daily_ret^2) over trailing 5 days )
    RV_20d = sqrt( sum(daily_ret^2) over trailing 20 days )
    RV_60d = sqrt( sum(daily_ret^2) over trailing 60 days )

Orthogonality factors (per task):
  - amp_imb_20d (v19.6 production sidecar) — 20d 振幅不对称, daily-ret based
  - JZF (v19.10 stacked sidecar) — overnight gap = (open - prev_close) / prev_close × 100

Gate (per task):
  - ICIR ≥ 0.4 AND n_months ≥ 60 AND |Spearman ρ| < 0.30 vs amp_imb_20d & JZF
  - 即使 ICIR 过 + n_months 过, 如果 Spearman 不过 → abort (类似 52w high)

READ-ONLY: paper_trade*.py, strategy_v*.py, baidu_kline.parquet
NEW outputs:
  examples/v22_rv_ic.csv
  examples/v22_rv_monthly.csv
  examples/v22_rv_spearman.csv
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
BACKFILL_PATH = ROOT / "data_cache" / "kline_2014_2017_csi300_backfill.parquet"

OUT_GRID = ROOT / "examples" / "v22_rv_ic.csv"
OUT_MONTHLY = ROOT / "examples" / "v22_rv_monthly.csv"
OUT_SPEARMAN = ROOT / "examples" / "v22_rv_spearman.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 21
MIN_MONTHLY_OBS = 30
BUFFER_POST = 35
BUFFER_PRE_DAYS = 120

RV_FACTORS = ["rv_5d", "rv_20d", "rv_60d"]


def load_csi300_codes() -> set[str]:
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    return set(csi["code"].tolist())


def load_merged_kline_is() -> pd.DataFrame:
    """Merge backfill (2014-2017) + main (2018+) for CSI300 IS panel."""
    codes = load_csi300_codes()
    pre_start = IS_START - pd.Timedelta(days=BUFFER_PRE_DAYS)
    post_end = IS_END + pd.Timedelta(days=BUFFER_POST)
    print(f"[load] codes={len(codes)} (CSI300), range "
          f"{pre_start.date()} ~ {post_end.date()}", flush=True)

    bf = pd.read_parquet(
        BACKFILL_PATH,
        columns=["code", "date", "open", "close", "high", "low"],
    )
    bf["code"] = bf["code"].astype(str).str.zfill(6)
    bf = bf[bf["code"].isin(codes)]
    bf = bf[
        (bf["date"] >= pre_start)
        & (bf["date"] <= pd.Timestamp("2017-12-31"))
    ]
    print(f"[load] backfill 2014-2017 rows={len(bf):,} "
          f"codes={bf['code'].nunique()}", flush=True)

    main = pd.read_parquet(
        KLINE_PATH,
        columns=["code", "date", "open", "close", "high", "low"],
    )
    main["code"] = main["code"].astype(str).str.zfill(6)
    main = main[main["code"].isin(codes)]
    main_full = main.copy()

    main_2018 = main[
        (main["date"] >= pd.Timestamp("2018-01-01"))
        & (main["date"] <= post_end)
    ]
    print(f"[load] main 2018+ rows={len(main_2018):,} "
          f"codes={main_2018['code'].nunique()}", flush=True)

    main_pre18 = main_full[
        (main_full["date"] >= pre_start)
        & (main_full["date"] <= pd.Timestamp("2017-12-31"))
    ]
    bf_keys = pd.MultiIndex.from_frame(bf[["code", "date"]])
    main_pre18_keys = pd.MultiIndex.from_frame(main_pre18[["code", "date"]])
    main_pre18_unique = main_pre18[~main_pre18_keys.isin(bf_keys)]
    print(f"[load] main 2014-2017 unique (not in backfill) "
          f"rows={len(main_pre18_unique):,}", flush=True)

    k = pd.concat([bf, main_pre18_unique, main_2018], ignore_index=True)
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    k = k.drop_duplicates(subset=["code", "date"], keep="first")
    print(f"[load] MERGED rows={len(k):,} codes={k['code'].nunique()} "
          f"date {k['date'].min().date()} ~ {k['date'].max().date()}",
          flush=True)

    k_in_is = k[(k["date"] >= IS_START) & (k["date"] <= IS_END)].copy()
    k_in_is["year"] = k_in_is["date"].dt.year
    cov = k_in_is.groupby("year")["code"].nunique()
    print(f"[load] yearly CSI300 stock coverage IS 2014-2020:\n{cov.to_string()}",
          flush=True)
    return k


def compute_rv_factors(k: pd.DataFrame) -> pd.DataFrame:
    """Realized vol = sqrt( sum(daily_ret^2) over trailing N days ).

    Backward-only rolling, no look-ahead. daily_ret[T] = close[T]/close[T-1]-1
    is known at T close, so legitimate for cross-section IC at month-end T
    against forward 21d return.
    """
    print("[factor] computing rv_5/20/60 ...", flush=True)
    k = k.copy()
    k["close"] = k["close"].astype(float)
    grp = k.groupby("code", sort=False)
    k["prev_close"] = grp["close"].shift(1)
    k["daily_ret"] = np.where(
        k["prev_close"] > 0, k["close"] / k["prev_close"] - 1, np.nan
    )
    k["sq_ret"] = k["daily_ret"].pow(2)
    grp2 = k.groupby("code", sort=False)
    for N in (5, 20, 60):
        s = grp2["sq_ret"].transform(
            lambda s, n=N: s.rolling(n, min_periods=n).sum()
        )
        k[f"rv_{N}d"] = np.sqrt(s)

    for f in RV_FACTORS:
        s = k[f]
        if s.notna().any():
            lo, hi = s.quantile(0.005), s.quantile(0.995)
            k[f] = s.clip(lo, hi)

    k = k.drop(columns=["prev_close", "daily_ret", "sq_ret"])
    total = len(k)
    print(f"[factor] panel rows={total:,}; coverage:")
    for f in RV_FACTORS:
        n = int(k[f].notna().sum())
        print(f"  {f:10s} {n:8,d}  ({n/total*100:.1f}%)")
    return k


def compute_amp_imb_20d(k: pd.DataFrame) -> pd.DataFrame:
    """v19.6 sidecar 因子: amp imbalance 20d."""
    print("[orth] computing amp_imb_20d ...", flush=True)
    df = k[["code", "date", "open", "high", "low", "close"]].copy()
    df["prev_close"] = df.groupby("code", sort=False)["close"].shift(1)
    valid = df["prev_close"].notna() & (df["prev_close"] > 0) & (df["open"] > 0)
    amp = (df["high"] - df["low"]) / df["prev_close"]
    delta = df["close"] - df["prev_close"]
    amp_up = np.where(delta > 0, delta, 0.0) / df["prev_close"] * amp
    amp_dn = np.where(delta < 0, -delta, 0.0) / df["prev_close"] * amp
    df["amp"] = amp
    df["amp_up"] = amp_up
    df["amp_dn"] = amp_dn
    df.loc[~valid, ["amp", "amp_up", "amp_dn"]] = np.nan
    grp = df.groupby("code", sort=False)
    sum_amp = grp["amp"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    sum_up = grp["amp_up"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    sum_dn = grp["amp_dn"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    df["amp_imb_20d"] = np.where(sum_amp > 0, (sum_up - sum_dn) / sum_amp, np.nan)
    return df[["code", "date", "amp_imb_20d"]]


def compute_jzf(k: pd.DataFrame) -> pd.DataFrame:
    """v19.10 sidecar 因子: overnight gap."""
    print("[orth] computing JZF (overnight gap) ...", flush=True)
    df = k[["code", "date", "open", "close"]].copy()
    df["prev_close"] = df.groupby("code", sort=False)["close"].shift(1)
    df["jzf"] = np.where(
        df["prev_close"] > 0,
        (df["open"] - df["prev_close"]) / df["prev_close"] * 100,
        np.nan,
    )
    return df[["code", "date", "jzf"]]


def build_monthly_panel(
    factor_df: pd.DataFrame,
    amp_df: pd.DataFrame,
    jzf_df: pd.DataFrame,
) -> pd.DataFrame:
    print("[panel] building monthly panel...", flush=True)
    is_dates = pd.DatetimeIndex(sorted(factor_df["date"].unique()))
    is_dates_in = is_dates[(is_dates >= IS_START) & (is_dates <= IS_END)]
    months = pd.Series(is_dates_in).dt.to_period("M")
    month_last = (
        pd.Series(is_dates_in).groupby(months).last().reset_index(drop=True)
    )
    print(f"[panel] {len(month_last)} monthly anchor points (month-end): "
          f"{month_last.iloc[0].date()} → {month_last.iloc[-1].date()}",
          flush=True)

    wide = factor_df.pivot_table(
        index="date", columns="code", values="close", aggfunc="first"
    ).sort_index()

    f_anchor = factor_df[factor_df["date"].isin(month_last)][
        ["code", "date"] + RV_FACTORS
    ]
    a_anchor = amp_df[amp_df["date"].isin(month_last)][
        ["code", "date", "amp_imb_20d"]
    ]
    j_anchor = jzf_df[jzf_df["date"].isin(month_last)][
        ["code", "date", "jzf"]
    ]
    f_groups = dict(list(f_anchor.groupby("date", sort=False)))
    a_groups = dict(list(a_anchor.groupby("date", sort=False)))
    j_groups = dict(list(j_anchor.groupby("date", sort=False)))

    rows = []
    for T in month_last:
        idx = wide.index.get_indexer([T])[0]
        if idx < 0 or idx + FORWARD_DAYS >= len(wide.index):
            continue
        T_close = wide.iloc[idx]
        T_plus = wide.iloc[idx + FORWARD_DAYS]
        fwd = T_plus / T_close - 1
        df_T = pd.DataFrame({"fwd_ret": fwd})
        df_T["code"] = df_T.index.astype(str)
        df_T = df_T.dropna(subset=["fwd_ret"]).reset_index(drop=True)
        f_T = f_groups.get(T)
        if f_T is None or f_T.empty:
            continue
        df_T = df_T.merge(f_T.drop(columns=["date"]), on="code", how="left")
        a_T = a_groups.get(T)
        if a_T is not None and not a_T.empty:
            df_T = df_T.merge(a_T.drop(columns=["date"]), on="code", how="left")
        else:
            df_T["amp_imb_20d"] = np.nan
        j_T = j_groups.get(T)
        if j_T is not None and not j_T.empty:
            df_T = df_T.merge(j_T.drop(columns=["date"]), on="code", how="left")
        else:
            df_T["jzf"] = np.nan
        df_T["month_start"] = T
        rows.append(df_T)

    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if panel.empty:
        return panel

    print("[panel] cross-section z-score per month ...", flush=True)
    for f in RV_FACTORS:
        zcol = f"z_{f}"
        panel[zcol] = panel.groupby("month_start")[f].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0
        )
    print(f"[panel] rows={len(panel):,}, codes={panel['code'].nunique()}, "
          f"months={panel['month_start'].nunique()}", flush=True)
    return panel


def monthly_ic(
    panel: pd.DataFrame, factor_col: str, sign: int = 1
) -> tuple[pd.Series, dict]:
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    if df.empty:
        return pd.Series(dtype=float), {
            "factor": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
            "ic_pos_pct": 0.0, "ic_neg_pct": 0.0,
            "avg_obs_per_month": 0.0,
        }
    df["signed"] = sign * df[factor_col]

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < MIN_MONTHLY_OBS or g["signed"].std() <= 0:
            return np.nan
        return g["signed"].corr(g["fwd_ret"], method="spearman")

    obs = df.groupby("month_start").size()
    monthly = df.groupby("month_start").apply(_corr).dropna()
    if len(monthly) == 0:
        return monthly, {
            "factor": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
            "ic_pos_pct": 0.0, "ic_neg_pct": 0.0,
            "avg_obs_per_month": float(obs.mean()) if len(obs) else 0.0,
        }
    mean = float(monthly.mean())
    std = float(monthly.std())
    icir = mean / std * np.sqrt(12) if std > 0 else 0.0
    return monthly, {
        "factor": factor_col, "sign": sign, "n_months": int(len(monthly)),
        "ic_mean": mean, "ic_std": std, "icir": icir,
        "ic_pos_pct": float((monthly > 0).mean() * 100),
        "ic_neg_pct": float((monthly < 0).mean() * 100),
        "avg_obs_per_month": float(obs.mean()),
    }


def spearman_vs_existing(panel: pd.DataFrame) -> pd.DataFrame:
    """月度横截面 Spearman corr(z_rv, amp_imb_20d) + (z_rv, jzf).

    关键 — RV 与 amp_imb_20d 重叠风险 (都是 daily price-based).
    """
    rows = []
    for f in RV_FACTORS:
        zcol = f"z_{f}"
        for other_col in ("amp_imb_20d", "jzf"):
            sub = panel.dropna(subset=[zcol, other_col])
            if sub.empty:
                rows.append({
                    "factor": f, "vs": other_col, "n_months": 0,
                    "mean_rho": np.nan, "mean_abs_rho": np.nan,
                    "max_abs_rho": np.nan,
                })
                continue
            per_month = sub.groupby("month_start").apply(
                lambda g: g[zcol].corr(g[other_col], method="spearman")
                if len(g) >= MIN_MONTHLY_OBS
                and g[zcol].std() > 0
                and g[other_col].std() > 0
                else np.nan
            ).dropna()
            rows.append({
                "factor": f, "vs": other_col,
                "n_months": int(len(per_month)),
                "mean_rho": float(per_month.mean()) if len(per_month) else np.nan,
                "mean_abs_rho": float(per_month.abs().mean()) if len(per_month) else np.nan,
                "max_abs_rho": float(per_month.abs().max()) if len(per_month) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> int:
    if not KLINE_PATH.exists():
        print(f"FATAL: 缺 {KLINE_PATH}", file=sys.stderr); return 1
    if not BACKFILL_PATH.exists():
        print(f"FATAL: 缺 {BACKFILL_PATH}", file=sys.stderr); return 1
    if not CSI300_PATH.exists():
        print(f"FATAL: 缺 {CSI300_PATH}", file=sys.stderr); return 1

    k = load_merged_kline_is()
    factor_df = compute_rv_factors(k)
    amp_df = compute_amp_imb_20d(k)
    jzf_df = compute_jzf(k)
    panel = build_monthly_panel(factor_df, amp_df, jzf_df)
    if panel.empty:
        print("FATAL: panel 为空", file=sys.stderr); return 1

    print("\n[step 2] monthly cross-section IC sweep (3 horizons × 2 signs)...",
          flush=True)
    summaries: list[dict] = []
    monthly_series: dict[str, pd.Series] = {}
    for f in RV_FACTORS:
        zcol = f"z_{f}"
        for sign in (+1, -1):
            m, s = monthly_ic(panel, zcol, sign=sign)
            s["factor"] = f
            s["zfactor"] = zcol
            s["sign"] = sign
            summaries.append(s)
            monthly_series[f"{f}_sign{sign:+d}"] = m

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.reindex(
        summary_df["icir"].abs().sort_values(ascending=False).index
    )
    summary_df = summary_df[
        ["factor", "zfactor", "sign", "n_months",
         "ic_mean", "ic_std", "icir",
         "ic_pos_pct", "ic_neg_pct", "avg_obs_per_month"]
    ].round(4)
    summary_df.to_csv(OUT_GRID, index=False)
    print(f"\n[output] IC grid -> {OUT_GRID}")
    print(summary_df.to_string(index=False))

    monthly_rows = []
    for key, ser in monthly_series.items():
        for mth, v in ser.items():
            monthly_rows.append({"variant": key, "month_start": mth, "ic": v})
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly IC -> {OUT_MONTHLY}")

    print("\n[step 3] orthogonality vs amp_imb_20d + JZF (CRITICAL — overlap risk)...",
          flush=True)
    orth = spearman_vs_existing(panel)
    orth.to_csv(OUT_SPEARMAN, index=False)
    print(orth.to_string(index=False))
    print(f"[output] Spearman -> {OUT_SPEARMAN}")

    print("\n[step 4] verdict (gate: ICIR ≥ 0.4 AND n_months ≥ 60 AND |rho|<0.30)",
          flush=True)
    top = summary_df.iloc[0]
    print(f"  top variant: factor={top['factor']} sign={int(top['sign'])} "
          f"n_months={int(top['n_months'])} ICIR={float(top['icir']):.3f}")
    top_factor = top["factor"]
    rho_amp = orth[(orth["factor"] == top_factor) & (orth["vs"] == "amp_imb_20d")]
    rho_jzf = orth[(orth["factor"] == top_factor) & (orth["vs"] == "jzf")]
    if not rho_amp.empty:
        print(f"  vs amp_imb_20d: mean_rho={rho_amp.iloc[0]['mean_rho']:+.3f}  "
              f"mean_abs_rho={rho_amp.iloc[0]['mean_abs_rho']:.3f}  "
              f"max_abs_rho={rho_amp.iloc[0]['max_abs_rho']:.3f}")
    if not rho_jzf.empty:
        print(f"  vs JZF:         mean_rho={rho_jzf.iloc[0]['mean_rho']:+.3f}  "
              f"mean_abs_rho={rho_jzf.iloc[0]['mean_abs_rho']:.3f}  "
              f"max_abs_rho={rho_jzf.iloc[0]['max_abs_rho']:.3f}")
    print(f"\n[summary] panel rows={len(panel):,}, "
          f"months={panel['month_start'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
