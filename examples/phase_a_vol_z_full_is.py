"""CSI300 volume z-score 因子 Phase A IS IC 分析 — full 84 月 verification.

Goal: verify whether previous Phase A vol_z_5d ICIR=+0.668 was a thin-sample
artifact (only 35 months covered after MIN_MONTHLY_OBS=30 filter).

主表 baidu_kline.parquet IS 期 CSI300 2014-2017 仅 10-25 stocks (大蓝筹 first_date
≥ 2018) → 35 月通过 MIN_OBS=30 阈值. backfill 文件
`data_cache/kline_2014_2017_csi300_backfill.parquet` 含 235 codes × 2014-2017
of CSI300 (无 turnoverratio 但有 vol). 合并后期望 covered months ~84.

Merge 策略:
  - backfill (2014-2017 CSI300 235 codes) + main (全期, 2018+ 用作 IS 主体)
  - 主表 2014-2017 也有少量 (10-25 stock) 数据: 与 backfill union, dedup by (code, date)
    backfill 优先 (它是为 CSI300 补缺 mood 数据)

5 vol factors (task spec):
    vol_z_5d   = (vol[T] - mean(vol_lag1, 5d))  / std(vol_lag1, 5d)
    vol_z_20d  = (vol[T] - mean(vol_lag1, 20d)) / std(vol_lag1, 20d)
    vol_z_60d  = (vol[T] - mean(vol_lag1, 60d)) / std(vol_lag1, 60d)
    vol_chg_5d = (mean(vol_lag1, 5d)  / mean(vol_lag1, 20d)) - 1   (task spec)
    vol_chg_20d= (mean(vol_lag1, 20d) / mean(vol_lag1, 60d)) - 1   (task spec)

(注意: task spec 的 vol_chg 是 mean-ratio, 与旧脚本 vol[T]/vol[T-N]-1 不同 — 这里
按 task 要求实现新口径, 与 vol_z 横截面 IC 直接对比.)

Orthogonality factors (per task):
  - amp_imb_20d  (v19.6 production sidecar)
  - JZF (overnight gap, v19.10 stacked sidecar)
        = (open - prev_close) / prev_close × 100

读: baidu_kline.parquet + kline_2014_2017_csi300_backfill.parquet (READ-ONLY).
写: examples/v22_vol_z_full_is_{ic, monthly, spearman}.csv. 不修改其他文件.
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

OUT_GRID = ROOT / "examples" / "v22_vol_z_full_is_ic.csv"
OUT_MONTHLY = ROOT / "examples" / "v22_vol_z_full_is_monthly.csv"
OUT_SPEARMAN = ROOT / "examples" / "v22_vol_z_full_is_spearman.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 21
MIN_MONTHLY_OBS = 30
BUFFER_POST = 35
BUFFER_PRE_DAYS = 120

VOL_FACTORS = ["vol_z_5d", "vol_z_20d", "vol_z_60d", "vol_chg_5d", "vol_chg_20d"]


def load_csi300_codes() -> set[str]:
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    return set(csi["code"].tolist())


def load_merged_kline_is() -> pd.DataFrame:
    """合并 backfill (2014-2017) + main (2018+) for CSI300 IS panel."""
    codes = load_csi300_codes()
    pre_start = IS_START - pd.Timedelta(days=BUFFER_PRE_DAYS)
    post_end = IS_END + pd.Timedelta(days=BUFFER_POST)
    print(f"[load] codes={len(codes)} (CSI300), date range "
          f"{pre_start.date()} ~ {post_end.date()}", flush=True)

    bf = pd.read_parquet(
        BACKFILL_PATH,
        columns=["code", "date", "open", "close", "high", "low", "vol"],
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
        columns=["code", "date", "open", "close", "high", "low", "vol"],
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


def compute_volume_factors(k: pd.DataFrame) -> pd.DataFrame:
    """5 vol 因子 (point-in-time, backward-only rolling)."""
    print("[factor] computing vol_z_5/20/60 + vol_chg_5/20 ...", flush=True)
    k = k.copy()
    k["vol"] = k["vol"].astype(float)
    grp = k.groupby("code", sort=False)
    k["_vol_lag1"] = grp["vol"].shift(1)
    grp2 = k.groupby("code", sort=False)

    for N in (5, 20, 60):
        rmean = grp2["_vol_lag1"].transform(
            lambda s, n=N: s.rolling(n, min_periods=n).mean()
        )
        rstd = grp2["_vol_lag1"].transform(
            lambda s, n=N: s.rolling(n, min_periods=n).std()
        )
        k[f"vol_z_{N}d"] = (k["vol"] - rmean) / rstd.replace(0, np.nan)

    rmean5 = grp2["_vol_lag1"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )
    rmean20 = grp2["_vol_lag1"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    rmean60 = grp2["_vol_lag1"].transform(
        lambda s: s.rolling(60, min_periods=60).mean()
    )
    k["vol_chg_5d"] = rmean5 / rmean20.replace(0, np.nan) - 1
    k["vol_chg_20d"] = rmean20 / rmean60.replace(0, np.nan) - 1

    for f in VOL_FACTORS:
        s = k[f]
        if s.notna().any():
            lo, hi = s.quantile(0.005), s.quantile(0.995)
            k[f] = s.clip(lo, hi)

    k = k.drop(columns=["_vol_lag1"])
    total = len(k)
    print(f"[factor] panel rows={total:,}; coverage:")
    for f in VOL_FACTORS:
        n = int(k[f].notna().sum())
        print(f"  {f:14s} {n:8,d}  ({n/total*100:.1f}%)")
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
    month_first = (
        pd.Series(is_dates_in).groupby(months).first().reset_index(drop=True)
    )
    print(f"[panel] {len(month_first)} monthly anchor points: "
          f"{month_first.iloc[0].date()} → {month_first.iloc[-1].date()}",
          flush=True)

    wide = factor_df.pivot_table(
        index="date", columns="code", values="close", aggfunc="first"
    ).sort_index()

    f_anchor = factor_df[factor_df["date"].isin(month_first)][
        ["code", "date"] + VOL_FACTORS
    ]
    a_anchor = amp_df[amp_df["date"].isin(month_first)][
        ["code", "date", "amp_imb_20d"]
    ]
    j_anchor = jzf_df[jzf_df["date"].isin(month_first)][
        ["code", "date", "jzf"]
    ]
    f_groups = dict(list(f_anchor.groupby("date", sort=False)))
    a_groups = dict(list(a_anchor.groupby("date", sort=False)))
    j_groups = dict(list(j_anchor.groupby("date", sort=False)))

    rows = []
    for T in month_first:
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

    print("[panel] cross-section z-score per month for each factor...", flush=True)
    for f in VOL_FACTORS:
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
    """月度横截面 Spearman corr(z_vol_factor, amp_imb_20d) + (z_vol_factor, jzf)."""
    rows = []
    for f in VOL_FACTORS:
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
    factor_df = compute_volume_factors(k)
    amp_df = compute_amp_imb_20d(k)
    jzf_df = compute_jzf(k)
    panel = build_monthly_panel(factor_df, amp_df, jzf_df)
    if panel.empty:
        print("FATAL: panel 为空", file=sys.stderr); return 1

    print("\n[step 2] monthly cross-section IC sweep (5 horizons × 2 signs)...",
          flush=True)
    summaries: list[dict] = []
    monthly_series: dict[str, pd.Series] = {}
    for f in VOL_FACTORS:
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

    print("\n[step 3] orthogonality vs amp_imb_20d + JZF ...", flush=True)
    orth = spearman_vs_existing(panel)
    orth.to_csv(OUT_SPEARMAN, index=False)
    print(orth.to_string(index=False))
    print(f"[output] Spearman -> {OUT_SPEARMAN}")

    print("\n[step 4] verdict (vs old thin-sample ICIR=+0.668 for vol_z_5d sign=+1) ...",
          flush=True)
    row5p = summary_df[
        (summary_df["factor"] == "vol_z_5d") & (summary_df["sign"] == 1)
    ]
    if not row5p.empty:
        new_icir = float(row5p.iloc[0]["icir"])
        n = int(row5p.iloc[0]["n_months"])
        print(f"  vol_z_5d sign=+1: OLD thin (n=35) ICIR=+0.668  "
              f"NEW full (n={n}) ICIR={new_icir:+.3f}")
        ratio = abs(new_icir) / 0.668
        sign_flip = (new_icir < 0)
        print(f"    decay ratio = {ratio:.2%}; sign_flip = {sign_flip}")

    top = summary_df.iloc[0]
    print(f"  top variant overall: factor={top['factor']} "
          f"sign={int(top['sign'])} n_months={int(top['n_months'])} "
          f"ICIR={float(top['icir']):.3f}")
    print(f"\n[summary] panel rows={len(panel):,}, "
          f"months={panel['month_start'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
