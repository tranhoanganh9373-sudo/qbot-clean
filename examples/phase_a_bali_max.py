"""CSI300 Bali (2011) MAX 彩票因子 Phase A IS IC 分析 — 严格 OOS 协议.

Bali Cakici Whitelaw (JFE 2011): 单日极端高涨幅 → lottery preference → subsequent
reversal. 经典 sign = -1 (lottery aversion).

因子 (3 variants):
  daily_ret = close / prev_close - 1   (PIT, shift(1))
  MAX_1d_5d    = max(daily_ret) over T-5..T-1     (single best day, past week)
  MAX_1d_20d   = max(daily_ret) over T-20..T-1    (single best day, past month)
  MAX_top5_20d = mean(top-5 daily_ret) over T-20..T-1  (Bali variant)

严格约束:
- IS 期 (因子选择): 2014-01-01 ~ 2020-12-31 (~84 月)
- OOS (留作 backtest): 2021-05-01 ~ 2026-04-30 → 本脚本绝不触碰
- 不修改 production / sidecar / 主 cache, 只读 baidu_kline.parquet + backfill
- 期望合并 backfill (2014-2017) + main (2018+) 覆盖 full 84 月

PIT 严格性:
- daily_ret = close[T] / close[T-1] - 1 (group shift(1))
- rolling window 严格 backward-only: rolling over daily_ret.shift(1).rolling(N)
  → 窗口为 T-1..T-N, 不含当前日
- 月度采样在月首交易日 T, factor 已是 T 日可观察值
- forward return = close(T + 21 trading days) / close(T) - 1

正交性 check (vs v19.10 production 因子):
- amp_imb_20d  (主 sidecar, λ=-0.30)
- JZF = (open - prev_close)/prev_close × 100  (stacked, λ=+0.10)

Phase A 通过门槛 (4 项全过):
    |ICIR| >= 0.40
    n_months >= 60
    mean_abs_rho vs amp_imb_20d < 0.30
    mean_abs_rho vs JZF          < 0.30

Artifact warning (memory project_phase_a_2026_05_27_round):
- PB / vol_z_5d 都是 thin-sample (n=35) ICIR 强 → backfill 后 sign flip / dilution
- 本脚本默认合并 backfill → 期望 full 84 月. 若 n < 60 → 严格 ABORT.

输出:
    examples/v22_bali_max_ic.csv         (3 variants × 2 signs = 6 行)
    examples/v22_bali_max_monthly.csv    long-format monthly IC
    examples/v22_bali_max_spearman.csv   vs amp_imb_20d + JZF

运行 (~5-10 min on CSI300 IS panel):
    .venv/bin/python examples/phase_a_bali_max.py
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

OUT_IC = ROOT / "examples" / "v22_bali_max_ic.csv"
OUT_MONTHLY = ROOT / "examples" / "v22_bali_max_monthly.csv"
OUT_SPEARMAN = ROOT / "examples" / "v22_bali_max_spearman.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 21  # 月度 forward return (~1 个月 trading days)
MIN_MONTHLY_OBS = 30  # 每月最少 30 股横截面才算 IC
# rolling需要 20d 历史 + 1d shift → 30 calendar buffer; amp_imb_20d 也只需 ~30
BUFFER_PRE_DAYS = 60
BUFFER_POST = 35

MAX_FACTORS = ["max_1d_5d", "max_1d_20d", "max_top5_20d"]


def load_csi300_codes() -> set[str]:
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    return set(csi["code"].tolist())


def load_merged_kline_is() -> pd.DataFrame:
    """合并 backfill (2014-2017 优先) + main (2018+) for CSI300 IS panel.

    Returns: code, date, open, close, high, low.
    backfill 优先用于 2014-2017 (baidu_kline 2014-2017 CSI300 sparse).
    """
    codes = load_csi300_codes()
    pre_start = IS_START - pd.Timedelta(days=BUFFER_PRE_DAYS)
    post_end = IS_END + pd.Timedelta(days=BUFFER_POST)
    print(
        f"[load] codes={len(codes)} (CSI300), date range "
        f"{pre_start.date()} ~ {post_end.date()}",
        flush=True,
    )

    cols = ["code", "date", "open", "close", "high", "low"]

    bf = pd.read_parquet(BACKFILL_PATH, columns=cols)
    bf["code"] = bf["code"].astype(str).str.zfill(6)
    bf = bf[bf["code"].isin(codes)]
    bf = bf[
        (bf["date"] >= pre_start) & (bf["date"] <= pd.Timestamp("2017-12-31"))
    ]
    print(
        f"[load] backfill 2014-2017 rows={len(bf):,} "
        f"codes={bf['code'].nunique()}",
        flush=True,
    )

    main = pd.read_parquet(KLINE_PATH, columns=cols)
    main["code"] = main["code"].astype(str).str.zfill(6)
    main = main[main["code"].isin(codes)]

    main_2018 = main[
        (main["date"] >= pd.Timestamp("2018-01-01"))
        & (main["date"] <= post_end)
    ]
    print(
        f"[load] main 2018+ rows={len(main_2018):,} "
        f"codes={main_2018['code'].nunique()}",
        flush=True,
    )

    main_pre18 = main[
        (main["date"] >= pre_start)
        & (main["date"] <= pd.Timestamp("2017-12-31"))
    ]
    bf_keys = pd.MultiIndex.from_frame(bf[["code", "date"]])
    main_pre18_keys = pd.MultiIndex.from_frame(main_pre18[["code", "date"]])
    main_pre18_unique = main_pre18[~main_pre18_keys.isin(bf_keys)]
    print(
        f"[load] main 2014-2017 unique (not in backfill) "
        f"rows={len(main_pre18_unique):,}",
        flush=True,
    )

    k = pd.concat([bf, main_pre18_unique, main_2018], ignore_index=True)
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    k = k.drop_duplicates(subset=["code", "date"], keep="first")
    print(
        f"[load] MERGED rows={len(k):,} codes={k['code'].nunique()} "
        f"date {k['date'].min().date()} ~ {k['date'].max().date()}",
        flush=True,
    )

    k_in_is = k[(k["date"] >= IS_START) & (k["date"] <= IS_END)].copy()
    k_in_is["year"] = k_in_is["date"].dt.year
    cov = k_in_is.groupby("year")["code"].nunique()
    print(
        f"[load] yearly CSI300 stock coverage IS 2014-2020:\n{cov.to_string()}",
        flush=True,
    )
    return k


def compute_bali_max_factors(k: pd.DataFrame) -> pd.DataFrame:
    """计算 3 个 Bali MAX 因子 (point-in-time, backward-only rolling).

    daily_ret = close[T] / close[T-1] - 1
    max_1d_5d    = max(daily_ret[T-5..T-1])      single best day past week
    max_1d_20d   = max(daily_ret[T-20..T-1])     single best day past month
    max_top5_20d = mean(top5(daily_ret[T-20..T-1]))  Bali variant

    关键: daily_ret.shift(1).rolling(N) → 窗口 T-1..T-N (PIT 严格).
    """
    print("[factor] computing Bali MAX 因子 (1d_5d, 1d_20d, top5_20d)...", flush=True)
    k = k.copy()
    k["close"] = k["close"].astype(float)
    grp = k.groupby("code", sort=False)
    k["_close_lag1"] = grp["close"].shift(1)
    k["_daily_ret"] = k["close"] / k["_close_lag1"] - 1.0
    grp2 = k.groupby("code", sort=False)
    k["_ret_lag1"] = grp2["_daily_ret"].shift(1)

    grp3 = k.groupby("code", sort=False)
    k["max_1d_5d"] = grp3["_ret_lag1"].transform(
        lambda s: s.rolling(5, min_periods=5).max()
    )
    k["max_1d_20d"] = grp3["_ret_lag1"].transform(
        lambda s: s.rolling(20, min_periods=20).max()
    )

    def _top5_mean(s: pd.Series) -> pd.Series:
        return s.rolling(20, min_periods=20).apply(
            lambda x: np.mean(np.sort(x)[-5:]), raw=True
        )

    k["max_top5_20d"] = grp3["_ret_lag1"].transform(_top5_mean)

    for f in MAX_FACTORS:
        s = k[f]
        if s.notna().any():
            lo, hi = s.quantile(0.005), s.quantile(0.995)
            k[f] = s.clip(lo, hi)

    k = k.drop(columns=["_close_lag1", "_daily_ret", "_ret_lag1"])
    total = len(k)
    print(f"[factor] panel rows={total:,}; coverage:")
    for f in MAX_FACTORS:
        n = int(k[f].notna().sum())
        print(f"  {f:14s} {n:8,d}  ({n/total*100:.1f}%)")
    return k


def compute_amp_imb_20d(k: pd.DataFrame) -> pd.DataFrame:
    """v19.10 production sidecar 因子 (主, λ=-0.30)."""
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
    """v19.10 stacked sidecar 因子 (JZF overnight gap, λ=+0.10)."""
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
    """月度 panel: T = 月首交易日; fwd_ret = close(T+21d)/close(T)-1."""
    print("[panel] building monthly panel (T = 月首交易日)...", flush=True)
    is_dates = pd.DatetimeIndex(sorted(factor_df["date"].unique()))
    is_dates_in = is_dates[(is_dates >= IS_START) & (is_dates <= IS_END)]
    months = pd.Series(is_dates_in).dt.to_period("M")
    month_first = (
        pd.Series(is_dates_in).groupby(months).first().reset_index(drop=True)
    )
    print(
        f"[panel] {len(month_first)} monthly anchor points: "
        f"{month_first.iloc[0].date()} → {month_first.iloc[-1].date()}",
        flush=True,
    )

    wide = factor_df.pivot_table(
        index="date", columns="code", values="close", aggfunc="first"
    ).sort_index()

    f_anchor = factor_df[factor_df["date"].isin(month_first)][
        ["code", "date"] + MAX_FACTORS
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
    for f in MAX_FACTORS:
        zcol = f"z_{f}"
        panel[zcol] = panel.groupby("month_start")[f].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0
        )
    print(
        f"[panel] rows={len(panel):,}, codes={panel['code'].nunique()}, "
        f"months={panel['month_start'].nunique()}",
        flush=True,
    )
    return panel


def monthly_ic(
    panel: pd.DataFrame, factor_col: str, sign: int = 1
) -> tuple[pd.Series, dict]:
    """月度横截面 Spearman corr(factor*sign, fwd_ret)."""
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
        if len(g) < MIN_MONTHLY_OBS:
            return np.nan
        if g["signed"].std() <= 0:
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


def spearman_vs(panel: pd.DataFrame, other_col: str) -> pd.DataFrame:
    rows = []
    for f in MAX_FACTORS:
        zcol = f"z_{f}"
        sub = panel.dropna(subset=[zcol, other_col])
        if sub.empty:
            rows.append({
                "factor": f, "vs": other_col, "n_months": 0,
                "mean_rho": np.nan, "mean_abs_rho": np.nan, "max_abs_rho": np.nan,
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
            "factor": f,
            "vs": other_col,
            "n_months": int(len(per_month)),
            "mean_rho": float(per_month.mean()) if len(per_month) else np.nan,
            "mean_abs_rho": (
                float(per_month.abs().mean()) if len(per_month) else np.nan
            ),
            "max_abs_rho": (
                float(per_month.abs().max()) if len(per_month) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def main() -> int:
    if not KLINE_PATH.exists():
        print(f"FATAL: 缺 {KLINE_PATH}", file=sys.stderr)
        return 1
    if not BACKFILL_PATH.exists():
        print(f"FATAL: 缺 {BACKFILL_PATH}", file=sys.stderr)
        return 1
    if not CSI300_PATH.exists():
        print(f"FATAL: 缺 {CSI300_PATH}", file=sys.stderr)
        return 1

    kline = load_merged_kline_is()
    factor_df = compute_bali_max_factors(kline)
    amp_df = compute_amp_imb_20d(kline)
    jzf_df = compute_jzf(kline)
    panel = build_monthly_panel(factor_df, amp_df, jzf_df)
    if panel.empty:
        print("FATAL: panel 为空", file=sys.stderr)
        return 1

    print("\n[step 2] monthly IC sweep (3 variants × 2 signs = 6 combos)...", flush=True)
    summaries: list[dict] = []
    monthly_series: dict[str, pd.Series] = {}
    for f in MAX_FACTORS:
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
        [
            "factor", "zfactor", "sign", "n_months",
            "ic_mean", "ic_std", "icir",
            "ic_pos_pct", "ic_neg_pct", "avg_obs_per_month",
        ]
    ].round(4)
    summary_df.to_csv(OUT_IC, index=False)
    print(f"\n[output] IC grid -> {OUT_IC}")
    print(summary_df.to_string(index=False))

    monthly_rows = []
    for key, ser in monthly_series.items():
        for m, v in ser.items():
            monthly_rows.append({"variant": key, "month_start": m, "ic": v})
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly IC -> {OUT_MONTHLY}")

    print(
        "\n[step 3] orthogonality vs amp_imb_20d + JZF (v19.10 production)...",
        flush=True,
    )
    orth_amp = spearman_vs(panel, "amp_imb_20d")
    orth_jzf = spearman_vs(panel, "jzf")
    orth_all = pd.concat([orth_amp, orth_jzf], ignore_index=True).round(4)
    orth_all.to_csv(OUT_SPEARMAN, index=False)
    print(orth_all.to_string(index=False))
    print(f"[output] Spearman -> {OUT_SPEARMAN}")

    print("\n[step 4] verdict ...", flush=True)
    top = summary_df.iloc[0]
    icir_abs = abs(float(top["icir"]))
    top_factor = str(top["factor"])
    top_sign = int(top["sign"])
    top_n_months = int(top["n_months"])
    print(
        f"  top variant: factor={top_factor} sign={top_sign:+d} "
        f"ICIR={float(top['icir']):.3f} n_months={top_n_months}"
    )

    amp_row = orth_amp[orth_amp["factor"] == top_factor].iloc[0]
    jzf_row = orth_jzf[orth_jzf["factor"] == top_factor].iloc[0]
    amp_rho = (
        float(amp_row["mean_abs_rho"]) if pd.notna(amp_row["mean_abs_rho"]) else 1.0
    )
    jzf_rho = (
        float(jzf_row["mean_abs_rho"]) if pd.notna(jzf_row["mean_abs_rho"]) else 1.0
    )

    gate1 = icir_abs >= 0.40
    gate2 = top_n_months >= 60
    gate3 = amp_rho < 0.30
    gate4 = jzf_rho < 0.30
    print(f"  gate1 |ICIR|>=0.40: {icir_abs:.3f} → {'PASS' if gate1 else 'FAIL'}")
    print(f"  gate2 n_months>=60: {top_n_months} → {'PASS' if gate2 else 'FAIL'}")
    print(
        f"  gate3 |rho| vs amp_imb_20d < 0.30: {amp_rho:.3f} → "
        f"{'PASS' if gate3 else 'FAIL'}"
    )
    print(
        f"  gate4 |rho| vs JZF < 0.30: {jzf_rho:.3f} → "
        f"{'PASS' if gate4 else 'FAIL'}"
    )
    all_pass = gate1 and gate2 and gate3 and gate4

    if all_pass:
        print("\n  VERDICT: PASS Phase A, RECOMMEND Phase B sidecar OOS test")
        print(f"    factor = {top_factor}")
        print(f"    sign   = {top_sign:+d}")
        print(f"    λ candidates ∈ [0.10, 0.20, 0.30]")
    else:
        fails = []
        if not gate1:
            fails.append(f"|ICIR|={icir_abs:.3f}<0.40")
        if not gate2:
            fails.append(f"n_months={top_n_months}<60")
        if not gate3:
            fails.append(f"|rho|_amp={amp_rho:.3f}>=0.30")
        if not gate4:
            fails.append(f"|rho|_jzf={jzf_rho:.3f}>=0.30")
        print(f"\n  VERDICT: ABORT_Phase_A — {', '.join(fails)}")

    print(
        f"\n[summary] panel rows={len(panel):,}, "
        f"months={panel['month_start'].nunique()}"
    )
    n_recommended = (summary_df["icir"].abs() >= 0.40).sum()
    print(f"[summary] {n_recommended}/6 variants with |ICIR|>=0.40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
