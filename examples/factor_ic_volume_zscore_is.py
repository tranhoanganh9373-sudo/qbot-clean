"""CSI300 volume z-score 因子 IS IC 分析 (Phase A) — 严格 OOS 协议.

任务: 评估 volume z-score / pct-change 多 horizon 在 IS 期 2014-2020 的 alpha,
为 Phase B sidecar 决定 (factor, sign, horizon, λ candidate).

严格约束:
- IS 期 (因子选择): 2014-01-01 ~ 2020-12-31 (84 月)
- OOS (留作 backtest): 2021-01-01 ~ 2026-04-30 → **本脚本绝不触碰**
- 不修改任何 production 文件 / sidecar 脚本 / 数据 cache
- 只读 baidu_kline.parquet (v3 clean hfq, 4621 codes / 7.9M rows)

候选因子 (5 horizon):
    vol_z_5d   = (vol[T] - mean(vol, T-5..T-1)) / std(vol, T-5..T-1)
    vol_z_20d  = 20d backward-only window
    vol_z_60d  = 60d backward-only window
    vol_chg_5d = vol[T] / vol[T-5] - 1
    vol_chg_20d= vol[T] / vol[T-20] - 1

机制可能性 (sign 双向都试):
    +1: 量价共振 (放量上涨,持续强势)
    -1: 放量反转 (异常放量后均值回归)

PIT (point-in-time) 严格性:
- rolling window 不含当前日 (.shift(1) + rolling(N)) → 避免 look-ahead
- factor 在月度采样日 T 上计算, 但 rolling 用 T-N..T-1 数据
- forward return = close(T + 22 trading days) / close(T) - 1
- cross-section z-score: 每月横截面再标准化一次 (跨股相对强度)

方法 (与 factor_ic_technical_csi300_is.py 一致):
- 月度采样 T = 每月第 1 个 IS 交易日
- 每月 cross-section Spearman corr(z_factor * sign, fwd_22d_ret) → 月度 IC
- ICIR = mean(IC) / std(IC) * sqrt(12)

判定:
- |ICIR| >= 0.40 + clean sign + same-sign 月 >= 55%: 推荐 Phase B
- 0.20 <= |ICIR| < 0.40 边际: 不推荐
- |ICIR| < 0.20: abort

输出:
    examples/v20_volume_zscore_is_grid.csv         summary 表 (5 horizon × 2 sign)
    examples/v20_volume_zscore_is_monthly.csv      long-format monthly IC
    examples/v20_volume_zscore_is_spearman.csv     vs amp_imb_20d Spearman |rho|

运行 (~15-25 min on full CSI300 panel):
    .venv/bin/python examples/factor_ic_volume_zscore_is.py
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
OUT_GRID = ROOT / "examples" / "v20_volume_zscore_is_grid.csv"
OUT_MONTHLY = ROOT / "examples" / "v20_volume_zscore_is_monthly.csv"
OUT_SPEARMAN = ROOT / "examples" / "v20_volume_zscore_is_spearman.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 22  # 月度 forward return
MIN_MONTHLY_OBS = 30  # 每月最少 30 股横截面才算 IC
BUFFER_DAYS = 35  # for forward 22 trading days

VOL_FACTORS = ["vol_z_5d", "vol_z_20d", "vol_z_60d", "vol_chg_5d", "vol_chg_20d"]


def load_csi300_codes() -> list[str]:
    """Load CSI300 6-digit codes."""
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    return csi["code"].tolist()


def load_kline_is_with_buffer() -> pd.DataFrame:
    """读 baidu_kline.parquet IS 期 CSI300, 含前后 buffer.

    Pre-buffer: 120 calendar days for vol_z_60d rolling.
    Post-buffer: 35 calendar days for forward 22d return.
    """
    codes = set(load_csi300_codes())
    pre_start = IS_START - pd.Timedelta(days=120)
    post_end = IS_END + pd.Timedelta(days=BUFFER_DAYS)
    k = pd.read_parquet(
        KLINE_PATH, columns=["code", "date", "close", "vol"]
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    k = k[(k["date"] >= pre_start) & (k["date"] <= post_end)]
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    print(
        f"[load] kline shape={k.shape}, "
        f"date range {k['date'].min().date()} ~ {k['date'].max().date()}, "
        f"codes={k['code'].nunique()}",
        flush=True,
    )
    return k


def compute_volume_factors(kline: pd.DataFrame) -> pd.DataFrame:
    """计算 5 个 volume 因子 (point-in-time, backward-only rolling).

    关键: rolling 窗口 .shift(1) → 不含当前日, T 日 factor 只用 T-1..T-N.
    """
    print("[factor] computing volume z-score + pct_change factors...", flush=True)
    k = kline.copy()
    k["vol"] = k["vol"].astype(float)
    grp = k.groupby("code", sort=False)

    # backward-only lag(1) of vol → rolling on lagged series gives T-N..T-1 stats
    k["_vol_lag1"] = grp["vol"].shift(1)
    grp2 = k.groupby("code", sort=False)

    for N in (5, 20, 60):
        rmean = grp2["_vol_lag1"].transform(
            lambda s: s.rolling(N, min_periods=N).mean()
        )
        rstd = grp2["_vol_lag1"].transform(
            lambda s: s.rolling(N, min_periods=N).std()
        )
        # z-score = (vol[T] - mean(vol, T-N..T-1)) / std(vol, T-N..T-1)
        k[f"vol_z_{N}d"] = (k["vol"] - rmean) / rstd.replace(0, np.nan)

    # pct_change: vol[T] / vol[T-N] - 1
    for N in (5, 20):
        vol_lagN = grp["vol"].shift(N)
        k[f"vol_chg_{N}d"] = k["vol"] / vol_lagN.replace(0, np.nan) - 1

    # clip extreme outliers (>= 99.5% / <= 0.5% pct) to limit noise
    for f in VOL_FACTORS:
        s = k[f]
        if s.notna().any():
            lo, hi = s.quantile(0.005), s.quantile(0.995)
            k[f] = s.clip(lo, hi)

    k = k.drop(columns=["_vol_lag1"])
    cov = {f: int(k[f].notna().sum()) for f in VOL_FACTORS}
    total = len(k)
    print(f"[factor] panel rows={total:,}; coverage:")
    for f, n in cov.items():
        print(f"  {f:14s} {n:8,d}  ({n/total*100:.1f}%)")
    return k[["code", "date", "close"] + VOL_FACTORS]


def build_amp_factor_for_orthogonality() -> pd.DataFrame:
    """计算 amp_imb_20d (v19.6 production sidecar 因子), 用于 Spearman 正交性.

    与 _factor_kline_panel.build_kline_factors 实现一致 — 重读 kline OHLC.
    """
    print("[orth] computing amp_imb_20d for Spearman orthogonality check...", flush=True)
    codes = set(load_csi300_codes())
    pre_start = IS_START - pd.Timedelta(days=60)
    k = pd.read_parquet(
        KLINE_PATH,
        columns=["code", "date", "open", "high", "low", "close"],
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    k = k[(k["date"] >= pre_start) & (k["date"] <= IS_END + pd.Timedelta(days=5))]
    k = k.sort_values(["code", "date"]).reset_index(drop=True)

    k["prev_close"] = k.groupby("code", sort=False)["close"].shift(1)
    valid = (
        k["prev_close"].notna() & (k["prev_close"] > 0) & (k["open"] > 0)
    )
    amp = (k["high"] - k["low"]) / k["prev_close"]
    delta = k["close"] - k["prev_close"]
    amp_up = np.where(delta > 0, delta, 0.0) / k["prev_close"] * amp
    amp_dn = np.where(delta < 0, -delta, 0.0) / k["prev_close"] * amp
    k["amp"] = amp
    k["amp_up"] = amp_up
    k["amp_dn"] = amp_dn
    k.loc[~valid, ["amp", "amp_up", "amp_dn"]] = np.nan

    grp = k.groupby("code", sort=False)
    k["amp_sum_20d"] = grp["amp"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["amp_up_sum_20d"] = grp["amp_up"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["amp_dn_sum_20d"] = grp["amp_dn"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["amp_imb_20d"] = np.where(
        k["amp_sum_20d"] > 0,
        (k["amp_up_sum_20d"] - k["amp_dn_sum_20d"]) / k["amp_sum_20d"],
        np.nan,
    )
    return k[["code", "date", "amp_imb_20d"]]


def build_monthly_panel(
    factor_kline: pd.DataFrame, amp_df: pd.DataFrame
) -> pd.DataFrame:
    """构造 (T, code, vol_z_5d..vol_chg_20d, amp_imb_20d, fwd_ret) panel.

    T = 每月第 1 个 IS 期交易日.
    fwd_ret = close(T + FORWARD_DAYS trading days) / close(T) - 1
    """
    print("[panel] building monthly panel...", flush=True)
    is_dates = pd.DatetimeIndex(sorted(factor_kline["date"].unique()))
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

    # wide close per (date, code) for fwd_ret
    wide = (
        factor_kline.pivot_table(
            index="date", columns="code", values="close", aggfunc="first"
        ).sort_index()
    )

    # filter factor & amp to anchor dates, group-by date for O(1) per-date lookup
    f_anchor = factor_kline[factor_kline["date"].isin(month_first)][
        ["code", "date"] + VOL_FACTORS
    ].copy()
    a_anchor = amp_df[amp_df["date"].isin(month_first)][
        ["code", "date", "amp_imb_20d"]
    ].copy()
    f_groups = dict(list(f_anchor.groupby("date", sort=False)))
    a_groups = dict(list(a_anchor.groupby("date", sort=False)))

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
        df_T = df_T.merge(
            f_T.drop(columns=["date"]), on="code", how="left"
        )

        a_T = a_groups.get(T)
        if a_T is not None and not a_T.empty:
            df_T = df_T.merge(
                a_T.drop(columns=["date"]), on="code", how="left"
            )
        else:
            df_T["amp_imb_20d"] = np.nan

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
    print(
        f"[panel] rows={len(panel):,}, codes={panel['code'].nunique()}, "
        f"months={panel['month_start'].nunique()}",
        flush=True,
    )
    return panel


def monthly_ic(
    panel: pd.DataFrame, factor_col: str, sign: int = 1
) -> tuple[pd.Series, dict]:
    """每月横截面 Spearman corr(factor*sign, fwd_ret)."""
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    if df.empty:
        return pd.Series(dtype=float), {
            "factor": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
            "ic_pos_pct": 0.0, "ic_neg_pct": 0.0,
            "avg_obs_per_month": 0.0,
        }
    df["signed_factor"] = sign * df[factor_col]

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < MIN_MONTHLY_OBS:
            return np.nan
        if g["signed_factor"].std() <= 0:
            return np.nan
        return g["signed_factor"].corr(g["fwd_ret"], method="spearman")

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


def spearman_vs_amp(panel: pd.DataFrame) -> pd.DataFrame:
    """月度横截面 Spearman corr(z_vol_factor, amp_imb_20d) → mean |rho|."""
    rows = []
    for f in VOL_FACTORS:
        zcol = f"z_{f}"
        sub = panel.dropna(subset=[zcol, "amp_imb_20d"])
        if sub.empty:
            rows.append({
                "factor": f, "n_months": 0, "mean_rho": np.nan,
                "mean_abs_rho": np.nan, "max_abs_rho": np.nan,
            })
            continue
        per_month = sub.groupby("month_start").apply(
            lambda g: g[zcol].corr(g["amp_imb_20d"], method="spearman")
            if len(g) >= MIN_MONTHLY_OBS
            and g[zcol].std() > 0
            and g["amp_imb_20d"].std() > 0
            else np.nan
        ).dropna()
        rows.append({
            "factor": f,
            "n_months": int(len(per_month)),
            "mean_rho": float(per_month.mean()) if len(per_month) else np.nan,
            "mean_abs_rho": float(per_month.abs().mean()) if len(per_month) else np.nan,
            "max_abs_rho": float(per_month.abs().max()) if len(per_month) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    if not KLINE_PATH.exists():
        print(f"FATAL: 缺 {KLINE_PATH}", file=sys.stderr)
        return 1
    if not CSI300_PATH.exists():
        print(f"FATAL: 缺 {CSI300_PATH}", file=sys.stderr)
        return 1

    kline = load_kline_is_with_buffer()
    factor_df = compute_volume_factors(kline)
    amp_df = build_amp_factor_for_orthogonality()
    panel = build_monthly_panel(factor_df, amp_df)
    if panel.empty:
        print("FATAL: panel 为空", file=sys.stderr)
        return 1

    print("\n[step 2] monthly cross-section IC sweep (5 horizons × 2 signs)...", flush=True)
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
        [
            "factor", "zfactor", "sign", "n_months",
            "ic_mean", "ic_std", "icir",
            "ic_pos_pct", "ic_neg_pct", "avg_obs_per_month",
        ]
    ].round(4)
    summary_df.to_csv(OUT_GRID, index=False)
    print(f"\n[output] IC grid -> {OUT_GRID}")
    print(summary_df.to_string(index=False))

    monthly_rows = []
    for key, ser in monthly_series.items():
        for m, v in ser.items():
            monthly_rows.append({"variant": key, "month_start": m, "ic": v})
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly IC -> {OUT_MONTHLY}")

    print("\n[step 3] orthogonality vs amp_imb_20d (v19.6 production sidecar)...", flush=True)
    orth = spearman_vs_amp(panel)
    orth.to_csv(OUT_SPEARMAN, index=False)
    print(orth.to_string(index=False))
    print(f"[output] Spearman -> {OUT_SPEARMAN}")

    print("\n[step 4] verdict ...", flush=True)
    top = summary_df.iloc[0]
    icir_abs = abs(float(top["icir"]))
    print(f"  top variant: factor={top['factor']} sign={int(top['sign'])} "
          f"ICIR={float(top['icir']):.3f} pos%={float(top['ic_pos_pct']):.1f}")
    n_recommended = (summary_df["icir"].abs() >= 0.40).sum()
    if icir_abs >= 0.40:
        sig = int(top["sign"])
        same_sign_pct = (
            float(top["ic_pos_pct"]) if sig == +1
            else float(top["ic_neg_pct"])
        )
        stable = same_sign_pct >= 55.0
        print(f"  same-sign month pct: {same_sign_pct:.1f}%  (stable >=55%: {stable})")
        if stable:
            print("\n  VERDICT: 推荐 Phase B sidecar")
            print(f"    factor={top['factor']}, sign={sig}, "
                  f"λ candidates ∈ [0.10, 0.20, 0.30]")
        else:
            print(f"\n  VERDICT: |ICIR|={icir_abs:.3f} 达标但 sign 不稳定 (<55% same-sign month) → 不推荐")
    elif icir_abs >= 0.20:
        print(f"\n  VERDICT: |ICIR|={icir_abs:.3f} 边际 (0.20-0.40), 不推荐 Phase B")
    else:
        print(f"\n  VERDICT: |ICIR|={icir_abs:.3f} < 0.20, 因子无 alpha, abort")

    print(f"\n[summary] panel rows={len(panel):,}, months={panel['month_start'].nunique()}")
    print(f"[summary] {n_recommended} variants with |ICIR|>=0.40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
