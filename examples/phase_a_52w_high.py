"""CSI300 52-week high distance 因子 IS IC 分析 (Phase A) — 严格 OOS 协议.

George & Hwang (2004) 经典: 股价距离 52 周高点的距离 (negative pct) 是 momentum
proxy. 文献结论混合 — A 股可能反转 (放量回升) 或延续 (突破)，需双向 sign 测试.

严格约束:
- IS 期 (因子选择): 2014-01-01 ~ 2020-12-31 (84 月)
- OOS (留作 backtest): 2021-05-01 ~ 2026-04-30 → **本脚本绝不触碰**
- 不修改任何 production 文件 / sidecar 脚本 / 数据 cache
- 只读 baidu_kline.parquet

候选因子 (3 horizon):
    dist_252d = (close - max(close, T-252..T-1)) / max(close, T-252..T-1)
    dist_60d  = same with 60d window
    dist_20d  = same with 20d window
取值范围 [-1, 0]; 0 = 当下是窗口高点, 越负 = 距高点越远.

机制可能性 (sign 双向都试):
    +1: 距高点越近 (越接近 0) → fwd_ret 越高 → 延续/突破
    -1: 距高点越近 (越接近 0) → fwd_ret 越低 → 反转/回归
两种符号都跑, 看月度 IC 数据说话.

PIT (point-in-time) 严格性:
- rolling window 严格 backward-only (shift(1) + rolling(N)) → 不含 T 日自身
- factor 在月度采样日 T 上算, 用 T-1..T-N 窗口高点
- forward return = close(T + 22 trading days) / close(T) - 1
- cross-section z-score 每月再标准化一次

正交性 check:
- vs amp_imb_20d (v19.10 主 sidecar 因子, λ=-0.30)
- vs JZF = (open - prev_close) / prev_close × 100 (v19.10 stacked 因子, λ=+0.10)
- 月度横截面 Spearman corr, 报 mean / mean_abs / max_abs

Phase A 通过门槛 (4 项全过):
    |ICIR| >= 0.40
    n_months >= 60
    mean_abs_rho vs amp_imb_20d < 0.30
    mean_abs_rho vs JZF          < 0.30

输出:
    examples/v21_52w_high_phase_a_ic.csv           summary (3 horizon × 2 sign)
    examples/v21_52w_high_phase_a_spearman.csv     vs amp_imb_20d + JZF
    examples/v21_52w_high_phase_a_monthly.csv      long-format monthly IC (辅助)

运行 (~10-15 min on CSI300 IS panel):
    .venv/bin/python examples/phase_a_52w_high.py
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
OUT_IC = ROOT / "examples" / "v21_52w_high_phase_a_ic.csv"
OUT_SPEARMAN = ROOT / "examples" / "v21_52w_high_phase_a_spearman.csv"
OUT_MONTHLY = ROOT / "examples" / "v21_52w_high_phase_a_monthly.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 22  # 月度 forward return (22 trading days ~ 1 个月)
MIN_MONTHLY_OBS = 30  # 每月最少 30 股横截面才算 IC
# pre-buffer: 252 trading days ≈ 365 calendar days, 加 buffer 留 30 天非交易
PRE_BUFFER_DAYS = 400
POST_BUFFER_DAYS = 35  # 22 trading days fwd

DIST_FACTORS = ["dist_252d", "dist_60d", "dist_20d"]


def load_csi300_codes() -> list[str]:
    """Load CSI300 6-digit codes."""
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    return csi["code"].tolist()


def load_kline_is_with_buffer() -> pd.DataFrame:
    """读 baidu_kline.parquet IS 期 CSI300 + pre/post buffer.

    Pre-buffer: 400 calendar days for dist_252d (need 252 trading days history).
    Post-buffer: 35 calendar days for forward 22d return.
    Cols: code, date, open, close (close 用于 high rolling + fwd_ret; open 用于 JZF).
    """
    codes = set(load_csi300_codes())
    pre_start = IS_START - pd.Timedelta(days=PRE_BUFFER_DAYS)
    post_end = IS_END + pd.Timedelta(days=POST_BUFFER_DAYS)
    k = pd.read_parquet(
        KLINE_PATH, columns=["code", "date", "open", "close"]
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


def compute_distance_factors(kline: pd.DataFrame) -> pd.DataFrame:
    """计算 3 个 52-week high distance 因子 (point-in-time, backward-only rolling).

    dist_N = (close[T] - max(close, T-N..T-1)) / max(close, T-N..T-1)
    范围 [-1, 0], 0 = 突破窗口高, -0.X = 距高点 X% 远.

    关键: rolling .shift(1) → 窗口为 T-1..T-N, 不含当前日 (PIT 严格).
    """
    print("[factor] computing 52-week high distance factors (252d/60d/20d)...", flush=True)
    k = kline.copy()
    k["close"] = k["close"].astype(float)
    grp = k.groupby("code", sort=False)

    # backward-only lag(1) of close
    k["_close_lag1"] = grp["close"].shift(1)
    grp2 = k.groupby("code", sort=False)

    for N in (252, 60, 20):
        rmax = grp2["_close_lag1"].transform(
            lambda s: s.rolling(N, min_periods=N).max()
        )
        # dist 应 ∈ [-1, 0]; close > rmax 时为正 (突破新高), 实务正常少见但允许
        k[f"dist_{N}d"] = (k["close"] - rmax) / rmax.replace(0, np.nan)

    # clip extreme outliers (>= 99.5% / <= 0.5%) — 主要为 IPO 早期或 corrupt 残留
    for f in DIST_FACTORS:
        s = k[f]
        if s.notna().any():
            lo, hi = s.quantile(0.005), s.quantile(0.995)
            k[f] = s.clip(lo, hi)

    k = k.drop(columns=["_close_lag1"])
    cov = {f: int(k[f].notna().sum()) for f in DIST_FACTORS}
    total = len(k)
    print(f"[factor] panel rows={total:,}; coverage:")
    for f, n in cov.items():
        print(f"  {f:12s} {n:8,d}  ({n/total*100:.1f}%)")
    return k[["code", "date", "open", "close"] + DIST_FACTORS]


def build_orthogonality_factors() -> pd.DataFrame:
    """计算 amp_imb_20d (v19.10 主 sidecar) + JZF (v19.10 stacked) 用于 Spearman 正交性.

    amp_imb_20d 公式与 production sidecar 一致 (20 日累计上涨/下跌振幅净失衡比).
    JZF = (open - prev_close) / prev_close × 100 (集合竞价跳空 %).
    """
    print("[orth] computing amp_imb_20d + JZF for Spearman orthogonality...", flush=True)
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
    # JZF = (open - prev_close) / prev_close * 100  (% 单位, 与 production 一致)
    k["jzf"] = np.where(
        valid,
        (k["open"] - k["prev_close"]) / k["prev_close"] * 100.0,
        np.nan,
    )
    return k[["code", "date", "amp_imb_20d", "jzf"]]


def build_monthly_panel(
    factor_kline: pd.DataFrame, orth_df: pd.DataFrame
) -> pd.DataFrame:
    """构造 (T, code, dist_*, amp_imb_20d, jzf, fwd_ret) panel.

    T = 每月第 1 个 IS 期交易日.
    fwd_ret = close(T + FORWARD_DAYS trading days) / close(T) - 1.
    每月横截面再 z-score 一次 (cross-section 相对强度).
    """
    print("[panel] building monthly panel (T = 月首交易日)...", flush=True)
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

    # filter factor & orth to anchor dates
    f_anchor = factor_kline[factor_kline["date"].isin(month_first)][
        ["code", "date"] + DIST_FACTORS
    ].copy()
    o_anchor = orth_df[orth_df["date"].isin(month_first)][
        ["code", "date", "amp_imb_20d", "jzf"]
    ].copy()
    f_groups = dict(list(f_anchor.groupby("date", sort=False)))
    o_groups = dict(list(o_anchor.groupby("date", sort=False)))

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

        o_T = o_groups.get(T)
        if o_T is not None and not o_T.empty:
            df_T = df_T.merge(o_T.drop(columns=["date"]), on="code", how="left")
        else:
            df_T["amp_imb_20d"] = np.nan
            df_T["jzf"] = np.nan

        df_T["month_start"] = T
        rows.append(df_T)

    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if panel.empty:
        return panel

    print("[panel] cross-section z-score per month for each factor...", flush=True)
    for f in DIST_FACTORS:
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


def spearman_vs(
    panel: pd.DataFrame, other_col: str
) -> pd.DataFrame:
    """月度横截面 Spearman corr(z_dist_*, other_col) → mean / mean_abs / max_abs."""
    rows = []
    for f in DIST_FACTORS:
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
    factor_df = compute_distance_factors(kline)
    orth_df = build_orthogonality_factors()
    panel = build_monthly_panel(factor_df, orth_df)
    if panel.empty:
        print("FATAL: panel 为空", file=sys.stderr)
        return 1

    print("\n[step 2] monthly IC sweep (3 horizons × 2 signs = 6 combos)...", flush=True)
    summaries: list[dict] = []
    monthly_series: dict[str, pd.Series] = {}
    for f in DIST_FACTORS:
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

    print("\n[step 3] orthogonality vs amp_imb_20d + JZF (v19.10 production)...", flush=True)
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
    print(f"  top variant: factor={top_factor} sign={top_sign:+d} "
          f"ICIR={float(top['icir']):.3f} n_months={top_n_months}")

    # 4 项门槛全过才 RECOMMEND
    amp_row = orth_amp[orth_amp["factor"] == top_factor].iloc[0]
    jzf_row = orth_jzf[orth_jzf["factor"] == top_factor].iloc[0]
    amp_rho = float(amp_row["mean_abs_rho"]) if pd.notna(amp_row["mean_abs_rho"]) else 1.0
    jzf_rho = float(jzf_row["mean_abs_rho"]) if pd.notna(jzf_row["mean_abs_rho"]) else 1.0

    gate1 = icir_abs >= 0.40
    gate2 = top_n_months >= 60
    gate3 = amp_rho < 0.30
    gate4 = jzf_rho < 0.30
    print(f"  gate1 |ICIR|>=0.40: {icir_abs:.3f} → {'PASS' if gate1 else 'FAIL'}")
    print(f"  gate2 n_months>=60: {top_n_months} → {'PASS' if gate2 else 'FAIL'}")
    print(f"  gate3 |rho| vs amp_imb_20d < 0.30: {amp_rho:.3f} → {'PASS' if gate3 else 'FAIL'}")
    print(f"  gate4 |rho| vs JZF < 0.30: {jzf_rho:.3f} → {'PASS' if gate4 else 'FAIL'}")
    all_pass = gate1 and gate2 and gate3 and gate4

    if all_pass:
        print("\n  VERDICT: RECOMMEND Phase B sidecar OOS test")
        print(f"    factor = {top_factor}")
        print(f"    sign   = {top_sign:+d}")
        print(f"    λ candidates ∈ [0.05, 0.10, 0.20, 0.30]")
    else:
        fails = []
        if not gate1: fails.append(f"|ICIR|={icir_abs:.3f}<0.40")
        if not gate2: fails.append(f"n_months={top_n_months}<60")
        if not gate3: fails.append(f"|rho|_amp={amp_rho:.3f}>=0.30")
        if not gate4: fails.append(f"|rho|_jzf={jzf_rho:.3f}>=0.30")
        print(f"\n  VERDICT: ABORT — {', '.join(fails)}")

    print(f"\n[summary] panel rows={len(panel):,}, months={panel['month_start'].nunique()}")
    n_recommended = (summary_df["icir"].abs() >= 0.40).sum()
    print(f"[summary] {n_recommended}/6 variants with |ICIR|>=0.40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
