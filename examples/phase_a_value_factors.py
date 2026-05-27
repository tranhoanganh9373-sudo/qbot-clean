"""Phase A IS IC analysis for VALUE factors (PE_static / PE_TTM / PB).

Strict OOS protocol (memory: feedback_strict_oos_backtest):
- IS = 2014-01-01 ~ 2020-12-31 (84 月) — factor selection only
- OOS = 2021-05-01 ~ 2026-04-30 — NEVER touched in Phase A.

Value factors per (month_start T, code):
  PE_static = close(T) / annual_eps_basic  (most-recent annual EPS, PIT lag=60d)
  PE_TTM    = close(T) / TTM_eps           (TTM = sum of trailing 4 single-quarter EPS)
  PB        = close(T) / bps               (PIT bps)

  Single-quarter EPS derived by differencing cumulative YTD eps_basic within the
  same fiscal year (Q4-Q3, Q3-Q2, Q2-Q1; Q1=Q1).

Dividend yield: NOT in fundamentals cache schema → SKIP per task spec.

For VALUE factors, lower = cheaper = bullish, so expected sign = -1.

Forward return:  ~1-month = 21 trading days, fwd_ret = close(T+21)/close(T) - 1.

Orthogonality:
  Monthly cross-sectional Spearman corr(value_factor, X) for X in
  {amp_imb_20d (v19.6 production), JZF = (open - prev_close) / prev_close * 100}.

Phase A pass criteria:
  |ICIR| >= 0.4 AND n_months >= 60 AND mean |rho| < 0.30 vs ALL existing factors.

Outputs:
  examples/v21_value_phase_a_ic.csv         per-factor IC summary
  examples/v21_value_phase_a_spearman.csv   orthogonality matrix
  examples/v21_value_phase_a_monthly_ic.csv long-format monthly IC

Run:
  .venv/bin/python examples/phase_a_value_factors.py
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
FUND_DIR = ROOT / "data_cache" / "fundamentals"

OUT_IC = ROOT / "examples" / "v21_value_phase_a_ic.csv"
OUT_SPEARMAN = ROOT / "examples" / "v21_value_phase_a_spearman.csv"
OUT_MONTHLY = ROOT / "examples" / "v21_value_phase_a_monthly_ic.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 21          # ~1-month forward return (trading days)
ANNOUNCE_LAG_DAYS = 60
MIN_MONTHLY_OBS = 30       # minimum cross-section size

VALUE_FACTORS = ["PE_static", "PE_TTM", "PB"]
# For value factors: low value = cheap = bullish → sign = -1
VALUE_SIGNS = {"PE_static": -1, "PE_TTM": -1, "PB": -1}


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def load_csi300_codes() -> list[str]:
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    return sorted(set(csi["code"].tolist()))


def load_kline_is(codes: set[str]) -> pd.DataFrame:
    """Read baidu_kline (OHLC), filter to IS + small buffer for fwd / JZF."""
    pre = IS_START - pd.Timedelta(days=60)
    post = IS_END + pd.Timedelta(days=45)   # buffer for FORWARD_DAYS
    k = pd.read_parquet(
        KLINE_PATH,
        columns=["code", "date", "open", "high", "low", "close"],
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    k = k[(k["date"] >= pre) & (k["date"] <= post)]
    return k.sort_values(["code", "date"]).reset_index(drop=True)


def load_fundamentals(codes: list[str]) -> dict[str, pd.DataFrame]:
    """Load per-stock fundamentals parquet → {code: df}."""
    out: dict[str, pd.DataFrame] = {}
    missing = []
    for code in codes:
        p = FUND_DIR / f"{code}.parquet"
        if not p.exists():
            missing.append(code)
            continue
        d = pd.read_parquet(p)
        if d.empty:
            missing.append(code)
            continue
        d["code"] = d["code"].astype(str).str.zfill(6)
        d["report_date"] = pd.to_datetime(d["report_date"])
        d = d.sort_values("report_date").reset_index(drop=True)
        out[code] = d
    if missing:
        print(f"[warn] {len(missing)} codes missing fundamentals cache "
              f"(first 5: {missing[:5]})", flush=True)
    return out


# ----------------------------------------------------------------------
# Single-quarter EPS + TTM EPS + annual EPS (PIT)
# ----------------------------------------------------------------------

def build_single_quarter_eps(df: pd.DataFrame) -> pd.DataFrame:
    """Add `eps_sq` column: single-quarter EPS by differencing cumulative YTD.

    eps_basic is cumulative YTD (resets Q1). Single-quarter:
       Q1 = Q1                 (report_date month=3)
       Q2 = Q2_cum - Q1        (month=6)
       Q3 = Q3_cum - Q2_cum    (month=9)
       Q4 = Q4_cum - Q3_cum    (month=12)
    """
    out = df.copy()
    out["fiscal_year"] = out["report_date"].dt.year
    out["fiscal_q"] = out["report_date"].dt.month  # 3 6 9 12
    out = out.sort_values(["fiscal_year", "fiscal_q"]).reset_index(drop=True)
    out["eps_sq"] = out.groupby("fiscal_year")["eps_basic"].diff()
    out.loc[out["fiscal_q"] == 3, "eps_sq"] = out.loc[
        out["fiscal_q"] == 3, "eps_basic"
    ]
    return out


def get_value_inputs_at_date(
    df: pd.DataFrame,
    query_date: pd.Timestamp,
    announce_lag_days: int = ANNOUNCE_LAG_DAYS,
) -> dict | None:
    """PIT value inputs visible at T: (annual_eps, ttm_eps, bps).

    - annual_eps: most-recent Q4 (December) report visible at T.
        If no Q4 visible yet, NaN (PE_static = NaN).
    - ttm_eps: sum of last 4 single-quarter EPS reports visible at T.
    - bps: PIT bps (latest visible report).
    """
    cutoff = query_date - pd.Timedelta(days=announce_lag_days)
    visible = df[df["report_date"] <= cutoff]
    if visible.empty:
        return None

    latest = visible.iloc[-1]
    bps = latest.get("bps")
    bps = float(bps) if bps is not None and pd.notna(bps) else np.nan

    q4s = visible[visible["fiscal_q"] == 12]
    if q4s.empty:
        annual_eps = np.nan
    else:
        annual_eps = q4s.iloc[-1].get("eps_basic")
        annual_eps = (
            float(annual_eps)
            if annual_eps is not None and pd.notna(annual_eps)
            else np.nan
        )

    sq = visible.dropna(subset=["eps_sq"])
    if len(sq) < 4:
        ttm_eps = np.nan
    else:
        ttm_eps = float(sq["eps_sq"].iloc[-4:].sum())

    return {"annual_eps": annual_eps, "ttm_eps": ttm_eps, "bps": bps}


# ----------------------------------------------------------------------
# amp_imb_20d (v19.6 production sidecar factor) + JZF (gap %) builders
# ----------------------------------------------------------------------

def add_amp_imb_20d(k: pd.DataFrame) -> pd.DataFrame:
    """v19.6 production formula — matches examples/_factor_kline_panel.py."""
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    k["prev_close"] = k.groupby("code", sort=False)["close"].shift(1)
    valid = (k["prev_close"].notna()) & (k["prev_close"] > 0) & (k["open"] > 0)
    amp = (k["high"] - k["low"]) / k["prev_close"]
    delta = k["close"] - k["prev_close"]
    amp_up = np.where(delta > 0, delta, 0.0) / k["prev_close"] * amp
    amp_dn = np.where(delta < 0, -delta, 0.0) / k["prev_close"] * amp
    k["_amp"] = amp
    k["_amp_up"] = amp_up
    k["_amp_dn"] = amp_dn
    k.loc[~valid, ["_amp", "_amp_up", "_amp_dn"]] = np.nan
    grp = k.groupby("code", sort=False)
    k["_amp_sum_20d"] = grp["_amp"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["_amp_up_sum_20d"] = grp["_amp_up"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["_amp_dn_sum_20d"] = grp["_amp_dn"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["amp_imb_20d"] = np.where(
        k["_amp_sum_20d"] > 0,
        (k["_amp_up_sum_20d"] - k["_amp_dn_sum_20d"]) / k["_amp_sum_20d"],
        np.nan,
    )
    k["JZF"] = np.where(
        valid, (k["open"] - k["prev_close"]) / k["prev_close"] * 100.0, np.nan
    )
    return k.drop(columns=[c for c in k.columns if c.startswith("_")])


# ----------------------------------------------------------------------
# Build monthly panel
# ----------------------------------------------------------------------

def build_panel(
    kline: pd.DataFrame, fund_map: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Return long panel: month_start, code, PE_static, PE_TTM, PB,
    amp_imb_20d, JZF, fwd_ret.

    T = first IS-period trading day of each month.
    """
    print("[panel] building single-quarter EPS for each code...", flush=True)
    fund_sq = {
        code: build_single_quarter_eps(df) for code, df in fund_map.items()
    }

    is_dates = pd.DatetimeIndex(sorted(kline["date"].unique()))
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

    wide_close = kline.pivot_table(
        index="date", columns="code", values="close", aggfunc="first"
    ).sort_index()

    k_anchor = kline[kline["date"].isin(month_first)][
        ["code", "date", "amp_imb_20d", "JZF"]
    ].copy()
    anchor_groups = dict(list(k_anchor.groupby("date", sort=False)))

    rows = []
    for T in month_first:
        idx = wide_close.index.get_indexer([T])[0]
        if idx < 0 or idx + FORWARD_DAYS >= len(wide_close.index):
            continue
        T_close = wide_close.iloc[idx]
        T_plus = wide_close.iloc[idx + FORWARD_DAYS]
        fwd = T_plus / T_close - 1

        df_T = pd.DataFrame({"fwd_ret": fwd})
        df_T["code"] = df_T.index.astype(str)
        df_T["close_T"] = T_close.values
        df_T = df_T.dropna(subset=["fwd_ret", "close_T"]).reset_index(drop=True)

        val_rows = []
        for code in df_T["code"]:
            fdf = fund_sq.get(code)
            if fdf is None:
                val_rows.append(
                    {"annual_eps": np.nan, "ttm_eps": np.nan, "bps": np.nan}
                )
                continue
            v = get_value_inputs_at_date(fdf, T)
            if v is None:
                val_rows.append(
                    {"annual_eps": np.nan, "ttm_eps": np.nan, "bps": np.nan}
                )
            else:
                val_rows.append(v)
        val_df = pd.DataFrame(val_rows)
        df_T = pd.concat([df_T, val_df], axis=1)

        def _safe_pe(close: float, eps: float) -> float:
            if pd.isna(close) or pd.isna(eps) or eps <= 0:
                return np.nan
            return close / eps

        def _safe_pb(close: float, bps: float) -> float:
            if pd.isna(close) or pd.isna(bps) or bps <= 0:
                return np.nan
            return close / bps

        df_T["PE_static"] = [
            _safe_pe(c, e) for c, e in zip(
                df_T["close_T"], df_T["annual_eps"], strict=True
            )
        ]
        df_T["PE_TTM"] = [
            _safe_pe(c, e) for c, e in zip(
                df_T["close_T"], df_T["ttm_eps"], strict=True
            )
        ]
        df_T["PB"] = [
            _safe_pb(c, b) for c, b in zip(
                df_T["close_T"], df_T["bps"], strict=True
            )
        ]

        a_T = anchor_groups.get(T)
        if a_T is not None and not a_T.empty:
            df_T = df_T.merge(
                a_T.drop(columns=["date"]), on="code", how="left"
            )
        else:
            df_T["amp_imb_20d"] = np.nan
            df_T["JZF"] = np.nan

        df_T["month_start"] = T
        rows.append(df_T)

    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if panel.empty:
        return panel
    print(
        f"[panel] rows={len(panel):,}, codes={panel['code'].nunique()}, "
        f"months={panel['month_start'].nunique()}",
        flush=True,
    )
    return panel[
        ["month_start", "code", "fwd_ret",
         "PE_static", "PE_TTM", "PB",
         "amp_imb_20d", "JZF"]
    ]


# ----------------------------------------------------------------------
# IC & orthogonality
# ----------------------------------------------------------------------

def monthly_ic(
    panel: pd.DataFrame, factor_col: str, sign: int
) -> tuple[pd.Series, dict]:
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    if df.empty:
        return pd.Series(dtype=float), {
            "factor": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
            "ic_pos_pct": 0.0, "avg_obs_per_month": 0.0,
        }
    df["signed_factor"] = sign * df[factor_col]
    obs = df.groupby("month_start").size()

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < MIN_MONTHLY_OBS:
            return np.nan
        if g["signed_factor"].std() <= 0:
            return np.nan
        return g["signed_factor"].corr(g["fwd_ret"], method="spearman")

    monthly = df.groupby("month_start").apply(_corr).dropna()
    if len(monthly) == 0:
        return monthly, {
            "factor": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
            "ic_pos_pct": 0.0, "avg_obs_per_month": float(obs.mean()),
        }
    mean = float(monthly.mean())
    std = float(monthly.std())
    icir = mean / std * np.sqrt(12) if std > 0 else 0.0
    return monthly, {
        "factor": factor_col, "sign": sign, "n_months": int(len(monthly)),
        "ic_mean": mean, "ic_std": std, "icir": icir,
        "ic_pos_pct": float((monthly > 0).mean() * 100),
        "avg_obs_per_month": float(obs.mean()),
    }


def spearman_orthogonality(
    panel: pd.DataFrame, factors: list[str], reference_factors: list[str]
) -> pd.DataFrame:
    """Monthly cross-section Spearman corr(factor, ref) → mean |rho|."""
    rows = []
    for f in factors:
        for ref in reference_factors:
            sub = panel.dropna(subset=[f, ref])
            if sub.empty:
                rows.append({
                    "factor": f, "vs": ref, "n_months": 0,
                    "mean_rho": np.nan, "mean_abs_rho": np.nan,
                    "max_abs_rho": np.nan,
                })
                continue
            per_month = sub.groupby("month_start").apply(
                lambda g: g[f].corr(g[ref], method="spearman")
                if len(g) >= MIN_MONTHLY_OBS
                and g[f].std() > 0 and g[ref].std() > 0
                else np.nan
            ).dropna()
            rows.append({
                "factor": f, "vs": ref,
                "n_months": int(len(per_month)),
                "mean_rho": float(per_month.mean()) if len(per_month) else np.nan,
                "mean_abs_rho": float(per_month.abs().mean()) if len(per_month) else np.nan,
                "max_abs_rho": float(per_month.abs().max()) if len(per_month) else np.nan,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    if not KLINE_PATH.exists():
        print(f"FATAL: kline cache missing: {KLINE_PATH}", file=sys.stderr)
        return 1
    if not CSI300_PATH.exists():
        print(f"FATAL: csi300 list missing: {CSI300_PATH}", file=sys.stderr)
        return 1
    if not FUND_DIR.exists():
        print(f"FATAL: fundamentals dir missing: {FUND_DIR}", file=sys.stderr)
        return 1

    codes = load_csi300_codes()
    print(f"[load] CSI300 universe: {len(codes)} codes", flush=True)

    print("[load] kline (OHLC, IS + buffer)...", flush=True)
    kline = load_kline_is(set(codes))
    print(f"[load] kline shape={kline.shape}", flush=True)

    print("[compute] amp_imb_20d + JZF on kline...", flush=True)
    kline = add_amp_imb_20d(kline)

    print("[load] fundamentals for CSI300...", flush=True)
    fund_map = load_fundamentals(codes)
    print(f"[load] {len(fund_map)} stocks with fundamentals cache", flush=True)
    if len(fund_map) < 50:
        print(f"FATAL: only {len(fund_map)} stocks have cache", file=sys.stderr)
        return 1

    panel = build_panel(kline, fund_map)
    if panel.empty:
        print("FATAL: panel empty", file=sys.stderr)
        return 1

    summaries = []
    monthly_series: dict[str, pd.Series] = {}
    for f in VALUE_FACTORS:
        s_pos = VALUE_SIGNS[f]
        m, summary = monthly_ic(panel, f, sign=s_pos)
        summaries.append(summary)
        monthly_series[f] = m

    summary_df = pd.DataFrame(summaries).round(4)
    summary_df = summary_df.reindex(
        summary_df["icir"].abs().sort_values(ascending=False).index
    )
    summary_df.to_csv(OUT_IC, index=False)
    print(f"\n[output] IC summary -> {OUT_IC}")
    print(summary_df.to_string(index=False))

    monthly_rows = []
    for f, ser in monthly_series.items():
        for m, v in ser.items():
            monthly_rows.append({"factor": f, "month_start": m, "ic": v})
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly IC -> {OUT_MONTHLY}")

    print("\n[orth] Spearman value-factor vs (amp_imb_20d, JZF)...", flush=True)
    ortho = spearman_orthogonality(
        panel, VALUE_FACTORS, ["amp_imb_20d", "JZF"]
    ).round(4)
    ortho.to_csv(OUT_SPEARMAN, index=False)
    print(f"[output] orthogonality -> {OUT_SPEARMAN}")
    print(ortho.to_string(index=False))

    print("\n[verdict] Phase A pass criteria:")
    print("  |ICIR| >= 0.4, n_months >= 60, mean |rho| < 0.30 vs all refs")
    for _, row in summary_df.iterrows():
        f = row["factor"]
        passes_icir = abs(row["icir"]) >= 0.4
        passes_n = row["n_months"] >= 60
        rhos = ortho[ortho["factor"] == f]
        max_abs_rho = (
            float(rhos["mean_abs_rho"].max()) if not rhos.empty else np.nan
        )
        passes_orth = (
            pd.notna(max_abs_rho) and max_abs_rho < 0.30
        )
        verdict = (
            "PASS → Phase B" if (passes_icir and passes_n and passes_orth)
            else "ABORT"
        )
        print(
            f"  {f}: ICIR={row['icir']:+.3f} "
            f"(|.|>=0.4: {passes_icir}) "
            f"n_months={int(row['n_months'])} (>=60: {passes_n}) "
            f"max_mean_|rho|={max_abs_rho:.3f} (<0.30: {passes_orth}) "
            f"→ {verdict}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
