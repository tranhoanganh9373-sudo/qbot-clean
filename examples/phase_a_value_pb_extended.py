"""Phase A IS IC for VALUE PB factor — EXTENDED kline (2014-2017 backfill + 2018+).

修复 phase_a_value_factors.py 的 n_months=35 痛点:
  baidu_kline.parquet CSI300 2014-2017 仅 10-22 codes/月 → 实际 IC 月数 35.
  本脚本合并 kline_2014_2017_csi300_backfill.parquet (2014-2017 235 stocks/月)
  + baidu_kline.parquet (2018-2020+) → 期望 n_months 接近 84.

Strict OOS protocol:
  IS = 2014-01-01 ~ 2020-12-31 (84 月)
  OOS = 2021-05-01 ~ 2026-04-30 — 严格不动 (本脚本只读 IS).

PB = close(T) / bps,  bps PIT lag = 60 天 (使用上一季度财报).
PB 因子 sign = -1 (低 PB = 便宜 = 看多).

Gate (per task spec):
  |ICIR| >= 0.4 AND n_months >= 60 AND mean |rho| < 0.30 vs (amp_imb_20d, JZF).

Outputs:
  examples/v21_value_pb_extended_ic.csv
  examples/v21_value_pb_extended_monthly.csv
  examples/v21_value_pb_extended_spearman.csv

Run: .venv/bin/python examples/phase_a_value_pb_extended.py
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
KLINE_MAIN = ROOT / "data_cache" / "baidu_kline.parquet"
KLINE_BACKFILL = ROOT / "data_cache" / "kline_2014_2017_csi300_backfill.parquet"
FUND_DIR = ROOT / "data_cache" / "fundamentals"

OUT_IC = ROOT / "examples" / "v21_value_pb_extended_ic.csv"
OUT_SPEARMAN = ROOT / "examples" / "v21_value_pb_extended_spearman.csv"
OUT_MONTHLY = ROOT / "examples" / "v21_value_pb_extended_monthly.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
BACKFILL_CUTOFF = pd.Timestamp("2018-01-01")  # backfill covers <, main >=
FORWARD_DAYS = 21
ANNOUNCE_LAG_DAYS = 60
MIN_MONTHLY_OBS = 30

FACTOR_NAME = "PB"
FACTOR_SIGN = -1  # low PB = cheap = bullish


# ──────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────

def load_csi300_codes() -> list[str]:
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    return sorted(set(csi["code"].tolist()))


def load_kline_extended(codes: set[str]) -> pd.DataFrame:
    """Read baidu_kline (>=2018) + backfill (<2018), combine."""
    pre = IS_START - pd.Timedelta(days=60)
    post = IS_END + pd.Timedelta(days=45)
    cols = ["code", "date", "open", "high", "low", "close"]

    k_main = pd.read_parquet(KLINE_MAIN, columns=cols)
    k_main["code"] = k_main["code"].astype(str).str.zfill(6)
    k_main = k_main[k_main["code"].isin(codes)]
    k_main = k_main[(k_main["date"] >= BACKFILL_CUTOFF) & (k_main["date"] <= post)]

    k_bf = pd.read_parquet(KLINE_BACKFILL, columns=cols)
    k_bf["code"] = k_bf["code"].astype(str).str.zfill(6)
    k_bf = k_bf[k_bf["code"].isin(codes)]
    k_bf = k_bf[(k_bf["date"] >= pre) & (k_bf["date"] < BACKFILL_CUTOFF)]

    print(f"[load] kline_main (post-{BACKFILL_CUTOFF.date()}): shape={k_main.shape}, "
          f"codes={k_main['code'].nunique()}", flush=True)
    print(f"[load] kline_backfill (pre-{BACKFILL_CUTOFF.date()}): shape={k_bf.shape}, "
          f"codes={k_bf['code'].nunique()}", flush=True)

    combined = pd.concat([k_bf, k_main], ignore_index=True)
    combined = combined.drop_duplicates(subset=["code", "date"], keep="last")
    combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
    return combined


def load_fundamentals(codes: list[str]) -> dict[str, pd.DataFrame]:
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
        print(f"[warn] {len(missing)} codes missing fundamentals "
              f"(first 5: {missing[:5]})", flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────────
# PIT bps + amp_imb_20d + JZF
# ──────────────────────────────────────────────────────────────────────────

def get_pit_bps(df: pd.DataFrame, query_date: pd.Timestamp,
                lag_days: int = ANNOUNCE_LAG_DAYS) -> float:
    cutoff = query_date - pd.Timedelta(days=lag_days)
    visible = df[df["report_date"] <= cutoff]
    if visible.empty:
        return np.nan
    bps = visible.iloc[-1].get("bps")
    if bps is None or pd.isna(bps):
        return np.nan
    return float(bps)


def add_amp_imb_20d(k: pd.DataFrame) -> pd.DataFrame:
    """Same formula as phase_a_value_factors.py / v19.6 production."""
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
    k["_amp_sum"] = grp["_amp"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    k["_amp_up_sum"] = grp["_amp_up"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    k["_amp_dn_sum"] = grp["_amp_dn"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    k["amp_imb_20d"] = np.where(
        k["_amp_sum"] > 0,
        (k["_amp_up_sum"] - k["_amp_dn_sum"]) / k["_amp_sum"],
        np.nan,
    )
    k["JZF"] = np.where(
        valid, (k["open"] - k["prev_close"]) / k["prev_close"] * 100.0, np.nan,
    )
    return k.drop(columns=[c for c in k.columns if c.startswith("_")])


# ──────────────────────────────────────────────────────────────────────────
# Build monthly panel
# ──────────────────────────────────────────────────────────────────────────

def build_panel(kline: pd.DataFrame,
                fund_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    is_dates = pd.DatetimeIndex(sorted(kline["date"].unique()))
    is_dates_in = is_dates[(is_dates >= IS_START) & (is_dates <= IS_END)]
    months = pd.Series(is_dates_in).dt.to_period("M")
    month_first = (
        pd.Series(is_dates_in).groupby(months).first().reset_index(drop=True)
    )
    print(f"[panel] {len(month_first)} monthly anchors: "
          f"{month_first.iloc[0].date()} → {month_first.iloc[-1].date()}", flush=True)

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

        bps_list = []
        for code in df_T["code"]:
            fdf = fund_map.get(code)
            bps_list.append(get_pit_bps(fdf, T) if fdf is not None else np.nan)
        df_T["bps"] = bps_list

        def _safe_pb(c: float, b: float) -> float:
            if pd.isna(c) or pd.isna(b) or b <= 0:
                return np.nan
            return c / b

        df_T["PB"] = [_safe_pb(c, b) for c, b in zip(df_T["close_T"], df_T["bps"], strict=True)]

        a_T = anchor_groups.get(T)
        if a_T is not None and not a_T.empty:
            df_T = df_T.merge(a_T.drop(columns=["date"]), on="code", how="left")
        else:
            df_T["amp_imb_20d"] = np.nan
            df_T["JZF"] = np.nan

        df_T["month_start"] = T
        rows.append(df_T)

    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if panel.empty:
        return panel
    print(f"[panel] rows={len(panel):,}, codes={panel['code'].nunique()}, "
          f"months={panel['month_start'].nunique()}", flush=True)
    return panel[["month_start", "code", "fwd_ret", "PB", "amp_imb_20d", "JZF"]]


# ──────────────────────────────────────────────────────────────────────────
# IC + orthogonality
# ──────────────────────────────────────────────────────────────────────────

def monthly_ic(panel: pd.DataFrame, factor_col: str, sign: int):
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    df["signed_factor"] = sign * df[factor_col]
    obs = df.groupby("month_start").size()

    def _corr(g):
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
            "ic_pos_pct": 0.0, "avg_obs_per_month": float(obs.mean()) if len(obs) else 0,
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


def spearman_orth(panel: pd.DataFrame, factor: str, refs: list[str]) -> pd.DataFrame:
    rows = []
    for ref in refs:
        sub = panel.dropna(subset=[factor, ref])
        per_month = sub.groupby("month_start").apply(
            lambda g: g[factor].corr(g[ref], method="spearman")
            if len(g) >= MIN_MONTHLY_OBS
            and g[factor].std() > 0 and g[ref].std() > 0
            else np.nan
        ).dropna()
        rows.append({
            "factor": factor, "vs": ref,
            "n_months": int(len(per_month)),
            "mean_rho": float(per_month.mean()) if len(per_month) else np.nan,
            "mean_abs_rho": float(per_month.abs().mean()) if len(per_month) else np.nan,
            "max_abs_rho": float(per_month.abs().max()) if len(per_month) else np.nan,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    for p in [KLINE_MAIN, KLINE_BACKFILL, CSI300_PATH, FUND_DIR]:
        if not p.exists():
            print(f"FATAL: missing {p}", file=sys.stderr)
            return 1

    codes = load_csi300_codes()
    print(f"[load] CSI300 universe: {len(codes)} codes", flush=True)

    kline = load_kline_extended(set(codes))
    print(f"[load] combined kline shape={kline.shape}, "
          f"codes={kline['code'].nunique()}, "
          f"date {kline['date'].min().date()} → {kline['date'].max().date()}", flush=True)
    mc = kline.groupby(pd.Grouper(key="date", freq="MS"))["code"].nunique()
    print(f"[load] monthly code coverage: min={mc.min()}, max={mc.max()}, "
          f"median={int(mc.median())}", flush=True)

    print("[compute] amp_imb_20d + JZF...", flush=True)
    kline = add_amp_imb_20d(kline)

    print("[load] fundamentals...", flush=True)
    fund_map = load_fundamentals(codes)
    print(f"[load] {len(fund_map)} stocks with fundamentals cache", flush=True)

    panel = build_panel(kline, fund_map)
    if panel.empty:
        print("FATAL: panel empty", file=sys.stderr)
        return 1

    monthly, summary = monthly_ic(panel, FACTOR_NAME, sign=FACTOR_SIGN)
    summary_df = pd.DataFrame([summary]).round(4)
    summary_df.to_csv(OUT_IC, index=False)
    print(f"\n[output] IC summary -> {OUT_IC}")
    print(summary_df.to_string(index=False))

    monthly_rows = [{"factor": FACTOR_NAME, "month_start": m, "ic": v}
                    for m, v in monthly.items()]
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly IC -> {OUT_MONTHLY}")

    print("\n[orth] Spearman PB vs (amp_imb_20d, JZF)...", flush=True)
    ortho = spearman_orth(panel, FACTOR_NAME, ["amp_imb_20d", "JZF"]).round(4)
    ortho.to_csv(OUT_SPEARMAN, index=False)
    print(f"[output] orthogonality -> {OUT_SPEARMAN}")
    print(ortho.to_string(index=False))

    print("\n[verdict] Phase A pass criteria: |ICIR|>=0.4, n_months>=60, mean|rho|<0.30")
    icir = summary["icir"]
    n_mo = summary["n_months"]
    max_abs_rho = float(ortho["mean_abs_rho"].max()) if not ortho.empty else np.nan
    passes_icir = abs(icir) >= 0.4
    passes_n = n_mo >= 60
    passes_orth = pd.notna(max_abs_rho) and max_abs_rho < 0.30
    verdict = "PASS → Phase B sidecar OOS" if (passes_icir and passes_n and passes_orth) else "ABORT"
    print(f"  PB: ICIR={icir:+.3f} (|.|>=0.4: {passes_icir}) "
          f"n_months={n_mo} (>=60: {passes_n}) "
          f"max_mean_|rho|={max_abs_rho:.3f} (<0.30: {passes_orth}) → {verdict}")

    print(f"\n[baseline-compare] previous PB Phase A: n_months=35, ICIR=-0.483")
    print(f"[baseline-compare] this run:           n_months={n_mo}, ICIR={icir:+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
