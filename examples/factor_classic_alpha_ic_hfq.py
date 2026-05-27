"""8 经典 alpha factor Phase A IC — hfq base, 2014-2018 (Discovery, IS-only).

复刻 `factor_classic_alpha_ic.py` 全部 logic, 只改:
  - KLINE_PATH: baidu_kline.parquet (qfq)        → baidu_kline_extended_hfq.parquet (hfq)
  - IS_END:     2020-12-31                       → 2018-12-31
  - OUT_CSV:    classic_ic_results.csv           → classic_ic_results_hfq.csv
  - OUT_SPEARMAN: classic_ic_spearman.csv        → classic_ic_spearman_hfq.csv

参数严格保持 (避免重 sweep): horizon=20d, MIN_OBS=30, MIN_MONTHS=24.

目的: 在 mootdx hfq base 上验证 8 经典因子 (Amihud / 52w / idio_vol / RSI /
Bollinger / WR / MFI / OBV slope), 看 IS n_months 能否破 60 月天花板.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

KLINE_PATH = ROOT / "data_cache" / "baidu_kline_extended_hfq.parquet"
UNIVERSE_PATH = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "all_no_st.txt"
MARGIN_PATH = ROOT / "data_cache" / "margin_180_backfill.parquet"
OUT_DIR = ROOT / "data_cache" / "factors"
OUT_CSV = OUT_DIR / "classic_ic_results_hfq.csv"
OUT_SPEARMAN = OUT_DIR / "classic_ic_spearman_hfq.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2018-12-31")
PRE_BUFFER_DAYS = 380
POST_BUFFER_DAYS = 35
FORWARD_DAYS = 20
MIN_MONTHLY_OBS = 30
MIN_MONTHS = 24

FACTOR_HYPOTHESIS: dict[str, int] = {
    "amihud_20d":     +1,
    "high_52w_dist":  +1,
    "idio_vol_20d":   -1,
    "rsi_14":         -1,
    "bb_pos_20d":     -1,
    "williams_r_14":  +1,
    "mfi_14":         -1,
    "obv_slope_60d":  +1,
}

FACTORS = list(FACTOR_HYPOTHESIS.keys())


def load_universe() -> set[str]:
    with UNIVERSE_PATH.open() as f:
        codes = [ln.strip().split("\t")[0] for ln in f if ln.strip()]
    out = set()
    for c in codes:
        if c[:2] in ("SH", "SZ", "BJ"):
            c = c[2:]
        if c.startswith("688") or c.startswith("8"):
            continue
        if not c.isdigit() or len(c) != 6:
            continue
        out.add(c)
    return out


def load_kline_is_with_buffer(codes: set[str]) -> pd.DataFrame:
    pre_start = IS_START - pd.Timedelta(days=PRE_BUFFER_DAYS)
    post_end = IS_END + pd.Timedelta(days=POST_BUFFER_DAYS)
    k = pd.read_parquet(
        KLINE_PATH,
        columns=["code", "date", "open", "high", "low", "close", "vol"],
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    k = k[(k["date"] >= pre_start) & (k["date"] <= post_end)]
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    for col in ("open", "high", "low", "close", "vol"):
        k[col] = k[col].astype(float)
    print(
        f"[load] kline shape={k.shape}, "
        f"date {k['date'].min().date()} ~ {k['date'].max().date()}, "
        f"codes={k['code'].nunique()}",
        flush=True,
    )
    return k


# ===== per-code factor functions =====

def _amihud_20d(sub: pd.DataFrame) -> pd.Series:
    close = sub["close"]
    vol = sub["vol"]
    ret = close.pct_change()
    amount = close * vol
    illiq = (ret.abs() / amount.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return illiq.shift(1).rolling(20, min_periods=20).mean()


def _high_52w_dist(sub: pd.DataFrame) -> pd.Series:
    close = sub["close"]
    max252 = close.shift(1).rolling(252, min_periods=252).max()
    return (close - max252) / max252


def _daily_ret_for_idio(sub: pd.DataFrame) -> pd.Series:
    return sub["close"].pct_change()


def _rsi_14(sub: pd.DataFrame) -> pd.Series:
    diff = sub["close"].diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)
    avg_g = gain.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    avg_l = loss.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _bb_pos_20d(sub: pd.DataFrame) -> pd.Series:
    close = sub["close"]
    ma = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std()
    return (close - ma) / sd.replace(0, np.nan)


def _williams_r_14(sub: pd.DataFrame) -> pd.Series:
    hhv = sub["high"].rolling(14, min_periods=14).max()
    llv = sub["low"].rolling(14, min_periods=14).min()
    denom = (hhv - llv).replace(0, np.nan)
    return (hhv - sub["close"]) / denom * -100.0


def _mfi_14(sub: pd.DataFrame) -> pd.Series:
    typical = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    mf = typical * sub["vol"]
    diff = typical.diff()
    pos_mf = mf.where(diff > 0, 0.0)
    neg_mf = mf.where(diff < 0, 0.0)
    pos_sum = pos_mf.rolling(14, min_periods=14).sum()
    neg_sum = neg_mf.rolling(14, min_periods=14).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


def _obv_slope_60d(sub: pd.DataFrame) -> pd.Series:
    diff = sub["close"].diff()
    sign = np.sign(diff.fillna(0))
    obv = (sign * sub["vol"]).cumsum().to_numpy()
    n = 60
    L = len(obv)
    if L < n:
        return pd.Series(np.full(L, np.nan), index=sub.index)
    x = np.arange(n, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = n * sum_x2 - sum_x * sum_x
    sum_y_roll = np.convolve(obv, np.ones(n)[::-1], mode="valid")
    sum_xy_roll = np.convolve(obv, x[::-1], mode="valid")
    slope_valid = (n * sum_xy_roll - sum_x * sum_y_roll) / denom
    out = np.full(L, np.nan)
    out[n - 1 :] = slope_valid
    return pd.Series(out, index=sub.index)


def compute_factors(kline: pd.DataFrame) -> pd.DataFrame:
    print("[factor] computing 8 classic factors per groupby code ...", flush=True)
    t0 = time.time()
    k = kline.copy()

    grp = k.groupby("code", sort=False, group_keys=False)

    k["amihud_20d"] = grp.apply(_amihud_20d)
    print(f"  [1/8] amihud_20d        done ({time.time()-t0:.1f}s)", flush=True)

    k["high_52w_dist"] = grp.apply(_high_52w_dist)
    print(f"  [2/8] high_52w_dist     done ({time.time()-t0:.1f}s)", flush=True)

    k["_ret"] = grp.apply(_daily_ret_for_idio)
    xs_mean = k.groupby("date")["_ret"].transform("mean")
    k["_ret_resid"] = k["_ret"] - xs_mean
    k["idio_vol_20d"] = k.groupby("code", sort=False, group_keys=False)[
        "_ret_resid"
    ].transform(lambda s: s.rolling(20, min_periods=20).std())
    k = k.drop(columns=["_ret", "_ret_resid"])
    print(f"  [3/8] idio_vol_20d      done ({time.time()-t0:.1f}s)", flush=True)

    k["rsi_14"] = k.groupby("code", sort=False, group_keys=False).apply(_rsi_14)
    print(f"  [4/8] rsi_14            done ({time.time()-t0:.1f}s)", flush=True)

    k["bb_pos_20d"] = k.groupby("code", sort=False, group_keys=False).apply(_bb_pos_20d)
    print(f"  [5/8] bb_pos_20d        done ({time.time()-t0:.1f}s)", flush=True)

    k["williams_r_14"] = k.groupby("code", sort=False, group_keys=False).apply(_williams_r_14)
    print(f"  [6/8] williams_r_14     done ({time.time()-t0:.1f}s)", flush=True)

    k["mfi_14"] = k.groupby("code", sort=False, group_keys=False).apply(_mfi_14)
    print(f"  [7/8] mfi_14            done ({time.time()-t0:.1f}s)", flush=True)

    k["obv_slope_60d"] = k.groupby("code", sort=False, group_keys=False).apply(_obv_slope_60d)
    print(f"  [8/8] obv_slope_60d     done ({time.time()-t0:.1f}s)", flush=True)

    for f in FACTORS:
        k[f] = k[f].replace([np.inf, -np.inf], np.nan)

    cov_total = len(k)
    print(f"[factor] panel rows={cov_total:,}; coverage:")
    for f in FACTORS:
        n = int(k[f].notna().sum())
        print(f"  {f:18s} {n:>10,d}  ({n/cov_total*100:.1f}%)")
    return k[["code", "date", "close"] + FACTORS]


def build_amp_imb_20d(codes: set[str]) -> pd.DataFrame:
    print("[orth] computing amp_imb_20d ...", flush=True)
    pre_start = IS_START - pd.Timedelta(days=60)
    k = pd.read_parquet(
        KLINE_PATH, columns=["code", "date", "high", "low", "close"]
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    k = k[(k["date"] >= pre_start) & (k["date"] <= IS_END + pd.Timedelta(days=5))]
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    k["prev_close"] = k.groupby("code", sort=False)["close"].shift(1)
    valid = k["prev_close"].notna() & (k["prev_close"] > 0)
    amp = (k["high"] - k["low"]) / k["prev_close"]
    delta = k["close"] - k["prev_close"]
    amp_up = np.where(delta > 0, delta, 0.0) / k["prev_close"] * amp
    amp_dn = np.where(delta < 0, -delta, 0.0) / k["prev_close"] * amp
    k["amp"] = amp
    k["amp_up"] = amp_up
    k["amp_dn"] = amp_dn
    k.loc[~valid, ["amp", "amp_up", "amp_dn"]] = np.nan
    g = k.groupby("code", sort=False)
    k["amp_sum_20d"] = g["amp"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    k["amp_up_sum_20d"] = g["amp_up"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    k["amp_dn_sum_20d"] = g["amp_dn"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    k["amp_imb_20d"] = np.where(
        k["amp_sum_20d"] > 0,
        (k["amp_up_sum_20d"] - k["amp_dn_sum_20d"]) / k["amp_sum_20d"],
        np.nan,
    )
    return k[["code", "date", "amp_imb_20d"]]


def load_margin_factors() -> pd.DataFrame:
    if not MARGIN_PATH.exists():
        return pd.DataFrame(columns=["code", "date", "margin_5d_chg", "margin_20d_chg"])
    m = pd.read_parquet(MARGIN_PATH)
    m["code"] = m["code"].astype(str).str.zfill(6)
    m = m[(m["date"] >= IS_START) & (m["date"] <= IS_END + pd.Timedelta(days=5))]
    return m[["code", "date", "margin_5d_chg", "margin_20d_chg"]]


def build_monthly_panel(
    factor_kline: pd.DataFrame,
    amp_df: pd.DataFrame,
    margin_df: pd.DataFrame,
) -> pd.DataFrame:
    print("[panel] building monthly panel ...", flush=True)
    is_dates = pd.DatetimeIndex(sorted(factor_kline["date"].unique()))
    is_dates_in = is_dates[(is_dates >= IS_START) & (is_dates <= IS_END)]
    months = pd.Series(is_dates_in).dt.to_period("M")
    month_first = pd.Series(is_dates_in).groupby(months).first().reset_index(drop=True)
    print(
        f"[panel] {len(month_first)} monthly anchors: "
        f"{month_first.iloc[0].date()} → {month_first.iloc[-1].date()}",
        flush=True,
    )

    wide = factor_kline.pivot_table(
        index="date", columns="code", values="close", aggfunc="first"
    ).sort_index()

    anchor_set = set(month_first.tolist())
    f_anchor = factor_kline[factor_kline["date"].isin(anchor_set)][
        ["code", "date"] + FACTORS
    ].copy()
    f_groups = dict(list(f_anchor.groupby("date", sort=False)))
    a_groups: dict = {}
    if not amp_df.empty:
        a_anchor = amp_df[amp_df["date"].isin(anchor_set)][
            ["code", "date", "amp_imb_20d"]
        ].copy()
        if not a_anchor.empty:
            a_groups = dict(list(a_anchor.groupby("date", sort=False)))
    m_groups: dict = {}
    if not margin_df.empty:
        m_anchor = margin_df[margin_df["date"].isin(anchor_set)][
            ["code", "date", "margin_5d_chg", "margin_20d_chg"]
        ].copy()
        if not m_anchor.empty:
            m_groups = dict(list(m_anchor.groupby("date", sort=False)))

    rows = []
    for T in month_first:
        idx_arr = wide.index.get_indexer([T])
        idx = int(idx_arr[0]) if len(idx_arr) else -1
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

        a_T = a_groups.get(T) if a_groups else None
        if a_T is not None and not a_T.empty:
            df_T = df_T.merge(a_T.drop(columns=["date"]), on="code", how="left")
        else:
            df_T["amp_imb_20d"] = np.nan

        m_T = m_groups.get(T) if m_groups else None
        if m_T is not None and not m_T.empty:
            df_T = df_T.merge(m_T.drop(columns=["date"]), on="code", how="left")
        else:
            df_T["margin_5d_chg"] = np.nan
            df_T["margin_20d_chg"] = np.nan

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
    return panel


def monthly_ic(panel: pd.DataFrame, factor_col: str, sign: int) -> dict:
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    if df.empty:
        return {
            "factor_name": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
            "top10_pos_pct": 0.0, "avg_obs_per_month": 0.0,
        }
    df["sf"] = sign * df[factor_col]

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < MIN_MONTHLY_OBS:
            return np.nan
        if g["sf"].std() <= 0 or g["fwd_ret"].std() <= 0:
            return np.nan
        return g["sf"].corr(g["fwd_ret"], method="spearman")

    obs = df.groupby("month_start").size()
    monthly = df.groupby("month_start").apply(_corr).dropna()
    if len(monthly) == 0:
        return {
            "factor_name": factor_col, "sign": sign, "n_months": 0,
            "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "top10_pos_pct": 0.0,
            "avg_obs_per_month": float(obs.mean()) if len(obs) else 0.0,
        }
    mean = float(monthly.mean())
    std = float(monthly.std(ddof=1))
    icir = mean / std * np.sqrt(12) if std > 0 else 0.0
    return {
        "factor_name": factor_col, "sign": sign, "n_months": int(len(monthly)),
        "ic_mean": mean, "ic_std": std, "icir": icir,
        "top10_pos_pct": float((monthly > 0).mean() * 100),
        "avg_obs_per_month": float(obs.mean()),
    }


def spearman_orth(panel: pd.DataFrame, ref_col: str) -> pd.DataFrame:
    rows = []
    for f in FACTORS:
        sub = panel.dropna(subset=[f, ref_col])
        if sub.empty:
            rows.append({
                "factor_name": f, "ref": ref_col, "n_months": 0,
                "mean_rho": np.nan, "mean_abs_rho": np.nan, "max_abs_rho": np.nan,
            })
            continue
        per_month = sub.groupby("month_start").apply(
            lambda g: g[f].corr(g[ref_col], method="spearman")
            if len(g) >= MIN_MONTHLY_OBS and g[f].std() > 0 and g[ref_col].std() > 0
            else np.nan
        ).dropna()
        rows.append({
            "factor_name": f, "ref": ref_col, "n_months": int(len(per_month)),
            "mean_rho": float(per_month.mean()) if len(per_month) else np.nan,
            "mean_abs_rho": float(per_month.abs().mean()) if len(per_month) else np.nan,
            "max_abs_rho": float(per_month.abs().max()) if len(per_month) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    if not KLINE_PATH.exists():
        print(f"FATAL: missing {KLINE_PATH}", file=sys.stderr)
        return 1
    if not UNIVERSE_PATH.exists():
        print(f"FATAL: missing {UNIVERSE_PATH}", file=sys.stderr)
        return 1

    print(f"[start] {pd.Timestamp.now()}", flush=True)
    print(f"[config] IS {IS_START.date()} ~ {IS_END.date()}, horizon={FORWARD_DAYS}d")
    print(f"[config] KLINE_PATH={KLINE_PATH.name}", flush=True)
    print(f"[config] MIN_OBS={MIN_MONTHLY_OBS}, MIN_MONTHS={MIN_MONTHS}", flush=True)

    universe = load_universe()
    print(f"[universe] all_no_st minus 688/8 = {len(universe)} codes", flush=True)

    kline = load_kline_is_with_buffer(universe)
    factor_df = compute_factors(kline)
    amp_df = build_amp_imb_20d(universe)
    margin_df = load_margin_factors()
    print(
        f"[orth] amp_imb_20d rows={len(amp_df):,}, "
        f"margin rows={len(margin_df):,}",
        flush=True,
    )

    panel = build_monthly_panel(factor_df, amp_df, margin_df)
    if panel.empty:
        print("FATAL: panel is empty", file=sys.stderr)
        return 1

    print("\n[step IC] monthly cross-section IC (signed per hypothesis) ...", flush=True)
    rows: list[dict] = []
    for f, sign in FACTOR_HYPOTHESIS.items():
        summ = monthly_ic(panel, f, sign=sign)
        if summ["n_months"] < MIN_MONTHS:
            summ["hypothesis_match"] = "no_data"
        else:
            summ["hypothesis_match"] = "yes" if summ["ic_mean"] > 0 else "no"
        rows.append(summ)

    res = pd.DataFrame(rows)
    res = res[[
        "factor_name", "sign", "ic_mean", "ic_std", "icir", "n_months",
        "top10_pos_pct", "avg_obs_per_month", "hypothesis_match",
    ]].round(4)
    res = res.sort_values("icir", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    print(f"\n[output] {OUT_CSV}")
    try:
        print(res.to_markdown(index=False))
    except ImportError:
        print(res.to_string(index=False))

    print("\n[step Spearman] vs known sidecar factors ...", flush=True)
    parts = []
    for ref in ("amp_imb_20d", "margin_5d_chg", "margin_20d_chg"):
        if ref not in panel.columns or panel[ref].notna().sum() == 0:
            print(f"  [skip] {ref}: no data in panel")
            continue
        orth = spearman_orth(panel, ref)
        parts.append(orth)
    if parts:
        orth_df = pd.concat(parts, ignore_index=True).round(4)
        orth_df.to_csv(OUT_SPEARMAN, index=False)
        print(f"[output] {OUT_SPEARMAN}")
        try:
            print(orth_df.to_markdown(index=False))
        except ImportError:
            print(orth_df.to_string(index=False))
    else:
        print("[skip] orthogonality (no reference factors available)")

    print("\n[verdict] Phase B candidates (|ICIR| > 1.0):", flush=True)
    phase_b = res[res["icir"].abs() > 1.0]
    if phase_b.empty:
        print("  none — no factor with |ICIR| > 1.0")
    else:
        for _, r in phase_b.iterrows():
            print(
                f"  {r['factor_name']:18s} sign={int(r['sign']):+d} "
                f"ICIR={r['icir']:+.3f} ic_mean={r['ic_mean']:+.4f} "
                f"n_months={int(r['n_months'])} "
                f"top10_pos_pct={r['top10_pos_pct']:.1f}% "
                f"hypothesis_match={r['hypothesis_match']}"
            )

    elapsed = time.time() - t0
    print(f"\n[done] wall {elapsed/60:.2f} min ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
