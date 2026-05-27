"""Shared helper: 从 baidu_kline (hfq) 计算 intraday/overnight/amplitude 因子.

供 v19.5 (team_coin) + v19.6 (amplitude) sidecar 共用.

参考: hugo2046/QuantsPlaybook 5.1k★ 仓库 "球队硬币" 与 "振幅隐藏结构" 笔记本.

因子定义:
    intraday_ret = close / open - 1         # 日内 (开→收)
    overnight_ret = open / prev_close - 1   # 隔夜 (上日收→今开)
    team_coin = intraday_ret * overnight_ret
    team_coin_5d / 20d = rolling sum (per code)

    amp = (high - low) / prev_close
    amp_up = max(0, close - prev_close) / prev_close * amp
    amp_dn = max(0, prev_close - close) / prev_close * amp
    amp_imb_20d = (sum(amp_up) - sum(amp_dn)) / sum(amp)
    amp_imb_5d  = same on 5d window

ICIR 文献参考: team_coin -4.73, amp -2.97. **反向使用 (sign -1)** 在 sidecar.

hfq vs qfq: 同一股内复权 factor 一致 → ratio (close/open, high/low) 不变.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KLINE_PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"


def _zscore_cs(s: pd.Series) -> pd.Series:
    """Cross-section z-score within a group, NaN-safe."""
    mu = s.mean()
    sd = s.std()
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _instrument_to_code6(inst: str) -> str:
    return str(inst)[-6:].zfill(6)


def _code6_to_instrument(c: str) -> str:
    c = str(c).zfill(6)
    if c.startswith("6"):
        return f"SH{c}"
    return f"SZ{c}"


def build_kline_factors(min_date: str = "2016-01-01") -> pd.DataFrame:
    """计算每个 (code, date) 上的 5 个因子.

    Returns columns:
        code, date, team_coin_5d, team_coin_20d,
        amp_imb_5d, amp_imb_20d
    """
    print(f"[kline] loading {KLINE_PARQUET.name}...", flush=True)
    k = pd.read_parquet(
        KLINE_PARQUET,
        columns=["code", "date", "open", "high", "low", "close"],
    )
    k["code"] = k["code"].astype(str).str.zfill(6)
    k["date"] = pd.to_datetime(k["date"])
    k = k[k["date"] >= pd.Timestamp(min_date)].copy()
    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"[kline] rows={len(k):,}  codes={k['code'].nunique()}  "
          f"date range {k['date'].min().date()} ~ {k['date'].max().date()}",
          flush=True)

    # prev_close (per code)
    k["prev_close"] = k.groupby("code", sort=False)["close"].shift(1)
    valid = k["prev_close"].notna() & (k["prev_close"] > 0) & (k["open"] > 0)
    n_inv = (~valid).sum()
    if n_inv:
        print(f"[kline] drop {n_inv:,} rows with NaN/0 prev_close|open",
              flush=True)

    # team_coin daily
    intraday = k["close"] / k["open"] - 1
    overnight = k["open"] / k["prev_close"] - 1
    k["team_coin"] = intraday * overnight
    k.loc[~valid, "team_coin"] = np.nan

    # amplitude daily decomposition
    amp = (k["high"] - k["low"]) / k["prev_close"]
    delta = k["close"] - k["prev_close"]
    amp_up = np.where(delta > 0, delta, 0.0) / k["prev_close"] * amp
    amp_dn = np.where(delta < 0, -delta, 0.0) / k["prev_close"] * amp
    k["amp"] = amp
    k["amp_up"] = amp_up
    k["amp_dn"] = amp_dn
    k.loc[~valid, ["amp", "amp_up", "amp_dn"]] = np.nan

    # Rolling 5d / 20d (per code)
    grp = k.groupby("code", sort=False)
    print("[kline] computing rolling 5d/20d sums...", flush=True)
    k["team_coin_5d"] = grp["team_coin"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    k["team_coin_20d"] = grp["team_coin"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["amp_sum_5d"] = grp["amp"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    k["amp_sum_20d"] = grp["amp"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["amp_up_sum_5d"] = grp["amp_up"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    k["amp_up_sum_20d"] = grp["amp_up"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["amp_dn_sum_5d"] = grp["amp_dn"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    k["amp_dn_sum_20d"] = grp["amp_dn"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )

    # amp imbalance — guard div-by-zero
    def _imb(num, den):
        return np.where(den > 0, num / den, np.nan)

    k["amp_imb_5d"] = _imb(
        k["amp_up_sum_5d"] - k["amp_dn_sum_5d"], k["amp_sum_5d"]
    )
    k["amp_imb_20d"] = _imb(
        k["amp_up_sum_20d"] - k["amp_dn_sum_20d"], k["amp_sum_20d"]
    )

    out = k[[
        "code", "date",
        "team_coin_5d", "team_coin_20d",
        "amp_imb_5d", "amp_imb_20d",
    ]].copy()
    print(f"[kline] factor rows={len(out):,}; "
          f"team_coin_5d coverage={out['team_coin_5d'].notna().mean()*100:.1f}%, "
          f"team_coin_20d={out['team_coin_20d'].notna().mean()*100:.1f}%, "
          f"amp_imb_20d={out['amp_imb_20d'].notna().mean()*100:.1f}%",
          flush=True)
    return out


def build_pit_panel_on_pred_axis(
    pred_path: Path, factor_cols: list[str], min_date: str = "2016-01-01",
) -> pd.DataFrame:
    """构造 (datetime, instrument) × factor_cols panel, 在 pred axis 上 point-in-time.

    对每个 pred (datetime, instrument), 取该 code 的 date <= datetime 的最新因子值.
    然后做 datetime 内横截面 z-score, fillna(0).

    Returns: DataFrame with [datetime, instrument, z_{col}] for each col in factor_cols.
    """
    factors = build_kline_factors(min_date=min_date)

    pred = pd.read_parquet(pred_path, columns=["datetime", "instrument"])
    pred["code"] = pred["instrument"].apply(_instrument_to_code6)
    pred_dt = pd.DatetimeIndex(sorted(pred["datetime"].unique()))
    print(f"[panel] pred dates: {len(pred_dt)}, "
          f"unique instruments: {pred['instrument'].nunique()}", flush=True)

    # PIT join: for each (code), searchsorted on factor dates vs pred_dt
    print(f"[panel] PIT join factors -> pred axis "
          f"(cols={factor_cols})...", flush=True)
    parts = []
    factors_sorted = factors.sort_values(["code", "date"])
    for code, sub in factors_sorted.groupby("code", sort=False):
        dates_arr = sub["date"].values
        idx = np.searchsorted(dates_arr, pred_dt.values, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        safe_idx = np.clip(idx, 0, len(sub) - 1)
        row = {"datetime": pred_dt, "code": code}
        for c in factor_cols:
            v = sub[c].values
            row[c] = np.where(valid, v[safe_idx], np.nan)
        parts.append(pd.DataFrame(row))
    if not parts:
        raise RuntimeError("[panel] no PIT factor rows built")
    panel = pd.concat(parts, ignore_index=True)
    print(f"[panel] PIT panel rows={len(panel):,}", flush=True)

    # Join into pred axis
    pred_axis = pred[["datetime", "instrument", "code"]].drop_duplicates()
    out = pred_axis.merge(panel, on=["datetime", "code"], how="left")

    n_total = len(out)
    for c in factor_cols:
        n_ok = out[c].notna().sum()
        print(f"[panel] {c}: {n_ok:,} ({n_ok/n_total*100:.1f}%)", flush=True)

    # Cross-section z-scores per datetime
    print("[panel] computing cross-section z-scores per date...", flush=True)
    for c in factor_cols:
        zcol = f"z_{c}"
        out[zcol] = out.groupby("datetime")[c].transform(_zscore_cs)
        out[zcol] = out[zcol].fillna(0.0)
    keep = ["datetime", "instrument"] + [f"z_{c}" for c in factor_cols]
    return out[keep]


if __name__ == "__main__":
    # smoke test
    f = build_kline_factors()
    print(f.head(10))
    print(f.tail(5))
