"""mt180 Phase A IC 评估 (Discovery, IS only).

对 parse_coverage.csv 中 status='ok' 的 indicators 计算 IS 月度横截面 Spearman IC:
- 期间: 2017-01-01 ~ 2020-12-31 (IS 严格, 不触 OOS 2021+)
- universe: CSI300
- 输入: baidu_kline.parquet (OHLCV)
- factor = compile_tdx(formula)(df) 第一个输出列
- forward return = 20d (close to close)
- IC = month-level cross-sectional Spearman corr(factor, fwd_ret)
- ICIR = IC.mean() * sqrt(12) / IC.std()

输出 `data_cache/mt180/ic_results.csv`.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from claude_finance.tdx_parser import compile_tdx

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/Volumes/SSD/finance/claude_finance")
INDICATORS_FILE = ROOT / "data_cache" / "mt180" / "top_n.jsonl"
COVERAGE_FILE = ROOT / "data_cache" / "mt180" / "parse_coverage.csv"
KLINE_FILE = ROOT / "data_cache" / "baidu_kline.parquet"
UNIVERSE_FILE = ROOT / "data_cache" / "csi300_constituents.csv"
OUTPUT = ROOT / "data_cache" / "mt180" / "ic_results.csv"

IS_START = "2017-01-01"
IS_END = "2020-12-31"
HORIZON_DAYS = 20
MIN_OBS_PER_MONTH = 30
MIN_MONTHS = 24


def _load_universe() -> list[str]:
    df = pd.read_csv(UNIVERSE_FILE, dtype={"code": str})
    return df["code"].tolist()


def _load_kline(codes: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(KLINE_FILE)
    df = df[df["code"].isin(codes)]
    df = df[(df["date"] >= IS_START) & (df["date"] <= IS_END)].copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def _load_indicators() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with INDICATORS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["id"]] = d
    return out


def _compute_factor_panel(cf, kline_groups, output_col: str) -> pd.DataFrame:
    """跑 groupby code 算出 factor, return panel cols [code,date,factor]."""
    parts: list[pd.DataFrame] = []
    for code, sub in kline_groups:
        sub = sub.set_index("date")
        try:
            results = cf(sub)
            if not results or output_col not in results:
                continue
            fact = results[output_col]
            df = pd.DataFrame({
                "code": code,
                "date": sub.index,
                "factor": fact.values,
            })
            parts.append(df)
        except Exception:  # noqa: BLE001
            continue
    if not parts:
        return pd.DataFrame(columns=["code", "date", "factor"])
    return pd.concat(parts, ignore_index=True)


def _compute_ic(
    factor_panel: pd.DataFrame,
    ret_panel: pd.DataFrame,
) -> tuple[float, float, float, int, float] | None:
    if factor_panel.empty:
        return None
    merged = factor_panel.merge(ret_panel, on=["code", "date"], how="inner")
    merged = merged.dropna(subset=["factor", "fwd_ret"])
    if merged.empty:
        return None
    # filter inf
    merged = merged[np.isfinite(merged["factor"]) & np.isfinite(merged["fwd_ret"])]
    if merged.empty:
        return None
    merged["month"] = merged["date"].dt.to_period("M")
    last_per_code_month = merged.groupby(["code", "month"])["date"].max()
    monthly = merged.merge(
        last_per_code_month.rename("month_end").reset_index(),
        on=["code", "month"],
    )
    monthly = monthly[monthly["date"] == monthly["month_end"]].copy()
    if monthly.empty:
        return None

    ic_per_month: list[float] = []
    for _, grp in monthly.groupby("month"):
        if len(grp) < MIN_OBS_PER_MONTH:
            continue
        if grp["factor"].nunique() < 5 or grp["fwd_ret"].nunique() < 5:
            continue
        ic = grp["factor"].corr(grp["fwd_ret"], method="spearman")
        if pd.notna(ic):
            ic_per_month.append(ic)
    if len(ic_per_month) < MIN_MONTHS:
        return None
    arr = np.array(ic_per_month)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    icir = float(mean * np.sqrt(12) / std) if std > 1e-9 else 0.0
    top10_pos = float((arr > 0).mean())
    return mean, std, icir, len(arr), top10_pos


def main() -> None:
    t0 = time.time()
    print(f"Loading universe & kline ({IS_START} ~ {IS_END}) …")
    universe = _load_universe()
    kline = _load_kline(universe)
    print(f"  kline rows={len(kline):,} codes={kline['code'].nunique()}")

    print("Computing forward returns …")
    kline = kline.sort_values(["code", "date"]).reset_index(drop=True)
    kline["fwd_close"] = kline.groupby("code")["close"].shift(-HORIZON_DAYS)
    kline["fwd_ret"] = kline["fwd_close"] / kline["close"] - 1
    ret_panel = kline[["code", "date", "fwd_ret"]].copy()

    # 预 group, 跨 indicator 复用
    kline_groups = list(kline.groupby("code", sort=False))
    print(f"  pre-grouped: {len(kline_groups)} codes")

    cov = pd.read_csv(COVERAGE_FILE)
    ok = cov[cov["parse_status"] == "ok"].copy()
    indicators = _load_indicators()
    print(f"  ok indicators: {len(ok)}")

    results: list[dict] = []
    n_attempted = 0
    n_skipped_empty = 0
    n_skipped_short = 0
    n_compile_fail = 0

    for _, row in ok.iterrows():
        iid = row["id"]
        name = row["name"]
        output_cols_str = str(row["output_cols"]) if pd.notna(row["output_cols"]) else ""
        if not output_cols_str:
            continue
        first_col = output_cols_str.split("|")[0]
        formula = indicators.get(iid, {}).get("formula", "")
        if not formula:
            continue
        n_attempted += 1
        try:
            cf = compile_tdx(formula)
            if cf.status != "ok":
                n_compile_fail += 1
                continue
            factor_panel = _compute_factor_panel(cf, kline_groups, first_col)
            if factor_panel.empty:
                n_skipped_empty += 1
                continue
            ic_res = _compute_ic(factor_panel, ret_panel)
            if ic_res is None:
                n_skipped_short += 1
                continue
            mean, std, icir, n_m, top_pos = ic_res
            results.append({
                "id": iid,
                "name": name,
                "output_col": first_col,
                "ic_mean": mean,
                "ic_std": std,
                "icir": icir,
                "n_months": n_m,
                "top10_pos_pct": top_pos,
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name}: {exc}")
            continue
        if n_attempted % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_attempted}/{len(ok)}] elapsed={elapsed:.1f}s, "
                  f"results={len(results)}, "
                  f"empty={n_skipped_empty}, short={n_skipped_short}")

    df = pd.DataFrame(results)
    if not df.empty:
        df["abs_icir"] = df["icir"].abs()
        df = df.sort_values("abs_icir", ascending=False)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT, index=False)
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s. attempted={n_attempted}, "
              f"ic_results={len(df)}, empty={n_skipped_empty}, "
              f"short={n_skipped_short}, compile_fail={n_compile_fail}")
        print(f"wrote: {OUTPUT}")
        print("\n=== Top 20 by |ICIR| ===\n")
        print(df.head(20)[["name", "output_col", "ic_mean", "icir",
                           "n_months", "top10_pos_pct"]].to_markdown(index=False))
    else:
        print("no IC results!")


if __name__ == "__main__":
    main()
