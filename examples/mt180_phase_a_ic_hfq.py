"""mt180 Phase A IC — hfq base, 2014-2018 (Discovery, IS-only).

目的: 在 mootdx hfq base 上跑 mt180 Phase A IC, 破 baidu qfq 主表 2014-2017 稀疏
(23-44 codes/月) 导致历史 n_months ≤ 60 的天花板.

数据源:
  - `data_cache/baidu_kline_extended_hfq.parquet`  (mootdx hfq 全 universe 2014-2018)
  - 不读 baidu_kline.parquet (qfq base 不能混)
  - 不读 2019+ (单数据源 hfq base 一致)

参数严格保持 mt180_phase_a_ic_extended.py 一致 (避免重 sweep):
  - horizon = 20d
  - MIN_OBS_PER_MONTH = 30
  - MIN_MONTHS = 24
  - universe = all_no_st
  - 月度横截面 Spearman IC
  - ICIR = mean × √12 / std

输出: `data_cache/mt180/ic_results_hfq_2014_2018.csv`
"""

from __future__ import annotations

import json
import multiprocessing as mp
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
KLINE_FILE = ROOT / "data_cache" / "baidu_kline_extended_hfq.parquet"
UNIVERSE_FILE = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "all_no_st.txt"
PREV_IC_FILE = ROOT / "data_cache" / "mt180" / "ic_results_extended.csv"
OUTPUT = ROOT / "data_cache" / "mt180" / "ic_results_hfq_2014_2018.csv"

IS_START = "2014-01-01"
IS_END = "2018-12-31"
IS_PERIOD_STR = f"{IS_START[:7]}~{IS_END[:7]}"
UNIVERSE_NAME = "all_no_st"
HORIZON_DAYS = 20
MIN_OBS_PER_MONTH = 30
MIN_MONTHS = 24
N_WORKERS = 8

# ---- worker-side global state ----
_KLINE_GROUPS: list[tuple[str, pd.DataFrame]] | None = None
_RET_PANEL: pd.DataFrame | None = None


def _load_universe() -> list[str]:
    codes: list[str] = []
    with UNIVERSE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = line.split("\t")[0].strip()
            if raw[:2] in ("SH", "SZ", "BJ"):
                codes.append(raw[2:])
            else:
                codes.append(raw)
    return codes


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
        except Exception:
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


def _worker_init(universe_codes: list[str]) -> None:
    global _KLINE_GROUPS, _RET_PANEL
    df = pd.read_parquet(KLINE_FILE)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["code"].isin(universe_codes)]
    df = df[(df["date"] >= IS_START) & (df["date"] <= IS_END)].copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["fwd_close"] = df.groupby("code")["close"].shift(-HORIZON_DAYS)
    df["fwd_ret"] = df["fwd_close"] / df["close"] - 1
    _RET_PANEL = df[["code", "date", "fwd_ret"]].copy()
    _KLINE_GROUPS = list(df.groupby("code", sort=False))


def _worker_task(args: tuple[str, str, str, str]) -> dict | None:
    iid, name, first_col, formula = args
    try:
        cf = compile_tdx(formula)
        if cf.status != "ok":
            return {"_status": "compile_fail", "id": iid, "name": name}
        factor_panel = _compute_factor_panel(cf, _KLINE_GROUPS, first_col)
        if factor_panel.empty:
            return {"_status": "empty", "id": iid, "name": name}
        ic_res = _compute_ic(factor_panel, _RET_PANEL)
        if ic_res is None:
            return {"_status": "short", "id": iid, "name": name}
        mean, std, icir, n_m, top_pos = ic_res
        return {
            "_status": "ok",
            "id": iid,
            "name": name,
            "output_col": first_col,
            "ic_mean": mean,
            "ic_std": std,
            "icir": icir,
            "n_months": n_m,
            "top10_pos_pct": top_pos,
            "universe": UNIVERSE_NAME,
            "is_period_str": IS_PERIOD_STR,
        }
    except Exception as exc:
        return {"_status": "error", "id": iid, "name": name, "error": str(exc)[:120]}


def _build_tasks() -> list[tuple[str, str, str, str]]:
    cov = pd.read_csv(COVERAGE_FILE)
    ok = cov[cov["parse_status"] == "ok"].copy()
    indicators = _load_indicators()
    tasks: list[tuple[str, str, str, str]] = []
    for _, row in ok.iterrows():
        iid = row["id"]
        output_cols_str = str(row["output_cols"]) if pd.notna(row["output_cols"]) else ""
        if not output_cols_str:
            continue
        first_col = output_cols_str.split("|")[0]
        formula = indicators.get(iid, {}).get("formula", "")
        if not formula:
            continue
        tasks.append((iid, row["name"], first_col, formula))
    return tasks


def _compare_with_prev(df_new: pd.DataFrame) -> str:
    """Compare Top 20 rank between new (hfq 2014-2018) and prev (qfq 2014-2020 extended)."""
    if not PREV_IC_FILE.exists():
        return "(prev ic_results_extended.csv not found — skip comparison)"
    prev = pd.read_csv(PREV_IC_FILE)
    prev["abs_icir"] = prev["icir"].abs()
    prev = prev.sort_values("abs_icir", ascending=False).reset_index(drop=True)
    prev["rank_prev"] = prev.index + 1
    prev_top20_ids = set(prev.head(20)["id"])

    new = df_new.copy()
    new["abs_icir"] = new["icir"].abs()
    new = new.sort_values("abs_icir", ascending=False).reset_index(drop=True)
    new["rank_new"] = new.index + 1
    new_top20_ids = set(new.head(20)["id"])

    overlap = prev_top20_ids & new_top20_ids
    only_new = new_top20_ids - prev_top20_ids
    only_prev = prev_top20_ids - new_top20_ids

    joined = new[["id", "name", "icir", "rank_new", "abs_icir"]].merge(
        prev[["id", "icir", "rank_prev", "abs_icir"]].rename(
            columns={"icir": "icir_prev", "abs_icir": "abs_icir_prev"}
        ),
        on="id",
        how="left",
    )
    joined["rank_delta"] = joined["rank_new"] - joined["rank_prev"]

    lines: list[str] = []
    lines.append(f"\n## Top 20 rank stability: hfq 2014-2018 vs qfq 2014-2020 extended\n")
    lines.append(f"- overlap in Top 20: {len(overlap)}/20")
    lines.append(f"- only in new hfq Top 20 (newly emerged): {len(only_new)}")
    lines.append(f"- only in prev qfq Top 20 (lost rank): {len(only_prev)}")
    lines.append("")
    stable = joined[
        joined["rank_new"].le(20)
        & joined["rank_prev"].le(20)
        & joined["rank_delta"].abs().le(10)
    ].sort_values("rank_new")
    if not stable.empty:
        lines.append("### Stable (rank Δ ≤ 10, both in Top 20):\n")
        try:
            lines.append(
                stable[["name", "rank_prev", "rank_new", "rank_delta", "icir_prev", "icir"]]
                .head(20)
                .to_markdown(index=False)
            )
        except ImportError:
            lines.append(
                stable[["name", "rank_prev", "rank_new", "rank_delta", "icir_prev", "icir"]]
                .head(20)
                .to_string(index=False)
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    print(f"=== mt180 Phase A IC HFQ (2014-2018) ===")
    print(f"Universe: {UNIVERSE_NAME}  ({UNIVERSE_FILE.name})")
    print(f"IS period: {IS_PERIOD_STR}  (5 years)")
    print(f"Horizon: {HORIZON_DAYS}d, MIN_OBS_PER_MONTH={MIN_OBS_PER_MONTH}, MIN_MONTHS={MIN_MONTHS}")
    print(f"KLINE_FILE: {KLINE_FILE.name}")
    print(f"Workers: {N_WORKERS}")
    print()

    if not KLINE_FILE.exists():
        print(f"FATAL: {KLINE_FILE} not found")
        return

    universe = _load_universe()
    bk_codes = set(pd.read_parquet(KLINE_FILE, columns=["code"])["code"].astype(str).str.zfill(6).unique())
    universe = [c for c in universe if c in bk_codes]
    print(f"Universe size (intersect hfq kline): {len(universe)}")

    tasks = _build_tasks()
    print(f"Tasks (ok indicators with output_cols): {len(tasks)}")

    print(f"\nLaunching {N_WORKERS} workers …")
    ctx = mp.get_context("spawn")
    results: list[dict] = []
    n_compile_fail = 0
    n_empty = 0
    n_short = 0
    n_error = 0
    n_ok = 0
    with ctx.Pool(
        processes=N_WORKERS,
        initializer=_worker_init,
        initargs=(universe,),
    ) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker_task, tasks, chunksize=2)):
            if res is None:
                continue
            status = res.pop("_status", None)
            if status == "ok":
                n_ok += 1
                results.append(res)
            elif status == "compile_fail":
                n_compile_fail += 1
            elif status == "empty":
                n_empty += 1
            elif status == "short":
                n_short += 1
            elif status == "error":
                n_error += 1
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate if rate > 0 else float("nan")
                print(
                    f"  [{i+1}/{len(tasks)}] elapsed={elapsed:.1f}s, "
                    f"ok={n_ok}, empty={n_empty}, short={n_short}, "
                    f"compile_fail={n_compile_fail}, error={n_error}, "
                    f"eta={eta:.1f}s"
                )

    df = pd.DataFrame(results)
    if df.empty:
        print("no IC results!")
        return

    df["abs_icir"] = df["icir"].abs()
    df = df.sort_values("abs_icir", ascending=False).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"n_attempted={len(tasks)}, ic_results={len(df)}, "
          f"empty={n_empty}, short={n_short}, "
          f"compile_fail={n_compile_fail}, error={n_error}")
    print(f"wrote: {OUTPUT}")

    n_dist = df["n_months"].value_counts().sort_index()
    print(f"\n## n_months distribution\n")
    print(f"  min={df['n_months'].min()}, median={df['n_months'].median():.0f}, "
          f"max={df['n_months'].max()}")
    print(f"  full dist: {n_dist.to_dict()}")
    buckets = [(0, 35), (36, 36), (37, 47), (48, 59), (60, 60), (61, 84)]
    print(f"  bucketed:")
    for lo, hi in buckets:
        n = ((df["n_months"] >= lo) & (df["n_months"] <= hi)).sum()
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        pct = 100 * n / len(df) if len(df) else 0
        print(f"    {label:>10s}: {n:4d}  ({pct:.1f}%)")

    print(f"\n## Top 20 by |ICIR|\n")
    top20 = df.head(20)[
        ["name", "output_col", "ic_mean", "icir", "n_months", "top10_pos_pct"]
    ]
    try:
        print(top20.to_markdown(index=False))
    except ImportError:
        print(top20.to_string(index=False))

    print(_compare_with_prev(df))


if __name__ == "__main__":
    main()
