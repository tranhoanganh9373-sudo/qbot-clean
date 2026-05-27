"""CSI300 解禁因子 IC 分析 — 严格 IS 2014-2020.

方法 (point-in-time, no lookahead):
- 每月第 1 个交易日 T:
  - 因子值 = forward_unlock_metrics(code, T, windows=(5,20,60))
    (T 时点已知的 unlock 时间表; 用未来 5/20/60 日 calendar 解禁汇总)
  - 前向收益 = close(T+5) / close(T) - 1
- 横截面 Spearman corr(factor_z, fwd_z) → monthly IC
- 月度 IC 序列 → mean / std / ICIR = mean / std * sqrt(12) / %>0

候选因子 (8 个):
  raw (sign=+1 评估; 文献期望 IC 负, 即 大解禁后跌):
    1. unlock_pct_next_5
    2. unlock_pct_next_20
    3. unlock_pct_next_60
    4. unlock_imminent_5      (binary)
    5. unlock_imminent_20
    6. unlock_value_next_5    (亿元)
    7. unlock_value_next_20
  combo:
    8. combo_neg_pct = -(z20 + z60)/2  (反向: 期望 IC > 0)

判定:
  best |ICIR| > 0.5 → 进 Phase B
  0.3 < |ICIR| < 0.5 → 边际
  |ICIR| < 0.3 → abort
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

from claude_finance import unlock  # noqa: E402

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline_v2.parquet"
OUT_CSV = ROOT / "examples" / "factor_ic_unlock_csi300_is.csv"
OUT_MONTHLY = ROOT / "examples" / "factor_ic_unlock_csi300_is_monthly.csv"
OUT_REPORT = ROOT / "examples" / "factor_ic_unlock_csi300_is_report.md"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 5

SINGLE_FACTORS = [
    ("unlock_pct_next_5", +1),
    ("unlock_pct_next_20", +1),
    ("unlock_pct_next_60", +1),
    ("unlock_imminent_5", +1),
    ("unlock_imminent_20", +1),
    ("unlock_value_next_5", +1),
    ("unlock_value_next_20", +1),
]


def load_kline_is() -> pd.DataFrame:
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = set(csi["code"].tolist())

    k = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    k = k[(k["date"] >= IS_START) &
          (k["date"] <= IS_END + pd.Timedelta(days=15))]
    return k.sort_values(["code", "date"]).reset_index(drop=True)


def build_panel(kline: pd.DataFrame,
                unlock_df: pd.DataFrame) -> pd.DataFrame:
    """构造 panel: (asof_date, code, factor1..N, fwd_ret)."""
    kline_dates = pd.DatetimeIndex(sorted(kline["date"].unique()))
    is_dates = kline_dates[
        (kline_dates >= IS_START) & (kline_dates <= IS_END)
    ]
    months = pd.Series(is_dates).dt.to_period("M")
    month_first_day = (
        pd.Series(is_dates).groupby(months).first().reset_index(drop=True)
    )
    print(f"[panel] {len(month_first_day)} 月度采样点 "
          f"({month_first_day.iloc[0].date()} → "
          f"{month_first_day.iloc[-1].date()})", flush=True)

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = csi["code"].tolist()

    factor_panel = unlock.build_factor_panel(
        unlock_df, codes, list(month_first_day),
        windows_days=(5, 20, 60),
    )

    wide = kline.pivot_table(
        index="date", columns="code", values="close", aggfunc="first",
    ).sort_index()
    fwd_rows = []
    for T in month_first_day:
        idx = wide.index.get_indexer([T])[0]
        if idx < 0 or idx + FORWARD_DAYS >= len(wide.index):
            continue
        T_close = wide.iloc[idx]
        T_plus = wide.iloc[idx + FORWARD_DAYS]
        for code in codes:
            c_now = T_close.get(code)
            c_fut = T_plus.get(code)
            if pd.isna(c_now) or pd.isna(c_fut) or c_now <= 0:
                continue
            fwd_rows.append({
                "asof_date": T, "code": code,
                "fwd_ret": c_fut / c_now - 1,
            })
    fwd = pd.DataFrame(fwd_rows)
    panel = factor_panel.merge(fwd, on=["asof_date", "code"], how="inner")
    print(f"[panel] {len(panel)} rows × "
          f"{panel['code'].nunique()} codes × "
          f"{panel['asof_date'].nunique()} months", flush=True)
    return panel


def monthly_ic(panel: pd.DataFrame, factor_col: str, sign: int = 1):
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    if df.empty:
        return pd.Series(dtype=float), {"factor": factor_col, "n_months": 0}
    df["signed_factor"] = sign * df[factor_col]

    def _corr(g):
        if len(g) < 10:
            return np.nan
        # 2 个 unique 值 (binary 因子) 允许 Spearman; 但纯常数 NaN
        if g["signed_factor"].nunique() < 2:
            return np.nan
        return g["signed_factor"].corr(g["fwd_ret"], method="spearman")

    monthly = df.groupby("asof_date").apply(_corr).dropna()
    if monthly.empty:
        return monthly, {"factor": factor_col, "sign": sign,
                         "n_months": 0, "ic_mean": np.nan, "ic_std": np.nan,
                         "icir": np.nan, "ic_pos_pct": np.nan}
    mean = float(monthly.mean())
    std = float(monthly.std())
    icir = mean / std * np.sqrt(12) if std > 0 else 0.0
    return monthly, {
        "factor": factor_col,
        "sign": sign,
        "n_months": int(len(monthly)),
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "ic_pos_pct": float((monthly > 0).mean() * 100),
    }


def main() -> int:
    if not KLINE_PATH.exists():
        print(f"FATAL: kline {KLINE_PATH} 缺", file=sys.stderr)
        return 1
    cache = unlock.load_cache()
    if cache.empty:
        print(f"FATAL: unlock cache 空, 先跑 fetch_unlock_csi300.py",
              file=sys.stderr)
        return 1
    # 严格 IS: 任何 unlock_date > IS_END+60d 的事件对 IS 因子不影响 (window 上界)
    # 但读 full cache 是 OK 的 — build_panel 内部 asof_dates 严格 ≤ IS_END.
    print(f"[load] unlock cache: {len(cache):,} rows × "
          f"{cache['code'].nunique():,} stocks")
    kline = load_kline_is()
    print(f"[load] kline IS shape={kline.shape}")
    panel = build_panel(kline, cache)
    if panel.empty:
        print("FATAL: panel 空", file=sys.stderr)
        return 1

    print("\n[sparsity] 因子非零率:")
    for f, _ in SINGLE_FACTORS:
        nz = (panel[f] > 0).mean() * 100
        print(f"  {f}: {nz:.1f}% nonzero")

    summaries: list[dict] = []
    monthly_series: dict[str, pd.Series] = {}

    for f, s in SINGLE_FACTORS:
        m, summary = monthly_ic(panel, f, sign=s)
        summaries.append(summary)
        monthly_series[f] = m

    def _zscore(g):
        if g.std(ddof=0) <= 0:
            return pd.Series(0.0, index=g.index)
        return (g - g.mean()) / g.std(ddof=0)

    panel = panel.copy()
    z20 = panel.groupby("asof_date")["unlock_pct_next_20"].transform(_zscore)
    z60 = panel.groupby("asof_date")["unlock_pct_next_60"].transform(_zscore)
    panel["combo_neg_pct"] = -(z20 + z60) / 2
    m, summary = monthly_ic(panel, "combo_neg_pct", sign=+1)
    summaries.append(summary)
    monthly_series["combo_neg_pct"] = m

    summary_df = pd.DataFrame(summaries).round(4)
    summary_df = summary_df.reindex(
        summary_df["icir"].abs().sort_values(
            ascending=False, na_position="last",
        ).index
    )
    summary_df.to_csv(OUT_CSV, index=False)
    print(f"\n[output] summary → {OUT_CSV}")
    print(summary_df.to_string(index=False))

    monthly_rows = []
    for f, ser in monthly_series.items():
        for m, v in ser.items():
            monthly_rows.append({"factor": f, "asof_date": m, "ic": v})
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly → {OUT_MONTHLY}")

    best = summary_df.iloc[0]
    best_icir_abs = abs(best["icir"]) if pd.notna(best["icir"]) else 0.0
    if best_icir_abs > 0.5:
        verdict = "STRONG — 推荐进 Phase B sidecar OOS"
    elif best_icir_abs > 0.3:
        verdict = "MARGINAL — 可选, 风险中等"
    else:
        verdict = "WEAK — abort, 不做 Phase B"

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    csi_with_events = len(set(csi["code"]) & set(cache["code"].unique()))

    with open(OUT_REPORT, "w") as out:
        out.write("# CSI300 Unlock 因子 IC 分析 (IS 2014-2020)\n\n")
        out.write(f"**Universe:** CSI300 (300 只)\n\n")
        out.write(f"**IS 期:** {IS_START.date()} → {IS_END.date()} (84 月)\n\n")
        out.write(f"**Forward return:** {FORWARD_DAYS} 日\n\n")
        out.write(f"**Unlock cache:** {len(cache):,} 事件 × "
                  f"{cache['code'].nunique():,} unique stocks; "
                  f"CSI300 中 {csi_with_events}/300 有 ≥1 事件 "
                  f"({csi_with_events/300*100:.1f}%)\n\n")
        out.write(f"**Panel:** {len(panel)} rows × "
                  f"{panel['code'].nunique()} codes × "
                  f"{panel['asof_date'].nunique()} months\n\n")
        out.write("## 因子非零率 (sparsity)\n\n")
        out.write("| factor | nonzero% |\n|---|---|\n")
        for f, _ in SINGLE_FACTORS:
            out.write(f"| {f} | {(panel[f]>0).mean()*100:.1f}% |\n")
        out.write("\n## IC 表 (按 |ICIR| 排序)\n\n")
        out.write(summary_df.to_markdown(index=False))
        out.write("\n\n## 判定规则\n\n")
        out.write("- |ICIR| > 0.5: 强, 进 Phase B sidecar OOS\n")
        out.write("- 0.3 < |ICIR| < 0.5: 边际\n")
        out.write("- |ICIR| < 0.3: 弱, abort\n\n")
        out.write(f"## 结论\n\n**Best factor:** `{best['factor']}` "
                  f"(ICIR = {best['icir']:.3f}, "
                  f"IC mean = {best['ic_mean']:.4f}, "
                  f"n_months = {best['n_months']})\n\n")
        out.write(f"**Verdict:** {verdict}\n")
    print(f"[output] report → {OUT_REPORT}")
    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"Best factor: {best['factor']} | ICIR={best['icir']:.3f} | "
          f"IC mean={best['ic_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
