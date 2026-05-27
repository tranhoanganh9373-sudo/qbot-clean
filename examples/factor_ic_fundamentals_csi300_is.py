"""CSI300 fundamentals IC 分析 - 严格 IS 2014-2020 only.

CLAUDE.md rule 5 (严格 OOS 协议):
- IS (因子选择期): 2014-01-01 ~ 2020-12-31 (84 月)
- OOS (留作 backtest): 2021-01-01 ~ 2026-04-30 → 本脚本绝不触碰

方法 (point-in-time, no lookahead):
- 每月第 1 个交易日 T:
  - 因子值 = fundamentals.get_quarterly_at_date(code, T, lag=60)
    (T 时点已披露的最新季报, 60 天 lag 保守)
  - 前向收益 = close(T+5) / close(T) - 1 (5 日 forward return)
- 横截面 Spearman corr(factor_z, forward_z) → monthly IC
- 月度 IC 序列 → mean / std / ICIR = mean / std * sqrt(12) / %>0

因子 (7 个 + 3 个 combo):
  正向 (高值看多):
    - roe                  ROE 加权 %
    - roa                  ROA (总资产净利率) %
    - net_margin           销售净利率
    - gross_margin         毛利率
    - revenue_yoy          营收同比
    - net_profit_yoy       净利润同比
  反向 (低值看多, IC 用 -factor):
    - debt_to_asset        资产负债率

注 (vs 用户原始 spec): 原 spec 含 pe_ttm/pb 但 stock_value_em 日频 PE/PB 仅
2018+, 不能覆盖 2014-2017 IS, 故本脚本用 roe_weighted/net_margin 替代,
全 IS 期可用.

输出:
  examples/factor_ic_fundamentals_csi300_is.csv          (per-factor summary)
  examples/factor_ic_fundamentals_csi300_is_monthly.csv  (long-format monthly)
  examples/factor_ic_fundamentals_csi300_is_report.md    (人读)

run:
  python examples/factor_ic_fundamentals_csi300_is.py
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

from claude_finance import fundamentals  # noqa: E402

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
OUT_CSV = ROOT / "examples" / "factor_ic_fundamentals_csi300_is.csv"
OUT_MONTHLY = ROOT / "examples" / "factor_ic_fundamentals_csi300_is_monthly.csv"
OUT_REPORT = ROOT / "examples" / "factor_ic_fundamentals_csi300_is_report.md"

# 严格 IS 边界, OOS 绝不触碰
IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 5
ANNOUNCE_LAG_DAYS = 60

POSITIVE_FACTORS = [
    "roe", "roa", "net_margin", "gross_margin",
    "revenue_yoy", "net_profit_yoy",
]
NEGATIVE_FACTORS = ["debt_to_asset"]
ALL_FACTORS = POSITIVE_FACTORS + NEGATIVE_FACTORS


def load_kline_is() -> pd.DataFrame:
    """读 baidu_kline.parquet, filter 到 IS 期间 + CSI300 codes.

    多留 10 天 buffer 给 T+5 forward 计算 (forward 5d 跨 IS_END 那几天).
    """
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = set(csi["code"].tolist())

    k = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    k = k[(k["date"] >= IS_START) & (k["date"] <= IS_END + pd.Timedelta(days=10))]
    return k.sort_values(["code", "date"]).reset_index(drop=True)


def load_fundamentals_for_csi300() -> dict[str, pd.DataFrame]:
    """读所有 CSI300 cache, key=code, value=df. 缺 cache 的略过."""
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    out: dict[str, pd.DataFrame] = {}
    missing = []
    for code in csi["code"]:
        d = fundamentals.load_cached(code)
        if d is None or d.empty:
            missing.append(code)
            continue
        out[code] = d
    if missing:
        print(
            f"[warn] {len(missing)} CSI300 股无 cache: {missing[:5]}..."
        )
    return out


def build_factor_returns_panel(
    kline: pd.DataFrame,
    fund_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """构造 monthly panel: (month_start, code, factor1..N, fwd_5d_ret).

    每月: 取该月第 1 个 IS-期 交易日 T, 对每股查 point-in-time fundamentals
    + forward 5d return.
    """
    kline_dates = pd.DatetimeIndex(sorted(kline["date"].unique()))
    is_dates = kline_dates[
        (kline_dates >= IS_START) & (kline_dates <= IS_END)
    ]
    months = pd.Series(is_dates).dt.to_period("M")
    month_first_day = (
        pd.Series(is_dates)
        .groupby(months)
        .first()
        .reset_index(drop=True)
    )

    print(
        f"[panel] {len(month_first_day)} 月度采样点 "
        f"({month_first_day.iloc[0].date()} → {month_first_day.iloc[-1].date()})",
        flush=True,
    )

    # wide pivot: date × code → close
    wide = kline.pivot_table(
        index="date", columns="code", values="close", aggfunc="first",
    ).sort_index()

    rows = []
    for T in month_first_day:
        idx = wide.index.get_indexer([T])[0]
        if idx < 0 or idx + FORWARD_DAYS >= len(wide.index):
            continue
        T_close = wide.iloc[idx]
        T_plus = wide.iloc[idx + FORWARD_DAYS]
        for code in wide.columns:
            c_now = T_close.get(code)
            c_fut = T_plus.get(code)
            if pd.isna(c_now) or pd.isna(c_fut) or c_now <= 0:
                continue
            fwd_ret = c_fut / c_now - 1
            fund_df = fund_map.get(code)
            if fund_df is None:
                continue
            q = fundamentals.get_quarterly_at_date(
                code, T, df=fund_df, announce_lag_days=ANNOUNCE_LAG_DAYS,
            )
            if q is None:
                continue
            row = {"month_start": T, "code": code, "fwd_ret": fwd_ret}
            for f in ALL_FACTORS:
                v = q.get(f)
                row[f] = v if v is not None and not pd.isna(v) else np.nan
            rows.append(row)

    panel = pd.DataFrame(rows)
    print(
        f"[panel] built {len(panel)} rows × "
        f"{panel['code'].nunique()} codes × "
        f"{panel['month_start'].nunique()} months",
        flush=True,
    )
    return panel


def monthly_ic(
    panel: pd.DataFrame,
    factor_col: str,
    sign: int = 1,
) -> tuple[pd.Series, dict]:
    """每月横截面 Spearman corr(factor*sign, fwd_ret).

    sign=+1 正向因子; sign=-1 反向因子 (用 -factor).
    """
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    if df.empty:
        return pd.Series(dtype=float), {"n_months": 0}
    df["signed_factor"] = sign * df[factor_col]

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < 10:
            return np.nan
        return g["signed_factor"].corr(g["fwd_ret"], method="spearman")

    monthly = df.groupby("month_start").apply(_corr).dropna()
    if len(monthly) == 0:
        return monthly, {"n_months": 0}
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


def build_combo_zscore(
    panel: pd.DataFrame, factors: list[str], signs: list[int],
) -> pd.Series:
    """每月对每因子做 rank z-score, 乘 sign 等权求和 → 组合."""
    df = panel.copy()
    z_parts = []
    for f, s in zip(factors, signs, strict=True):
        def _z(g: pd.Series) -> pd.Series:
            r = g.rank()
            mu = r.mean()
            sd = r.std(ddof=0)
            if sd <= 0:
                return pd.Series(np.zeros(len(g)), index=g.index)
            return (r - mu) / sd
        z = df.groupby("month_start")[f].transform(_z)
        z_parts.append(s * z)
    return sum(z_parts) / max(len(z_parts), 1)


def main() -> int:
    if not KLINE_PATH.exists():
        print(f"FATAL: kline cache 缺 {KLINE_PATH}", file=sys.stderr)
        return 1
    if not CSI300_PATH.exists():
        print(f"FATAL: csi300 list 缺 {CSI300_PATH}", file=sys.stderr)
        return 1

    print("[load] kline...")
    kline = load_kline_is()
    print(f"[load] kline shape={kline.shape}")

    print("[load] fundamentals for CSI300...")
    fund_map = load_fundamentals_for_csi300()
    print(f"[load] {len(fund_map)} stocks with fundamentals cache")

    if len(fund_map) < 50:
        print(
            f"FATAL: 仅 {len(fund_map)} 股有 fundamentals cache; "
            f"先跑 fetch_fundamentals_csi300.py",
            file=sys.stderr,
        )
        return 1

    panel = build_factor_returns_panel(kline, fund_map)
    if panel.empty:
        print("FATAL: panel 为空", file=sys.stderr)
        return 1

    summaries = []
    monthly_series: dict[str, pd.Series] = {}
    for f in POSITIVE_FACTORS:
        m, s = monthly_ic(panel, f, sign=+1)
        summaries.append(s)
        monthly_series[f] = m
    for f in NEGATIVE_FACTORS:
        m, s = monthly_ic(panel, f, sign=-1)
        summaries.append(s)
        monthly_series[f] = m

    # Combos
    panel = panel.copy()
    panel["combo_all"] = build_combo_zscore(
        panel,
        ALL_FACTORS,
        [1] * len(POSITIVE_FACTORS) + [-1] * len(NEGATIVE_FACTORS),
    )
    panel["combo_quality"] = build_combo_zscore(
        panel, ["roe", "net_margin", "gross_margin"], [1, 1, 1],
    )
    panel["combo_growth"] = build_combo_zscore(
        panel, ["revenue_yoy", "net_profit_yoy"], [1, 1],
    )

    for combo in ["combo_all", "combo_quality", "combo_growth"]:
        m, s = monthly_ic(panel, combo, sign=+1)
        summaries.append(s)
        monthly_series[combo] = m

    summary_df = pd.DataFrame(summaries).round(4)
    summary_df = summary_df.reindex(
        summary_df["icir"].abs().sort_values(ascending=False).index
    )
    summary_df.to_csv(OUT_CSV, index=False)
    print(f"\n[output] summary -> {OUT_CSV}")
    print(summary_df.to_string(index=False))

    monthly_rows = []
    for f, ser in monthly_series.items():
        for m, v in ser.items():
            monthly_rows.append({"factor": f, "month_start": m, "ic": v})
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly series -> {OUT_MONTHLY}")

    with open(OUT_REPORT, "w") as out:
        out.write("# CSI300 Fundamentals IC 分析 (IS 2014-2020)\n\n")
        out.write(f"**Universe:** CSI300 (300 股, {len(fund_map)} 有 cache)\n\n")
        out.write(f"**IS 期:** {IS_START.date()} → {IS_END.date()} (84 月)\n\n")
        out.write(f"**Forward return:** {FORWARD_DAYS} 日\n\n")
        out.write(f"**Announce lag:** {ANNOUNCE_LAG_DAYS} 天 (point-in-time)\n\n")
        out.write(
            f"**Panel:** {len(panel)} rows × "
            f"{panel['code'].nunique()} codes × "
            f"{panel['month_start'].nunique()} months\n\n"
        )
        out.write("## IC 表\n\n")
        out.write(summary_df.to_markdown(index=False))
        out.write("\n\n## 判定规则\n\n")
        out.write("- ICIR > 0.5: 因子有效, 建议纳入 combo\n")
        out.write("- 0.2 < ICIR < 0.5: 弱有效, 谨慎\n")
        out.write("- ICIR < 0.2: 无效, 弃用\n")
    print(f"[output] report -> {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
