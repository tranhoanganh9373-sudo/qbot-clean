"""CSI300 技术/事件因子 IC 分析 - 严格 IS 2014-2020 only.

CLAUDE.md rule 5 (严格 OOS 协议):
- IS (因子选择期): 2014-01-01 ~ 2020-12-31 (84 月)
- OOS (留作 backtest): 2021-01-01 ~ 2026-04-30 → 本脚本绝不触碰

方法 (point-in-time, no lookahead):
- 月度采样: 每月第 1 个交易日 T
- 因子值: T 时点的滚动事件特征 (用 T 之前的数据, 不含 T+future)
- 前向收益: close(T+5) / close(T) - 1
- 横截面 Spearman corr(factor, fwd_5d_ret) per month → 月度 IC 序列
- ICIR = ic.mean() / ic.std() * sqrt(12)

数据源 (sandbox-OK):
- 龙虎榜: data_cache/dragon_tiger/{code}.parquet
  (fetch_dragon_tiger_csi300_is.py 抓的 IS 期)
- 融资融券: data_cache/csi300_margin_14yr.parquet (Phase 2 已 cache)

未实现 (sandbox 不可达, 见模块 docstring):
- EPS 一致预期: src/claude_finance/eps_consensus.py (端点 snapshot-only)
- 日级资金流: src/claude_finance/fund_flow.py (push2his 不通)

候选因子 (12 个):

  C 轨 龙虎榜 (event-driven, 7 个):
    - on_list_today                当日是否上榜 (binary)
    - net_buy_pct_evt              当日净买入占成交额 % (非上榜日 0)
    - top_list_count_30d           滚动 30 日上榜次数
    - top_list_count_60d           滚动 60 日上榜次数
    - net_buy_sum_30d_wan          滚动 30 日累计净买额 (万元)
    - net_buy_sum_60d_wan          滚动 60 日累计净买额 (万元)
    - net_buy_pct_evt_30d_avg      30 日平均净买占比 (含 0)

  B' 轨 融资融券 (substitute for fund flow, 5 个):
    - margin_5d_chg                融资余额 5 日变化率
    - margin_20d_chg               融资余额 20 日变化率
    - rzye_log                     融资余额 log
    - rzmre_5d_avg_log             融资买入 5 日均值 log
    - margin_5d_minus_20d          5d_chg - 20d_chg (加速度)

  A 轨 EPS 一致预期: 不可行, 输出 feasibility note

输出:
  examples/factor_ic_technical_csi300_is.csv          summary
  examples/factor_ic_technical_csi300_is_monthly.csv  long-format monthly
  examples/factor_ic_technical_csi300_is_report.md    人读
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

from claude_finance import dragon_tiger  # noqa: E402
from claude_finance import eps_consensus  # noqa: E402
from claude_finance import fund_flow  # noqa: E402

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
MARGIN_PATH = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
DT_CACHE_DIR = ROOT / "data_cache" / "dragon_tiger"
OUT_CSV = ROOT / "examples" / "factor_ic_technical_csi300_is.csv"
OUT_MONTHLY = ROOT / "examples" / "factor_ic_technical_csi300_is_monthly.csv"
OUT_REPORT = ROOT / "examples" / "factor_ic_technical_csi300_is_report.md"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 5
MIN_MONTHLY_OBS = 30

TIGER_FACTORS_SIGN = {
    "on_list_today": +1,
    "net_buy_pct_evt": +1,
    "top_list_count_30d": +1,
    "top_list_count_60d": +1,
    "net_buy_sum_30d_wan": +1,
    "net_buy_sum_60d_wan": +1,
    "net_buy_pct_evt_30d_avg": +1,
}

MARGIN_FACTORS_SIGN = {
    "margin_5d_chg": +1,
    "margin_20d_chg": +1,
    "rzye_log": +1,
    "rzmre_5d_avg_log": +1,
    "margin_5d_minus_20d": +1,
}


def load_kline_is() -> pd.DataFrame:
    """读 baidu_kline.parquet, filter IS + CSI300, +10d buffer for T+5."""
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = set(csi["code"].tolist())
    k = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
    k["code"] = k["code"].astype(str).str.zfill(6)
    k = k[k["code"].isin(codes)]
    k = k[
        (k["date"] >= IS_START)
        & (k["date"] <= IS_END + pd.Timedelta(days=10))
    ]
    return k.sort_values(["code", "date"]).reset_index(drop=True)


def load_dragon_tiger_cache() -> pd.DataFrame:
    """合并 CSI300 所有 dragon_tiger/{code}.parquet → 单 df."""
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    parts: list[pd.DataFrame] = []
    n_with_events = 0
    n_no_cache = 0
    n_empty = 0
    for code in csi["code"]:
        p = DT_CACHE_DIR / f"{code}.parquet"
        if not p.exists():
            n_no_cache += 1
            continue
        df = pd.read_parquet(p)
        if df.empty:
            n_empty += 1
            continue
        parts.append(df)
        n_with_events += 1
    print(
        f"[dt-load] {n_with_events} 股有 events, "
        f"{n_empty} 空 (IS 无龙虎榜), {n_no_cache} 缺 cache",
        flush=True,
    )
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out[(out["date"] >= IS_START) & (out["date"] <= IS_END)]
    return out.reset_index(drop=True)


def load_margin_is() -> pd.DataFrame | None:
    """读 csi300_margin_14yr.parquet, filter IS + CSI300."""
    if not MARGIN_PATH.exists():
        return None
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    m = pd.read_parquet(MARGIN_PATH)
    m["code"] = m["code"].astype(str).str.zfill(6)
    m = m[m["code"].isin(csi["code"])]
    m = m[(m["date"] >= IS_START) & (m["date"] <= IS_END)]
    return m.sort_values(["code", "date"]).reset_index(drop=True)


def build_panel(
    kline: pd.DataFrame,
    tiger_events: pd.DataFrame,
    margin_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """构造 monthly panel: (T, code, factor1..N, fwd_5d_ret).

    T = 每月第 1 个 IS 期交易日.
    Factor values are at T (using info available <= T, no lookahead).
    """
    kline_dates = pd.DatetimeIndex(sorted(kline["date"].unique()))
    is_dates = kline_dates[
        (kline_dates >= IS_START) & (kline_dates <= IS_END)
    ]
    months = pd.Series(is_dates).dt.to_period("M")
    month_first = (
        pd.Series(is_dates)
        .groupby(months)
        .first()
        .reset_index(drop=True)
    )
    print(
        f"[panel] {len(month_first)} 月度采样点 "
        f"{month_first.iloc[0].date()} → {month_first.iloc[-1].date()}",
        flush=True,
    )

    # tiger panel (dense at trading-day frequency)
    tiger_daily = dragon_tiger.daily_features(tiger_events)
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    all_codes = csi["code"].tolist()

    full_idx = pd.DatetimeIndex(sorted(kline["date"].unique()))
    full_idx_is = full_idx[
        (full_idx >= IS_START) & (full_idx <= IS_END)
    ]
    tiger_panel = dragon_tiger.rolling_event_features(
        tiger_daily, all_codes, full_idx_is
    )
    tiger_panel = tiger_panel.sort_values(["code", "date"])
    grp = tiger_panel.groupby("code", sort=False)
    tiger_panel["net_buy_pct_evt_30d_avg"] = grp[
        "net_buy_pct_evt"
    ].transform(lambda x: x.rolling(30, min_periods=1).mean())

    # margin panel (daily already)
    margin_panel = None
    if margin_df is not None and not margin_df.empty:
        mp = margin_df.copy()
        mp["rzye_log"] = np.log(mp["rzye"].clip(lower=1.0))
        mp["rzmre_5d_avg"] = (
            mp.groupby("code")["rzmre"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        mp["rzmre_5d_avg_log"] = np.log(
            mp["rzmre_5d_avg"].clip(lower=1.0)
        )
        mp["margin_5d_minus_20d"] = (
            mp["margin_5d_chg"] - mp["margin_20d_chg"]
        )
        margin_panel = mp[
            [
                "code",
                "date",
                "margin_5d_chg",
                "margin_20d_chg",
                "rzye_log",
                "rzmre_5d_avg_log",
                "margin_5d_minus_20d",
            ]
        ]

    wide = (
        kline.pivot_table(
            index="date",
            columns="code",
            values="close",
            aggfunc="first",
        )
        .sort_index()
    )

    rows = []
    for T in month_first:
        idx = wide.index.get_indexer([T])[0]
        if idx < 0 or idx + FORWARD_DAYS >= len(wide.index):
            continue
        T_close = wide.iloc[idx]
        T_plus = wide.iloc[idx + FORWARD_DAYS]
        tiger_at_T = (
            tiger_panel[tiger_panel["date"] == T]
            .set_index("code")
        )
        margin_at_T = None
        if margin_panel is not None:
            recent = margin_panel[
                (margin_panel["date"] <= T)
                & (margin_panel["date"] >= T - pd.Timedelta(days=10))
            ]
            if not recent.empty:
                margin_at_T = (
                    recent.sort_values("date")
                    .groupby("code", as_index=True)
                    .tail(1)
                    .set_index("code")
                )

        for code in wide.columns:
            c_now = T_close.get(code)
            c_fut = T_plus.get(code)
            if pd.isna(c_now) or pd.isna(c_fut) or c_now <= 0:
                continue
            fwd_ret = c_fut / c_now - 1
            row = {"month_start": T, "code": code, "fwd_ret": fwd_ret}
            if code in tiger_at_T.index:
                tr = tiger_at_T.loc[code]
                for f in TIGER_FACTORS_SIGN:
                    row[f] = (
                        float(tr[f])
                        if f in tr and not pd.isna(tr[f])
                        else np.nan
                    )
            else:
                for f in TIGER_FACTORS_SIGN:
                    row[f] = np.nan
            if margin_at_T is not None and code in margin_at_T.index:
                mr = margin_at_T.loc[code]
                for f in MARGIN_FACTORS_SIGN:
                    row[f] = (
                        float(mr[f])
                        if f in mr and not pd.isna(mr[f])
                        else np.nan
                    )
            else:
                for f in MARGIN_FACTORS_SIGN:
                    row[f] = np.nan
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
    panel: pd.DataFrame, factor_col: str, sign: int = 1
) -> tuple[pd.Series, dict]:
    """每月横截面 Spearman corr(factor*sign, fwd_ret)."""
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    if df.empty:
        return pd.Series(dtype=float), {
            "factor": factor_col,
            "sign": sign,
            "n_months": 0,
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "icir": 0.0,
            "ic_pos_pct": 0.0,
            "avg_obs_per_month": 0.0,
        }
    df["signed_factor"] = sign * df[factor_col]

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < MIN_MONTHLY_OBS:
            return np.nan
        if g["signed_factor"].std() <= 0:
            return np.nan
        return g["signed_factor"].corr(
            g["fwd_ret"], method="spearman"
        )

    obs = df.groupby("month_start").size()
    monthly = df.groupby("month_start").apply(_corr).dropna()
    if len(monthly) == 0:
        return monthly, {
            "factor": factor_col,
            "sign": sign,
            "n_months": 0,
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "icir": 0.0,
            "ic_pos_pct": 0.0,
            "avg_obs_per_month": float(obs.mean()) if len(obs) else 0.0,
        }
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
        "avg_obs_per_month": float(obs.mean()),
    }


def main() -> int:
    if not KLINE_PATH.exists():
        print(f"FATAL: 缺 {KLINE_PATH}", file=sys.stderr)
        return 1
    if not CSI300_PATH.exists():
        print(f"FATAL: 缺 {CSI300_PATH}", file=sys.stderr)
        return 1

    print("\n=== Track A (EPS consensus) ===")
    print(eps_consensus.is_ic_feasibility_note())
    print("\n=== Track B (fund flow daily) ===")
    print(fund_flow.is_ic_feasibility_note())
    print()

    print("[load] kline...")
    kline = load_kline_is()
    print(f"[load] kline shape={kline.shape}")

    print("[load] dragon-tiger events...")
    tiger_events = load_dragon_tiger_cache()
    print(f"[load] tiger events: {len(tiger_events)} rows")

    print("[load] margin (substitute for fund flow)...")
    margin_df = load_margin_is()
    if margin_df is not None:
        print(
            f"[load] margin: {len(margin_df)} rows × "
            f"{margin_df['code'].nunique()} codes"
        )
    else:
        print("[load] margin: cache 不存在")

    panel = build_panel(kline, tiger_events, margin_df)
    if panel.empty:
        print("FATAL: panel 为空", file=sys.stderr)
        return 1

    cov = {}
    for f in {**TIGER_FACTORS_SIGN, **MARGIN_FACTORS_SIGN}:
        cov[f] = int(panel[f].notna().sum())
    print("\n[panel] factor coverage (non-NaN rows):")
    for f, n in sorted(cov.items(), key=lambda kv: -kv[1]):
        print(f"  {f:30s} {n:6d} / {len(panel)}")

    summaries: list[dict] = []
    monthly_series: dict[str, pd.Series] = {}
    for f, sign in {**TIGER_FACTORS_SIGN, **MARGIN_FACTORS_SIGN}.items():
        m, s = monthly_ic(panel, f, sign=sign)
        summaries.append(s)
        monthly_series[f] = m

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
            monthly_rows.append(
                {"factor": f, "month_start": m, "ic": v}
            )
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly series -> {OUT_MONTHLY}")

    with open(OUT_REPORT, "w") as out:
        out.write(
            "# CSI300 Technical/Event Factors IC 分析 "
            "(IS 2014-2020)\n\n"
        )
        out.write(
            "**Universe:** CSI300 (300 股, current constituents — "
            "survivorship bias 注明)\n\n"
        )
        out.write(
            f"**IS 期:** {IS_START.date()} → {IS_END.date()} "
            f"({panel['month_start'].nunique()} 月)\n\n"
        )
        out.write(f"**Forward return:** {FORWARD_DAYS} 日\n\n")
        out.write(
            f"**Panel:** {len(panel)} rows × "
            f"{panel['code'].nunique()} codes × "
            f"{panel['month_start'].nunique()} months\n\n"
        )
        out.write("## Track A: EPS 一致预期\n\n")
        out.write("**status:** NOT FEASIBLE in sandbox.\n\n")
        out.write(f"> {eps_consensus.is_ic_feasibility_note()}\n\n")
        out.write("## Track B: 资金流 (daily)\n\n")
        out.write("**status:** NOT FEASIBLE in sandbox.\n\n")
        out.write(f"> {fund_flow.is_ic_feasibility_note()}\n\n")
        out.write(
            "**substitute:** 融资融券 (margin trading) — "
            "已 cache 在 `data_cache/csi300_margin_14yr.parquet`"
        )
        out.write(" (99 股 IS 期).\n\n")
        out.write("## Track C: 龙虎榜\n\n")
        out.write(
            "**endpoint:** `datacenter-web.eastmoney.com/api/data/v1/get"
            "?reportName=RPT_DAILYBILLBOARD_DETAILSNEW`\n\n"
        )
        out.write(
            f"**cache:** `data_cache/dragon_tiger/*.parquet` "
            f"({len(tiger_events)} rows total)\n\n"
        )
        out.write("## IC 表 (按 |ICIR| 降序)\n\n")
        out.write(summary_df.to_markdown(index=False))
        out.write("\n\n## 判定规则\n\n")
        out.write("- ICIR > 0.5: 因子有效, 建议纳入 Phase 4 sidecar\n")
        out.write("- 0.3 < ICIR < 0.5: 弱有效, 谨慎\n")
        out.write("- ICIR < 0.3: 无效, 弃用\n\n")
        out.write("## 推荐 Phase 4 sidecar 候选\n\n")
        strong = summary_df[summary_df["icir"].abs() > 0.5]
        mid = summary_df[
            (summary_df["icir"].abs() > 0.3)
            & (summary_df["icir"].abs() <= 0.5)
        ]
        out.write(f"**strong (|ICIR|>0.5):** {len(strong)} 个\n\n")
        if not strong.empty:
            out.write(
                strong[
                    ["factor", "ic_mean", "icir", "n_months"]
                ].to_markdown(index=False)
            )
            out.write("\n\n")
        out.write(f"**mid (0.3<|ICIR|<=0.5):** {len(mid)} 个\n\n")
        if not mid.empty:
            out.write(
                mid[
                    ["factor", "ic_mean", "icir", "n_months"]
                ].to_markdown(index=False)
            )
            out.write("\n\n")
        if strong.empty and mid.empty:
            out.write(
                "\n**所有 12 因子 |ICIR| < 0.3, 技术/事件因子方向"
                "在 IS 上未显著. 与 Phase 3 fundamentals 失败一致, "
                "production 接受 Calmar 1.96 现状.**\n"
            )
    print(f"[output] report -> {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
