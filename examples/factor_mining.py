"""自定义因子挖掘 — 算 daily IC vs 未来 5 日收益.

支持因子:
  margin    融资余额 5/20 日变化率 (东财 datacenter, sandbox 可用)
  dragon    龙虎榜机构净买 (东财 datacenter, 数据稀疏)
  iwencai   NL 选股 indicator (需 IWENCAI_KEY env, sandbox 多半不可用)

输入:
  data_cache/csi300_constituents.csv (code, name)
  data_cache/baidu_kline.parquet (code, date, close → forward returns)

输出:
  examples/factor_<name>_ic.csv  daily IC series
  examples/factor_<name>_report.md  IC mean / ICIR / quantile spread

run:
  python examples/factor_mining.py --factor margin --n-stocks 50 --days 60
  python examples/factor_mining.py --factor dragon --n-stocks 100 --days 60
"""
from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
OUT_DIR = Path(__file__).resolve().parent

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def eastmoney_datacenter(report_name: str, filter_str: str = "",
                          page_size: int = 50, sort_columns: str = "",
                          sort_types: str = "-1", retries: int = 3) -> list[dict]:
    params = {
        "reportName": report_name, "columns": "ALL",
        "filter": filter_str, "pageNumber": "1", "pageSize": str(page_size),
        "sortColumns": sort_columns, "sortTypes": sort_types,
        "source": "WEB", "client": "WEB",
    }
    for attempt in range(retries):
        try:
            r = requests.get(DATACENTER_URL, params=params,
                              headers={"User-Agent": UA}, timeout=15)
            d = r.json()
            if d.get("result") and d["result"].get("data"):
                return d["result"]["data"]
            return []
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(0.5)
    return []


def margin_trading(code: str, page_size: int = 500, max_pages: int = 15) -> pd.DataFrame:
    """融资融券明细 - pagination 拉满历史 (endpoint 实际 cap ~500/页).

    14 年历史 = ~3500 天 = 7 页 (page_size=500). max_pages=15 留 buffer.
    """
    all_data = []
    for pn in range(1, max_pages + 1):
        params = {
            "reportName": "RPTA_WEB_RZRQ_GGMX",
            "columns": "ALL",
            "filter": f'(SCODE="{code}")',
            "pageNumber": str(pn), "pageSize": str(page_size),
            "sortColumns": "DATE", "sortTypes": "-1",
            "source": "WEB", "client": "WEB",
        }
        rows = None
        for _ in range(3):
            try:
                r = requests.get(DATACENTER_URL, params=params,
                                  headers={"User-Agent": UA}, timeout=15)
                rows = (r.json().get("result") or {}).get("data") or []
                break
            except Exception:
                time.sleep(0.5)
        if rows is None or not rows:
            break
        all_data.extend(rows)
        if len(rows) < page_size:
            break
        time.sleep(0.05)
    if not all_data:
        return pd.DataFrame()
    out_rows = []
    for r in all_data:
        out_rows.append({
            "code": code,
            "date": pd.to_datetime(str(r.get("DATE", ""))[:10]),
            "rzye": float(r.get("RZYE") or 0),
            "rzmre": float(r.get("RZMRE") or 0),
            "rzche": float(r.get("RZCHE") or 0),
        })
    return pd.DataFrame(out_rows).drop_duplicates(subset=["code", "date"])


def dragon_tiger_records(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = eastmoney_datacenter(
        "RPT_DAILYBILLBOARD_DETAILSNEW",
        filter_str=f'(TRADE_DATE>=\'{start_date}\')(TRADE_DATE<=\'{end_date}\')(SECURITY_CODE="{code}")',
        page_size=200,
        sort_columns="TRADE_DATE", sort_types="-1",
    )
    if not data:
        return pd.DataFrame()
    rows = []
    for r in data:
        rows.append({
            "code": code,
            "date": pd.to_datetime(str(r.get("TRADE_DATE", ""))[:10]),
            "net_buy": float(r.get("BILLBOARD_NET_AMT") or 0),
            "turnover": float(r.get("TURNOVERRATE") or 0),
        })
    return pd.DataFrame(rows)


def load_csi300_codes(n: int) -> list[str]:
    df = pd.read_csv(CSI300_PATH, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df["code"].head(n).tolist()


def load_forward_returns(codes: list[str], end_date: str,
                          n_days_back: int, fwd_horizon: int = 5) -> pd.DataFrame:
    kl = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
    kl["code"] = kl["code"].astype(str).str.zfill(6)
    kl["date"] = pd.to_datetime(kl["date"])
    kl = kl[kl["code"].isin(codes)]
    end_ts = pd.to_datetime(end_date)
    start_ts = end_ts - pd.Timedelta(days=n_days_back + fwd_horizon + 5)
    kl = kl[(kl["date"] >= start_ts) &
            (kl["date"] <= end_ts + pd.Timedelta(days=fwd_horizon + 3))]
    kl = kl.sort_values(["code", "date"]).reset_index(drop=True)
    kl["fwd_close"] = kl.groupby("code")["close"].shift(-fwd_horizon)
    kl["fwd_ret"] = kl["fwd_close"] / kl["close"] - 1
    return kl[["code", "date", "close", "fwd_ret"]].dropna()


def factor_margin(codes: list[str], days: int = 0) -> pd.DataFrame:
    """days 参数已弃用 (现在 margin_trading 自动 pagination 取全 14 年历史)."""
    all_rows = []
    failed = 0
    for i, c in enumerate(codes):
        df = margin_trading(c)  # 不再传 page_size, 用 margin_trading 内部 pagination
        if df.empty:
            failed += 1
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df["margin_5d_chg"] = df["rzye"].pct_change(5)
        df["margin_20d_chg"] = df["rzye"].pct_change(20)
        # 保留 rzye/rzmre/rzche 给下游 (容许重算其他周期 / 衍生因子)
        all_rows.append(df[["code", "date", "rzye", "rzmre", "rzche", "margin_5d_chg", "margin_20d_chg"]])
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(codes)}] ok (累计失败 {failed})", flush=True)
        time.sleep(0.05)
    print(f"  完成: {len(codes)} 只, 成功 {len(codes)-failed}, 失败 {failed}")
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def factor_dragon(codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    all_rows = []
    hits = 0
    for i, c in enumerate(codes):
        df = dragon_tiger_records(c, start_date, end_date)
        if not df.empty:
            all_rows.append(df)
            hits += 1
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(codes)}] 累计上榜股 {hits}", flush=True)
        time.sleep(0.05)
    print(f"  完成: {len(codes)} 只查询, {hits} 只有上榜记录")
    if not all_rows:
        return pd.DataFrame()
    out = pd.concat(all_rows, ignore_index=True)
    out["lhb_signal"] = np.sign(out["net_buy"]) * np.log1p(out["net_buy"].abs() / 1e6)
    return out[["code", "date", "lhb_signal", "net_buy"]]


def monthly_ic_breakdown(daily_ic: pd.Series) -> pd.DataFrame:
    """按月聚合 daily IC: monthly mean / std / ICIR / pos_pct."""
    df = daily_ic.to_frame("ic")
    df["month"] = df.index.to_period("M").astype(str)
    grp = df.groupby("month")["ic"]
    out = pd.DataFrame({
        "n_days": grp.size(),
        "ic_mean": grp.mean(),
        "ic_std": grp.std(),
        "ic_pos_pct": grp.apply(lambda x: (x > 0).mean() * 100),
    }).round(4)
    out["icir_m"] = (out["ic_mean"] / out["ic_std"] * np.sqrt(21)).round(3)
    return out.reset_index()


def compute_ic(factor_df: pd.DataFrame, returns_df: pd.DataFrame,
                factor_col: str) -> tuple[pd.Series, dict]:
    df = factor_df.merge(returns_df[["code", "date", "fwd_ret"]],
                          on=["code", "date"], how="inner")
    df = df.dropna(subset=[factor_col, "fwd_ret"])
    if df.empty:
        return pd.Series(dtype=float), {}
    daily_ic = df.groupby("date").apply(
        lambda g: g[factor_col].corr(g["fwd_ret"], method="spearman") if len(g) > 5 else np.nan
    ).dropna()
    if len(daily_ic) == 0:
        return daily_ic, {"n_days": 0}
    summary = {
        "n_days": int(len(daily_ic)),
        "n_pairs": int(len(df)),
        "ic_mean": float(daily_ic.mean()),
        "ic_std": float(daily_ic.std()),
        "icir": float(daily_ic.mean() / daily_ic.std() * np.sqrt(252))
                if daily_ic.std() > 0 else 0,
        "ic_pos_pct": float((daily_ic > 0).mean() * 100),
    }
    df["q"] = df.groupby("date")[factor_col].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop") + 1
        if len(x) >= 5 else np.nan)
    q_ret = df.groupby("q")["fwd_ret"].mean() * 10000
    if len(q_ret) == 5:
        summary["q5_minus_q1_bps"] = float(q_ret[5] - q_ret[1])
        summary["q5_bps"] = float(q_ret[5])
        summary["q1_bps"] = float(q_ret[1])
    return daily_ic, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", required=True, choices=["margin", "dragon"])
    parser.add_argument("--n-stocks", type=int, default=50)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--wf60m", action="store_true",
                         help="拉全历史 page_size=2000, filter 2021-05→2026-04, 加 monthly IC breakdown")
    args = parser.parse_args()

    # wf60m: fetch 全历史 (page_size 2000) + filter 到 2021-05 → 2026-04 (60 月)
    if args.wf60m:
        fetch_days = 2000
        lookback_days = 1900
        wf_start = pd.Timestamp("2021-05-01")
        wf_end = pd.Timestamp("2026-04-30")
        tag_suffix = "_wf60m"
    else:
        fetch_days = args.days
        lookback_days = args.days
        wf_start, wf_end = None, None
        tag_suffix = ""

    print(f"[1/3] 加载 CSI300 前 {args.n_stocks} 只 + baidu_kline forward returns "
          f"({'wf60m' if args.wf60m else f'{args.days}d'})")
    codes = load_csi300_codes(args.n_stocks)
    end_date = datetime.now().strftime("%Y-%m-%d")
    returns_df = load_forward_returns(codes, end_date, lookback_days, fwd_horizon=5)
    print(f"  forward returns: {len(returns_df):,} rows  "
          f"covering {returns_df['date'].min().date()} → {returns_df['date'].max().date()}")

    print(f"\n[2/3] 拉 {args.factor} 数据 (page_size={fetch_days}) ...")
    if args.factor == "margin":
        factor_df = factor_margin(codes, days=fetch_days)
        factor_col = "margin_5d_chg"
    else:
        start_ds = (datetime.now() - timedelta(days=fetch_days)).strftime("%Y-%m-%d")
        factor_df = factor_dragon(codes, start_ds, end_date)
        factor_col = "lhb_signal"

    if args.wf60m and not factor_df.empty:
        # filter 到 60 月窗口
        factor_df = factor_df[(factor_df["date"] >= wf_start) & (factor_df["date"] <= wf_end)]
        returns_df = returns_df[(returns_df["date"] >= wf_start) & (returns_df["date"] <= wf_end)]

    if factor_df.empty:
        print(f"\n❌ 因子数据为空 — sandbox 网络 / endpoint 失效")
        return
    print(f"  factor rows: {len(factor_df):,}  "
          f"{factor_df['date'].min().date()} → {factor_df['date'].max().date()}")

    print(f"\n[3/3] 算 daily IC ...")
    daily_ic, summary = compute_ic(factor_df, returns_df, factor_col)
    if not summary or summary.get("n_days", 0) == 0:
        print("❌ IC 无法计算 — 每日横截面 < 5 stock")
        return

    print(f"\n=== Summary: factor={args.factor} ({factor_col}) ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<20} {v:+.4f}")
        else:
            print(f"  {k:<20} {v}")

    ic_path = OUT_DIR / f"factor_{args.factor}{tag_suffix}_ic.csv"
    daily_ic.to_csv(ic_path, header=["ic"])
    print(f"\n输出: {ic_path.name}")

    # 月度 IC 拆分 (--wf60m 必跑, 否则只在 n_days >= 60 时跑)
    monthly_df = None
    if args.wf60m or summary.get("n_days", 0) >= 60:
        monthly_df = monthly_ic_breakdown(daily_ic)
        m_path = OUT_DIR / f"factor_{args.factor}{tag_suffix}_monthly.csv"
        monthly_df.to_csv(m_path, index=False)
        print(f"输出: {m_path.name}")
        # monthly stability stats
        m_ic = monthly_df["ic_mean"]
        summary["monthly_ic_mean"] = float(m_ic.mean())
        summary["monthly_ic_std"] = float(m_ic.std())
        summary["%months_ic>0"] = float((m_ic > 0).mean() * 100)
        summary["max_monthly_ic"] = float(m_ic.max())
        summary["min_monthly_ic"] = float(m_ic.min())
        # 子段 stability
        print("\n=== Monthly IC stability ===")
        print(f"  cross-month ic_mean: {m_ic.mean():+.4f}  std: {m_ic.std():.4f}")
        print(f"  % months IC > 0:    {(m_ic > 0).mean() * 100:.1f}%")
        print(f"  best month:  {monthly_df.loc[m_ic.idxmax(), 'month']} ic={m_ic.max():+.4f}")
        print(f"  worst month: {monthly_df.loc[m_ic.idxmin(), 'month']} ic={m_ic.min():+.4f}")

    md = [
        f"# Factor mining: {args.factor} ({factor_col}){' — wf60m' if args.wf60m else ''}",
        "",
        f"**Universe**: CSI300 前 {args.n_stocks} 只",
        f"**Period**: {factor_df['date'].min().date()} → {factor_df['date'].max().date()}",
        f"**Forward horizon**: 5 日",
        "",
        "## IC Summary",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ]
    for k, v in summary.items():
        line = f"| {k} | "
        line += f"{v:+.4f}" if isinstance(v, float) else f"{v}"
        line += " |"
        md.append(line)
    if monthly_df is not None:
        md += ["", "## Monthly IC (前 12 / 后 12 月)", ""]
        md.append(monthly_df.head(12).to_markdown(index=False))
        md += ["", "...", ""]
        md.append(monthly_df.tail(12).to_markdown(index=False))
    md += [
        "",
        "## 解读",
        "- ICIR > 1.0 强, > 0.5 可用, < 0.3 噪声",
        "- Q5 - Q1 bps spread 正向且大 → 因子能区分赢家输家",
        "- IC > 0 占比 > 60% → 因子稳定不靠极端日",
        "- **% months IC > 0 < 50%** → 因子在多数月份反向, 整体方向取决于尾部月份",
    ]
    md_path = OUT_DIR / f"factor_{args.factor}{tag_suffix}_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"输出: {md_path.name}")


if __name__ == "__main__":
    main()
