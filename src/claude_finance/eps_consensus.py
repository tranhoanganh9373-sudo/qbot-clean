"""EPS 一致预期因子探索 — sandbox 可达性记录 + 端点封装.

# 结论 (2026-05-25 探查)

**Track A (EPS 一致预期 history 2014-2020) 不可行**，原因记录：

1. **同花顺 basic.10jqka.com.cn `/new/{code}/worth.html`**:
   - sandbox ✅ 可达 (200 OK, 3.8s)
   - 但返回的是 **当前快照** 一致预期 (2026/2027/2028 三年预测均值)
   - 没有历史 snapshot — 无法构造 eps_consensus_change_1m / change_3m 时间序列
   - 不能用于 2014-2020 IS 期 IC 分析

2. **东财 reportapi.eastmoney.com `/report/list`**:
   - sandbox ✅ 可达
   - 每篇 research report 含 `predictThisYearEps` 字段, 理论上可构造预期 history
   - **但实测 2024 之前所有研报 EPS 字段全部空字符串** (probed 2014/2015/2018/2020/2022 都是 with_eps=0)
   - 2024 年 6 月起字段开始有值, 但 IS 期 (2014-2020) 完全空白
   - 不可用

3. **akshare ak.stock_zh_a_concept_consensus / 类似**:
   - 同样底层走 push2his / 同花顺接口, 同样无 IS 历史

**结论:** sandbox 内无法获取 2014-2020 EPS 一致预期 history. 需付费 wind/choice 终端
或第三方数据 (uqer/jq/tushare pro).

# 备选: 当前快照 helper

仍保留 `fetch_current_consensus(code)` 用于实盘/scan, 抓"今日"的一致预期表。
仅供 production v19.1 之外的实时决策, **不用于 IS IC 分析**.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko)"
)

CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "data_cache" / "eps_consensus"
)


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(code: str) -> Path:
    """单股 cache 路径 data_cache/eps_consensus/{code}.parquet."""
    code_str = str(code).zfill(6)
    return CACHE_DIR / f"{code_str}.parquet"


def fetch_current_consensus(code: str, timeout: int = 15) -> pd.DataFrame:
    """抓同花顺 worth.html → DataFrame: 年度/预测机构数/最小/均值/最大.

    返回 columns: ['code', 'fetched_date', 'year', 'analyst_count',
                   'eps_min', 'eps_mean', 'eps_max']

    若无机构覆盖 (小盘/ST), 返回空 DataFrame.
    若 HTTP 失败, raise requests.RequestException.
    """
    code_str = str(code).zfill(6)
    url = f"https://basic.10jqka.com.cn/new/{code_str}/worth.html"
    headers = {
        "User-Agent": UA,
        "Referer": "https://basic.10jqka.com.cn/",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "gbk"

    try:
        dfs = pd.read_html(resp.text)
    except ValueError:
        return pd.DataFrame()

    # 找含"年度/均值"的 EPS 预测表 (通常 table 0 是 EPS)
    target = None
    for df in dfs:
        cols = [str(c) for c in df.columns]
        if "年度" in cols and "均值" in cols and "预测机构数" in cols:
            target = df
            break
    if target is None or target.empty:
        return pd.DataFrame()

    today = pd.Timestamp.today().normalize()
    out_rows = []
    for _, row in target.iterrows():
        try:
            year = int(row["年度"])
            count = (
                int(row["预测机构数"])
                if pd.notna(row["预测机构数"])
                else 0
            )
            eps_min = (
                float(row["最小值"]) if pd.notna(row["最小值"]) else None
            )
            eps_mean = (
                float(row["均值"]) if pd.notna(row["均值"]) else None
            )
            eps_max = (
                float(row["最大值"]) if pd.notna(row["最大值"]) else None
            )
        except (ValueError, TypeError, KeyError):
            continue
        out_rows.append(
            {
                "code": code_str,
                "fetched_date": today,
                "year": year,
                "analyst_count": count,
                "eps_min": eps_min,
                "eps_mean": eps_mean,
                "eps_max": eps_max,
            }
        )
    return pd.DataFrame(out_rows)


def load_cached(code: str) -> pd.DataFrame | None:
    """读单股 cache, miss 返回 None."""
    p = cache_path(code)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def save_cached(code: str, df: pd.DataFrame) -> None:
    """写单股 cache, 覆盖式. 仅保留 code 匹配的行."""
    _ensure_cache_dir()
    code_str = str(code).zfill(6)
    if df is None or df.empty:
        return
    out = df.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out = out[out["code"] == code_str].reset_index(drop=True)
    if out.empty:
        return
    out.to_parquet(cache_path(code_str), index=False)


def is_ic_feasibility_note() -> str:
    """Returns the feasibility verdict for IS (2014-2020) IC analysis.

    See module docstring for full reasoning.
    """
    return (
        "EPS consensus history NOT FEASIBLE for IS 2014-2020 in sandbox: "
        "(1) THS basic.10jqka.com.cn returns current snapshot only, no "
        "historical revisions; (2) EM reportapi.eastmoney.com EPS fields "
        "empty pre-2024; (3) akshare routes through push2his "
        "(proxy-blocked). Need paid wind/choice or tushare-pro for IS."
    )
