"""基本面因子 per-stock cache + akshare 抓取 + 季频对齐.

设计动机:
- akshare 单股财报 (sina stock_financial_analysis_indicator) 每股 ~9 秒, CSI300
  跑一遍 ~45 min. 必须 per-stock parquet + 增量更新 才能复用.
- factor IC 分析需把"季度披露数据"对齐到"每月初的最新已知季报"
  (point-in-time, 避免用 future 财报 → lookahead leak).

主要 endpoint (Phase 3 探查后确定):
- 东财 (EM): ak.stock_financial_analysis_indicator_em(symbol="600519.SH",
  indicator="按报告期"). 季频, 1998-至今, 141 字段, ~0.5 sec/股, 无 IP 限.
- (Sina endpoint stock_financial_analysis_indicator 也覆盖 2014-2020 但
  连续抓 ~20 股后被 sina 反爬封 IP 5-60 min; 已废弃改用 EM.)
- (stock_value_em 日频 PE/PB 仅回溯 2018, 不覆盖 2014-2017, 故本 cache
  不保 pe_ttm/pb; 如要做 PE/PB 因子须另起 cache 且 IC 分析限 2018+.)

Schema (per-stock parquet, data_cache/fundamentals/{code}.parquet):
    code             str   (6 位补零)
    report_date      datetime64[ns]   财报截止日 (季度末)
    roe              float (净资产收益率加权 %)
    net_margin       float (销售净利率 %)
    gross_margin     float (销售毛利率 %)
    debt_to_asset    float (资产负债率 %)
    current_ratio    float (流动比率)
    quick_ratio      float (速动比率)
    revenue_yoy      float (营业总收入同比 %)
    net_profit_yoy   float (归母净利润同比 %)
    eps_basic        float (基本每股收益)
    bps              float (每股净资产)
    ocf_per_share    float (每股经营现金净流量)
    roa              float (总资产净利率 %)
    fetched_at       datetime64[ns]   抓取时间戳

约束:
- 本模块只做读/写/合并/字段映射, 不做 schedule
- akshare 调用在 fetch_fundamentals(); 失败 retry 2 次后抛 RuntimeError
- get_quarterly_at_date 做 backward fill (point-in-time): 季报披露 lag
  默认 60 天 (季报最长 1 个月, 年报最长 4 个月; 60 天保守可见性).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "fundamentals"

# 季报披露 lag (天). 保守值: 季报实际公开日 ≈ report_date + 60 天.
ANNOUNCE_LAG_DAYS = 60

# EM (东财) → 本模块 schema 字段映射 (EM 列名, 本模块字段名, 类型)
EM_FIELD_MAP = [
    ("REPORT_DATE", "report_date", "date"),
    ("ROEJQ", "roe", "float"),
    ("XSJLL", "net_margin", "float"),
    ("XSMLL", "gross_margin", "float"),
    ("ZCFZL", "debt_to_asset", "float"),
    ("LD", "current_ratio", "float"),
    ("SD", "quick_ratio", "float"),
    ("TOTALOPERATEREVETZ", "revenue_yoy", "float"),
    ("PARENTNETPROFITTZ", "net_profit_yoy", "float"),
    ("EPSJB", "eps_basic", "float"),
    ("BPS", "bps", "float"),
    ("MGJYXJJE", "ocf_per_share", "float"),
    ("ZZCJLL", "roa", "float"),
]

SCHEMA_COLUMNS = (
    ["code"]
    + [m[1] for m in EM_FIELD_MAP]
    + ["fetched_at"]
)


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(code: str) -> Path:
    """单股 cache 路径 data_cache/fundamentals/{code}.parquet."""
    code_str = str(code).zfill(6)
    return CACHE_DIR / f"{code_str}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe by (code, report_date) + sort asc. 不动入参 (返回新 df)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    out = df.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["report_date"] = pd.to_datetime(out["report_date"])
    out = (
        out.drop_duplicates(subset=["code", "report_date"], keep="last")
        .sort_values(["code", "report_date"])
        .reset_index(drop=True)
    )
    cols_present = [c for c in SCHEMA_COLUMNS if c in out.columns]
    return out[cols_present]


def load_cached(code: str) -> pd.DataFrame | None:
    """读单股 cache, miss 返回 None."""
    p = cache_path(code)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def save_cached(code: str, df: pd.DataFrame) -> None:
    """写单股 cache, 覆盖式. 内部 dedupe + sort."""
    _ensure_cache_dir()
    code_str = str(code).zfill(6)
    out = _normalize(df)
    out = out[out["code"] == code_str].reset_index(drop=True)
    out.to_parquet(cache_path(code_str), index=False)


def merge_incremental(code: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """合并新 df 到 cache, 写盘, 返回合并后的 df.

    去重 by (code, report_date); new_df 优先 (keep='last').
    """
    old = load_cached(code)
    if old is None:
        merged = new_df
    else:
        merged = pd.concat([old, new_df], ignore_index=True)
    out = _normalize(merged)
    code_str = str(code).zfill(6)
    out = out[out["code"] == code_str].reset_index(drop=True)
    _ensure_cache_dir()
    out.to_parquet(cache_path(code_str), index=False)
    return out


def needs_refresh(
    code: str,
    today: pd.Timestamp,
    *,
    stale_days: int = 90,
) -> bool:
    """判断 cache 是否需要更新.

    True 如果:
      - cache 不存在或空, 或
      - cache 内 max(report_date) + stale_days < today
        (季报每 90 天更新, 默认 stale_days=90)
    """
    df = load_cached(code)
    if df is None or df.empty or "report_date" not in df.columns:
        return True
    today = pd.to_datetime(today).normalize()
    max_date = pd.to_datetime(df["report_date"]).max().normalize()
    return max_date + pd.Timedelta(days=stale_days) < today


def _to_em_symbol(code: str) -> str:
    """6 位 code → EM 格式 SECUCODE (6XXXXX.SH / 0XXXXX/3XXXXX.SZ).

    688xxx (科创板), 6xxxxx → .SH
    0xxxxx, 3xxxxx (创业板) → .SZ
    """
    c = str(code).zfill(6)
    if c.startswith("6"):
        return f"{c}.SH"
    if c.startswith(("0", "3")):
        return f"{c}.SZ"
    if c.startswith("4") or c.startswith("8"):  # 北交所
        return f"{c}.BJ"
    return f"{c}.SZ"  # 默认


def _map_em_df(code: str, em_df: pd.DataFrame) -> pd.DataFrame:
    """EM stock_financial_analysis_indicator_em 输出 → 本模块 schema."""
    if em_df is None or em_df.empty:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    out = pd.DataFrame()
    out["code"] = [str(code).zfill(6)] * len(em_df)
    for em_col, our_col, kind in EM_FIELD_MAP:
        if em_col not in em_df.columns:
            out[our_col] = pd.NA
            continue
        if kind == "date":
            out[our_col] = pd.to_datetime(em_df[em_col], errors="coerce")
        else:
            out[our_col] = pd.to_numeric(em_df[em_col], errors="coerce")
    out["fetched_at"] = pd.Timestamp.now()
    out = out.dropna(subset=["report_date"]).reset_index(drop=True)
    return out[SCHEMA_COLUMNS]


def fetch_fundamentals(
    code: str,
    *,
    retries: int = 2,
    retry_sleep: float = 2.0,
) -> pd.DataFrame:
    """通过 akshare (东财 EM) 抓单股财务指标, 转 schema, 返回 df.

    不写 cache (调用方决定). 失败 retry `retries` 次后抛 RuntimeError.
    """
    import akshare as ak

    last_err: Exception | None = None
    em_symbol = _to_em_symbol(code)
    for attempt in range(retries + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = ak.stock_financial_analysis_indicator_em(
                    symbol=em_symbol,
                    indicator="按报告期",
                )
            return _map_em_df(code, raw)
        except Exception as e:  # noqa: BLE001 - 任何异常都 retry
            last_err = e
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(
        f"fetch_fundamentals({code}/{em_symbol}) "
        f"{retries + 1} 次后失败: {last_err}"
    )


def get_quarterly_at_date(
    code: str,
    query_date: pd.Timestamp,
    *,
    df: pd.DataFrame | None = None,
    announce_lag_days: int = ANNOUNCE_LAG_DAYS,
) -> dict | None:
    """Point-in-time 查询: 查询日可见的最新季报 (含披露 lag).

    "可见" 定义: report_date + announce_lag_days <= query_date.
      - announce_lag_days=60 时 query_date=2014-07-01 看不到
        report_date=2014-06-30 的季报 (披露日 2014-08-29), 只能看
        2014-03-31 (披露日 2014-05-30, 可见).

    若 df 参数给出, 用它代替读 cache (用于批量 / 测试).
    返回 dict (所有 schema 字段), 或 None (无可见季报).
    """
    if df is None:
        df = load_cached(code)
    if df is None or df.empty:
        return None
    query_date = pd.to_datetime(query_date).normalize()
    cutoff = query_date - pd.Timedelta(days=announce_lag_days)
    visible = df[pd.to_datetime(df["report_date"]) <= cutoff]
    if visible.empty:
        return None
    latest = visible.sort_values("report_date").iloc[-1]
    return latest.to_dict()


def load_many(codes: list[str]) -> pd.DataFrame:
    """批量读多个股 cache, 缺失股略过. 返回长 df."""
    frames = []
    for c in codes:
        d = load_cached(c)
        if d is None or d.empty:
            continue
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["code"] = out["code"].astype(str).str.zfill(6)
    return out.sort_values(["code", "report_date"]).reset_index(drop=True)
