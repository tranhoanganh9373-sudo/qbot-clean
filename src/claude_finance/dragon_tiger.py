"""龙虎榜数据 per-stock cache + 东财 datacenter 抓取 + IC 因子辅助.

设计动机:
- 龙虎榜是事件驱动信号 (单股仅在触发涨跌幅/换手率/异常波动时上榜).
- 端点 https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=
  RPT_DAILYBILLBOARD_DETAILSNEW sandbox ✅ 可用, IS 2014-2020 全覆盖.
- per-stock parquet 避免每次重抓; 一次 14-yr 拉完后增量更新即可.

Schema (per-stock parquet, data_cache/dragon_tiger/{code}.parquet):
    code              str   (6 位补零)
    date              datetime64[ns]   交易日 (TRADE_DATE)
    reason            str   上榜原因 (EXPLANATION)
    net_amt           float 龙虎榜净额 (BILLBOARD_NET_AMT, 单位 元)
    buy_amt           float 龙虎榜买入额 (BILLBOARD_BUY_AMT, 元)
    sell_amt          float 龙虎榜卖出额 (BILLBOARD_SELL_AMT, 元)
    accum_amount      float 全天总成交额 (ACCUM_AMOUNT, 元)
    turnover_pct      float 换手率 (TURNOVERRATE, %)
    change_rate       float 当日涨跌幅 (CHANGE_RATE, %)
    fetched_at        datetime64[ns]   抓取时间戳

派生字段 (在 panel 构造时计算, 不入 cache):
    top_buyer_pct       = buy_amt / accum_amount * 100      龙虎榜买方占比
    top_seller_pct      = sell_amt / accum_amount * 100     龙虎榜卖方占比
    net_buy_pct         = net_amt / accum_amount * 100      净买入占比

约束:
- 任何写盘前必须 dedupe(code,date) + sort by date asc
- 全天单股可能多条上榜记录 (不同 reason), cache 内可重复 → IC panel 聚合
  时按 (code, date) groupby sum.
- net_amt 可正可负, 单位是元 (不是万元).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko)"
)

DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "data_cache" / "dragon_tiger"
)

SCHEMA_COLUMNS = [
    "code",
    "date",
    "reason",
    "net_amt",
    "buy_amt",
    "sell_amt",
    "accum_amount",
    "turnover_pct",
    "change_rate",
    "fetched_at",
]


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(code: str) -> Path:
    """单股 cache 路径 data_cache/dragon_tiger/{code}.parquet."""
    code_str = str(code).zfill(6)
    return CACHE_DIR / f"{code_str}.parquet"


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe by (code, date, reason) + sort by date asc.

    返回新 df (immutable, 不动入参).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    out = df.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"])
    out = (
        out.drop_duplicates(
            subset=["code", "date", "reason"], keep="last"
        )
        .sort_values(["code", "date", "reason"])
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
    """写单股 cache, 覆盖式 (仅保留 code 匹配行, dedupe + sort)."""
    _ensure_cache_dir()
    code_str = str(code).zfill(6)
    out = _normalize(df)
    out = out[out["code"] == code_str].reset_index(drop=True)
    out.to_parquet(cache_path(code_str), index=False)


def merge_incremental(code: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """合并新拉的 df 到已 cache 的, 写盘, 返回合并后的 df."""
    old = load_cached(code)
    merged = (
        new_df if old is None else pd.concat([old, new_df], ignore_index=True)
    )
    out = _normalize(merged)
    code_str = str(code).zfill(6)
    out = out[out["code"] == code_str].reset_index(drop=True)
    _ensure_cache_dir()
    out.to_parquet(cache_path(code_str), index=False)
    return out


def fetch_dragon_tiger(
    code: str,
    start_date: str,
    end_date: str,
    page_size: int = 500,
    retries: int = 3,
    sleep_between_pages: float = 0.3,
    timeout: int = 20,
) -> pd.DataFrame:
    """从东财 datacenter 抓单股龙虎榜历史 (含 IS 期).

    参数:
        code: 6 位股票代码
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        page_size: 每页大小, 上限 500
        retries: 单页失败重试次数
        sleep_between_pages: 翻页间隔秒
        timeout: 单次 HTTP timeout

    返回 DataFrame, schema 见 SCHEMA_COLUMNS. 若无数据返回空 DataFrame.

    raise: requests.RequestException 若 retries 后仍失败.
    """
    code_str = str(code).zfill(6)
    rows: list[dict] = []
    page = 1
    fetched_at = pd.Timestamp.now()
    while True:
        params = {
            "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
            "columns": (
                "TRADE_DATE,SECURITY_CODE,EXPLANATION,"
                "BILLBOARD_NET_AMT,BILLBOARD_BUY_AMT,BILLBOARD_SELL_AMT,"
                "ACCUM_AMOUNT,TURNOVERRATE,CHANGE_RATE"
            ),
            "filter": (
                f'(SECURITY_CODE="{code_str}")'
                f"(TRADE_DATE>='{start_date}')"
                f"(TRADE_DATE<='{end_date}')"
            ),
            "pageNumber": str(page),
            "pageSize": str(page_size),
            "sortColumns": "TRADE_DATE",
            "sortTypes": "-1",
            "source": "WEB",
            "client": "WEB",
        }
        last_err: Exception | None = None
        payload = None
        for attempt in range(retries):
            try:
                resp = requests.get(
                    DATACENTER_URL,
                    params=params,
                    headers={"User-Agent": UA},
                    timeout=timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
                last_err = None
                break
            except (requests.RequestException, ValueError) as e:
                last_err = e
                if attempt + 1 < retries:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    raise
        if last_err is not None:
            raise last_err
        assert payload is not None

        result = payload.get("result") or {}
        data = result.get("data") or []
        if not data:
            break

        for d in data:
            rows.append(
                {
                    "code": code_str,
                    "date": pd.to_datetime(d.get("TRADE_DATE")).normalize(),
                    "reason": str(d.get("EXPLANATION") or ""),
                    "net_amt": _safe_float(d.get("BILLBOARD_NET_AMT")),
                    "buy_amt": _safe_float(d.get("BILLBOARD_BUY_AMT")),
                    "sell_amt": _safe_float(d.get("BILLBOARD_SELL_AMT")),
                    "accum_amount": _safe_float(d.get("ACCUM_AMOUNT")),
                    "turnover_pct": _safe_float(d.get("TURNOVERRATE")),
                    "change_rate": _safe_float(d.get("CHANGE_RATE")),
                    "fetched_at": fetched_at,
                }
            )

        # paginate
        if len(data) < page_size:
            break
        page += 1
        time.sleep(sleep_between_pages)

    return _normalize(pd.DataFrame(rows))


def fetch_and_cache(
    code: str,
    start_date: str,
    end_date: str,
    skip_if_recent: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """抓 + 增量合并到 cache, 返回 cache 全量 df.

    skip_if_recent: True → 若 cache 已存且 max(date) >= end_date, 跳过.
    """
    code_str = str(code).zfill(6)
    if skip_if_recent:
        old = load_cached(code_str)
        if old is not None and not old.empty:
            max_d = pd.to_datetime(old["date"]).max()
            if max_d >= pd.to_datetime(end_date):
                return old
    new_df = fetch_dragon_tiger(code_str, start_date, end_date, **kwargs)
    if new_df.empty:
        # 无龙虎榜记录, 写一个空文件方便下次 skip
        _ensure_cache_dir()
        empty = pd.DataFrame(columns=SCHEMA_COLUMNS)
        empty.to_parquet(cache_path(code_str), index=False)
        return empty
    return merge_incremental(code_str, new_df)


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """把多 reason 的 daily 记录聚合到 (code, date) 级别.

    Aggregation:
        net_amt        = sum  (各 reason 净额相加)
        buy_amt        = sum
        sell_amt       = sum
        accum_amount   = max  (同一天 ACCUM_AMOUNT 相同, 取 max 防异常)
        turnover_pct   = max
        change_rate    = max  (绝对值最大)
        n_reasons      = count

    派生:
        net_buy_pct    = net_amt / accum_amount * 100
        top_buyer_pct  = buy_amt / accum_amount * 100
        top_seller_pct = sell_amt / accum_amount * 100
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "code",
                "date",
                "net_amt",
                "buy_amt",
                "sell_amt",
                "accum_amount",
                "turnover_pct",
                "change_rate",
                "n_reasons",
                "net_buy_pct",
                "top_buyer_pct",
                "top_seller_pct",
            ]
        )
    agg = (
        df.groupby(["code", "date"], as_index=False).agg(
            net_amt=("net_amt", "sum"),
            buy_amt=("buy_amt", "sum"),
            sell_amt=("sell_amt", "sum"),
            accum_amount=("accum_amount", "max"),
            turnover_pct=("turnover_pct", "max"),
            change_rate=("change_rate", "max"),
            n_reasons=("reason", "count"),
        )
    )
    accum = agg["accum_amount"].where(agg["accum_amount"] > 0)
    agg["net_buy_pct"] = agg["net_amt"] / accum * 100
    agg["top_buyer_pct"] = agg["buy_amt"] / accum * 100
    agg["top_seller_pct"] = agg["sell_amt"] / accum * 100
    return agg


def rolling_event_features(
    daily_agg: pd.DataFrame,
    all_codes: list[str],
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """构造 daily panel (every code × every trading day) 的滚动事件特征.

    针对 IC 分析, 把"事件驱动" cache 摊平为日级别因子, 在非上榜日填 0.

    输入:
        daily_agg: daily_features() 的输出 (sparse, 只有上榜日)
        all_codes: panel 包含的所有股代码 (6 位)
        date_index: panel 时间轴 (trading days)

    输出 panel: (date, code) × [
        net_buy_pct_evt,             当日上榜净买占比 (非上榜日 0)
        on_list_today,               当日是否上榜 (0/1)
        top_list_count_30d,          滚动 30 日上榜次数
        top_list_count_60d,          滚动 60 日上榜次数
        net_buy_sum_30d_wan,         滚动 30 日累计 net_amt (万元)
        net_buy_sum_60d_wan,         滚动 60 日累计 net_amt (万元)
    ]
    """
    codes = [str(c).zfill(6) for c in all_codes]
    full_idx = pd.MultiIndex.from_product(
        [date_index, codes], names=["date", "code"]
    )
    flat = pd.DataFrame(index=full_idx).reset_index()
    if daily_agg is not None and not daily_agg.empty:
        sparse = daily_agg.copy()
        sparse["code"] = sparse["code"].astype(str).str.zfill(6)
        sparse["date"] = pd.to_datetime(sparse["date"]).dt.normalize()
        merged = flat.merge(
            sparse[["date", "code", "net_amt", "net_buy_pct"]],
            on=["date", "code"],
            how="left",
        )
    else:
        merged = flat.copy()
        merged["net_amt"] = float("nan")
        merged["net_buy_pct"] = float("nan")

    merged["on_list_today"] = merged["net_amt"].notna().astype(int)
    merged["net_buy_pct_evt"] = merged["net_buy_pct"].fillna(0.0)
    merged["net_amt"] = merged["net_amt"].fillna(0.0)

    merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
    grp = merged.groupby("code", sort=False)
    merged["top_list_count_30d"] = grp["on_list_today"].transform(
        lambda x: x.rolling(30, min_periods=1).sum()
    )
    merged["top_list_count_60d"] = grp["on_list_today"].transform(
        lambda x: x.rolling(60, min_periods=1).sum()
    )
    merged["net_buy_sum_30d_wan"] = grp["net_amt"].transform(
        lambda x: x.rolling(30, min_periods=1).sum() / 1e4
    )
    merged["net_buy_sum_60d_wan"] = grp["net_amt"].transform(
        lambda x: x.rolling(60, min_periods=1).sum() / 1e4
    )

    return merged[
        [
            "date",
            "code",
            "on_list_today",
            "net_buy_pct_evt",
            "top_list_count_30d",
            "top_list_count_60d",
            "net_buy_sum_30d_wan",
            "net_buy_sum_60d_wan",
        ]
    ]
