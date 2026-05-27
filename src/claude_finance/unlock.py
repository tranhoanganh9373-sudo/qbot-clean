"""限售股解禁(unlock / share-lockup expiration)数据 fetch + 因子工程.

数据源
------
akshare:
  - ak.stock_restricted_release_detail_em(start_date, end_date)
    全市场单期间解禁明细 (datacenter-web; 推荐, 一次拉全市场).
    返回字段:
        股票代码, 股票简称, 解禁时间, 限售股类型,
        解禁数量, 实际解禁数量, 实际解禁市值, 占解禁前流通市值比例,
        解禁前一交易日收盘价, 解禁前20日涨跌幅, 解禁后20日涨跌幅
    注意: `解禁后20日涨跌幅` 是 lookahead 字段, 因子工程绝不使用.

Cache schema (单 parquet, 全市场)
-------------------------------
    code               str   (6位补零)
    unlock_date        datetime64[ns]
    unlock_shares      float (实际解禁数量, 股)
    unlock_value       float (实际解禁市值, 元)
    unlock_ratio_cap   float (占解禁前流通市值比例, 0..1)
    unlock_type        str   (ipo / spo / incentive / other)
    close_before       float (解禁前一交易日收盘价 — 仅记录, 不做因子)

说明: 用 akshare 自带的 `占解禁前流通市值比例` 替代
`unlock_shares / outstanding_share` 自算, 原因:
  1. long_history.parquet 仅 2023-04+, 不覆盖 IS 2014-2020.
  2. baidu_kline_v2.turnoverratio 100% NaN, 不能反算 outstanding_share.
  3. akshare 字段 point-in-time 安全且权威 (东财计算口径).

约束
----
- 本模块负责 fetch + cache + 因子点查询, 不做 IC 计算.
- 因子 lookback: 给定 T 日, 仅看 t > T (forward window) 的解禁事件.
- 任何写盘前 dedupe(code, unlock_date, unlock_type, unlock_shares) + sort.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "unlock"
CACHE_FILE = CACHE_DIR / "unlock_detail_em.parquet"

# 类型映射 (中文 → 英文 tag)
TYPE_MAP = {
    "首发原股东限售股份": "ipo",
    "首发战略配售股份": "ipo",
    "首发机构配售股份": "ipo",
    "定向增发机构配售股份": "spo",
    "股权激励限售股份": "incentive",
    "追加承诺限售股份上市流通": "other",
    "股权分置限售股份": "other",
}


def _classify_type(zh: str) -> str:
    """中文 limit-share 类型 → 4 类英文 tag."""
    if zh in TYPE_MAP:
        return TYPE_MAP[zh]
    s = str(zh)
    if "首发" in s:
        return "ipo"
    if "定向增发" in s or "定增" in s:
        return "spo"
    if "激励" in s:
        return "incentive"
    return "other"


def fetch_detail_range(
    start_date: str,
    end_date: str,
    max_retries: int = 4,
    retry_sleep: float = 3.0,
) -> pd.DataFrame:
    """一次区间拉全市场解禁明细; 自带 SSL 重试.

    参数
    ----
    start_date : str  "YYYYMMDD"
    end_date   : str  "YYYYMMDD"

    返回标准化 schema 的 DataFrame; 失败抛 RuntimeError.
    """
    import akshare as ak  # 延迟导入

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = ak.stock_restricted_release_detail_em(
                start_date=start_date, end_date=end_date,
            )
            return _normalize(raw)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(
        f"fetch_detail_range({start_date},{end_date}) failed after "
        f"{max_retries} tries: {last_err}"
    )


def _normalize(raw: pd.DataFrame | None) -> pd.DataFrame:
    """raw akshare → cache schema."""
    cols = [
        "code", "unlock_date", "unlock_shares", "unlock_value",
        "unlock_ratio_cap", "unlock_type", "close_before",
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)

    df = raw.rename(columns={
        "股票代码": "code",
        "解禁时间": "unlock_date",
        "限售股类型": "unlock_type_zh",
        "实际解禁数量": "unlock_shares",
        "实际解禁市值": "unlock_value",
        "占解禁前流通市值比例": "unlock_ratio_cap",
        "解禁前一交易日收盘价": "close_before",
    })
    keep = [
        "code", "unlock_date", "unlock_type_zh",
        "unlock_shares", "unlock_value", "unlock_ratio_cap", "close_before",
    ]
    # 容忍源字段缺失
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[keep].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["unlock_date"] = pd.to_datetime(df["unlock_date"], errors="coerce")
    df["unlock_type"] = df["unlock_type_zh"].astype(str).map(_classify_type)
    df = df.drop(columns=["unlock_type_zh"])
    for col in [
        "unlock_shares", "unlock_value", "unlock_ratio_cap", "close_before",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["unlock_date"]).reset_index(drop=True)
    return df[cols].reset_index(drop=True)


def load_cache() -> pd.DataFrame:
    """读全市场解禁 cache; 不存在返回空 DataFrame."""
    if not CACHE_FILE.exists():
        return _normalize(None)
    df = pd.read_parquet(CACHE_FILE)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["unlock_date"] = pd.to_datetime(df["unlock_date"])
    return df


def save_cache(df: pd.DataFrame) -> None:
    """落盘 (dedupe + sort + 创建目录)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["code", "unlock_date", "unlock_type", "unlock_shares"]
    df = (
        df.drop_duplicates(subset=keys, keep="last")
        .sort_values(["code", "unlock_date"])
        .reset_index(drop=True)
    )
    df.to_parquet(CACHE_FILE, index=False)


def merge_into_cache(new_df: pd.DataFrame) -> pd.DataFrame:
    """合并新数据到 cache, 返回合并后的全集."""
    cur = load_cache()
    merged = pd.concat([cur, new_df], ignore_index=True)
    save_cache(merged)
    return load_cache()


def filter_codes(df: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """筛某个 universe."""
    codes_set = {str(c).zfill(6) for c in codes}
    out = df[df["code"].isin(codes_set)].copy()
    return out.reset_index(drop=True)


# ---------- 因子工程 ----------


def forward_unlock_metrics(
    unlock_df: pd.DataFrame,
    code: str,
    asof_date: pd.Timestamp,
    windows_days: tuple[int, ...] = (5, 20, 60),
) -> dict[str, float]:
    """对单股 code 在 T 日, 返回未来 N 日的解禁因子.

    返回字段 (per window N):
        unlock_pct_next_N   : sum(unlock_ratio_cap) over (T, T+N 日历日]
        unlock_value_next_N : sum(unlock_value, 亿元) over (T, T+N 日历日]
        unlock_imminent_N   : 1 if any unlock in (T, T+N] else 0

    `占解禁前流通市值比例` 已是 point-in-time 量, 直接 sum
    (假设同窗口内 unlock 视为同时点叠加; 合理近似, 大窗口可能略夸大).
    """
    code = str(code).zfill(6)
    asof_date = pd.Timestamp(asof_date)
    sub = unlock_df[unlock_df["code"] == code]
    if sub.empty:
        return _zero_metrics(windows_days)

    out: dict[str, float] = {}
    for n in windows_days:
        win = sub[
            (sub["unlock_date"] > asof_date)
            & (sub["unlock_date"] <= asof_date + pd.Timedelta(days=n))
        ]
        if win.empty:
            out[f"unlock_pct_next_{n}"] = 0.0
            out[f"unlock_value_next_{n}"] = 0.0
            out[f"unlock_imminent_{n}"] = 0.0
        else:
            out[f"unlock_pct_next_{n}"] = float(
                win["unlock_ratio_cap"].fillna(0).sum()
            )
            out[f"unlock_value_next_{n}"] = float(
                win["unlock_value"].fillna(0).sum() / 1e8
            )
            out[f"unlock_imminent_{n}"] = 1.0
    return out


def _zero_metrics(windows_days: tuple[int, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for n in windows_days:
        out[f"unlock_pct_next_{n}"] = 0.0
        out[f"unlock_value_next_{n}"] = 0.0
        out[f"unlock_imminent_{n}"] = 0.0
    return out


def build_factor_panel(
    unlock_df: pd.DataFrame,
    codes: list[str],
    asof_dates: list[pd.Timestamp],
    windows_days: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    """构造 panel: (asof_date, code) → factor columns.

    对每个 asof_date 做一次切片 + 按 code groupby; 比 nested loop 快.
    """
    asof_dates = sorted({pd.Timestamp(d) for d in asof_dates})
    codes = [str(c).zfill(6) for c in codes]
    codes_set = set(codes)
    sub = unlock_df[unlock_df["code"].isin(codes_set)].copy()
    sub = sub.sort_values(["unlock_date"]).reset_index(drop=True)
    max_window = max(windows_days)

    rows: list[dict] = []
    for asof in asof_dates:
        upper = asof + pd.Timedelta(days=max_window)
        slab = sub[(sub["unlock_date"] > asof) & (sub["unlock_date"] <= upper)]
        codes_with_events = set(slab["code"].unique())
        # 为有事件的 code 预算 per-window 累积 (一次扫表)
        events_by_code: dict[str, pd.DataFrame] = {
            c: slab[slab["code"] == c] for c in codes_with_events
        }
        for code in codes:
            row = {"asof_date": asof, "code": code}
            if code not in codes_with_events:
                row.update(_zero_metrics(windows_days))
            else:
                csub = events_by_code[code]
                for n in windows_days:
                    win = csub[
                        csub["unlock_date"] <= asof + pd.Timedelta(days=n)
                    ]
                    if win.empty:
                        row[f"unlock_pct_next_{n}"] = 0.0
                        row[f"unlock_value_next_{n}"] = 0.0
                        row[f"unlock_imminent_{n}"] = 0.0
                    else:
                        row[f"unlock_pct_next_{n}"] = float(
                            win["unlock_ratio_cap"].fillna(0).sum()
                        )
                        row[f"unlock_value_next_{n}"] = float(
                            win["unlock_value"].fillna(0).sum() / 1e8
                        )
                        row[f"unlock_imminent_{n}"] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)
