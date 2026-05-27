"""融资融券数据本地缓存 (per-stock parquet + 增量更新).

设计动机:
- factor_margin() 每次调用都走网络 (东财 RPT_MARGIN_TOTAL),
  CSI300 跑一遍 14 年历史 ≈ 数十分钟.
- per-stock parquet (code.parquet) 拆分 + 增量合并, 把秒级响应做出来,
  网络只补 cache miss / 增量更新.

Schema (与 examples/factor_mining.py:factor_margin 输出完全兼容):
    code            str   (6 位补零)
    date            datetime64[ns]
    rzye            float (融资余额)
    rzmre           float (融资买入额)
    rzche           float (融资偿还额)
    margin_5d_chg   float (rzye 的 5 日 pct_change)
    margin_20d_chg  float (rzye 的 20 日 pct_change)

约束:
- 本模块只做读/写/合并, 不做网络抓取
- 任何写盘前必须 dedupe(code,date) + sort by date asc
- 合并新数据后必须重算 5d/20d 变化率 (因为新数据末尾可能改变 rolling)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "margin_eastmoney"

SCHEMA_COLUMNS = [
    "code", "date", "rzye", "rzmre", "rzche",
    "margin_5d_chg", "margin_20d_chg",
]


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(code: str) -> Path:
    """单股 cache 路径 data_cache/margin_eastmoney/{code}.parquet."""
    code_str = str(code).zfill(6)
    return CACHE_DIR / f"{code_str}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe by (code, date) + sort by date asc + 重算 5d/20d 变化率.

    返回新 df (immutable, 不动入参).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    out = df.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"])
    # dedupe + sort
    out = (
        out.drop_duplicates(subset=["code", "date"], keep="last")
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )
    # 重算变化率 (按 code 分组)
    out["margin_5d_chg"] = out.groupby("code")["rzye"].pct_change(5)
    out["margin_20d_chg"] = out.groupby("code")["rzye"].pct_change(20)
    # 仅保留 schema 列, 顺序固定
    cols_present = [c for c in SCHEMA_COLUMNS if c in out.columns]
    return out[cols_present]


def load_cached(code: str) -> pd.DataFrame | None:
    """读单股 cache, miss 返回 None."""
    p = cache_path(code)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def save_cached(code: str, df: pd.DataFrame) -> None:
    """写单股 cache, 覆盖式. 内部会 dedupe + sort + 重算变化率."""
    _ensure_cache_dir()
    code_str = str(code).zfill(6)
    out = _normalize(df)
    # 仅保留该 code 的行
    out = out[out["code"] == code_str].reset_index(drop=True)
    out.to_parquet(cache_path(code_str), index=False)


def merge_incremental(code: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """合并新拉的 df 到已 cache 的, 写盘, 返回合并后的 df.

    - 去重 by (code, date), new_df 优先 (keep='last')
    - 重算 margin_5d_chg / margin_20d_chg
    - 自动 sort by date asc
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


def load_many(codes: list[str]) -> pd.DataFrame:
    """批量读多个股 cache, 缺失股从结果中略 (不抓网络). 返回长 df."""
    frames = []
    for c in codes:
        df = load_cached(c)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["code"] = out["code"].astype(str).str.zfill(6)
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def needs_refresh(code: str, today: pd.Timestamp) -> bool:
    """判断 cache 是否需要增量更新.

    返回 True 如果:
      - cache 不存在, 或
      - cache 为空, 或
      - cache 内 max(date) < today - 1 day  (留 1 天容差给 T+1 数据)
    """
    df = load_cached(code)
    if df is None or df.empty or "date" not in df.columns:
        return True
    today = pd.to_datetime(today).normalize()
    max_date = pd.to_datetime(df["date"]).max().normalize()
    return max_date < today - pd.Timedelta(days=1)


def bootstrap_from_bulk_parquet(bulk_path: Path) -> int:
    """一次性从 csi300_margin_14yr.parquet 这种 bulk 文件 split 成 per-stock cache.

    用法: OOS subagent 跑完 14 年抓取后, 把 bulk parquet bootstrap 成 per-stock cache.

    参数:
        bulk_path: 一个长 df parquet, 必须含 code/date/rzye/rzmre/rzche 列;
                   margin_5d_chg/margin_20d_chg 可缺 (会重算).

    返回: 写入的股数.
    """
    bulk_path = Path(bulk_path)
    if not bulk_path.exists():
        raise FileNotFoundError(f"bulk parquet 不存在: {bulk_path}")
    bulk = pd.read_parquet(bulk_path)
    required = {"code", "date", "rzye", "rzmre", "rzche"}
    missing = required - set(bulk.columns)
    if missing:
        raise ValueError(f"bulk parquet 缺列: {missing}")
    bulk["code"] = bulk["code"].astype(str).str.zfill(6)
    bulk["date"] = pd.to_datetime(bulk["date"])
    _ensure_cache_dir()
    n = 0
    for code, sub in bulk.groupby("code", sort=False):
        save_cached(code, sub)
        n += 1
    return n
