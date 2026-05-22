"""A-share stock universe, with primary/fallback source and basic filters."""
from __future__ import annotations

import time

import pandas as pd


def _fetch_primary() -> pd.DataFrame:
    """Eastmoney real-time snapshot (faster, more stable than szse.cn)."""
    import akshare as ak

    df = ak.stock_zh_a_spot_em()
    df = df.rename(columns={"代码": "code", "名称": "name"})
    return df[["code", "name"]].copy()


def _fetch_fallback() -> pd.DataFrame:
    """Fallback: szse.cn (occasionally times out)."""
    import akshare as ak

    return ak.stock_info_a_code_name()


def get_stock_universe(
    *,
    drop_st: bool = True,
    drop_beijing: bool = True,
    drop_chinext: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return A-share code/name table with common filters applied.

    drop_st        — 剔除 ST/退市
    drop_beijing   — 剔除北交所（8 / 4 开头）
    drop_chinext   — 剔除创业板（30 开头）
    """
    if verbose:
        print("[1/3] 拉取 A 股代码全列表...")
    last_err: Exception | None = None
    df: pd.DataFrame | None = None
    for attempt in range(3):
        try:
            df = _fetch_primary()
            if verbose:
                print(f"  数据源: 东财快照接口  原始数量: {len(df)}")
            break
        except Exception as e:
            last_err = e
            if verbose:
                print(f"  东财接口第 {attempt + 1} 次失败: {str(e)[:80]}，重试...")
            time.sleep(2 * (attempt + 1))
    if df is None:
        if verbose:
            print("  东财接口全部失败，切换到深交所兜底...")
        try:
            df = _fetch_fallback()
            if verbose:
                print(f"  数据源: 深交所官网  原始数量: {len(df)}")
        except Exception as e:
            raise RuntimeError(f"全部数据源失败。主源: {last_err}; 兜底: {e}") from e

    if drop_st:
        before = len(df)
        df = df[~df["name"].str.contains("ST|退", regex=True, na=False)]
        if verbose:
            print(f"  剔除 ST/退市: -{before - len(df)}")

    if drop_beijing:
        before = len(df)
        df = df[~df["code"].str.startswith(("8", "4"))]
        if verbose:
            print(f"  剔除北交所: -{before - len(df)}")

    if drop_chinext:
        before = len(df)
        df = df[~df["code"].str.startswith("30")]
        if verbose:
            print(f"  剔除创业板: -{before - len(df)}")

    df = df.reset_index(drop=True)
    if verbose:
        print(f"  最终扫描数量: {len(df)}")
    return df
