"""通用 fetch cache 工具 — TTL-based, sandbox-safe.

设计动机:
- 探查类脚本 (scan_sectors / qbot_decision / strategy_recommended ...) 反复跑同样的
  网络抓取 (ak.stock_board_industry_cons_em / ak.index_stock_cons_weight_csindex /
  ak.stock_market_pe_lg ...) , 沙盒里慢 + 经常被拦截.
- 这里提供 *最小侵入式* 的 wrapper: ``cache_or_fetch(key, fetcher, ttl_hours=...)``.
- TTL 过期 -> 重新抓; fetcher 抛异常 + 有 stale cache -> 警告后用 stale.

cache 文件落 ``data_cache/scan_cache/{key}.parquet`` (or .csv) +
``data_cache/scan_cache/{key}.meta.json`` (存 fetched_at timestamp + serializer).

约束:
- 这里只做 fetch wrapper, 不做任何业务计算
- key 不能含特殊字符 (会做 _sanitize)
- fetcher 必须是 *无参 callable*, 返回 pd.DataFrame
"""
from __future__ import annotations

import json
import re
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "scan_cache"

_KEY_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_key(key: str) -> str:
    """key 里把非 [A-Za-z0-9_.-] 全替换成 _ , 防止路径越权 / windows 非法字符."""
    if not key:
        raise ValueError("key 不能为空")
    return _KEY_SAFE_RE.sub("_", key)


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _paths(key: str, serializer: str) -> tuple[Path, Path]:
    """返回 (data_path, meta_path).  serializer ∈ {parquet, csv}."""
    ext = {"parquet": "parquet", "csv": "csv"}[serializer]
    safe = _sanitize_key(key)
    return CACHE_DIR / f"{safe}.{ext}", CACHE_DIR / f"{safe}.meta.json"


def _read_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_meta(meta_path: Path, fetched_at: float, serializer: str) -> None:
    meta_path.write_text(
        json.dumps({"fetched_at": fetched_at, "serializer": serializer}, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_df(data_path: Path, serializer: str) -> pd.DataFrame:
    if serializer == "parquet":
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path)


def _write_df(df: pd.DataFrame, data_path: Path, serializer: str) -> None:
    if serializer == "parquet":
        df.to_parquet(data_path, index=False)
    else:
        df.to_csv(data_path, index=False)


def _age_hours(meta: dict, now: float | None = None) -> float:
    now = time.time() if now is None else now
    return (now - float(meta.get("fetched_at", 0))) / 3600.0


def cache_or_fetch(
    key: str,
    fetcher: Callable[[], pd.DataFrame],
    ttl_hours: float = 24.0,
    serializer: Literal["parquet", "csv"] = "parquet",
) -> pd.DataFrame:
    """读 cache; 不存在 / 过期 -> 调 fetcher 抓新.

    参数:
        key: cache 文件名 (会被 _sanitize_key 清洗)
        fetcher: 无参 callable, 返回 pd.DataFrame
        ttl_hours: cache 有效期 (小时); 超过则 stale.  必须 > 0
        serializer: ``parquet`` (默认, 保留 dtype) 或 ``csv``

    返回:
        pd.DataFrame.  来源优先级:
          1. fresh cache (age < ttl_hours)  -> 读 cache
          2. fetcher 抓成功  -> 落盘 + 返回
          3. fetcher 抛异常 + stale cache 存在  -> warn + 返回 stale
          4. fetcher 抛异常 + 无 cache  -> 抛原异常

    Raises:
        ValueError: ttl_hours <= 0 / serializer 不合法 / key 空
    """
    if ttl_hours <= 0:
        raise ValueError(f"ttl_hours 必须 > 0, got {ttl_hours}")
    if serializer not in ("parquet", "csv"):
        raise ValueError(f"serializer 只能是 parquet/csv, got {serializer!r}")

    data_path, meta_path = _paths(key, serializer)
    meta = _read_meta(meta_path)
    cache_exists = data_path.exists() and meta is not None

    # 1. fresh cache  -> 直接读
    if cache_exists and _age_hours(meta) < ttl_hours:
        return _read_df(data_path, serializer)

    # 2. 否则抓新
    try:
        df = fetcher()
    except Exception as exc:
        # 3. 抓失败 + 有 stale cache  -> warn + 用 stale
        if cache_exists:
            age = _age_hours(meta)
            warnings.warn(
                f"[scan_cache] fetcher 失败 ({type(exc).__name__}: {str(exc)[:80]}), "
                f"用 stale cache (age={age:.1f}h) key={key!r}",
                stacklevel=2,
            )
            return _read_df(data_path, serializer)
        # 4. 抓失败 + 无 cache  -> 抛
        raise

    # 抓成功 -> 落盘
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"fetcher 必须返回 pd.DataFrame, got {type(df).__name__}")
    _ensure_cache_dir()
    _write_df(df, data_path, serializer)
    _write_meta(meta_path, fetched_at=time.time(), serializer=serializer)
    return df


def list_cache() -> pd.DataFrame:
    """列出 ``CACHE_DIR`` 下所有 cache 项 (按 meta.json 枚举).

    返回列: key / fetched_at (iso str) / age_hours (float) / size_kb (float) /
    serializer (str).
    """
    cols = ["key", "fetched_at", "age_hours", "size_kb", "serializer"]
    if not CACHE_DIR.exists():
        return pd.DataFrame(columns=cols)

    rows = []
    now = time.time()
    for meta_path in sorted(CACHE_DIR.glob("*.meta.json")):
        meta = _read_meta(meta_path)
        if meta is None:
            continue
        key = meta_path.name.removesuffix(".meta.json")
        serializer = meta.get("serializer", "parquet")
        data_path = CACHE_DIR / f"{key}.{serializer}"
        size_kb = data_path.stat().st_size / 1024 if data_path.exists() else 0.0
        fetched_at = float(meta.get("fetched_at", 0))
        rows.append(
            {
                "key": key,
                "fetched_at": pd.Timestamp(fetched_at, unit="s").isoformat(),
                "age_hours": round((now - fetched_at) / 3600.0, 3),
                "size_kb": round(size_kb, 2),
                "serializer": serializer,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def clear_stale(max_age_hours: float = 168.0) -> int:
    """清除 cache 目录里 *严格超过* max_age_hours 的项.

    "严格超过" = age > max_age_hours.  age == max_age_hours 的项保留.

    返回: 清除的 cache 项数 (一项 = data + meta 一对).
    """
    if not CACHE_DIR.exists():
        return 0
    now = time.time()
    n_cleared = 0
    for meta_path in list(CACHE_DIR.glob("*.meta.json")):
        meta = _read_meta(meta_path)
        if meta is None:
            continue
        age = _age_hours(meta, now=now)
        if age <= max_age_hours:
            continue
        serializer = meta.get("serializer", "parquet")
        key = meta_path.name.removesuffix(".meta.json")
        data_path = CACHE_DIR / f"{key}.{serializer}"
        if data_path.exists():
            data_path.unlink()
        meta_path.unlink()
        n_cleared += 1
    return n_cleared
