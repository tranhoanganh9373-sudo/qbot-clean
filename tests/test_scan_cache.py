"""单元测试: scan_cache 模块 (通用 TTL 缓存 wrapper).

不依赖网络. 使用 tmp_path 隔离 CACHE_DIR. AAA pattern.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import pytest

from claude_finance import scan_cache


# -------------------- 通用 fixture --------------------

@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    iso = tmp_path / "scan_cache"
    iso.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(scan_cache, "CACHE_DIR", iso)
    return iso


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({"code": ["sh600000", "sz000001"], "name": ["浦发", "平安"]})


def _set_meta_age(cache_dir: Path, key: str, age_hours: float) -> None:
    """直接改 meta.json 的 fetched_at, 把 cache 'age' 调成想要的小时数.

    age 是相对 ``time.time()`` 的小时数; 因为 scan_cache 内部也调 ``time.time()``,
    边界 case 会有 µs 级抖动 — 测试时给精确边界 case 用 ``_freeze_time_at`` fixture.
    """
    meta_path = cache_dir / f"{key}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = time.time() - age_hours * 3600.0
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _set_meta_fetched_at(cache_dir: Path, key: str, fetched_at: float) -> None:
    """直接把 fetched_at 设到绝对 unix epoch (用于配 monkeypatch time.time)."""
    meta_path = cache_dir / f"{key}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = fetched_at
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


# -------------------- cache_or_fetch --------------------

def test_first_call_invokes_fetcher_and_writes_cache(_isolated_cache_dir: Path):
    # Arrange
    calls = {"n": 0}
    def fetcher():
        calls["n"] += 1
        return _sample_df()

    # Act
    out = scan_cache.cache_or_fetch("k1", fetcher, ttl_hours=24.0)

    # Assert
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(out, _sample_df())
    assert (_isolated_cache_dir / "k1.parquet").exists()
    assert (_isolated_cache_dir / "k1.meta.json").exists()


def test_fresh_cache_skips_fetcher(_isolated_cache_dir: Path):
    # Arrange
    calls = {"n": 0}
    def fetcher():
        calls["n"] += 1
        return _sample_df()
    scan_cache.cache_or_fetch("k2", fetcher, ttl_hours=24.0)
    assert calls["n"] == 1

    # Act: 第二次, 立即 (age ≈ 0) 调
    out = scan_cache.cache_or_fetch("k2", fetcher, ttl_hours=24.0)

    # Assert: fetcher 没再被调
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(out, _sample_df())


def test_stale_cache_triggers_refetch(_isolated_cache_dir: Path):
    # Arrange: 第一次填满 cache
    calls = {"n": 0}
    df_v1 = pd.DataFrame({"x": [1, 2, 3]})
    df_v2 = pd.DataFrame({"x": [10, 20, 30]})
    def fetcher():
        calls["n"] += 1
        return df_v2 if calls["n"] >= 2 else df_v1
    scan_cache.cache_or_fetch("k3", fetcher, ttl_hours=24.0)
    # 把 meta 的 fetched_at 改到 25 小时前
    _set_meta_age(_isolated_cache_dir, "k3", age_hours=25.0)

    # Act
    out = scan_cache.cache_or_fetch("k3", fetcher, ttl_hours=24.0)

    # Assert: fetcher 被第二次调, 返回 df_v2
    assert calls["n"] == 2
    pd.testing.assert_frame_equal(out, df_v2)


def test_fetcher_failure_with_stale_cache_returns_stale_with_warning(_isolated_cache_dir: Path):
    # Arrange
    df = _sample_df()
    scan_cache.cache_or_fetch("k4", lambda: df, ttl_hours=24.0)
    _set_meta_age(_isolated_cache_dir, "k4", age_hours=48.0)

    def failing():
        raise ConnectionError("network blocked")

    # Act / Assert
    with pytest.warns(UserWarning, match="fetcher 失败"):
        out = scan_cache.cache_or_fetch("k4", failing, ttl_hours=24.0)
    pd.testing.assert_frame_equal(out, df)


def test_fetcher_failure_no_cache_reraises(_isolated_cache_dir: Path):
    # Arrange
    def failing():
        raise ConnectionError("network blocked")

    # Act / Assert
    with pytest.raises(ConnectionError, match="network blocked"):
        scan_cache.cache_or_fetch("k5_no_cache", failing, ttl_hours=24.0)


def test_csv_serializer_roundtrip(_isolated_cache_dir: Path):
    # Arrange
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    # Act
    scan_cache.cache_or_fetch("k6", lambda: df, ttl_hours=24.0, serializer="csv")
    out = scan_cache.cache_or_fetch(
        "k6", lambda: pd.DataFrame(), ttl_hours=24.0, serializer="csv"
    )

    # Assert
    assert (_isolated_cache_dir / "k6.csv").exists()
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))


def test_invalid_ttl_raises():
    with pytest.raises(ValueError, match="ttl_hours"):
        scan_cache.cache_or_fetch("k7", lambda: pd.DataFrame(), ttl_hours=0)
    with pytest.raises(ValueError, match="ttl_hours"):
        scan_cache.cache_or_fetch("k7", lambda: pd.DataFrame(), ttl_hours=-1.0)


def test_invalid_serializer_raises():
    with pytest.raises(ValueError, match="serializer"):
        scan_cache.cache_or_fetch(
            "k8", lambda: pd.DataFrame(), ttl_hours=1.0, serializer="json",  # type: ignore[arg-type]
        )


def test_key_with_special_chars_is_sanitized(_isolated_cache_dir: Path):
    # Arrange
    df = _sample_df()

    # Act
    scan_cache.cache_or_fetch(
        "ak_stock_board:industry/cons em?symbol=白酒", lambda: df, ttl_hours=24.0
    )

    # Assert: 没有路径越权, 文件名只含 [A-Za-z0-9_.-]
    written = list(_isolated_cache_dir.glob("*.parquet"))
    assert len(written) == 1
    fname = written[0].name
    assert re.fullmatch(r"[A-Za-z0-9_.-]+\.parquet", fname), fname


def test_fetcher_must_return_dataframe(_isolated_cache_dir: Path):
    with pytest.raises(TypeError, match="DataFrame"):
        scan_cache.cache_or_fetch("k9", lambda: {"not": "df"}, ttl_hours=24.0)  # type: ignore[arg-type, return-value]


# -------------------- list_cache --------------------

def test_list_cache_returns_correct_schema(_isolated_cache_dir: Path):
    # Arrange
    scan_cache.cache_or_fetch("ka", lambda: _sample_df(), ttl_hours=24.0)
    scan_cache.cache_or_fetch("kb", lambda: _sample_df(), ttl_hours=24.0, serializer="csv")

    # Act
    out = scan_cache.list_cache()

    # Assert
    assert list(out.columns) == ["key", "fetched_at", "age_hours", "size_kb", "serializer"]
    assert set(out["key"]) == {"ka", "kb"}
    assert (out["age_hours"] >= 0).all()
    assert (out["size_kb"] > 0).all()
    assert set(out["serializer"]) == {"parquet", "csv"}


def test_list_cache_empty_dir(_isolated_cache_dir: Path):
    out = scan_cache.list_cache()
    assert out.empty
    assert list(out.columns) == ["key", "fetched_at", "age_hours", "size_kb", "serializer"]


# -------------------- clear_stale --------------------

def test_clear_stale_removes_only_older_than_threshold(_isolated_cache_dir: Path):
    # Arrange: 3 项 cache, age 分别 10h / 100h / 200h
    scan_cache.cache_or_fetch("young", lambda: _sample_df(), ttl_hours=999)
    scan_cache.cache_or_fetch("middle", lambda: _sample_df(), ttl_hours=999)
    scan_cache.cache_or_fetch("old", lambda: _sample_df(), ttl_hours=999)
    _set_meta_age(_isolated_cache_dir, "young", age_hours=10.0)
    _set_meta_age(_isolated_cache_dir, "middle", age_hours=100.0)
    _set_meta_age(_isolated_cache_dir, "old", age_hours=200.0)

    # Act: 清 > 168h 的
    n = scan_cache.clear_stale(max_age_hours=168.0)

    # Assert
    assert n == 1
    remaining_keys = set(scan_cache.list_cache()["key"])
    assert remaining_keys == {"young", "middle"}
    # 文件确实被删
    assert not (_isolated_cache_dir / "old.parquet").exists()
    assert not (_isolated_cache_dir / "old.meta.json").exists()


def test_clear_stale_exact_boundary_kept(
    _isolated_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """age == max_age_hours 的项应该保留 (我们用 '严格超过' 语义).

    精确边界 case: 用 monkeypatch 锁死 time.time(), 避免 µs 抖动.
    """
    # Arrange: 锁死 time.time() = T_NOW
    T_NOW = 1_700_000_000.0
    scan_cache.cache_or_fetch("exact", lambda: _sample_df(), ttl_hours=999)
    # 把 fetched_at 设到 T_NOW - 168h 整
    _set_meta_fetched_at(_isolated_cache_dir, "exact", T_NOW - 168.0 * 3600.0)
    monkeypatch.setattr(scan_cache.time, "time", lambda: T_NOW)

    # Act
    n = scan_cache.clear_stale(max_age_hours=168.0)

    # Assert
    assert n == 0
    assert (_isolated_cache_dir / "exact.parquet").exists()


def test_clear_stale_below_threshold_kept(_isolated_cache_dir: Path):
    # Arrange
    scan_cache.cache_or_fetch("below", lambda: _sample_df(), ttl_hours=999)
    _set_meta_age(_isolated_cache_dir, "below", age_hours=167.5)

    # Act
    n = scan_cache.clear_stale(max_age_hours=168.0)

    # Assert
    assert n == 0


def test_clear_stale_above_threshold_removed(_isolated_cache_dir: Path):
    # Arrange
    scan_cache.cache_or_fetch("above", lambda: _sample_df(), ttl_hours=999)
    _set_meta_age(_isolated_cache_dir, "above", age_hours=168.5)

    # Act
    n = scan_cache.clear_stale(max_age_hours=168.0)

    # Assert
    assert n == 1


def test_clear_stale_no_cache_dir_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Arrange: 完全不存在的目录
    monkeypatch.setattr(scan_cache, "CACHE_DIR", tmp_path / "does_not_exist")

    # Act
    n = scan_cache.clear_stale(max_age_hours=1.0)

    # Assert
    assert n == 0
