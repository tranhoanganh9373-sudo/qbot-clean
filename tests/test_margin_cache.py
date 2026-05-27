"""单元测试: margin_cache 模块 (per-stock parquet 读写/合并/bootstrap).

不依赖网络. 使用 tmp_path 隔离 cache 目录, 避免污染真实 data_cache/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from claude_finance import margin_cache


# -------------------- 通用 fixture --------------------

@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """每个测试用独立 tmp_path, monkeypatch CACHE_DIR, 保证测试间互不影响."""
    iso = tmp_path / "margin_eastmoney"
    iso.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(margin_cache, "CACHE_DIR", iso)
    return iso


def _make_df(code: str, n_days: int, start: str = "2024-01-01",
             seed: int = 0) -> pd.DataFrame:
    """造一个 synthetic margin df (含 code/date/rzye/rzmre/rzche)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="B")  # business days
    rzye = 1e8 + np.cumsum(rng.normal(0, 1e6, n_days))
    rzmre = np.abs(rng.normal(5e6, 1e6, n_days))
    rzche = np.abs(rng.normal(5e6, 1e6, n_days))
    return pd.DataFrame({
        "code": [str(code).zfill(6)] * n_days,
        "date": dates,
        "rzye": rzye,
        "rzmre": rzmre,
        "rzche": rzche,
    })


# -------------------- save_cached / load_cached 来回一致 --------------------

def test_save_and_load_roundtrip_preserves_data():
    # Arrange
    code = "000001"
    df = _make_df(code, n_days=10)

    # Act
    margin_cache.save_cached(code, df)
    loaded = margin_cache.load_cached(code)

    # Assert
    assert loaded is not None
    assert len(loaded) == 10
    assert list(loaded["code"].unique()) == [code]
    np.testing.assert_array_almost_equal(
        loaded["rzye"].to_numpy(), df["rzye"].to_numpy(),
    )
    np.testing.assert_array_almost_equal(
        loaded["rzmre"].to_numpy(), df["rzmre"].to_numpy(),
    )
    np.testing.assert_array_almost_equal(
        loaded["rzche"].to_numpy(), df["rzche"].to_numpy(),
    )
    assert loaded["date"].is_monotonic_increasing


def test_load_cached_returns_none_when_missing():
    # Arrange (no save)
    code = "999999"

    # Act
    result = margin_cache.load_cached(code)

    # Assert
    assert result is None


def test_save_idempotent_overwrite():
    """对同一 code 多次 save 后 load, 结果完全一致 (含 dtype)."""
    # Arrange
    code = "000002"
    df = _make_df(code, n_days=20, seed=42)

    # Act
    margin_cache.save_cached(code, df)
    first = margin_cache.load_cached(code)
    margin_cache.save_cached(code, df)
    second = margin_cache.load_cached(code)

    # Assert
    pd.testing.assert_frame_equal(first, second, check_dtype=True)


# -------------------- merge_incremental 去重 + 重算 --------------------

def test_merge_incremental_dedupes_and_extends():
    """old 100 天 + new 含 50 天 overlap + 30 天新 → 130 天."""
    # Arrange
    code = "000003"
    old = _make_df(code, n_days=100, start="2024-01-01", seed=1)
    margin_cache.save_cached(code, old)
    overlap_start = old["date"].iloc[50]
    new = _make_df(
        code,
        n_days=80,
        start=overlap_start.strftime("%Y-%m-%d"),
        seed=2,
    )

    # Act
    merged = margin_cache.merge_incremental(code, new)

    # Assert: dedupe → 50 (old 前段) + 80 (new) = 130
    assert len(merged) == 130
    assert merged["date"].is_monotonic_increasing
    assert merged["code"].nunique() == 1
    loaded = margin_cache.load_cached(code)
    assert len(loaded) == 130


def test_merge_incremental_recomputes_5d_and_20d_chg():
    """合并后 margin_5d_chg / margin_20d_chg = rzye 的 pct_change(5/20)."""
    # Arrange
    code = "000004"
    df = _make_df(code, n_days=40, seed=7)

    # Act
    merged = margin_cache.merge_incremental(code, df)

    # Assert: 整列等于手算 pct_change
    expected_5d = merged["rzye"].pct_change(5)
    expected_20d = merged["rzye"].pct_change(20)
    np.testing.assert_array_almost_equal(
        merged["margin_5d_chg"].to_numpy(),
        expected_5d.to_numpy(),
    )
    np.testing.assert_array_almost_equal(
        merged["margin_20d_chg"].to_numpy(),
        expected_20d.to_numpy(),
    )


def test_merge_incremental_no_existing_cache():
    """没有 old 时, merge_incremental 等价于 save."""
    # Arrange
    code = "000005"
    new = _make_df(code, n_days=15, seed=3)

    # Act
    merged = margin_cache.merge_incremental(code, new)

    # Assert
    assert len(merged) == 15
    loaded = margin_cache.load_cached(code)
    assert loaded is not None
    assert len(loaded) == 15


# -------------------- needs_refresh 三种 case --------------------

def test_needs_refresh_no_cache_returns_true():
    # Arrange
    code = "888888"
    today = pd.Timestamp("2026-05-24")

    # Act
    result = margin_cache.needs_refresh(code, today)

    # Assert
    assert result is True


def test_needs_refresh_stale_cache_returns_true():
    # Arrange: cache 末尾是 2024-01-01, today 是 2026-05-24
    code = "777777"
    old = _make_df(code, n_days=10, start="2024-01-01")
    margin_cache.save_cached(code, old)
    today = pd.Timestamp("2026-05-24")

    # Act
    result = margin_cache.needs_refresh(code, today)

    # Assert
    assert result is True


def test_needs_refresh_fresh_cache_returns_false():
    # Arrange: cache 末尾就是 today (用一个交易日 Wed 2026-05-20 避免 bdate_range 退到周五)
    code = "666666"
    today = pd.Timestamp("2026-05-20")
    dates = pd.bdate_range(end=today, periods=10)
    df = pd.DataFrame({
        "code": [code] * 10,
        "date": dates,
        "rzye": np.linspace(1e8, 1.1e8, 10),
        "rzmre": np.ones(10) * 5e6,
        "rzche": np.ones(10) * 5e6,
    })
    margin_cache.save_cached(code, df)

    # Act
    result = margin_cache.needs_refresh(code, today)

    # Assert
    assert result is False


# -------------------- bootstrap_from_bulk_parquet --------------------

def test_bootstrap_from_bulk_parquet_splits_into_per_stock(tmp_path: Path):
    """synthetic bulk df (3 股 × 100 天) → 3 个 per-stock cache 文件."""
    # Arrange
    codes = ["100001", "100002", "100003"]
    frames = [_make_df(c, n_days=100, seed=int(c)) for c in codes]
    bulk = pd.concat(frames, ignore_index=True)
    bulk_path = tmp_path / "bulk.parquet"
    bulk.to_parquet(bulk_path, index=False)

    # Act
    n = margin_cache.bootstrap_from_bulk_parquet(bulk_path)

    # Assert
    assert n == 3
    for c in codes:
        assert margin_cache.cache_path(c).exists()
        loaded = margin_cache.load_cached(c)
        assert loaded is not None
        assert len(loaded) == 100
        assert (loaded["code"] == c).all()


def test_bootstrap_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        margin_cache.bootstrap_from_bulk_parquet(tmp_path / "nope.parquet")


def test_bootstrap_bulk_missing_columns_raises(tmp_path: Path):
    # Arrange: bulk 缺 rzye 列
    bad = pd.DataFrame({
        "code": ["000001"] * 5,
        "date": pd.date_range("2024-01-01", periods=5, freq="B"),
        "rzmre": [1.0] * 5,
        "rzche": [1.0] * 5,
    })
    p = tmp_path / "bad.parquet"
    bad.to_parquet(p, index=False)

    # Act / Assert
    with pytest.raises(ValueError, match="缺列"):
        margin_cache.bootstrap_from_bulk_parquet(p)


# -------------------- load_many 批量读 --------------------

def test_load_many_returns_long_df_for_existing_codes():
    # Arrange
    margin_cache.save_cached("000010", _make_df("000010", n_days=30, seed=10))
    margin_cache.save_cached("000011", _make_df("000011", n_days=40, seed=11))

    # Act
    out = margin_cache.load_many(["000010", "000011", "999998"])  # 1 个 miss

    # Assert
    assert set(out["code"].unique()) == {"000010", "000011"}
    assert len(out) == 70  # 30 + 40
    assert out.sort_values(["code", "date"]).reset_index(drop=True).equals(out)


def test_load_many_empty_when_all_missing():
    # Act
    out = margin_cache.load_many(["999996", "999997"])

    # Assert
    assert out.empty
    assert list(out.columns) == margin_cache.SCHEMA_COLUMNS
