"""单元测试: fundamentals 模块 (per-stock parquet 读写 + point-in-time 对齐).

不依赖网络. 使用 tmp_path 隔离 cache 目录, 复用 test_margin_cache 模式.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from claude_finance import fundamentals


# -------------------- fixtures --------------------

@pytest.fixture(autouse=True)
def _isolated_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    iso = tmp_path / "fundamentals"
    iso.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fundamentals, "CACHE_DIR", iso)
    return iso


def _make_df(
    code: str,
    n_quarters: int,
    start: str = "2014-03-31",
    seed: int = 0,
) -> pd.DataFrame:
    """构造 synthetic quarterly fundamentals df (schema 完整)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_quarters, freq="QE")
    base = {
        "code": [str(code).zfill(6)] * n_quarters,
        "report_date": dates,
        "roe": rng.uniform(5, 25, n_quarters),
        "net_margin": rng.uniform(5, 50, n_quarters),
        "gross_margin": rng.uniform(20, 90, n_quarters),
        "debt_to_asset": rng.uniform(10, 70, n_quarters),
        "current_ratio": rng.uniform(0.5, 5, n_quarters),
        "quick_ratio": rng.uniform(0.3, 4, n_quarters),
        "revenue_yoy": rng.normal(10, 20, n_quarters),
        "net_profit_yoy": rng.normal(10, 30, n_quarters),
        "eps_basic": rng.uniform(0.1, 5, n_quarters),
        "bps": rng.uniform(2, 50, n_quarters),
        "ocf_per_share": rng.uniform(-1, 5, n_quarters),
        "roa": rng.uniform(1, 15, n_quarters),
        "fetched_at": [pd.Timestamp("2026-05-25 12:00:00")] * n_quarters,
    }
    return pd.DataFrame(base)


# -------------------- save / load roundtrip --------------------

def test_save_and_load_roundtrip_preserves_data():
    # Arrange
    code = "600519"
    df = _make_df(code, n_quarters=20)

    # Act
    fundamentals.save_cached(code, df)
    loaded = fundamentals.load_cached(code)

    # Assert
    assert loaded is not None
    assert len(loaded) == 20
    assert list(loaded["code"].unique()) == [code]
    np.testing.assert_array_almost_equal(
        loaded["roe"].to_numpy(), df["roe"].to_numpy(),
    )
    np.testing.assert_array_almost_equal(
        loaded["debt_to_asset"].to_numpy(), df["debt_to_asset"].to_numpy(),
    )
    assert loaded["report_date"].is_monotonic_increasing


def test_load_cached_returns_none_when_missing():
    assert fundamentals.load_cached("999999") is None


def test_save_dedupes_duplicate_report_dates():
    """同股同 report_date 出现 2 次 → 保留 last."""
    # Arrange
    code = "000001"
    df1 = _make_df(code, n_quarters=4, seed=1)
    df2 = _make_df(code, n_quarters=4, seed=2)  # 同样 4 个季度
    combined = pd.concat([df1, df2], ignore_index=True)

    # Act
    fundamentals.save_cached(code, combined)
    loaded = fundamentals.load_cached(code)

    # Assert: 应保留 4 行 (dedupe), 且 last 优先 (seed=2 数据)
    assert len(loaded) == 4
    np.testing.assert_array_almost_equal(
        loaded["roe"].to_numpy(), df2["roe"].to_numpy(),
    )


# -------------------- needs_refresh 三种 case --------------------

def test_needs_refresh_no_cache_returns_true():
    assert fundamentals.needs_refresh("999999", pd.Timestamp("2026-05-25"))


def test_needs_refresh_stale_cache_returns_true():
    """cache 末尾 2024-03-31, today 2026-05-25, stale_days=90 → True."""
    # Arrange
    code = "777777"
    df = _make_df(code, n_quarters=4, start="2023-06-30")  # 末尾 2024-03-31
    fundamentals.save_cached(code, df)

    # Act
    result = fundamentals.needs_refresh(
        code, pd.Timestamp("2026-05-25"), stale_days=90,
    )

    # Assert
    assert result is True


def test_needs_refresh_fresh_cache_returns_false():
    """cache 末尾 2026-03-31, today 2026-05-25, 55 天 < 90 → False."""
    # Arrange
    code = "666666"
    df = _make_df(code, n_quarters=8, start="2024-06-30")  # 末尾 2026-03-31
    fundamentals.save_cached(code, df)

    # Act
    result = fundamentals.needs_refresh(
        code, pd.Timestamp("2026-05-25"), stale_days=90,
    )

    # Assert
    assert result is False


# -------------------- get_quarterly_at_date 边界 --------------------

def test_get_quarterly_at_date_returns_none_if_no_cache():
    out = fundamentals.get_quarterly_at_date(
        "999999", pd.Timestamp("2020-01-01"),
    )
    assert out is None


def test_get_quarterly_at_date_uses_prior_report_when_query_too_early():
    """query=2014-08-15, lag=60 → cutoff=2014-06-16.
    应取 2014-03-31 (≤ cutoff); 2014-06-30 不可见 (> cutoff)."""
    # Arrange
    code = "600519"
    df = _make_df(code, n_quarters=12, start="2014-03-31")
    fundamentals.save_cached(code, df)

    # Act
    res = fundamentals.get_quarterly_at_date(
        code, pd.Timestamp("2014-08-15"), announce_lag_days=60,
    )

    # Assert
    assert res is not None
    assert pd.Timestamp(res["report_date"]) == pd.Timestamp("2014-03-31")


def test_get_quarterly_at_date_returns_none_when_query_before_first_report():
    """查询日早于所有可见季报 → None."""
    # Arrange
    code = "600519"
    df = _make_df(code, n_quarters=4, start="2020-03-31")
    fundamentals.save_cached(code, df)

    # Act: query=2019-01-01, 所有 report_date 都 > cutoff → None
    res = fundamentals.get_quarterly_at_date(
        code, pd.Timestamp("2019-01-01"), announce_lag_days=60,
    )

    # Assert
    assert res is None


def test_get_quarterly_at_date_returns_latest_visible():
    """查询日很晚 → 返回最新季报."""
    # Arrange
    code = "600519"
    df = _make_df(code, n_quarters=8, start="2014-03-31")  # 末尾 2015-12-31
    fundamentals.save_cached(code, df)

    # Act: query=2026-01-01, 所有都可见, 应取 max(report_date)
    res = fundamentals.get_quarterly_at_date(
        code, pd.Timestamp("2026-01-01"), announce_lag_days=60,
    )

    # Assert
    assert res is not None
    assert pd.Timestamp(res["report_date"]) == pd.Timestamp("2015-12-31")


def test_get_quarterly_at_date_respects_announce_lag():
    """同 query, lag=0 vs lag=60 → 不同结果."""
    # Arrange
    code = "600519"
    df = _make_df(code, n_quarters=8, start="2014-03-31")
    fundamentals.save_cached(code, df)

    # Act:
    # lag=0  → cutoff=2014-07-01 → 可见 2014-06-30
    # lag=60 → cutoff=2014-05-02 → 只可见 2014-03-31
    res_lag0 = fundamentals.get_quarterly_at_date(
        code, pd.Timestamp("2014-07-01"), announce_lag_days=0,
    )
    res_lag60 = fundamentals.get_quarterly_at_date(
        code, pd.Timestamp("2014-07-01"), announce_lag_days=60,
    )

    # Assert
    assert pd.Timestamp(res_lag0["report_date"]) == pd.Timestamp("2014-06-30")
    assert pd.Timestamp(res_lag60["report_date"]) == pd.Timestamp("2014-03-31")


def test_get_quarterly_at_date_with_explicit_df_skips_cache():
    """传 df 参数 → 用 df 不读 cache."""
    # Arrange
    code = "888888"  # 没 cache
    df = _make_df(code, n_quarters=4, start="2020-03-31")

    # Act
    res = fundamentals.get_quarterly_at_date(
        code, pd.Timestamp("2026-01-01"), df=df, announce_lag_days=60,
    )

    # Assert
    assert res is not None
    assert pd.Timestamp(res["report_date"]) == pd.Timestamp("2020-12-31")


# -------------------- merge_incremental --------------------

def test_merge_incremental_extends_cache():
    # Arrange
    code = "300001"
    old = _make_df(code, n_quarters=4, start="2014-03-31", seed=1)
    fundamentals.save_cached(code, old)
    new = _make_df(code, n_quarters=4, start="2015-03-31", seed=2)

    # Act
    merged = fundamentals.merge_incremental(code, new)

    # Assert
    assert len(merged) == 8
    assert merged["report_date"].is_monotonic_increasing


def test_merge_incremental_dedupes_overlap():
    """8 old + 8 new 重叠 4 → 12."""
    # Arrange
    code = "300002"
    old = _make_df(code, n_quarters=8, start="2014-03-31", seed=1)
    fundamentals.save_cached(code, old)
    new = _make_df(code, n_quarters=8, start="2015-03-31", seed=2)

    # Act
    merged = fundamentals.merge_incremental(code, new)

    # Assert
    assert len(merged) == 12


# -------------------- _map_em_df 字段映射 --------------------

def test_map_em_df_translates_fields():
    """EM 原始列名 → 本模块字段名."""
    # Arrange
    em = pd.DataFrame({
        "REPORT_DATE": ["2014-03-31", "2014-06-30"],
        "ROEJQ": [7.99, 15.97],
        "XSJLL": [53.06, 53.52],
        "XSMLL": [93.20, 93.10],
        "ZCFZL": [15.31, 15.56],
        "LD": [4.73, 4.51],
        "SD": [3.24, 2.96],
        "TOTALOPERATEREVETZ": [3.96, 1.37],
        "PARENTNETPROFITTZ": [4.76, 0.57],
        "EPSJB": [3.81, 6.71],
        "BPS": [44.59, 39.65],
        "MGJYXJJE": [0.17, 3.76],
        "ZZCJLL": [7.05, 13.88],
    })

    # Act
    out = fundamentals._map_em_df("600519", em)

    # Assert
    assert len(out) == 2
    assert list(out["code"]) == ["600519", "600519"]
    assert out.loc[0, "roe"] == pytest.approx(7.99)
    assert out.loc[1, "net_margin"] == pytest.approx(53.52)
    assert out.loc[0, "roa"] == pytest.approx(7.05)
    assert pd.Timestamp(out.loc[0, "report_date"]) == pd.Timestamp("2014-03-31")


def test_map_em_df_handles_missing_columns():
    """EM 缺一些列 → 对应字段 NaN, 不抛."""
    # Arrange
    em = pd.DataFrame({
        "REPORT_DATE": ["2014-03-31"],
        "ROEJQ": [7.99],
    })

    # Act
    out = fundamentals._map_em_df("600519", em)

    # Assert
    assert len(out) == 1
    assert out.loc[0, "roe"] == pytest.approx(7.99)
    assert pd.isna(out.loc[0, "net_margin"])


def test_map_em_df_empty_input():
    out = fundamentals._map_em_df("600519", pd.DataFrame())
    assert out.empty
    assert list(out.columns) == fundamentals.SCHEMA_COLUMNS


def test_to_em_symbol_routes_market_correctly():
    """600519 → .SH; 000001 → .SZ; 300001 → .SZ."""
    assert fundamentals._to_em_symbol("600519") == "600519.SH"
    assert fundamentals._to_em_symbol("000001") == "000001.SZ"
    assert fundamentals._to_em_symbol("300001") == "300001.SZ"
    assert fundamentals._to_em_symbol("688981") == "688981.SH"


# -------------------- load_many 批量 --------------------

def test_load_many_returns_long_df():
    # Arrange
    codes = ["100001", "100002", "100003"]
    for c in codes:
        df = _make_df(c, n_quarters=4, seed=int(c) % 100)
        fundamentals.save_cached(c, df)

    # Act
    out = fundamentals.load_many(codes)

    # Assert
    assert len(out) == 12
    assert set(out["code"].unique()) == set(codes)


def test_load_many_skips_missing_codes():
    # Arrange
    fundamentals.save_cached("100001", _make_df("100001", 4))
    # 100002 不写

    # Act
    out = fundamentals.load_many(["100001", "100002"])

    # Assert
    assert len(out) == 4
    assert list(out["code"].unique()) == ["100001"]
