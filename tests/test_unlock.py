"""单元测试: unlock 模块 (类型映射 / normalize / 因子计算).

不依赖网络. 使用 tmp_path 隔离 cache 文件.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from claude_finance import unlock


# -------------------- fixture --------------------


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    iso_dir = tmp_path / "unlock"
    iso_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(unlock, "CACHE_DIR", iso_dir)
    monkeypatch.setattr(unlock, "CACHE_FILE", iso_dir / "unlock_detail_em.parquet")
    return iso_dir


def _make_raw(rows: list[dict]) -> pd.DataFrame:
    """造 akshare-shaped raw frame."""
    cols = [
        "序号", "股票代码", "股票简称", "解禁时间", "限售股类型",
        "解禁数量", "实际解禁数量", "实际解禁市值", "占解禁前流通市值比例",
        "解禁前一交易日收盘价", "解禁前20日涨跌幅", "解禁后20日涨跌幅",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


# -------------------- 类型分类 --------------------


def test_classify_type_known_mappings():
    assert unlock._classify_type("首发原股东限售股份") == "ipo"
    assert unlock._classify_type("定向增发机构配售股份") == "spo"
    assert unlock._classify_type("股权激励限售股份") == "incentive"
    assert unlock._classify_type("追加承诺限售股份上市流通") == "other"


def test_classify_type_fuzzy_match():
    assert unlock._classify_type("首发战投") == "ipo"
    assert unlock._classify_type("定增机构配售股份-另增") == "spo"
    assert unlock._classify_type("股权激励解锁") == "incentive"
    assert unlock._classify_type("奇怪类型") == "other"


# -------------------- normalize --------------------


def test_normalize_empty_returns_empty_with_schema():
    out = unlock._normalize(None)
    assert out.empty
    assert set(out.columns) == {
        "code", "unlock_date", "unlock_shares", "unlock_value",
        "unlock_ratio_cap", "unlock_type", "close_before",
    }


def test_normalize_pads_code_and_parses_date():
    raw = _make_raw([{
        "股票代码": "1",
        "解禁时间": "2018-05-21",
        "限售股类型": "定向增发机构配售股份",
        "实际解禁数量": 1000.0,
        "实际解禁市值": 2e8,
        "占解禁前流通市值比例": 0.025,
        "解禁前一交易日收盘价": 10.5,
    }])
    out = unlock._normalize(raw)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["code"] == "000001"
    assert row["unlock_date"] == pd.Timestamp("2018-05-21")
    assert row["unlock_type"] == "spo"
    assert row["unlock_value"] == pytest.approx(2e8)
    assert row["unlock_ratio_cap"] == pytest.approx(0.025)


def test_normalize_drops_invalid_date():
    raw = _make_raw([
        {"股票代码": "000001", "解禁时间": "bogus", "限售股类型": "首发",
         "实际解禁数量": 1, "实际解禁市值": 1, "占解禁前流通市值比例": 0.01},
        {"股票代码": "000002", "解禁时间": "2020-01-02", "限售股类型": "首发",
         "实际解禁数量": 1, "实际解禁市值": 1, "占解禁前流通市值比例": 0.01},
    ])
    out = unlock._normalize(raw)
    assert len(out) == 1
    assert out.iloc[0]["code"] == "000002"


# -------------------- cache I/O --------------------


def test_save_load_roundtrip():
    raw = _make_raw([
        {"股票代码": "600519", "解禁时间": "2014-08-01",
         "限售股类型": "首发原股东限售股份",
         "实际解禁数量": 1e6, "实际解禁市值": 5e8,
         "占解禁前流通市值比例": 0.04, "解禁前一交易日收盘价": 200.0},
        {"股票代码": "000001", "解禁时间": "2014-09-01",
         "限售股类型": "定向增发机构配售股份",
         "实际解禁数量": 2e6, "实际解禁市值": 6e8,
         "占解禁前流通市值比例": 0.03, "解禁前一交易日收盘价": 10.0},
    ])
    df = unlock._normalize(raw)
    unlock.save_cache(df)
    loaded = unlock.load_cache()
    assert len(loaded) == 2
    assert set(loaded["code"]) == {"600519", "000001"}
    assert loaded.iloc[0]["code"] == "000001"  # 排序后字典序


def test_merge_dedupes_on_key():
    raw1 = _make_raw([{
        "股票代码": "000001", "解禁时间": "2014-09-01",
        "限售股类型": "定向增发机构配售股份",
        "实际解禁数量": 1000, "实际解禁市值": 5e7,
        "占解禁前流通市值比例": 0.02, "解禁前一交易日收盘价": 9.0,
    }])
    raw2 = _make_raw([{
        "股票代码": "000001", "解禁时间": "2014-09-01",
        "限售股类型": "定向增发机构配售股份",
        "实际解禁数量": 1000, "实际解禁市值": 5e7,
        "占解禁前流通市值比例": 0.02, "解禁前一交易日收盘价": 9.0,
    }])
    unlock.merge_into_cache(unlock._normalize(raw1))
    final = unlock.merge_into_cache(unlock._normalize(raw2))
    assert len(final) == 1


# -------------------- 因子计算 --------------------


def _toy_unlock_df() -> pd.DataFrame:
    return pd.DataFrame({
        "code": ["000001", "000001", "000001", "600036"],
        "unlock_date": pd.to_datetime([
            "2018-03-05", "2018-03-15", "2018-05-01", "2018-04-01",
        ]),
        "unlock_shares": [1e6, 2e6, 3e6, 5e6],
        "unlock_value": [1e8, 2e8, 3e8, 5e8],
        "unlock_ratio_cap": [0.01, 0.02, 0.03, 0.05],
        "unlock_type": ["spo", "spo", "ipo", "ipo"],
        "close_before": [10.0, 11.0, 12.0, 30.0],
    })


def test_forward_unlock_metrics_5d_20d_60d_windows():
    df = _toy_unlock_df()
    # T=2018-03-01: +5d=03-06 含 03-05; +20d=03-21 含 03-05+03-15;
    # +60d=04-30 含 03-05+03-15 (05-01 边界外)
    m = unlock.forward_unlock_metrics(
        df, "000001", pd.Timestamp("2018-03-01"), windows_days=(5, 20, 60),
    )
    assert m["unlock_pct_next_5"] == pytest.approx(0.01)
    assert m["unlock_value_next_5"] == pytest.approx(1.0)
    assert m["unlock_imminent_5"] == 1.0
    assert m["unlock_pct_next_20"] == pytest.approx(0.03)
    assert m["unlock_value_next_20"] == pytest.approx(3.0)
    assert m["unlock_pct_next_60"] == pytest.approx(0.03)


def test_forward_unlock_metrics_empty_returns_zero():
    df = _toy_unlock_df()
    m = unlock.forward_unlock_metrics(
        df, "600519", pd.Timestamp("2018-03-01"), windows_days=(5, 20),
    )
    assert m["unlock_pct_next_5"] == 0.0
    assert m["unlock_imminent_5"] == 0.0
    assert m["unlock_pct_next_20"] == 0.0
    assert m["unlock_imminent_20"] == 0.0


def test_forward_unlock_metrics_strictly_forward_no_lookback():
    df = _toy_unlock_df()
    # T=2018-04-01: 03-05/03-15 在过去, 应忽略; 05-01 在 +60d (30d 后)
    m = unlock.forward_unlock_metrics(
        df, "000001", pd.Timestamp("2018-04-01"), windows_days=(5, 20, 60),
    )
    assert m["unlock_pct_next_5"] == 0.0
    assert m["unlock_pct_next_20"] == 0.0
    assert m["unlock_pct_next_60"] == pytest.approx(0.03)


def test_build_factor_panel_shape_and_zero_for_unknown_code():
    df = _toy_unlock_df()
    codes = ["000001", "600036", "600519"]
    asof_dates = [pd.Timestamp("2018-03-01"), pd.Timestamp("2018-04-01")]
    panel = unlock.build_factor_panel(
        df, codes, asof_dates, windows_days=(5, 20),
    )
    assert len(panel) == 6  # 2 dates × 3 codes
    assert set(panel["code"]) == {"000001", "600036", "600519"}
    # 600519 无事件
    row = panel[(panel["code"] == "600519")].iloc[0]
    assert row["unlock_pct_next_5"] == 0.0
    assert row["unlock_pct_next_20"] == 0.0
    # 600036 04-01 同日 (strict >) → 0
    row = panel[(panel["code"] == "600036") &
                (panel["asof_date"] == pd.Timestamp("2018-04-01"))].iloc[0]
    assert row["unlock_pct_next_5"] == 0.0


def test_filter_codes_zfills_and_filters():
    df = _toy_unlock_df()
    out = unlock.filter_codes(df, [1, "600036"])  # 1 → 000001
    assert set(out["code"]) == {"000001", "600036"}
    assert len(out) == 4
