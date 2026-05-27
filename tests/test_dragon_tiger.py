"""单元测试: dragon_tiger 模块 (cache 读写, daily_features, rolling).

不依赖网络. 使用 tmp_path 隔离 cache 目录.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from claude_finance import dragon_tiger


@pytest.fixture(autouse=True)
def _isolated_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    iso = tmp_path / "dragon_tiger"
    iso.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dragon_tiger, "CACHE_DIR", iso)
    return iso


def _make_event_df(
    code: str, dates: list[str], reasons: list[str] | None = None
) -> pd.DataFrame:
    """造合成龙虎榜 df (按 SCHEMA_COLUMNS schema)."""
    n = len(dates)
    if reasons is None:
        reasons = ["日涨幅偏离值达到7%的前五只证券"] * n
    rng = np.random.default_rng(seed=hash(code) % 2**32)
    return pd.DataFrame(
        {
            "code": [str(code).zfill(6)] * n,
            "date": pd.to_datetime(dates),
            "reason": reasons,
            "net_amt": rng.normal(0, 5e7, n),
            "buy_amt": np.abs(rng.normal(1e8, 3e7, n)),
            "sell_amt": np.abs(rng.normal(1e8, 3e7, n)),
            "accum_amount": np.abs(rng.normal(5e8, 1e8, n)) + 1e8,
            "turnover_pct": rng.uniform(2, 15, n),
            "change_rate": rng.uniform(-10, 10, n),
            "fetched_at": pd.Timestamp("2026-05-25"),
        }
    )


# ---------- normalize ----------

def test_normalize_empty_returns_schema_columns() -> None:
    out = dragon_tiger._normalize(pd.DataFrame())
    assert list(out.columns) == dragon_tiger.SCHEMA_COLUMNS
    assert len(out) == 0


def test_normalize_dedupes_by_code_date_reason() -> None:
    df = _make_event_df("300059", ["2020-01-15", "2020-01-15"])
    df.loc[1, "net_amt"] = 999.0  # newer
    out = dragon_tiger._normalize(df)
    assert len(out) == 1
    assert out.iloc[0]["net_amt"] == 999.0


def test_normalize_sorts_by_date_asc() -> None:
    df = _make_event_df(
        "300059", ["2020-03-01", "2020-01-01", "2020-02-01"]
    )
    out = dragon_tiger._normalize(df)
    assert list(out["date"]) == sorted(out["date"])


def test_normalize_zero_pads_code() -> None:
    df = _make_event_df("059", ["2020-01-15"])
    df["code"] = "59"  # short
    out = dragon_tiger._normalize(df)
    assert out.iloc[0]["code"] == "000059"


# ---------- save / load ----------

def test_save_then_load_round_trip(_isolated_cache_dir: Path) -> None:
    df = _make_event_df("300059", ["2020-01-15", "2020-03-20"])
    dragon_tiger.save_cached("300059", df)
    loaded = dragon_tiger.load_cached("300059")
    assert loaded is not None
    assert len(loaded) == 2
    assert set(loaded.columns) == set(dragon_tiger.SCHEMA_COLUMNS)


def test_load_missing_returns_none() -> None:
    assert dragon_tiger.load_cached("999999") is None


def test_save_filters_to_target_code(_isolated_cache_dir: Path) -> None:
    df = _make_event_df("300059", ["2020-01-15"])
    other = _make_event_df("000001", ["2020-01-15"])
    combined = pd.concat([df, other], ignore_index=True)
    dragon_tiger.save_cached("300059", combined)
    loaded = dragon_tiger.load_cached("300059")
    assert loaded is not None
    assert (loaded["code"] == "300059").all()
    assert len(loaded) == 1


# ---------- merge_incremental ----------

def test_merge_incremental_combines_old_and_new(
    _isolated_cache_dir: Path,
) -> None:
    old = _make_event_df("300059", ["2020-01-15"])
    dragon_tiger.save_cached("300059", old)

    new = _make_event_df("300059", ["2020-03-20"])
    merged = dragon_tiger.merge_incremental("300059", new)
    assert len(merged) == 2
    assert "2020-01-15" in [str(d.date()) for d in merged["date"]]
    assert "2020-03-20" in [str(d.date()) for d in merged["date"]]


def test_merge_incremental_dedupes_overlap(
    _isolated_cache_dir: Path,
) -> None:
    old = _make_event_df("300059", ["2020-01-15"])
    old.loc[0, "net_amt"] = 100.0
    dragon_tiger.save_cached("300059", old)

    new = _make_event_df("300059", ["2020-01-15", "2020-02-01"])
    new.loc[0, "net_amt"] = 200.0  # update
    merged = dragon_tiger.merge_incremental("300059", new)
    assert len(merged) == 2  # 1 + 1, dedup overlap
    jan_row = merged[merged["date"] == pd.Timestamp("2020-01-15")]
    assert jan_row.iloc[0]["net_amt"] == 200.0  # keep new


# ---------- daily_features ----------

def test_daily_features_empty() -> None:
    out = dragon_tiger.daily_features(pd.DataFrame())
    assert "net_buy_pct" in out.columns
    assert len(out) == 0


def test_daily_features_aggregates_multi_reason() -> None:
    df = _make_event_df(
        "300059",
        ["2020-01-15", "2020-01-15"],
        reasons=["A", "B"],  # 同日两条
    )
    df.loc[0, "net_amt"] = 100.0
    df.loc[1, "net_amt"] = 200.0
    df.loc[0, "buy_amt"] = 500.0
    df.loc[1, "buy_amt"] = 700.0
    df.loc[:, "accum_amount"] = 10000.0
    agg = dragon_tiger.daily_features(df)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["net_amt"] == 300.0
    assert row["buy_amt"] == 1200.0
    assert row["n_reasons"] == 2
    assert row["net_buy_pct"] == pytest.approx(3.0)
    assert row["top_buyer_pct"] == pytest.approx(12.0)


def test_daily_features_handles_zero_accum() -> None:
    df = _make_event_df("300059", ["2020-01-15"])
    df.loc[:, "accum_amount"] = 0.0
    agg = dragon_tiger.daily_features(df)
    assert pd.isna(agg.iloc[0]["net_buy_pct"])
    assert pd.isna(agg.iloc[0]["top_buyer_pct"])


# ---------- rolling_event_features ----------

def test_rolling_event_features_dense_panel_shape() -> None:
    daily = dragon_tiger.daily_features(
        _make_event_df("300059", ["2020-01-15"])
    )
    dates = pd.date_range("2020-01-01", "2020-01-20", freq="B")
    panel = dragon_tiger.rolling_event_features(
        daily, ["300059", "000001"], dates
    )
    expected = len(dates) * 2
    assert len(panel) == expected
    assert set(panel.columns) >= {
        "date",
        "code",
        "on_list_today",
        "net_buy_pct_evt",
        "top_list_count_30d",
        "top_list_count_60d",
    }


def test_rolling_event_features_zero_fill_on_non_event_days() -> None:
    daily = dragon_tiger.daily_features(
        _make_event_df("300059", ["2020-01-15"])
    )
    dates = pd.date_range("2020-01-01", "2020-01-31", freq="B")
    panel = dragon_tiger.rolling_event_features(
        daily, ["300059"], dates
    )
    non_event = panel[panel["date"] != pd.Timestamp("2020-01-15")]
    assert (non_event["on_list_today"] == 0).all()
    assert (non_event["net_buy_pct_evt"] == 0.0).all()


def test_rolling_event_features_count_rolls() -> None:
    daily = dragon_tiger.daily_features(
        _make_event_df(
            "300059",
            ["2020-01-15", "2020-01-22", "2020-02-05"],
        )
    )
    dates = pd.date_range("2020-01-01", "2020-03-31", freq="B")
    panel = dragon_tiger.rolling_event_features(
        daily, ["300059"], dates
    )
    panel = panel[panel["code"] == "300059"].sort_values("date")
    feb5 = panel[panel["date"] == pd.Timestamp("2020-02-05")]
    assert feb5.iloc[0]["top_list_count_30d"] == 3
    assert feb5.iloc[0]["top_list_count_60d"] == 3


def test_rolling_event_features_empty_daily() -> None:
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
    panel = dragon_tiger.rolling_event_features(
        pd.DataFrame(), ["300059", "000001"], dates
    )
    assert (panel["on_list_today"] == 0).all()
    assert (panel["net_buy_pct_evt"] == 0.0).all()
    assert (panel["top_list_count_30d"] == 0.0).all()


# ---------- fetch_and_cache (mocked) ----------

def test_fetch_and_cache_skips_when_cache_recent(
    _isolated_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = _make_event_df("300059", ["2020-12-31"])
    dragon_tiger.save_cached("300059", existing)

    def _boom(*args, **kwargs):
        raise AssertionError("不该被调")

    monkeypatch.setattr(dragon_tiger, "fetch_dragon_tiger", _boom)

    out = dragon_tiger.fetch_and_cache(
        "300059", "2014-01-01", "2020-12-31", skip_if_recent=True
    )
    assert not out.empty


def test_fetch_and_cache_writes_empty_when_no_events(
    _isolated_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        dragon_tiger,
        "fetch_dragon_tiger",
        lambda *a, **kw: pd.DataFrame(
            columns=dragon_tiger.SCHEMA_COLUMNS
        ),
    )
    out = dragon_tiger.fetch_and_cache(
        "999999", "2014-01-01", "2020-12-31", skip_if_recent=False
    )
    assert out.empty
    assert dragon_tiger.cache_path("999999").exists()


# ---------- feasibility note modules ----------

def test_eps_feasibility_note_returns_string() -> None:
    from claude_finance import eps_consensus

    note = eps_consensus.is_ic_feasibility_note()
    assert isinstance(note, str)
    assert "NOT FEASIBLE" in note


def test_fund_flow_feasibility_note_returns_string() -> None:
    from claude_finance import fund_flow

    note = fund_flow.is_ic_feasibility_note()
    assert isinstance(note, str)
    assert "NOT FEASIBLE" in note
