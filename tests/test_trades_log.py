"""Tests for src/claude_finance/trades_log.py."""
from __future__ import annotations

import json

import pytest

from claude_finance.trades_log import (
    TradeValidationError,
    aggregate_positions,
    append_trade,
    get_trades_for_date,
    load_trades,
)


@pytest.fixture
def tmp_jsonl(tmp_path):
    return tmp_path / "trades.jsonl"


# ────────────────────── validation ──────────────────────


def test_append_valid_trade_returns_normalized(tmp_jsonl):
    t = {"date": "2026-05-26", "sym": "SH600547", "action": "BUY",
         "price": 29.95, "shares": 200, "note": "开仓"}
    out = append_trade(t, path=tmp_jsonl)
    assert out["id"]
    assert out["recorded_at"]
    assert out["price"] == 29.95
    assert out["shares"] == 200
    assert out["name"] == ""


def test_append_rejects_missing_required(tmp_jsonl):
    with pytest.raises(TradeValidationError, match="missing required field: price"):
        append_trade({"date": "2026-05-26", "sym": "SH600547",
                      "action": "BUY", "shares": 100}, path=tmp_jsonl)


def test_append_rejects_bad_action(tmp_jsonl):
    with pytest.raises(TradeValidationError, match="action must be one of"):
        append_trade({"date": "2026-05-26", "sym": "SH600547", "action": "HOLD",
                      "price": 30, "shares": 100}, path=tmp_jsonl)


def test_append_rejects_negative_price(tmp_jsonl):
    with pytest.raises(TradeValidationError, match="price must be > 0"):
        append_trade({"date": "2026-05-26", "sym": "SH600547", "action": "BUY",
                      "price": -1, "shares": 100}, path=tmp_jsonl)


def test_append_rejects_zero_shares(tmp_jsonl):
    with pytest.raises(TradeValidationError, match="shares must be > 0"):
        append_trade({"date": "2026-05-26", "sym": "SH600547", "action": "BUY",
                      "price": 30, "shares": 0}, path=tmp_jsonl)


def test_append_rejects_bad_date(tmp_jsonl):
    with pytest.raises(TradeValidationError, match="date must be YYYY-MM-DD"):
        append_trade({"date": "20260526", "sym": "SH600547", "action": "BUY",
                      "price": 30, "shares": 100}, path=tmp_jsonl)


def test_append_rejects_bad_sym(tmp_jsonl):
    with pytest.raises(TradeValidationError, match="sym must be SH/SZ"):
        append_trade({"date": "2026-05-26", "sym": "600547", "action": "BUY",
                      "price": 30, "shares": 100}, path=tmp_jsonl)


# ────────────────────── persistence ──────────────────────


def test_append_then_load_roundtrip(tmp_jsonl):
    t = {"date": "2026-05-26", "sym": "SH600547", "action": "BUY",
         "price": 29.95, "shares": 200, "name": "山东黄金"}
    append_trade(t, path=tmp_jsonl)
    loaded = load_trades(path=tmp_jsonl)
    assert len(loaded) == 1
    assert loaded[0]["sym"] == "SH600547"
    assert loaded[0]["price"] == 29.95


def test_load_empty_file_returns_empty(tmp_path):
    assert load_trades(path=tmp_path / "missing.jsonl") == []


def test_load_skips_malformed_lines(tmp_jsonl):
    tmp_jsonl.write_text(
        '{"id":"a","date":"2026-05-26","sym":"SH600547","action":"BUY","price":30,'
        '"shares":100,"recorded_at":"2026-05-26T01:00:00+00:00"}\n'
        "not-a-json-line\n"
        '{"id":"b","date":"2026-05-26","sym":"SZ300347","action":"BUY","price":42,'
        '"shares":200,"recorded_at":"2026-05-26T02:00:00+00:00"}\n'
    )
    loaded = load_trades(path=tmp_jsonl)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "a"


def test_load_sorts_by_recorded_at(tmp_jsonl):
    tmp_jsonl.write_text(
        '{"id":"z","sym":"SH600547","action":"BUY","price":30,"shares":100,'
        '"date":"2026-05-26","recorded_at":"2026-05-26T05:00:00+00:00"}\n'
        '{"id":"a","sym":"SH600547","action":"BUY","price":31,"shares":100,'
        '"date":"2026-05-26","recorded_at":"2026-05-26T01:00:00+00:00"}\n'
    )
    loaded = load_trades(path=tmp_jsonl)
    assert loaded[0]["id"] == "a"
    assert loaded[1]["id"] == "z"


def test_get_trades_for_date_filters(tmp_jsonl):
    append_trade({"date": "2026-05-26", "sym": "SH600547", "action": "BUY",
                  "price": 30, "shares": 100}, path=tmp_jsonl)
    append_trade({"date": "2026-05-27", "sym": "SH600547", "action": "BUY",
                  "price": 31, "shares": 100}, path=tmp_jsonl)
    on_26 = get_trades_for_date("2026-05-26", path=tmp_jsonl)
    assert len(on_26) == 1
    assert on_26[0]["price"] == 30


# ────────────────────── aggregation ──────────────────────


def _t(sym, action, price, shares, date="2026-05-26", recorded_at=None):
    return {
        "id": f"{sym}-{action}-{price}-{shares}",
        "sym": sym, "action": action,
        "price": price, "shares": shares, "date": date,
        "recorded_at": recorded_at or f"2026-05-26T{shares:02d}:00:00+00:00",
    }


def test_aggregate_single_buy():
    trades = [_t("SH600547", "BUY", 29.95, 200)]
    pos = aggregate_positions(trades)["SH600547"]
    assert pos["net_shares"] == 200
    assert pos["weighted_avg_cost"] == 29.95
    assert pos["total_buy_cost"] == 5990.00
    assert pos["status"] == "holding"


def test_aggregate_two_buys_weighted_avg():
    """Buy 200@29.95 + Buy 100@30.10 → WAC = (200*29.95 + 100*30.10)/300."""
    trades = [
        _t("SH600547", "BUY", 29.95, 200, recorded_at="2026-05-26T09:30:00+00:00"),
        _t("SH600547", "BUY", 30.10, 100, recorded_at="2026-05-26T11:00:00+00:00"),
    ]
    pos = aggregate_positions(trades)["SH600547"]
    assert pos["net_shares"] == 300
    expected_avg = (200 * 29.95 + 100 * 30.10) / 300
    assert pos["weighted_avg_cost"] == round(expected_avg, 4)
    assert pos["total_buy_shares"] == 300


def test_aggregate_buy_then_partial_sell_realized_pnl():
    """Buy 200@29.95, Sell 150@30.50 → realized = (30.50-29.95)*150 = 82.50."""
    trades = [
        _t("SH600547", "BUY", 29.95, 200, recorded_at="2026-05-26T09:30:00+00:00"),
        _t("SH600547", "SELL", 30.50, 150, recorded_at="2026-05-26T14:00:00+00:00"),
    ]
    pos = aggregate_positions(trades)["SH600547"]
    assert pos["net_shares"] == 50
    assert pos["realized_pnl"] == 82.50
    assert pos["weighted_avg_cost"] == 29.95
    assert pos["status"] == "holding"


def test_aggregate_full_close():
    trades = [
        _t("SH600547", "BUY", 29.95, 200, recorded_at="2026-05-26T09:30:00+00:00"),
        _t("SH600547", "SELL", 30.50, 200, recorded_at="2026-05-26T14:00:00+00:00"),
    ]
    pos = aggregate_positions(trades)["SH600547"]
    assert pos["net_shares"] == 0
    assert pos["realized_pnl"] == 110.00
    assert pos["status"] == "closed"


def test_aggregate_multi_day_intraday_with_re_entry():
    """Buy 200@30 → Sell 200@31 (close) → Buy 100@28 (re-entry).
       重入仓后 avg 重置 = 28 (不是 30 的延续)."""
    trades = [
        _t("SH600547", "BUY", 30.0, 200, recorded_at="2026-05-26T09:30:00+00:00"),
        _t("SH600547", "SELL", 31.0, 200, recorded_at="2026-05-26T14:00:00+00:00"),
        _t("SH600547", "BUY", 28.0, 100, recorded_at="2026-05-27T09:30:00+00:00"),
    ]
    pos = aggregate_positions(trades)["SH600547"]
    assert pos["net_shares"] == 100
    assert pos["weighted_avg_cost"] == 28.0
    assert pos["realized_pnl"] == 200.0
    assert pos["status"] == "holding"


def test_aggregate_multi_sym_isolated():
    trades = [
        _t("SH600547", "BUY", 30, 100, recorded_at="2026-05-26T09:00:00+00:00"),
        _t("SZ300347", "BUY", 42, 200, recorded_at="2026-05-26T10:00:00+00:00"),
    ]
    agg = aggregate_positions(trades)
    assert set(agg.keys()) == {"SH600547", "SZ300347"}
    assert agg["SH600547"]["net_shares"] == 100
    assert agg["SZ300347"]["net_shares"] == 200


def test_aggregate_empty_returns_empty():
    assert aggregate_positions([]) == {}


def test_aggregate_oversold_marks_short():
    """异常 SELL > 持仓 → net_shares < 0, status='short'."""
    trades = [_t("SH600547", "SELL", 30, 200)]
    pos = aggregate_positions(trades)["SH600547"]
    assert pos["net_shares"] == -200
    assert pos["status"] == "short"
    assert pos["realized_pnl"] == 0.0


def test_aggregate_trade_count_tracks():
    trades = [
        _t("SH600547", "BUY", 30, 100, recorded_at="2026-05-26T01:00:00+00:00"),
        _t("SH600547", "BUY", 31, 100, recorded_at="2026-05-26T02:00:00+00:00"),
        _t("SH600547", "SELL", 32, 50, recorded_at="2026-05-26T03:00:00+00:00"),
    ]
    pos = aggregate_positions(trades)["SH600547"]
    assert pos["trade_count"] == 3
    assert pos["total_buy_shares"] == 200
    assert pos["total_sell_shares"] == 50


# ────────────────────── file ops ──────────────────────


def test_append_creates_parent_dir(tmp_path):
    nested = tmp_path / "nested" / "deep" / "trades.jsonl"
    append_trade({"date": "2026-05-26", "sym": "SH600547", "action": "BUY",
                  "price": 30, "shares": 100}, path=nested)
    assert nested.exists()


def test_append_preserves_existing_lines(tmp_jsonl):
    append_trade({"date": "2026-05-26", "sym": "SH600547", "action": "BUY",
                  "price": 30, "shares": 100}, path=tmp_jsonl)
    append_trade({"date": "2026-05-26", "sym": "SZ300347", "action": "BUY",
                  "price": 42, "shares": 200}, path=tmp_jsonl)
    lines = tmp_jsonl.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["sym"] == "SH600547"
    assert json.loads(lines[1])["sym"] == "SZ300347"
