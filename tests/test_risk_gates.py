"""Portfolio risk gates — AAA pattern unit tests.

覆盖:
  - check_drawdown: insufficient_history / 边界值 / disabled
  - check_daily_loss: 触发 / 通过 / 0 prev / disabled
  - check_position_weight: 触发 / 通过 / 空 / disabled
  - audit log: header + row 持久化
  - compute_nav_series_from_log: 合成 trade log + kline
  - self_check exit codes
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from claude_finance.risk.gates import (
    AUDIT_FIELDS,
    GateResult,
    PortfolioRiskGate,
    compute_nav_series_from_log,
)


# ============ check_drawdown ============

def test_drawdown_insufficient_history_passes():
    # Arrange
    gate = PortfolioRiskGate(audit_log=None)
    nav = [100.0] * 5

    # Act
    r = gate.check_drawdown(nav)

    # Assert
    assert r.ok
    assert "insufficient_history" in r.reason
    assert r.gate_id == "drawdown"


def test_drawdown_triggers_at_exactly_threshold():
    gate = PortfolioRiskGate(audit_log=None, min_history=10)
    nav = [100.0] * 10 + [85.0]
    r = gate.check_drawdown(nav)
    assert not r.ok
    assert "drawdown" in r.reason
    assert r.metric == pytest.approx(-0.15, abs=1e-9)


def test_drawdown_passes_at_minus_14_pct():
    gate = PortfolioRiskGate(audit_log=None, min_history=10)
    nav = [100.0] * 10 + [86.0]
    r = gate.check_drawdown(nav)
    assert r.ok
    assert r.metric == pytest.approx(-0.14, abs=1e-9)


def test_drawdown_peak_is_max_not_start():
    gate = PortfolioRiskGate(audit_log=None, min_history=5)
    nav = [100.0, 110.0, 120.0, 115.0, 105.0, 100.0]
    r = gate.check_drawdown(nav)
    assert not r.ok
    assert r.detail["peak"] == 120.0


def test_drawdown_disabled_bypass():
    gate = PortfolioRiskGate(audit_log=None, enabled=False, min_history=5)
    nav = [100.0] * 5 + [50.0]
    r = gate.check_drawdown(nav)
    assert r.ok
    assert r.reason == "disabled"


# ============ check_daily_loss ============

def test_daily_loss_triggers_at_minus_5_pct():
    gate = PortfolioRiskGate(audit_log=None)
    r = gate.check_daily_loss(95.0, 100.0)
    assert not r.ok
    assert r.metric == pytest.approx(-0.05, abs=1e-9)


def test_daily_loss_passes_at_minus_3_pct():
    gate = PortfolioRiskGate(audit_log=None)
    r = gate.check_daily_loss(97.0, 100.0)
    assert r.ok


def test_daily_loss_prev_zero_bypass():
    gate = PortfolioRiskGate(audit_log=None)
    r = gate.check_daily_loss(50.0, 0.0)
    assert r.ok
    assert "prev_nav_zero_or_negative" in r.reason


def test_daily_loss_disabled_bypass():
    gate = PortfolioRiskGate(audit_log=None, enabled=False)
    r = gate.check_daily_loss(10.0, 100.0)
    assert r.ok
    assert r.reason == "disabled"


# ============ check_position_weight ============

def test_position_weight_triggers_at_25_pct():
    gate = PortfolioRiskGate(audit_log=None)
    weights = {"SH600000": 0.25, "SZ000001": 0.10}
    r = gate.check_position_weight(weights)
    assert not r.ok
    assert r.detail["sym"] == "SH600000"


def test_position_weight_passes_at_15_pct():
    gate = PortfolioRiskGate(audit_log=None)
    weights = {f"SH60000{i}": 0.125 for i in range(8)}
    r = gate.check_position_weight(weights)
    assert r.ok


def test_position_weight_empty_bypass():
    gate = PortfolioRiskGate(audit_log=None)
    r = gate.check_position_weight({})
    assert r.ok
    assert "no_positions" in r.reason


def test_position_weight_disabled_bypass():
    gate = PortfolioRiskGate(audit_log=None, enabled=False)
    r = gate.check_position_weight({"SH600000": 0.99})
    assert r.ok


# ============ audit log ============

def test_audit_creates_csv_with_header(tmp_path: Path):
    log_path = tmp_path / "risk_event_log.csv"
    gate = PortfolioRiskGate(audit_log=log_path)
    result = GateResult(False, "test_reason", "drawdown",
                        metric=-0.20, detail={"peak": 100, "cur": 80})
    gate.audit(result)
    assert log_path.exists()
    rows = list(csv.reader(log_path.open()))
    assert rows[0] == list(AUDIT_FIELDS)
    assert len(rows) == 2
    assert rows[1][1] == "drawdown"
    assert rows[1][3] == "1"


def test_audit_appends_multiple_rows(tmp_path: Path):
    log_path = tmp_path / "risk_event_log.csv"
    gate = PortfolioRiskGate(audit_log=log_path)
    for i in range(3):
        gate.audit(GateResult(True, f"row{i}", "drawdown"))
    rows = list(csv.reader(log_path.open()))
    assert len(rows) == 4


def test_audit_disabled_when_log_path_none():
    gate = PortfolioRiskGate(audit_log=None)
    gate.audit(GateResult(False, "test", "drawdown"))


# ============ compute_nav_series_from_log ============

def _make_synthetic_kline(tmp_path: Path, codes: list,
                          start: str = "2026-01-01", n_days: int = 60,
                          base_price: float = 10.0) -> Path:
    dates = pd.date_range(start, periods=n_days, freq="B")
    rows = []
    for code in codes:
        for i, d in enumerate(dates):
            price = base_price + 0.05 * i
            rows.append({"code": code, "date": d, "close": price})
    df = pd.DataFrame(rows)
    path = tmp_path / "baidu_kline.parquet"
    df.to_parquet(path)
    return path


def _make_synthetic_trade_log(tmp_path: Path, trades: list) -> Path:
    df = pd.DataFrame(trades)
    path = tmp_path / "paper_trade_log.csv"
    df.to_csv(path, index=False)
    return path


def test_nav_series_empty_when_no_trade_log(tmp_path: Path):
    nav = compute_nav_series_from_log(
        trade_log=tmp_path / "nonexistent.csv",
        kline=tmp_path / "nonexistent.parquet",
    )
    assert nav == []


def test_nav_series_single_buy_then_kline_rising(tmp_path: Path):
    kline = _make_synthetic_kline(tmp_path, ["600000"], n_days=10, base_price=10.0)
    trades = [
        {"date": "2026-01-01", "action": "BUY", "symbol": "SH600000",
         "name": "x", "score": 0.1, "price": 10.0},
    ]
    log = _make_synthetic_trade_log(tmp_path, trades)
    nav = compute_nav_series_from_log(capital=100000.0, trade_log=log, kline=kline)
    assert len(nav) >= 5
    first_nav = nav[0][1]
    assert first_nav == pytest.approx(100000.0, abs=500.0)
    assert nav[-1][1] > first_nav


def test_nav_series_skips_price_zero_rows(tmp_path: Path):
    kline = _make_synthetic_kline(tmp_path, ["600000"], n_days=5, base_price=10.0)
    trades = [
        {"date": "2026-01-01", "action": "BUY", "symbol": "SH600000",
         "name": "x", "score": 0.1, "price": 10.0},
        {"date": "2026-01-02", "action": "SELL", "symbol": "SH600000",
         "name": "x", "score": 0.0, "price": 0.0},
    ]
    log = _make_synthetic_trade_log(tmp_path, trades)
    nav = compute_nav_series_from_log(capital=100000.0, trade_log=log, kline=kline)
    assert len(nav) >= 3
    assert nav[-1][1] >= nav[0][1]


# ============ RISK_ENABLED global toggle ============

def test_global_risk_enabled_false(monkeypatch):
    monkeypatch.setattr(
        "claude_finance.risk.gates.RISK_ENABLED", False,
    )
    # ctor 时读 RISK_ENABLED
    gate = PortfolioRiskGate(audit_log=None, enabled=True, min_history=5)
    r = gate.check_drawdown([100.0] * 5 + [50.0])
    assert r.ok
    assert r.reason == "disabled"


# ============ self_check exit codes ============

def test_self_check_returns_int(monkeypatch, tmp_path):
    """self_check 必须返回 int 而非抛, 走 audit_log 默认路径写 tmp."""
    # 把 audit log 默认路径 monkey patch 到 tmp, 不污染生产
    monkeypatch.setattr(
        "claude_finance.risk.gates.DEFAULT_AUDIT_LOG",
        tmp_path / "risk_event_log.csv",
    )
    from claude_finance.risk.gates import self_check
    code = self_check(verbose=False)
    assert isinstance(code, int)
    assert code in (0, 1, 2)
