"""Phase B gate — AAA pattern unit tests.

覆盖:
  - Candidate derived: icir_abs / lambda_at_max
  - 4 历史失败案例 reject 验证
  - 3 历史成功案例 pass 验证
  - 单 mode 隔离触发
  - audit log 持久化
  - PHASE_B_GATE_ENABLED 全局回滚
"""
from __future__ import annotations

import csv
from pathlib import Path

from claude_finance.phase_b_gate import (
    CHECK_LOG_FIELDS,
    Candidate,
    CheckResult,
    PhaseBGate,
)


# ============ Candidate derived properties ============

def test_candidate_icir_abs():
    c = Candidate(name="x", n_months=60, is_calmar=0.5, icir=-0.7)
    assert c.icir_abs == 0.7


def test_candidate_lambda_at_max_scalar():
    c = Candidate(name="x", n_months=60, is_calmar=0.5,
                  lambda_locked=0.30, lambda_max=0.30)
    assert c.lambda_at_max is True


def test_candidate_lambda_at_max_below():
    c = Candidate(name="x", n_months=60, is_calmar=0.5,
                  lambda_locked=0.20, lambda_max=0.30)
    assert c.lambda_at_max is False


def test_candidate_lambda_at_max_list():
    c = Candidate(name="x", n_months=60, is_calmar=0.5,
                  lambda_locked=[0.30, 0.10], lambda_max=[0.30, 0.10])
    assert c.lambda_at_max is True


def test_candidate_lambda_at_max_list_mismatch():
    c = Candidate(name="x", n_months=60, is_calmar=0.5,
                  lambda_locked=[0.30, 0.10], lambda_max=[0.30, 0.20])
    assert c.lambda_at_max is False


def test_candidate_lambda_at_max_none():
    c = Candidate(name="x", n_months=60, is_calmar=0.5,
                  lambda_locked=0.30, lambda_max=None)
    assert c.lambda_at_max is False


# ============ 4 历史失败 reject 验证 ============

def test_v19_9_unlock_rejected(tmp_path: Path):
    gate = PhaseBGate(audit_log=tmp_path / "log.csv")
    cand = Candidate(
        name="v19.9_unlock", n_months=49, is_calmar=2.70,
        lambda_locked=0.30, lambda_max=0.30,
        icir=1.14, spearman_max_abs=0.05,
    )
    r = gate.check(cand)
    assert not r.pass_overall
    assert "thin_sample_overfit" in r.rejects
    assert "extreme_is_calmar_alarm" in r.rejects
    assert "icir_too_strong_unverified" in r.rejects
    assert "lambda_at_sweep_max" in r.warns


def test_super_big_net_rejected(tmp_path: Path):
    gate = PhaseBGate(audit_log=tmp_path / "log.csv")
    cand = Candidate(
        name="super_big_net", n_months=36, is_calmar=2.01,
        lambda_locked=0.30, lambda_max=0.30,
        icir=0.66, spearman_max_abs=0.05,
    )
    r = gate.check(cand)
    assert not r.pass_overall
    assert "thin_sample_overfit" in r.rejects


def test_shareholders_rejected_red_alarm(tmp_path: Path):
    gate = PhaseBGate(audit_log=tmp_path / "log.csv")
    cand = Candidate(
        name="v20_shareholders", n_months=49, is_calmar=6.09,
        lambda_locked=0.30, lambda_max=0.30,
        icir=1.044, spearman_max_abs=0.02,
    )
    r = gate.check(cand)
    assert not r.pass_overall
    assert "extreme_is_calmar_red_alarm" in r.rejects


def test_v19_7_passes_but_oos_failed(tmp_path: Path):
    """v19.7 OOS 实际 -17.7% 但 gate 抓不到 — 文档化 known gap."""
    gate = PhaseBGate(audit_log=tmp_path / "log.csv")
    cand = Candidate(
        name="v19.7", n_months=60, is_calmar=0.76,
        lambda_locked=[0.20, 0.20], lambda_max=[0.30, 0.20],
        icir=0.50, spearman_max_abs=0.08,
    )
    r = gate.check(cand)
    assert r.pass_overall


# ============ 3 历史成功 pass 验证 ============

def test_v19_4_passes(tmp_path: Path):
    gate = PhaseBGate(audit_log=tmp_path / "log.csv")
    cand = Candidate(
        name="v19.4", n_months=60, is_calmar=0.85,
        lambda_locked=[0.10, 0.10], lambda_max=[0.20, 0.20],
        icir=0.55, spearman_max_abs=0.10,
    )
    r = gate.check(cand)
    assert r.pass_overall


def test_v19_6_passes_with_lambda_max_warn(tmp_path: Path):
    gate = PhaseBGate(audit_log=tmp_path / "log.csv")
    cand = Candidate(
        name="v19.6", n_months=60, is_calmar=0.85,
        lambda_locked=0.30, lambda_max=0.30,
        icir=0.50, spearman_max_abs=0.05,
    )
    r = gate.check(cand)
    assert r.pass_overall
    assert "lambda_at_sweep_max" in r.warns


def test_v19_10_passes(tmp_path: Path):
    gate = PhaseBGate(audit_log=tmp_path / "log.csv")
    cand = Candidate(
        name="v19.10", n_months=60, is_calmar=0.85,
        lambda_locked=[0.30, 0.10], lambda_max=[0.30, 0.20],
        icir=0.50, spearman_max_abs=0.05,
    )
    r = gate.check(cand)
    assert r.pass_overall
    assert "lambda_at_sweep_max" not in r.fired_modes


# ============ 单 mode 隔离触发 ============

def test_isolated_icir_below_min(tmp_path: Path):
    gate = PhaseBGate(audit_log=tmp_path / "log.csv")
    cand = Candidate(
        name="weak_factor", n_months=80, is_calmar=0.5,
        lambda_locked=0.10, lambda_max=0.30,
        icir=0.20, spearman_max_abs=0.05,
    )
    r = gate.check(cand)
    assert "icir_below_min" in r.rejects


def test_isolated_spearman_overlap_warn(tmp_path: Path):
    gate = PhaseBGate(audit_log=tmp_path / "log.csv")
    cand = Candidate(
        name="overlap_factor", n_months=80, is_calmar=0.5,
        lambda_locked=0.10, lambda_max=0.30,
        icir=0.55, spearman_max_abs=0.35,
    )
    r = gate.check(cand)
    assert "spearman_overlap_with_production" in r.warns
    # 仅 warn, 整体 pass
    assert r.pass_overall


# ============ audit log 持久化 ============

def test_audit_log_writes_header_and_row(tmp_path: Path):
    log_path = tmp_path / "phase_b_check.csv"
    gate = PhaseBGate(audit_log=log_path)
    cand = Candidate(
        name="test_x", n_months=49, is_calmar=2.5,
        lambda_locked=0.30, lambda_max=0.30,
    )
    gate.check(cand)
    assert log_path.exists()
    rows = list(csv.reader(log_path.open()))
    assert rows[0] == list(CHECK_LOG_FIELDS)
    assert len(rows) >= 2
    assert rows[1][1] == "test_x"


def test_audit_log_none_no_write():
    gate = PhaseBGate(audit_log=None)
    cand = Candidate(name="x", n_months=50, is_calmar=1.0)
    r = gate.check(cand)
    assert isinstance(r, CheckResult)


# ============ 全局回滚 ============

def test_global_phase_b_disabled_passes_anything(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "claude_finance.phase_b_gate.PHASE_B_GATE_ENABLED", False,
    )
    gate = PhaseBGate(audit_log=tmp_path / "log.csv", enabled=True)
    cand = Candidate(
        name="extreme_overfit",
        n_months=10, is_calmar=99.0, icir=10.0,
        lambda_locked=0.99, lambda_max=0.99,
        spearman_max_abs=0.99,
    )
    r = gate.check(cand)
    assert r.pass_overall
    assert r.fired_modes == []


# ============ CheckResult summary ============

def test_check_result_summary_pass():
    r = CheckResult(True, [], {}, {})
    assert "PASS" in r.summary()


def test_check_result_summary_reject_warn():
    r = CheckResult(
        False,
        ["thin_sample_overfit", "lambda_at_sweep_max"],
        {"thin_sample_overfit": "reject", "lambda_at_sweep_max": "warn"},
        {"thin_sample_overfit": "...", "lambda_at_sweep_max": "..."},
    )
    s = r.summary()
    assert "REJECT" in s
    assert "WARN" in s
