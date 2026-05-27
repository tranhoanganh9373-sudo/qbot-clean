"""Multi-agent debate veto — AAA pattern unit tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from claude_finance.debate_veto import (
    AUDIT_FIELDS,
    DebateVeto,
    VetoResult,
    filter_picks,
    load_debate_votes,
)


def _write_log(tmp_path: Path, records: list) -> Path:
    path = tmp_path / "debate.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _r(agent: str, sym: str, vote: str, ts: str = "2026-05-27T10:00:00+00:00") -> dict:
    return {"agent": agent, "ts": ts, "sym": sym, "vote": vote}


# ============ load_debate_votes ============

def test_load_empty_log_returns_empty(tmp_path: Path):
    votes, source = load_debate_votes(tmp_path / "ghost.jsonl")
    assert votes == {}
    assert source == ""


def test_load_single_day_full_3_agents(tmp_path: Path):
    log = _write_log(tmp_path, [
        _r("bull", "SH600000", "BUY"),
        _r("bear", "SH600000", "SELL"),
        _r("neutral", "SH600000", "HOLD"),
    ])
    votes, source = load_debate_votes(log)
    assert source == "2026-05-27"
    assert votes == {"SH600000": {"bull": "BUY", "bear": "SELL", "neutral": "HOLD"}}


def test_load_picks_latest_date_when_multi_day(tmp_path: Path):
    log = _write_log(tmp_path, [
        _r("neutral", "SH600000", "BUY", ts="2026-05-20T10:00:00+00:00"),
        _r("neutral", "SZ000001", "SELL", ts="2026-05-27T10:00:00+00:00"),
    ])
    votes, source = load_debate_votes(log)
    assert source == "2026-05-27"
    assert votes == {"SZ000001": {"neutral": "SELL"}}


def test_load_specific_date(tmp_path: Path):
    log = _write_log(tmp_path, [
        _r("neutral", "SH600000", "BUY", ts="2026-05-20T10:00:00+00:00"),
        _r("neutral", "SZ000001", "SELL", ts="2026-05-27T10:00:00+00:00"),
    ])
    votes, source = load_debate_votes(log, date_str="2026-05-20")
    assert source == "2026-05-20"
    assert "SH600000" in votes
    assert "SZ000001" not in votes


def test_load_nonexistent_date(tmp_path: Path):
    log = _write_log(tmp_path, [_r("neutral", "SH600000", "BUY")])
    votes, source = load_debate_votes(log, date_str="2099-01-01")
    assert votes == {}
    assert source == ""


def test_load_skips_malformed_jsonl(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(_r("neutral", "SH600000", "BUY")) + "\n"
        "this is not json\n"
        + json.dumps(_r("neutral", "SZ000001", "SELL")) + "\n",
        encoding="utf-8",
    )
    votes, source = load_debate_votes(path)
    assert len(votes) == 2


# ============ DebateVeto.filter_picks ============

def test_filter_neutral_sell_vetoed(tmp_path: Path):
    log = _write_log(tmp_path, [
        _r("neutral", "SH600000", "BUY"),
        _r("neutral", "SZ000001", "SELL"),
        _r("neutral", "SZ000002", "HOLD"),
    ])
    veto = DebateVeto(log_path=log, audit_log=tmp_path / "audit.csv")
    result = veto.filter_picks(["SH600000", "SZ000001", "SZ000002"])
    assert result.n_kept == 2
    assert result.n_vetoed == 1
    assert result.vetoed[0]["sym"] == "SZ000001"


def test_filter_keeps_when_neutral_buy_or_hold(tmp_path: Path):
    log = _write_log(tmp_path, [
        _r("neutral", "SH600000", "BUY"),
        _r("neutral", "SZ000001", "HOLD"),
    ])
    veto = DebateVeto(log_path=log, audit_log=None)
    result = veto.filter_picks(["SH600000", "SZ000001"])
    assert result.n_kept == 2
    assert result.n_vetoed == 0


def test_filter_unknown_sym_kept(tmp_path: Path):
    log = _write_log(tmp_path, [_r("neutral", "SH600000", "BUY")])
    veto = DebateVeto(log_path=log, audit_log=None)
    result = veto.filter_picks(["SH600000", "SZ999999"])
    assert result.n_kept == 2


def test_filter_picks_empty(tmp_path: Path):
    log = _write_log(tmp_path, [_r("neutral", "SH600000", "BUY")])
    veto = DebateVeto(log_path=log, audit_log=None)
    result = veto.filter_picks([])
    assert result.n_kept == 0
    assert result.n_vetoed == 0


def test_filter_no_log_skips(tmp_path: Path):
    veto = DebateVeto(log_path=tmp_path / "ghost.jsonl", audit_log=None)
    result = veto.filter_picks(["SH600000"])
    assert result.skipped
    assert result.n_kept == 1


def test_filter_disabled_passes_through(tmp_path: Path):
    log = _write_log(tmp_path, [_r("neutral", "SH600000", "SELL")])
    veto = DebateVeto(log_path=log, audit_log=None, enabled=False)
    result = veto.filter_picks(["SH600000"])
    assert result.skipped
    assert result.n_kept == 1
    assert result.n_vetoed == 0


# ============ veto_vote 自定义 ============

def test_custom_veto_vote_hold_also_rejected(tmp_path: Path):
    log = _write_log(tmp_path, [_r("neutral", "SH600000", "HOLD")])
    veto = DebateVeto(log_path=log, audit_log=None, veto_vote="HOLD")
    result = veto.filter_picks(["SH600000"])
    assert result.n_vetoed == 1


# ============ audit log ============

def test_audit_log_writes_header_and_kept(tmp_path: Path):
    log = _write_log(tmp_path, [
        _r("bull", "SH600000", "BUY"),
        _r("bear", "SH600000", "SELL"),
        _r("neutral", "SH600000", "HOLD"),
    ])
    audit = tmp_path / "audit.csv"
    veto = DebateVeto(log_path=log, audit_log=audit)
    veto.filter_picks(["SH600000"])
    assert audit.exists()
    rows = list(csv.reader(audit.open()))
    assert rows[0] == list(AUDIT_FIELDS)
    assert rows[1][6] == "kept"
    assert rows[1][3] == "BUY"
    assert rows[1][4] == "SELL"
    assert rows[1][5] == "HOLD"


def test_audit_log_writes_VETOED(tmp_path: Path):
    log = _write_log(tmp_path, [_r("neutral", "SH600000", "SELL")])
    audit = tmp_path / "audit.csv"
    veto = DebateVeto(log_path=log, audit_log=audit)
    veto.filter_picks(["SH600000"])
    rows = list(csv.reader(audit.open()))
    assert rows[1][6] == "VETOED"


def test_preview_does_not_write_audit(tmp_path: Path):
    log = _write_log(tmp_path, [_r("neutral", "SH600000", "SELL")])
    audit = tmp_path / "audit.csv"
    veto = DebateVeto(log_path=log, audit_log=audit)
    result = veto.preview(["SH600000"])
    assert result.n_vetoed == 1
    assert not audit.exists()


# ============ 全局回滚 ============

def test_global_debate_veto_disabled(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "claude_finance.debate_veto.DEBATE_VETO_ENABLED", False,
    )
    log = _write_log(tmp_path, [_r("neutral", "SH600000", "SELL")])
    veto = DebateVeto(log_path=log, audit_log=None, enabled=True)
    result = veto.filter_picks(["SH600000"])
    assert result.skipped
    assert result.n_kept == 1


# ============ filter_picks convenience ============

def test_convenience_filter_picks(tmp_path: Path):
    log = _write_log(tmp_path, [_r("neutral", "SH600000", "SELL")])
    result = filter_picks(["SH600000"], log_path=log, audit_log=None)
    assert isinstance(result, VetoResult)
    assert result.n_vetoed == 1
