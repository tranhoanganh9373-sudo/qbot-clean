"""Static leak scanner — AAA pattern unit tests."""
from __future__ import annotations

import json
from pathlib import Path

from tools.static_leak_check import (
    SEV_INFO,
    SEV_REJECT,
    SEV_WARN,
    Finding,
    ScanReport,
    scan_file,
    scan_paths,
    scan_text,
    write_report,
)


# ============ regex patterns ============

def test_shift_negative_rejects():
    code = "df['ret_t1'] = df['close'].shift(-1) / df['close'] - 1"
    findings = scan_text(Path("synthetic.py"), code)
    rejects = [f for f in findings if f.severity == SEV_REJECT]
    assert any(f.pattern_id == "shift_negative" for f in rejects)


def test_shift_negative_large():
    code = "future_5 = df['close'].shift(-5)"
    findings = scan_text(Path("synthetic.py"), code)
    rejects = [f for f in findings if f.severity == SEV_REJECT]
    assert len(rejects) >= 1


def test_shift_positive_ok():
    code = "lag_5 = df['close'].shift(5)"
    findings = scan_text(Path("synthetic.py"), code)
    rejects = [f for f in findings if f.pattern_id == "shift_negative"]
    assert len(rejects) == 0


def test_iloc_positive_offset_rejects():
    code = "future_val = df.iloc[+3]"
    findings = scan_text(Path("synthetic.py"), code)
    rejects = [f for f in findings if f.severity == SEV_REJECT]
    assert any(f.pattern_id == "iloc_positive_offset" for f in rejects)


def test_loc_future_slice_rejects():
    code = "df_future = df.loc[future_date:, 'close']"
    findings = scan_text(Path("synthetic.py"), code)
    rejects = [f for f in findings if f.severity == SEV_REJECT]
    assert any(f.pattern_id == "loc_future_slice" for f in rejects)


def test_forward_returns_keyword_warn():
    code = "forward_returns = compute_alpha(panel)"
    findings = scan_text(Path("synthetic.py"), code)
    warns = [f for f in findings if f.severity == SEV_WARN]
    assert any(f.pattern_id == "forward_keyword" for f in warns)


def test_future_returns_keyword_warn():
    code = "future_returns_t5 = df['close'].pct_change(5)"
    findings = scan_text(Path("synthetic.py"), code)
    warns = [f for f in findings if f.severity == SEV_WARN]
    assert any(f.pattern_id == "forward_keyword" for f in warns)


def test_transform_last_warn():
    code = "df['month_last'] = df.groupby('code').transform('last')"
    findings = scan_text(Path("synthetic.py"), code)
    warns = [f for f in findings if f.severity == SEV_WARN]
    assert any(f.pattern_id == "transform_last" for f in warns)


def test_comment_match_demoted_to_info():
    code = '# this is a comment about future_returns lookahead bias'
    findings = scan_text(Path("synthetic.py"), code)
    rejects = [f for f in findings if f.severity == SEV_REJECT]
    assert rejects == []
    info_or_lower = [f for f in findings if f.severity == SEV_INFO]
    assert len(info_or_lower) >= 1


def test_clean_code_no_findings():
    code = """
import pandas as pd

def compute_zscore(df, window=20):
    rolling = df['close'].rolling(window).mean()
    return (df['close'] - rolling) / df['close'].rolling(window).std()
"""
    findings = scan_text(Path("synthetic.py"), code)
    rejects = [f for f in findings if f.severity == SEV_REJECT]
    warns = [f for f in findings if f.severity == SEV_WARN]
    assert rejects == []
    assert warns == []


def test_lookahead_keyword_in_string_info():
    code = "msg = '防 lookahead bias'"
    findings = scan_text(Path("synthetic.py"), code)
    rejects = [f for f in findings if f.severity == SEV_REJECT]
    assert rejects == []


# ============ scan_file / scan_paths ============

def test_scan_file_synthetic(tmp_path: Path):
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("df.shift(-1)\n")
    findings = scan_file(bad_file)
    assert len(findings) == 1
    assert findings[0].pattern_id == "shift_negative"
    assert findings[0].file == str(bad_file)


def test_scan_paths_multi_file(tmp_path: Path):
    good = tmp_path / "good.py"
    good.write_text("df.shift(1)\n")
    bad = tmp_path / "bad.py"
    bad.write_text("df.shift(-1)\n")
    report = scan_paths([good, bad])
    assert report.n_files == 2
    assert report.n_rejects == 1
    assert report.n_findings == 1


def test_scan_paths_directory(tmp_path: Path):
    (tmp_path / "x.py").write_text("df.shift(-2)\n")
    (tmp_path / "y.py").write_text("z = future_returns\n")
    report = scan_paths([tmp_path])
    assert report.n_files == 2
    assert report.n_rejects == 1
    assert report.n_warns == 1


def test_scan_paths_nonexistent_file_silent(tmp_path: Path):
    report = scan_paths([tmp_path / "ghost.py"])
    assert report.n_findings == 0


# ============ ScanReport ============

def test_scan_report_properties():
    r = ScanReport(
        scanned_at="2026-05-27T10:00:00",
        files_scanned=["a.py", "b.py"],
        findings=[
            Finding(file="a.py", line=1, severity=SEV_REJECT,
                    pattern_id="shift_negative", matched=".shift(-1)", description=""),
            Finding(file="b.py", line=2, severity=SEV_WARN,
                    pattern_id="forward_keyword", matched="forward_returns", description=""),
            Finding(file="b.py", line=3, severity=SEV_INFO,
                    pattern_id="lookahead_keyword_mention", matched="lookahead", description=""),
        ],
    )
    assert r.n_files == 2
    assert r.n_findings == 3
    assert r.n_rejects == 1
    assert r.n_warns == 1


def test_scan_report_to_dict_round_trip():
    r = ScanReport(
        scanned_at="2026-05-27T10:00:00",
        files_scanned=["a.py"],
        findings=[
            Finding(file="a.py", line=1, severity=SEV_REJECT,
                    pattern_id="x", matched="m", description="d"),
        ],
    )
    d = r.to_dict()
    assert d["n_files"] == 1
    assert d["n_rejects"] == 1
    assert d["findings"][0]["pattern_id"] == "x"


def test_write_report_persists(tmp_path: Path):
    r = ScanReport(
        scanned_at="2026-05-27T10:00:00",
        files_scanned=["a.py"],
        findings=[],
    )
    out = tmp_path / "report.json"
    write_report(r, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["n_files"] == 1


# ============ 全局回滚 ============

def test_leak_check_disabled_bypass(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "tools.static_leak_check.LEAK_CHECK_ENABLED", False,
    )
    bad = tmp_path / "bad.py"
    bad.write_text("df.shift(-1)\nfuture_returns = 1\n")
    report = scan_paths([bad])
    assert report.n_findings == 0
