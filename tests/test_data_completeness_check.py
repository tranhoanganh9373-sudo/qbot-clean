"""Data completeness check — unit tests (AAA pattern).

无网络, 无 production parquet 依赖 (合成 DataFrame + tmp_path + monkeypatch).
覆盖:
  - compute_completeness: OK / WARN / CRITICAL 阈值
  - 大蓝筹 10/9/8/7/0
  - 板块分类
  - log csv schema + append
  - macOS notification 触发
  - 缺文件 → critical
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

# 加载 examples/data_completeness_check.py
_SPEC = importlib.util.spec_from_file_location(
    "data_completeness_check_mod",
    Path(__file__).resolve().parent.parent / "examples" / "data_completeness_check.py",
)
assert _SPEC and _SPEC.loader
dcc = importlib.util.module_from_spec(_SPEC)
sys.modules["data_completeness_check_mod"] = dcc
_SPEC.loader.exec_module(dcc)


# ---------- helpers ----------

BLUE_CHIPS = list(dcc.MUST_HAVE_BLUE_CHIPS.keys())


def make_csi300_set(n: int = 300) -> set[str]:
    """合成 n 只 CSI300 codes (用 200001..200300 避免和板块冲突)."""
    return {f"{200000 + i:06d}" for i in range(1, n + 1)}


def make_universe_set(n: int = 5000) -> set[str]:
    """合成 n 只 universe codes (包含全部 CSI300 + 全部蓝筹)."""
    base = {f"{200000 + i:06d}" for i in range(1, 301)}
    base.update(set(BLUE_CHIPS))
    i = 0
    while len(base) < n:
        c = f"{900000 + i:06d}"
        base.add(c)
        i += 1
    return base


def _patch_paths(monkeypatch, tmp_path: Path, log_name: str = "data_completeness_log.csv"):
    """LOG_PATH → tmp_path (隔离 log)."""
    log_path = tmp_path / log_name
    monkeypatch.setattr(dcc, "LOG_PATH", log_path)
    return log_path


def _write_files(
    tmp_path: Path,
    kline_codes: set[str] | None,
    universe_codes: set[str] | None,
    csi300_codes: set[str] | None,
) -> tuple[Path, Path, Path]:
    """生成 parquet/csv 文件,None = 不写."""
    kline_path = tmp_path / "baidu_kline.parquet"
    universe_path = tmp_path / "universe.csv"
    csi300_path = tmp_path / "csi300.csv"

    if kline_codes is not None:
        df = pd.DataFrame({"code": list(kline_codes), "close": 10.0})
        df.to_parquet(kline_path)
    if universe_codes is not None:
        if universe_codes:
            uni_lines = ["code,name"] + [f"{c},S" for c in universe_codes]
        else:
            uni_lines = ["code,name"]
        universe_path.write_text("\n".join(uni_lines) + "\n")
    if csi300_codes is not None:
        csi_lines = ["code,name"] + [f"{c},S" for c in csi300_codes]
        csi300_path.write_text("\n".join(csi_lines) + "\n")

    return kline_path, universe_path, csi300_path


# ============================================================
# 1. 全 cover → exit 0
# ============================================================

def test_full_coverage_returns_ok():
    # Arrange
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    kline = set(universe)

    # Act
    r = dcc.compute_completeness(kline, universe, csi300)

    # Assert
    assert r["verdict"] == "OK"
    assert r["exit_code"] == 0
    assert r["bluechip_covered"] == 10


# ============================================================
# 2. CSI300 95% → OK (边界)
# ============================================================

def test_csi300_95pct_returns_ok():
    # Arrange — CSI300 285/300 = 95.0%
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    csi_remove = set(list(csi300)[:15])
    kline = set(universe) - csi_remove

    # Act
    r = dcc.compute_completeness(kline, universe, csi300)

    # Assert
    assert r["csi300_pct"] == pytest.approx(0.95)
    assert r["verdict"] == "OK"
    assert r["exit_code"] == 0


# ============================================================
# 3. CSI300 94% + universe 95% → WARNING
# ============================================================

def test_csi300_94pct_universe_95pct_returns_warning():
    # Arrange — CSI300 282/300 = 94.0%, 蓝筹 10/10, universe > 90%
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    csi_remove = set(list(csi300)[:18])  # 94%
    kline = set(universe) - csi_remove

    # Act
    r = dcc.compute_completeness(kline, universe, csi300)

    # Assert
    assert r["csi300_pct"] == pytest.approx(0.94)
    assert r["universe_pct"] >= 0.90
    assert r["verdict"] == "WARNING"
    assert r["exit_code"] == 1


# ============================================================
# 4. CSI300 79% → CRITICAL
# ============================================================

def test_csi300_79pct_returns_critical():
    # Arrange — CSI300 237/300 = 79.0%
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    csi_remove = set(list(csi300)[:63])  # 21% missing → 79%
    kline = set(universe) - csi_remove

    # Act
    r = dcc.compute_completeness(kline, universe, csi300)

    # Assert
    assert r["csi300_pct"] == pytest.approx(0.79)
    assert r["verdict"] == "CRITICAL"
    assert r["exit_code"] == 99


# ============================================================
# 5. 大蓝筹 8/10 → WARNING (border)
# ============================================================

def test_bluechip_eight_returns_warning():
    # Arrange — 缺 2 只蓝筹
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    missing = set(BLUE_CHIPS[:2])
    kline = set(universe) - missing

    # Act
    r = dcc.compute_completeness(kline, universe, csi300)

    # Assert
    assert r["bluechip_covered"] == 8
    assert r["verdict"] == "WARNING"
    assert r["exit_code"] == 1


# ============================================================
# 6. 大蓝筹 7/10 → CRITICAL
# ============================================================

def test_bluechip_seven_returns_critical():
    # Arrange — 缺 3 只蓝筹
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    missing = set(BLUE_CHIPS[:3])
    kline = set(universe) - missing

    # Act
    r = dcc.compute_completeness(kline, universe, csi300)

    # Assert
    assert r["bluechip_covered"] == 7
    assert r["verdict"] == "CRITICAL"
    assert r["exit_code"] == 99


# ============================================================
# 7. 大蓝筹完全缺 → CRITICAL
# ============================================================

def test_bluechip_all_missing_returns_critical():
    # Arrange
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    missing = set(BLUE_CHIPS)
    kline = set(universe) - missing

    # Act
    r = dcc.compute_completeness(kline, universe, csi300)

    # Assert
    assert r["bluechip_covered"] == 0
    assert r["verdict"] == "CRITICAL"
    assert r["exit_code"] == 99


# ============================================================
# 8. log csv schema 正确
# ============================================================

def test_log_csv_schema_correct(monkeypatch, tmp_path: Path):
    # Arrange — 全 cover
    log_path = _patch_paths(monkeypatch, tmp_path)
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    kline = set(universe)
    kline_path, universe_path, csi300_path = _write_files(tmp_path, kline, universe, csi300)
    monkeypatch.setattr(dcc, "KLINE_PATH", kline_path)
    monkeypatch.setattr(dcc, "UNIVERSE_PATH", universe_path)
    monkeypatch.setattr(dcc, "CSI300_PATH", csi300_path)
    monkeypatch.setattr(dcc, "_notify_macos", lambda *a, **k: None)

    # Act
    dcc.main()
    log = pd.read_csv(log_path)

    # Assert
    expected_cols = ["date", "total", "universe_pct", "csi300_pct", "bluechip_pct", "verdict", "details"]
    assert list(log.columns) == expected_cols
    assert len(log) == 1
    assert log.iloc[0]["verdict"] == "OK"


# ============================================================
# 9. log csv append (不覆盖)
# ============================================================

def test_log_csv_appends_not_overwrites(monkeypatch, tmp_path: Path):
    # Arrange
    log_path = _patch_paths(monkeypatch, tmp_path)
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    kline = set(universe)
    kline_path, universe_path, csi300_path = _write_files(tmp_path, kline, universe, csi300)
    monkeypatch.setattr(dcc, "KLINE_PATH", kline_path)
    monkeypatch.setattr(dcc, "UNIVERSE_PATH", universe_path)
    monkeypatch.setattr(dcc, "CSI300_PATH", csi300_path)
    monkeypatch.setattr(dcc, "_notify_macos", lambda *a, **k: None)

    # Act
    dcc.main()
    dcc.main()
    log = pd.read_csv(log_path)

    # Assert
    assert len(log) == 2


# ============================================================
# 10. macOS notification 仅 warn/critical 触发 (green 不弹)
# ============================================================

def test_notification_silent_on_ok(monkeypatch, tmp_path: Path):
    # Arrange — 全 cover → OK
    _patch_paths(monkeypatch, tmp_path)
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    kline = set(universe)
    kline_path, universe_path, csi300_path = _write_files(tmp_path, kline, universe, csi300)
    monkeypatch.setattr(dcc, "KLINE_PATH", kline_path)
    monkeypatch.setattr(dcc, "UNIVERSE_PATH", universe_path)
    monkeypatch.setattr(dcc, "CSI300_PATH", csi300_path)

    calls: list[tuple] = []
    monkeypatch.setattr(dcc, "_notify_macos", lambda title, msg: calls.append((title, msg)))

    # Act
    code = dcc.main()

    # Assert
    assert code == 0
    assert calls == []


def test_notification_fires_on_critical(monkeypatch, tmp_path: Path):
    # Arrange — 蓝筹全缺 → CRITICAL
    _patch_paths(monkeypatch, tmp_path)
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    kline = set(universe) - set(BLUE_CHIPS)
    kline_path, universe_path, csi300_path = _write_files(tmp_path, kline, universe, csi300)
    monkeypatch.setattr(dcc, "KLINE_PATH", kline_path)
    monkeypatch.setattr(dcc, "UNIVERSE_PATH", universe_path)
    monkeypatch.setattr(dcc, "CSI300_PATH", csi300_path)

    calls: list[tuple] = []
    monkeypatch.setattr(dcc, "_notify_macos", lambda title, msg: calls.append((title, msg)))

    # Act
    code = dcc.main()

    # Assert
    assert code == 99
    assert len(calls) == 1
    assert "CRITICAL" in calls[0][0]


# ============================================================
# 11. baidu_kline 缺失 → critical
# ============================================================

def test_missing_baidu_kline_returns_critical(monkeypatch, tmp_path: Path):
    # Arrange
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(dcc, "KLINE_PATH", tmp_path / "missing_kline.parquet")
    monkeypatch.setattr(dcc, "UNIVERSE_PATH", tmp_path / "universe.csv")
    monkeypatch.setattr(dcc, "CSI300_PATH", tmp_path / "csi300.csv")
    monkeypatch.setattr(dcc, "_notify_macos", lambda *a, **k: None)

    # Act
    code = dcc.main()

    # Assert
    assert code == 99


# ============================================================
# 12. universe.csv 缺失 → critical
# ============================================================

def test_missing_universe_returns_critical(monkeypatch, tmp_path: Path):
    # Arrange
    _patch_paths(monkeypatch, tmp_path)
    kline_path, _, _ = _write_files(tmp_path, {"600519", "000001"}, None, None)
    monkeypatch.setattr(dcc, "KLINE_PATH", kline_path)
    monkeypatch.setattr(dcc, "UNIVERSE_PATH", tmp_path / "missing_universe.csv")
    monkeypatch.setattr(dcc, "CSI300_PATH", tmp_path / "missing_csi300.csv")
    monkeypatch.setattr(dcc, "_notify_macos", lambda *a, **k: None)

    # Act
    code = dcc.main()

    # Assert
    assert code == 99


# ============================================================
# 13. empty universe → critical
# ============================================================

def test_empty_universe_returns_critical(monkeypatch, tmp_path: Path):
    # Arrange — universe.csv 仅 header
    _patch_paths(monkeypatch, tmp_path)
    kline_path, universe_path, csi300_path = _write_files(
        tmp_path, {"600519"}, set(), make_csi300_set(300),
    )
    monkeypatch.setattr(dcc, "KLINE_PATH", kline_path)
    monkeypatch.setattr(dcc, "UNIVERSE_PATH", universe_path)
    monkeypatch.setattr(dcc, "CSI300_PATH", csi300_path)
    monkeypatch.setattr(dcc, "_notify_macos", lambda *a, **k: None)

    # Act
    code = dcc.main()

    # Assert
    assert code == 99


# ============================================================
# 14. 板块分布正确
# ============================================================

def test_board_classification_correct():
    # Arrange — 5 板块各 2 只
    universe = {
        "000001", "000002",   # 深主板
        "300001", "300750",   # 创业板
        "600519", "601398",   # 沪主板
        "688001", "688002",   # 科创板
        "830001", "430001",   # 北交所
    }
    csi300: set[str] = set()
    kline = {"000001", "300750", "600519", "688001", "430001"}  # 每板块 1/2

    # Act
    r = dcc.compute_completeness(kline, universe, csi300)

    # Assert
    board = r["board_dist"]
    assert board["深主板"]["covered"] == 1 and board["深主板"]["total"] == 2
    assert board["创业板"]["covered"] == 1 and board["创业板"]["total"] == 2
    assert board["沪主板"]["covered"] == 1 and board["沪主板"]["total"] == 2
    assert board["科创板"]["covered"] == 1 and board["科创板"]["total"] == 2
    assert board["北交所"]["covered"] == 1 and board["北交所"]["total"] == 2


# ============================================================
# 15. 输出格式人类可读 (capsys 捕获 STDOUT)
# ============================================================

def test_print_report_human_readable(capsys):
    # Arrange
    universe = make_universe_set(500)
    csi300 = make_csi300_set(300)
    kline = set(universe)
    r = dcc.compute_completeness(kline, universe, csi300)

    # Act
    dcc._print_report(r)
    captured = capsys.readouterr()

    # Assert — 4 section header 全在
    out = captured.out
    assert "[1] Overall" in out
    assert "[2] By board" in out
    assert "[3] CSI300" in out
    assert "[4] 必持大蓝筹" in out
    assert "Verdict:" in out
