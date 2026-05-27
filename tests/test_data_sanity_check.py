"""Data sanity check — pure-function unit tests (AAA pattern).

无网络, 无 production parquet 依赖 (合成 DataFrame + tmp_path).
覆盖:
  - check_no_neg_close: 干净 / 1 条 neg / 100 条 neg / days 窗口隔离
  - check_no_extreme_jump: 正常 / 12% jump / 10% 涨停不算
  - check_data_freshness: today / yesterday / 5 天前
  - check_coverage: 100% / 80% 边界 / 50%
  - check_no_extreme_low: 全 >0.5 / 1 条 0.3
  - main strict vs lenient: 全 pass / critical fail / non-critical fail
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

# 加载 examples/data_sanity_check.py
_SPEC = importlib.util.spec_from_file_location(
    "data_sanity_check_mod",
    Path(__file__).resolve().parent.parent / "examples" / "data_sanity_check.py",
)
assert _SPEC and _SPEC.loader
dsc = importlib.util.module_from_spec(_SPEC)
sys.modules["data_sanity_check_mod"] = dsc
_SPEC.loader.exec_module(dsc)


# ---------- helpers ----------

def make_kline(rows: list[dict]) -> pd.DataFrame:
    """合成 K 线 DF, 自动转 date dtype."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def make_clean_recent(n_stocks: int = 5, n_days: int = 10, today: str = "2026-05-22") -> pd.DataFrame:
    """生成 n_stocks 只 × n_days 天的干净数据 (close ~ 10 元, 0.5% 递增)."""
    end = pd.Timestamp(today)
    dates = pd.date_range(end - pd.Timedelta(days=n_days - 1), end, freq="D")
    rows = []
    for s in range(n_stocks):
        code = f"00000{s}"[-6:]
        for i, d in enumerate(dates):
            rows.append({"code": code, "date": d, "close": 10.0 + i * 0.05})
    return pd.DataFrame(rows)


# ---------- check_no_neg_close ----------

def test_check_no_neg_close_clean_passes():
    # Arrange
    df = make_clean_recent()

    # Act
    passed, msg = dsc.check_no_neg_close(df)

    # Assert
    assert passed is True
    assert "OK" in msg
    assert "0 rows" in msg


def test_check_no_neg_close_one_negative_fails():
    # Arrange
    df = make_clean_recent()
    df.loc[0, "close"] = -1.5

    # Act
    passed, msg = dsc.check_no_neg_close(df)

    # Assert
    assert passed is False
    assert "FAIL" in msg
    assert "1 rows" in msg


def test_check_no_neg_close_one_hundred_negatives_fails():
    # Arrange
    df = make_clean_recent(n_stocks=20, n_days=10)
    df.loc[df.index[:100], "close"] = -0.5

    # Act
    passed, msg = dsc.check_no_neg_close(df)

    # Assert
    assert passed is False
    assert "100 rows" in msg


def test_check_no_neg_close_days_window_excludes_old_corruption():
    """days=30 时, 旧的 neg 不在窗口内 → pass; full-scan 应能捕到."""
    # Arrange
    df = make_clean_recent(today="2026-05-22")
    old_neg = make_kline([{"code": "000001", "date": "2019-06-03", "close": -0.06}])
    df = pd.concat([df, old_neg], ignore_index=True)

    # Act
    passed_window, _ = dsc.check_no_neg_close(df, days=30)
    passed_full, _ = dsc.check_no_neg_close(df, days=None)

    # Assert
    assert passed_window is True
    assert passed_full is False


# ---------- check_no_extreme_jump ----------

def test_check_no_extreme_jump_normal_passes():
    # Arrange — 干净数据
    df = make_clean_recent()

    # Act
    passed, msg = dsc.check_no_extreme_jump(df)

    # Assert
    assert passed is True
    assert "OK" in msg


def test_check_no_extreme_jump_twelve_pct_jump_fails():
    # Arrange — stock 000000 第 2 天插入 12% 跳变
    df = make_clean_recent(n_stocks=1, n_days=5)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df.loc[1, "close"] = df.loc[0, "close"] * 1.12

    # Act
    passed, msg = dsc.check_no_extreme_jump(df)

    # Assert
    assert passed is False
    assert "FAIL" in msg


def test_check_no_extreme_jump_ten_pct_passes():
    """涨停 10% 不被误判 (threshold 0.11)."""
    # Arrange
    df = make_clean_recent(n_stocks=1, n_days=5)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df.loc[1, "close"] = df.loc[0, "close"] * 1.10

    # Act
    passed, _ = dsc.check_no_extreme_jump(df)

    # Assert
    assert passed is True


# ---------- check_data_freshness ----------

def test_check_data_freshness_today_passes():
    # Arrange
    today = pd.Timestamp("2026-05-22")
    df = make_kline([{"code": "000001", "date": "2026-05-22", "close": 10.0}])

    # Act
    passed, msg = dsc.check_data_freshness(df, today=today)

    # Assert
    assert passed is True
    assert "lag=0d" in msg


def test_check_data_freshness_yesterday_passes():
    # Arrange — yesterday OK (lag=1d <= 3d)
    today = pd.Timestamp("2026-05-22")
    df = make_kline([{"code": "000001", "date": "2026-05-21", "close": 10.0}])

    # Act
    passed, _ = dsc.check_data_freshness(df, today=today)

    # Assert
    assert passed is True


def test_check_data_freshness_five_days_old_fails():
    # Arrange — 5 天前 (> 3 天阈值)
    today = pd.Timestamp("2026-05-22")
    df = make_kline([{"code": "000001", "date": "2026-05-17", "close": 10.0}])

    # Act
    passed, msg = dsc.check_data_freshness(df, today=today)

    # Assert
    assert passed is False
    assert "lag=5d" in msg


# ---------- check_coverage ----------

def test_check_coverage_full_passes(tmp_path: Path):
    # Arrange — universe 5, all present
    universe_csv = tmp_path / "universe.csv"
    universe_csv.write_text("code,name\n000001,A\n000002,B\n000003,C\n000004,D\n000005,E\n")
    df = make_kline([
        {"code": f"00000{i}", "date": "2026-05-22", "close": 10.0} for i in range(1, 6)
    ])

    # Act
    passed, msg = dsc.check_coverage(df, universe_path=universe_csv)

    # Assert
    assert passed is True
    assert "100.0%" in msg


def test_check_coverage_eighty_pct_boundary_passes(tmp_path: Path):
    # Arrange — universe 5, 4 present (80%)
    universe_csv = tmp_path / "universe.csv"
    universe_csv.write_text("code,name\n000001,A\n000002,B\n000003,C\n000004,D\n000005,E\n")
    df = make_kline([
        {"code": f"00000{i}", "date": "2026-05-22", "close": 10.0} for i in range(1, 5)
    ])

    # Act
    passed, msg = dsc.check_coverage(df, universe_path=universe_csv)

    # Assert
    assert passed is True
    assert "80.0%" in msg


def test_check_coverage_fifty_pct_fails(tmp_path: Path):
    # Arrange — universe 4, 2 present (50%)
    universe_csv = tmp_path / "universe.csv"
    universe_csv.write_text("code,name\n000001,A\n000002,B\n000003,C\n000004,D\n")
    df = make_kline([
        {"code": f"00000{i}", "date": "2026-05-22", "close": 10.0} for i in range(1, 3)
    ])

    # Act
    passed, msg = dsc.check_coverage(df, universe_path=universe_csv)

    # Assert
    assert passed is False
    assert "50.0%" in msg


# ---------- check_no_extreme_low ----------

def test_check_no_extreme_low_clean_passes():
    # Arrange
    df = make_clean_recent()

    # Act
    passed, _ = dsc.check_no_extreme_low(df)

    # Assert
    assert passed is True


def test_check_no_extreme_low_one_low_fails():
    # Arrange
    df = make_clean_recent()
    df.loc[0, "close"] = 0.3

    # Act
    passed, msg = dsc.check_no_extreme_low(df)

    # Assert
    assert passed is False
    assert "1 rows" in msg


# ---------- main strict vs lenient ----------

def _patch_paths(monkeypatch, tmp_path: Path, kl_df: pd.DataFrame, universe_rows: int = 5):
    """KLINE_PATH / UNIVERSE_PATH / LOG_PATH → tmp_path."""
    kline_path = tmp_path / "baidu_kline.parquet"
    kl_df.to_parquet(kline_path)
    universe_path = tmp_path / "universe.csv"
    lines = ["code,name"] + [f"00000{i},S{i}" for i in range(1, universe_rows + 1)]
    universe_path.write_text("\n".join(lines) + "\n")
    log_path = tmp_path / "sanity_check_log.csv"

    monkeypatch.setattr(dsc, "KLINE_PATH", kline_path)
    monkeypatch.setattr(dsc, "UNIVERSE_PATH", universe_path)
    monkeypatch.setattr(dsc, "LOG_PATH", log_path)
    return log_path


def test_main_all_pass_returns_zero(monkeypatch, tmp_path: Path):
    # Arrange — universe 5 stocks, latest date = today, clean
    today = pd.Timestamp.today().normalize()
    rows = []
    for s in range(1, 6):
        code = f"00000{s}"
        for i in range(10):
            rows.append({"code": code, "date": today - pd.Timedelta(days=9 - i), "close": 10.0 + i * 0.05})
    df = pd.DataFrame(rows)
    log_path = _patch_paths(monkeypatch, tmp_path, df, universe_rows=5)

    # Act
    code = dsc.main(strict=True, quiet=True)

    # Assert
    assert code == 0
    log = pd.read_csv(log_path)
    assert bool(log.iloc[-1]["overall_pass"]) is True


def test_main_critical_fail_blocks_both_modes(monkeypatch, tmp_path: Path):
    # Arrange — 注入 negative close (CRITICAL)
    today = pd.Timestamp.today().normalize()
    rows = []
    for s in range(1, 6):
        code = f"00000{s}"
        for i in range(10):
            rows.append({"code": code, "date": today - pd.Timedelta(days=9 - i), "close": 10.0})
    rows[0]["close"] = -1.0
    df = pd.DataFrame(rows)
    _patch_paths(monkeypatch, tmp_path, df, universe_rows=5)

    # Act
    code_strict = dsc.main(strict=True, quiet=True)
    code_lenient = dsc.main(strict=False, quiet=True)

    # Assert
    assert code_strict == 99
    assert code_lenient == 99


def test_main_non_critical_fail_strict_blocks_lenient_passes(monkeypatch, tmp_path: Path):
    # Arrange — 注入 freshness fail (latest 10 天前, 非 critical)
    today = pd.Timestamp.today().normalize()
    old = today - pd.Timedelta(days=10)
    rows = []
    for s in range(1, 6):
        code = f"00000{s}"
        for i in range(5):
            rows.append({"code": code, "date": old - pd.Timedelta(days=4 - i), "close": 10.0 + i * 0.01})
    df = pd.DataFrame(rows)
    _patch_paths(monkeypatch, tmp_path, df, universe_rows=5)

    # Act
    code_strict = dsc.main(strict=True, quiet=True)
    code_lenient = dsc.main(strict=False, quiet=True)

    # Assert
    assert code_strict == 99
    assert code_lenient == 0


def test_main_log_csv_schema(monkeypatch, tmp_path: Path):
    # Arrange — 干净数据
    today = pd.Timestamp.today().normalize()
    rows = []
    for s in range(1, 6):
        code = f"00000{s}"
        for i in range(5):
            rows.append({"code": code, "date": today - pd.Timedelta(days=4 - i), "close": 10.0 + i * 0.05})
    df = pd.DataFrame(rows)
    log_path = _patch_paths(monkeypatch, tmp_path, df, universe_rows=5)

    # Act
    dsc.main(strict=False, quiet=True)
    log = pd.read_csv(log_path)

    # Assert — 列顺序匹配 schema
    expected_cols = [
        "date", "check_neg_close", "check_extreme_jump", "check_freshness",
        "check_coverage", "check_extreme_low", "overall_pass", "fail_details",
    ]
    assert list(log.columns) == expected_cols
    assert len(log) == 1
