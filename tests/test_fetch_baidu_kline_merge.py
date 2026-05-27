"""Test merge-mode behavior of fetch_baidu_kline.main().

验证 3 个 root-cause fix:
  1. existing OUT_PATH 数据保留 (fetch 失败的 code 不会丢)
  2. new fetch 覆盖同 (code, date) 旧数据 (dedup keep='last')
  3. corrupt_codes_v3.txt 的股 保留 existing hfq fix, 不被 new qfq 覆盖

通过 monkeypatch fetch_one + 写临时 OUT_PATH/universe/corrupt_codes 实现,
不发任何真实 HTTP 请求.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "examples" / "fetch_baidu_kline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_baidu_kline_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_baidu_kline_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_kline(code: str, dates: list[str], close: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame({
        "code": code,
        "date": pd.to_datetime(dates),
        "open": close, "close": close, "high": close * 1.01, "low": close * 0.99,
        "vol": 1000.0, "amount": 10000.0, "ma5": close, "ma10": close, "ma20": close,
        "turnoverratio": 1.0,
    })


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Redirect module paths to tmp dir + provide universe + corrupt-codes file."""
    mod = _load_module()

    out_path = tmp_path / "baidu_kline.parquet"
    checkpoint = tmp_path / "baidu_kline_partial.parquet"
    universe = tmp_path / "universe.csv"
    corrupt = tmp_path / "corrupt_codes_v3.txt"

    monkeypatch.setattr(mod, "OUT_PATH", out_path)
    monkeypatch.setattr(mod, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(mod, "UNIVERSE_PATH", universe)
    monkeypatch.setattr(mod, "CORRUPT_CODES_PATH", corrupt)

    # universe: 5 股
    universe.write_text(
        "code,name,market\n"
        "000001,平安银行,深主板\n"
        "000002,万科A,深主板\n"
        "000858,五粮液,深主板\n"
        "002756,股A,深主板\n"
        "600519,贵州茅台,沪主板\n"
    )
    # corrupt list: 002756 + 600519 受保护(测试用)
    corrupt.write_text("002756\n600519\n")
    return {
        "mod": mod, "tmp": tmp_path,
        "out_path": out_path, "universe": universe, "corrupt": corrupt,
    }


def _baidu_raw(code: str, dates: list[str], close: str = "11.0") -> pd.DataFrame:
    """Mock baidu API 返回的 raw shape (走 normalize() 转 long format)."""
    n = len(dates)
    return pd.DataFrame({
        "code": code,
        "time": dates,
        "open": [close] * n, "close": [close] * n,
        "high": [str(float(close) + 0.1)] * n, "low": [str(float(close) - 0.1)] * n,
        "volume": ["1100"] * n, "amount": ["12100"] * n,
        "ma5avgprice": [close] * n, "ma10avgprice": [close] * n, "ma20avgprice": [close] * n,
        "turnoverratio": ["1.0"] * n,
    })


def test_merge_preserves_existing_codes_when_fetch_fails(env, monkeypatch):
    """fetch 失败的 code 整支保留 existing 数据."""
    mod = env["mod"]
    out_path = env["out_path"]

    # existing: 5 股各 2 天
    existing = pd.concat([
        _make_kline(c, ["2026-05-22", "2026-05-25"])
        for c in ["000001", "000002", "000858", "002756", "600519"]
    ], ignore_index=True)
    existing.to_parquet(out_path, index=False)

    # Mock: 只 000001 + 000002 + 000858 成功, 002756 + 600519 失败 (None)
    def mock_fetch_one(code, retries=3):
        if code in ("000001", "000002", "000858"):
            return _baidu_raw(code, ["2026-05-26"], close="11.0")
        return None
    monkeypatch.setattr(mod, "fetch_one", mock_fetch_one)

    monkeypatch.setattr(sys, "argv", ["fetch_baidu_kline.py"])
    mod.main()

    result = pd.read_parquet(out_path)
    result["code"] = result["code"].astype(str).str.zfill(6)

    assert "002756" in result["code"].values, "002756 fetch 失败但应保留 existing"
    assert "600519" in result["code"].values, "600519 fetch 失败但应保留 existing"
    assert set(result["code"].unique()) >= {
        "000001", "000002", "000858", "002756", "600519"
    }


def test_merge_new_overwrites_same_date(env, monkeypatch):
    """new fetch 覆盖同 (code, date) 的 existing 行."""
    mod = env["mod"]
    out_path = env["out_path"]

    # existing: 000001 在 2026-05-26 上 close=10
    existing = _make_kline(
        "000001", ["2026-05-22", "2026-05-25", "2026-05-26"], close=10.0
    )
    existing.to_parquet(out_path, index=False)

    # new fetch: 000001 在 2026-05-26 上 close=11 (应覆盖)
    def mock_fetch_one(code, retries=3):
        if code == "000001":
            return _baidu_raw(code, ["2026-05-26"], close="11.0")
        return None
    monkeypatch.setattr(mod, "fetch_one", mock_fetch_one)

    monkeypatch.setattr(sys, "argv", ["fetch_baidu_kline.py"])
    mod.main()

    result = pd.read_parquet(out_path)
    row = result[(result["code"] == "000001") & (result["date"] == "2026-05-26")]
    assert len(row) == 1, "(code,date) 应去重为 1 行"
    assert row.iloc[0]["close"] == 11.0, "new fetch close=11 应覆盖 existing close=10"


def test_corrupt_codes_preserved_from_existing(env, monkeypatch):
    """corrupt_codes_v3.txt 的股: existing 数据保留, 不被 new fetch 覆盖."""
    mod = env["mod"]
    out_path = env["out_path"]

    # existing: 002756 close=99 (modeled hfq fix value)
    existing = _make_kline("002756", ["2026-05-22", "2026-05-25"], close=99.0)
    existing.to_parquet(out_path, index=False)

    # new fetch: 002756 给 corrupt 数据 close=-0.5 — 必须被跳过
    def mock_fetch_one(code, retries=3):
        if code == "002756":
            return _baidu_raw(code, ["2026-05-22", "2026-05-25"], close="-0.5")
        return None
    monkeypatch.setattr(mod, "fetch_one", mock_fetch_one)

    monkeypatch.setattr(sys, "argv", ["fetch_baidu_kline.py"])
    mod.main()

    result = pd.read_parquet(out_path)
    rows = result[result["code"] == "002756"]
    assert len(rows) >= 2, "002756 existing 2 行必须保留"
    assert (rows["close"] >= 0).all(), "002756 不能被 corrupt qfq (close<0) 覆盖"
    assert (rows["close"] == 99.0).all(), "002756 应保持 existing hfq 值 99"


def test_atomic_write_no_tmp_left_after_success(env, monkeypatch):
    """成功写入后 .tmp 不应残留."""
    mod = env["mod"]
    out_path = env["out_path"]
    tmp_artifact = out_path.with_suffix(".parquet.tmp")

    existing = _make_kline("000001", ["2026-05-25"])
    existing.to_parquet(out_path, index=False)

    def mock_fetch_one(code, retries=3):
        if code == "000001":
            return _baidu_raw(code, ["2026-05-26"], close="12.0")
        return None
    monkeypatch.setattr(mod, "fetch_one", mock_fetch_one)

    monkeypatch.setattr(sys, "argv", ["fetch_baidu_kline.py"])
    mod.main()

    assert out_path.exists()
    assert not tmp_artifact.exists(), ".tmp 应在成功 rename 后消失"


def test_first_run_no_existing_writes_clean(env, monkeypatch):
    """首次运行 (OUT_PATH 不存在): 直接写, 无 merge 错误."""
    mod = env["mod"]
    out_path = env["out_path"]
    assert not out_path.exists()

    def mock_fetch_one(code, retries=3):
        return _baidu_raw(code, ["2026-05-26"], close="10.0")
    monkeypatch.setattr(mod, "fetch_one", mock_fetch_one)

    monkeypatch.setattr(sys, "argv", ["fetch_baidu_kline.py"])
    mod.main()

    result = pd.read_parquet(out_path)
    codes = set(result["code"].astype(str).str.zfill(6).unique())
    # corrupt-protected 002756 + 600519 在 new_big 里被 filter 掉, 首次运行也无 existing
    assert "000001" in codes
    assert "000002" in codes
    assert "000858" in codes
