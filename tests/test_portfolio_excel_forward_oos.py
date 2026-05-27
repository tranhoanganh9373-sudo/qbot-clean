"""Forward OOS Track sheet — pure-function tests.

无网络, 无文件 I/O. 直接喂合成 log_df / kline_df 给 compute_forward_oos_track +
compute_forward_oos_stats, 验证:
  - 空 log → 返回空 DataFrame, 不崩
  - 1 月数据 → monthly_return / cum_return / start_value / end_value 算对
  - 3 月数据 → ann_return 是几何 (1+m1)(1+m2)(1+m3) ^ (12/3) - 1
  - 6 月数据 → Sharpe = mean/std * sqrt(12) 算对
  - 12 月含负月 → MDD / Calmar 算对
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 直接从 examples/portfolio_excel.py 加载 (它不是 package 模块)
_SPEC = importlib.util.spec_from_file_location(
    "portfolio_excel_mod",
    Path(__file__).resolve().parent.parent / "examples" / "portfolio_excel.py",
)
assert _SPEC and _SPEC.loader
portfolio_excel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(portfolio_excel)

compute_forward_oos_track = portfolio_excel.compute_forward_oos_track
compute_forward_oos_stats = portfolio_excel.compute_forward_oos_stats
FORWARD_OOS_HEADERS = portfolio_excel.FORWARD_OOS_HEADERS


def _make_kline(specs: dict[str, list[tuple[str, float]]]) -> pd.DataFrame:
    """specs: code -> [(date_str, close), ...] → DataFrame [code, date, close]."""
    rows = []
    for code, points in specs.items():
        for d, c in points:
            rows.append({"code": code, "date": pd.Timestamp(d), "close": float(c)})
    if not rows:
        return pd.DataFrame(columns=["code", "date", "close"])
    return pd.DataFrame(rows)


def _make_log(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """rows: [(date, action, symbol, price), ...] (score=0.5, name='X' 填充)."""
    return pd.DataFrame([
        {"date": d, "action": a, "symbol": s, "name": "X", "score": 0.5, "price": p}
        for (d, a, s, p) in rows
    ])


# ---------- 1. empty log ----------
def test_empty_log_returns_empty_df():
    log_df = pd.DataFrame(columns=["date", "action", "symbol", "name", "score", "price"])
    kline_df = pd.DataFrame(columns=["code", "date", "close"])
    out = compute_forward_oos_track(log_df, kline_df, holdings=[])
    assert list(out.columns) == FORWARD_OOS_HEADERS
    assert len(out) == 0


def test_log_only_before_start_returns_empty():
    """log 全部在 2026-05-25 之前 → 视作还没真 OOS, 空 DataFrame."""
    log_df = _make_log([
        ("2026-05-22", "BUY", "SH600000", 10.0),
        ("2026-05-23", "BUY", "SH600001", 20.0),
    ])
    kline_df = _make_kline({"600000": [("2026-05-22", 10), ("2026-05-23", 11)]})
    out = compute_forward_oos_track(log_df, kline_df, holdings=[])
    assert len(out) == 0


def test_empty_log_stats_no_crash():
    log_df = pd.DataFrame(columns=["date", "action", "symbol", "name", "score", "price"])
    kline_df = pd.DataFrame(columns=["code", "date", "close"])
    track = compute_forward_oos_track(log_df, kline_df, holdings=[])
    stats = compute_forward_oos_stats(track)
    assert stats["累积月数"] == 0
    assert stats["forward cum return %"] == 0.0
    assert math.isnan(stats["forward Sharpe"])
    assert math.isnan(stats["forward ann return %"])


# ---------- 2. 1 month ----------
def test_one_month_monthly_and_cum_return():
    """月初 BUY SH600000 @ 100, 月末 close 110 → +10%.
    start_value=50000, end_value=55000, cum=10%.
    """
    log_df = _make_log([
        ("2026-06-01", "BUY", "SH600000", 100.0),
    ])
    kline_df = _make_kline({
        "600000": [
            ("2026-06-01", 100.0),
            ("2026-06-15", 105.0),
            ("2026-06-30", 110.0),
        ],
    })
    out = compute_forward_oos_track(log_df, kline_df, holdings=["SH600000"],
                                     total_capital=50000.0)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["start_value"] == 50000.0
    assert row["end_value"] == pytest.approx(55000.0, rel=1e-6)
    assert row["monthly_return"] == pytest.approx(0.10, rel=1e-6)
    assert row["cum_return"] == pytest.approx(0.10, rel=1e-6)
    assert row["n_picks_used"] == 1
    assert row["picks_taken"] == "SH600000"

    stats = compute_forward_oos_stats(out)
    assert stats["累积月数"] == 1
    assert stats["forward cum return %"] == pytest.approx(10.0, rel=1e-4)
    # n<3 → ann/MDD/Sharpe 都 NaN
    assert math.isnan(stats["forward ann return %"])
    assert math.isnan(stats["forward Sharpe"])
    assert math.isnan(stats["forward MDD %"])


def test_no_picks_month_returns_start_value():
    """有 log 行但都 SELL → picks 空 → 月末值 = 月初值 (现金不变)."""
    log_df = _make_log([
        ("2026-06-01", "SELL", "SH600000", 0.0),
        ("2026-06-05", "SELL", "SH600001", 0.0),
    ])
    kline_df = _make_kline({"600000": [("2026-06-01", 100)]})
    out = compute_forward_oos_track(log_df, kline_df, holdings=[],
                                     total_capital=50000.0)
    assert len(out) == 1
    assert out.iloc[0]["start_value"] == 50000.0
    assert out.iloc[0]["end_value"] == 50000.0
    assert out.iloc[0]["monthly_return"] == 0.0
    assert out.iloc[0]["n_picks_used"] == 0


# ---------- 3. 3 months → ann return ----------
def _build_n_month_log_and_kline(monthly_rets: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """合成 N 月 log + kline, 每月 1 只 SH600000, entry=100, exit=100*(1+r).
    每月 BUY 在该月 1 号; kline 给该月 1 号 + 末日两点.
    """
    log_rows: list[tuple[str, str, str, float]] = []
    kline_rows: dict[str, list[tuple[str, float]]] = {"600000": []}
    base = pd.Timestamp("2026-06-01")
    for i, r in enumerate(monthly_rets):
        month_start = base + pd.DateOffset(months=i)
        # 月末: 月初 + 27 天 (落入同月)
        month_end = month_start + pd.Timedelta(days=27)
        log_rows.append(
            (month_start.strftime("%Y-%m-%d"), "BUY", "SH600000", 100.0)
        )
        kline_rows["600000"].append((month_start.strftime("%Y-%m-%d"), 100.0))
        kline_rows["600000"].append(
            (month_end.strftime("%Y-%m-%d"), 100.0 * (1.0 + r))
        )
    return _make_log(log_rows), _make_kline(kline_rows)


def test_three_months_ann_return_geometric():
    """3 个月: +5%, +3%, -2% → cum = 1.05*1.03*0.98 - 1; ann = cum_factor^(12/3) - 1."""
    rets = [0.05, 0.03, -0.02]
    log_df, kline_df = _build_n_month_log_and_kline(rets)
    out = compute_forward_oos_track(log_df, kline_df, holdings=["SH600000"],
                                     total_capital=50000.0)
    assert len(out) == 3
    # 每月 monthly_return 校验
    for i, r in enumerate(rets):
        assert out.iloc[i]["monthly_return"] == pytest.approx(r, abs=1e-6)
    # cum_return 几何累乘
    expected_cum = 1.0
    for r in rets:
        expected_cum *= (1.0 + r)
    expected_cum -= 1.0
    assert out.iloc[-1]["cum_return"] == pytest.approx(expected_cum, abs=1e-6)

    stats = compute_forward_oos_stats(out)
    assert stats["累积月数"] == 3
    expected_ann = ((1.0 + expected_cum) ** (12.0 / 3.0) - 1.0) * 100
    assert stats["forward ann return %"] == pytest.approx(expected_ann, rel=1e-3)
    # n=3 → MDD 出, Sharpe 仍 NaN
    assert not math.isnan(stats["forward MDD %"])
    assert math.isnan(stats["forward Sharpe"])


# ---------- 4. 6 months → Sharpe ----------
def test_six_months_sharpe_formula():
    """6 个月固定收益, 验证 Sharpe = mean/std * sqrt(12)."""
    rets = [0.04, -0.01, 0.03, 0.02, -0.02, 0.05]
    log_df, kline_df = _build_n_month_log_and_kline(rets)
    out = compute_forward_oos_track(log_df, kline_df, holdings=["SH600000"],
                                     total_capital=50000.0)
    assert len(out) == 6
    stats = compute_forward_oos_stats(out)
    assert stats["累积月数"] == 6

    arr = np.array(rets)
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    expected_sharpe = (mu / sigma) * math.sqrt(12)
    assert not math.isnan(stats["forward Sharpe"])
    assert stats["forward Sharpe"] == pytest.approx(expected_sharpe, rel=1e-3)


# ---------- 5. 12 months with negative → MDD/Calmar ----------
def test_twelve_months_mdd_calmar():
    """12 月含负月: +5,+3,-10,-5,+2,+4,-8,+6,+3,-2,+5,+4
    校验 MDD 是 cum_curve 上的最大相对回撤; Calmar = ann / |MDD|.
    """
    rets = [0.05, 0.03, -0.10, -0.05, 0.02, 0.04, -0.08, 0.06, 0.03, -0.02, 0.05, 0.04]
    log_df, kline_df = _build_n_month_log_and_kline(rets)
    out = compute_forward_oos_track(log_df, kline_df, holdings=["SH600000"],
                                     total_capital=50000.0)
    assert len(out) == 12
    stats = compute_forward_oos_stats(out)
    assert stats["累积月数"] == 12

    # MDD 手算
    curve = np.cumprod(1 + np.array(rets))
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    expected_mdd_pct = dd.min() * 100
    assert stats["forward MDD %"] == pytest.approx(expected_mdd_pct, rel=1e-3)
    assert stats["forward MDD %"] < 0

    # ann return / Calmar
    cum_factor = curve[-1]
    expected_ann = (cum_factor ** (12.0 / 12.0) - 1.0) * 100
    assert stats["forward ann return %"] == pytest.approx(expected_ann, rel=1e-3)
    expected_calmar = (expected_ann / 100) / abs(expected_mdd_pct / 100)
    assert stats["forward Calmar"] == pytest.approx(expected_calmar, rel=1e-3)


# ---------- 6. defensive: 价格缺失 / 多 picks ----------
def test_no_close_for_pick_falls_back_to_per_stock_cash():
    """picks 找不到 kline → 该股按 per_stock 现金对待, 整月收益 0."""
    log_df = _make_log([
        ("2026-06-01", "BUY", "SH999999", 50.0),
    ])
    kline_df = _make_kline({"600000": [("2026-06-01", 100), ("2026-06-30", 110)]})
    out = compute_forward_oos_track(log_df, kline_df, holdings=["SH999999"],
                                     total_capital=50000.0)
    assert len(out) == 1
    # 找不到价 → 视作整月空仓 → end_value = start_value
    assert out.iloc[0]["end_value"] == 50000.0
    assert out.iloc[0]["monthly_return"] == 0.0


def test_two_picks_equal_weight():
    """两只 picks 等权, A +20%, B 找不到 → 整月 +10% (一半涨 20, 一半不动)."""
    log_df = _make_log([
        ("2026-06-01", "BUY", "SH600000", 100.0),
        ("2026-06-02", "BUY", "SH999999", 50.0),
    ])
    kline_df = _make_kline({
        "600000": [("2026-06-01", 100), ("2026-06-30", 120)],
    })
    out = compute_forward_oos_track(log_df, kline_df,
                                     holdings=["SH600000", "SH999999"],
                                     total_capital=50000.0)
    assert len(out) == 1
    # 一半 25000 涨 20% → 30000, 另一半 25000 不动 → 总 55000
    assert out.iloc[0]["end_value"] == pytest.approx(55000.0, rel=1e-6)
    assert out.iloc[0]["monthly_return"] == pytest.approx(0.10, abs=1e-6)
