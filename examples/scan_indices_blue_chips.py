"""Market-context scan: 3 indices + equal-weight all-A proxy + 4 blue chips.

Tries to fetch the 3 indices (上证 / 沪深300 / 创业板) live from akshare; if
the network is blocked (sandboxed environments), falls back to the equal-weight
all-A synthetic index built from data/deepseek_trading.csv.

The 4 blue chips and the synthetic index always come from the CSV.

Run:  python examples/scan_indices_blue_chips.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from claude_finance.data import load_market_csv
from claude_finance.decision import (
    SYMBOLS,
    analyze_one,
    render_decision_report,
)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "deepseek_trading.csv"
TAIL_BARS = 180


def _try_fetch_index(ak_symbol: str) -> pd.DataFrame | None:
    """Best-effort akshare fetch; returns None on any failure (e.g. blocked network)."""
    try:
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol=ak_symbol)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").tail(TAIL_BARS)
    except Exception as e:
        print(f"  ✗ {ak_symbol} 联网拉取失败 ({str(e)[:60]})")
        return None


def _synthetic_all_a(market: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weighted daily OHLCV of every stock in the CSV.

    This is a market-breadth proxy — closer to a small-cap-tilted index than
    沪深300 (which is large-cap weighted), but useful when the real indices
    aren't reachable.
    """
    all_close = pd.concat({c: d["close"] for c, d in market.items()}, axis=1).mean(axis=1)
    all_open = pd.concat({c: d["open"] for c, d in market.items()}, axis=1).mean(axis=1)
    all_high = pd.concat({c: d["high"] for c, d in market.items()}, axis=1).mean(axis=1)
    all_low = pd.concat({c: d["low"] for c, d in market.items()}, axis=1).mean(axis=1)
    all_vol = pd.concat({c: d["volume"] for c, d in market.items()}, axis=1).sum(axis=1).astype("int64")

    df = pd.DataFrame(
        {
            "open": all_open,
            "high": all_high,
            "low": all_low,
            "close": all_close,
            "volume": all_vol,
        }
    ).dropna()
    df.index.name = "date"
    return df


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    report_md = out_dir / "indices_report.md"
    results_json = out_dir / "indices_results.json"

    print("[1/4] loading CSV...")
    t = time.time()
    market = load_market_csv(CSV_PATH, strip_prefix=False, min_bars=90)
    print(f"  ok ({time.time() - t:.1f}s) — {len(market)} stocks")

    print("\n[2/4] 尝试联网拉 3 个真实指数...")
    real_indices: dict[str, pd.DataFrame] = {}
    for s in SYMBOLS:
        if s.kind != "index":
            continue
        df = _try_fetch_index(s.ak_symbol)
        if df is not None:
            real_indices[s.code] = df
            print(f"  ✓ {s.name} ({s.code}): {len(df)} bars")

    print("\n[3/4] 合成等权全 A 宽基指数 + 提取 4 蓝筹...")
    synth_index = _synthetic_all_a(market)
    print(
        f"  ✓ 等权全A: {len(synth_index)} bars  "
        f"{synth_index.index[0].date()} -> {synth_index.index[-1].date()}"
    )

    csv_codes_for_blue = {
        "600519": "sh600519",
        "300750": "sz300750",
        "600036": "sh600036",
        "601318": "sh601318",
    }

    print("\n[4/4] 跑多策略融合...")
    results: list[dict] = []

    # 真实指数（如果拉到）
    for s in SYMBOLS:
        if s.kind == "index" and s.code in real_indices:
            res = analyze_one(real_indices[s.code], s.name)
            res["code"] = s.code
            results.append(res)
            print(
                f"  {s.name}: {res['signal']}  买分={res['buy_score']} "
                f"卖分={res['sell_score']} 趋势={res['trend_ma60']}"
            )

    # 合成宽基（永远有）
    res = analyze_one(synth_index.tail(TAIL_BARS), "等权全A宽基")
    res["code"] = "SYNTH-ALLA"
    results.append(res)
    print(
        f"  等权全A宽基: {res['signal']}  买分={res['buy_score']} "
        f"卖分={res['sell_score']} 趋势={res['trend_ma60']}"
    )

    # 4 蓝筹
    for s in SYMBOLS:
        if s.kind != "stock":
            continue
        csv_code = csv_codes_for_blue.get(s.code)
        if csv_code is None or csv_code not in market:
            print(f"  ✗ {s.name} ({s.code}): CSV 无数据")
            results.append({"name": s.name, "code": s.code, "error": "CSV 无数据"})
            continue
        res = analyze_one(market[csv_code].tail(TAIL_BARS), s.name)
        res["code"] = s.code
        results.append(res)
        print(
            f"  {s.name}: {res['signal']}  买分={res['buy_score']} "
            f"卖分={res['sell_score']} 趋势={res['trend_ma60']}"
        )

    print("\n写报告...")
    results_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    report_md.write_text(render_decision_report(results), encoding="utf-8")
    print(f"  报告: {report_md}")
    print(f"  原始: {results_json}")


if __name__ == "__main__":
    main()
