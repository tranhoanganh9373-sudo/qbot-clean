"""Qbot 多策略融合中线决策报告。

Run:  python examples/qbot_decision.py

输入: claude_finance.decision.SYMBOLS (默认 7 个标的 — 3 指数 + 4 蓝筹)
输出: examples/qbot_report.md  +  examples/qbot_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from claude_finance.decision import (
    SYMBOLS,
    analyze_one,
    render_decision_report,
)
from claude_finance.scan_cache import cache_or_fetch

QBOT_TTL_HOURS = 6.0  # interactive CLI: 半天内重复跑命中 cache


def _fetch_index(ak_symbol: str) -> pd.DataFrame:
    """Cache: ``ak_stock_zh_index_daily_{ak_symbol}.parquet``, TTL 6h."""
    import akshare as ak

    df = cache_or_fetch(
        key=f"ak_stock_zh_index_daily_{ak_symbol}",
        fetcher=lambda: ak.stock_zh_index_daily(symbol=ak_symbol),
        ttl_hours=QBOT_TTL_HOURS,
    )
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").tail(180)


def _fetch_stock(code: str, retries: int = 3) -> pd.DataFrame:
    """Cache: ``ak_stock_zh_a_hist_{code}_qfq.parquet``, TTL 6h.

    日期窗口故意不进 cache key — cache hit 时返回最近一次抓的 300d 窗口数据,
    取 tail(180) 仍然能覆盖中线决策需求.  TTL 6h 内不会跨日, 窗口偏移可忽略.
    """
    import akshare as ak

    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=300)).strftime("%Y%m%d")

    def _fetch() -> pd.DataFrame:
        return ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq",
        )

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = cache_or_fetch(
                key=f"ak_stock_zh_a_hist_{code}_qfq",
                fetcher=_fetch,
                ttl_hours=QBOT_TTL_HOURS,
            )
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date").tail(180)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err  # type: ignore[misc]


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results: list[dict] = []
    for s in SYMBOLS:
        try:
            df = _fetch_index(s.ak_symbol) if s.kind == "index" else _fetch_stock(s.code)
            res = analyze_one(df, s.name)
            res["code"] = s.code
            results.append(res)
            print(
                f"✓ {s.name} ({s.code}): {res['signal']}  价={res['price']}  "
                f"买分={res['buy_score']}/卖分={res['sell_score']}"
            )
        except Exception as e:
            print(f"✗ {s.name} ({s.code}) 失败: {e}")
            results.append({"name": s.name, "code": s.code, "error": str(e)})
        time.sleep(0.3)

    (out_dir / "qbot_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "qbot_report.md").write_text(render_decision_report(results), encoding="utf-8")
    print(f"\n报告: {out_dir / 'qbot_report.md'}")
    print(f"原始: {out_dir / 'qbot_results.json'}")


if __name__ == "__main__":
    main()
