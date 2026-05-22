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


def _fetch_index(ak_symbol: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_zh_index_daily(symbol=ak_symbol)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").tail(180)


def _fetch_stock(code: str, retries: int = 3) -> pd.DataFrame:
    import akshare as ak

    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=300)).strftime("%Y%m%d")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq"
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
