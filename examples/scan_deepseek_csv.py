"""Offline full-market scan using a pre-downloaded CSV (no network).

Reads data/deepseek_trading.csv (5,518 A-share codes, ~13 months) and runs
the multi-strategy fusion engine on every stock with >= 90 bars, then writes
a Top-50 BUY / SELL report.

Run:  python examples/scan_deepseek_csv.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from claude_finance.data import load_market_csv
from claude_finance.decision import analyze_one, render_scan_report

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "deepseek_trading.csv"
TOP_N = 50
TAIL_BARS = 180  # use the most recent 180 bars per stock (qbot convention)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    report_md = out_dir / "deepseek_scan_report.md"
    results_json = out_dir / "deepseek_scan_results.json"

    print(f"[1/3] loading {CSV_PATH.name}...")
    t = time.time()
    market = load_market_csv(CSV_PATH, strip_prefix=False, min_bars=90)
    print(f"  ok ({time.time() - t:.1f}s) — {len(market)} stocks with >= 90 bars")

    print(f"[2/3] running multi-strategy fusion on {len(market)} stocks...")
    t = time.time()
    results: list[dict] = []
    fail = 0
    for i, (code, df) in enumerate(market.items(), 1):
        try:
            res = analyze_one(df.tail(TAIL_BARS), code)
            res["code"] = code
            results.append(res)
        except Exception as e:
            results.append({"code": code, "name": code, "error": str(e)[:120]})
            fail += 1
        if i % 500 == 0:
            elapsed = time.time() - t
            print(f"  [{i}/{len(market)}] ok={i - fail} fail={fail}  ({elapsed:.1f}s, {i / elapsed:.0f}/s)")

    elapsed = time.time() - t
    print(f"  done ({elapsed:.1f}s, {len(results) / elapsed:.0f}/s)  ok={len(results) - fail} fail={fail}")

    print("[3/3] writing report...")
    results_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    report_md.write_text(render_scan_report(results, top_n=TOP_N), encoding="utf-8")
    print(f"\n报告: {report_md}")
    print(f"原始: {results_json}")


if __name__ == "__main__":
    main()
