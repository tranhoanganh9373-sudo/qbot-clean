"""Qbot 全市场扫描（A 股）— 拉所有 A 股 → 跑多策略融合 → 输出 Top BUY/SELL。

Run:  python examples/qbot_scan_all.py

预计耗时: 30-60 分钟（5 线程，~5000 股）
输出:
  - examples/qbot_scan_report.md
  - examples/qbot_scan_results.json
  - data_cache/qbot_scan_cache.parquet  (断点续跑)
"""

from __future__ import annotations

import json
from pathlib import Path

from claude_finance.decision import render_scan_report
from claude_finance.scan import get_stock_universe, run_scan

TOP_N = 50


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    project_root = out_dir.parent
    cache_file = project_root / "data_cache" / "qbot_scan_cache.parquet"
    report_md = out_dir / "qbot_scan_report.md"
    results_json = out_dir / "qbot_scan_results.json"

    universe = get_stock_universe()
    results = run_scan(universe, cache_file=cache_file)

    results_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    report_md.write_text(render_scan_report(results, top_n=TOP_N), encoding="utf-8")
    print(f"\n报告: {report_md}")
    print(f"原始: {results_json}")
    print(f"缓存: {cache_file}  (再跑会从这里恢复)")


if __name__ == "__main__":
    main()
