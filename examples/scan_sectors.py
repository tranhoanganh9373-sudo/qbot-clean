"""Slice the full A-share scan by Eastmoney industry sector.

Live-fetches each sector's constituent list from akshare. The eastmoney
push2 endpoint is sometimes blocked at the subdomain level (17/29.push2...)
in sandboxes, so each sector is retried 3 times with backoff and silently
skipped if all attempts fail.

Run:  python examples/scan_sectors.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from claude_finance.decision import render_scan_report

OUT_DIR = Path(__file__).resolve().parent
SCAN_RESULTS = OUT_DIR / "deepseek_scan_results.json"

SECTORS = [
    ("白酒", "baijiu"),
    ("银行", "bank"),
    ("半导体", "semi"),
    ("医药", "pharma"),
    ("房地产开发", "realestate"),
    ("光伏设备", "solar"),
    ("汽车整车", "auto"),
    ("有色金属", "metals"),
]
TOP_N = 20
MAX_RETRIES = 3


def _add_prefix(bare_code: str) -> str:
    """A-share bare code -> CSV-style prefixed code (sh/sz/bj)."""
    if bare_code.startswith("6"):
        return "sh" + bare_code
    if bare_code.startswith(("0", "3", "1", "2")):
        return "sz" + bare_code
    if bare_code.startswith(("8", "9", "4")):
        return "bj" + bare_code
    return bare_code


def fetch_sector(name: str) -> dict[str, str] | None:
    """Return {csv_code: stock_name} or None after retries fail."""
    import akshare as ak

    last_err = ""
    for attempt in range(MAX_RETRIES):
        try:
            raw = ak.stock_board_industry_cons_em(symbol=name)
            out = {}
            for _, row in raw.iterrows():
                csv_code = _add_prefix(str(row["代码"]))
                out[csv_code] = row["名称"]
            return out
        except Exception as e:
            last_err = str(e)[:80]
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    print(f"    ✗ 全部重试失败: {last_err}")
    return None


def slice_and_report(
    label: str, slug: str, components: dict[str, str], all_results: list[dict]
) -> dict:
    subset = []
    for r in all_results:
        code = r.get("code", "")
        if code in components:
            r = dict(r)
            r["name"] = components[code]
            subset.append(r)

    ok = [r for r in subset if "error" not in r]
    n_buy = sum(1 for r in ok if r["signal"] == "BUY")
    n_sell = sum(1 for r in ok if r["signal"] == "SELL")
    n_uptrend = sum(1 for r in ok if r["trend_ma60"] == "上升")
    avg_chg = sum(r["change_pct"] for r in ok) / len(ok) if ok else 0
    avg_buy = sum(r["buy_score"] for r in ok) / len(ok) if ok else 0
    avg_sell = sum(r["sell_score"] for r in ok) / len(ok) if ok else 0

    report_md = OUT_DIR / f"sector_{slug}_report.md"
    results_json = OUT_DIR / f"sector_{slug}_results.json"
    results_json.write_text(
        json.dumps(subset, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    report_md.write_text(render_scan_report(subset, top_n=TOP_N), encoding="utf-8")

    return {
        "label": label,
        "matched": len(subset),
        "components_total": len(components),
        "ok": len(ok),
        "buy_pct": n_buy / len(ok) * 100 if ok else 0,
        "sell_pct": n_sell / len(ok) * 100 if ok else 0,
        "uptrend_pct": n_uptrend / len(ok) * 100 if ok else 0,
        "avg_chg": avg_chg,
        "avg_buy": avg_buy,
        "avg_sell": avg_sell,
        "report_md": str(report_md),
    }


def main() -> None:
    print(f"[1/{len(SECTORS) + 2}] loading scan results...")
    all_results = json.loads(SCAN_RESULTS.read_text(encoding="utf-8"))

    summary = []
    failed = []
    for i, (name, slug) in enumerate(SECTORS, 2):
        print(f"\n[{i}/{len(SECTORS) + 2}] 拉 {name}...")
        components = fetch_sector(name)
        if components is None:
            failed.append(name)
            continue
        print(f"  ✓ {len(components)} 只成分股")
        s = slice_and_report(name, slug, components, all_results)
        summary.append(s)
        print(
            f"  匹配 {s['matched']}/{s['components_total']}  "
            f"BUY={s['buy_pct']:.1f}% SELL={s['sell_pct']:.1f}% 上升={s['uptrend_pct']:.1f}%  "
            f"均涨跌={s['avg_chg']:+.2f}%  均买/卖分={s['avg_buy']:.3f}/{s['avg_sell']:.3f}"
        )

    print(f"\n\n[{len(SECTORS) + 2}/{len(SECTORS) + 2}] 汇总...")
    if not summary:
        print("  ✗ 没有任何行业拉取成功")
        return

    df = pd.DataFrame(summary).round(3)
    df["bias"] = df["avg_buy"] - df["avg_sell"]
    df = df.sort_values("bias", ascending=False)

    summary_md = ["# 行业切片对照", "", "排序: 多空 bias (买分 - 卖分) 降序", ""]
    summary_md.append("| 行业 | 成分 | 匹配 | BUY% | SELL% | 上升% | 均涨跌 | 买分 | 卖分 | bias |")
    summary_md.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        summary_md.append(
            f"| {r['label']} | {r['components_total']} | {r['matched']} | "
            f"{r['buy_pct']:.1f}% | {r['sell_pct']:.1f}% | {r['uptrend_pct']:.1f}% | "
            f"{r['avg_chg']:+.2f}% | {r['avg_buy']:.3f} | {r['avg_sell']:.3f} | {r['bias']:+.3f} |"
        )
    if failed:
        summary_md += ["", f"**拉取失败的行业**（沙盒拦截）: {', '.join(failed)}"]
    (OUT_DIR / "sector_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    print("\n=== 对照表 ===")
    print(
        df[
            [
                "label",
                "matched",
                "buy_pct",
                "sell_pct",
                "uptrend_pct",
                "avg_chg",
                "avg_buy",
                "avg_sell",
                "bias",
            ]
        ].to_string(index=False)
    )
    if failed:
        print(f"\n拉取失败: {failed}")
    print(f"\n汇总: {OUT_DIR / 'sector_summary.md'}")


if __name__ == "__main__":
    main()
