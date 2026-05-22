"""Slice the full A-share scan by index membership (HS300 / CSI500 / CSI1000).

Live-fetches each constituent list from akshare (csindex endpoint, stable),
joins against the existing deepseek scan results, renders one report per index.

Run:  python examples/scan_multi_subset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from claude_finance.decision import render_scan_report

OUT_DIR = Path(__file__).resolve().parent
SCAN_RESULTS = OUT_DIR / "deepseek_scan_results.json"

INDICES = [
    ("沪深300", "000300", "hs300", 30),
    ("中证500", "000905", "csi500", 30),
    ("中证1000", "000852", "csi1000", 30),
]


def _exchange_prefix(exchange: str) -> str:
    if "上海" in exchange:
        return "sh"
    if "深圳" in exchange:
        return "sz"
    if "北京" in exchange:
        return "bj"
    raise ValueError(f"unknown exchange: {exchange!r}")


def fetch_index_components(index_code: str) -> dict[str, dict]:
    """Return {csv_code: {'name': str, 'weight': float}}."""
    import akshare as ak

    raw = ak.index_stock_cons_weight_csindex(symbol=index_code)
    out = {}
    for _, row in raw.iterrows():
        csv_code = _exchange_prefix(row["交易所"]) + str(row["成分券代码"])
        out[csv_code] = {"name": row["成分券名称"], "weight": float(row["权重"])}
    return out


def slice_and_report(
    label: str, slug: str, components: dict[str, dict], all_results: list[dict], top_n: int
) -> dict:
    """Filter scan results to this index's constituents and write report files."""
    subset = []
    for r in all_results:
        code = r.get("code", "")
        if code in components:
            r = dict(r)
            r["name"] = components[code]["name"]
            r["index_weight"] = components[code]["weight"]
            subset.append(r)

    ok = [r for r in subset if "error" not in r]
    n_buy = sum(1 for r in ok if r["signal"] == "BUY")
    n_sell = sum(1 for r in ok if r["signal"] == "SELL")
    n_hold = sum(1 for r in ok if r["signal"] == "HOLD")
    n_uptrend = sum(1 for r in ok if r["trend_ma60"] == "上升")

    w = sum(r["index_weight"] for r in ok) or 1.0
    weighted_chg = sum(r["change_pct"] * r["index_weight"] for r in ok) / w
    weighted_buy = sum(r["buy_score"] * r["index_weight"] for r in ok) / w
    weighted_sell = sum(r["sell_score"] * r["index_weight"] for r in ok) / w

    report_md = OUT_DIR / f"{slug}_report.md"
    results_json = OUT_DIR / f"{slug}_results.json"
    results_json.write_text(
        json.dumps(subset, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    report_md.write_text(render_scan_report(subset, top_n=top_n), encoding="utf-8")

    return {
        "label": label,
        "matched": len(subset),
        "components_total": len(components),
        "ok": len(ok),
        "buy_pct": n_buy / len(ok) * 100 if ok else 0,
        "sell_pct": n_sell / len(ok) * 100 if ok else 0,
        "hold_pct": n_hold / len(ok) * 100 if ok else 0,
        "uptrend_pct": n_uptrend / len(ok) * 100 if ok else 0,
        "w_chg": weighted_chg,
        "w_buy": weighted_buy,
        "w_sell": weighted_sell,
        "report_md": str(report_md),
    }


def main() -> None:
    print(f"[1/{len(INDICES) + 1}] loading scan results...")
    all_results = json.loads(SCAN_RESULTS.read_text(encoding="utf-8"))
    print(f"  全市场扫描: {len(all_results)} 行")

    summary = []
    for i, (label, code, slug, top_n) in enumerate(INDICES, 2):
        print(f"\n[{i}/{len(INDICES) + 1}] 拉 {label} ({code}) 成分股...")
        components = fetch_index_components(code)
        print(f"  ✓ {len(components)} 只")
        s = slice_and_report(label, slug, components, all_results, top_n)
        summary.append(s)
        print(
            f"  匹配 {s['matched']}/{s['components_total']}  "
            f"BUY={s['buy_pct']:.1f}% SELL={s['sell_pct']:.1f}% 上升={s['uptrend_pct']:.1f}%  "
            f"加权涨跌={s['w_chg']:+.2f}%  买/卖分={s['w_buy']:.3f}/{s['w_sell']:.3f}"
        )

    print("\n\n=== 对照表 ===")
    df = pd.DataFrame(summary).round(2)
    print(
        df[
            [
                "label",
                "matched",
                "components_total",
                "buy_pct",
                "sell_pct",
                "uptrend_pct",
                "w_chg",
                "w_buy",
                "w_sell",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
