"""Filter the full A-share scan down to CSI 300 constituents only.

Live-fetches the CSI 300 constituent list from akshare (sina-style endpoint
that works even in sandboxes that block eastmoney), joins it against the
already-saved deepseek scan results, and re-renders a Top-30 BUY/SELL report
restricted to large-caps.

Run:  python examples/scan_hs300_subset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from claude_finance.decision import render_scan_report
from claude_finance.scan_cache import cache_or_fetch

SCAN_RESULTS = Path(__file__).resolve().parent / "deepseek_scan_results.json"
REPORT_MD = Path(__file__).resolve().parent / "hs300_report.md"
RESULTS_JSON = Path(__file__).resolve().parent / "hs300_results.json"


def _exchange_prefix(exchange: str) -> str:
    if "上海" in exchange:
        return "sh"
    if "深圳" in exchange:
        return "sz"
    if "北京" in exchange:
        return "bj"
    raise ValueError(f"unknown exchange: {exchange!r}")


def fetch_hs300_components() -> pd.DataFrame:
    """Return DataFrame with columns: csv_code, name, weight.

    Cache: ``data_cache/scan_cache/ak_index_stock_cons_weight_csindex_000300.parquet``,
    TTL 24h (HS300 季度调样, 1 天足够).
    """
    import akshare as ak

    raw = cache_or_fetch(
        key="ak_index_stock_cons_weight_csindex_000300",
        fetcher=lambda: ak.index_stock_cons_weight_csindex(symbol="000300"),
        ttl_hours=24.0,
    )
    raw["csv_code"] = raw.apply(
        lambda r: _exchange_prefix(r["交易所"]) + str(r["成分券代码"]), axis=1
    )
    return raw[["csv_code", "成分券名称", "权重"]].rename(
        columns={"成分券名称": "name", "权重": "weight"}
    )


def main() -> None:
    print("[1/3] 联网拉沪深300成分股...")
    components = fetch_hs300_components()
    print(f"  ✓ {len(components)} 只成分股")
    code_to_name = dict(zip(components["csv_code"], components["name"], strict=True))
    code_to_weight = dict(zip(components["csv_code"], components["weight"], strict=True))

    print(f"[2/3] 加载已有扫描结果 ({SCAN_RESULTS.name})...")
    all_results = json.loads(SCAN_RESULTS.read_text(encoding="utf-8"))
    print(f"  全市场扫描结果: {len(all_results)} 行")

    hs300_results = []
    for r in all_results:
        code = r.get("code", "")
        if code in code_to_name:
            r = dict(r)
            r["name"] = code_to_name[code]
            r["hs300_weight"] = code_to_weight[code]
            hs300_results.append(r)

    matched_codes = {r["code"] for r in hs300_results}
    missing = [c for c in code_to_name if c not in matched_codes]
    print(
        f"  匹配到 {len(hs300_results)} 只 (缺 {len(missing)} 只: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''})"
    )

    ok = [r for r in hs300_results if "error" not in r]
    n_buy = sum(1 for r in ok if r["signal"] == "BUY")
    n_sell = sum(1 for r in ok if r["signal"] == "SELL")
    n_hold = sum(1 for r in ok if r["signal"] == "HOLD")
    n_uptrend = sum(1 for r in ok if r["trend_ma60"] == "上升")

    if ok:
        w = sum(r["hs300_weight"] for r in ok)
        weighted_chg = sum(r["change_pct"] * r["hs300_weight"] for r in ok) / w
        weighted_buy = sum(r["buy_score"] * r["hs300_weight"] for r in ok) / w
        weighted_sell = sum(r["sell_score"] * r["hs300_weight"] for r in ok) / w
    else:
        weighted_chg = weighted_buy = weighted_sell = 0

    print(f"\n  HS300 成分股信号分布 (n={len(ok)}):")
    print(f"    🟢 BUY  : {n_buy:>3} ({n_buy / len(ok) * 100:>5.1f}%)")
    print(f"    🔴 SELL : {n_sell:>3} ({n_sell / len(ok) * 100:>5.1f}%)")
    print(f"    ⚪ HOLD : {n_hold:>3} ({n_hold / len(ok) * 100:>5.1f}%)")
    print(f"    上升趋势: {n_uptrend} ({n_uptrend / len(ok) * 100:.1f}%)")
    print("\n  权重加权(更接近真实指数):")
    print(f"    当日加权涨跌: {weighted_chg:+.2f}%")
    print(f"    加权买分: {weighted_buy:.3f}   加权卖分: {weighted_sell:.3f}")

    print("\n[3/3] 写报告...")
    RESULTS_JSON.write_text(
        json.dumps(hs300_results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    REPORT_MD.write_text(render_scan_report(hs300_results, top_n=30), encoding="utf-8")
    print(f"  报告: {REPORT_MD}")
    print(f"  原始: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
