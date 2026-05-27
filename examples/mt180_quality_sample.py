"""mt180 数据 quality 抽样 + Top-N 提取.

Quality 统计:
  - formula 字段空值率
  - formula 长度分布
  - category 分布
  - sales / fav 分布
  - 重复名 / 重复公式

Top-N 提取:
  - 按 (salesCount + 10 * favoritesCount) 排
  - 输出 top-N 到 data_cache/mt180/top_n.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data_cache" / "mt180"
DETAIL_PATH = OUT_DIR / "indicators_detail.jsonl"
REPORT_PATH = OUT_DIR / "quality_report.txt"
TOPN_PATH = OUT_DIR / "top_n.jsonl"

HEAT_W_SALES = 1.0
HEAT_W_FAV = 10.0


def _load_details() -> list[dict]:
    if not DETAIL_PATH.exists():
        print(f"ERR: {DETAIL_PATH} not found")
        return []
    out: list[dict] = []
    with DETAIL_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _heat_score(d: dict) -> float:
    return (
        HEAT_W_SALES * (d.get("salesCount") or 0)
        + HEAT_W_FAV * (d.get("favoritesCount") or 0)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()

    indicators = _load_details()
    n = len(indicators)
    print(f"loaded {n:,} indicators from {DETAIL_PATH.name}")
    if not n:
        return

    formula_lengths: list[int] = []
    no_formula = 0
    has_drawing_only = 0
    name_counter: collections.Counter[str] = collections.Counter()
    formula_counter: collections.Counter[str] = collections.Counter()
    type_counter: collections.Counter[int] = collections.Counter()

    DRAWING_FUNCS = ("STICKLINE", "DRAWICON", "DRAWTEXT", "DRAWBAND",
                     "DRAWNUMBER", "DRAWGBK", "DRAWLINE", "POLYLINE")

    for d in indicators:
        f = d.get("formula") or ""
        formula_lengths.append(len(f))
        if not f.strip():
            no_formula += 1
        else:
            calc_lines = [
                ln for ln in f.split("\n")
                if ":" in ln and not any(dr in ln.upper() for dr in DRAWING_FUNCS)
            ]
            if not calc_lines:
                has_drawing_only += 1
        name_counter[d.get("name", "")] += 1
        formula_counter[f] += 1
        type_counter[d.get("indicatorType", -1)] += 1

    formula_lengths.sort()
    p50 = formula_lengths[n // 2] if n else 0
    p90 = formula_lengths[int(n * 0.9)] if n else 0
    p99 = formula_lengths[int(n * 0.99)] if n else 0
    max_len = formula_lengths[-1] if n else 0

    dup_names = sum(1 for c in name_counter.values() if c > 1)
    dup_formulas = sum(1 for c in formula_counter.values() if c > 1)

    sales = [d.get("salesCount") or 0 for d in indicators]
    favs = [d.get("favoritesCount") or 0 for d in indicators]
    sales_top10 = sorted(sales, reverse=True)[:10]
    favs_top10 = sorted(favs, reverse=True)[:10]

    lines = [
        "mt180 indicators quality report",
        "=" * 50, "",
        f"Total: {n:,}", "",
        "## Formula 字段",
        f"  - 空 formula:        {no_formula:,} ({no_formula/n*100:.1f}%)",
        f"  - 仅绘图 (无计算):    {has_drawing_only:,} ({has_drawing_only/n*100:.1f}%)",
        f"  - 长度分位 (chars):  p50={p50}  p90={p90}  p99={p99}  max={max_len:,}",
        "",
        "## 重复",
        f"  - 重复名:            {dup_names:,} 唯一名 ({len(name_counter):,} unique)",
        f"  - 完全相同 formula:  {dup_formulas:,} 唯一公式 ({len(formula_counter):,} unique)",
        "",
        "## indicatorType 分布",
    ]
    for tp, c in sorted(type_counter.items()):
        lines.append(f"  - type={tp}:  {c:,} ({c/n*100:.1f}%)")
    lines.extend(["", "## 热度 Top 10 (by salesCount)"])
    for s in sales_top10:
        lines.append(f"  - {s:,}")
    lines.extend(["", "## 热度 Top 10 (by favoritesCount)"])
    for v in favs_top10:
        lines.append(f"  - {v:,}")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {REPORT_PATH} ({REPORT_PATH.stat().st_size} bytes)")

    indicators_sorted = sorted(indicators, key=_heat_score, reverse=True)
    top_n = indicators_sorted[: args.top_n]
    with TOPN_PATH.open("w", encoding="utf-8") as fh:
        for d in top_n:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  → {TOPN_PATH} ({args.top_n} indicators, by heat score)")
    print()
    print("Top 5 sample (name | sales | fav | formula head):")
    for d in top_n[:5]:
        f = (d.get("formula") or "").replace("\n", " ").strip()
        print(f"  · {d.get('name','?'):<20s}  sales={d.get('salesCount'):>6,}  "
              f"fav={d.get('favoritesCount'):>5,}  | {f[:80]}")


if __name__ == "__main__":
    main()
