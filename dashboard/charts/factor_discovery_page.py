"""Dashboard Factor Discovery 新 tab page — Top 100 mt180 公式跨 universe 比较.

数据源 (全只读):
  - data_cache/mt180/ic_results_extended.csv  (all_no_st / IS 2014-2020, 主排序)
  - data_cache/mt180/ic_results.csv           (CSI300 / IS 2017-2020, 跨 univ 对比)
  - data_cache/mt180/indicators_detail.jsonl  (TDX 公式 head)

视觉:
  - Filter chip: All / Clean / Spurious / VOL leak / 派生 / Sign-flip
  - Top 100 表: # / name / output / ICIR all / ICIR csi300 / rank Δ / n_m / ⇋ / status / 公式 head
"""
from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
IC_EXT_CSV = ROOT / "data_cache" / "mt180" / "ic_results_extended.csv"
IC_PREV_CSV = ROOT / "data_cache" / "mt180" / "ic_results.csv"
DETAIL_JSONL = ROOT / "data_cache" / "mt180" / "indicators_detail.jsonl"

TOP_N = 100

CLEAN_NAMES = {"CYS", "竞价勾魂翻"}
DEMOTED_NAMES = {"CT3", "量能均线突破", "牛股启动幅图", "土系趋势"}
VOLUME_LEAK_OUTPUTS = {
    "VOLUME", "VOLVME", "NOTEXTV", "成交量", "总量", "成交", "主力", "主动买盘",
    "单位时间总量", "量能",
}


def _placeholder(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{message}"
        "</div>"
    )


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_formula_heads(ids: set[str]) -> dict[str, str]:
    if not DETAIL_JSONL.exists() or not ids:
        return {}
    out: dict[str, str] = {}
    with DETAIL_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = d.get("id")
            if iid in ids:
                f = (d.get("formula") or "").replace("\n", " ").strip()
                f = re.sub(r"\s+", " ", f)
                out[iid] = f
                if len(out) == len(ids):
                    break
    return out


def _classify(row: dict) -> tuple[str, str, str]:
    name = row.get("name", "")
    output_col = row.get("output_col", "")
    if name in CLEAN_NAMES:
        return ("✓ Clean", "#16a34a", "clean")
    if name in DEMOTED_NAMES:
        return ("✗ Spurious", "#f59e0b", "spurious")
    if output_col in VOLUME_LEAK_OUTPUTS:
        return ("⚠ VOL leak", "#dc2626", "vol-leak")
    return ("派生", "#6b7280", "derived")


def build_factor_discovery_page() -> str:
    rows = _load_csv(IC_EXT_CSV)
    if not rows:
        return _placeholder(
            "无 mt180 IC extended 结果 — "
            "跑 <code>python examples/mt180_phase_a_ic_extended.py</code> 后此 page 自动填充."
        )

    prev_rows = _load_csv(IC_PREV_CSV)
    prev_rank_by_id: dict[str, int] = {}
    prev_icir_by_id: dict[str, float] = {}
    for i, r in enumerate(prev_rows, 1):
        iid = r["id"]
        prev_rank_by_id[iid] = i
        prev_icir_by_id[iid] = float(r["icir"])

    rows.sort(key=lambda r: -abs(float(r["icir"])))
    top = rows[:TOP_N]
    needed_ids = {r["id"] for r in top}
    formula_heads = _load_formula_heads(needed_ids)

    n_total = len(rows)
    n_clean = sum(1 for r in top if r.get("name") in CLEAN_NAMES)
    n_spurious = sum(1 for r in top if r.get("name") in DEMOTED_NAMES)
    n_leak = sum(1 for r in top if r.get("output_col") in VOLUME_LEAK_OUTPUTS)
    n_derived = TOP_N - n_clean - n_spurious - n_leak

    n_sign_flip = 0
    n_no_prev = 0
    for r in top:
        iid = r["id"]
        new_icir = float(r["icir"])
        prev_icir = prev_icir_by_id.get(iid)
        if prev_icir is None:
            n_no_prev += 1
        elif (new_icir > 0) != (prev_icir > 0) and min(abs(new_icir), abs(prev_icir)) > 0.3:
            n_sign_flip += 1

    summary = (
        '<div style="background:rgba(37,99,235,0.06); border-left:3px solid #2563eb; '
        'padding:12px 16px; margin-bottom:14px; border-radius:0 4px 4px 0; font-size:13px;">'
        f"<strong>Top {TOP_N} / 共 {n_total} mt180 公式</strong> 跨 universe 比较<br>"
        '<span style="font-size:12px; color:var(--muted, #6b7280);">'
        f"分类: <strong style='color:#16a34a;'>{n_clean} Clean</strong> · "
        f"<strong style='color:#f59e0b;'>{n_spurious} Spurious</strong> · "
        f"<strong style='color:#dc2626;'>{n_leak} VOL Leak</strong> · "
        f"<span style='color:#6b7280;'>{n_derived} 派生</span> · "
        f"<strong style='color:#7c3aed;'>{n_sign_flip} Sign-flip</strong> "
        f"({n_no_prev} 仅 all_no_st 新出现)<br>"
        "主排序: <code>all_no_st</code> / IS 2014-2020 / 横截面 Spearman / horizon=20d<br>"
        "对比: <code>CSI300</code> / IS 2017-2020 (上轮)"
        "</span>"
        "</div>"
    )

    filter_html = (
        '<div class="discovery-filter" style="margin-bottom:12px; display:flex; gap:8px; flex-wrap:wrap;">'
        f'<button data-filter="all" class="filter-btn active" type="button">全部 ({TOP_N})</button>'
        '<button data-filter="clean" class="filter-btn" type="button" '
        f'style="border-color:#16a34a;">✓ Clean ({n_clean})</button>'
        '<button data-filter="spurious" class="filter-btn" type="button" '
        f'style="border-color:#f59e0b;">✗ Spurious ({n_spurious})</button>'
        '<button data-filter="vol-leak" class="filter-btn" type="button" '
        f'style="border-color:#dc2626;">⚠ VOL Leak ({n_leak})</button>'
        f'<button data-filter="derived" class="filter-btn" type="button">派生 ({n_derived})</button>'
        '<button data-filter="sign-flip" class="filter-btn" type="button" '
        f'style="border-color:#7c3aed;">⇋ Sign-flip ({n_sign_flip})</button>'
        '</div>'
    )

    css = """
<style>
.factor-discovery-page .filter-btn {
    padding: 6px 14px; font-size: 12px; cursor: pointer;
    background: var(--bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 16px;
    transition: background 0.15s, color 0.15s;
}
.factor-discovery-page .filter-btn:hover { background: var(--accent); color: #fff; }
.factor-discovery-page .filter-btn.active { background: var(--accent); color: #fff; }
.factor-discovery-page table.data { table-layout: fixed; width: 100%; }
.factor-discovery-page table.data th,
.factor-discovery-page table.data td {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.factor-discovery-page tr.row-clean    { background: rgba(22,163,74,0.05); }
.factor-discovery-page tr.row-spurious { background: rgba(245,158,11,0.05); }
.factor-discovery-page tr.row-vol-leak { background: rgba(220,38,38,0.05); }
.factor-discovery-page tr.row-sign-flip { box-shadow: inset 4px 0 0 #7c3aed; }
.factor-discovery-page tr.hidden { display: none; }
.factor-discovery-page td.formula-head {
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
    font-size: 11px; color: var(--muted, #6b7280);
}
</style>
"""

    cols = [
        ("#", 4), ("name", 12), ("output", 10),
        ("ICIR all", 7), ("ICIR csi300", 7), ("rank Δ", 6),
        ("n_m", 4), ("⇋", 3), ("status", 10), ("公式 head", 37),
    ]
    assert sum(w for _, w in cols) == 100
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _, w in cols
    ) + "</colgroup>"
    thead = "".join(f"<th>{html.escape(label)}</th>" for label, _ in cols)

    body_rows: list[str] = []
    for i, r in enumerate(top, 1):
        iid = r["id"]
        status_label, status_color, filter_class = _classify(r)
        icir_new = float(r["icir"])
        icir_color = "#16a34a" if icir_new > 0 else "#dc2626"
        name = (r.get("name") or "")[:14]
        output = (r.get("output_col") or "")[:10]
        n_m = int(r["n_months"])
        formula = formula_heads.get(iid, "")[:120]

        prev_rank = prev_rank_by_id.get(iid)
        prev_icir = prev_icir_by_id.get(iid)
        if prev_icir is not None:
            prev_color = "#16a34a" if prev_icir > 0 else "#dc2626"
            prev_icir_str = (
                f"<span style='color:{prev_color};'>{prev_icir:+.2f}</span>"
            )
            rank_delta = (prev_rank or 0) - i
            if abs(rank_delta) >= 30:
                rd_color = "#dc2626"
            elif abs(rank_delta) >= 15:
                rd_color = "#f59e0b"
            else:
                rd_color = "var(--muted, #6b7280)"
            rank_delta_str = (
                f"<span style='color:{rd_color};'>{rank_delta:+d}</span>"
            )
        else:
            prev_icir_str = "<span style='color:#6b7280;'>—</span>"
            rank_delta_str = "<span style='color:#7c3aed;'>NEW</span>"

        sign_flip = (
            prev_icir is not None
            and (icir_new > 0) != (prev_icir > 0)
            and min(abs(icir_new), abs(prev_icir)) > 0.3
        )
        flip_str = (
            "<span style='color:#7c3aed; font-weight:700;'>⇋</span>"
            if sign_flip else ""
        )

        row_classes = [f"row-{filter_class}"]
        if sign_flip:
            row_classes.append("row-sign-flip")
        filter_keys = [filter_class] + (["sign-flip"] if sign_flip else [])

        body_rows.append(
            f'<tr class="{" ".join(row_classes)}" data-filters="{",".join(filter_keys)}">'
            f"<td>{i}</td>"
            f"<td>{html.escape(name)}</td>"
            f"<td><code style='font-size:11px;'>{html.escape(output)}</code></td>"
            f"<td style='color:{icir_color}; font-weight:600; text-align:right;'>"
            f"{icir_new:+.2f}</td>"
            f"<td style='text-align:right;'>{prev_icir_str}</td>"
            f"<td style='text-align:right;'>{rank_delta_str}</td>"
            f"<td style='text-align:right;'>{n_m}</td>"
            f"<td style='text-align:center;'>{flip_str}</td>"
            f"<td style='color:{status_color}; font-weight:600;'>{status_label}</td>"
            f"<td class='formula-head'>{html.escape(formula)}</td>"
            "</tr>"
        )
    table_html = (
        '<table class="data">'
        + colgroup
        + f"<thead><tr>{thead}</tr></thead>"
        + f"<tbody>{''.join(body_rows)}</tbody>"
        + "</table>"
    )

    footer = (
        '<div style="margin-top:10px; font-size:11px; color:var(--muted, #6b7280);">'
        "📊 <strong>判断</strong>: <strong style='color:#16a34a;'>Clean</strong> = 跨 universe 稳定; "
        "<strong style='color:#f59e0b;'>Spurious</strong> = CSI300 上有 ICIR 但 all_no_st rank 大降; "
        "<strong style='color:#dc2626;'>VOL Leak</strong> = output 是 raw VOL projection; "
        "<strong style='color:#7c3aed;'>Sign-flip</strong> = 跨 universe 方向反转 (不稳定)<br>"
        "🔢 <strong>rank Δ</strong>: CSI300 rank → all_no_st rank; 正 = rank 升 (更 robust)<br>"
        "⚠️ <strong>n_months</strong> ≤ 48 (baidu_kline 2014-2016 backfill 缺口), "
        "Phase B 60 月门槛未破 — 仅 discovery 用, 不进 production"
        "</div>"
    )

    return (
        css
        + '<div class="factor-discovery-page">'
        + summary
        + filter_html
        + table_html
        + footer
        + "</div>"
    )
