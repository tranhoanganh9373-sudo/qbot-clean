"""Dashboard panel — mt180 Top-N factors (read-only experimental panel).

数据源:
  - data_cache/mt180/ic_results.csv (Phase A IC 评估结果, IS 2017-2020)
  - data_cache/mt180/indicators_detail.jsonl (TDX 公式 head)

标注 caveat:
  - n_months = 36 (低样本) — 所有候选不达 Phase B 60 月门槛
  - 大量 VOLUME spec leak (Top 9 ICIR=-2.14 等价 raw VOL)
  - 仅 5 个 candidate 公式干净 (无 spec leak)
  - 此 panel 仅 read-only 参考, 不进 production
"""
from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
IC_CSV = ROOT / "data_cache" / "mt180" / "ic_results.csv"
DETAIL_JSONL = ROOT / "data_cache" / "mt180" / "indicators_detail.jsonl"

TOP_N = 30

# Cherry-picked clean candidates (subagent 双轮验证: CSI300 + all_no_st 跨 universe 稳定)
# 历史: 上轮 5 个 picks 中 4 个 (CT3 / 量能均线突破 / 牛股启动幅图 / 土系趋势) 在 all_no_st
# universe 上 rank 大降或 sign-flip → 是 CSI300 spurious. 已 demote.
CLEAN_CANDIDATE_NAMES = {
    "CYS",            # 跨 universe 唯一稳定 (CSI300 -4.45 / all_no_st -3.51, rank 1→1)
    "竞价勾魂翻",       # all_no_st 浮出 唯一独立正向 (rank 24→11, ICIR +1.17)
}

# Demoted (CSI300 spurious, 已被 all_no_st 验证为 size-effect contaminated)
DEMOTED_NAMES = {
    "CT3", "量能均线突破", "牛股启动幅图", "土系趋势",
}

# VOLUME spec leak symptoms (output 是 raw VOL projection)
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


def _load_ic() -> list[dict]:
    if not IC_CSV.exists():
        return []
    with IC_CSV.open(encoding="utf-8") as fh:
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
                out[iid] = f[:120]
                if len(out) == len(ids):
                    break
    return out


def _classify(row: dict) -> tuple[str, str]:
    name = row.get("name", "")
    output_col = row.get("output_col", "")
    if name in CLEAN_CANDIDATE_NAMES:
        return ("✓ 跨 univ 稳定", "#16a34a")
    if name in DEMOTED_NAMES:
        return ("✗ CSI300 spurious", "#f59e0b")
    if output_col in VOLUME_LEAK_OUTPUTS:
        return ("⚠ VOL leak", "#dc2626")
    return ("派生", "#6b7280")


def build_mt180_top_factors_section() -> str:
    rows = _load_ic()
    if not rows:
        return _placeholder(
            "无 mt180 IC 结果 — 跑 <code>python examples/mt180_phase_a_ic.py</code> "
            "产生 <code>data_cache/mt180/ic_results.csv</code> 后此 panel 自动填充."
        )

    top = rows[:TOP_N]
    needed_ids = {r["id"] for r in top}
    formula_heads = _load_formula_heads(needed_ids)

    n_total = len(rows)
    n_clean = sum(1 for r in rows if r.get("name") in CLEAN_CANDIDATE_NAMES)
    n_leak = sum(1 for r in rows if r.get("output_col") in VOLUME_LEAK_OUTPUTS)
    banner = (
        '<div style="background:rgba(245,158,11,0.10); border-left:3px solid #f59e0b; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        '<strong style="color:#f59e0b;">⚠️ Experimental — 不进 production</strong><br>'
        f"<code>{n_total}</code> 个 indicators IC 通过 (Top 500 → ok 70.4%). "
        f"<strong style='color:#16a34a;'>{n_clean}</strong> 跨 universe 稳定 / "
        f"<strong style='color:#f59e0b;'>{len(DEMOTED_NAMES)}</strong> CSI300 spurious / "
        f"<strong style='color:#dc2626;'>{n_leak}</strong> raw VOLUME leak. "
        "<br>"
        "双轮验证: <code>CSI300/IS=2017-2020</code> + <code>all_no_st/IS=2014-2020</code>. "
        "n_months cap=48 (baidu_kline 2014-2016 backfill 缺口, 未破 60 门槛) — 仅参考用."
        "</div>"
    )

    cols = [
        ("#", 4), ("name", 14), ("output", 10), ("ICIR", 8),
        ("IC mean", 8), ("top10%", 7), ("status", 9), ("公式 head", 40),
    ]
    assert sum(w for _, w in cols) == 100
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _, w in cols
    ) + "</colgroup>"
    thead = "".join(f"<th>{html.escape(label)}</th>" for label, _ in cols)

    body_rows: list[str] = []
    for i, r in enumerate(top, 1):
        status_label, status_color = _classify(r)
        icir = float(r["icir"])
        icir_color = "#16a34a" if icir > 0 else "#dc2626"
        name = (r.get("name") or "")[:14]
        output = (r.get("output_col") or "")[:10]
        ic_mean = float(r["ic_mean"])
        top10 = float(r["top10_pos_pct"])
        formula = formula_heads.get(r["id"], "")[:120]
        row_bg = (
            ' style="background:rgba(22,163,74,0.06);"'
            if status_color == "#16a34a" else ""
        )
        body_rows.append(
            f"<tr{row_bg}>"
            f"<td>{i}</td>"
            f"<td>{html.escape(name)}</td>"
            f"<td><code style='font-size:11px;'>{html.escape(output)}</code></td>"
            f"<td style='color:{icir_color}; font-weight:600; text-align:right;'>"
            f"{icir:+.3f}</td>"
            f"<td style='text-align:right;'>{ic_mean:+.4f}</td>"
            f"<td style='text-align:right;'>{top10:.3f}</td>"
            f"<td style='color:{status_color}; font-weight:600;'>{status_label}</td>"
            f"<td style='font-size:11px; color:var(--muted, #6b7280); "
            "font-family:\"SF Mono\", Menlo, monospace;'>"
            f"{html.escape(formula)}</td>"
            "</tr>"
        )
    table_html = (
        '<table class="data">'
        + colgroup
        + f"<thead><tr>{thead}</tr></thead>"
        + f"<tbody>{''.join(body_rows)}</tbody>"
        + "</table>"
    )

    clean_summary_rows: list[str] = []
    for r in rows:
        if r.get("name") not in CLEAN_CANDIDATE_NAMES:
            continue
        icir = float(r["icir"])
        icir_color = "#16a34a" if icir > 0 else "#dc2626"
        clean_summary_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(r.get('name',''))}</strong></td>"
            f"<td><code>{html.escape(r.get('output_col',''))}</code></td>"
            f"<td style='color:{icir_color}; font-weight:600;'>{icir:+.3f}</td>"
            f"<td>{float(r['ic_mean']):+.4f}</td>"
            "</tr>"
        )
    clean_panel = ""
    if clean_summary_rows:
        clean_panel = (
            '<h3 style="font-size:13px; margin:18px 0 6px 0; color:#16a34a;">'
            "✓ Cherry-picked 干净 candidates (无 spec leak, 可进 Phase B 候选池)"
            "</h3>"
            '<table class="data" style="margin-bottom:8px;">'
            "<thead><tr><th>name</th><th>output</th><th>ICIR</th><th>IC mean</th></tr></thead>"
            f"<tbody>{''.join(clean_summary_rows)}</tbody>"
            "</table>"
            '<div style="font-size:11px; color:var(--muted, #6b7280); margin-bottom:12px;">'
            "建议: 先扩 universe / IS 期使 n_months ≥ 60, 再单独 OOS 测试. "
            "当前 n_months=36 与历史 4 次 Phase B abort 模式一致 — 不直接进 Phase B."
            "</div>"
        )

    footer = (
        '<div style="margin-top:8px; font-size:11px; color:var(--muted, #6b7280);">'
        f"显示 Top {len(top)} / 共 {n_total} 通过 IC 的 candidates · "
        "数据源 <code>data_cache/mt180/ic_results.csv</code> · "
        "TDX parser <code>src/claude_finance/tdx_parser.py</code> 70.4% Top-500 覆盖率 · "
        "mt180.com 公开 API 抓取 23,890 公式 (PII strip)"
        "</div>"
    )

    return banner + clean_panel + table_html + footer
