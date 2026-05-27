"""Dashboard panel — Phase B 失败模式 + 自动 gate 历史.

数据源 (只读):
  - data_cache/phase_b_failure_modes.json (P1.1 结构化 4 失败 + 3 成功 + 7 modes)
  - data_cache/phase_b_check_log.csv (P1.2 gate 调用审计)

显示:
  1. 顶部 banner: gate 状态 + 最近一次 check
  2. 阈值速查
  3. 失败 vs 成功 对比 table
  4. 7 个 failure mode 规则展示 (折叠)
  5. 最近 gate check 记录 (audit log tail)
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODES_PATH = ROOT / "data_cache" / "phase_b_failure_modes.json"
CHECK_LOG_PATH = ROOT / "data_cache" / "phase_b_check_log.csv"

MAX_AUDIT_ROWS = 20


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _load_modes() -> dict | None:
    if not MODES_PATH.exists():
        return None
    try:
        return json.loads(MODES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_audit() -> list[dict]:
    if not CHECK_LOG_PATH.exists():
        return []
    try:
        with CHECK_LOG_PATH.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def build_phase_b_history_section() -> str:
    data = _load_modes()
    if data is None:
        return _placeholder(
            "<code>data_cache/phase_b_failure_modes.json</code> 不存在."
        )

    modes = data.get("failure_modes", [])
    history = data.get("history", {})
    failures = history.get("failures", [])
    successes = history.get("successes", [])
    thresholds = data.get("thresholds", {})
    audit = _load_audit()

    # ===== 顶部 banner =====
    latest_check = ""
    if audit:
        last = audit[-1]
        pass_overall = last.get("pass_overall", "1") == "1"
        marker = "✓ PASS" if pass_overall else "✗ REJECT"
        marker_color = "#16a34a" if pass_overall else "#dc2626"
        latest_check = (
            f'<br>最近 check: <code>{html.escape(last.get("candidate", "?"))}</code> '
            f'@ <code>{html.escape(last.get("dt", "")[:19])}</code> · '
            f'<span style="color:{marker_color}; font-weight:600;">{marker}</span>'
            f' fired: <code>{html.escape(last.get("fired_modes", "") or "(none)")}</code>'
        )

    banner = (
        '<div style="background:rgba(59,130,246,0.08); border-left:3px solid #3b82f6; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        '<strong>🔬 Phase B 失败模式自动 gate</strong> · '
        f'modes <strong>{len(modes)}</strong> · '
        f'历史失败 <strong style="color:#dc2626;">{len(failures)}</strong> · '
        f'成功 <strong style="color:#16a34a;">{len(successes)}</strong> · '
        f'gate 调用 <strong>{len(audit)}</strong> 次'
        f'{latest_check}'
        '<br><span style="color:var(--muted, #6b7280); font-size:11px;">'
        '借鉴 jin-ce-zhi-suan AnalysisAgent → prompt_context_patch, 只做 reject gate '
        '(不 LLM-driven 因子生成, 严格 OOS 协议保留). '
        'CLI: <code>python -m claude_finance.phase_b_gate --help</code>. '
        '一行回滚: <code>PHASE_B_GATE_ENABLED=False</code>.'
        '</span></div>'
    )

    # ===== 阈值速查 =====
    thresholds_html = (
        '<h3 style="font-size:13px; margin:14px 0 6px 0; color:var(--muted, #6b7280);">'
        '📏 阈值速查</h3>'
        '<div style="background:rgba(107,114,128,0.06); padding:8px 12px; '
        'border-left:3px solid #6b7280; border-radius:0 4px 4px 0; '
        'font-size:11px; line-height:1.7;">'
        + " · ".join(
            f"<code>{html.escape(k)}</code>: <strong>{v}</strong>"
            for k, v in thresholds.items()
        )
        + "</div>"
    )

    # ===== 历史对比 table =====
    fail_rows = []
    for f in failures:
        oos = f.get("oos_calmar", 0)
        is_c = f.get("is_calmar", 0)
        decay = "—"
        if isinstance(oos, (int, float)) and isinstance(is_c, (int, float)) and is_c:
            decay_pct = (oos - is_c) / is_c * 100
            decay = f"{decay_pct:+.0f}%"
        fired = ", ".join(f.get("fired_modes", [])) or "(gate 未捕获)"
        fail_rows.append(
            '<tr style="background:rgba(220,38,38,0.04);">'
            f'<td><code>{html.escape(f.get("candidate", "?"))}</code></td>'
            f'<td style="text-align:right;">{f.get("n_months", "?")}</td>'
            f'<td style="text-align:right; font-family:monospace; color:#dc2626;">{is_c}</td>'
            f'<td style="text-align:right; font-family:monospace; color:#dc2626;">{oos}</td>'
            f'<td style="text-align:right; color:#dc2626;">{decay}</td>'
            f'<td style="font-size:11px;">{html.escape(fired)}</td>'
            "</tr>"
        )

    success_rows = []
    for s in successes:
        fired = ", ".join(s.get("fired_modes", [])) or "(clean)"
        success_rows.append(
            '<tr style="background:rgba(22,163,74,0.04);">'
            f'<td><code>{html.escape(s.get("candidate", "?"))}</code></td>'
            f'<td style="text-align:right;">{s.get("n_months", "?")}</td>'
            f'<td style="text-align:right; font-family:monospace;">{s.get("is_calmar", "?")}</td>'
            f'<td style="text-align:right; font-family:monospace; color:#16a34a;">{s.get("oos_calmar", "?")}</td>'
            f'<td style="text-align:right; color:#16a34a;">✓ {html.escape(str(s.get("verdict", ""))[:30])}</td>'
            f'<td style="font-size:11px;">{html.escape(fired)}</td>'
            "</tr>"
        )

    history_table = (
        '<h3 style="font-size:13px; margin:14px 0 6px 0; color:var(--muted, #6b7280);">'
        '📜 历史 case (失败 vs 成功)</h3>'
        '<table class="data"><colgroup>'
        '<col style="width:22%;"><col style="width:8%;"><col style="width:10%;">'
        '<col style="width:10%;"><col style="width:18%;"><col style="width:32%;">'
        '</colgroup>'
        '<thead><tr><th>候选</th><th>n_months</th><th>IS Calmar</th>'
        '<th>OOS Calmar</th><th>verdict/decay</th><th>fired modes</th></tr></thead>'
        f"<tbody>{''.join(fail_rows)}{''.join(success_rows)}</tbody></table>"
    )

    # ===== Modes 规则展示 (折叠) =====
    mode_rows = []
    for m in modes:
        sev = m.get("severity", "reject")
        sev_color = "#dc2626" if sev == "reject" else "#f59e0b"
        rule_strs = []
        for r in m.get("rules", []):
            k = r.get("key", "?")
            op = r.get("op", "?")
            th = r.get("threshold", "?")
            rule_strs.append(
                f"<code>{html.escape(k)} {html.escape(op)} {html.escape(str(th))}</code>"
            )
        logic = html.escape(m.get("logic", "all")).upper()
        rules_joined = (" " + logic + " ").join(rule_strs)
        examples = ", ".join(m.get("examples", []))[:80]
        mode_rows.append(
            f'<tr>'
            f'<td><code>{html.escape(m.get("id", "?"))}</code></td>'
            f'<td style="color:{sev_color}; font-weight:600;">{html.escape(sev)}</td>'
            f'<td style="font-size:11px;">{rules_joined}</td>'
            f'<td style="font-size:11px; color:var(--muted, #6b7280);">{html.escape(examples)}</td>'
            f'</tr>'
        )
    modes_table = (
        '<details style="margin-top:14px;">'
        '<summary style="cursor:pointer; font-size:13px; color:var(--muted, #6b7280);">'
        f'🔍 展开 {len(modes)} 个 failure modes 规则定义'
        '</summary>'
        '<table class="data" style="margin-top:8px;">'
        '<colgroup>'
        '<col style="width:22%;"><col style="width:8%;">'
        '<col style="width:35%;"><col style="width:35%;">'
        '</colgroup>'
        '<thead><tr><th>mode_id</th><th>severity</th><th>rule</th><th>examples</th></tr></thead>'
        f'<tbody>{"".join(mode_rows)}</tbody></table></details>'
    )

    # ===== Audit log tail =====
    if audit:
        recent = list(reversed(audit))[:MAX_AUDIT_ROWS]
        audit_rows = []
        for r in recent:
            pass_overall = r.get("pass_overall", "1") == "1"
            row_bg = "transparent" if pass_overall else "rgba(220,38,38,0.06)"
            marker = (
                '<span style="color:#16a34a;">✓ pass</span>'
                if pass_overall else
                '<span style="color:#dc2626; font-weight:600;">✗ REJECT</span>'
            )
            audit_rows.append(
                f'<tr style="background:{row_bg};">'
                f'<td style="font-family:monospace; font-size:11px;">{html.escape(r.get("dt", "")[:19])}</td>'
                f'<td><code>{html.escape(r.get("candidate", "?"))}</code></td>'
                f'<td style="text-align:right;">{html.escape(r.get("n_months", ""))}</td>'
                f'<td style="text-align:right; font-family:monospace;">{html.escape(r.get("is_calmar", ""))}</td>'
                f'<td style="text-align:center;">{marker}</td>'
                f'<td style="font-size:11px; color:#dc2626;">{html.escape(r.get("rejects", "") or "—")}</td>'
                f'<td style="font-size:11px; color:#f59e0b;">{html.escape(r.get("warns", "") or "—")}</td>'
                f'</tr>'
            )
        audit_html = (
            '<h3 style="font-size:13px; margin:14px 0 6px 0; color:var(--muted, #6b7280);">'
            f'📋 最近 {len(recent)} 次 gate check (共 {len(audit)})</h3>'
            '<table class="data">'
            '<colgroup>'
            '<col style="width:14%;"><col style="width:20%;"><col style="width:8%;">'
            '<col style="width:10%;"><col style="width:10%;">'
            '<col style="width:24%;"><col style="width:14%;">'
            '</colgroup>'
            '<thead><tr>'
            '<th>时间</th><th>候选</th><th>n_months</th><th>IS Calmar</th>'
            '<th>结果</th><th>rejects</th><th>warns</th>'
            '</tr></thead>'
            f"<tbody>{''.join(audit_rows)}</tbody></table>"
        )
    else:
        audit_html = (
            '<div style="margin-top:14px; padding:10px 14px; '
            'background:rgba(107,114,128,0.06); border-left:3px solid #6b7280; '
            'border-radius:0 4px 4px 0; font-size:11px; color:var(--muted, #6b7280);">'
            '尚无 gate 调用记录. CLI: <code>python -m claude_finance.phase_b_gate --help</code>'
            '</div>'
        )

    return banner + thresholds_html + history_table + modes_table + audit_html
