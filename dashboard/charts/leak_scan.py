"""Dashboard panel — Static leak scanner 结果展示.

数据源 (只读): data_cache/leak_scan_report.json
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_PATH = ROOT / "data_cache" / "leak_scan_report.json"

MAX_FINDINGS_DISPLAY = 30


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _load() -> dict | None:
    if not REPORT_PATH.exists():
        return None
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_leak_scan_section() -> str:
    data = _load()
    if data is None:
        return _placeholder(
            "<code>data_cache/leak_scan_report.json</code> 不存在 — "
            "跑 <code>python -m tools.static_leak_check --all</code> 产出."
        )

    n_files = data.get("n_files", 0)
    n_rejects = data.get("n_rejects", 0)
    n_warns = data.get("n_warns", 0)
    n_findings = data.get("n_findings", 0)
    scanned_at = data.get("scanned_at", "")
    findings = data.get("findings", [])

    if n_rejects > 0:
        status_color = "#dc2626"
        status_text = f"⚠ {n_rejects} REJECT — 必须修复"
    elif n_warns > 0:
        status_color = "#f59e0b"
        status_text = f"⚠ {n_warns} WARN — 需人工 review"
    else:
        status_color = "#16a34a"
        status_text = "✓ 全绿 — 0 lookahead 风险"

    banner = (
        '<div style="background:rgba(107,114,128,0.08); border-left:3px solid '
        f'{status_color}; padding:10px 14px; margin-bottom:12px; '
        'border-radius:0 4px 4px 0; font-size:12px;">'
        f'<strong>🔎 Static Leak Scan</strong> · <strong style="color:{status_color};">'
        f'{status_text}</strong>'
        f'<br>扫 <strong>{n_files}</strong> 文件 · '
        f'findings <strong>{n_findings}</strong> '
        f'(reject <span style="color:#dc2626;">{n_rejects}</span> / '
        f'warn <span style="color:#f59e0b;">{n_warns}</span> / '
        f'info <span style="color:var(--muted, #6b7280);">'
        f'{n_findings - n_rejects - n_warns}</span>)'
        f' · 最近扫: <code>{html.escape(scanned_at[:19])}</code>'
        '<br><span style="color:var(--muted, #6b7280); font-size:11px;">'
        '借鉴 jin-ce-zhi-suan critic.py:74-128 regex+AST 静态扫. '
        '检 lookahead bias: <code>shift(-N)</code>, <code>iloc[+N]</code>, '
        '<code>future_*</code> 关键字, <code>.loc[future_:]</code>. '
        'CLI: <code>python -m tools.static_leak_check --all</code>. '
        '一行回滚: <code>LEAK_CHECK_ENABLED=False</code>.'
        '</span></div>'
    )

    severe = [f for f in findings if f.get("severity") in ("reject", "warn")]
    if severe:
        rows = []
        for f in severe[:MAX_FINDINGS_DISPLAY]:
            sev = f.get("severity", "?")
            sev_color = "#dc2626" if sev == "reject" else "#f59e0b"
            file_short = f.get("file", "").split("/")[-1]
            rows.append(
                '<tr>'
                f'<td style="color:{sev_color}; font-weight:600;">'
                f'{html.escape(sev)}</td>'
                f'<td style="font-family:monospace; font-size:11px;">'
                f'{html.escape(file_short)}:{f.get("line", "?")}</td>'
                f'<td><code>{html.escape(f.get("pattern_id", "?"))}</code></td>'
                f'<td><code style="color:#dc2626;">'
                f'{html.escape(f.get("matched", "")[:40])}</code></td>'
                f'<td style="font-size:11px;">'
                f'{html.escape(f.get("description", "")[:80])}</td>'
                '</tr>'
            )
        severe_table = (
            '<h3 style="font-size:13px; margin:14px 0 6px 0; '
            'color:var(--muted, #6b7280);">'
            f'🚨 Reject + Warn findings ({len(severe)})</h3>'
            '<table class="data">'
            '<colgroup>'
            '<col style="width:8%;"><col style="width:25%;"><col style="width:15%;">'
            '<col style="width:20%;"><col style="width:32%;">'
            '</colgroup>'
            '<thead><tr><th>severity</th><th>file:line</th>'
            '<th>pattern</th><th>matched</th><th>description</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    else:
        severe_table = ""

    info_list = [f for f in findings if f.get("severity") == "info"]
    if info_list:
        info_rows = []
        for f in info_list[:MAX_FINDINGS_DISPLAY]:
            file_short = f.get("file", "").split("/")[-1]
            info_rows.append(
                f'<tr>'
                f'<td style="color:var(--muted, #6b7280);">info</td>'
                f'<td style="font-family:monospace; font-size:11px;">'
                f'{html.escape(file_short)}:{f.get("line", "?")}</td>'
                f'<td><code>{html.escape(f.get("pattern_id", "?"))}</code></td>'
                f'<td><code>{html.escape(f.get("matched", "")[:40])}</code></td>'
                f'</tr>'
            )
        info_html = (
            '<details style="margin-top:14px;">'
            '<summary style="cursor:pointer; font-size:13px; '
            'color:var(--muted, #6b7280);">'
            f'💬 展开 {len(info_list)} info findings (注释/字符串/字面 \'lookahead\')'
            '</summary>'
            '<table class="data" style="margin-top:8px; font-size:11px;">'
            '<thead><tr><th>severity</th><th>file:line</th>'
            '<th>pattern</th><th>matched</th></tr></thead>'
            f"<tbody>{''.join(info_rows)}</tbody></table></details>"
        )
    else:
        info_html = ""

    files_html = (
        '<details style="margin-top:14px;">'
        '<summary style="cursor:pointer; font-size:13px; '
        'color:var(--muted, #6b7280);">'
        f'📂 展开扫描文件列表 ({n_files})'
        '</summary>'
        '<div style="margin-top:8px; font-size:11px; color:var(--muted, #6b7280); '
        'font-family:monospace; line-height:1.6;">'
        + "<br>".join(html.escape(p) for p in data.get("files_scanned", [])[:100])
        + "</div></details>"
    )

    return banner + severe_table + info_html + files_html
