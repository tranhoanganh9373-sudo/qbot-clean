"""Dashboard panel — Risk Events 风控事件审计 trail.

数据源 (只读): data_cache/risk_event_log.csv
  字段: dt(ISO), gate_id, symbol, blocked(0|1), nav, peak, dd_pct, reason

显示最近 30 行 + summary card (总事件 / blocked / 各 gate 触发次数 + 最近 trip).
跟 risk_metrics.py 互补:
  - risk_metrics: 当前 portfolio Beta/vol/VaR (前向)
  - risk_events: 历史 gate 触发审计 (后向)
"""
from __future__ import annotations

import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = ROOT / "data_cache" / "risk_event_log.csv"

MAX_ROWS = 30


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _load_events() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return rows
    except Exception:
        return []


def build_risk_events_section() -> str:
    events = _load_events()
    if not events:
        return _placeholder(
            "<code>data_cache/risk_event_log.csv</code> 不存在 — "
            "跑 <code>python -m claude_finance.risk.gates --self-check</code> "
            "或等 daily_check step 1.8 自动产出."
        )

    events_sorted = list(reversed(events))[:MAX_ROWS]
    total = len(events)
    blocked = sum(1 for r in events if r.get("blocked", "0") == "1")
    gate_counts: dict = {}
    last_block = None
    for r in events:
        gid = r.get("gate_id", "?")
        gate_counts[gid] = gate_counts.get(gid, 0) + 1
        if r.get("blocked", "0") == "1":
            last_block = r

    status_color = "#dc2626" if blocked > 0 else "#16a34a"
    status_text = "⚠ 有熔断事件" if blocked > 0 else "✓ 无熔断"
    last_block_html = ""
    if last_block:
        last_block_html = (
            f"<br>最近 trip: <code>{html.escape(last_block.get('gate_id', '?'))}</code> "
            f"@ <code>{html.escape(last_block.get('dt', '?')[:19])}</code> · "
            f"<span style='color:#dc2626;'>{html.escape(last_block.get('reason', '?')[:100])}</span>"
        )
    gate_breakdown = " · ".join(
        f"<code>{html.escape(g)}</code> ×{n}"
        for g, n in sorted(gate_counts.items())
    )

    banner = (
        '<div style="background:rgba(107,114,128,0.08); border-left:3px solid '
        f'{status_color}; padding:10px 14px; margin-bottom:12px; '
        'border-radius:0 4px 4px 0; font-size:12px;">'
        f'<strong>🛡️ Risk Events</strong> · <strong>{status_text}</strong> · '
        f'总事件 <strong>{total}</strong> · '
        f'熔断 <strong style="color:#dc2626;">{blocked}</strong> · '
        f'{gate_breakdown}'
        f'{last_block_html}'
        '<br><span style="color:var(--muted, #6b7280); font-size:11px;">'
        'data_cache/risk_event_log.csv · 阈值: MDD -15% / daily -4% / 单票 20%. '
        '一行回滚: <code>src/claude_finance/risk/gates.py: RISK_ENABLED=False</code>'
        '</span></div>'
    )

    rows_html = []
    for r in events_sorted:
        blocked_str = r.get("blocked", "0")
        row_bg = "rgba(220,38,38,0.08)" if blocked_str == "1" else "transparent"
        blocked_lbl = (
            '<span style="color:#dc2626; font-weight:600;">✗ BLOCK</span>'
            if blocked_str == "1"
            else '<span style="color:#16a34a;">✓ ok</span>'
        )
        dt_short = r.get("dt", "")[:19]
        gid = r.get("gate_id", "")
        sym = r.get("symbol", "") or ""
        dd_pct = r.get("dd_pct", "") or ""
        try:
            dd_pct_f = float(dd_pct)
            dd_pct_str = f"{dd_pct_f:+.2f}%"
            dd_color = "#dc2626" if dd_pct_f < -10 else "#f59e0b" if dd_pct_f < 0 else "#6b7280"
        except (ValueError, TypeError):
            dd_pct_str = "—"
            dd_color = "#6b7280"
        nav = r.get("nav", "0")
        peak = r.get("peak", "0")
        reason = r.get("reason", "")[:120]
        rows_html.append(
            f'<tr style="background:{row_bg};">'
            f'<td style="font-family:monospace; font-size:11px;">{html.escape(dt_short)}</td>'
            f'<td><code>{html.escape(gid)}</code></td>'
            f'<td style="font-family:monospace;">{html.escape(sym)}</td>'
            f'<td style="text-align:center;">{blocked_lbl}</td>'
            f'<td style="text-align:right; font-family:monospace; color:{dd_color};">{dd_pct_str}</td>'
            f'<td style="text-align:right; font-family:monospace; font-size:11px;">{html.escape(nav)}</td>'
            f'<td style="text-align:right; font-family:monospace; font-size:11px;">{html.escape(peak)}</td>'
            f'<td style="font-size:11px; color:var(--muted, #6b7280);">{html.escape(reason)}</td>'
            "</tr>"
        )

    table = (
        '<table class="data">'
        '<colgroup>'
        '<col style="width:14%;"><col style="width:11%;"><col style="width:10%;">'
        '<col style="width:8%;"><col style="width:8%;">'
        '<col style="width:10%;"><col style="width:10%;"><col style="width:29%;">'
        '</colgroup>'
        '<thead><tr>'
        '<th>时间</th><th>gate</th><th>sym</th><th>状态</th>'
        '<th>dd%/chg%</th><th>nav</th><th>peak</th><th>reason</th>'
        '</tr></thead>'
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )

    footer = (
        '<div style="margin-top:10px; font-size:11px; color:var(--muted, #6b7280);">'
        f'显示最近 {len(events_sorted)} 条 (总 {total}). '
        '历史教训: 4 次 Phase B OOS 失败 (v19.7/v19.9/super_big_net/shareholders) 均跑到 -33%~-37% MDD, '
        '15% 熔断会全部截胡. WS push 触发实时刷新.'
        '</div>'
    )

    return banner + table + footer
