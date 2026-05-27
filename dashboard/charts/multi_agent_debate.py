"""Dashboard panel — Multi-Agent Debate transcript.

读 data_cache/multi_agent_log.jsonl (由 examples/multi_agent_debate.py 产生).
Schema per line: {agent, ts, sym, name, msg, vote, score}.

视觉:
  - 顶部: 投票汇总 (per sym, BUY/HOLD/SELL 计数)
  - 主体: transcript 按 sym 分组, 每组 3 agents 的 message + vote
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = ROOT / "data_cache" / "multi_agent_log.jsonl"
NAMES_PATH = ROOT / "data_cache" / "stock_names.json"

AGENT_LABEL = {"bull": "🐂 看多", "bear": "🐻 看空", "neutral": "⚖️ 综合"}
AGENT_COLOR = {"bull": "#16a34a", "bear": "#dc2626", "neutral": "#2563eb"}
VOTE_COLOR = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#6b7280"}


def _load_names_map() -> dict[str, str]:
    """sym → 中文名 fallback. multi_agent_log 的 name 可能为空 (xlsx 缺)."""
    if not NAMES_PATH.exists():
        return {}
    try:
        return json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _placeholder(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{message}"
        "</div>"
    )


def _load_lines() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    out: list[dict] = []
    with LOG_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def build_multi_agent_debate_section() -> str:
    lines = _load_lines()
    if not lines:
        return _placeholder(
            "无 multi-agent debate 数据 — 跑 <code>python examples/multi_agent_debate.py</code> "
            "产生 <code>data_cache/multi_agent_log.jsonl</code> 后此 panel 自动填充."
        )

    # log 是 append-only, 含历史多日 debates. 仅显示最新一日数据.
    all_ts = [ln.get("ts", "") for ln in lines if ln.get("ts")]
    if not all_ts:
        return _placeholder("multi_agent_log.jsonl 中无 ts 字段")
    data_date_str = max(all_ts)[:10]
    data_full_ts = max(all_ts)[:19]
    # filter 仅保留最新日期的 entries
    latest_day_lines = [ln for ln in lines if ln.get("ts", "")[:10] == data_date_str]

    by_sym: dict[str, list[dict]] = {}
    for ln in latest_day_lines:
        sym = ln.get("sym", "")
        if not sym:
            continue
        by_sym.setdefault(sym, []).append(ln)

    name_fallback = _load_names_map()

    summary_rows: list[str] = []
    for sym in sorted(by_sym.keys()):
        agents_lines = by_sym[sym]
        name = next((x.get("name", "") for x in agents_lines if x.get("name")), "")
        if not name:  # log 里 name 空 → fallback stock_names.json
            name = name_fallback.get(sym, "?")
        votes = {a.get("agent", ""): a.get("vote", "?") for a in agents_lines}
        neutral_vote = votes.get("neutral", "?")
        vote_color = VOTE_COLOR.get(neutral_vote, "#6b7280")
        bull_score = next(
            (a["score"] for a in agents_lines if a.get("agent") == "bull"), 0
        )
        bear_score = next(
            (a["score"] for a in agents_lines if a.get("agent") == "bear"), 0
        )
        summary_rows.append(
            "<tr>"
            f"<td><code>{html.escape(sym)}</code></td>"
            f"<td>{html.escape(name)}</td>"
            f"<td style='text-align:center; color:#16a34a;'>{bull_score:.2f}</td>"
            f"<td style='text-align:center; color:#dc2626;'>{bear_score:.2f}</td>"
            f"<td style='text-align:center; font-weight:700; color:{vote_color};'>"
            f"{html.escape(neutral_vote)}</td>"
            "</tr>"
        )

    # 顶部数据日期 banner
    date_banner = (
        '<div style="background:rgba(37,99,235,0.06); border-left:3px solid #2563eb; '
        'padding:8px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        f'<strong>📅 数据日期</strong>: <strong>{html.escape(data_date_str)}</strong> '
        f'(最新 ts <code>{html.escape(data_full_ts)}</code>) · '
        f'<strong>{len(by_sym)}</strong> 只 picks 参与 debate · '
        f'<span style="color:var(--muted, #6b7280);">基于 paper_trade picks_today.json 当日 Top 8 + 持仓</span>'
        '</div>'
    )

    summary_table = (
        '<h3 style="font-size:13px; margin:0 0 8px 0; color:var(--muted, #6b7280);">'
        f"投票汇总 ({len(by_sym)} 只)"
        "</h3>"
        '<table class="data" style="margin-bottom:16px;">'
        '<colgroup>'
        '<col style="width:14%;"><col style="width:24%;">'
        '<col style="width:18%;"><col style="width:18%;"><col style="width:26%;">'
        '</colgroup>'
        "<thead><tr>"
        "<th>代码</th><th>名称</th>"
        "<th style='text-align:center;'>🐂 Bull 信心</th>"
        "<th style='text-align:center;'>🐻 Bear 信心</th>"
        "<th style='text-align:center;'>⚖️ 综合 vote</th>"
        "</tr></thead>"
        f"<tbody>{''.join(summary_rows)}</tbody>"
        "</table>"
    )

    transcript_html: list[str] = []
    transcript_html.append(
        '<h3 style="font-size:13px; margin:0 0 8px 0; color:var(--muted, #6b7280);">'
        "Transcript (按 sym 分组, 点击展开)</h3>"
    )
    agent_order = {"bull": 0, "bear": 1, "neutral": 2}
    for sym in sorted(by_sym.keys()):
        agents_lines = sorted(
            by_sym[sym], key=lambda x: agent_order.get(x.get("agent", ""), 9)
        )
        name = next((x.get("name", "") for x in agents_lines if x.get("name")), "")
        if not name:
            name = name_fallback.get(sym, "?")
        transcript_html.append(
            '<details style="margin-bottom:10px; '
            'background:rgba(0,0,0,0.02); border:1px solid var(--border, #e5e7eb); '
            'border-radius:6px; padding:10px 14px;">'
            "<summary style='cursor:pointer; font-weight:600; font-size:13px;'>"
            f"<code>{html.escape(sym)}</code> "
            f"<span style='font-weight:400; color:var(--muted, #6b7280);'>"
            f"{html.escape(name)}</span>"
            "</summary>"
        )
        for ln in agents_lines:
            agent = ln.get("agent", "")
            label = AGENT_LABEL.get(agent, agent)
            color = AGENT_COLOR.get(agent, "#6b7280")
            vote = ln.get("vote", "?")
            vote_color = VOTE_COLOR.get(vote, "#6b7280")
            score = ln.get("score", 0)
            msg = ln.get("msg", "")
            transcript_html.append(
                '<div style="margin-top:8px; padding:6px 10px; '
                f'border-left:3px solid {color}; '
                'background:rgba(255,255,255,0.5); border-radius:0 4px 4px 0;">'
                f"<div style='font-size:11px; color:{color}; font-weight:600;'>"
                f"{label} <span style='color:{vote_color};'>→ {html.escape(vote)} "
                f"({score:.2f})</span></div>"
                f"<div style='font-size:12px; margin-top:2px;'>{html.escape(msg)}</div>"
                "</div>"
            )
        transcript_html.append("</details>")
    transcript = "".join(transcript_html)

    n_lines = len(lines)
    first_ts = lines[0].get("ts", "")
    footer = (
        '<div style="margin-top:12px; font-size:11px; color:var(--muted, #6b7280);">'
        f"共 {n_lines} 条 transcript · 生成 {html.escape(first_ts[:19])} · "
        "数据源 <code>data_cache/multi_agent_log.jsonl</code> "
        "(MVP rule-based, future LLM upgrade)"
        "</div>"
    )

    return date_banner + summary_table + transcript + footer
