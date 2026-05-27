"""Dashboard panel — Multi-agent Debate Veto preview + shadow A/B status."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PICKS_PATH = ROOT / "data_cache" / "picks_today.json"
DEBATE_LOG = ROOT / "data_cache" / "multi_agent_log.jsonl"
AUDIT_LOG = ROOT / "data_cache" / "debate_veto_log.csv"
PAPER_TRADE_PY = ROOT / "examples" / "paper_trade_today.py"

MAX_AUDIT_ROWS = 30


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _read_use_debate_veto():
    if not PAPER_TRADE_PY.exists():
        return None
    import ast as _ast
    try:
        tree = _ast.parse(PAPER_TRADE_PY.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name) and t.id == "USE_DEBATE_VETO":
                    try:
                        return bool(_ast.literal_eval(node.value))
                    except (ValueError, SyntaxError):
                        return None
    return None


def _load_picks() -> list:
    if not PICKS_PATH.exists():
        return []
    try:
        d = json.loads(PICKS_PATH.read_text(encoding="utf-8"))
        return [p.get("sym") for p in d.get("picks", []) if p.get("sym")]
    except Exception:
        return []


def _load_audit() -> list:
    if not AUDIT_LOG.exists():
        return []
    try:
        with AUDIT_LOG.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def build_debate_veto_section() -> str:
    use_veto = _read_use_debate_veto()
    picks = _load_picks()

    try:
        from claude_finance.debate_veto import DebateVeto, load_debate_votes
        veto = DebateVeto()
        preview = veto.preview(picks) if picks else None
    except Exception as e:  # noqa: BLE001
        return _placeholder(
            f"DebateVeto import 失败: <code>{type(e).__name__}: {e}</code>"
        )

    if use_veto is True:
        toggle_color = "#16a34a"
        toggle_text = "✓ ON (production)"
    elif use_veto is False:
        toggle_color = "#6b7280"
        toggle_text = "OFF (默认, shadow 阶段)"
    else:
        toggle_color = "#f59e0b"
        toggle_text = "? 无法解析"

    if preview is None or not picks:
        veto_summary = "无 picks_today.json 数据"
    elif preview.skipped:
        veto_summary = f"⚠ skipped — debate log 缺数据 (source={preview.source_date or 'n/a'})"
    else:
        veto_summary = (
            f"今日 {preview.total_input} picks → kept "
            f'<strong style="color:#16a34a;">{preview.n_kept}</strong>, '
            f'veto <strong style="color:#dc2626;">{preview.n_vetoed}</strong> '
            f"(source={preview.source_date})"
        )

    audit = _load_audit()
    total_audit = len(audit)
    total_vetoed = sum(1 for r in audit if r.get("kept_or_vetoed") == "VETOED")
    shadow_summary = (
        f"shadow audit 累计 <strong>{total_audit}</strong> 决策 / "
        f'<strong style="color:#dc2626;">{total_vetoed}</strong> vetoed'
    )

    banner = (
        '<div style="background:rgba(168,85,247,0.08); border-left:3px solid #a855f7; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        '<strong>🗳️ Debate Veto (P2-opt)</strong> · '
        f'toggle <strong style="color:{toggle_color};">{toggle_text}</strong> · '
        f'{veto_summary}'
        f'<br>{shadow_summary}'
        '<br><span style="color:var(--muted, #6b7280); font-size:11px;">'
        '借鉴 jin-ce-zhi-suan 多 agent 决策, 把 multi_agent_debate panel 从 "展示" 升级 "第二风控闸". '
        '逻辑: neutral 投 SELL → 移出 picks (K 缩小, 不 backfill). '
        '<strong>默认 OFF</strong> — production 需 30 日 shadow A/B 验证 Calmar/Sharpe 显著优于才升级. '
        '一行回滚: <code>USE_DEBATE_VETO=False</code> (paper_trade_today) 或 '
        '<code>DEBATE_VETO_ENABLED=False</code> (debate_veto.py 全局).'
        '</span></div>'
    )

    if preview and not preview.skipped and picks:
        votes_map, _ = load_debate_votes()
        rows = []
        kept_set = set(preview.kept)
        vote_color = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#6b7280"}
        for sym in picks:
            v = votes_map.get(sym, {})
            bull = v.get("bull", "?")
            bear = v.get("bear", "?")
            neutral = v.get("neutral", "?")
            kept = sym in kept_set
            row_bg = "transparent" if kept else "rgba(220,38,38,0.06)"
            status = (
                '<span style="color:#16a34a;">✓ kept</span>'
                if kept else
                '<span style="color:#dc2626; font-weight:600;">✗ VETOED</span>'
            )
            rows.append(
                f'<tr style="background:{row_bg};">'
                f'<td><code>{html.escape(sym)}</code></td>'
                f'<td style="text-align:center; color:{vote_color.get(bull, "#6b7280")};">'
                f'{html.escape(bull)}</td>'
                f'<td style="text-align:center; color:{vote_color.get(bear, "#6b7280")};">'
                f'{html.escape(bear)}</td>'
                f'<td style="text-align:center; color:{vote_color.get(neutral, "#6b7280")}; font-weight:600;">'
                f'{html.escape(neutral)}</td>'
                f'<td style="text-align:center;">{status}</td>'
                "</tr>"
            )
        picks_table = (
            '<h3 style="font-size:13px; margin:14px 0 6px 0; color:var(--muted, #6b7280);">'
            f'📊 今日 picks × 3 agents votes ({len(picks)})</h3>'
            '<table class="data">'
            '<colgroup>'
            '<col style="width:20%;"><col style="width:18%;">'
            '<col style="width:18%;"><col style="width:20%;"><col style="width:24%;">'
            '</colgroup>'
            '<thead><tr>'
            '<th>代码</th><th>🐂 bull</th><th>🐻 bear</th><th>⚖️ neutral</th><th>状态</th>'
            '</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    else:
        picks_table = ""

    if audit:
        recent = list(reversed(audit))[:MAX_AUDIT_ROWS]
        audit_rows = []
        for r in recent:
            vetoed = r.get("kept_or_vetoed") == "VETOED"
            row_bg = "transparent" if not vetoed else "rgba(220,38,38,0.06)"
            status = (
                '<span style="color:#dc2626; font-weight:600;">VETOED</span>'
                if vetoed else
                '<span style="color:#16a34a;">kept</span>'
            )
            audit_rows.append(
                f'<tr style="background:{row_bg};">'
                f'<td style="font-family:monospace; font-size:11px;">{html.escape(r.get("dt", "")[:19])}</td>'
                f'<td><code>{html.escape(r.get("sym", "?"))}</code></td>'
                f'<td style="font-family:monospace; font-size:11px;">{html.escape(r.get("date_of_debate", ""))}</td>'
                f'<td style="text-align:center;">{html.escape(r.get("bull_vote", ""))}</td>'
                f'<td style="text-align:center;">{html.escape(r.get("bear_vote", ""))}</td>'
                f'<td style="text-align:center; font-weight:600;">{html.escape(r.get("neutral_vote", ""))}</td>'
                f'<td style="text-align:center;">{status}</td>'
                "</tr>"
            )
        audit_html = (
            '<details style="margin-top:14px;">'
            '<summary style="cursor:pointer; font-size:13px; color:var(--muted, #6b7280);">'
            f'📋 展开 audit log (最近 {len(recent)} / 共 {total_audit})'
            '</summary>'
            '<table class="data" style="margin-top:8px; font-size:11px;">'
            '<thead><tr>'
            '<th>时间</th><th>sym</th><th>debate 日期</th>'
            '<th>bull</th><th>bear</th><th>neutral</th><th>状态</th>'
            '</tr></thead>'
            f"<tbody>{''.join(audit_rows)}</tbody></table></details>"
        )
    else:
        audit_html = (
            '<div style="margin-top:14px; padding:10px 14px; '
            'background:rgba(107,114,128,0.06); border-left:3px solid #6b7280; '
            'border-radius:0 4px 4px 0; font-size:11px; color:var(--muted, #6b7280);">'
            '尚无 audit 记录 (USE_DEBATE_VETO=False 时 paper_trade 不调用 veto). '
            '本 panel 显示 preview 不写 audit.'
            '</div>'
        )

    protocol_html = (
        '<details style="margin-top:14px;">'
        '<summary style="cursor:pointer; font-size:13px; color:var(--muted, #6b7280);">'
        '📜 展开 30 日 shadow A/B 升级 / 降级协议</summary>'
        '<div style="margin-top:8px; padding:10px 14px; '
        'background:rgba(168,85,247,0.06); border-left:3px solid #a855f7; '
        'border-radius:0 4px 4px 0; font-size:12px; line-height:1.7;">'
        '<strong>升级条件 (USE_DEBATE_VETO=True 需全部满足)</strong>:<br>'
        '1. shadow 期 ≥ 30 个交易日 audit log<br>'
        '2. neutral=SELL picks 后续 5d/10d 真实收益 < kept picks 平均 (信号有效)<br>'
        '3. vetoed picks 命中率 ≥ 65%<br>'
        '4. daily kept 中位数 ≥ 6 (不会因频繁 veto 仓位过空)<br>'
        '5. shadow simulate Calmar 比 v19.10 production 2.12 提升 ≥ 10%<br>'
        '<br>'
        '<strong>降级条件 (任一触发立即 OFF)</strong>:<br>'
        '1. 单日 veto 移除 ≥ K-2 picks<br>'
        '2. vetoed 命中率 < 50% (反信号)<br>'
        '3. neutral 当日 SELL 比例 > 30% (debate 失效)<br>'
        '</div></details>'
    )

    return banner + picks_table + audit_html + protocol_html
