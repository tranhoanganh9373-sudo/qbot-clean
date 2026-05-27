"""Dashboard panel — 选中股票 (Top 8 picks + 当前持仓) 的实时 5m 行情.

B 路线 execution overlay: 不动 production 选股 v19.10, 仅展示 5m 最新 bar
用于人工判断 (真跳空 / 假跳空 / intra-day breakout).

数据源:
  - data_cache/live_5m_picks.json (由 examples/poll_5m_picks.py daemon 每 60s 更新)

视觉:
  - 顶部 banner: 更新时间 + ok/fail + stale 警告 (>180s 视为掉线)
  - 表格列: # / sym / category / 5m datetime / open / high / low / close / vol /
            5m 涨跌% / JZF / confirm 标记
"""
from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
LIVE_PATH = ROOT / "data_cache" / "live_5m_picks.json"
NAMES_PATH = ROOT / "data_cache" / "stock_names.json"

STALE_THRESHOLD_SEC = 180


def _load_names() -> dict[str, str]:
    """读 sym → 中文名 cache. 容错: 文件不在或 parse 失败返空 dict."""
    if not NAMES_PATH.exists():
        return {}
    try:
        return json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sparkline_base64(closes: list[float]) -> str:
    """render 一只股票 close 序列 sparkline 为 base64 inline PNG.
    红/绿 取决于 last vs first close. 无 axis, 极简."""
    if not closes or len(closes) < 2:
        return ""
    color = "#16a34a" if closes[-1] >= closes[0] else "#dc2626"
    fig, ax = plt.subplots(figsize=(1.6, 0.35), dpi=80)
    ax.plot(closes, color=color, linewidth=1.0)
    ax.fill_between(range(len(closes)), closes, min(closes), alpha=0.15, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                facecolor="none", transparent=True)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="sparkline" '\
           f'style="display:block; height:24px; margin:0 auto;"/>'
CATEGORY_BADGE = {
    "pick": ("Top 8", "#2563eb", "rgba(37,99,235,0.10)"),
    "holding": ("持仓", "#f59e0b", "rgba(245,158,11,0.10)"),
    "both": ("持仓+Top 8", "#16a34a", "rgba(22,163,74,0.10)"),
}


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _format_secs_ago(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} 秒前"
    if seconds < 3600:
        return f"{int(seconds / 60)} 分钟前"
    return f"{seconds / 3600:.1f} 小时前"


def _row_html(idx: int, sym_obj: dict, name_map: dict[str, str]) -> str:
    sym = sym_obj.get("sym", "?")
    name = name_map.get(sym, "?")
    cat = sym_obj.get("category", "?")
    badge_label, badge_color, badge_bg = CATEGORY_BADGE.get(
        cat, (cat, "#6b7280", "rgba(107,114,128,0.10)"),
    )
    err = sym_obj.get("error")
    if err:
        return (
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><code>{html.escape(sym)}</code></td>"
            f"<td>{html.escape(name)}</td>"
            f'<td><span style="color:{badge_color}; background:{badge_bg}; '
            f'padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600;">'
            f"{badge_label}</span></td>"
            f'<td colspan="10" style="color:#dc2626; font-style:italic; font-size:12px;">'
            f"fetch fail: {html.escape(str(err))}</td>"
            "</tr>"
        )
    bar = sym_obj.get("latest_5m") or {}
    ret = bar.get("ret_5m_pct", 0.0)
    ret_color = "#16a34a" if ret > 0 else "#dc2626" if ret < 0 else "#6b7280"
    jzf = sym_obj.get("jzf")
    jzf_str = f"{jzf:+.2f}%" if isinstance(jzf, (int, float)) else "—"
    if jzf is None:
        conf_label, conf_color = "—", "#6b7280"
    elif jzf > 0 and ret >= -0.5:
        conf_label, conf_color = "✓ 真跳空", "#16a34a"
    elif jzf > 0 and ret < -0.5:
        conf_label, conf_color = "⚠ 假跳空", "#f59e0b"
    elif jzf < 0 and ret > 0.5:
        conf_label, conf_color = "⚠ 反转", "#a855f7"
    else:
        conf_label, conf_color = "—", "#6b7280"
    dt_str = bar.get("datetime", "?")
    if dt_str and len(dt_str) >= 5:
        dt_str = dt_str[-8:]
    return (
        "<tr>"
        f"<td>{idx}</td>"
        f"<td><code>{html.escape(sym)}</code></td>"
        f"<td>{html.escape(name)}</td>"
        f'<td><span style="color:{badge_color}; background:{badge_bg}; '
        f'padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600;">'
        f"{badge_label}</span></td>"
        f'<td style="font-size:11px; color:var(--muted, #6b7280);">'
        f"{html.escape(dt_str)}</td>"
        f'<td style="text-align:right;">{bar.get("open", 0):.2f}</td>'
        f'<td style="text-align:right;">{bar.get("high", 0):.2f}</td>'
        f'<td style="text-align:right;">{bar.get("low", 0):.2f}</td>'
        f'<td style="text-align:right; font-weight:700; font-size:14px; '
        f'color:{ret_color};">{bar.get("close", 0):.2f}</td>'
        f'<td style="text-align:right; font-size:11px;">{bar.get("volume", 0):,}</td>'
        f'<td style="text-align:right; color:{ret_color}; font-weight:600;">'
        f"{ret:+.2f}%</td>"
        f'<td style="text-align:right;">{jzf_str}</td>'
        f'<td style="color:{conf_color}; font-weight:600;">{conf_label}</td>'
        f"<td>{_sparkline_base64(sym_obj.get('sparkline_closes') or [])}</td>"
        "</tr>"
    )


def build_live_5m_picks_section() -> str:
    if not LIVE_PATH.exists():
        return _placeholder(
            "无 live 5m 数据 — 启动 poller: "
            "<code>.venv/bin/python examples/poll_5m_picks.py</code> 或 "
            "<code>launchctl load ~/Library/LaunchAgents/com.claude_finance.poll_5m_picks.plist</code>"
        )
    try:
        payload = json.loads(LIVE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return _placeholder(
            f"<code>live_5m_picks.json</code> 解析失败: "
            f"<code>{type(exc).__name__}: {html.escape(str(exc))}</code>"
        )

    updated_at = payload.get("updated_at", "?")
    try:
        upd_dt = datetime.fromisoformat(updated_at)
        elapsed = (datetime.now() - upd_dt).total_seconds()
        ago_str = _format_secs_ago(elapsed)
        stale = elapsed > STALE_THRESHOLD_SEC
    except Exception:
        ago_str, stale = "?", True

    n_polled = payload.get("n_polled", 0)
    n_ok = payload.get("n_ok", 0)
    n_fail = payload.get("n_fail", 0)

    banner_bg = "rgba(220,38,38,0.06)" if stale else "rgba(22,163,74,0.06)"
    banner_border = "#dc2626" if stale else "#16a34a"
    stale_warn = " ⚠ <strong style='color:#dc2626;'>STALE</strong>" if stale else ""
    banner_html = (
        f'<div style="background:{banner_bg}; border-left:3px solid {banner_border}; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        "<strong>📡 5m 行情 poller</strong>"
        f' · 更新 <code>{html.escape(updated_at)}</code> ({ago_str}){stale_warn}'
        f' · polled <strong>{n_polled}</strong> · ok <strong>{n_ok}</strong>'
        f' · fail <strong>{n_fail}</strong>'
        '<br><span style="color:var(--muted, #6b7280); font-size:11px;">'
        "B 路线 execution overlay: 不动 production 选股 v19.10, 仅供入场点 audit "
        "(真跳空/假跳空/intra-day 反转). poller daemon by launchd / 60s 间隔. "
        '<strong>现价 (close)</strong> = 最新完成 5m bar 收盘价 ≈ 实时价 (盘中 bar 形成中持续更新).'
        "</span></div>"
    )

    syms = payload.get("syms", [])
    if not syms:
        return banner_html + _placeholder(
            "syms 列表为空 — picks_today.json + portfolio_state.json 都没有数据."
        )

    cols = [
        ("#", 3), ("sym", 7), ("名称", 9), ("分类", 8),
        ("bar 时间", 6), ("open", 5), ("high", 5), ("low", 5), ("现价 (close)", 8),
        ("vol", 6), ("5m 涨跌", 7), ("JZF", 5), ("confirm", 10), ("K 线", 16),
    ]
    assert sum(w for _, w in cols) == 100, sum(w for _, w in cols)
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _, w in cols
    ) + "</colgroup>"
    thead_cells = []
    for label, _ in cols:
        if label in ("open", "high", "low", "现价 (close)", "vol", "5m 涨跌", "JZF"):
            thead_cells.append(f'<th style="text-align:right;">{label}</th>')
        elif label == "K 线":
            thead_cells.append(f'<th style="text-align:center;">{label}</th>')
        else:
            thead_cells.append(f"<th>{label}</th>")
    thead = "<tr>" + "".join(thead_cells) + "</tr>"

    name_map = _load_names()
    body_rows = "".join(_row_html(i, s, name_map) for i, s in enumerate(syms, 1))

    table_html = (
        '<table class="data">'
        + colgroup
        + f"<thead>{thead}</thead>"
        + f"<tbody>{body_rows}</tbody>"
        + "</table>"
    )

    footer = (
        '<div style="margin-top:10px; font-size:11px; color:var(--muted, #6b7280);">'
        "💡 <strong>confirm 列含义</strong>: "
        '<span style="color:#16a34a;">✓ 真跳空</span> = JZF&gt;0 + 5m ret≥-0.5% '
        "(信号一致, 9:35 入场); "
        '<span style="color:#f59e0b;">⚠ 假跳空</span> = JZF&gt;0 + 5m ret&lt;-0.5% '
        "(集合竞价开高后 5m 已回吐, 建议延后入场); "
        '<span style="color:#a855f7;">⚠ 反转</span> = JZF&lt;0 + 5m ret&gt;0.5% '
        "(开低后反弹).<br>"
        "数据源 <code>data_cache/live_5m_picks.json</code> · "
        "poller <code>examples/poll_5m_picks.py</code> · 不进 production scoring"
        "</div>"
    )

    # 包 wrapper div: dashboard 通过 AJAX 局部替换该 div 内容 (60s 自动更新)
    return (
        '<div id="live-5m-content">'
        + banner_html + table_html + footer
        + '</div>'
    )
