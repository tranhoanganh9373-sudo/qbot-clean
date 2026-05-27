"""Dashboard panel — 服务状态汇总 (launchd 3 services + fetch + watcher).

数据源 (全只读):
  - launchctl list (3 services)
  - ps -p $PID -o etime=,rss= (per-service uptime + RAM)
  - pgrep -f fetch_mootdx_5m_5y_backfill.py (fetch background process)
  - /tmp/post_fetch_chain.log tail (watcher chain state)
  - data_cache/picks_today.json + live_5m_picks.json mtime (新鲜度)
"""
from __future__ import annotations

import html
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PICKS_TODAY = ROOT / "data_cache" / "picks_today.json"
LIVE_5M = ROOT / "data_cache" / "live_5m_picks.json"
CHAIN_LOG = Path("/tmp/post_fetch_chain.log")

SERVICES = [
    ("com.claude_finance.dashboard_server", "Dashboard server (port 5557)",
     "GET /, POST /submit, hot-render"),
    ("com.claude_finance.poll_5m_picks", "5m poller daemon",
     "60s polling Top 8 + 持仓"),
    ("com.claude_finance.daily_check", "Daily check 16:30 cron",
     "paper_trade + render + forward OOS"),
]


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _launchctl_status() -> dict:
    try:
        out = subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return {}
    except Exception:
        return {}
    result: dict = {}
    for ln in out.stdout.splitlines():
        parts = ln.split("\t")
        if len(parts) >= 3 and parts[2].startswith("com.claude_finance."):
            pid, code, label = parts[0], parts[1], parts[2]
            result[label] = {"pid": pid, "code": code}
    return result


def _ps_etime_rss(pid: str) -> tuple[str, str]:
    if not pid or pid == "-":
        return "—", "—"
    try:
        out = subprocess.run(
            ["ps", "-p", pid, "-o", "etime=,rss="],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            parts = out.stdout.split()
            if len(parts) >= 2:
                etime, rss_kb = parts[0], parts[1]
                rss_mb = f"{int(rss_kb) / 1024:.0f} MB"
                return etime, rss_mb
    except Exception:
        pass
    return "?", "?"


def _fetch_status() -> dict:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "fetch_mootdx_5m_5y_backfill.py"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            pid = out.stdout.strip().split("\n")[0]
            etime, rss = _ps_etime_rss(pid)
            return {"running": True, "pid": pid, "etime": etime, "rss": rss}
    except Exception:
        pass
    return {"running": False, "pid": "—", "etime": "—", "rss": "—"}


def _watcher_state() -> str:
    if not CHAIN_LOG.exists():
        return "未启动"
    try:
        lines = CHAIN_LOG.read_text(encoding="utf-8").splitlines()
    except Exception:
        return "log 读取失败"
    if not lines:
        return "log 空"
    tail = lines[-1]
    if "DONE" in tail:
        return "✓ 已完成 (Hive rebuild + poller loaded)"
    if "Step 2/2" in tail or "load poller" in tail:
        return "🔄 loading poller plist"
    if "Step 1/2" in tail or "rebuild Hive" in tail:
        return "🔄 rebuilding 5m Hive"
    if "waiting" in tail:
        return "⏳ 等 fetch 退出"
    return tail[:80]


def _mtime_human(p: Path) -> str:
    if not p.exists():
        return "不存在"
    age = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds()
    if age < 60:
        return f"{int(age)} 秒前"
    if age < 3600:
        return f"{int(age / 60)} 分钟前"
    if age < 86400:
        return f"{age / 3600:.1f} 小时前"
    return f"{age / 86400:.1f} 天前"


def build_server_status_section() -> str:
    launchd = _launchctl_status()
    fetch = _fetch_status()
    chain_state = _watcher_state()
    picks_freshness = _mtime_human(PICKS_TODAY)
    live_5m_freshness = _mtime_human(LIVE_5M)

    n_loaded = sum(1 for s, _, _ in SERVICES if s in launchd)
    n_ok = sum(
        1 for s, _, _ in SERVICES if s in launchd and launchd[s]["code"] in ("0", "-")
    )
    overall_color = "#16a34a" if n_ok == 3 else "#f59e0b" if n_ok >= 2 else "#dc2626"
    overall_label = (
        "✓ ALL GREEN" if n_ok == 3 else
        f"⚠ 部分异常 ({n_ok}/3)" if n_ok >= 1 else
        "🔴 全部异常"
    )
    banner = (
        '<div style="background:rgba(107,114,128,0.08); border-left:3px solid '
        f'{overall_color}; padding:10px 14px; margin-bottom:12px; '
        'border-radius:0 4px 4px 0; font-size:12px;">'
        f'<strong>🖥️ 服务状态</strong> · <strong>{overall_label}</strong> · '
        f'已 load <strong>{n_loaded}/3</strong> · '
        f'正常 <strong>{n_ok}/3</strong>'
        '<br><span style="color:var(--muted, #6b7280); font-size:11px;">'
        f"production picks 新鲜度: <strong>{picks_freshness}</strong> · "
        f"live 5m 新鲜度: <strong>{live_5m_freshness}</strong>"
        "</span></div>"
    )

    cols = [
        ("服务", 28), ("状态", 8), ("PID", 7), ("uptime", 9),
        ("RAM", 8), ("exit code", 9), ("说明", 31),
    ]
    assert sum(w for _, w in cols) == 100
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _, w in cols
    ) + "</colgroup>"
    thead = "<tr>" + "".join(f"<th>{l}</th>" for l, _ in cols) + "</tr>"

    body_rows: list[str] = []
    for svc, label, desc in SERVICES:
        info = launchd.get(svc)
        if info is None:
            status_lbl, status_c = "未 load", "#dc2626"
            etime, rss = "—", "—"
            pid_show = "—"
            code_show = "—"
        else:
            pid = info["pid"]
            code = info["code"]
            if pid == "-":
                status_lbl, status_c = "已 load 待跑", "#6b7280"
                etime, rss = "—", "—"
                pid_show = "—"
            else:
                status_lbl, status_c = "✓ 运行", "#16a34a"
                etime, rss = _ps_etime_rss(pid)
                pid_show = pid
            if code in ("0", "-"):
                code_show = f'<span style="color:#16a34a;">{code}</span>'
            else:
                code_show = f'<span style="color:#dc2626;">{code}</span>'
        body_rows.append(
            "<tr>"
            f'<td style="font-size:12px;">{html.escape(label)}<br>'
            f'<code style="font-size:10px; color:var(--muted, #6b7280);">'
            f'{html.escape(svc)}</code></td>'
            f'<td style="color:{status_c}; font-weight:600; font-size:11px;">{status_lbl}</td>'
            f'<td><code>{html.escape(pid_show)}</code></td>'
            f'<td style="font-family:monospace; font-size:11px;">{html.escape(etime)}</td>'
            f'<td style="text-align:right; font-size:11px;">{html.escape(rss)}</td>'
            f'<td style="text-align:right; font-size:11px;">{code_show}</td>'
            f'<td style="font-size:11px; color:var(--muted, #6b7280);">{html.escape(desc)}</td>'
            "</tr>"
        )
    table = (
        '<table class="data">'
        + colgroup
        + f"<thead>{thead}</thead>"
        + f"<tbody>{''.join(body_rows)}</tbody>"
        + "</table>"
    )

    fetch_color = "#2563eb" if fetch["running"] else "#16a34a"
    fetch_lbl = "🔄 running" if fetch["running"] else "✓ 已完成 / 未跑"
    extras = (
        '<h3 style="font-size:13px; margin:14px 0 6px 0; color:var(--muted, #6b7280);">'
        "🔧 后台任务</h3>"
        '<div style="display:flex; gap:10px; flex-wrap:wrap;">'
        f'<div style="background:rgba(107,114,128,0.06); padding:10px 12px; '
        f'border-left:3px solid {fetch_color}; border-radius:0 4px 4px 0; flex:1; min-width:220px;">'
        f'<div style="font-weight:600; font-size:12px;">5m fetch backfill</div>'
        f'<div style="font-size:11px; color:{fetch_color}; font-weight:600;">{fetch_lbl}</div>'
        f'<div style="font-size:11px; color:var(--muted, #6b7280); margin-top:4px;">'
        f'PID: <code>{html.escape(fetch["pid"])}</code> · '
        f'uptime <code>{html.escape(fetch["etime"])}</code> · '
        f'RAM <code>{html.escape(fetch["rss"])}</code>'
        f'</div></div>'
        f'<div style="background:rgba(107,114,128,0.06); padding:10px 12px; '
        f'border-left:3px solid #f59e0b; border-radius:0 4px 4px 0; flex:1; min-width:220px;">'
        f'<div style="font-weight:600; font-size:12px;">post-fetch watcher chain</div>'
        f'<div style="font-size:11px;">{html.escape(chain_state)}</div>'
        f'<div style="font-size:11px; color:var(--muted, #6b7280); margin-top:4px;">'
        f'log: <code>/tmp/post_fetch_chain.log</code>'
        f'</div></div>'
        "</div>"
    )

    footer = (
        '<div style="margin-top:10px; font-size:11px; color:var(--muted, #6b7280);">'
        "stop/start: <code>launchctl unload/load -w ~/Library/LaunchAgents/&lt;label&gt;.plist</code>. "
        "logs: <code>/tmp/dashboard_server_*.log</code> · "
        "<code>/tmp/poll_5m_*.log</code> · "
        "<code>/tmp/daily_check_*.log</code>."
        "</div>"
    )

    return banner + table + extras + footer
