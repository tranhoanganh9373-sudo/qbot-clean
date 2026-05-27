"""Dashboard panel — 告警中心 (集中显示所有 alert sources).

聚合 4 处 alert + 服务状态 → 按 severity 排序统一展示.

数据源 (全只读):
  - data_cache/forward_oos_alerts.csv (5-level: green/yellow/orange/red/black)
  - data_cache/sanity_check_log.csv (daily 数据完整性, overall_pass=False 即 alert)
  - data_cache/kline_5m_failed.csv (5m fetch 失败 stocks)
  - launchctl list | grep claude_finance (三服务状态)
"""
from __future__ import annotations

import csv
import html
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FORWARD_OOS_ALERTS = ROOT / "data_cache" / "forward_oos_alerts.csv"
SANITY_LOG = ROOT / "data_cache" / "sanity_check_log.csv"
KLINE_5M_FAILED = ROOT / "data_cache" / "kline_5m_failed.csv"

SEVERITY_META = {
    0: ("info", "#2563eb", "rgba(37,99,235,0.10)"),
    1: ("warning", "#f59e0b", "rgba(245,158,11,0.10)"),
    2: ("error", "#dc2626", "rgba(220,38,38,0.10)"),
    3: ("critical", "#991b1b", "rgba(153,27,27,0.15)"),
}

LEVEL_SEVERITY = {
    "green": 0, "yellow": 1, "orange": 2, "red": 3, "black": 3,
}


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _read_csv_last(path: Path, n: int = 1) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return rows[-n:] if rows else []
    except Exception:
        return []


def _collect_forward_oos_alerts() -> list[dict]:
    rows = _read_csv_last(FORWARD_OOS_ALERTS, n=10)
    if not rows:
        return []
    latest = rows[-1]
    level = latest.get("level", "").lower()
    sev = LEVEL_SEVERITY.get(level, 0)
    portfolio = latest.get("portfolio_value", "")
    cum = latest.get("cum_60d", "")
    triggers = latest.get("triggers", "").strip() or "—"
    history_warnings = sum(
        1 for r in rows if LEVEL_SEVERITY.get(r.get("level", "").lower(), 0) >= 1
    )
    msg_parts = [f"level=<strong>{level}</strong>"]
    if portfolio:
        msg_parts.append(f"组合 ¥{portfolio}")
    if cum:
        msg_parts.append(f"60d cum {cum}")
    if triggers and triggers != "—":
        msg_parts.append(f"triggers: {triggers}")
    if history_warnings > 1:
        msg_parts.append(f"近 10 日 yellow+ 出现 {history_warnings} 次")
    return [{
        "source": "Forward OOS monitor",
        "severity": sev,
        "date": latest.get("date", ""),
        "summary": " · ".join(msg_parts),
    }]


def _collect_sanity_alerts() -> list[dict]:
    rows = _read_csv_last(SANITY_LOG, n=1)
    if not rows:
        return []
    r = rows[0]
    passed = r.get("overall_pass", "").lower() == "true"
    if passed:
        return [{
            "source": "Data sanity check",
            "severity": 0,
            "date": r.get("date", ""),
            "summary": "5 checks all PASS (neg/jump/freshness/coverage/low)",
        }]
    fails = []
    if r.get("check_neg_close", "").lower() != "true":
        fails.append("neg-close")
    if r.get("check_extreme_low", "").lower() != "true":
        fails.append("extreme-low")
    if r.get("check_extreme_jump", "").lower() != "true":
        fails.append("extreme-jump")
    if r.get("check_freshness", "").lower() != "true":
        fails.append("freshness")
    if r.get("check_coverage", "").lower() != "true":
        fails.append("coverage")
    detail = (r.get("fail_details", "") or "")[:180].replace("\n", " ")
    return [{
        "source": "Data sanity check",
        "severity": 2,
        "date": r.get("date", ""),
        "summary": f"FAIL: {', '.join(fails)} · {html.escape(detail)}…",
    }]


def _collect_5m_failed() -> list[dict]:
    if not KLINE_5M_FAILED.exists():
        return []
    try:
        n = sum(1 for _ in KLINE_5M_FAILED.open()) - 1
    except Exception:
        return []
    if n <= 0:
        return []
    severity = 0 if n < 10 else 1 if n < 100 else 2
    return [{
        "source": "5m fetch failures",
        "severity": severity,
        "date": datetime.fromtimestamp(
            KLINE_5M_FAILED.stat().st_mtime).strftime("%Y-%m-%d"),
        "summary": f"{n} stocks 5m 抓取失败 (post-2021 IPO / 退市股, 通常无需介入)",
    }]


def _collect_launchd_alerts() -> list[dict]:
    expected = [
        "com.claude_finance.dashboard_server",
        "com.claude_finance.daily_check",
        "com.claude_finance.poll_5m_picks",
    ]
    try:
        out = subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return [{
                "source": "launchd",
                "severity": 1,
                "date": "",
                "summary": "launchctl list 失败",
            }]
    except Exception:
        return []
    line_map: dict[str, tuple[str, str]] = {}
    for ln in out.stdout.splitlines():
        parts = ln.split("\t")
        if len(parts) >= 3:
            pid, code, label = parts[0], parts[1], parts[2]
            line_map[label] = (pid, code)
    alerts: list[dict] = []
    for svc in expected:
        if svc not in line_map:
            alerts.append({
                "source": "launchd",
                "severity": 1,
                "date": "",
                "summary": f"{html.escape(svc)} 未 load (等 fetch 完 watcher 自动 load 或手动 launchctl load -w)",
            })
            continue
        pid, code = line_map[svc]
        if code != "0" and code != "-":
            try:
                code_i = int(code)
                if code_i < 0:
                    msg = (f"{html.escape(svc)} last exit by signal {-code_i} "
                           "(可能 restart loop)")
                    sev = 2
                else:
                    msg = f"{html.escape(svc)} last exit code {code_i}"
                    sev = 1
            except ValueError:
                msg = f"{html.escape(svc)} exit code {html.escape(code)}"
                sev = 1
            alerts.append({"source": "launchd", "severity": sev, "date": "", "summary": msg})
    return alerts


def build_alert_center_section() -> str:
    all_alerts: list[dict] = []
    all_alerts.extend(_collect_forward_oos_alerts())
    all_alerts.extend(_collect_sanity_alerts())
    all_alerts.extend(_collect_5m_failed())
    all_alerts.extend(_collect_launchd_alerts())

    if not all_alerts:
        return _placeholder("🟢 无 alert — 所有 sources 检查通过 / 文件不存在.")

    all_alerts.sort(key=lambda a: (-a["severity"], a.get("date", "")))

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for a in all_alerts:
        counts[a["severity"]] = counts.get(a["severity"], 0) + 1

    critical_color = "#991b1b" if counts[3] > 0 else (
        "#dc2626" if counts[2] > 0 else (
            "#f59e0b" if counts[1] > 0 else "#16a34a"))
    banner = (
        '<div style="background:rgba(107,114,128,0.08); border-left:3px solid '
        f'{critical_color}; padding:10px 14px; margin-bottom:12px; '
        'border-radius:0 4px 4px 0; font-size:12px;">'
        f'<strong>🚨 告警中心</strong> · 共 <strong>{len(all_alerts)}</strong> 条 '
        f'(<span style="color:#991b1b; font-weight:600;">critical {counts[3]}</span> / '
        f'<span style="color:#dc2626; font-weight:600;">error {counts[2]}</span> / '
        f'<span style="color:#f59e0b; font-weight:600;">warning {counts[1]}</span> / '
        f'<span style="color:#2563eb; font-weight:600;">info {counts[0]}</span>)<br>'
        '<span style="color:var(--muted, #6b7280); font-size:11px;">'
        "汇总: forward_oos / data sanity / 5m fetch / launchd services. "
        "按 severity 倒序, 同级按日期倒序."
        "</span></div>"
    )

    cols = [
        ("#", 4), ("severity", 12), ("source", 18), ("date", 11), ("summary", 55),
    ]
    assert sum(w for _, w in cols) == 100
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _, w in cols
    ) + "</colgroup>"
    thead = "<tr>" + "".join(f"<th>{label}</th>" for label, _ in cols) + "</tr>"

    body_rows = []
    for i, a in enumerate(all_alerts, 1):
        sev_label, sev_color, sev_bg = SEVERITY_META.get(
            a["severity"], ("?", "#6b7280", "rgba(107,114,128,0.10)"),
        )
        body_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f'<td><span style="color:{sev_color}; background:{sev_bg}; '
            f'padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600;">'
            f"{sev_label}</span></td>"
            f'<td style="font-size:12px;">{html.escape(a["source"])}</td>'
            f'<td style="font-size:11px; color:var(--muted, #6b7280);">'
            f'{html.escape(a.get("date", ""))}</td>'
            f'<td style="font-size:12px;">{a["summary"]}</td>'
            "</tr>"
        )
    table = (
        '<table class="data">'
        + colgroup
        + f"<thead>{thead}</thead>"
        + f"<tbody>{''.join(body_rows)}</tbody>"
        + "</table>"
    )

    footer = (
        '<div style="margin-top:10px; font-size:11px; color:var(--muted, #6b7280);">'
        "🟢 info · 🟡 warning · 🔴 error · ⛔ critical. "
        "数据源: <code>forward_oos_alerts.csv</code> + <code>sanity_check_log.csv</code> + "
        "<code>kline_5m_failed.csv</code> + <code>launchctl list</code>."
        "</div>"
    )

    return banner + table + footer
