"""Daily check status panel — visualize 最后一次 daily_check.sh 运行状态.

数据源 (全只读):
  - `/tmp/daily_check_YYYYMMDD.log` (daily_check.sh tee 输出, 按日期 stamped)
  - `/tmp/daily_check_stdout.log` + `/tmp/daily_check_stderr.log` (launchd 兜底)
  - `launchctl print gui/501/com.claude_finance.daily_check` (subprocess, last exit)
  - `examples/com.claude_finance.daily_check.plist` (兜底 parse StartCalendarInterval)

行为:
  - 解析 step 行 `[1/4]...`, `[1.5/4]...`, `[3.5/4]...`
  - 统计每 step 状态: ok / warn / fail / unknown (从 step 内 FAIL/error/✗/CRITICAL/OK/✓ 关键字)
  - 提取 final exit code `=== ... Done (exit=N) ===`
  - 提取关键告警: 🚨 / SANITY / CRITICAL / Alert level
  - 推算 launchd 下次 trigger (StartCalendarInterval Hour/Minute, 默认 16:30)
  - log tail 12 行

绝不修改: production / data_cache / daily_check.sh / launchd plist.
"""
from __future__ import annotations

import html
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent

# 候选 log path — 按优先级.
# daily_check.sh 用 `/tmp/daily_check_$(date +%Y%m%d).log` 主写, launchd stdout/stderr 兜底.
LOG_CANDIDATES_STATIC = [
    Path("/tmp/daily_check_stdout.log"),
    Path("/tmp/daily_check_stderr.log"),
    ROOT / "logs" / "daily_check.log",
    Path.home() / "Library" / "Logs" / "com.claude_finance.daily_check.log",
]

LAUNCHD_LABEL = "com.claude_finance.daily_check"
PLIST_PATH = ROOT / "examples" / "com.claude_finance.daily_check.plist"

# 颜色 (与 template.html CSS var 协调, 但不依赖 — 直接 inline)
COLOR_OK = "#16a34a"
COLOR_WARN = "#f59e0b"
COLOR_FAIL = "#dc2626"
COLOR_UNKNOWN = "#6b7280"

STATUS_COLORS = {
    "ok": COLOR_OK,
    "warn": COLOR_WARN,
    "fail": COLOR_FAIL,
    "unknown": COLOR_UNKNOWN,
    "pending": "#94a3b8",
    "running": "#3b82f6",
    "skipped": "#6b7280",
}

STEP_PATTERN = re.compile(r"^\[(\d+(?:\.\d+)?)/(\d+)\]\s*(.+?)\s*$")
EXIT_PATTERN = re.compile(r"Done\s*\(exit=(-?\d+)\)|exit\s*[:=]\s*(-?\d+)")
ERROR_KEYWORDS = re.compile(r"(🚨|FAIL|✗|❌|Error|ERROR|CRITICAL|Traceback)")
WARN_KEYWORDS = re.compile(r"(⚠|WARN|WARNING|warning)")
SUCCESS_KEYWORDS = re.compile(r"(✓|\bOK\b|saved|done|complete|完成|green)")

# daily_check.sh 顶层 step 的 label 起始关键词 (用于过滤子脚本嵌套 [N/M] 输出).
# 顺序与 daily_check.sh 里 echo 顺序一致, 用于识别真正的 outer steps.
OUTER_STEP_LABEL_PREFIXES = (
    "Fetching today's kline",
    "Data sanity check",
    "Data completeness check",
    "Margin incremental fetch",
    "Running paper_trade signals",
    "Forward OOS monitoring",
    "Shadow v19.4 paper_trade",
    "Multi-Agent Debate",
    "Shadow v19.6 paper_trade",
    "Rendering dashboard report",
)

# Step label 英文 → 中文映射 (仅渲染时用; 内部匹配 log 仍用英文)
LABEL_ZH = {
    "Fetching today's kline": "拉取今日 K 线",
    "Data sanity check": "数据 sanity 检查",
    "Data completeness check": "数据覆盖率检查",
    "Margin incremental fetch": "融资余额增量拉取",
    "Running paper_trade signals": "跑 paper_trade 信号 (v19.10 主)",
    "Forward OOS monitoring": "Forward OOS 监控",
    "Shadow v19.4 paper_trade": "影子 v19.4 paper_trade",
    "Multi-Agent Debate": "Multi-Agent 多方讨论",
    "Shadow v19.6 paper_trade": "影子 v19.6 (v19.10 fallback)",
    "Rendering dashboard report": "重新生成 dashboard",
}


def _find_latest_log() -> Path | None:
    """搜 /tmp/daily_check_YYYYMMDD.log + 兜底 candidates, 取 mtime 最新."""
    candidates: list[Path] = []
    tmp = Path("/tmp")
    if tmp.exists():
        try:
            for p in tmp.glob("daily_check_*.log"):
                stem = p.stem  # daily_check_20260525
                date_part = stem.replace("daily_check_", "")
                if date_part.isdigit() and len(date_part) == 8:
                    candidates.append(p)
        except OSError:
            pass

    for p in LOG_CANDIDATES_STATIC:
        if p.exists():
            candidates.append(p)

    candidates = [c for c in candidates if c.exists() and c.stat().st_size > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _classify_step_lines(lines: list[str]) -> str:
    """根据 step 范围内的关键字推断状态.

    Verdict 行 (data_completeness_check.py 输出) 优先 — 它是子脚本的最终判决:
      `Verdict: ⚠️  WARNING (exit=1)` → warn (不论上文有多少 ✗ MISSING)
      `Verdict: ❌ CRITICAL (exit=99)` → fail
    """
    has_fail = False
    has_warn = False
    has_success = False
    verdict_override: str | None = None
    # sanity check lenient mode: 仅当 production blocked 才 fail, 否则 warn (允许 non-critical fail)
    has_lenient_mode = False
    has_production_blocked = False
    for line in lines:
        if "Verdict:" in line:
            if "CRITICAL" in line or "exit=99" in line:
                verdict_override = "fail"
            elif "WARNING" in line or "exit=1" in line:
                verdict_override = "warn"
            elif "OK" in line or "exit=0" in line:
                verdict_override = "ok"
        if "lenient mode" in line:
            has_lenient_mode = True
        if "production blocked" in line or "DATA SANITY CHECK FAILED" in line:
            has_production_blocked = True
        if ERROR_KEYWORDS.search(line):
            has_fail = True
        if WARN_KEYWORDS.search(line):
            has_warn = True
        if SUCCESS_KEYWORDS.search(line):
            has_success = True
    if verdict_override is not None:
        return verdict_override
    # Lenient mode: critical 全 OK 但 non-critical fail → warn (不是 fail)
    if has_lenient_mode and not has_production_blocked:
        return "warn"
    if has_fail:
        return "fail"
    if has_warn:
        return "warn"
    if has_success:
        return "ok"
    return "unknown"


def _slice_last_run(lines: list[str]) -> list[str]:
    """若日志含多次 run (tee -a), 只保留最后一次 run 的 lines.

    daily_check.sh 以 `=== ... Daily check starting ===` 分隔每次 run.
    """
    start_indices = [
        i for i, line in enumerate(lines) if "Daily check starting" in line
    ]
    if not start_indices:
        return lines
    return lines[start_indices[-1]:]


def _parse_log(log_path: Path) -> dict[str, Any]:
    """解析 daily_check.log: 每 step 状态 + final exit + tail."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    all_lines = text.splitlines()
    # 仅看最后一次 run, 避免 tee -a 累积多次 run 时把 step 重复算
    lines = _slice_last_run(all_lines)

    last_run = datetime.fromtimestamp(log_path.stat().st_mtime)

    # 分段: 每遇到 *顶层* [N/M] 开新 step, 收集到下一个顶层 [N/M] 之前.
    # daily_check.sh 调子脚本(fetch_baidu_kline.py / paper_trade_today.py)
    # 也会输出 [1/4]...[4/4] 嵌套, 必须用 OUTER_STEP_LABEL_PREFIXES 过滤.
    steps: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_step: dict[str, Any] | None = None

    for line in lines:
        m = STEP_PATTERN.match(line)
        is_outer_step = False
        if m:
            label = m.group(3).strip()
            is_outer_step = any(
                label.startswith(prefix) for prefix in OUTER_STEP_LABEL_PREFIXES
            )
        if is_outer_step and m is not None:
            if current_step is not None:
                current_step["status"] = _classify_step_lines(current_lines)
                steps.append(current_step)
            current_step = {
                "id": m.group(1),
                "label": m.group(3).strip(),
                "status": "unknown",
            }
            current_lines = [line]
        elif current_step is not None:
            current_lines.append(line)

    if current_step is not None:
        current_step["status"] = _classify_step_lines(current_lines)
        steps.append(current_step)

    # final exit code — 找最后一个 exit= 出现
    exit_code: int | None = None
    for line in reversed(lines):
        m = EXIT_PATTERN.search(line)
        if m:
            exit_code = int(m.group(1) or m.group(2))
            break

    # Pre-fill 全 8 个 expected step,缺的标 pending,最后一个无终态标 running
    EXPECTED_STEPS = [
        ("1", "Fetching today's kline"),
        ("1.5", "Data sanity check"),
        ("1.6", "Data completeness check"),
        ("1.7", "Margin incremental fetch"),
        ("2", "Running paper_trade signals"),
        ("3", "Forward OOS monitoring"),
        ("3.5", "Shadow v19.4 paper_trade"),
        ("3.6", "Multi-Agent Debate"),
        ("3.7", "Shadow v19.6 paper_trade"),
        ("4", "Rendering dashboard report"),
    ]
    is_finished = exit_code is not None
    full_steps: list[dict[str, Any]] = []
    for i, (sid, expected_label) in enumerate(EXPECTED_STEPS):
        found = None
        for s in steps:
            if s["label"].startswith(expected_label):
                found = s
                break
        if found:
            status = found["status"]
            # 最后一个找到的 step 如果整 run 未结束 → running
            if not is_finished and steps and found is steps[-1] and status == "unknown":
                status = "running"
            # daily_check.sh 用 `set -e -o pipefail`: 整 run exit=0 ⇒ 每个跑过的 step 都成功了.
            # paper_trade / forward_oos / shadow 等不打 OK/done 关键字, 默认会 unknown → 修正为 ok.
            if is_finished and exit_code == 0 and status == "unknown":
                status = "ok"
            full_steps.append({"id": sid, "label": expected_label, "status": status})
        else:
            # 未达到 → pending(如整 run 已结束没跑到 → skipped)
            status = "skipped" if is_finished else "pending"
            full_steps.append({"id": sid, "label": expected_label, "status": status})
    steps = full_steps

    # 提取关键告警
    alerts: list[str] = []
    for line in lines:
        if "🚨" in line or "SANITY CHECK FAILED" in line:
            alerts.append(line.strip())
        elif "Alert level code" in line:
            alerts.append(line.strip())
    seen: set[str] = set()
    alerts_unique = []
    for a in alerts:
        if a not in seen:
            seen.add(a)
            alerts_unique.append(a)

    return {
        "log_path": str(log_path),
        "log_size_kb": log_path.stat().st_size / 1024,
        "last_run": last_run,
        "steps": steps,
        "tail_lines": lines[-12:] if len(lines) >= 12 else lines,
        "all_lines": lines,  # 全量 (scrollable container 用)
        "exit_code": exit_code,
        "alerts": alerts_unique[:6],
    }


def _parse_plist_schedule() -> tuple[int, int]:
    """从 plist 文件 parse StartCalendarInterval Hour/Minute, fallback (16, 30)."""
    if not PLIST_PATH.exists():
        return (16, 30)
    try:
        text = PLIST_PATH.read_text(encoding="utf-8")
    except OSError:
        return (16, 30)
    hour_m = re.search(
        r"<key>Hour</key>\s*<integer>\s*(\d+)\s*</integer>", text
    )
    min_m = re.search(
        r"<key>Minute</key>\s*<integer>\s*(\d+)\s*</integer>", text
    )
    hour = int(hour_m.group(1)) if hour_m else 16
    minute = int(min_m.group(1)) if min_m else 30
    return (hour, minute)


def _next_trigger(hour: int, minute: int) -> datetime:
    """计算下次触发时刻 (本地时间). 兼容旧 caller, 不再使用."""
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    return target


def _next_cn_trigger() -> tuple[str, str]:
    """计算下次北京时间 16:30 工作日触发时刻.

    Wrapper (daily_check_cn_wrapper.sh) 用 TZ=Asia/Shanghai gate,
    实际只在北京 16:25-16:59 周一-周五 exec daily_check.sh.
    返回 (北京 ISO 字符串, 距今 H 小时 M 分钟字符串).
    """
    try:
        from zoneinfo import ZoneInfo
        cn_tz = ZoneInfo("Asia/Shanghai")
    except Exception:
        return ("(timezone lib missing)", "?")
    now_cn = datetime.now(cn_tz)
    target = now_cn.replace(hour=16, minute=30, second=0, microsecond=0)
    if now_cn >= target:
        target = target + timedelta(days=1)
    while target.weekday() >= 5:
        target = target + timedelta(days=1)
    delta = target - now_cn
    h = int(delta.total_seconds() // 3600)
    m = int((delta.total_seconds() % 3600) // 60)
    weekday_zh = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][target.weekday()]
    return (
        target.strftime(f"%Y-%m-%d %H:%M {weekday_zh}") + " 北京",
        f"{h} 小时 {m} 分钟后",
    )


def _check_data_live_status() -> dict:
    """快速检查 baidu_kline 当前数据健康 (vs log 时刻).

    Returns {'status': 'clean'|'dirty'|'unknown', 'msg': str, 'last_date': str|None}.
    """
    kline = ROOT / "data_cache" / "baidu_kline.parquet"
    if not kline.exists():
        return {"status": "unknown", "msg": "baidu_kline.parquet 不存在", "last_date": None}
    try:
        import pandas as pd  # local import 避免顶层 hard dep
        df = pd.read_parquet(kline, columns=["code", "close", "date"])
        neg = int((df["close"] <= 0).sum())
        low = int(((df["close"] > 0) & (df["close"] < 0.5)).sum())
        codes = int(df["code"].nunique())
        last = df["date"].max()
        last_str = last.date().isoformat() if hasattr(last, "date") else str(last)[:10]
        if neg == 0 and low == 0:
            return {
                "status": "clean",
                "msg": f"baidu_kline.parquet 当前 clean (0 neg · 0 low · {codes} codes · last={last_str})",
                "last_date": last_str,
            }
        return {
            "status": "dirty",
            "msg": f"baidu_kline.parquet 当前仍有 corruption: {neg} 行负价 + {low} 行 <0.5 · last={last_str}",
            "last_date": last_str,
        }
    except Exception as exc:
        return {"status": "unknown", "msg": f"读 baidu_kline 失败: {exc}", "last_date": None}


def _launchctl_last_exit() -> str | None:
    """launchctl print 拿 last exit code. 失败返回 None."""
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/501/{LAUNCHD_LABEL}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    m = re.search(r"last exit code\s*=\s*(-?\d+)", proc.stdout)
    if not m:
        return None
    return m.group(1)


def build_daily_check_status_section() -> str:
    """Section 12: Daily Check Status panel."""
    log_path = _find_latest_log()

    hour, minute = _parse_plist_schedule()
    next_trig = _next_trigger(hour, minute)
    next_trig_str = next_trig.strftime("%Y-%m-%d %H:%M")
    cn_trig_str, cn_trig_eta = _next_cn_trigger()
    launchctl_exit = _launchctl_last_exit()
    launchctl_exit_str = (
        f"last launchd exit={launchctl_exit}"
        if launchctl_exit is not None
        else "launchctl 不可查 (gui/501 未注册或 launchctl 不可用)"
    )

    if log_path is None:
        candidates_html = "<br>".join(
            f"<code>{html.escape(str(p))}</code>" for p in LOG_CANDIDATES_STATIC
        )
        return (
            '<div class="placeholder-content">'
            "未找到 daily_check log. 候选位置 (按优先级):<br>"
            f"{candidates_html}<br>"
            "及 <code>/tmp/daily_check_YYYYMMDD.log</code> (daily_check.sh 主写).<br><br>"
            f"<strong>下次 launchd 触发</strong>: {cn_trig_str} ({cn_trig_eta}) — "
            f"{html.escape(launchctl_exit_str)}"
            "</div>"
        )

    info = _parse_log(log_path)

    # 整体状态颜色
    if info["exit_code"] is None:
        overall_color = COLOR_UNKNOWN
        overall_text = "exit=? (未结束或日志不完整)"
    elif info["exit_code"] == 0:
        overall_color = COLOR_OK
        overall_text = "exit=0 (success)"
    elif info["exit_code"] in (1, 2):
        overall_color = COLOR_WARN
        overall_text = f"exit={info['exit_code']} (yellow/orange alert)"
    else:
        overall_color = COLOR_FAIL
        overall_text = f"exit={info['exit_code']} (red/black alert 或 fail)"

    # Live data freshness banner — log 是历史快照, 真实数据可能已修复
    live = _check_data_live_status()
    if live["status"] == "clean":
        banner_bg, banner_border, banner_icon = "rgba(22,163,74,0.10)", "#16a34a", "✓"
        banner_extra = ""
        # 如果 log 显示 fail 但 data 现在 clean,加修复提示
        if info["steps"]:
            sanity_step = next((s for s in info["steps"] if s["id"] == "1.5"), None)
            if sanity_step and sanity_step["status"] == "fail":
                banner_extra = (
                    " &middot; <strong>log 中 sanity fail 是历史</strong> — "
                    "明日 16:30 重跑后会显示绿"
                )
    elif live["status"] == "dirty":
        banner_bg, banner_border, banner_icon = "rgba(220,38,38,0.10)", "#dc2626", "⚠"
        banner_extra = " &middot; 需 restore / re-fetch"
    else:
        banner_bg, banner_border, banner_icon = "rgba(148,163,184,0.10)", "#6b7280", "?"
        banner_extra = ""
    live_banner = (
        f"<div style='background:{banner_bg}; border-left:3px solid {banner_border}; "
        f"padding:8px 12px; margin-bottom:12px; font-size:12px; border-radius:0 4px 4px 0;'>"
        f"<strong>{banner_icon} 数据实时状态</strong> "
        f"<span style='color:var(--muted, #6b7280);'>(log 上方是历史快照, 下面是当前数据)</span><br>"
        f"{html.escape(live['msg'])}{banner_extra}"
        f"</div>"
    )

    # 顶部 summary block
    log_path_esc = html.escape(info["log_path"])
    last_run_str = info["last_run"].strftime("%Y-%m-%d %H:%M:%S")
    summary_html = live_banner + f"""
<div style='display:flex; gap:24px; flex-wrap:wrap; margin-bottom:12px;'>
  <div style='flex:1; min-width:280px;'>
    <div style='font-size:12px; color:var(--muted, #6b7280);'>Last run</div>
    <div style='font-size:14px; font-weight:600;'>{last_run_str}</div>
  </div>
  <div style='flex:1; min-width:200px;'>
    <div style='font-size:12px; color:var(--muted, #6b7280);'>Overall</div>
    <div style='font-size:14px; font-weight:600; color:{overall_color};'>{overall_text}</div>
  </div>
  <div style='flex:1; min-width:280px;'>
    <div style='font-size:12px; color:var(--muted, #6b7280);'>下次自动触发 (北京时间)</div>
    <div style='font-size:14px; font-weight:600;'>{cn_trig_str}</div>
    <div style='font-size:11px; color:var(--muted, #6b7280);'>{cn_trig_eta}</div>
  </div>
</div>
<div style='font-size:12px; color:var(--muted, #6b7280); margin-bottom:8px;'>
Log: <code>{log_path_esc}</code> ({info['log_size_kb']:.1f} KB) &middot; {html.escape(launchctl_exit_str)}
</div>
"""

    # steps table
    STATUS_ICONS = {
        "ok": "✅", "warn": "⚠️", "fail": "❌", "unknown": "❓",
        "pending": "⏳", "running": "🔄", "skipped": "⏭️",
    }
    STATUS_LABEL_ZH = {
        "ok": "完成", "warn": "警告", "fail": "失败", "unknown": "未知",
        "pending": "未跑", "running": "运行中", "skipped": "跳过",
    }
    if info["steps"]:
        step_rows_html = ""
        for s in info["steps"]:
            color = STATUS_COLORS.get(s["status"], COLOR_UNKNOWN)
            icon = STATUS_ICONS.get(s["status"], "❓")
            zh_label = STATUS_LABEL_ZH.get(s["status"], s["status"])
            label_zh = LABEL_ZH.get(s["label"], s["label"])
            label_esc = html.escape(label_zh)
            step_rows_html += (
                "<tr>"
                f"<td><code>{html.escape(s['id'])}</code></td>"
                f"<td>{label_esc}</td>"
                f"<td style='color:{color}; font-weight:600;'>{icon} {s['status']} · {zh_label}</td>"
                "</tr>"
            )
        steps_html = f"""
<h3 style='font-size:13px; margin:14px 0 6px 0; color:var(--muted, #6b7280);'>
  步骤 ({len(info['steps'])})
</h3>
<table class='data'>
<thead><tr><th>step</th><th>label</th><th>status</th></tr></thead>
<tbody>{step_rows_html}</tbody>
</table>
"""
    else:
        steps_html = (
            "<div class='placeholder-content' style='padding:12px;'>"
            "日志中未找到 <code>[N/M]</code> step 行 — 可能 daily_check.sh 还没跑到主流程."
            "</div>"
        )

    # alerts — 每个告警可单独 ack (localStorage 持久化, key = 'cf_acked_alerts_v1')
    if info["alerts"]:
        alert_items_html = []
        for a in info["alerts"]:
            # ack id = simple hash of text (stable across renders)
            import hashlib
            ack_id = hashlib.md5(a.encode("utf-8")).hexdigest()[:12]
            alert_items_html.append(
                f"<li class='alert-item' data-ack-id='{ack_id}' "
                f"style='color:{COLOR_FAIL}; padding:4px 0; "
                "display:flex; align-items:center; gap:10px;'>"
                f"<span class='alert-text' style='flex:1;'>{html.escape(a)}</span>"
                f"<button class='ack-btn' data-ack-target='{ack_id}' type='button' "
                "style='padding:3px 10px; font-size:11px; cursor:pointer; "
                "background:rgba(220,38,38,0.10); color:#dc2626; "
                "border:1px solid rgba(220,38,38,0.30); border-radius:4px;'>"
                "✓ 确认</button></li>"
            )
        alerts_html = f"""
<h3 style='font-size:13px; margin:14px 0 6px 0; color:{COLOR_FAIL};'>
  告警 ({len(info['alerts'])}) <span style='font-weight:400; font-size:11px; color:var(--muted, #6b7280);'>· 点 ✓ 确认后本浏览器不再红色提示</span>
</h3>
<ul id='daily-check-alerts' style='margin:0; padding-left:18px; font-size:12px; list-style:none;'>{"".join(alert_items_html)}</ul>
"""
    else:
        alerts_html = ""

    # 全量日志 (scrollable container)
    full_lines = info.get("all_lines", info["tail_lines"])
    full_text = "\n".join(full_lines)
    full_esc = html.escape(full_text)
    tail_html = f"""
<h3 style='font-size:13px; margin:14px 0 6px 0; color:var(--muted, #6b7280);'>
  完整日志 ({len(full_lines)} 行 · 滚动查看)
</h3>
<pre style='background:#1e293b; color:#e5e7eb; padding:12px; border-radius:6px; \
max-height:400px; overflow-y:auto; overflow-x:auto; font-size:11px; line-height:1.4; margin:0; \
font-family:"SF Mono", Menlo, Monaco, Consolas, monospace;'>{full_esc}</pre>
"""

    return summary_html + steps_html + alerts_html + tail_html
