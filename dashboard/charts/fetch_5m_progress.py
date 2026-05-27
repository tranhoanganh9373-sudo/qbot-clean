"""Dashboard panel — 5m fetch 全量 backfill 进度 + watcher 接力状态.

数据源 (全只读):
  - data_cache/kline_5m_shards/*.parquet — 当前已落盘的 shards
  - data_cache/qlib_baidu/instruments/all_no_st.txt — 全 universe count
  - /tmp/mootdx_5m_24mo_fetch.log — fetch script log (worker progress)
  - /tmp/post_fetch_chain.log — watcher chain log (Hive rebuild + poller load)
  - ps PID (fetch process 通过 pgrep) 状态
"""
from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SHARDS_DIR = ROOT / "data_cache" / "kline_5m_shards"
UNIVERSE_PATH = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "all_no_st.txt"
FETCH_LOG = Path("/tmp/mootdx_5m_24mo_fetch.log")
CHAIN_LOG = Path("/tmp/post_fetch_chain.log")


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _count_shards() -> tuple[int, int]:
    if not SHARDS_DIR.exists():
        return 0, 0
    files = list(SHARDS_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in files)
    return len(files), total_size


def _universe_size() -> int:
    if not UNIVERSE_PATH.exists():
        return 0
    try:
        return sum(
            1 for line in UNIVERSE_PATH.read_text().splitlines()
            if line.strip() and len(line.strip().split("\t")[0]) >= 8
        )
    except Exception:
        return 0


def _fetch_pid_status() -> tuple[bool, str]:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "fetch_mootdx_5m_5y_backfill.py"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            pid = out.stdout.strip().split("\n")[0]
            etime_out = subprocess.run(
                ["ps", "-p", pid, "-o", "etime="],
                capture_output=True, text=True, timeout=3,
            )
            etime = etime_out.stdout.strip()
            return True, f"PID {pid}, 已跑 {etime}"
        return False, "未运行 (已完成或未启动)"
    except Exception as e:
        return False, f"ps query 失败: {type(e).__name__}"


def _log_tail(path: Path, n: int = 6) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def _parse_worker_progress(log_lines: list[str]) -> dict[str, str]:
    """从 log 抽 [w0]~[w7] 最新进度 — 形如 '[w0] progress 250/475 ok=247'."""
    worker_latest: dict[str, str] = {}
    for ln in log_lines:
        m = re.search(r"\[w(\d)\]\s+progress\s+(\d+/\d+)\s+ok=(\d+)", ln)
        if m:
            worker_latest[f"w{m.group(1)}"] = f"{m.group(2)} ok={m.group(3)}"
    return worker_latest


def build_fetch_5m_progress_section() -> str:
    n_shards, total_bytes = _count_shards()
    universe_n = _universe_size()
    pct = (n_shards / universe_n * 100) if universe_n > 0 else 0
    fetch_alive, fetch_status = _fetch_pid_status()

    if universe_n == 0:
        return _placeholder("universe.txt 未找到 — fetch 状态无法跟踪.")

    size_mb = total_bytes / 1024 / 1024
    est_full_mb = size_mb / max(n_shards, 1) * universe_n if n_shards > 0 else 0

    status_color = "#16a34a" if not fetch_alive else "#2563eb"
    status_label = "✓ DONE" if not fetch_alive else "🔄 RUNNING"
    banner_bg = "rgba(22,163,74,0.06)" if not fetch_alive else "rgba(37,99,235,0.06)"

    progress_bar = (
        f'<div style="background:#1f2937; border-radius:4px; height:18px; '
        f'overflow:hidden; margin:8px 0;">'
        f'<div style="background:{status_color}; height:100%; width:{pct:.1f}%; '
        f'transition: width 0.3s;"></div>'
        f"</div>"
    )

    banner = (
        f'<div style="background:{banner_bg}; border-left:3px solid {status_color}; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; '
        'font-size:12px;">'
        f'<strong>{status_label}</strong> · shards <strong>{n_shards:,}</strong> / '
        f'<strong>{universe_n:,}</strong> ({pct:.1f}%) · 已落盘 '
        f'<strong>{size_mb:.0f} MB</strong>'
        f' (估算全量 ~{est_full_mb:.0f} MB)<br>'
        f"fetch process: <code>{html.escape(fetch_status)}</code>"
        f"{progress_bar}"
        "</div>"
    )

    worker_html = ""
    fetch_lines = _log_tail(FETCH_LOG, 30)
    if fetch_lines:
        workers = _parse_worker_progress(fetch_lines)
        if workers:
            worker_rows = "".join(
                f"<tr><td><code>{html.escape(w)}</code></td>"
                f'<td style="text-align:right;">{html.escape(p)}</td></tr>'
                for w, p in sorted(workers.items())
            )
            worker_html = (
                '<h3 style="font-size:13px; margin:10px 0 4px 0; '
                'color:var(--muted, #6b7280);">8 worker 进度</h3>'
                '<table class="data" style="margin-bottom:12px;">'
                '<colgroup><col style="width:30%;"><col style="width:70%;"></colgroup>'
                '<thead><tr><th>worker</th>'
                '<th style="text-align:right;">progress</th></tr></thead>'
                f"<tbody>{worker_rows}</tbody></table>"
            )

    chain_lines = _log_tail(CHAIN_LOG, 6)
    if chain_lines:
        chain_html = (
            '<h3 style="font-size:13px; margin:10px 0 4px 0; '
            'color:var(--muted, #6b7280);">post-fetch chain (Hive rebuild + poller load)</h3>'
            '<pre style="font-size:11px; background:rgba(107,114,128,0.08); '
            'padding:8px 10px; border-radius:4px; overflow-x:auto; '
            'color:var(--fg, #cbd5e1); margin:0;">'
            + html.escape("\n".join(chain_lines))
            + "</pre>"
        )
    else:
        chain_html = (
            '<div style="font-size:11px; color:var(--muted, #6b7280);">'
            "post-fetch chain log 未创建 (watcher 待启动 / 或已 done)"
            "</div>"
        )

    footer = (
        '<div style="margin-top:10px; font-size:11px; color:var(--muted, #6b7280);">'
        "数据源 <code>data_cache/kline_5m_shards/</code> · log "
        "<code>/tmp/mootdx_5m_24mo_fetch.log</code> + <code>/tmp/post_fetch_chain.log</code> · "
        "fetch 完成后 watcher 自动: 1) <code>build_5m_hive_duckdb.py --force</code> "
        "重建 Hive, 2) <code>launchctl load</code> poller daemon."
        "</div>"
    )

    return banner + worker_html + chain_html + footer
