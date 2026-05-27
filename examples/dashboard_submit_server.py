"""Dashboard Trade Entry submit server — append trade to trades.jsonl → re-render dashboard.

用法 (在 claude_finance 根目录):
    python examples/dashboard_submit_server.py
    # 或自定义端口:
    python examples/dashboard_submit_server.py 5557

Endpoints:
    GET  /health  → {"ok": true}                                    (liveness probe)
    POST /submit  body = trade JSON object (新 schema):
                    {"date": "2026-05-26", "sym": "SH600547",
                     "action": "BUY" | "SELL", "price": 29.95,
                     "shares": 200, "note": "open"}
                  → trades_log.append_trade() to data_cache/trades.jsonl
                  → spawn `.venv/bin/python dashboard/render_report.py`
                  → {"ok": bool, "trade_id": str, "elapsed_sec": float,
                     "elapsed_render_sec": float,
                     "stdout_tail": [...], "stderr_tail": [...]}

为什么用 stdlib http.server 而非 Flask:
  - 无新依赖, 单文件可移植
  - 100% 受用户本地 .venv 限制 (不暴露公网)
"""
from __future__ import annotations

# 所有时间显示用北京时间 (A 股交易时区), 不参考 Mac 本机时区.
import os
os.environ["TZ"] = "Asia/Shanghai"
import time
time.tzset()

import gzip
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 让 import claude_finance 走得通 — src/ 在 ROOT 下
sys.path.insert(0, str(ROOT / "src"))
from claude_finance.trades_log import TradeValidationError, append_trade

RENDER_REPORT = ROOT / "dashboard" / "render_report.py"
VENV_PY = ROOT / ".venv" / "bin" / "python"
RENDER_TIMEOUT_SEC = 60

DEFAULT_PORT = 5557
LATEST_HTML = ROOT / "reports" / "latest.html"
AUTO_REFRESH_SEC = 30
HOT_RENDER_DATA_FILES = [
    ROOT / "data_cache" / "picks_today.json",
    ROOT / "data_cache" / "portfolio_state.json",
    ROOT / "data_cache" / "trades.jsonl",
    ROOT / "data_cache" / "paper_trade_log.csv",
    ROOT / "data_cache" / "live_5m_picks.json",  # 5m live bar 更新
    ROOT / "data_cache" / "baidu_kline.parquet",  # daily_check step 1 完成时 mtime 变
    Path("/tmp/daily_check_stdout.log"),
]
# Glob pattern: daily_check_YYYYMMDD.log (日期变化, 用 glob 兼容历史/未来日期)
import glob as _glob
HOT_RENDER_GLOB_PATTERNS = [
    "/tmp/daily_check_*.log",
]
# 强制 render fallback: 即使 mtime 比对没新, 超过此秒数也 force re-render
# (用户调时钟回拨 / NTP sync / 长期 idle 等 mtime 不可信场景)
MAX_RENDER_STALENESS_SEC = 300  # 5 分钟
# HTML mtime > 现在系统时间 + 此阈值 → 视为时钟错乱 → force render
CLOCK_FUTURE_TOLERANCE_SEC = 60
import threading as _threading
_RENDER_LOCK = _threading.Lock()


def _maybe_hot_render() -> bool:
    """若任何数据 file mtime 新过 latest.html, spawn render_report.py.
    返回 did_render. 同时只跑一个 render, 已在跑时直接 serve 当前 latest.

    Fallback triggers (防 mtime 不可信场景):
      1) 数据 file mtime > HTML mtime + 0.5s  (正常 case)
      2) HTML age > MAX_RENDER_STALENESS_SEC  (long idle, 强制刷新)
      3) HTML mtime > now + CLOCK_FUTURE_TOLERANCE_SEC  (时钟回拨, future mtime)
    """
    if not LATEST_HTML.exists():
        return False
    now = time.time()
    latest_mtime = LATEST_HTML.stat().st_mtime
    newest_data_mtime = 0.0
    for p in HOT_RENDER_DATA_FILES:
        if p.exists():
            newest_data_mtime = max(newest_data_mtime, p.stat().st_mtime)
    # glob patterns (日期不固定 log)
    for pattern in HOT_RENDER_GLOB_PATTERNS:
        for fp in _glob.glob(pattern):
            try:
                newest_data_mtime = max(newest_data_mtime, os.path.getmtime(fp))
            except OSError:
                pass

    needs_render = False
    # 1) 数据更新 (normal mtime comparison)
    if newest_data_mtime > latest_mtime + 0.5:
        needs_render = True
    # 2) HTML 太老 (long idle / data 没动但要刷新 "更新 X 秒前" 等 badge)
    elif (now - latest_mtime) > MAX_RENDER_STALENESS_SEC:
        needs_render = True
    # 3) HTML mtime 在未来 (时钟回拨, mtime 比对永远 false → 卡死 hot-render)
    elif latest_mtime > now + CLOCK_FUTURE_TOLERANCE_SEC:
        needs_render = True

    if not needs_render:
        return False
    if not _RENDER_LOCK.acquire(blocking=False):
        return False
    try:
        if RENDER_REPORT.exists() and VENV_PY.exists():
            subprocess.run(
                [str(VENV_PY), str(RENDER_REPORT)],
                capture_output=True, text=True,
                timeout=30, cwd=str(ROOT),
            )
            return True
    except subprocess.TimeoutExpired:
        return False
    finally:
        _RENDER_LOCK.release()
    return False


def _inject_auto_refresh(html: str, interval_sec: int) -> str:
    """注入自动刷新逻辑 (按当前 active tab 不同策略).

    - active tab = 'realtime' (5m K 线 + 5m 行情): **整页 60s reload**
      用户预期实时数据高频更新, 整页 reload 拿 audit/live panel 最新.
    - active tab = 其他 (today/market/picks/models/etc): **不主动 reload**
      仅 daily 16:30-17:00 北京时间窗口内 检测一次, 拿 daily_check 跑完的新数据.
      其他时间 user 可手动 ⌘+R.
    - 整页 reload 前 save scrollY, reload 后 restore (sessionStorage).
    """
    script = (
        '<meta name="claude-finance-server" content="dashboard-submit-server">'
        '<script>'
        '(function(){'
        ' if (history && "scrollRestoration" in history) {'
        '   history.scrollRestoration = "manual";'
        ' }'
        ' function restoreScroll() {'
        '   var SY = sessionStorage.getItem("cfDashScrollY");'
        '   if (SY === null) return;'
        '   var y = parseInt(SY, 10) || 0;'
        '   window.scrollTo(0, y);'
        '   var maxY = document.documentElement.scrollHeight - window.innerHeight;'
        '   if (window.scrollY < y - 5 && maxY > 0) {'
        '     setTimeout(function(){ window.scrollTo(0, y); }, 50);'
        '     setTimeout(function(){ window.scrollTo(0, y); }, 200);'
        '     setTimeout(function(){ window.scrollTo(0, y); }, 500);'
        '     setTimeout(function(){ window.scrollTo(0, y); }, 1000);'
        '     setTimeout(function(){ window.scrollTo(0, y); }, 2000);'
        '   }'
        ' }'
        ' if (document.readyState === "complete") { restoreScroll(); }'
        ' else { window.addEventListener("load", restoreScroll); }'
        ' function getActiveTab() {'
        '   var p = document.querySelector(".tab-pane.active");'
        '   return p ? p.getAttribute("data-tab") : null;'
        ' }'
        # 可见 status indicator: 当前 tab 的刷新策略
        ' function injectRefreshStatus() {'
        '   if (document.getElementById("cf-refresh-status")) return;'
        '   var bar = document.createElement("div");'
        '   bar.id = "cf-refresh-status";'
        '   bar.style.cssText = "position:fixed; bottom:8px; right:14px; '
        'background:rgba(15,23,42,0.85); color:#cbd5e1; padding:4px 10px; '
        'border-radius:10px; font-size:11px; z-index:200; '
        'border:1px solid rgba(148,163,184,0.2); font-family:Inter,sans-serif;";'
        '   document.body.appendChild(bar);'
        '   updateRefreshStatus();'
        ' }'
        ' function updateRefreshStatus() {'
        '   var bar = document.getElementById("cf-refresh-status");'
        '   if (!bar) return;'
        '   var active = getActiveTab();'
        '   if (active === "realtime") {'
        '     bar.innerHTML = "🔄 实时 5m · <strong style=\\"color:#16a34a;\\">60s 自动刷新</strong>";'
        '   } else {'
        '     bar.innerHTML = "⏸ <span style=\\"color:#6b7280;\\">不自动刷新</span> · 仅北京 16:30 daily";'
        '   }'
        ' }'
        ' if (document.readyState === "complete") { injectRefreshStatus(); }'
        ' else { window.addEventListener("load", injectRefreshStatus); }'
        # tab 切换时立即更新 status
        ' document.addEventListener("click", function(e){'
        '   if (e.target && e.target.classList && e.target.classList.contains("tab-btn")) {'
        '     setTimeout(updateRefreshStatus, 50);'
        '   }'
        ' });'
        ' function doReload() {'
        '   sessionStorage.setItem("cfDashScrollY", String(window.scrollY));'
        '   if (location.hash) {'
        '     history.replaceState(null, "", location.pathname + location.search);'
        '   }'
        '   location.reload();'
        ' }'
        # 每 60s tick: realtime tab 整页 reload, 其他 tab 检测北京 16:30-17:00 daily 触发
        ' var dailyReloadedToday = false;'
        ' setInterval(function(){'
        '   var active = getActiveTab();'
        '   if (active === "realtime") {'
        '     doReload();'
        '     return;'
        '   }'
        # 其他 tab: 检测北京时间 16:30-17:00 daily 触发 reload 一次
        '   try {'
        '     var nowCN = new Date(new Date().toLocaleString("en-US", {timeZone: "Asia/Shanghai"}));'
        '     var h = nowCN.getHours();'
        '     var m = nowCN.getMinutes();'
        '     var inWindow = (h === 16 && m >= 30) || (h === 17 && m === 0);'
        '     if (inWindow && !dailyReloadedToday) {'
        '       dailyReloadedToday = true;'
        '       doReload();'
        '     }'
        '     if (h !== 16 && h !== 17) {'
        '       dailyReloadedToday = false;'  # reset for next day
        '     }'
        '   } catch (e) {}'
        ' }, 60000);'
        '})();'
        '</script>'
    )
    if "<head>" in html:
        return html.replace("<head>", f"<head>{script}", 1)
    return script + html


class SubmitHandler(BaseHTTPRequestHandler):
    """单线程实现, ThreadingHTTPServer 提供并发."""

    server_version = "ClaudeFinanceSubmit/1.0"

    def _cors_headers(self) -> None:
        # dashboard 是 file:// 协议, browser 视为 'null' origin — 必须 * 才能通过
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "3600")

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # CORS preflight
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _serve_dashboard(self) -> None:
        """GET / → 返回最新 dashboard HTML + auto-refresh meta tag."""
        _maybe_hot_render()  # lazy hot-render 如果数据 file 新过 latest.html
        if not LATEST_HTML.exists():
            self.send_response(503)
            self._cors_headers()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                ("Dashboard 未生成 — 跑 .venv/bin/python dashboard/render_report.py "
                 "后再访问.").encode("utf-8")
            )
            return
        try:
            html = LATEST_HTML.read_text(encoding="utf-8")
        except OSError as e:
            self.send_response(500)
            self._cors_headers()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"读取 latest.html 失败: {e}".encode("utf-8"))
            return
        html = _inject_auto_refresh(html, AUTO_REFRESH_SEC)
        body = html.encode("utf-8")
        self._write_compressed(body, "text/html; charset=utf-8", no_cache=True)

    def _serve_live_5m_panel(self):
        """Legacy endpoint (兼容旧 dashboard): 等价于 /api/partial/live_5m.html."""
        self._serve_partial("live_5m")

    # 白名单: panel_id → (module, fn)
    _PARTIAL_PANELS = {
        "live_5m": ("dashboard.charts.live_5m_picks", "build_live_5m_picks_section"),
        "picks": ("dashboard.charts.recommended_picks", "build_recommended_picks_section"),
        "alert": ("dashboard.charts.alert_center", "build_alert_center_section"),
        "freshness": ("dashboard.charts.data_freshness", "build_data_freshness_section"),
        "sandbox": ("dashboard.charts.factor_sandbox", "build_factor_sandbox_section"),
        "risk_events": ("dashboard.charts.risk_events", "build_risk_events_section"),
        "phase_b_history": ("dashboard.charts.phase_b_history", "build_phase_b_history_section"),
        "leak_scan": ("dashboard.charts.leak_scan", "build_leak_scan_section"),
        "debate_veto": ("dashboard.charts.debate_veto_panel", "build_debate_veto_section"),
    }

    def _serve_partial(self, panel_id: str):
        """通用 partial panel API.
        WS client 收到 typed event 后 fetch /api/partial/<id>.html, replace section innerHTML.
        白名单 + 同步 import + 任何异常返 placeholder, 永不抛 500.
        """
        entry = self._PARTIAL_PANELS.get(panel_id)
        if entry is None:
            self._json_response(404, {"ok": False, "error": f"unknown panel {panel_id!r}"})
            return
        module_name, fn_name = entry
        try:
            sys.path.insert(0, str(ROOT))
            mod = __import__(module_name, fromlist=[fn_name])
            builder = getattr(mod, fn_name)
            html_body = builder()
        except Exception as exc:
            html_body = (
                '<div class="placeholder-content" style="padding:24px 16px;">'
                f'partial panel {panel_id} 渲染失败: <code>{type(exc).__name__}: {exc}</code>'
                '</div>'
            )
        body = html_body.encode("utf-8")
        self._write_compressed(body, "text/html; charset=utf-8", no_cache=True)

    def _write_compressed(self, body: bytes, content_type: str, no_cache: bool = False) -> None:
        """Write response with gzip if client supports (1.74MB → ~400KB for base64 HTML)."""
        accept = self.headers.get("Accept-Encoding", "")
        use_gzip = "gzip" in accept and len(body) > 1024  # skip tiny responses
        if use_gzip:
            body = gzip.compress(body, compresslevel=6)
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", content_type)
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        if no_cache:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("/health", "/healthz"):
            self._json_response(200, {"ok": True, "service": "dashboard-submit"})
        elif path == "/api/live_5m_panel.html":
            self._serve_live_5m_panel()
        elif path.startswith("/api/partial/") and path.endswith(".html"):
            # /api/partial/<id>.html → look up whitelist
            panel_id = path[len("/api/partial/"):-len(".html")]
            self._serve_partial(panel_id)
        elif path in ("", "/", "/dashboard"):
            self._serve_dashboard()
        else:
            self._json_response(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/submit":
            self._json_response(404, {"ok": False, "error": "POST /submit only"})
            return
        # 1. read + parse body
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                self._json_response(400, {"ok": False, "error": "invalid body length"})
                return
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                self._json_response(400, {"ok": False, "error": "payload must be JSON object"})
                return
        except json.JSONDecodeError as exc:
            self._json_response(400, {"ok": False, "error": f"invalid JSON: {exc}"})
            return
        except Exception as exc:
            self._json_response(400, {"ok": False, "error": f"read body failed: {exc}"})
            return

        # 2. validate + append trade to trades.jsonl
        try:
            normalized = append_trade(payload)
        except TradeValidationError as exc:
            self._json_response(400, {"ok": False, "error": f"invalid trade: {exc}"})
            return
        except Exception as exc:
            self._json_response(500, {"ok": False, "error": f"append failed: {exc}"})
            return

        # 3. re-render dashboard HTML to reflect new position state
        elapsed_render = 0.0
        stdout_tail: list[str] = []
        stderr_tail: list[str] = []
        ok = True
        if RENDER_REPORT.exists() and VENV_PY.exists():
            t1 = time.time()
            try:
                render_proc = subprocess.run(
                    [str(VENV_PY), str(RENDER_REPORT)],
                    capture_output=True, text=True,
                    timeout=RENDER_TIMEOUT_SEC, cwd=str(ROOT),
                )
                elapsed_render = time.time() - t1
                stdout_tail = (render_proc.stdout or "").splitlines()[-8:]
                stderr_tail = (render_proc.stderr or "").splitlines()[-8:]
                if render_proc.returncode != 0:
                    ok = False
            except subprocess.TimeoutExpired:
                stderr_tail = [f"render_report.py timed out (>{RENDER_TIMEOUT_SEC}s)"]
                ok = False

        self._json_response(200 if ok else 500, {
            "ok": ok,
            "trade_id": normalized["id"],
            "trade": normalized,
            "elapsed_sec": round(elapsed_render, 2),
            "elapsed_render_sec": round(elapsed_render, 2),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        })

    def log_message(self, fmt, *args):  # quieter than default stderr spam
        sys.stderr.write(f"[submit-server] {self.address_string()} {fmt % args}\n")


def main():
    port = DEFAULT_PORT
    if len(sys.argv) >= 2:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"ERR: port must be int, got {sys.argv[1]!r}", file=sys.stderr)
            return 2
    addr = ("127.0.0.1", port)
    try:
        httpd = ThreadingHTTPServer(addr, SubmitHandler)
    except OSError as exc:
        if "Address already in use" in str(exc):
            print(f"ERR: port {port} 已被占用 (可能 server 已经在跑)", file=sys.stderr)
            return 1
        raise
    print(f"[dashboard server] listening on http://127.0.0.1:{port}")
    print(f"  GET  /          → 实时 dashboard ({AUTO_REFRESH_SEC}s auto-refresh, lazy hot-render)")
    print("  GET  /health    → liveness")
    print("  POST /submit    → append trade + re-render dashboard")
    print(f"  render script: {RENDER_REPORT}")
    print(f"  bookmark: http://127.0.0.1:{port}/")
    print("  Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard submit server] stopped.")
    httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
