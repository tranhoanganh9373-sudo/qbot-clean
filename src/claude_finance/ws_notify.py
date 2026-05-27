"""Thin client for ws_broadcaster.py — fail-tolerant.

Producers (paper_trade_today / poll_5m_picks / forward_oos_monitor) 调用:

    from claude_finance.ws_notify import ws_notify
    ws_notify("picks_changed", {"as_of_date": "...", "count": 8})

设计原则:
  * Failure 永不抛 — broadcast 失败时 producer 仍然完成主业务
  * Timeout 0.5s — 即使 ws_broadcaster 卡死也不拖慢 producer
  * 零外部依赖 (stdlib urllib only)
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8003/broadcast"
DEFAULT_TIMEOUT_SEC = 0.5


def ws_notify(
    type_: str,
    data: dict,
    url: str = DEFAULT_URL,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    verbose: bool = False,
) -> bool:
    """Send broadcast to ws_broadcaster IPC endpoint.

    Returns True on success, False on any failure (never raises).
    verbose=True 时把失败原因 print 到 stderr (调试用).
    """
    try:
        payload = json.dumps(
            {"type": type_, "data": data}, ensure_ascii=False
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            ConnectionRefusedError, OSError) as e:
        if verbose:
            import sys
            print(f"[ws_notify] {type_} failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
        return False
