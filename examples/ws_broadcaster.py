"""WebSocket Broadcaster — 实时 dashboard 推送中继.

架构:
  Producer (paper_trade_today / poll_5m_picks / forward_oos_monitor)
    --HTTP POST /broadcast--> ws_broadcaster (localhost:8003)
       --WS push--> Browser clients (ws://localhost:8002)

设计原则 (借鉴 jin-ce-zhi-suan ConnectionManager + 改进):
  * Typed channels: {"type": "bar_5m_update"|"picks_changed"|"oos_alert"|...,
                     "data": {...}, "ts": ISO}
  * Per-client asyncio.Queue (maxsize=2000, drop-oldest on full) + 5s send timeout
  * Heartbeat: websockets 库 ping_interval=30 自动 + 客户端不响应 60s 断开
  * 零持久化: producer 把数据落 parquet/json, ws 只通知"有新数据"
  * 双 port: WS 8002 给浏览器, HTTP 8003 给 Python 内部 IPC
  * 仅 localhost 监听 (不暴露给外网, 避免无认证风险)

运行:
  python examples/ws_broadcaster.py
  # 或由 launchd 自启 com.claude_finance.ws_broadcaster

测试 broadcast:
  curl -X POST http://127.0.0.1:8003/broadcast \
       -H 'Content-Type: application/json' \
       -d '{"type":"test","data":{"msg":"hello"}}'

回滚: launchctl unload ~/Library/LaunchAgents/com.claude_finance.ws_broadcaster.plist
       前端 template.html WS client 会 reconnect 失败但不影响其它功能.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from contextlib import suppress
from datetime import datetime
from http import HTTPStatus

# 北京时间统一 (跟 dashboard_submit_server / paper_trade_today 一致)
os.environ["TZ"] = "Asia/Shanghai"
time.tzset()

import websockets
from websockets.exceptions import ConnectionClosed

WS_HOST = "127.0.0.1"
WS_PORT = 8002
IPC_HOST = "127.0.0.1"
IPC_PORT = 8003

QUEUE_MAXSIZE = 2000  # 客户端积压上限
SEND_TIMEOUT_SEC = 5.0  # 单条消息推送超时
PING_INTERVAL_SEC = 30  # 心跳间隔
PING_TIMEOUT_SEC = 60   # 心跳无响应阈值
MAX_BODY_BYTES = 64 * 1024  # /broadcast HTTP body 上限 (64 KB)

VALID_TYPES = {
    "bar_5m_update",      # poll_5m_picks 5m K bar update
    "picks_changed",      # paper_trade_today 完成
    "oos_alert",          # forward_oos_monitor alert level change
    "sandbox_refresh",    # sandbox_factors.json 更新
    "risk_event",         # risk_gates.py veto/熔断 (P0-B 用)
    "system",             # 服务自身事件 (started/heartbeat 等)
    "test",               # 调试用
}

logger = logging.getLogger("ws_broadcaster")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class ConnectionManager:
    """管理 WS 客户端集合, 每个客户端独立队列 + 发送 task."""

    def __init__(self) -> None:
        self._clients: dict = {}
        self._lock = asyncio.Lock()
        self.total_broadcasts = 0
        self.total_drops = 0

    async def register(self, ws) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        async with self._lock:
            self._clients[ws] = q
        logger.info("client connect (total=%d) remote=%s",
                    len(self._clients), ws.remote_address)
        return q

    async def unregister(self, ws) -> None:
        async with self._lock:
            self._clients.pop(ws, None)
        logger.info("client disconnect (total=%d)", len(self._clients))

    async def broadcast(self, message: dict) -> dict:
        """非阻塞 broadcast: 每个客户端队列满则 drop-oldest."""
        payload = json.dumps(message, ensure_ascii=False)
        async with self._lock:
            clients = list(self._clients.items())
        n_sent = 0
        n_drop = 0
        for ws, q in clients:
            try:
                q.put_nowait(payload)
                n_sent += 1
            except asyncio.QueueFull:
                # drop-oldest: 取走最老的一条腾位
                with suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                with suppress(asyncio.QueueFull):
                    q.put_nowait(payload)
                n_drop += 1
        self.total_broadcasts += 1
        self.total_drops += n_drop
        return {"clients": len(clients), "sent": n_sent, "dropped": n_drop}

    def client_count(self) -> int:
        return len(self._clients)


manager = ConnectionManager()


async def _client_sender(ws, q: asyncio.Queue) -> None:
    """单个客户端的发送 task: 从队列取消息推 ws, 超时则断开."""
    try:
        while True:
            payload = await q.get()
            try:
                await asyncio.wait_for(ws.send(payload), timeout=SEND_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                logger.warning("send timeout, closing client %s", ws.remote_address)
                await ws.close(code=1011, reason="send timeout")
                return
            except ConnectionClosed:
                return
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("client sender error: %s", e)


async def ws_handler(ws) -> None:
    """每个 WS 连接的主 handler."""
    q = await manager.register(ws)
    # 立刻 push 一个 system 'connected' 消息让前端知道生效
    hello = json.dumps({
        "type": "system",
        "data": {"msg": "connected", "server_ts": datetime.now().isoformat()},
    }, ensure_ascii=False)
    with suppress(asyncio.QueueFull):
        q.put_nowait(hello)
    sender_task = asyncio.create_task(_client_sender(ws, q))
    try:
        async for raw in ws:
            # 客户端上行: 仅支持简单 ping. 不强制 schema, 容错处理.
            try:
                msg = json.loads(raw) if isinstance(raw, str) else {}
                if msg.get("type") == "ping":
                    pong = json.dumps({"type": "pong", "ts": datetime.now().isoformat()},
                                       ensure_ascii=False)
                    with suppress(asyncio.QueueFull):
                        q.put_nowait(pong)
            except (json.JSONDecodeError, AttributeError):
                pass
    except ConnectionClosed:
        pass
    finally:
        sender_task.cancel()
        with suppress(asyncio.CancelledError):
            await sender_task
        await manager.unregister(ws)


# === HTTP IPC server (localhost:8003) ===

async def _read_http_request(reader: asyncio.StreamReader):
    """读取 HTTP 请求, 返回 (method, path, headers, body)."""
    line = await reader.readline()
    if not line:
        raise ValueError("empty request")
    request_line = line.decode("ascii", errors="replace").rstrip("\r\n")
    parts = request_line.split(" ", 2)
    if len(parts) < 3:
        raise ValueError(f"bad request line: {request_line!r}")
    method, path, _ = parts
    headers: dict = {}
    while True:
        h = await reader.readline()
        if h in (b"\r\n", b"\n", b""):
            break
        try:
            k, v = h.decode("ascii", errors="replace").rstrip("\r\n").split(":", 1)
            headers[k.strip().lower()] = v.strip()
        except ValueError:
            continue
    body = b""
    content_length = int(headers.get("content-length", "0"))
    if content_length > 0:
        if content_length > MAX_BODY_BYTES:
            raise ValueError(f"body too large ({content_length} > {MAX_BODY_BYTES})")
        body = await reader.readexactly(content_length)
    return method, path, headers, body


def _http_response(status: HTTPStatus, body: dict) -> bytes:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    head = (
        f"HTTP/1.1 {status.value} {status.phrase}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("ascii")
    return head + payload


async def http_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """HTTP IPC handler — accepts POST /broadcast, GET /stats."""
    try:
        method, path, _headers, body = await _read_http_request(reader)
    except (ValueError, asyncio.IncompleteReadError) as e:
        writer.write(_http_response(HTTPStatus.BAD_REQUEST, {"error": str(e)}))
        await writer.drain()
        writer.close()
        return

    if method == "GET" and path == "/stats":
        resp = {
            "clients": manager.client_count(),
            "total_broadcasts": manager.total_broadcasts,
            "total_drops": manager.total_drops,
            "ts": datetime.now().isoformat(),
        }
        writer.write(_http_response(HTTPStatus.OK, resp))
        await writer.drain()
        writer.close()
        return

    if method == "POST" and path == "/broadcast":
        try:
            msg = json.loads(body.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            writer.write(_http_response(HTTPStatus.BAD_REQUEST, {"error": f"json: {e}"}))
            await writer.drain()
            writer.close()
            return
        msg_type = msg.get("type")
        if msg_type not in VALID_TYPES:
            writer.write(_http_response(
                HTTPStatus.BAD_REQUEST,
                {"error": f"unknown type {msg_type!r}, valid={sorted(VALID_TYPES)}"},
            ))
            await writer.drain()
            writer.close()
            return
        # 注入 server timestamp (覆盖客户端可能写错的 ts)
        msg["ts"] = datetime.now().isoformat()
        result = await manager.broadcast(msg)
        writer.write(_http_response(HTTPStatus.OK, {"ok": True, **result}))
        await writer.drain()
        writer.close()
        return

    writer.write(_http_response(
        HTTPStatus.NOT_FOUND,
        {"error": f"{method} {path} not found"},
    ))
    await writer.drain()
    writer.close()


async def main() -> None:
    # WS server
    ws_server = await websockets.serve(
        ws_handler,
        WS_HOST,
        WS_PORT,
        ping_interval=PING_INTERVAL_SEC,
        ping_timeout=PING_TIMEOUT_SEC,
        max_size=1 << 20,  # 1 MB inbound msg cap
    )
    logger.info("WS listening on ws://%s:%d", WS_HOST, WS_PORT)

    # HTTP IPC server
    http_server = await asyncio.start_server(http_handler, IPC_HOST, IPC_PORT)
    logger.info("HTTP IPC listening on http://%s:%d", IPC_HOST, IPC_PORT)

    # Periodic heartbeat broadcast (让前端知道服务还活着)
    async def heartbeat_loop() -> None:
        while True:
            await asyncio.sleep(60)
            with suppress(Exception):
                await manager.broadcast({
                    "type": "system",
                    "data": {
                        "msg": "heartbeat",
                        "clients": manager.client_count(),
                        "broadcasts": manager.total_broadcasts,
                    },
                })

    hb_task = asyncio.create_task(heartbeat_loop())

    # graceful shutdown on SIGTERM / SIGINT
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    try:
        await stop.wait()
    finally:
        logger.info("shutting down...")
        hb_task.cancel()
        with suppress(asyncio.CancelledError):
            await hb_task
        ws_server.close()
        await ws_server.wait_closed()
        http_server.close()
        await http_server.wait_closed()
        logger.info("bye.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
