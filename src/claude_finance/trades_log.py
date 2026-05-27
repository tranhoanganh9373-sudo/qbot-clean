"""Trade log — append-only JSON Lines source of truth for user trades.

Replaces the xlsx-based user_input.json (覆盖式) flow with append-only trade log,
supporting multiple BUY/SELL trades per stock per day.

Storage: data_cache/trades.jsonl, one JSON object per line. Never modified in place
(only append). Lines are not sorted on disk — sort by `recorded_at` when reading.

Schema (per trade):
    id          str    uuid4 hex
    date        str    'YYYY-MM-DD'      — 交易日 (用户填)
    sym         str    'SH600547' / 'SZ300347'
    name        str    可选 — '山东黄金'
    action      str    'BUY' | 'SELL'
    price       float  > 0
    shares      int    > 0 (BUY 加仓数, SELL 减仓数)
    note        str    可选自由文本
    recorded_at str    ISO8601 UTC — append 时间戳

Positions aggregation (perpetual weighted-average cost):
    on BUY:
        avg = (shares * avg + buy_shares * buy_price) / (shares + buy_shares)
        shares += buy_shares
    on SELL:
        realized_pnl += (sell_price - avg) * min(sell_shares, shares)
        shares -= sell_shares
        # avg 不变 (moving avg)
    status:
        net_shares > 0 → 'holding'
        net_shares == 0 AND total_buy_shares > 0 → 'closed'
        net_shares < 0 → 'short' (异常, 容忍)
        no trades → 'none' (来自外部 picks 推荐时映射)
"""
from __future__ import annotations

import fcntl
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
TRADES_PATH = ROOT / "data_cache" / "trades.jsonl"

REQUIRED_FIELDS = ("date", "sym", "action", "price", "shares")
VALID_ACTIONS = ("BUY", "SELL")


class TradeValidationError(ValueError):
    """trade dict 不符 schema."""


def _validate_trade(t: dict[str, Any]) -> None:
    """Raise TradeValidationError if missing required field or bad value."""
    for k in REQUIRED_FIELDS:
        if k not in t or t[k] is None or t[k] == "":
            raise TradeValidationError(f"missing required field: {k}")
    if t["action"] not in VALID_ACTIONS:
        raise TradeValidationError(
            f"action must be one of {VALID_ACTIONS}, got {t['action']!r}"
        )
    try:
        p = float(t["price"])
        s = int(t["shares"])
    except (TypeError, ValueError) as exc:
        raise TradeValidationError(f"price/shares not numeric: {exc}") from exc
    if p <= 0:
        raise TradeValidationError(f"price must be > 0, got {p}")
    if s <= 0:
        raise TradeValidationError(f"shares must be > 0, got {s}")
    try:
        datetime.strptime(str(t["date"]), "%Y-%m-%d")
    except ValueError as exc:
        raise TradeValidationError(f"date must be YYYY-MM-DD, got {t['date']!r}") from exc
    sym = str(t["sym"])
    if not (len(sym) == 8 and sym[:2] in ("SH", "SZ") and sym[2:].isdigit()):
        raise TradeValidationError(f"sym must be SH/SZ + 6 digits, got {sym!r}")


def _normalize_trade(t: dict[str, Any]) -> dict[str, Any]:
    """Fill auto fields (id, recorded_at) + coerce numeric types."""
    return {
        "id": t.get("id") or uuid.uuid4().hex,
        "date": str(t["date"]),
        "sym": str(t["sym"]),
        "name": str(t.get("name") or ""),
        "action": str(t["action"]),
        "price": round(float(t["price"]), 4),
        "shares": int(t["shares"]),
        "note": str(t.get("note") or ""),
        "recorded_at": t.get("recorded_at")
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def append_trade(trade: dict[str, Any], path: Path = TRADES_PATH) -> dict[str, Any]:
    """Validate + normalize + atomic append to JSON Lines.

    Returns the normalized trade dict (with id/recorded_at filled).
    Raises TradeValidationError on bad input.
    """
    _validate_trade(trade)
    normalized = _normalize_trade(trade)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(normalized, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(line)
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return normalized


def load_trades(path: Path = TRADES_PATH) -> list[dict[str, Any]]:
    """Read all trades sorted by recorded_at ASC (chronological).

    Empty list if file missing. Silently skip malformed JSON lines.
    """
    if not path.exists():
        return []
    trades: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if isinstance(t, dict):
                    trades.append(t)
            except json.JSONDecodeError:
                continue
    trades.sort(key=lambda t: t.get("recorded_at", ""))
    return trades


def aggregate_positions(trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate trades by sym → position state via perpetual weighted-avg cost.

    Returns {sym: {sym, name, net_shares, total_buy_shares, total_sell_shares,
                   weighted_avg_cost, total_buy_cost, total_sell_revenue,
                   realized_pnl, last_buy_date, last_sell_date, last_trade_date,
                   status, trade_count}}.
    """
    positions: dict[str, dict[str, Any]] = {}
    for t in trades:
        sym = t.get("sym")
        if not sym:
            continue
        p = positions.setdefault(sym, {
            "sym": sym,
            "name": t.get("name") or "",
            "net_shares": 0,
            "total_buy_shares": 0,
            "total_sell_shares": 0,
            "weighted_avg_cost": 0.0,
            "total_buy_cost": 0.0,
            "total_sell_revenue": 0.0,
            "realized_pnl": 0.0,
            "last_buy_date": None,
            "last_sell_date": None,
            "last_trade_date": None,
            "trade_count": 0,
        })
        if t.get("name") and not p["name"]:
            p["name"] = t["name"]
        action = t.get("action")
        price = float(t.get("price", 0))
        shares = int(t.get("shares", 0))
        date = t.get("date")
        p["trade_count"] += 1
        p["last_trade_date"] = date

        if action == "BUY":
            if p["net_shares"] > 0:
                new_avg = (
                    (p["net_shares"] * p["weighted_avg_cost"]) + (shares * price)
                ) / (p["net_shares"] + shares)
            else:
                new_avg = price  # restart avg when flat/short
            p["weighted_avg_cost"] = round(new_avg, 4)
            p["net_shares"] += shares
            p["total_buy_shares"] += shares
            p["total_buy_cost"] = round(p["total_buy_cost"] + shares * price, 2)
            p["last_buy_date"] = date
        elif action == "SELL":
            effective = min(shares, max(p["net_shares"], 0))
            if effective > 0:
                p["realized_pnl"] = round(
                    p["realized_pnl"] + (price - p["weighted_avg_cost"]) * effective, 2
                )
            p["net_shares"] -= shares
            p["total_sell_shares"] += shares
            p["total_sell_revenue"] = round(
                p["total_sell_revenue"] + shares * price, 2
            )
            p["last_sell_date"] = date

    for p in positions.values():
        if p["net_shares"] > 0:
            p["status"] = "holding"
        elif p["net_shares"] < 0:
            p["status"] = "short"
        elif p["total_buy_shares"] > 0:
            p["status"] = "closed"
        else:
            p["status"] = "none"
    return positions


def get_trades_for_date(date_str: str, path: Path = TRADES_PATH) -> list[dict[str, Any]]:
    """Filter trades by date == date_str ('YYYY-MM-DD')."""
    return [t for t in load_trades(path) if t.get("date") == date_str]
