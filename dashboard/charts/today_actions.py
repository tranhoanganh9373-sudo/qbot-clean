"""Dashboard panel — Today's Actions · 今日待办智能 checklist.

聚合多源信息生成今日决策清单, 按优先级排序:
  - URGENT (红): 持仓股跌破止损价 → 立即复核
  - WATCH  (黄): 持仓股接近止损 (5% 内)
  - ACTION (蓝): 今日新推荐股 → 考虑买入
  - INFO   (灰): 下次 daily_check / mt180 fetch 状态

数据源 (全只读):
  - trades.jsonl (聚合当前持仓)
  - portfolio.xlsx Positions sheet (推荐 picks metadata, frozen snapshot)
  - baidu_kline.parquet (今日 close)
  - com.claude_finance.daily_check.plist (下次 trigger)
  - data_cache/mt180/*.jsonl (subagent fetch 进度)
"""
from __future__ import annotations

import html
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from claude_finance.trades_log import aggregate_positions, load_trades  # noqa: E402

PORTFOLIO_XLSX = ROOT / "data_cache" / "portfolio.xlsx"
KLINE_PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"
PLIST_PATH = ROOT / "examples" / "com.claude_finance.daily_check.plist"
MT180_DETAIL_PATH = ROOT / "data_cache" / "mt180" / "indicators_detail.jsonl"
MT180_LIST_PATH = ROOT / "data_cache" / "mt180" / "indicators_list.jsonl"

COLOR_URGENT = "#dc2626"
COLOR_WATCH = "#f59e0b"
COLOR_ACTION = "#2563eb"
COLOR_INFO = "#6b7280"
COLOR_OK = "#16a34a"


def _placeholder(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{message}"
        "</div>"
    )


def _load_today_close() -> dict[str, float]:
    if not KLINE_PARQUET.exists():
        return {}
    try:
        df = pd.read_parquet(KLINE_PARQUET, columns=["code", "date", "close"])
        df = df.sort_values(["code", "date"])
        latest = df.groupby("code", sort=False).tail(1)
        out: dict[str, float] = {}
        for _, r in latest.iterrows():
            c = str(r["code"]).zfill(6)
            prefix = "SH" if c[0] in ("6", "9") else "SZ"
            out[f"{prefix}{c}"] = float(r["close"])
        return out
    except Exception:
        return {}


def _load_picks_metadata() -> dict[str, dict]:
    if not PORTFOLIO_XLSX.exists():
        return {}
    try:
        df = pd.read_excel(PORTFOLIO_XLSX, sheet_name="Positions")
    except Exception:
        return {}
    if df.empty or "代码" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        sym = str(r.get("代码") or "")
        if not sym:
            continue
        out[sym] = {
            "name": str(r.get("名称") or ""),
            "status_xlsx": str(r.get("状态") or ""),
            "rec_price": r.get("推荐价"),
            "stop_loss": r.get("止损价(-8%)"),
            "rec_date": str(r.get("推荐日期") or "")[:10],
        }
    return out


def _next_daily_check_trigger() -> datetime | None:
    if not PLIST_PATH.exists():
        return None
    try:
        text = PLIST_PATH.read_text(encoding="utf-8")
    except OSError:
        return None
    h = re.search(r"<key>Hour</key>\s*<integer>\s*(\d+)\s*</integer>", text)
    m = re.search(r"<key>Minute</key>\s*<integer>\s*(\d+)\s*</integer>", text)
    if not h or not m:
        return None
    hour, minute = int(h.group(1)), int(m.group(1))
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    return target


def _build_items() -> list[dict]:
    items: list[dict] = []
    trades = load_trades()
    positions = aggregate_positions(trades)
    picks_meta = _load_picks_metadata()
    today_close = _load_today_close()

    # URGENT + WATCH: 止损
    breached: list[tuple] = []
    near_stop: list[tuple] = []
    for sym, pos in positions.items():
        if pos.get("net_shares", 0) <= 0:
            continue
        meta = picks_meta.get(sym, {})
        stop = meta.get("stop_loss")
        close = today_close.get(sym)
        if not (isinstance(stop, (int, float)) and isinstance(close, (int, float))):
            continue
        if close < stop:
            breached.append((sym, meta.get("name", ""), close, stop))
        elif close < stop * 1.05:
            near_stop.append((sym, meta.get("name", ""), close, stop))
    for sym, name, close, stop in breached:
        items.append({
            "priority": 0, "icon": "🚨", "color": COLOR_URGENT,
            "title": f"{sym} {name} 跌破止损",
            "msg": f"当前 ¥{close:.2f} ≤ 止损 ¥{stop:.2f} — 立即复核仓位",
        })
    for sym, name, close, stop in near_stop:
        gap_pct = (close / stop - 1) * 100
        items.append({
            "priority": 1, "icon": "🟡", "color": COLOR_WATCH,
            "title": f"{sym} {name} 接近止损",
            "msg": f"当前 ¥{close:.2f} 距止损 ¥{stop:.2f} 仅 {gap_pct:+.1f}% — 观察",
        })

    # ACTION: 推荐股
    today_str = datetime.now().strftime("%Y-%m-%d")
    rec_today: list[tuple] = []
    rec_stale: list[tuple] = []
    held_syms = {s for s, p in positions.items() if p.get("net_shares", 0) > 0}
    for sym, meta in picks_meta.items():
        if meta.get("status_xlsx") != "推荐":
            continue
        if sym in held_syms:
            continue
        rec_date = meta.get("rec_date", "")
        if rec_date == today_str:
            rec_today.append((sym, meta))
        else:
            rec_stale.append((sym, meta))
    if rec_today:
        sample = ", ".join(f"{s} {m.get('name','')}" for s, m in rec_today[:3])
        more = "" if len(rec_today) <= 3 else f" 等 {len(rec_today)} 只"
        items.append({
            "priority": 2, "icon": "🎯", "color": COLOR_ACTION,
            "title": f"今日新推荐 {len(rec_today)} 只",
            "msg": f"{sample}{more} · 决策买入与否 → 见 Recommended Picks panel",
        })
    if rec_stale:
        items.append({
            "priority": 3, "icon": "📋", "color": COLOR_INFO,
            "title": f"前日推荐 {len(rec_stale)} 只 待决策",
            "msg": "上批 picks 尚未买入, 继续观察或放弃",
        })

    # INFO: 下次 daily_check
    next_trig = _next_daily_check_trigger()
    if next_trig:
        delta = next_trig - datetime.now()
        hours = delta.total_seconds() / 3600
        ts = next_trig.strftime("%m-%d %H:%M")
        color, icon = (COLOR_ACTION, "⏰") if hours < 6 else (COLOR_INFO, "⏳")
        items.append({
            "priority": 4, "icon": icon, "color": color,
            "title": f"下次 daily_check 触发: {ts}",
            "msg": f"距今约 {hours:.1f} 小时 (launchd 自动)",
        })

    # INFO: mt180 fetch
    if MT180_LIST_PATH.exists():
        with MT180_LIST_PATH.open("r", encoding="utf-8") as fh:
            list_lines = sum(1 for _ in fh)
        if MT180_DETAIL_PATH.exists():
            with MT180_DETAIL_PATH.open("r", encoding="utf-8") as fh:
                detail_lines = sum(1 for _ in fh)
        else:
            detail_lines = 0
        if list_lines > 0:
            pct = detail_lines / list_lines * 100
            if detail_lines < list_lines:
                items.append({
                    "priority": 5, "icon": "🛠", "color": COLOR_INFO,
                    "title": f"mt180 fetch 进行中: {detail_lines:,}/{list_lines:,} ({pct:.1f}%)",
                    "msg": "TDX 公式抓取后台进行, 完成后可做 Phase A IC 评估",
                })
            else:
                items.append({
                    "priority": 6, "icon": "✓", "color": COLOR_OK,
                    "title": f"mt180 fetch 完成: {detail_lines:,} 公式",
                    "msg": "next: TDX → Python factor parser → Phase A IC",
                })

    if not items:
        items.append({
            "priority": 99, "icon": "✓", "color": COLOR_OK,
            "title": "今日无紧急待办",
            "msg": "持仓全部在止损价之上, 无新推荐需决策",
        })
    items.sort(key=lambda x: x["priority"])
    return items


def build_today_actions_section() -> str:
    try:
        items = _build_items()
    except Exception as exc:  # noqa: BLE001
        return _placeholder(
            f"构建今日待办失败: <code>{type(exc).__name__}: "
            f"{html.escape(str(exc))}</code>"
        )

    rows_html: list[str] = []
    for it in items:
        rows_html.append(
            '<div style="display:flex; align-items:flex-start; gap:12px; '
            'padding:10px 14px; margin-bottom:8px; '
            f'background:rgba(0,0,0,0.02); border-left:3px solid {it["color"]}; '
            'border-radius:0 4px 4px 0;">'
            f'<span style="font-size:18px; line-height:1;">{it["icon"]}</span>'
            '<div style="flex:1; min-width:0;">'
            f'<div style="font-size:13px; font-weight:600; color:{it["color"]};">'
            f'{html.escape(it["title"])}</div>'
            f'<div style="font-size:12px; color:var(--muted, #6b7280); '
            f'margin-top:2px;">{html.escape(it["msg"])}</div>'
            '</div></div>'
        )

    counts: dict[str, int] = {}
    for it in items:
        p = it["priority"]
        cat = ("urgent" if p == 0 else "watch" if p == 1
               else "action" if p <= 3 else "info")
        counts[cat] = counts.get(cat, 0) + 1
    summary_parts: list[str] = []
    if counts.get("urgent"):
        summary_parts.append(f"<strong style='color:{COLOR_URGENT};'>{counts['urgent']} 紧急</strong>")
    if counts.get("watch"):
        summary_parts.append(f"<strong style='color:{COLOR_WATCH};'>{counts['watch']} 观察</strong>")
    if counts.get("action"):
        summary_parts.append(f"<strong style='color:{COLOR_ACTION};'>{counts['action']} 决策</strong>")
    if counts.get("info"):
        summary_parts.append(f"<span style='color:{COLOR_INFO};'>{counts['info']} 提示</span>")
    summary = (
        '<div style="margin-bottom:10px; font-size:12px; color:var(--muted, #6b7280);">'
        f"共 {len(items)} 项: {' · '.join(summary_parts)}"
        "</div>"
    )
    return summary + "".join(rows_html)
