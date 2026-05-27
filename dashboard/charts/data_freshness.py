"""Dashboard panel — 数据新鲜度总览.

显示 6 个核心数据源 last-mtime + 距今 + 状态色:
绿 fresh / 黄 stale / 红 outdated / 灰 缺失.

数据 I/O (全只读, 仅 stat mtime):
  - data_cache/baidu_kline.parquet
  - data_cache/picks_today.json
  - data_cache/live_5m_picks.json
  - data_cache/paper_trade_log.csv
  - data_cache/multi_agent_log.jsonl
  - data_cache/sanity_check_log.csv
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# (label, path, fresh_sec, warn_sec, kind)
SOURCES = [
    ("baidu_kline 日 K", ROOT / "data_cache" / "baidu_kline.parquet",
     3600 * 24, 3600 * 48, "daily"),
    ("picks_today 选股", ROOT / "data_cache" / "picks_today.json",
     3600 * 24, 3600 * 48, "daily"),
    ("live_5m 行情", ROOT / "data_cache" / "live_5m_picks.json",
     180, 600, "realtime"),
    ("paper_trade_log 交易", ROOT / "data_cache" / "paper_trade_log.csv",
     3600 * 24, 3600 * 48, "daily"),
    ("multi_agent 讨论", ROOT / "data_cache" / "multi_agent_log.jsonl",
     3600 * 24, 3600 * 48, "daily"),
    ("sanity_check 完整性", ROOT / "data_cache" / "sanity_check_log.csv",
     3600 * 24, 3600 * 48, "daily"),
]


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} 秒前"
    if seconds < 3600:
        return f"{int(seconds / 60)} 分钟前"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} 小时前"
    return f"{seconds / 86400:.1f} 天前"


def _color_for_age(age_sec: float, fresh_sec: int, warn_sec: int) -> tuple[str, str]:
    if age_sec < fresh_sec:
        return ("#16a34a", "✓ fresh")
    if age_sec < warn_sec:
        return ("#f59e0b", "⚠ stale")
    return ("#dc2626", "🔴 outdated")


def build_data_freshness_section() -> str:
    now = datetime.now()
    rows: list[dict] = []
    for label, path, fresh_sec, warn_sec, _kind in SOURCES:
        if not path.exists():
            rows.append({
                "label": label, "path": str(path.name),
                "age_str": "—", "color": "#6b7280", "status": "缺失",
                "ts_str": "—",
            })
            continue
        mtime = path.stat().st_mtime
        age_sec = now.timestamp() - mtime
        color, status = _color_for_age(age_sec, fresh_sec, warn_sec)
        ts_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append({
            "label": label, "path": path.name,
            "age_str": _fmt_age(age_sec),
            "color": color, "status": status,
            "ts_str": ts_str,
        })

    has_red = any(r["color"] == "#dc2626" for r in rows)
    has_warn = any(r["color"] == "#f59e0b" for r in rows)
    overall_color = "#dc2626" if has_red else "#f59e0b" if has_warn else "#16a34a"
    overall_label = (
        "🔴 部分数据过期" if has_red else
        "⚠ 部分数据 stale" if has_warn else
        "✓ 全部新鲜"
    )

    banner = (
        f'<div style="background:rgba(107,114,128,0.08); border-left:3px solid {overall_color}; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        f'<strong>📊 数据新鲜度</strong> · <strong>{overall_label}</strong>'
        '<br><span style="color:var(--muted, #6b7280); font-size:11px;">'
        'realtime: 5m 行情 < 3min 绿 / < 10min 黄. '
        'daily: < 24h 绿 / < 48h 黄.'
        '</span></div>'
    )

    cards = []
    for r in rows:
        cards.append(
            f'<div style="background:rgba(107,114,128,0.06); padding:8px 12px; '
            f'border-left:3px solid {r["color"]}; border-radius:0 4px 4px 0; '
            'flex:1 1 240px; min-width:240px;">'
            f'<div style="font-size:12px; font-weight:600;">{html.escape(r["label"])}</div>'
            f'<div style="font-size:11px; color:var(--muted, #6b7280); margin:2px 0;">'
            f'<code style="font-size:10px;">{html.escape(r["path"])}</code></div>'
            f'<div style="font-size:12px; color:{r["color"]}; font-weight:600;">'
            f'{r["status"]} · {r["age_str"]}</div>'
            f'<div style="font-size:10px; color:var(--muted, #6b7280);">'
            f'{html.escape(r["ts_str"])}'
            '</div>'
            '</div>'
        )
    cards_html = (
        '<div style="display:flex; gap:8px; flex-wrap:wrap;">'
        + "".join(cards)
        + '</div>'
    )

    return banner + cards_html
