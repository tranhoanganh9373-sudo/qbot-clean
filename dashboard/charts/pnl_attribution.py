"""Dashboard panel — PnL attribution by stock.

聚合 trades.jsonl 全部交易 → 每只 stock 的 已实现 + 浮动 PnL,
按总 PnL 排序 (winners 上, losers 下) + 横向 bar chart 显示占比.

数据源 (全只读):
  - data_cache/trades.jsonl → load_trades + aggregate_positions
  - data_cache/baidu_kline.parquet → 最新 close (浮动 PnL)
  - data_cache/stock_names.json → 中文名 fallback
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

TRADES_PATH = ROOT / "data_cache" / "trades.jsonl"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
NAMES_PATH = ROOT / "data_cache" / "stock_names.json"


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _load_names() -> dict[str, str]:
    if not NAMES_PATH.exists():
        return {}
    try:
        return json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_latest_close() -> dict[str, float]:
    """每 sym 最新 close. 优先 live_5m (实时 60s) 覆盖 baidu_kline (daily 兜底).

    baidu_kline schema: code (6-digit) / date / close → SH/SZ prefix 还原 full sym.
    live_5m schema: syms[*].sym (full SH/SZ) + latest_5m.close."""
    out: dict[str, float] = {}
    # 1) baidu_kline daily close 兜底
    if KLINE_PATH.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
            df = df.sort_values("date").drop_duplicates("code", keep="last")
            for code, close in zip(df["code"], df["close"].astype(float)):
                code = str(code).zfill(6)
                if code.startswith(("600", "601", "603", "605", "688", "689")):
                    out[f"SH{code}"] = close
                elif code.startswith(("000", "001", "002", "003", "300", "301")):
                    out[f"SZ{code}"] = close
        except Exception:
            pass
    # 2) live_5m_picks.json 覆盖 (5m close 最实时, 盘中尤其重要)
    import json
    live_path = ROOT / "data_cache" / "live_5m_picks.json"
    if live_path.exists():
        try:
            data = json.loads(live_path.read_text(encoding="utf-8"))
            for s in data.get("syms", []):
                sym = s.get("sym")
                bar = s.get("latest_5m") or {}
                close = bar.get("close")
                if sym and isinstance(close, (int, float)) and close > 0:
                    out[sym] = float(close)
        except Exception:
            pass
    return out


def _compute_attribution() -> list[dict] | None:
    try:
        from claude_finance.trades_log import load_trades, aggregate_positions
    except Exception:
        return None
    if not TRADES_PATH.exists():
        return []
    try:
        trades = load_trades()
    except Exception:
        return None
    if not trades:
        return []
    positions = aggregate_positions(trades)
    close_map = _load_latest_close()
    name_map = _load_names()
    out: list[dict] = []
    for sym, p in positions.items():
        net_shares = int(p.get("net_shares", 0))
        wac = float(p.get("weighted_avg_cost", 0))
        realized = float(p.get("realized_pnl", 0))
        cur_close = float(close_map.get(sym, 0))
        floating = (cur_close - wac) * net_shares if (net_shares > 0 and cur_close > 0) else 0
        total_pnl = realized + floating
        market_value = cur_close * net_shares if net_shares > 0 else 0
        cost_basis = wac * net_shares if net_shares > 0 else 0
        out.append({
            "sym": sym,
            "name": p.get("name") or name_map.get(sym, "?"),
            "status": p.get("status", "?"),
            "net_shares": net_shares,
            "wac": wac,
            "close": cur_close,
            "realized_pnl": realized,
            "floating_pnl": floating,
            "total_pnl": total_pnl,
            "cost_basis": cost_basis,
            "market_value": market_value,
        })
    out.sort(key=lambda x: x["total_pnl"], reverse=True)
    return out


def _bar_html(value: float, max_abs: float, width_pct: float) -> str:
    if max_abs <= 0:
        return ""
    half = width_pct / 2
    bar_pct = min(abs(value) / max_abs, 1.0) * half
    if value >= 0:
        return (
            f'<div style="display:flex; align-items:center; height:14px;">'
            f'<div style="width:{half}%;"></div>'
            f'<div style="width:{bar_pct:.1f}%; background:#16a34a; height:10px;"></div>'
            f"</div>"
        )
    return (
        f'<div style="display:flex; align-items:center; height:14px; justify-content:flex-end;">'
        f'<div style="width:{bar_pct:.1f}%; background:#dc2626; height:10px;"></div>'
        f'<div style="width:{half}%;"></div>'
        f"</div>"
    )


def build_pnl_attribution_section() -> str:
    rows = _compute_attribution()
    if rows is None:
        return _placeholder(
            "无法导入 <code>claude_finance.trades_log</code> 或读 trades — panel 暂不可用."
        )
    if not rows:
        return _placeholder(
            "<code>data_cache/trades.jsonl</code> 空 — 尚无交易记录."
        )

    total_realized = sum(r["realized_pnl"] for r in rows)
    total_floating = sum(r["floating_pnl"] for r in rows)
    grand_total = total_realized + total_floating
    max_abs_pnl = max((abs(r["total_pnl"]) for r in rows), default=1.0)

    total_color = "#16a34a" if grand_total >= 0 else "#dc2626"
    realized_color = "#16a34a" if total_realized >= 0 else "#dc2626"
    floating_color = "#16a34a" if total_floating >= 0 else "#dc2626"

    banner = (
        '<div style="background:rgba(107,114,128,0.08); border-left:3px solid #6b7280; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        '<strong>📊 累计 PnL</strong> · '
        f'总 PnL: <strong style="color:{total_color};">¥{grand_total:+,.0f}</strong> · '
        f'已实现: <strong style="color:{realized_color};">¥{total_realized:+,.0f}</strong> · '
        f'浮动: <strong style="color:{floating_color};">¥{total_floating:+,.0f}</strong> · '
        f'股票数 <strong>{len(rows)}</strong> '
        f'(持仓 {sum(1 for r in rows if r["status"]=="holding")} / '
        f'已平 {sum(1 for r in rows if r["status"]=="closed")})<br>'
        '<span style="color:var(--muted, #6b7280); font-size:11px;">'
        "按总 PnL (已实现 + 浮动) 降序, winners 上 losers 下. "
        "浮动 PnL 用 baidu_kline 最新 close 计算 (盘后停盘则为收盘价)."
        "</span></div>"
    )

    cols = [
        ("#", 3), ("sym", 7), ("名称", 10), ("status", 6),
        ("持仓", 6), ("WAC", 7), ("现价", 7),
        ("已实现 ¥", 9), ("浮动 ¥", 9), ("总 PnL ¥", 9),
        ("贡献", 27),
    ]
    assert sum(w for _, w in cols) == 100, sum(w for _, w in cols)
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _, w in cols
    ) + "</colgroup>"
    right_cols = {"持仓", "WAC", "现价", "已实现 ¥", "浮动 ¥", "总 PnL ¥"}
    thead = "<tr>" + "".join(
        f'<th style="text-align:right;">{label}</th>' if label in right_cols
        else f"<th>{label}</th>"
        for label, _ in cols
    ) + "</tr>"

    body_rows: list[str] = []
    status_badge = {
        "holding": ("持仓", "#f59e0b", "rgba(245,158,11,0.10)"),
        "closed": ("已平", "#6b7280", "rgba(107,114,128,0.10)"),
        "short": ("做空", "#dc2626", "rgba(220,38,38,0.10)"),
        "none": ("无", "#6b7280", "rgba(107,114,128,0.10)"),
    }
    for i, r in enumerate(rows, 1):
        sym = r["sym"]
        s_label, s_color, s_bg = status_badge.get(
            r["status"], (r["status"], "#6b7280", "rgba(107,114,128,0.10)"),
        )
        realized_c = "#16a34a" if r["realized_pnl"] >= 0 else "#dc2626"
        floating_c = "#16a34a" if r["floating_pnl"] >= 0 else "#dc2626"
        total_c = "#16a34a" if r["total_pnl"] >= 0 else "#dc2626"
        bar_html = _bar_html(r["total_pnl"], max_abs_pnl, width_pct=100)
        body_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td><code>{html.escape(sym)}</code></td>"
            f"<td>{html.escape(r['name'])}</td>"
            f'<td><span style="color:{s_color}; background:{s_bg}; '
            f'padding:1px 6px; border-radius:8px; font-size:10px; font-weight:600;">'
            f"{s_label}</span></td>"
            f'<td style="text-align:right;">{r["net_shares"]:,}</td>'
            f'<td style="text-align:right;">{r["wac"]:.2f}</td>'
            f'<td style="text-align:right;">{r["close"]:.2f}</td>'
            f'<td style="text-align:right; color:{realized_c};">{r["realized_pnl"]:+,.0f}</td>'
            f'<td style="text-align:right; color:{floating_c};">{r["floating_pnl"]:+,.0f}</td>'
            f'<td style="text-align:right; color:{total_c}; font-weight:600;">{r["total_pnl"]:+,.0f}</td>'
            f"<td>{bar_html}</td>"
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
        "已实现 PnL = SELL 时 (sell_price - WAC) × sell_shares 累计;<br>"
        "浮动 PnL = (current_close - WAC) × net_shares (仅 holding 状态有值);<br>"
        "贡献 bar = 按 max_abs(总 PnL) normalize 居中, 直观看哪只贡献最大 / 拖最多.<br>"
        "数据源 <code>data_cache/trades.jsonl</code> + <code>baidu_kline.parquet</code> 最新 close."
        "</div>"
    )

    return banner + table + footer
