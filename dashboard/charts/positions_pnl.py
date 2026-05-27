"""持仓 PnL 表 — Trade log driven (取代 xlsx user_input).

Panel A "当前持仓盈亏 (只读统计)" 数据流:
  - data_cache/trades.jsonl → aggregate_positions: 净持仓 / WAC / 已实现盈亏
  - portfolio.xlsx Positions sheet: 推荐 metadata (推荐价 / 止损价 / 权重) — paper_trade 写
  - baidu_kline.parquet: 最新交易日 close (用作当前价 / 市值 / 浮动 PnL)

Panel B "今日操作录入":
  - 单行新增交易表单 (sym select + action + price + shares + note)
  - 今日已录入交易历史 (filter trades.jsonl WHERE date == today, 倒序)
  - 提交 → POST /submit {trade} → trades_log.append_trade + render → reload

不再使用 inline input per row 的旧覆盖式 user_input.json 流.
"""
from __future__ import annotations

import html
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from claude_finance.trades_log import (  # noqa: E402
    aggregate_positions,
    get_trades_for_date,
    load_trades,
)

SHEET_NAME = "Positions"

# Panel B sym select 候选: status='推荐' (paper_trade 产) 全收 + 下列 pinned 即使持仓也保留
PANEL_B_PINNED_SYMS: set[str] = {"SH600547", "SZ300347"}

# Panel A 列定义: (label, internal_key, width_pct, css_class)
PANEL_A_COLS = [
    ("Code", "code", 8, ""),
    ("Name", "name", 8, ""),
    ("Status", "status", 7, ""),
    ("权重 %", "weight_pct", 6, ""),
    ("推荐价", "rec_price", 7, ""),
    ("止损价", "stop_loss", 7, ""),
    ("当前价", "today_close", 7, ""),     # 紧跟 止损价 — 决策对照: 推荐价 / 止损价 / 当前价
    ("持仓数量", "net_shares", 7, ""),
    ("平均成本", "wac", 7, ""),
    ("市值 ¥", "market_value", 7, ""),
    ("浮动 PnL %", "float_pnl_pct", 7, "pnl-cell"),
    ("浮动利润 ¥", "float_pnl_yuan", 7, "pnl-cell"),
    ("已实现盈亏 ¥", "realized_pnl", 7, "pnl-cell"),
    ("最后交易", "last_trade_date", 8, ""),
]
# sanity: widths sum to 100
assert sum(w for _, _, w, _ in PANEL_A_COLS) == 100, "PANEL_A widths must sum to 100"


def _pnl_class(pnl_pct) -> str:
    """根据浮盈% 选 CSS class."""
    if pnl_pct is None or (isinstance(pnl_pct, float) and math.isnan(pnl_pct)):
        return "pnl-flat"
    if pnl_pct > 0:
        return "pnl-pos"
    if pnl_pct < 0:
        return "pnl-neg"
    return "pnl-flat"


def _fmt(value, kind: str) -> str:
    """Format value by kind. 空 → '-'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if kind == "pct":
        return f"{value * 100:+.2f}%" if isinstance(value, (int, float)) else str(value)
    if kind == "money_signed":
        return f"{value:+,.0f}" if isinstance(value, (int, float)) else str(value)
    if kind == "money":
        return f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
    if kind == "price":
        return f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
    if kind == "shares":
        return f"{int(value):,}" if isinstance(value, (int, float)) else str(value)
    if kind == "weight":
        return f"{value:.1f}%" if isinstance(value, (int, float)) else str(value)
    if kind == "date":
        s = str(value)
        return html.escape(s[:10])
    return html.escape(str(value))


_KIND_BY_KEY = {
    "code": "raw", "name": "raw", "status": "raw",
    "weight_pct": "weight",
    "rec_price": "price", "stop_loss": "price", "wac": "price", "today_close": "price",
    "net_shares": "shares", "trade_count": "shares",
    "market_value": "money", "realized_pnl": "money_signed",
    "float_pnl_yuan": "money_signed",
    "float_pnl_pct": "pct",
    "last_trade_date": "date",
}


def _placeholder_html(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:32px 16px;">'
        f"{message}"
        "</div>"
    )


# ──────────────────── data load helpers ────────────────────


def _load_picks_from_xlsx(portfolio_xlsx: Path) -> dict[str, dict]:
    """从 xlsx Positions sheet 读 paper_trade-managed picks metadata.

    返回 {sym: {name, status_xlsx, rec_price, stop_loss, weight_pct, rec_date}}.
    用于 Panel A: 显示 推荐 状态 + 推荐价 + 止损价 + 权重(用户决策参考).
    """
    if not portfolio_xlsx.exists():
        return {}
    try:
        df = pd.read_excel(portfolio_xlsx, sheet_name=SHEET_NAME)
    except Exception:
        return {}
    if df.empty or "代码" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        sym = str(row.get("代码") or "")
        if not sym:
            continue
        out[sym] = {
            "name": str(row.get("名称") or ""),
            "status_xlsx": str(row.get("状态") or ""),
            "rec_price": row.get("推荐价") if "推荐价" in df.columns else None,
            "stop_loss": row.get("止损价(-8%)") if "止损价(-8%)" in df.columns else None,
            "weight_pct": row.get("Score权重%") if "Score权重%" in df.columns else None,
            "rec_date": row.get("推荐日期") if "推荐日期" in df.columns else None,
        }
    return out


def _load_today_close(kline_parquet: Path | None) -> dict[str, float]:
    """每股最新 close → {sym: close}.

    优先级 (盘中实时优于盘后 daily close):
      1) baidu_kline.parquet 最新 daily close (兜底)
      2) live_5m_picks.json (poller 60s 写, 含 5m bar 最新 close — 盘中最准, 覆盖 baidu)

    sym 加 SH/SZ 前缀对齐 trades.jsonl 格式.
    """
    import json
    out: dict[str, float] = {}

    # 1) baidu_kline daily close (兜底)
    if kline_parquet is not None and kline_parquet.exists():
        try:
            df = pd.read_parquet(kline_parquet, columns=["code", "date", "close"])
            if not df.empty:
                df = df.sort_values(["code", "date"])
                latest = df.groupby("code", sort=False).tail(1)
                for _, r in latest.iterrows():
                    c = str(r["code"]).zfill(6)
                    prefix = "SH" if c[0] in ("6", "9") else "SZ"
                    out[f"{prefix}{c}"] = float(r["close"])
        except Exception:
            pass

    # 2) live_5m_picks.json 覆盖 (5m close 最实时, 盘中 baidu_kline 当日 row 缺时尤其重要)
    live_path = (
        kline_parquet.parent / "live_5m_picks.json" if kline_parquet else None
    )
    if live_path is not None and live_path.exists():
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


def _compose_panel_a_rows(
    picks: dict[str, dict],
    positions: dict[str, dict],
    today_close: dict[str, float],
) -> list[dict]:
    """合并 trades 聚合 positions + baidu close → Panel A 行 list.

    **仅纳入有过交易的 sym** (trade_count > 0):
      - 持仓 (net_shares > 0): 显示 持仓数量 / WAC / 市值 / 浮动 PnL
      - 已平 (net_shares == 0): 仅显示 WAC / 已实现盈亏 (无市值/浮动 PnL)
      - 异常空仓 (net_shares < 0): 标 abnormal, 仍显示已实现盈亏
    纯推荐 (无 trade) 不在此显示 — 它们出现在 Panel B sym select 里.

    picks 仅供 metadata 引用 (推荐价 / 止损价 / 权重), 不决定是否入表.
    """
    rows: list[dict] = []
    for sym, pos in positions.items():
        trade_count = pos.get("trade_count", 0)
        if trade_count <= 0:
            continue  # 无 trade 不入 Panel A
        pick = picks.get(sym, {})
        name = pos.get("name") or pick.get("name") or ""
        net_shares = pos.get("net_shares", 0)
        wac = pos.get("weighted_avg_cost") or None
        realized_pnl = pos.get("realized_pnl", 0.0)
        last_trade_date = pos.get("last_trade_date")

        if net_shares > 0:
            status = "持仓"
        elif net_shares == 0:
            status = "已平"
        else:
            status = "异常空仓"

        close = today_close.get(sym)
        market_value = (
            round(net_shares * close, 2)
            if close is not None and net_shares > 0 else None
        )
        float_pnl_pct = None
        float_pnl_yuan = None
        if close is not None and wac and net_shares > 0:
            float_pnl_pct = (close - wac) / wac
            float_pnl_yuan = round((close - wac) * net_shares, 2)

        # 止损警报: 持仓股 当前价 < 止损价 → 风险提醒
        stop_loss = pick.get("stop_loss")
        stop_loss_breached = False
        if (
            status == "持仓"
            and isinstance(stop_loss, (int, float))
            and isinstance(close, (int, float))
            and close < stop_loss
        ):
            stop_loss_breached = True

        rows.append({
            "code": sym,
            "name": name,
            "status": status,
            "weight_pct": pick.get("weight_pct"),
            "rec_price": pick.get("rec_price"),
            "stop_loss": stop_loss,
            # 持仓数量: 仅 持仓 才有意义; 平均成本: 持仓 + 已平 都保留 (audit history)
            "net_shares": net_shares if net_shares > 0 else None,
            "wac": wac,  # 已平股保留历史平均成本 (audit value)
            "today_close": close,
            "market_value": market_value,
            "float_pnl_pct": float_pnl_pct,
            "float_pnl_yuan": float_pnl_yuan,
            "realized_pnl": realized_pnl,  # 总有值 (持仓 = 0 / 已平 = 真实数)
            "trade_count": trade_count,
            "last_trade_date": last_trade_date,
            "stop_loss_breached": stop_loss_breached,
        })
    # 排序: 持仓 → 已平 → 异常空仓; 同类按 code 升
    status_rank = {"持仓": 0, "已平": 1, "异常空仓": 2}
    rows.sort(key=lambda r: (status_rank.get(r["status"], 9), r["code"]))
    return rows


# ──────────────────── CSS (scoped) ────────────────────

_CSS = """
<style>
.positions-editable table.data, .positions-readonly table.data { table-layout: fixed; width: 100%; }
.positions-editable table.data th, .positions-editable table.data td,
.positions-readonly table.data th, .positions-readonly table.data td {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.positions-readonly tr.row-holding { background: rgba(22,163,74,0.04); }
.positions-readonly tr.row-closed { color: var(--muted, #6b7280); }
.positions-readonly tr.row-recommend { font-style: italic; }
.positions-readonly tr.row-stop-loss {
    background: rgba(220,38,38,0.10) !important;
    border-left: 3px solid #dc2626;
}
.positions-readonly td.stop-loss-breach {
    color: #dc2626; font-weight: 700;
}
.positions-editable .trade-form {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px 14px;
    padding: 14px 16px; margin-bottom: 14px;
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.20);
    border-radius: 6px;
}
.positions-editable .trade-form label {
    display: flex; flex-direction: column; gap: 4px;
    font-size: 12px; color: var(--muted, #6b7280);
}
.positions-editable .trade-form input,
.positions-editable .trade-form select {
    padding: 6px 8px; font: inherit; font-size: 13px;
    color: var(--fg); background: var(--bg);
    border: 1px solid var(--border); border-radius: 4px;
}
.positions-editable .trade-form .field-note { grid-column: 1 / -1; }
.positions-editable .trade-form .field-actions { grid-column: 1 / -1; display: flex; gap: 10px; align-items: center; }
.positions-editable .trade-form input:focus,
.positions-editable .trade-form select:focus { outline: none; border-color: var(--accent); }
.positions-editable .trade-form .action-buy { color: #16a34a; font-weight: 600; }
.positions-editable .trade-form .action-sell { color: #dc2626; font-weight: 600; }
.positions-editable .export-bar {
    margin-top: 12px;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
}
.positions-editable .export-btn {
    padding: 8px 16px;
    font: inherit;
    font-size: 13px;
    font-weight: 500;
    color: #fff;
    background: var(--accent);
    border: none;
    border-radius: 6px;
    cursor: pointer;
}
.positions-editable .export-btn:hover { filter: brightness(1.1); }
.positions-editable .export-btn.submit-btn { background: #16a34a; }
.positions-editable .export-btn.submit-btn:disabled {
    background: #94a3b8; cursor: not-allowed; filter: none;
}
.positions-editable .server-banner {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    padding: 10px 14px; margin-bottom: 10px;
    background: rgba(220, 38, 38, 0.10); border-left: 3px solid #dc2626;
    border-radius: 4px; font-size: 12px; color: var(--fg);
}
.positions-editable .server-banner.hidden { display: none; }
.positions-editable .banner-icon { font-size: 16px; }
.positions-editable .banner-msg { flex: 1; min-width: 280px; }
.positions-editable .banner-cmd {
    background: rgba(0,0,0,0.4); color: #93c5fd;
    padding: 4px 8px; border-radius: 3px; font-size: 11px;
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
}
.positions-editable .copy-btn,
.positions-editable .recheck-btn {
    padding: 4px 10px; font-size: 11px; cursor: pointer;
    background: var(--bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
}
.positions-editable .copy-btn:hover,
.positions-editable .recheck-btn:hover { background: var(--accent); color: #fff; }
.positions-editable .server-status-pill {
    font-size: 11px; padding: 3px 8px; border-radius: 10px;
    background: rgba(148,163,184,0.20); color: var(--muted);
}
.positions-editable .server-status-pill.ok {
    background: rgba(22,163,74,0.15); color: #16a34a; font-weight: 600;
}
.positions-editable .server-status-pill.down {
    background: rgba(220,38,38,0.15); color: #dc2626; font-weight: 600;
}
.positions-editable .submit-msg {
    font-size: 12px;
    color: var(--muted);
    margin-left: 4px;
}
.positions-editable .submit-msg.flash { color: var(--green, #16a34a); }
.positions-editable .cost-estimate {
    font-size: 13px; font-weight: 600; padding: 4px 10px;
    background: rgba(37,99,235,0.08); color: var(--accent, #2563eb);
    border-radius: 4px; font-variant-numeric: tabular-nums;
}
.positions-editable .cost-estimate.action-sell {
    background: rgba(220,38,38,0.08); color: #dc2626;
}
.positions-editable .trade-history-table {
    margin-top: 14px;
}
.positions-editable .trade-history-table th { font-size: 12px; }
.positions-editable .trade-history-table td.action-buy { color: #16a34a; font-weight: 600; }
.positions-editable .trade-history-table td.action-sell { color: #dc2626; font-weight: 600; }
</style>
"""


# ──────────────────── Panel A: build_positions_table ────────────────────


def build_positions_table(
    portfolio_xlsx: Path, kline_parquet: Path | None = None
) -> str:
    """Panel A — 当前持仓盈亏 (只读统计) 从 trades 聚合 + xlsx picks + baidu close.

    显示列: Code / Name / Status / 权重 / 推荐价 / 止损价 / 持仓数量 /
            平均成本 / 当前价 / 市值 / 浮动 PnL % / 浮动利润 ¥ /
            已实现盈亏 ¥ / 交易次数 / 最后交易.
    """
    picks = _load_picks_from_xlsx(portfolio_xlsx)
    trades = load_trades()
    positions = aggregate_positions(trades)
    today_close = _load_today_close(kline_parquet)

    rows_data = _compose_panel_a_rows(picks, positions, today_close)
    if not rows_data:
        return _placeholder_html(
            "暂无持仓 / 已平交易记录 (trades.jsonl 无交易). "
            "在下方 'Trade Entry · 今日操作录入' 录入第一笔买入即可."
        )

    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _l, _k, w, _c in PANEL_A_COLS
    ) + "</colgroup>"
    thead = "".join(f"<th>{label}</th>" for label, _k, _w, _c in PANEL_A_COLS)

    rows_html: list[str] = []
    breached_syms: list[tuple[str, str, float, float]] = []
    for row in rows_data:
        row_cls_map = {
            "持仓": "row-holding", "已平": "row-closed",
            "推荐": "row-recommend", "异常空仓": "row-closed",
        }
        row_cls_list = [row_cls_map.get(row["status"], "")]
        if row.get("stop_loss_breached"):
            row_cls_list.append("row-stop-loss")
            breached_syms.append((
                row["code"], row["name"],
                row.get("today_close") or 0, row.get("stop_loss") or 0,
            ))
        row_cls = " ".join(c for c in row_cls_list if c)
        pnl_pct = row.get("float_pnl_pct")
        pnl_cls = _pnl_class(pnl_pct)
        tds: list[str] = []
        for label, key, _w, css in PANEL_A_COLS:
            kind = _KIND_BY_KEY.get(key, "raw")
            val = row.get(key)
            val_str = _fmt(val, kind)
            cls_parts = []
            if css == "pnl-cell" and pnl_pct is not None:
                cls_parts.append(pnl_cls)
            # 当前价跌破止损: 加 stop-loss-breach class
            if key == "today_close" and row.get("stop_loss_breached"):
                cls_parts.append("stop-loss-breach")
            cls_attr = f' class="{" ".join(cls_parts)}"' if cls_parts else ""
            tds.append(f"<td{cls_attr}>{val_str}</td>")
        tr_class = f' class="{row_cls}"' if row_cls else ""
        rows_html.append(f"<tr{tr_class}>" + "".join(tds) + "</tr>")
    tbody = "".join(rows_html)

    # 顶部止损警报 banner
    if breached_syms:
        breach_items = " · ".join(
            f"<strong>{html.escape(s)} {html.escape(n)}</strong> "
            f"(当前 {c:.2f} ≤ 止损 {sl:.2f})"
            for s, n, c, sl in breached_syms
        )
        alert_banner = (
            '<div style="background:rgba(220,38,38,0.12); border-left:4px solid #dc2626;'
            ' padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0;'
            ' font-size:13px; color:var(--fg);">'
            f'<strong style="color:#dc2626;">⚠️ 止损警报 ({len(breached_syms)} 只)</strong>'
            f': {breach_items}'
            ' &middot; <span style="color:var(--muted, #6b7280);">'
            '止损价 = 推荐价 × 0.92, 跌破后建议复核仓位 (非机械触发)</span>'
            '</div>'
        )
    else:
        alert_banner = ""

    # summary
    n_holding = sum(1 for r in rows_data if r["status"] == "持仓")
    n_closed = sum(1 for r in rows_data if r["status"] == "已平")
    n_recommend = sum(1 for r in rows_data if r["status"] == "推荐")
    realized_total = sum(
        r["realized_pnl"] for r in rows_data if r.get("realized_pnl")
    )
    float_total = sum(
        r["float_pnl_yuan"] for r in rows_data if r.get("float_pnl_yuan")
    )
    summary = (
        '<div style="margin-top:8px; font-size:12px; color:var(--muted);">'
        f"持仓 {n_holding} 行 / 已平 {n_closed} 行 / 推荐 {n_recommend} 行 / 共 {len(rows_data)} 行"
        f" &middot; 累计已实现 ¥{realized_total:+,.0f}"
        f" &middot; 当前浮动 ¥{float_total:+,.0f}"
        "</div>"
    )

    return (
        _CSS
        + '<div class="positions-readonly">'
        + alert_banner
        + '<table class="data">'
        + colgroup
        + f"<thead><tr>{thead}</tr></thead>"
        + f"<tbody>{tbody}</tbody>"
        + "</table>"
        + summary
        + "</div>"
    )


# ──────────────────── Panel B: build_trade_input_table ────────────────────


def _trade_form_html(candidate_syms: list[tuple[str, str]], today_str: str) -> str:
    """单行新增交易表单 HTML.

    candidate_syms: [(sym, display_label)] — sym select 候选列表.
    """
    opt_html = "".join(
        f'<option value="{html.escape(s)}">{html.escape(label)}</option>'
        for s, label in candidate_syms
    )
    return (
        '<form id="trade-form" class="trade-form" onsubmit="return false;">'
        '<label>日期<input type="date" id="trade-date" value="' + today_str + '" required></label>'
        f'<label>代码<select id="trade-sym" required><option value="">--选择--</option>{opt_html}</select></label>'
        '<label>方向<select id="trade-action" required>'
        '<option value="BUY">BUY 买入</option>'
        '<option value="SELL">SELL 卖出</option>'
        '</select></label>'
        '<label>价格<input type="number" id="trade-price" step="0.01" min="0.01" placeholder="29.95" required></label>'
        '<label>数量 (股)<input type="number" id="trade-shares" step="100" min="1" placeholder="200" required></label>'
        '<label class="field-note">备注<input type="text" id="trade-note" maxlength="80" placeholder="开仓 / 加仓 / 减仓 / 清仓"></label>'
        '<div class="field-actions">'
        '<button type="button" id="submit-trade-btn" class="export-btn submit-btn" title="Ctrl/Cmd+Enter 也可提交">🚀 提交交易</button>'
        '<button type="button" id="reset-trade-btn" class="export-btn" style="background:#6b7280;">清空</button>'
        '<span id="cost-estimate" class="cost-estimate">估算: ¥-</span>'
        '<span id="submit-msg" class="submit-msg">未提交</span>'
        '<span id="server-status-pill" class="server-status-pill">检测中...</span>'
        '</div>'
        '</form>'
    )


def _trade_history_table_html(trades_today: list[dict]) -> str:
    """今日已录入交易历史表 (倒序)."""
    if not trades_today:
        return (
            '<div class="trade-history-table">'
            '<h4 style="margin:14px 0 6px; font-size:13px; color:var(--muted);">'
            '今日已录入交易 (0 笔)</h4>'
            '<div class="placeholder-content" style="padding:14px;">'
            '今日尚无交易. 填上表 → 点 🚀 提交交易.'
            '</div></div>'
        )
    rows = []
    for t in reversed(trades_today):
        action = t.get("action", "")
        action_cls = "action-buy" if action == "BUY" else "action-sell"
        rec_at = (t.get("recorded_at") or "")[11:19]  # HH:MM:SS
        rows.append(
            "<tr>"
            f"<td>{html.escape(rec_at)}</td>"
            f"<td>{html.escape(str(t.get('sym','')))}</td>"
            f"<td>{html.escape(str(t.get('name','')))}</td>"
            f'<td class="{action_cls}">{html.escape(action)}</td>'
            f"<td>{_fmt(t.get('price'), 'price')}</td>"
            f"<td>{_fmt(t.get('shares'), 'shares')}</td>"
            f"<td>{html.escape(str(t.get('note','') or ''))}</td>"
            "</tr>"
        )
    tbody = "".join(rows)
    return (
        '<div class="trade-history-table">'
        f'<h4 style="margin:14px 0 6px; font-size:13px; color:var(--muted);">'
        f'今日已录入交易 ({len(trades_today)} 笔, 倒序)</h4>'
        '<table class="data">'
        '<colgroup>'
        '<col style="width:10%;"><col style="width:10%;"><col style="width:14%;">'
        '<col style="width:8%;"><col style="width:10%;"><col style="width:10%;">'
        '<col style="width:38%;">'
        '</colgroup>'
        '<thead><tr><th>时间</th><th>代码</th><th>名称</th><th>方向</th>'
        '<th>价格</th><th>数量</th><th>备注</th></tr></thead>'
        f'<tbody>{tbody}</tbody>'
        '</table></div>'
    )


def _server_banner_html() -> str:
    return (
        '<div id="submit-server-banner" class="server-banner hidden">'
        '<span class="banner-icon">⚠️</span>'
        '<span class="banner-msg">'
        'Submit server 未运行 — 无法提交交易. 请在 claude_finance 根目录 终端启动:'
        '</span>'
        '<code class="banner-cmd" id="submit-server-cmd">'
        'python examples/dashboard_submit_server.py'
        '</code>'
        '<button type="button" id="copy-server-cmd" class="copy-btn">📋 复制</button>'
        '<button type="button" id="recheck-server" class="recheck-btn">🔄 重新检测</button>'
        '</div>'
    )


def build_trade_input_table(portfolio_xlsx: Path) -> str:
    """Panel B — 今日操作录入.

    显示:
      - 红色 server-banner (server down 时 JS show)
      - 单行新增交易表单 (date / sym / action / price / shares / note)
      - 今日 trades 历史表 (倒序)

    sym select 候选 = xlsx Positions status='推荐' rows + PANEL_B_PINNED_SYMS +
    现有持仓 (from trades aggregate, net_shares>0).
    """
    picks = _load_picks_from_xlsx(portfolio_xlsx)
    trades = load_trades()
    positions = aggregate_positions(trades)

    # candidate sym list = 推荐 + 持仓 + pinned
    candidates: dict[str, str] = {}  # sym → "SH600547 山东黄金"
    for sym, p in picks.items():
        # 状态 = '推荐' 或 pinned 才纳入
        if p.get("status_xlsx") == "推荐" or sym in PANEL_B_PINNED_SYMS:
            candidates[sym] = f"{sym} {p.get('name', '')}"
    for sym, pos in positions.items():
        if pos.get("net_shares", 0) > 0:
            candidates[sym] = f"{sym} {pos.get('name', '') or candidates.get(sym, '').split(' ', 1)[-1]}"
    # ensure pinned are present even if not in picks/positions
    for sym in PANEL_B_PINNED_SYMS:
        if sym not in candidates:
            name = picks.get(sym, {}).get("name") or positions.get(sym, {}).get("name") or ""
            candidates[sym] = f"{sym} {name}"
    sorted_candidates = sorted(candidates.items(), key=lambda kv: kv[0])

    today_str = datetime.now().strftime("%Y-%m-%d")
    trades_today = get_trades_for_date(today_str)

    form_html = _trade_form_html(sorted_candidates, today_str)
    history_html = _trade_history_table_html(trades_today)
    banner_html = _server_banner_html()

    return (
        '<div class="positions-editable">'
        + banner_html
        + form_html
        + history_html
        + "</div>"
    )
