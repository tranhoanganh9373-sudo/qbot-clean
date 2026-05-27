"""Dashboard panel — 推荐股票清单 (paper_trade 模型当日 v19.10 picks).

数据源 (只读):
  - data_cache/picks_today.json (paper_trade_today.py 当日 dump, 含 8 picks + sidecar)
  - data_cache/stock_names.json (sym → 中文名 fallback)

显示列:
  # / sym / 名称 / final_score / z_pred / z_amp / z_jzf / JZF% / 推荐价 (close) /
  止损价 (-8%) / 等权 % / 预算 ¥ / 手数 (建议)
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PICKS_TODAY = ROOT / "data_cache" / "picks_today.json"
NAMES_PATH = ROOT / "data_cache" / "stock_names.json"
TRADE_LOG = ROOT / "data_cache" / "paper_trade_log.csv"
PORTFOLIO_STATE = ROOT / "data_cache" / "portfolio_state.json"

CAPITAL_PER_BUDGET = 50000  # 默认 50k 等权预算
N_DROP = 2  # paper_trade_today.py 默认每天最多换 2 只


def _load_holdings() -> tuple[list[str], str]:
    """读 portfolio_state.json 拿实际持仓 set + state 日期."""
    if not PORTFOLIO_STATE.exists():
        return [], ""
    try:
        d = json.loads(PORTFOLIO_STATE.read_text(encoding="utf-8"))
        h = d.get("holdings") or []
        date = str(d.get("date", ""))
        if isinstance(h, list):
            return [s for s in h if isinstance(s, str)], date
    except Exception:
        pass
    return [], ""


def _load_today_actions(today_date: str) -> tuple[list[dict], list[dict]]:
    """读 paper_trade_log.csv 拿 today_date 当天的 BUY / SELL actions."""
    if not TRADE_LOG.exists() or not today_date:
        return [], []
    try:
        import csv
        buys, sells = [], []
        with TRADE_LOG.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("date") != today_date:
                    continue
                if row.get("action") == "BUY":
                    buys.append(row)
                elif row.get("action") == "SELL":
                    sells.append(row)
        return buys, sells
    except Exception:
        return [], []


def _load_last_buy_dates() -> dict[str, str]:
    """读 paper_trade_log.csv 拿每 sym 最近 BUY 日期 (=入选 picks 日)."""
    if not TRADE_LOG.exists():
        return {}
    try:
        import csv
        latest: dict[str, str] = {}
        with TRADE_LOG.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("action") != "BUY":
                    continue
                sym = row.get("symbol", "")
                date = row.get("date", "")
                if sym and date:
                    # 取最大 date (后面 row 覆盖前面 = 累计最新)
                    if sym not in latest or date > latest[sym]:
                        latest[sym] = date
        return latest
    except Exception:
        return {}


def _placeholder(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{message}"
        "</div>"
    )


def _load_names() -> dict[str, str]:
    if not NAMES_PATH.exists():
        return {}
    try:
        return json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_recommended_picks_section() -> str:
    if not PICKS_TODAY.exists():
        return _placeholder(
            f"未找到 <code>{PICKS_TODAY}</code> — paper_trade_today.py 尚未跑"
        )
    try:
        d = json.loads(PICKS_TODAY.read_text(encoding="utf-8"))
    except Exception as exc:
        return _placeholder(
            f"picks_today.json 解析失败: <code>{type(exc).__name__}: {exc}</code>"
        )

    picks = d.get("picks", [])
    if not picks:
        return _placeholder(
            "picks_today.json 中无 picks — paper_trade 尚未生成今日信号"
        )

    name_map = _load_names()
    last_buy_dates = _load_last_buy_dates()
    holdings_list, holdings_date = _load_holdings()
    holdings_set = set(holdings_list)
    as_of = str(d.get("as_of_date", "?"))
    today_buys, today_sells = _load_today_actions(as_of)
    today_buy_syms = {b.get("symbol") for b in today_buys}
    today_sell_syms = {s.get("symbol") for s in today_sells}
    version = d.get("production_version", "?")
    sidecar = d.get("sidecar", {})
    sidecar_applied = sidecar.get("applied", False)
    factor = sidecar.get("factor", "?")
    lam = sidecar.get("lambda", "?")
    sign = sidecar.get("sign", "?")
    n_picks = len(picks)
    per_budget = CAPITAL_PER_BUDGET / n_picks if n_picks > 0 else 0
    equal_w = 100.0 / n_picks if n_picks > 0 else 0

    cols = [
        ("#", 3), ("代码", 7), ("名称", 10), ("状态", 9), ("推荐日期", 10),
        ("final score", 8), ("z_pred", 6), ("z_amp", 5), ("z_jzf", 5),
        ("JZF %", 5), ("现价", 8), ("止损 -8%", 6),
        ("等权 %", 5), ("预算 ¥", 8), ("手数", 5),
    ]
    assert sum(w for _, w in cols) == 100, sum(w for _, w in cols)
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _, w in cols
    ) + "</colgroup>"
    right_cols = {
        "final score", "z_pred", "z_amp", "z_jzf", "JZF %",
        "现价", "止损 -8%", "等权 %", "预算 ¥", "手数",
    }
    thead = "<tr>" + "".join(
        f'<th style="text-align:right;">{l}</th>' if l in right_cols
        else f"<th>{l}</th>"
        for l, _ in cols
    ) + "</tr>"

    body_rows: list[str] = []
    total_budget = 0.0
    for i, p in enumerate(picks, 1):
        sym = p.get("sym", "?")
        name = name_map.get(sym, "?")
        close = float(p.get("close_today", 0))
        final_score = float(p.get("final_score", 0))
        z_pred = float(p.get("z_pred", 0))
        z_amp = float(p.get("z_amp", 0)) if p.get("z_amp") is not None else 0.0
        z_jzf = float(p.get("z_jzf", 0)) if p.get("z_jzf") is not None else 0.0
        jzf_raw = p.get("jzf")
        jzf_str = f"{jzf_raw:+.2f}%" if isinstance(jzf_raw, (int, float)) else "—"
        stop_loss = close * 0.92 if close > 0 else 0
        shares = max(1, int(per_budget / close / 100)) if close > 0 else 1
        actual_budget = shares * 100 * close
        total_budget += actual_budget
        # 推荐日期: paper_trade_log 最近一次 BUY 日, fallback 用今日 as_of_date
        rec_date = last_buy_dates.get(sym, str(as_of))
        is_new_today = (rec_date == str(as_of))
        date_color = "#16a34a" if is_new_today else "var(--muted, #6b7280)"
        # 状态分类: 今日 BUY / 持仓中 / 未持仓 (model ranking 在 Top 8 但实际不持)
        if sym in today_buy_syms:
            status_label, status_color, status_bg = "🟢 今日 BUY", "#16a34a", "rgba(22,163,74,0.15)"
        elif sym in holdings_set:
            status_label, status_color, status_bg = "持有", "#f59e0b", "rgba(245,158,11,0.10)"
        else:
            status_label, status_color, status_bg = "未持仓", "#6b7280", "rgba(107,114,128,0.10)"
        body_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td><code>{html.escape(sym)}</code></td>"
            f"<td>{html.escape(name)}</td>"
            f'<td><span style="color:{status_color}; background:{status_bg}; '
            f'padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600;">'
            f'{status_label}</span></td>'
            f'<td style="color:{date_color}; font-size:11px;">{html.escape(rec_date)}</td>'
            f'<td style="text-align:right; font-weight:600;">{final_score:+.3f}</td>'
            f'<td style="text-align:right; color:var(--muted, #6b7280);">{z_pred:+.2f}</td>'
            f'<td style="text-align:right; color:var(--muted, #6b7280);">{z_amp:+.2f}</td>'
            f'<td style="text-align:right; color:var(--muted, #6b7280);">{z_jzf:+.2f}</td>'
            f'<td style="text-align:right;">{jzf_str}</td>'
            f'<td style="text-align:right; font-weight:600;">{close:.2f}</td>'
            f'<td style="text-align:right; color:#dc2626;">{stop_loss:.2f}</td>'
            f'<td style="text-align:right;">{equal_w:.1f}%</td>'
            f'<td style="text-align:right;">{actual_budget:,.0f}</td>'
            f'<td style="text-align:right;">{shares}</td>'
            "</tr>"
        )

    table = (
        '<table class="data">'
        + colgroup
        + f"<thead>{thead}</thead>"
        + f"<tbody>{''.join(body_rows)}</tbody>"
        + "</table>"
    )

    sidecar_label = (
        f"{version} sidecar: {factor} λ={lam} sign={sign}"
        if sidecar_applied else
        f"{version} no sidecar"
    )

    # ─── Banner 解释 TopK + N_DROP 机制 ───────────────────────────────
    pick_syms_set = {p.get("sym") for p in picks}
    overlap_holdings = pick_syms_set & holdings_set
    in_picks_not_held = pick_syms_set - holdings_set  # model 推荐但未持
    held_not_in_picks = holdings_set - pick_syms_set  # 持有但已跌出 Top 8

    name_lookup = lambda s: name_map.get(s, s)
    explanation_banner = (
        '<div style="background:rgba(37,99,235,0.06); border-left:3px solid #2563eb; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        '<strong>💡 TopK + N_DROP 机制说明</strong>'
        ' (一次推荐 8 只 ≠ 全买入 — 实际操作有 sticky portfolio 限制)<br>'
        '<span style="font-size:11px; line-height:1.6;">'
        '• <strong>model Top 8</strong> = 当日打分排名前 8(本表显示) — 信息参考<br>'
        f'• <strong>N_DROP={N_DROP}</strong> = 每天最多换 {N_DROP} 只 (SELL 持仓里 score 最低 {N_DROP}, '
        f'BUY 新 Top 8 里 score 最高的 {N_DROP})<br>'
        '• <strong>实际持仓</strong> = 上日持仓 + 今日 BUY - 今日 SELL (一般 75% 重叠于昨日, '
        '股票通常持有 1-2 周)<br>'
        '• <strong>换手成本</strong> ~0.20%/round-trip, '
        f'每月 ~500 元摩擦 (60 月 OOS Calmar 2.12 已 net-of-friction)<br>'
        '• <strong>实操</strong>: 跟 paper_trade_log BUY/SELL events 即可, '
        '不要每天全换 Top 8'
        '</span></div>'
    )

    # ─── 当日 actions 块 ─────────────────────────────────────────────
    actions_html = ""
    if today_buys or today_sells:
        sells_str = " · ".join(
            f"<strong>{html.escape(s.get('symbol', '?'))}</strong> "
            f"{html.escape(s.get('name', '?'))}"
            for s in today_sells
        ) or "<span style='color:var(--muted, #6b7280);'>无</span>"
        buys_str = " · ".join(
            f"<strong>{html.escape(b.get('symbol', '?'))}</strong> "
            f"{html.escape(b.get('name', '?'))}"
            f" @ {float(b.get('price', 0)):.2f}"
            for b in today_buys
        ) or "<span style='color:var(--muted, #6b7280);'>无</span>"
        actions_html = (
            '<div style="background:rgba(245,158,11,0.06); '
            'border-left:3px solid #f59e0b; padding:10px 14px; '
            'margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
            f'<strong>📋 {as_of} 真实换手 (paper_trade_log)</strong> · '
            f'SELL {len(today_sells)} · BUY {len(today_buys)} · '
            f'N_DROP 上限 {N_DROP}<br>'
            f'<span style="color:#dc2626;">SELL</span>: {sells_str}<br>'
            f'<span style="color:#16a34a;">BUY</span>: {buys_str}'
            '</div>'
        )

    # ─── 持仓 vs Top 8 对比块 ────────────────────────────────────────
    held_outside = ""
    if held_not_in_picks:
        items = ", ".join(
            f"{html.escape(s)} {html.escape(name_lookup(s))}"
            for s in sorted(held_not_in_picks)
        )
        held_outside = (
            '<div style="background:rgba(107,114,128,0.08); '
            'border-left:3px solid #6b7280; padding:8px 14px; '
            'margin-bottom:12px; border-radius:0 4px 4px 0; font-size:11px;">'
            f'<strong>📦 持仓但已跌出 Top 8 ({len(held_not_in_picks)} 只)</strong>: {items}<br>'
            '<span style="color:var(--muted, #6b7280);">'
            'N_DROP 限制下不立即 SELL, 等下一轮 ranking 再被淘汰</span>'
            '</div>'
        )

    summary = (
        '<div style="margin-top:8px; font-size:12px; color:var(--muted, #6b7280);">'
        f"model Top 8 ranking <strong>{n_picks}</strong> 只 · "
        f"实际持仓 <strong>{len(holdings_list)}</strong> 只 · "
        f"重叠 <strong>{len(overlap_holdings)}/{n_picks}</strong> "
        f"({len(overlap_holdings)/max(n_picks,1)*100:.0f}%)<br>"
        f"信号日期: <strong>{html.escape(str(as_of))}</strong> · "
        f"production: <strong>{html.escape(sidecar_label)}</strong> · "
        f"生成时间: {html.escape(str(d.get('generated_at', '?'))[:19])}"
        "</div>"
    )

    footer = (
        '<div style="margin-top:6px; font-size:11px; color:var(--muted, #6b7280);">'
        "数据源 <code>data_cache/picks_today.json</code> "
        "(paper_trade_today.py daily_check step 2 实时 dump, 含 v19.10 stacked sidecar 分量).<br>"
        "止损 -8% = close × 0.92; 手数按 100 股/手 + 50k 等权预算估算.<br>"
        "final_score = z(pred) − 0.30·z(amp_imb_20d) + 0.10·z(JZF) (v19.10 公式)."
        "</div>"
    )

    return (
        explanation_banner
        + actions_html
        + held_outside
        + table
        + summary
        + footer
    )
