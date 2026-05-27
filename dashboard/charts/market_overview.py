"""Dashboard panel — 大盘指数 + 申万行业静态指标.

数据源 (全只读, 不抢 mootdx 在跑的 fetch):
  - data_cache/index_kline.parquet — 4 个主指数 daily OHLCV
    codes: sh000001(上证), sh000300(CSI300), sz399001(深证成指), sz399006(创业板)
  - data_cache/industry/industry_boards.parquet — 申万 31 个一级行业 静态指标
"""
from __future__ import annotations

import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_PATH = ROOT / "data_cache" / "index_kline.parquet"
INDUSTRY_BOARDS_PATH = ROOT / "data_cache" / "industry" / "industry_boards.parquet"

INDEX_LABELS = {
    "sh000001": ("上证指数", "Shanghai Composite"),
    "sh000300": ("沪深 300", "CSI 300 · production universe"),
    "sz399001": ("深证成指", "Shenzhen Component"),
    "sz399006": ("创业板指", "ChiNext"),
}


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _build_index_cards() -> str:
    if not INDEX_PATH.exists():
        return ""
    try:
        import pandas as pd
        df = pd.read_parquet(INDEX_PATH)
    except Exception:
        return ""
    cards: list[str] = []
    for code, (name, en) in INDEX_LABELS.items():
        sub = df[df["code"] == code].sort_values("date")
        if len(sub) < 2:
            cards.append(
                '<div style="background:rgba(107,114,128,0.06); padding:10px; '
                'border-radius:6px; min-width:170px;">'
                f"<div style='font-weight:600;'>{html.escape(name)}</div>"
                "<div style='font-size:11px; color:var(--muted, #6b7280);'>无数据</div>"
                "</div>"
            )
            continue
        last = sub.iloc[-1]
        prev = sub.iloc[-2]
        change_pct = (float(last["close"]) / float(prev["close"]) - 1) * 100
        color = "#16a34a" if change_pct >= 0 else "#dc2626"
        last_date = str(last["date"])[:10]
        cards.append(
            f'<div style="background:rgba(107,114,128,0.06); padding:12px; '
            f'border-radius:6px; min-width:180px; flex:1; border-left:3px solid {color};">'
            f'<div style="font-weight:600; font-size:13px;">'
            f"{html.escape(name)} "
            f'<span style="color:var(--muted, #6b7280); font-weight:400; '
            f'font-size:10px;">{html.escape(code)}</span></div>'
            f'<div style="font-size:11px; color:var(--muted, #6b7280);">'
            f"{html.escape(en)}</div>"
            f'<div style="font-size:18px; font-weight:700; margin-top:4px;">'
            f'{float(last["close"]):,.2f}</div>'
            f'<div style="font-size:13px; color:{color}; font-weight:600;">'
            f"{change_pct:+.2f}%</div>"
            f'<div style="font-size:10px; color:var(--muted, #6b7280); margin-top:2px;">'
            f"as of {html.escape(last_date)}</div>"
            "</div>"
        )
    return (
        '<div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px;">'
        + "".join(cards)
        + "</div>"
    )


def _build_industry_table() -> str:
    if not INDUSTRY_BOARDS_PATH.exists():
        return (
            '<div style="font-size:11px; color:var(--muted, #6b7280);">'
            "industry_boards.parquet 不存在 — 行业表跳过."
            "</div>"
        )
    try:
        import pandas as pd
        df = pd.read_parquet(INDUSTRY_BOARDS_PATH)
    except Exception as exc:
        return (
            f'<div class="placeholder-content">行业表读取失败: '
            f"{type(exc).__name__}</div>"
        )
    df = df.sort_values("静态市盈率", ascending=False)
    cols = [
        ("#", 4), ("代码", 12), ("行业", 14), ("成份数", 9),
        ("静态 PE", 11), ("TTM PE", 11), ("PB", 9), ("股息率%", 10),
        ("估值评级", 20),
    ]
    assert sum(w for _, w in cols) == 100
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _, w in cols
    ) + "</colgroup>"
    right_cols = {"成份数", "静态 PE", "TTM PE", "PB", "股息率%"}
    thead = "<tr>" + "".join(
        f'<th style="text-align:right;">{l}</th>' if l in right_cols
        else f"<th>{l}</th>"
        for l, _ in cols
    ) + "</tr>"
    body_rows: list[str] = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        pe = float(row[3]) if row[3] is not None else 0
        if pe > 35:
            tag_lbl, tag_color = "高估", "#dc2626"
        elif pe > 25:
            tag_lbl, tag_color = "略高", "#f59e0b"
        elif pe > 15:
            tag_lbl, tag_color = "合理", "#16a34a"
        elif pe > 0:
            tag_lbl, tag_color = "低估", "#2563eb"
        else:
            tag_lbl, tag_color = "?", "#6b7280"
        body_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td><code style='font-size:10px;'>{html.escape(str(row[0]))}</code></td>"
            f"<td>{html.escape(str(row[1]))}</td>"
            f'<td style="text-align:right;">{int(row[2])}</td>'
            f'<td style="text-align:right;">{pe:.2f}</td>'
            f'<td style="text-align:right;">{float(row[4]):.2f}</td>'
            f'<td style="text-align:right;">{float(row[5]):.2f}</td>'
            f'<td style="text-align:right;">{float(row[6]):.2f}%</td>'
            f'<td><span style="color:{tag_color}; font-weight:600; font-size:11px;">'
            f"{tag_lbl}</span></td>"
            "</tr>"
        )
    return (
        '<h3 style="font-size:13px; margin:10px 0 6px 0; color:var(--muted, #6b7280);">'
        "申万 31 一级行业 (按静态 PE 倒序)"
        "</h3>"
        '<table class="data">'
        + colgroup
        + f"<thead>{thead}</thead>"
        + f"<tbody>{''.join(body_rows)}</tbody>"
        + "</table>"
    )


def build_market_overview_section() -> str:
    cards_html = _build_index_cards()
    industry_html = _build_industry_table()
    if not cards_html and not industry_html:
        return _placeholder(
            "无 index_kline.parquet + industry_boards.parquet — panel 无数据."
        )

    banner = (
        '<div style="background:rgba(37,99,235,0.06); border-left:3px solid #2563eb; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        "<strong>🌏 大盘 + 板块强弱</strong> · "
        "4 主指数最近 1 日 change% + 申万 31 行业静态估值<br>"
        '<span style="color:var(--muted, #6b7280); font-size:11px;">'
        "数据为日级 (非实时), 来自 <code>index_kline.parquet</code> + "
        "<code>industry/industry_boards.parquet</code>. 静态 PE 评级: "
        "&gt;35 高估 / 25-35 略高 / 15-25 合理 / &lt;15 低估."
        "</span></div>"
    )
    return banner + cards_html + industry_html
