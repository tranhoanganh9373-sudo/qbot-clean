"""Dashboard panel — 影子 v19.4 paper_trade 模拟交易历史.

显示 paper_trade_v19_4.py (v19.4 shadow,与 v19.6 production 平行跑) 的全 BUY/SELL 记录.
v19.4 是 forward OOS A/B shadow,用于跟 v19.6 production 对比真实 paper trading 效果.

数据源 (只读):
  - data_cache/paper_trade_log_v19_4.csv (schema: date,action,symbol,name,score,price)
  - 由 examples/paper_trade_v19_4.py 每日 daily_check step 3.5 append, 不删不改.

显示:
  全部 trades 列表 (倒序), 列: 日期 / 方向 / 代码 / 名称 / score / 价格 / 距今
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = ROOT / "data_cache" / "paper_trade_log_v19_4.csv"

COLOR_BUY = "#16a34a"
COLOR_SELL = "#dc2626"

MAX_ROWS = 50


def _placeholder(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{message}"
        "</div>"
    )


def build_shadow_paper_trade_section() -> str:
    if not LOG_PATH.exists():
        return _placeholder(
            f"未找到 <code>{LOG_PATH}</code>. "
            "影子 v19.4 paper_trade 尚未生成 (daily_check step 3.5)."
        )
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception as exc:  # noqa: BLE001
        return _placeholder(f"读取 v19.4 log 失败: <code>{type(exc).__name__}: {exc}</code>")
    if df.empty:
        return _placeholder("影子 v19.4 paper_trade log 为空.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "action"])
    df = df.sort_values("date", ascending=False).head(MAX_ROWS).reset_index(drop=True)

    cols = [
        ("日期", 12),
        ("方向", 8),
        ("代码", 14),
        ("名称", 22),
        ("Score", 14),
        ("价格", 14),
        ("距今", 16),
    ]
    assert sum(w for _, w in cols) == 100
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _, w in cols
    ) + "</colgroup>"
    thead = "".join(f"<th>{label}</th>" for label, _ in cols)

    today = pd.Timestamp.now().normalize()
    rows_html: list[str] = []
    for _, r in df.iterrows():
        action = str(r["action"]).upper()
        color = COLOR_BUY if action == "BUY" else COLOR_SELL
        sym = str(r["symbol"])
        name = str(r.get("name", "") or "")
        try:
            score = f"{float(r['score']):.4f}"
        except (TypeError, ValueError):
            score = "-"
        try:
            price = f"{float(r['price']):.2f}"
        except (TypeError, ValueError):
            price = "-"
        date_str = r["date"].strftime("%Y-%m-%d")
        days_ago = (today - r["date"]).days
        days_str = "今日" if days_ago == 0 else f"{days_ago} 天前"
        rows_html.append(
            "<tr>"
            f"<td>{date_str}</td>"
            f"<td style='color:{color}; font-weight:600;'>{action}</td>"
            f"<td>{html.escape(sym)}</td>"
            f"<td>{html.escape(name)}</td>"
            f"<td style='text-align:right;'>{score}</td>"
            f"<td style='text-align:right;'>{price}</td>"
            f"<td style='color:var(--muted, #6b7280);'>{days_str}</td>"
            "</tr>"
        )
    tbody = "".join(rows_html)

    total = len(df)
    n_buy = int((df["action"].str.upper() == "BUY").sum())
    n_sell = int((df["action"].str.upper() == "SELL").sum())
    summary = (
        '<div style="margin-top:8px; font-size:12px; color:var(--muted, #6b7280);">'
        f"显示最近 {total} 笔 (BUY {n_buy} / SELL {n_sell}). "
        f"数据源 <code>data_cache/paper_trade_log_v19_4.csv</code> · "
        "由 <code>examples/paper_trade_v19_4.py</code> 每日 daily_check step 3.5 append. "
        "影子 v19.4 (m5+m20 sidecar) 跟 v19.6 production (amp_imb_20d) 平行跑, "
        "用于 forward OOS A/B 对比."
        "</div>"
    )

    return (
        '<table class="data">'
        + colgroup
        + f"<thead><tr>{thead}</tr></thead>"
        + f"<tbody>{tbody}</tbody>"
        + "</table>"
        + summary
    )
