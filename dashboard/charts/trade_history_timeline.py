"""Dashboard Section 11 — Trade history timeline (用户真实交易).

可视化 trades.jsonl 全 BUY/SELL events,看用户买卖时间线 +
哪些 stocks 实际交易过。

数据源 (全只读):
- data_cache/trades.jsonl (schema: id/date/sym/name/action/price/shares/note/recorded_at)
- dashboard/utils/kline_fast.get_stock_kline(sym) — Hive 分区单股 kline

视觉:
- 上半 scatter: x=date, y=stock, marker=^ (BUY) / v (SELL), color=green/red.
- 下半 line: top 6 交易 stocks close 路径 (normalized 起点=1.0) + 实际 BUY/SELL 标记.
- 下方 summary table 列每个涉及 stock 的 BUY/SELL 次数与最后 action.

注意:
- trades.jsonl 由 dashboard_submit_server append, append-only 用户真实交易记录.
- dashboard 仅读, 绝不写 / 改.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from dashboard.utils import fig_to_base64  # noqa: E402
from dashboard.utils.kline_fast import get_stock_kline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from claude_finance.trades_log import load_trades  # noqa: E402

DC = ROOT / "data_cache"

# Top-N picked stocks 显示 close 路径
TOP_LINE_N = 6
# Summary table 最多显示行数
TABLE_MAX_ROWS = 15
# 下半 close 路径横轴起点提前 N 天 (留 buffer)
HORIZON_LEAD_DAYS = 10

COLOR_BUY = "#27ae60"
COLOR_SELL = "#e74c3c"


def _placeholder(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{message}"
        "</div>"
    )


def _load_log() -> pd.DataFrame:
    """读 trades.jsonl → DataFrame (映射到旧 paper_trade_log column 命名以复用下游).

    schema:
      sym  → symbol
      name, action, price, shares, note 保留
      date 转 pd.Timestamp
      recorded_at 也保留 (timestamp 排序用)
    """
    trades = load_trades()
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    if "sym" in df.columns:
        df = df.rename(columns={"sym": "symbol"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "action"])
    return df.sort_values(["recorded_at", "date"]).reset_index(drop=True)


def _build_timeline_fig(df: pd.DataFrame, unique_syms: list[str]) -> plt.Figure:
    """两块 subplot: scatter timeline + top-N close 路径."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    # === 上半 scatter ===
    sym_to_idx = {s: i for i, s in enumerate(unique_syms)}
    for _, row in df.iterrows():
        y = sym_to_idx.get(row["symbol"], -1)
        if y < 0:
            continue
        is_buy = str(row["action"]).upper() == "BUY"
        ax1.scatter(
            row["date"], y,
            c=COLOR_BUY if is_buy else COLOR_SELL,
            marker="^" if is_buy else "v",
            s=80, alpha=0.85,
            edgecolors="black", linewidths=0.5,
        )

    from dashboard.utils.stock_names import code_with_name
    ax1.set_yticks(range(len(unique_syms)))
    ax1.set_yticklabels([code_with_name(s) for s in unique_syms], fontsize=8)
    ax1.set_title(
        f"Trade Events Timeline ({len(df)} events, {len(unique_syms)} stocks)",
        fontsize=12,
    )
    ax1.grid(True, alpha=0.2)
    legend_elems = [
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=COLOR_BUY, markersize=10, label="BUY",
               markeredgecolor="black"),
        Line2D([0], [0], marker="v", color="w",
               markerfacecolor=COLOR_SELL, markersize=10, label="SELL",
               markeredgecolor="black"),
    ]
    ax1.legend(handles=legend_elems, loc="upper right", fontsize=9)

    # === 下半 close 路径 ===
    sym_counts = df.groupby("symbol").size().sort_values(ascending=False)
    top_syms = sym_counts.head(TOP_LINE_N).index.tolist()

    log_min_date = df["date"].min()
    horizon_start = log_min_date - pd.Timedelta(days=HORIZON_LEAD_DAYS)

    plotted = 0
    for sym in top_syms:
        kline = get_stock_kline(sym)
        if kline.empty:
            continue
        sub = kline[kline["date"] >= horizon_start].copy()
        if sub.empty:
            continue
        first_close = float(sub["close"].iloc[0])
        if first_close <= 0:
            continue
        sub["close_norm"] = sub["close"] / first_close
        ax2.plot(
            sub["date"], sub["close_norm"],
            label=f"{sym} ({int(sym_counts[sym])}×)",
            alpha=0.75, linewidth=1.5,
        )
        # 标 BUY/SELL on this stock 的 events
        sym_events = df[df["symbol"] == sym]
        for _, r in sym_events.iterrows():
            is_buy = str(r["action"]).upper() == "BUY"
            close_today = sub[sub["date"].dt.date == r["date"].date()]
            if close_today.empty:
                continue
            ax2.scatter(
                r["date"], close_today["close_norm"].iloc[0],
                c=COLOR_BUY if is_buy else COLOR_SELL,
                marker="^" if is_buy else "v",
                s=80, edgecolors="black", linewidths=0.5, zorder=5,
            )
        plotted += 1

    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Close (normalized to start = 1.0)")
    ax2.set_title(
        f"Top {plotted} Picked Stocks — Close 路径 + 实际 BUY/SELL 标记",
        fontsize=12,
    )
    if plotted > 0:
        ax2.legend(fontsize=8, loc="upper left", ncol=2)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


def _build_summary_rows(df: pd.DataFrame, unique_syms: list[str]) -> str:
    """Summary 表 HTML rows (按 last-event 日期降序, 最多 TABLE_MAX_ROWS 行)."""
    rows = []
    last_event = df.groupby("symbol")["date"].max()
    order = last_event.sort_values(ascending=False).index.tolist()
    for sym in order[:TABLE_MAX_ROWS]:
        sub = df[df["symbol"] == sym]
        n_buy = int((sub["action"].str.upper() == "BUY").sum())
        n_sell = int((sub["action"].str.upper() == "SELL").sum())
        name_series = sub["name"].dropna() if "name" in sub.columns else pd.Series(dtype=object)
        name = str(name_series.iloc[0]) if not name_series.empty else "?"
        last_action = str(sub.iloc[-1]["action"]).upper()
        last_date = sub.iloc[-1]["date"].strftime("%Y-%m-%d")
        action_color = COLOR_BUY if last_action == "BUY" else COLOR_SELL
        rows.append(
            "<tr>"
            f"<td>{html.escape(sym)}</td>"
            f"<td>{html.escape(name)}</td>"
            f"<td style='text-align:right'>{n_buy}</td>"
            f"<td style='text-align:right'>{n_sell}</td>"
            f"<td style='color:{action_color};font-weight:600'>{last_action}</td>"
            f"<td>{last_date}</td>"
            "</tr>"
        )
    return "".join(rows)


def build_trade_timeline_section() -> str:
    """Section 11 入口 — 渲染 trade history timeline 完整 HTML 片段."""
    df = _load_log()
    if df.empty:
        return _placeholder(
            "trades.jsonl 未找到 / 空 — 在上方 'Trade Entry' 录入第一笔买卖即可."
        )

    unique_syms = sorted(df["symbol"].dropna().unique().tolist())
    if not unique_syms:
        return _placeholder("trades.jsonl 无有效 symbol")

    try:
        fig = _build_timeline_fig(df, unique_syms)
        uri = fig_to_base64(fig)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        return _placeholder(
            "绘制 timeline 失败: "
            f"<code>{type(exc).__name__}: {html.escape(str(exc))}</code>"
        )

    img_html = f'<img src="{uri}" alt="Trade history timeline"/>'

    summary_rows = _build_summary_rows(df, unique_syms)
    table_html = (
        '<table class="data" style="margin-top:12px;">'
        "<thead><tr>"
        "<th>代码</th><th>名称</th>"
        "<th style='text-align:right'>BUY 次</th>"
        "<th style='text-align:right'>SELL 次</th>"
        "<th>最后 action</th><th>最后日期</th>"
        "</tr></thead>"
        f"<tbody>{summary_rows}</tbody>"
        "</table>"
    )

    n_events = len(df)
    n_stocks = len(unique_syms)
    explanation = (
        '<div class="placeholder-content" '
        'style="margin-top:12px;text-align:left;border:none;'
        'padding:8px 0;color:var(--muted);">'
        "▲ 绿 = BUY 买入, ▼ 红 = SELL 卖出. 上半 scatter: 全部交易时间轴. "
        f"下半 line: 最常交易 {TOP_LINE_N} 只股 close 路径 "
        "(归一化起点 = 1.0) + 实际 BUY/SELL 标记. "
        f"累计 {n_events} 笔交易 / {n_stocks} 只股票. "
        "数据源 <code>data_cache/trades.jsonl</code> — append-only, 用户在 Trade Entry 录入."
        "</div>"
    )

    return img_html + table_html + explanation
