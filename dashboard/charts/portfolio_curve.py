"""Dashboard panel — 持仓配置饼图 + 累计 PnL 曲线.

数据源:
  - data_cache/trades.jsonl (via trades_log.load_trades + aggregate_positions)
  - data_cache/baidu_kline.parquet (历史每日 close, 用于 mark-to-market)

视觉:
  - 左饼图: 当前持仓 (net_shares > 0) by market value 占比
  - 右曲线: daily portfolio value vs cumulative net cost,填充区域 = 浮动 PnL
  - trades < 2 dates 时退化为只显示饼图 + 数字摘要
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dashboard.utils import fig_to_base64  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from claude_finance.trades_log import aggregate_positions, load_trades  # noqa: E402

KLINE_PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"

COLOR_VALUE = "#2563eb"
COLOR_COST = "#94a3b8"
COLOR_GAIN = "#16a34a"
COLOR_LOSS = "#dc2626"


def _placeholder(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{message}"
        "</div>"
    )


def _strip_prefix(sym: str) -> str:
    return sym[2:] if sym[:2] in ("SH", "SZ") else sym


def _load_kline_subset(syms: set[str]) -> pd.DataFrame:
    """读 baidu_kline 仅 涉及 syms 的 close 数据."""
    if not KLINE_PARQUET.exists() or not syms:
        return pd.DataFrame()
    codes = {_strip_prefix(s) for s in syms}
    df = pd.read_parquet(KLINE_PARQUET, columns=["code", "date", "close"])
    df = df[df["code"].isin(codes)].copy()
    df["sym"] = df["code"].apply(
        lambda c: ("SH" if str(c)[0] in ("6", "9") else "SZ") + str(c).zfill(6)
    )
    return df[["sym", "date", "close"]].sort_values(["sym", "date"])


def _build_daily_curve(trades: list[dict], kline: pd.DataFrame) -> pd.DataFrame:
    """重建每日 portfolio value + cumulative net cost.

    返回 DataFrame columns: date / market_value / cost_basis / pnl.
    """
    if not trades or kline.empty:
        return pd.DataFrame()
    tdf = pd.DataFrame(trades)
    tdf["date"] = pd.to_datetime(tdf["date"])
    tdf["shares_signed"] = tdf.apply(
        lambda r: r["shares"] if r["action"] == "BUY" else -r["shares"], axis=1
    )
    tdf["cash_delta"] = tdf.apply(
        lambda r: -r["price"] * r["shares"] if r["action"] == "BUY"
        else r["price"] * r["shares"], axis=1
    )

    daily_cash = tdf.groupby("date")["cash_delta"].sum().sort_index()
    cum_cash = daily_cash.cumsum()
    cost_basis_series = -cum_cash  # 净投入 (正数)

    start_date = tdf["date"].min()
    today = pd.Timestamp.now().normalize()
    date_range = pd.date_range(start_date, today, freq="D")

    sym_value_frames = []
    for sym in tdf["sym"].unique():
        sym_trades = tdf[tdf["sym"] == sym].sort_values("date")
        sym_daily_delta = sym_trades.groupby("date")["shares_signed"].sum()
        sym_cumul = sym_daily_delta.cumsum()
        sym_cumul_full = sym_cumul.reindex(date_range, method="ffill").fillna(0)

        sym_kline = kline[kline["sym"] == sym].set_index("date")["close"]
        if sym_kline.empty:
            continue
        sym_close = sym_kline.reindex(date_range, method="ffill")
        sym_value = sym_cumul_full * sym_close
        sym_value_frames.append(
            pd.DataFrame({"date": date_range, "sym": sym, "value": sym_value.values})
        )

    if not sym_value_frames:
        return pd.DataFrame()
    all_value = pd.concat(sym_value_frames)
    daily_value = all_value.groupby("date")["value"].sum()

    cost_basis_full = cost_basis_series.reindex(date_range, method="ffill").fillna(0)
    result = pd.DataFrame({
        "date": date_range,
        "market_value": daily_value.values,
        "cost_basis": cost_basis_full.values,
    })
    result["pnl"] = result["market_value"] - result["cost_basis"]
    return result


def _plot(positions: dict, curve: pd.DataFrame, kline: pd.DataFrame) -> str:
    """matplotlib 渲染左饼图 + 右曲线 → base64 image."""
    has_curve = not curve.empty and len(curve) >= 2

    if has_curve:
        fig, (ax_pie, ax_line) = plt.subplots(
            1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 2]}
        )
    else:
        # 退化模式 (< 2 数据点): 只画饼图, figsize 缩小
        fig, ax_pie = plt.subplots(1, 1, figsize=(3.5, 3.5))
        ax_line = None

    # 当前持仓饼图: 用最新 close × net_shares 作为 market value (准确)
    holdings = [(sym, p) for sym, p in positions.items() if p.get("net_shares", 0) > 0]
    if holdings:
        labels: list[str] = []
        values: list[float] = []
        for sym, p in holdings:
            sym_kline = kline[kline["sym"] == sym].sort_values("date")
            latest_close = float(sym_kline["close"].iloc[-1]) if not sym_kline.empty else 0
            if latest_close <= 0:
                latest_close = float(p.get("weighted_avg_cost") or 0)
            mv = latest_close * p.get("net_shares", 0)
            labels.append(f"{sym[2:]} {p.get('name','')}"[:12])
            values.append(max(mv, 1))
        ax_pie.pie(
            values, labels=labels, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 9},
            colors=plt.cm.tab10.colors[: len(values)],
        )
        ax_pie.set_title(f"持仓配置 ({len(holdings)} 只)", fontsize=11)
    else:
        ax_pie.text(0.5, 0.5, "暂无持仓", ha="center", va="center",
                    transform=ax_pie.transAxes, fontsize=12, color="#6b7280")
        ax_pie.set_xticks([])
        ax_pie.set_yticks([])

    if has_curve and ax_line is not None:
        ax_line.plot(curve["date"], curve["market_value"],
                     color=COLOR_VALUE, linewidth=1.8, label="市值")
        ax_line.plot(curve["date"], curve["cost_basis"],
                     color=COLOR_COST, linewidth=1.2, linestyle="--", label="成本基础")
        positive_mask = curve["market_value"] >= curve["cost_basis"]
        ax_line.fill_between(
            curve["date"], curve["cost_basis"], curve["market_value"],
            where=positive_mask, color=COLOR_GAIN, alpha=0.20, label="浮盈",
        )
        ax_line.fill_between(
            curve["date"], curve["cost_basis"], curve["market_value"],
            where=~positive_mask, color=COLOR_LOSS, alpha=0.20, label="浮亏",
        )
        ax_line.set_title(
            f"累计 PnL 曲线 (mark-to-market, "
            f"{curve['date'].min().date()} → {curve['date'].max().date()})",
            fontsize=11,
        )
        ax_line.set_ylabel("¥")
        ax_line.legend(fontsize=9, loc="upper left")
        ax_line.grid(True, alpha=0.2)

    fig.tight_layout()
    uri = fig_to_base64(fig)
    plt.close(fig)
    return uri


def build_portfolio_curve_section() -> str:
    trades = load_trades()
    if not trades:
        return _placeholder(
            "无交易记录 — trades.jsonl 为空. "
            "在上方 'Trade Entry' 录入第一笔买入后即可看到饼图 + PnL 曲线."
        )

    positions = aggregate_positions(trades)
    all_syms = set(positions.keys())
    kline = _load_kline_subset(all_syms)
    curve = _build_daily_curve(trades, kline)

    try:
        img_uri = _plot(positions, curve, kline)
    except Exception as exc:  # noqa: BLE001
        return _placeholder(
            f"绘图失败: <code>{type(exc).__name__}: {html.escape(str(exc))}</code>"
        )
    has_curve = not curve.empty and len(curve) >= 2
    # 退化模式 (单饼图) 限制最大宽度 25% 防止 panel full-width 撑大
    img_style = "" if has_curve else 'style="max-width:25%; height:auto; display:block;"'
    img_html = f'<img src="{img_uri}" alt="持仓配置 + PnL 曲线" {img_style}/>'

    if not curve.empty:
        latest = curve.iloc[-1]
        latest_mv = latest["market_value"]
        latest_cost = latest["cost_basis"]
        latest_pnl = latest["pnl"]
        pnl_pct = (latest_pnl / latest_cost * 100) if latest_cost else 0
        n_days = len(curve)
        pnl_color = "#16a34a" if latest_pnl >= 0 else "#dc2626"
        summary = (
            '<div style="margin-top:8px; font-size:12px; color:var(--muted, #6b7280);">'
            f"今日 ({latest['date'].date()}) 持仓: 市值 ¥{latest_mv:,.0f} / "
            f"成本 ¥{latest_cost:,.0f} / "
            f"<strong style='color:{pnl_color};'>"
            f"浮动 ¥{latest_pnl:+,.0f} ({pnl_pct:+.2f}%)</strong> · "
            f"曲线 {n_days} 个日历日 (含周末 ffill)"
            "</div>"
        )
    else:
        holding_n = sum(1 for p in positions.values() if p.get("net_shares", 0) > 0)
        summary = (
            '<div style="margin-top:8px; font-size:12px; color:var(--muted, #6b7280);">'
            f"持仓 {holding_n} 只 · baidu_kline 缺相关股 close 数据"
            "</div>"
        )
    return img_html + summary
