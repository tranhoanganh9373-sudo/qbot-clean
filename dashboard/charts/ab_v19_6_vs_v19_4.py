"""Dashboard Stage 3 — v19.6 main vs v19.4 shadow A/B 比较.

可视化 forward A/B 跟踪 (production 主 + shadow 实时对比真实表现).

数据源 (全只读):
- data_cache/portfolio_state.json          (v19.6 main holdings)
- data_cache/portfolio_state_v19_4.json    (v19.4 shadow holdings)
- data_cache/paper_trade_log.csv           (v19.6 BUY/SELL log)
- data_cache/paper_trade_log_v19_4.csv     (v19.4 BUY/SELL log)

由于 forward shadow 才刚启动 (累积 < 5 个交易日), cum return 双线图会进入 fallback
"积累中" 提示, 但 Venn diagram + symbols 表始终可用.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from dashboard.utils import fig_to_base64
from dashboard.utils.kline_fast import get_stock_kline

ROOT = Path(__file__).resolve().parent.parent.parent
DC = ROOT / "data_cache"

STATE_MAIN = DC / "portfolio_state.json"
STATE_SHADOW = DC / "portfolio_state_v19_4.json"
LOG_MAIN = DC / "paper_trade_log.csv"
LOG_SHADOW = DC / "paper_trade_log_v19_4.csv"

MIN_DAYS_FOR_CHART = 5
COLOR_MAIN = "#2563eb"     # v19.6 蓝
COLOR_SHADOW = "#f59e0b"   # v19.4 橙
COLOR_OVERLAP = "#16a34a"  # 重叠绿


def _placeholder(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{message}"
        "</div>"
    )


def _load_holdings(path: Path) -> tuple[set[str], str]:
    """读 state json → (holdings set, as_of_date)."""
    if not path.exists():
        return set(), ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set(), ""
    return set(data.get("holdings", []) or []), str(data.get("date", "") or "")


def _venn_figure(main_h: set[str], shadow_h: set[str]) -> str:
    """画手工 2-圆 Venn (避免 matplotlib_venn 依赖), 返回 base64 PNG data URI."""
    n_main = len(main_h)
    n_shadow = len(shadow_h)
    n_common = len(main_h & shadow_h)
    n_main_only = n_main - n_common
    n_shadow_only = n_shadow - n_common

    fig, ax = plt.subplots(figsize=(7, 4))

    r = 1.4
    c_main = (-0.9, 0.0)
    c_shadow = (0.9, 0.0)

    circle_main = mpatches.Circle(
        c_main, r, facecolor=COLOR_MAIN, edgecolor=COLOR_MAIN,
        alpha=0.35, linewidth=2,
    )
    circle_shadow = mpatches.Circle(
        c_shadow, r, facecolor=COLOR_SHADOW, edgecolor=COLOR_SHADOW,
        alpha=0.35, linewidth=2,
    )
    ax.add_patch(circle_main)
    ax.add_patch(circle_shadow)

    # 区域数字: 左唯一 / 重叠 / 右唯一
    ax.text(-1.5, 0.0, str(n_main_only), ha="center", va="center",
            fontsize=22, fontweight="bold", color="#1a1a1a")
    ax.text(0.0, 0.0, str(n_common), ha="center", va="center",
            fontsize=22, fontweight="bold", color="#1a1a1a")
    ax.text(1.5, 0.0, str(n_shadow_only), ha="center", va="center",
            fontsize=22, fontweight="bold", color="#1a1a1a")

    # Label
    ax.text(-1.5, 1.65, f"v19.6 main ({n_main})",
            ha="center", va="bottom", fontsize=12,
            color=COLOR_MAIN, fontweight="bold")
    ax.text(1.5, 1.65, f"v19.4 shadow ({n_shadow})",
            ha="center", va="bottom", fontsize=12,
            color=COLOR_SHADOW, fontweight="bold")
    ax.text(0.0, -1.85, f"overlap = {n_common}",
            ha="center", va="top", fontsize=11,
            color=COLOR_OVERLAP, fontweight="bold")

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    uri = fig_to_base64(fig)
    plt.close(fig)
    return uri


def _symbols_table(main_h: set[str], shadow_h: set[str]) -> str:
    """3-列 symbol 分类表 HTML."""
    common = sorted(main_h & shadow_h)
    main_only = sorted(main_h - shadow_h)
    shadow_only = sorted(shadow_h - main_h)

    from dashboard.utils.stock_names import code_with_name
    def _cell(symbols: list[str]) -> str:
        if not symbols:
            return '<span style="color:var(--muted);">—</span>'
        return "<br>".join(html.escape(code_with_name(s)) for s in symbols)

    return (
        '<table class="data" style="margin-top:12px;">'
        "<thead><tr>"
        f'<th style="color:{COLOR_MAIN};">v19.6 only ({len(main_only)})</th>'
        f'<th style="color:{COLOR_OVERLAP};">overlap ({len(common)})</th>'
        f'<th style="color:{COLOR_SHADOW};">v19.4 only ({len(shadow_only)})</th>'
        "</tr></thead>"
        "<tbody><tr>"
        f'<td style="vertical-align:top;">{_cell(main_only)}</td>'
        f'<td style="vertical-align:top;">{_cell(common)}</td>'
        f'<td style="vertical-align:top;">{_cell(shadow_only)}</td>'
        "</tr></tbody>"
        "</table>"
    )


def _load_log(path: Path) -> pd.DataFrame:
    """读 paper_trade_log csv, 返回空 DF 若文件 missing / unreadable."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or not {"date", "action", "symbol"}.issubset(df.columns):
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date"]).reset_index(drop=True)


def _replay_positions(log: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    """从 BUY/SELL log 重放每日 EOD 持仓 set. 同日先 SELL 后 BUY."""
    positions: set[str] = set()
    snapshots: dict[pd.Timestamp, set[str]] = {}
    log = log.copy()
    log["_prio"] = log["action"].map({"SELL": 0, "BUY": 1}).fillna(2)
    log = log.sort_values(["date", "_prio"]).reset_index(drop=True)
    for date, day in log.groupby("date"):
        for _, r in day.iterrows():
            sym = str(r["symbol"])
            if r["action"] == "BUY":
                positions.add(sym)
            elif r["action"] == "SELL":
                positions.discard(sym)
        snapshots[date] = set(positions)
    return snapshots


def _build_daily_cum_curve(log: pd.DataFrame) -> pd.DataFrame:
    """等权重 daily cum return, 按 kline close MTM.

    返回列: date, cum_return (decimal). 空 DF 若 log 不足 / kline 全缺.
    """
    if log.empty:
        return pd.DataFrame()
    snapshots = _replay_positions(log)
    if not snapshots:
        return pd.DataFrame()

    log_dates = sorted(snapshots.keys())
    start = log_dates[0]
    end = log_dates[-1]
    if end <= start:
        return pd.DataFrame()

    # 取任意一只有 kline 的股做 trading-day calendar
    calendar_dates: pd.DatetimeIndex | None = None
    for sym in log["symbol"].drop_duplicates().tolist():
        df = get_stock_kline(sym)
        if df.empty:
            continue
        mask = (df["date"] >= start) & (df["date"] <= end)
        sub = df.loc[mask, "date"]
        if not sub.empty:
            calendar_dates = pd.DatetimeIndex(sorted(sub.unique()))
            break
    if calendar_dates is None or len(calendar_dates) == 0:
        return pd.DataFrame()

    # 预读所有相关股 close series
    syms = sorted({s for snap in snapshots.values() for s in snap}
                  | {str(s) for s in log["symbol"].drop_duplicates()})
    kline_map: dict[str, pd.Series] = {}
    for sym in syms:
        df = get_stock_kline(sym)
        if df.empty:
            continue
        kline_map[sym] = df.set_index("date")["close"].astype(float)

    if not kline_map:
        return pd.DataFrame()

    snap_dates_sorted = sorted(snapshots.keys())

    def _positions_at(d: pd.Timestamp) -> set[str]:
        valid = [sd for sd in snap_dates_sorted if sd <= d]
        if not valid:
            return set()
        return snapshots[valid[-1]]

    daily_ret = []
    prev_close: dict[str, float] = {}
    for i, d in enumerate(calendar_dates):
        pos = _positions_at(d)
        rets = []
        for sym in pos:
            if sym not in kline_map:
                continue
            s = kline_map[sym]
            if d not in s.index:
                continue
            cur = float(s.loc[d])
            if sym in prev_close and prev_close[sym] > 0:
                rets.append(cur / prev_close[sym] - 1.0)
            prev_close[sym] = cur
        if i == 0 or not rets:
            daily_ret.append(0.0)
        else:
            daily_ret.append(sum(rets) / len(rets))

    cum = (pd.Series(daily_ret) + 1.0).cumprod() - 1.0
    return pd.DataFrame({"date": calendar_dates, "cum_return": cum.values})


def _cum_chart(curve_main: pd.DataFrame, curve_shadow: pd.DataFrame) -> str:
    """双线 cum return chart, 返回 base64 PNG data URI."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if not curve_main.empty:
        ax.plot(curve_main["date"], curve_main["cum_return"] * 100,
                color=COLOR_MAIN, linewidth=2, marker="o", markersize=4,
                label=f"v19.6 main ({len(curve_main)}d)")
    if not curve_shadow.empty:
        ax.plot(curve_shadow["date"], curve_shadow["cum_return"] * 100,
                color=COLOR_SHADOW, linewidth=2, marker="s", markersize=4,
                label=f"v19.4 shadow ({len(curve_shadow)}d)")
    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Trading Day")
    ax.set_title("v19.6 main vs v19.4 shadow — forward equal-weight cum return",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    uri = fig_to_base64(fig)
    plt.close(fig)
    return uri


def build_ab_section() -> str:
    """主入口: v19.6 main vs v19.4 shadow 对比 HTML fragment.

    包含: Venn diagram + symbols 分类表 + 双 cum 线图 (或 fallback).
    """
    main_h, main_date = _load_holdings(STATE_MAIN)
    shadow_h, shadow_date = _load_holdings(STATE_SHADOW)

    if not main_h and not shadow_h:
        return _placeholder(
            "未找到 portfolio_state.json / portfolio_state_v19_4.json, "
            "shadow 跟踪尚未启动."
        )

    venn_uri = _venn_figure(main_h, shadow_h)
    venn_html = f'<img src="{venn_uri}" alt="v19.6 vs v19.4 holdings Venn"/>'

    table_html = _symbols_table(main_h, shadow_h)

    meta_html = (
        '<div style="margin-top:8px; font-size:12px; color:var(--muted);">'
        f"v19.6 state as_of <b>{html.escape(main_date or 'n/a')}</b> "
        f"({len(main_h)} stocks) &middot; "
        f"v19.4 shadow state as_of <b>{html.escape(shadow_date or 'n/a')}</b> "
        f"({len(shadow_h)} stocks)"
        "</div>"
    )

    log_main = _load_log(LOG_MAIN)
    log_shadow = _load_log(LOG_SHADOW)

    main_days = log_main["date"].nunique() if not log_main.empty else 0
    shadow_days = log_shadow["date"].nunique() if not log_shadow.empty else 0
    forward_days = max(main_days, shadow_days)

    if forward_days < MIN_DAYS_FOR_CHART:
        chart_html = _placeholder(
            f"双 cum return 曲线 — 积累中 <b>{forward_days} / "
            f"{MIN_DAYS_FOR_CHART}</b> 交易日<br>"
            f"v19.6 log: {main_days} log-day / {len(log_main)} actions &middot; "
            f"v19.4 log: {shadow_days} log-day / {len(log_shadow)} actions<br>"
            f"<span style='font-size:11px;'>"
            f"累积 ≥ {MIN_DAYS_FOR_CHART} 交易日后将自动切换为双线 MTM 走势."
            "</span>"
        )
    else:
        try:
            curve_main = _build_daily_cum_curve(log_main)
            curve_shadow = _build_daily_cum_curve(log_shadow)
        except Exception as exc:  # noqa: BLE001
            return (
                venn_html + meta_html + table_html
                + _placeholder(
                    "cum curve build failed: "
                    f"<code>{html.escape(type(exc).__name__)}: "
                    f"{html.escape(str(exc))}</code>"
                )
            )
        if curve_main.empty and curve_shadow.empty:
            chart_html = _placeholder(
                "两条 log 都无法构建 cum curve (kline 缺数据)."
            )
        else:
            chart_uri = _cum_chart(curve_main, curve_shadow)
            chart_html = f'<img src="{chart_uri}" alt="v19.6 vs v19.4 cum return"/>'

    return (
        '<div style="margin-bottom:12px;">'
        f"{venn_html}"
        f"{meta_html}"
        f"{table_html}"
        "</div>"
        f"{chart_html}"
    )
