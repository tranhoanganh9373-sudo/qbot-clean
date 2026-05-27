"""Dashboard panel — Daily PnL calendar heatmap.

每个方格 = 1 个交易日, 颜色深浅 = daily PnL 大小 (绿正红负).
GitHub-style 7-row × N-week 日历布局.

数据源 (全只读):
  - data_cache/paper_trade_log.csv (date, action, sym, price) → 构建 daily 持仓 timeline
  - data_cache/baidu_kline.parquet (code, date, close) → mark-to-market daily delta
"""
from __future__ import annotations

import base64
import io
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
TRADE_LOG = ROOT / "data_cache" / "paper_trade_log.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _sym_to_code(sym: str) -> str:
    return sym[2:] if len(sym) >= 8 and sym[:2] in ("SH", "SZ") else sym


def _compute_daily_pnl():
    if not TRADE_LOG.exists():
        return None
    try:
        import pandas as pd
        trades = pd.read_csv(TRADE_LOG)
        trades["date"] = pd.to_datetime(trades["date"])
        kline = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
        kline["date"] = pd.to_datetime(kline["date"])
    except Exception:
        return None
    if len(trades) == 0:
        return None

    syms_traded = sorted(set(trades["symbol"]))
    qty_by_date: dict = {}
    cur: dict[str, int] = {s: 0 for s in syms_traded}
    trade_dates = sorted(trades["date"].unique())
    for d in trade_dates:
        day_trades = trades[trades["date"] == d]
        for _, t in day_trades.iterrows():
            sym = t["symbol"]
            if t["action"] == "BUY":
                cur[sym] = cur.get(sym, 0) + 1
            elif t["action"] == "SELL":
                cur[sym] = cur.get(sym, 0) - 1
        qty_by_date[d] = dict(cur)

    import pandas as pd
    last_trade_date = max(trade_dates)
    today = pd.Timestamp.now().normalize()
    end_date = max(last_trade_date, today)
    cutoff_start = min(trade_dates) - timedelta(days=1)

    codes = {_sym_to_code(s) for s in syms_traded}
    sub = kline[
        kline["code"].isin(codes)
        & (kline["date"] >= cutoff_start)
        & (kline["date"] <= end_date)
    ].sort_values(["code", "date"])
    if sub.empty:
        return None
    sub["prev_close"] = sub.groupby("code", observed=True)["close"].shift(1)
    sub = sub.dropna(subset=["prev_close"])
    sub["delta_close"] = sub["close"] - sub["prev_close"]
    sub["sym"] = sub["code"].apply(
        lambda c: ("SH" if str(c).startswith(("6", "688", "689"))
                   else "SZ") + str(c).zfill(6)
    )

    pnl_by_date: dict = {}
    sorted_trade_dates = sorted(qty_by_date.keys())

    def qty_at_date(d):
        applicable = [td for td in sorted_trade_dates if td <= d]
        if not applicable:
            return {}
        return qty_by_date[applicable[-1]]

    for d, day_sub in sub.groupby("date"):
        qty_map = qty_at_date(d - timedelta(days=1))
        pnl = 0.0
        for _, r in day_sub.iterrows():
            q = qty_map.get(r["sym"], 0)
            if q > 0:
                pnl += q * r["delta_close"] * 100
        pnl_by_date[d] = pnl
    return pnl_by_date


def _render_heatmap_png(pnl_by_date: dict) -> str:
    if not pnl_by_date:
        return ""
    dates = sorted(pnl_by_date.keys())
    if len(dates) < 2:
        return ""
    pnls = [pnl_by_date[d] for d in dates]
    max_abs = max(abs(p) for p in pnls) or 1.0

    weeks: dict = {}
    min_d = dates[0]
    for d in dates:
        wkday = d.weekday()
        week_idx = (d - min_d).days // 7
        weeks[(wkday, week_idx)] = pnl_by_date[d]
    n_weeks = max(k[1] for k in weeks.keys()) + 1

    fig, ax = plt.subplots(figsize=(12, 2.4), dpi=90)
    for (wkday, wk), pnl in weeks.items():
        if abs(pnl) < 1e-9:
            color = "#374151"
        elif pnl > 0:
            alpha = min(abs(pnl) / max_abs, 1.0)
            color = (0.086, 0.639, 0.290, 0.2 + alpha * 0.8)
        else:
            alpha = min(abs(pnl) / max_abs, 1.0)
            color = (0.863, 0.149, 0.149, 0.2 + alpha * 0.8)
        ax.add_patch(plt.Rectangle(
            (wk, 6 - wkday), 0.9, 0.9, facecolor=color, edgecolor="none",
        ))

    ax.set_xlim(-0.5, n_weeks + 0.5)
    ax.set_ylim(-0.5, 7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([6 - i for i in range(7)])
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                       fontsize=8, color="#9ca3af")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, colors="#9ca3af")
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1,
                facecolor="none", transparent=True)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="daily pnl heatmap" '\
           f'style="display:block; max-width:100%; height:auto;"/>'


def build_daily_pnl_heatmap_section() -> str:
    pnl_by_date = _compute_daily_pnl()
    if pnl_by_date is None:
        return _placeholder(
            "无 paper_trade_log.csv 数据 — 至少需 1 笔交易 + baidu_kline 历史 close 才可计算 daily PnL."
        )
    if not pnl_by_date:
        return _placeholder("PnL 时间序列为空 — 数据不足.")

    img_html = _render_heatmap_png(pnl_by_date)

    dates = sorted(pnl_by_date.keys())
    pnls = list(pnl_by_date.values())
    n_pos = sum(1 for p in pnls if p > 0)
    n_neg = sum(1 for p in pnls if p < 0)
    n_zero = sum(1 for p in pnls if abs(p) < 1e-9)
    total_pnl = sum(pnls)
    win_rate = (n_pos / (n_pos + n_neg) * 100) if (n_pos + n_neg) > 0 else 0
    best_day_pnl = max(pnls) if pnls else 0
    worst_day_pnl = min(pnls) if pnls else 0
    best_day_date = dates[pnls.index(best_day_pnl)].strftime("%Y-%m-%d") if pnls else ""
    worst_day_date = dates[pnls.index(worst_day_pnl)].strftime("%Y-%m-%d") if pnls else ""

    total_color = "#16a34a" if total_pnl >= 0 else "#dc2626"
    banner = (
        '<div style="background:rgba(107,114,128,0.08); border-left:3px solid #6b7280; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        '<strong>📅 Daily PnL Heatmap</strong> · '
        f'累计 PnL <strong style="color:{total_color};">¥{total_pnl:+,.0f}</strong> · '
        f'{len(dates)} 个交易日 · '
        f'<span style="color:#16a34a; font-weight:600;">盈 {n_pos}</span> / '
        f'<span style="color:#dc2626; font-weight:600;">亏 {n_neg}</span> / '
        f'平 {n_zero} · 胜率 <strong>{win_rate:.0f}%</strong><br>'
        '<span style="color:var(--muted, #6b7280); font-size:11px;">'
        f"最佳日 <strong>{best_day_date}</strong> ¥{best_day_pnl:+,.0f} · "
        f"最差日 <strong>{worst_day_date}</strong> ¥{worst_day_pnl:+,.0f} · "
        "颜色深浅 ∝ |PnL|, 绿涨红跌 · GitHub-style 日历布局."
        "</span></div>"
    )
    footer = (
        '<div style="margin-top:10px; font-size:11px; color:var(--muted, #6b7280);">'
        "假设每笔 BUY = 100 shares (1 lot), PnL_T = Σ qty_at_start_of_T × Δclose. "
        "数据源 <code>paper_trade_log.csv</code> + <code>baidu_kline.parquet</code>."
        "</div>"
    )
    return banner + img_html + footer
