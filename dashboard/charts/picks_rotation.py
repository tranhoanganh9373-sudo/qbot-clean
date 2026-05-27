"""Daily picks rotation diff — yesterday vs today holdings Venn + lists.

数据源 (只读):
  - data_cache/portfolio_state.json — 当前 holdings + date
      { "date": "YYYY-MM-DD", "holdings": ["SH600547", "SZ300347", ...] }
  - data_cache/portfolio_state.json.bak — (optional) 昨日 holdings (same shape)
  - data_cache/paper_trade_log.csv — fallback. cols: date,action,symbol,name,score,price.
      用最近一日的 BUY/SELL 反推 yesterday holdings = (today - today_BUYs) ∪ today_SELLs.

输出:
  - 自绘 2-set Venn diagram (matplotlib Circle patches, 不依赖 matplotlib_venn).
  - 三栏 list: only-yesterday (SELL), overlap (HOLD), only-today (BUY).

只读, 不修改 production.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PORTFOLIO_STATE = ROOT / "data_cache" / "portfolio_state.json"
PORTFOLIO_STATE_BAK = ROOT / "data_cache" / "portfolio_state.json.bak"
PAPER_TRADE_LOG = ROOT / "data_cache" / "paper_trade_log.csv"

COLOR_SELL = "#dc2626"   # only-yesterday → sold
COLOR_HOLD = "#2563eb"   # overlap → still held
COLOR_BUY = "#16a34a"    # only-today → newly bought


def _load_state(path: Path) -> tuple[str, set[str]] | None:
    """returns (date_str, holdings_set) or None."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    date_str = str(data.get("date", "?"))
    holdings = set(data.get("holdings", []))
    return date_str, holdings


def _derive_yesterday_from_log(
    today: set[str],
) -> tuple[set[str], str, set[str], set[str]] | None:
    """returns (yesterday_holdings, prev_date_str, today_buys, today_sells) or None.

    yesterday = (today - today_BUYs) ∪ today_SELLs.
    勘注: 该公式假定 paper_trade_log 最新日期 = today's portfolio_state.date.
    """
    if not PAPER_TRADE_LOG.exists():
        return None
    try:
        df = pd.read_csv(PAPER_TRADE_LOG)
    except Exception:  # noqa: BLE001
        return None
    if df.empty or "date" not in df.columns or "action" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    latest = df["date"].max()
    distinct_dates = sorted(df["date"].unique())
    prev = distinct_dates[-2] if len(distinct_dates) >= 2 else None

    today_events = df[df["date"] == latest]
    buys = set(today_events.loc[today_events["action"] == "BUY", "symbol"])
    sells = set(today_events.loc[today_events["action"] == "SELL", "symbol"])

    yesterday = (today - buys) | sells
    prev_str = str(prev)[:10] if prev is not None else "prior"
    return yesterday, prev_str, buys, sells


def _draw_venn(yesterday: set[str], today: set[str]) -> str:
    """自绘 2-circle Venn → base64 PNG."""
    overlap = yesterday & today
    only_y = yesterday - today
    only_t = today - yesterday

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=110)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    c_left = mpatches.Circle((3.5, 2.5), 2.0, facecolor=COLOR_SELL,
                             edgecolor=COLOR_SELL, alpha=0.30, linewidth=2)
    c_right = mpatches.Circle((6.5, 2.5), 2.0, facecolor=COLOR_BUY,
                              edgecolor=COLOR_BUY, alpha=0.30, linewidth=2)
    ax.add_patch(c_left)
    ax.add_patch(c_right)

    ax.text(2.3, 2.5, f"{len(only_y)}", ha="center", va="center",
            fontsize=26, fontweight="bold", color=COLOR_SELL)
    ax.text(7.7, 2.5, f"{len(only_t)}", ha="center", va="center",
            fontsize=26, fontweight="bold", color=COLOR_BUY)
    ax.text(5.0, 2.5, f"{len(overlap)}", ha="center", va="center",
            fontsize=26, fontweight="bold", color=COLOR_HOLD)

    ax.text(2.3, 4.7, "Yesterday only\n(SELL)", ha="center", va="top",
            fontsize=10, color=COLOR_SELL, fontweight="bold")
    ax.text(7.7, 4.7, "Today only\n(BUY)", ha="center", va="top",
            fontsize=10, color=COLOR_BUY, fontweight="bold")
    ax.text(5.0, 0.4, "Overlap\n(HOLD)", ha="center", va="bottom",
            fontsize=10, color=COLOR_HOLD, fontweight="bold")

    ax.set_title(
        f"Holdings rotation — yesterday ({len(yesterday)}) vs today ({len(today)})",
        fontsize=11,
    )

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _symbol_list_html(symbols: set[str], color: str) -> str:
    if not symbols:
        return "<span style='color:var(--muted, #6b7280); font-size:12px;'>(none)</span>"
    from dashboard.utils.stock_names import code_with_name
    sorted_syms = sorted(symbols)
    items = "".join(
        f"<li style='font-family:\"SF Mono\", Menlo, Monaco, Consolas, monospace; "
        f"font-size:12px; color:{color};'>{html.escape(code_with_name(s))}</li>"
        for s in sorted_syms
    )
    return f"<ul style='margin:0; padding-left:20px;'>{items}</ul>"


def build_picks_rotation_section() -> str:
    """Section 15: yesterday vs today rotation Venn + diff list."""
    today_pair = _load_state(PORTFOLIO_STATE)
    if today_pair is None:
        return (
            "<div class='placeholder-content'>"
            f"未找到 <code>{PORTFOLIO_STATE.name}</code> — 无法绘制 picks rotation."
            "</div>"
        )
    today_date, today = today_pair

    yesterday_pair = _load_state(PORTFOLIO_STATE_BAK)
    if yesterday_pair is not None:
        yesterday_date, yesterday = yesterday_pair
        derived_source = "portfolio_state.json.bak"
        today_buys = today - yesterday  # noqa: F841 (kept for parity)
        today_sells = yesterday - today  # noqa: F841
    else:
        derived = _derive_yesterday_from_log(today)
        if derived is None:
            return (
                "<div class='placeholder-content'>"
                "portfolio_state.json.bak 不存在, paper_trade_log 也不可读 — "
                "无法推算 yesterday holdings."
                "</div>"
            )
        yesterday, yesterday_date, today_buys, today_sells = derived
        derived_source = (
            "paper_trade_log.csv 反推 "
            f"(today_BUYs={len(today_buys)} today_SELLs={len(today_sells)})"
        )

    overlap = yesterday & today
    only_y = yesterday - today
    only_t = today - yesterday

    img_b64 = _draw_venn(yesterday, today)
    img_html = (
        f'<img src="data:image/png;base64,{img_b64}" alt="picks rotation venn"/>'
    )

    rotation_pct = (
        100.0 * (len(only_t) + len(only_y)) / max(len(yesterday | today), 1)
    )

    meta = f"""
<div style='display:flex; gap:12px; flex-wrap:wrap; margin-bottom:10px;
            font-size:12px; color:var(--muted, #6b7280);'>
  <div><strong>yesterday</strong> ({html.escape(yesterday_date)}): {len(yesterday)} 只</div>
  <div><strong>today</strong> ({html.escape(today_date)}): {len(today)} 只</div>
  <div><strong>rotation%</strong>:
       <span style='color:{COLOR_HOLD};font-weight:600;'>{rotation_pct:.1f}%</span></div>
  <div>source: <code>{html.escape(derived_source)}</code></div>
</div>
"""

    cols_html = f"""
<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
            gap:12px; margin-top:14px;'>
  <div style='border:1px solid {COLOR_SELL}; border-radius:6px; padding:10px;
              background:rgba(220,38,38,0.04);'>
    <div style='color:{COLOR_SELL}; font-weight:600; font-size:13px; margin-bottom:6px;'>
      SELL &mdash; {len(only_y)} 只 (only yesterday)
    </div>
    {_symbol_list_html(only_y, COLOR_SELL)}
  </div>
  <div style='border:1px solid {COLOR_HOLD}; border-radius:6px; padding:10px;
              background:rgba(37,99,235,0.04);'>
    <div style='color:{COLOR_HOLD}; font-weight:600; font-size:13px; margin-bottom:6px;'>
      HOLD &mdash; {len(overlap)} 只 (overlap)
    </div>
    {_symbol_list_html(overlap, COLOR_HOLD)}
  </div>
  <div style='border:1px solid {COLOR_BUY}; border-radius:6px; padding:10px;
              background:rgba(22,163,74,0.04);'>
    <div style='color:{COLOR_BUY}; font-weight:600; font-size:13px; margin-bottom:6px;'>
      BUY &mdash; {len(only_t)} 只 (only today)
    </div>
    {_symbol_list_html(only_t, COLOR_BUY)}
  </div>
</div>
<div style='font-size:11px; color:var(--muted, #6b7280); margin-top:10px;'>
  rotation% = (BUY + SELL) / union(yesterday, today).
  高 rotation 提示因子 churn 风险, 低 rotation 提示稳定的 conviction.
</div>
"""
    return meta + img_html + cols_html
