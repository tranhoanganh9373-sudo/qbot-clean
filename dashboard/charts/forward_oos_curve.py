"""Forward OOS 累计收益 + drawdown 曲线 (Section C enhanced).

数据源: data_cache/portfolio.xlsx 'Forward OOS Track' (header=1).

行为:
  - n_months == 0: 完全空状态 placeholder.
  - n_months < 3: fallback 卡片 + 1/3/6/12 月 milestone timeline.
  - n_months >= 3: 多 panel pyfolio-style — cumulative / drawdown / monthly-return bar.
    (>= 6 月 进一步追加 rolling Sharpe 3M; >= 12 月 追加 monthly heatmap.)

只读, 不修改 production. fallback to placeholder string on any error.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dashboard.utils import fig_to_base64

MIN_MONTHS_FOR_CHART = 3
SHEET_NAME = "Forward OOS Track"
START_DATE_DEFAULT = date(2026, 5, 25)
MILESTONES_MONTHS = [
    (1, "符号检查"),
    (3, "首次曲线"),
    (6, "首份 Sharpe"),
    (12, "干净 Sharpe/Calmar"),
]
COLOR_OK = "#16a34a"
COLOR_FAIL = "#dc2626"
COLOR_ACCENT = "#2563eb"
COLOR_MUTED = "#6b7280"


def _placeholder_html(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:48px 16px;">'
        f"{message}"
        "</div>"
    )


def _load_forward_oos(portfolio_xlsx: Path) -> pd.DataFrame:
    if not portfolio_xlsx.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(portfolio_xlsx, sheet_name=SHEET_NAME, header=1)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df.empty or "cum_return" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["month_end_date", "cum_return"]).copy()
    df["month_end_date"] = pd.to_datetime(df["month_end_date"])
    df = df.sort_values("month_end_date").reset_index(drop=True)
    return df


def _build_drawdown(cum_return: pd.Series) -> pd.Series:
    equity = 1.0 + cum_return
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd


def _milestone_timeline_html(n_months: int) -> str:
    """渲染 1/3/6/12 月 milestone timeline. 完成→绿 / 当前→蓝 / 未到→灰."""
    cells = []
    for m, label in MILESTONES_MONTHS:
        if n_months > m:
            color = COLOR_OK
            icon = "✓"
            state = "完成"
        elif n_months == m or (m == 3 and 1 <= n_months < 3):
            color = COLOR_ACCENT
            icon = "●"
            state = "进行中" if n_months < m else "当前"
        else:
            color = COLOR_MUTED
            icon = "○"
            state = f"还需 {m - n_months} 月"
        cells.append(f"""
<div style='flex:1; min-width:120px; text-align:center;
            padding:10px; border:1px solid var(--border, #e5e7eb);
            border-radius:6px; background:rgba(0,0,0,0.02);'>
  <div style='font-size:20px; color:{color};'>{icon}</div>
  <div style='font-size:13px; font-weight:600; color:{color};'>{m} 月</div>
  <div style='font-size:11px; color:var(--muted, #6b7280); margin-top:2px;'>{label}</div>
  <div style='font-size:10px; color:{color}; margin-top:2px; font-weight:500;'>{state}</div>
</div>
""")
    return (
        "<div style='display:flex; gap:8px; flex-wrap:wrap; margin-top:12px;'>"
        + "".join(cells)
        + "</div>"
    )


def _short_fallback_html(df: pd.DataFrame, n_months: int) -> str:
    """< 3 月: summary + milestone timeline."""
    latest = df.iloc[-1]
    cum_pct = float(latest["cum_return"]) * 100
    cum_color = COLOR_OK if cum_pct >= 0 else COLOR_FAIL
    alert = str(latest.get("alert_level", "n/a"))
    monthly = (
        float(latest["monthly_return"]) * 100
        if "monthly_return" in df.columns and pd.notna(latest.get("monthly_return"))
        else None
    )
    monthly_str = f"{monthly:+.2f}%" if monthly is not None else "n/a"

    head_card = f"""
<div style='display:flex; gap:12px; flex-wrap:wrap;'>
  <div style='flex:1; min-width:140px; padding:12px; background:rgba(0,0,0,0.02);
              border:1px solid var(--border, #e5e7eb); border-radius:6px;'>
    <div style='font-size:11px; color:var(--muted, #6b7280); text-transform:uppercase;
                letter-spacing:0.5px;'>累积月数</div>
    <div style='font-size:20px; font-weight:700; color:{COLOR_ACCENT}; margin-top:4px;'>
      {n_months} / {MIN_MONTHS_FOR_CHART}+ 起绘曲线
    </div>
  </div>
  <div style='flex:1; min-width:140px; padding:12px; background:rgba(0,0,0,0.02);
              border:1px solid var(--border, #e5e7eb); border-radius:6px;'>
    <div style='font-size:11px; color:var(--muted, #6b7280); text-transform:uppercase;
                letter-spacing:0.5px;'>当前累计</div>
    <div style='font-size:20px; font-weight:700; color:{cum_color}; margin-top:4px;'>
      {cum_pct:+.2f}%
    </div>
  </div>
  <div style='flex:1; min-width:140px; padding:12px; background:rgba(0,0,0,0.02);
              border:1px solid var(--border, #e5e7eb); border-radius:6px;'>
    <div style='font-size:11px; color:var(--muted, #6b7280); text-transform:uppercase;
                letter-spacing:0.5px;'>最近月度</div>
    <div style='font-size:20px; font-weight:700; color:{cum_color}; margin-top:4px;'>
      {monthly_str}
    </div>
  </div>
  <div style='flex:1; min-width:140px; padding:12px; background:rgba(0,0,0,0.02);
              border:1px solid var(--border, #e5e7eb); border-radius:6px;'>
    <div style='font-size:11px; color:var(--muted, #6b7280); text-transform:uppercase;
                letter-spacing:0.5px;'>alert_level</div>
    <div style='font-size:20px; font-weight:700; color:{cum_color}; margin-top:4px;'>
      {alert}
    </div>
  </div>
</div>
"""
    timeline = _milestone_timeline_html(n_months)
    note = (
        f"<div style='margin-top:12px; font-size:12px; color:var(--muted, #6b7280);'>"
        f"≥ {MIN_MONTHS_FOR_CHART} 月后自动切换 cumulative + drawdown + monthly-return 多子图. "
        f"6 月后追加 rolling Sharpe; 12 月追加 monthly heatmap (pyfolio-style)."
        "</div>"
    )
    return head_card + timeline + note


def _build_full_chart(df: pd.DataFrame, n_months: int) -> str:
    """>= 3 月: pyfolio-style 多子图 (cum / drawdown / monthly bar [+ rolling Sharpe + heatmap])."""
    cum = df["cum_return"].astype(float)
    dates = df["month_end_date"]
    dd = _build_drawdown(cum)
    monthly_ret = df["monthly_return"].astype(float) if "monthly_return" in df.columns \
        else cum.diff().fillna(cum.iloc[0])

    n_panels = 3
    if n_months >= 6:
        n_panels = 4  # add rolling Sharpe
    has_heatmap = n_months >= 12

    height = 7 if n_panels == 3 else 9
    if has_heatmap:
        height += 2.5

    height_ratios = [2, 1, 1] if n_panels == 3 else [2, 1, 1, 1]
    if has_heatmap:
        height_ratios.append(1.6)

    fig, axes = plt.subplots(
        len(height_ratios), 1, figsize=(10, height), sharex=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    axes = list(axes)

    ax_cum = axes[0]
    ax_dd = axes[1]
    ax_mon = axes[2]

    ax_cum.plot(dates, cum * 100, color=COLOR_ACCENT, linewidth=2, marker="o")
    ax_cum.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax_cum.fill_between(dates, cum * 100, 0,
                        where=(cum >= 0), color=COLOR_OK, alpha=0.15)
    ax_cum.fill_between(dates, cum * 100, 0,
                        where=(cum < 0), color=COLOR_FAIL, alpha=0.15)
    ax_cum.set_ylabel("Cumulative (%)")
    ax_cum.set_title(f"Forward OOS — {n_months} months (pyfolio-style)", fontsize=12)
    ax_cum.grid(True, alpha=0.3)

    ax_dd.fill_between(dates, dd * 100, 0, color=COLOR_FAIL, alpha=0.5)
    ax_dd.plot(dates, dd * 100, color=COLOR_FAIL, linewidth=1.2)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(True, alpha=0.3)

    colors_mon = [COLOR_OK if v >= 0 else COLOR_FAIL for v in monthly_ret]
    ax_mon.bar(dates, monthly_ret * 100, color=colors_mon, alpha=0.75, width=20)
    ax_mon.axhline(0, color="#9ca3af", linewidth=0.8)
    ax_mon.set_ylabel("Monthly (%)")
    ax_mon.grid(True, alpha=0.3)

    if n_panels >= 4:
        ax_sh = axes[3]
        rolling_window = 3
        # 月度近似 Sharpe = mean / std * sqrt(12); 用 expanding/rolling 3 月窗口.
        roll_mean = monthly_ret.rolling(rolling_window).mean()
        roll_std = monthly_ret.rolling(rolling_window).std()
        roll_sh = (roll_mean / roll_std) * np.sqrt(12)
        ax_sh.plot(dates, roll_sh, color=COLOR_ACCENT, linewidth=1.5, marker=".")
        ax_sh.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
        ax_sh.axhline(0.5, color=COLOR_OK, linewidth=0.6, linestyle=":")
        ax_sh.set_ylabel(f"Rolling Sharpe ({rolling_window}M)")
        ax_sh.grid(True, alpha=0.3)

    ax_mon.set_xlabel("Month End")
    if n_panels >= 4:
        axes[3].set_xlabel("Month End")

    if has_heatmap:
        ax_hm = axes[-1]
        # build year × month grid
        df_hm = pd.DataFrame({"date": dates, "ret": monthly_ret.values}).copy()
        df_hm["year"] = df_hm["date"].dt.year
        df_hm["month"] = df_hm["date"].dt.month
        pivot = df_hm.pivot_table(
            index="year", columns="month", values="ret", aggfunc="last",
        )
        # ensure all 12 cols
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan
        pivot = pivot[sorted(pivot.columns)]
        # imshow
        data = pivot.values * 100
        vmax = float(np.nanmax(np.abs(data))) if np.isfinite(np.nanmax(np.abs(data))) else 1.0
        im = ax_hm.imshow(data, aspect="auto", cmap="RdYlGn",
                          vmin=-vmax, vmax=vmax)
        ax_hm.set_xticks(range(12))
        ax_hm.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax_hm.set_yticks(range(len(pivot.index)))
        ax_hm.set_yticklabels(pivot.index)
        ax_hm.set_title("Monthly Return Heatmap (%)", fontsize=10)
        fig.colorbar(im, ax=ax_hm, shrink=0.7)

    for ax in axes[:3 if not has_heatmap else 3]:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
    fig.autofmt_xdate()
    fig.tight_layout()

    uri = fig_to_base64(fig)
    plt.close(fig)

    # max DD table footer
    max_dd_pct = float(dd.min()) * 100
    sharpe_est = float(monthly_ret.mean() / monthly_ret.std() * np.sqrt(12)) \
        if monthly_ret.std() > 0 else float("nan")

    footer = f"""
<div style='display:flex; gap:12px; margin-top:12px; flex-wrap:wrap; font-size:12px;'>
  <div><strong>累积月数</strong>: {n_months}</div>
  <div><strong>累计收益</strong>:
       <span style='color:{COLOR_OK if cum.iloc[-1] >= 0 else COLOR_FAIL};'>
       {cum.iloc[-1] * 100:+.2f}%</span></div>
  <div><strong>Max DD</strong>: <span style='color:{COLOR_FAIL};'>{max_dd_pct:.2f}%</span></div>
  <div><strong>Sharpe (annualized est)</strong>:
       <span style='color:{COLOR_OK if sharpe_est >= 0.5 else COLOR_FAIL};'>
       {sharpe_est:.2f}</span></div>
</div>
"""
    return f'<img src="{uri}" alt="Forward OOS chart"/>' + footer


def build_forward_oos_chart(portfolio_xlsx: Path) -> str:
    """主入口: 返回可直接塞到 template `{{forward_oos_chart}}` 的 HTML 片段."""
    df = _load_forward_oos(portfolio_xlsx)
    n_months = len(df)

    if n_months == 0:
        return _placeholder_html(
            "Forward OOS Track 尚无数据 (portfolio.xlsx 该 sheet 为空).<br>"
            "需 paper_trade 累积至少 1 个完整月份后才会显示."
            + _milestone_timeline_html(0)
        )

    if n_months < MIN_MONTHS_FOR_CHART:
        return _short_fallback_html(df, n_months)

    return _build_full_chart(df, n_months)
