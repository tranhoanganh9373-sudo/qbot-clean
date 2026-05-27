"""Dashboard Stage 3+ — Model Leaderboard.

可视化所有 sidecar 候选 model 的 OOS Calmar 排名 + IS→OOS 衰减程度.

设计要点:
- 一眼看清 production / 候选 / 假 alpha / catastrophic 分布
- IS 强 ≠ OOS 强 (Phase B 教训视觉警示)
- 数据硬编码 (11 model 中只有 4 个在 Training sheet, 其余来自 Phase B memory)

数据源:
- 4 行 Training sheet 数字 (baseline / v19.6 / v19.4 / v19.7) 来自
  data_cache/portfolio.xlsx 的 Training sheet, 2026-05-26 verified.
- 其余 Phase B abort/catastrophic 数字来自 examples/v19_*_oos_stats.csv 与
  ~/.claude memory project_phase4_sidecar_v* 系列. 全是 v3 cache 后的 OOS 数字.
"""
from __future__ import annotations

import html
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dashboard.utils import fig_to_base64


# ---------------------------------------------------------------------------
# 数据 (硬编码 — 见模块文档 "数据源" 段)
# ---------------------------------------------------------------------------

MODEL_DATA: Final[list[dict]] = [
    {
        "name": "v19.6 amp_imb_20d",
        "oos_calmar": 1.29,
        "is_calmar": 0.79,
        "sharpe": 0.92,
        "ann": 34.36,
        "mdd": -26.66,
        "cum": 337.80,
        "tag": "production",
    },
    {
        "name": "v19.7 stacked (a20+m5)",
        "oos_calmar": 0.96,
        "is_calmar": 0.76,
        "sharpe": 0.87,
        "ann": 30.65,
        "mdd": -32.03,
        "cum": 280.70,
        "tag": "候选",
    },
    {
        "name": "baseline (train24 纯)",
        "oos_calmar": 0.77,
        "is_calmar": None,
        "sharpe": 0.69,
        "ann": 23.02,
        "mdd": -30.01,
        "cum": 258.96,
        "tag": "fallback",
    },
    {
        "name": "v20 industry_60d",
        "oos_calmar": 0.65,
        "is_calmar": 2.95,
        "sharpe": 0.68,
        "ann": 22.10,
        "mdd": -33.93,
        "cum": 191.00,
        "tag": "假 alpha",
    },
    {
        "name": "v19.4 m5+m20",
        "oos_calmar": 0.62,
        "is_calmar": None,
        "sharpe": 0.62,
        "ann": 19.35,
        "mdd": -31.21,
        "cum": 142.16,
        "tag": "shadow",
    },
    {
        "name": "v19.8 dragontiger",
        "oos_calmar": 0.42,
        "is_calmar": 0.94,
        "sharpe": 0.50,
        "ann": 14.50,
        "mdd": -34.50,
        "cum": 80.00,
        "tag": "data gap",
    },
    {
        "name": "v20 shareholders",
        "oos_calmar": 0.39,
        "is_calmar": 6.09,
        "sharpe": 0.50,
        "ann": 15.80,
        "mdd": -40.32,
        "cum": 85.46,
        "tag": "abort",
    },
    {
        "name": "v19.5 team_coin",
        "oos_calmar": 0.38,
        "is_calmar": 0.77,
        "sharpe": 0.57,
        "ann": 17.20,
        "mdd": -45.06,
        "cum": 113.86,
        "tag": "假 alpha",
    },
    {
        "name": "v20 vol_z_5d",
        "oos_calmar": 0.32,
        "is_calmar": 3.39,
        "sharpe": 0.46,
        "ann": 14.80,
        "mdd": -45.96,
        "cum": 74.45,
        "tag": "假 alpha",
    },
    {
        "name": "v19.9 unlock",
        "oos_calmar": 0.09,
        "is_calmar": 2.70,
        "sharpe": 0.25,
        "ann": 3.17,
        "mdd": -34.42,
        "cum": 16.88,
        "tag": "catastrophic",
    },
    {
        "name": "v20 super_big_net",
        "oos_calmar": -0.07,
        "is_calmar": 2.01,
        "sharpe": 0.04,
        "ann": -2.40,
        "mdd": -33.20,
        "cum": -12.43,
        "tag": "catastrophic",
    },
]


TAG_COLORS: Final[dict[str, str]] = {
    "production": "#27ae60",     # 绿
    "候选": "#16a085",            # 蓝绿
    "fallback": "#3498db",       # 蓝
    "shadow": "#9b59b6",         # 紫
    "假 alpha": "#e67e22",        # 橙
    "data gap": "#95a5a6",       # 灰
    "abort": "#e74c3c",          # 红
    "catastrophic": "#7f0000",   # 深红
}

BASELINE_CALMAR: Final[float] = 0.77
PRODUCTION_CALMAR: Final[float] = 1.29


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


TOP_N_DISPLAY = 5  # 2026-05-26: 仅展示 OOS Calmar top 5 (其余 6 个 abort/catastrophic 见 section 9 IS→OOS scatter)


def build_leaderboard_fig() -> plt.Figure:
    """Build horizontal-bar Figure: OOS Calmar top-N 排名, 颜色=tag, 右侧标 IS→OOS gap."""
    df = (
        pd.DataFrame(MODEL_DATA)
        .sort_values("oos_calmar", ascending=False)
        .head(TOP_N_DISPLAY)
        .sort_values("oos_calmar", ascending=True)  # barh: ascending → top on top
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    y = range(len(df))
    colors = [TAG_COLORS.get(t, "#999999") for t in df["tag"]]
    ax.barh(
        y, df["oos_calmar"], color=colors, alpha=0.85,
        edgecolor="#333333", linewidth=0.5,
    )

    # 参考线
    ax.axvline(BASELINE_CALMAR, ls="--", color="#3498db", alpha=0.6,
               label=f"baseline {BASELINE_CALMAR:.2f}")
    ax.axvline(PRODUCTION_CALMAR, ls="--", color="#27ae60", alpha=0.6,
               label=f"v19.6 production {PRODUCTION_CALMAR:.2f}")
    ax.axvline(0, color="#7f7f7f", linewidth=0.8)

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["name"].tolist(), fontsize=9)
    ax.set_xlabel("OOS Calmar (60 月 2021-05~2026-04, v3 cache)")
    ax.set_title(
        "Model Leaderboard — OOS Calmar 排名 (颜色=tag, Δ%=IS→OOS 衰减)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # 在 bar 右侧 (或左侧 if 负) 标 OOS 数字 + IS gap
    x_max = max(df["oos_calmar"].max(), PRODUCTION_CALMAR) + 0.05
    x_min = min(df["oos_calmar"].min(), 0.0)
    ax.set_xlim(x_min - 0.05, x_max + 0.55)

    for i, row in df.iterrows():
        oos = row["oos_calmar"]
        is_v = row["is_calmar"]
        label = f"{oos:+.2f}"
        if is_v is not None and is_v != 0:
            gap_pct = (oos - is_v) / is_v * 100
            label += f"  (IS {is_v:.2f}, Δ{gap_pct:+.0f}%)"
        text_x = oos + 0.03 if oos >= 0 else oos - 0.03
        ha = "left" if oos >= 0 else "right"
        ax.text(text_x, i, label, fontsize=8, va="center", ha=ha)

    fig.tight_layout()
    return fig


def _decision_matrix_table(df: pd.DataFrame) -> str:
    """决策矩阵 HTML 表 — 按 OOS Calmar 降序."""
    rows: list[str] = []
    for _, r in df.iterrows():
        tag = r["tag"]
        color = TAG_COLORS.get(tag, "#999999")
        is_v = r["is_calmar"]
        if is_v is not None and is_v != 0:
            gap_pct = (r["oos_calmar"] - is_v) / is_v * 100
            is_cell = f"{is_v:.2f}"
            gap_cell = f"{gap_pct:+.0f}%"
        else:
            is_cell = "—"
            gap_cell = "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(r['name'])}</td>"
            f'<td><span class="badge" style="background:{color};color:#fff;">'
            f"{html.escape(tag)}</span></td>"
            f"<td>{r['oos_calmar']:+.2f}</td>"
            f"<td>{is_cell}</td>"
            f"<td>{gap_cell}</td>"
            f"<td>{r['sharpe']:.2f}</td>"
            f"<td>{r['cum']:+.2f}%</td>"
            f"<td>{r['mdd']:+.2f}%</td>"
            "</tr>"
        )
    return (
        '<table class="data" style="margin-top:8px;">'
        "<thead><tr>"
        "<th>Model</th><th>Tag</th><th>OOS Calmar</th><th>IS Calmar</th>"
        "<th>IS→OOS Δ</th><th>Sharpe</th><th>Cum %</th><th>MDD %</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _interpretation_html() -> str:
    """Chart 下方解读."""
    return (
        '<ol style="margin-top:12px; font-size:13px; line-height:1.7;">'
        "<li><b>v19.6 是当前 production</b> (OOS Calmar 1.29 vs baseline 0.77, "
        "<b>+67%</b>); v19.7 stacked 候选 0.96 在评估, 目前不上.</li>"
        '<li><b>颜色编码</b>: <span style="color:#27ae60;">绿=真 alpha</span> / '
        '<span style="color:#9b59b6;">紫=shadow A/B</span> / '
        '<span style="color:#e67e22;">橙=假 alpha (Phase B abort)</span> / '
        '<span style="color:#7f0000;">深红=catastrophic</span>. '
        "区分<b>真 alpha vs IS 数据虚高</b>是 Phase B 的核心教训.</li>"
        "<li><b>IS→OOS 衰减</b>: industry/vol/shareholders/super_big_net/unlock "
        "5 个模型 Δ 全部 <b>-78% ~ -103%</b> catastrophic — IS 高 Calmar "
        "(2.0~6.1) 在 OOS 期完全消失.</li>"
        "<li><b>经验法则</b> (Phase B 后): "
        "<i>IS 期 &lt; 60 月 + IS Calmar &gt; 1.5</i> → "
        "高概率 OOS catastrophic. 真 alpha (v19.6) IS 反而仅 0.79 — "
        "<b>不夸张的 IS 才稳定外推</b>.</li>"
        "<li><b>当前决策</b>: 维持 v19.6 + v19.4 shadow A/B, "
        "不轻易换 production; 新候选必须先 strict OOS 验证 IS→OOS gap "
        "&lt; 30% 才进 shadow.</li>"
        "</ol>"
    )


def build_leaderboard_section() -> str:
    """主入口: 返回 HTML fragment (chart + 解读 + 决策矩阵 table)."""
    fig = build_leaderboard_fig()
    img_uri = fig_to_base64(fig)
    plt.close(fig)

    df_desc = (
        pd.DataFrame(MODEL_DATA)
        .sort_values("oos_calmar", ascending=False)
        .head(TOP_N_DISPLAY)
        .reset_index(drop=True)
    )

    return (
        f'<img src="{img_uri}" alt="Model Leaderboard — OOS Calmar 排名"/>'
        f"{_interpretation_html()}"
        '<div style="margin-top:16px;">'
        '<div style="font-size:13px; color:var(--muted); margin-bottom:4px;">'
        "决策矩阵 (按 OOS Calmar 降序):"
        "</div>"
        f"{_decision_matrix_table(df_desc)}"
        "</div>"
    )
