"""Dashboard Section 9 — Per-Factor IS→OOS Calmar Gap Scatter.

10 个 sidecar candidate (跟 model_leaderboard 同源, 但剔除 baseline 因为 baseline 无
sidecar 即无 IS Calmar 数字) 的 IS Calmar vs OOS Calmar 散点图.

设计要点:
- 45° 参考线 (y=x): 上方 = OOS > IS (真 alpha); 下方 = IS > OOS (overfit).
- 颜色 = tag, 跟 model_leaderboard.TAG_COLORS 保持一致 (production / 候选 / shadow /
  假 alpha / data gap / abort / catastrophic).
- 阈值线: y=0.5 (OOS abort line) / y=0.77 (baseline OOS).
- 文本 label 在 dot 旁边.
- 一眼看 distance 远的 dots: catastrophic 区 (IS > 2 + OOS < 0.5).

数据源 (硬编码):
- 10 行跟 model_leaderboard.MODEL_DATA 中 is_calmar 非 None 的子集严格一致
  (2026-05-26 verified, v3 cache).
- baseline (train24 纯) 无 IS sidecar → 不入 scatter.
"""
from __future__ import annotations

from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dashboard.utils import fig_to_base64


# ---------------------------------------------------------------------------
# 数据 (硬编码 — 与 model_leaderboard.MODEL_DATA is_calmar 非 None 子集对齐)
# ---------------------------------------------------------------------------

SCATTER_DATA: Final[list[dict]] = [
    {"name": "v19.6 amp_imb_20d", "is_calmar": 0.79, "oos_calmar": 1.29, "tag": "production"},
    {"name": "v19.7 stacked", "is_calmar": 0.76, "oos_calmar": 0.96, "tag": "候选"},
    {"name": "v20 industry_60d", "is_calmar": 2.95, "oos_calmar": 0.65, "tag": "假 alpha"},
    {"name": "v19.4 m5+m20", "is_calmar": 1.67, "oos_calmar": 0.62, "tag": "shadow"},
    {"name": "v19.8 dragontiger", "is_calmar": 0.94, "oos_calmar": 0.42, "tag": "data gap"},
    {"name": "v20 shareholders", "is_calmar": 6.09, "oos_calmar": 0.39, "tag": "abort"},
    {"name": "v19.5 team_coin", "is_calmar": 0.77, "oos_calmar": 0.38, "tag": "假 alpha"},
    {"name": "v20 vol_z_5d", "is_calmar": 3.39, "oos_calmar": 0.32, "tag": "假 alpha"},
    {"name": "v19.9 unlock", "is_calmar": 2.70, "oos_calmar": 0.09, "tag": "catastrophic"},
    {"name": "v20 super_big_net", "is_calmar": 2.01, "oos_calmar": -0.07, "tag": "catastrophic"},
]


TAG_COLORS: Final[dict[str, str]] = {
    "production": "#27ae60",
    "候选": "#16a085",
    "fallback": "#3498db",
    "shadow": "#9b59b6",
    "假 alpha": "#e67e22",
    "data gap": "#95a5a6",
    "abort": "#e74c3c",
    "catastrophic": "#7f0000",
}

OOS_ABORT_LINE: Final[float] = 0.5
BASELINE_OOS: Final[float] = 0.77


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def _ensure_cjk_font() -> None:
    """把常见 CJK 字体 prepend 到 matplotlib sans-serif 列表, 防中文方框."""
    cjk_candidates = [
        "PingFang SC", "Heiti SC", "STHeiti", "Hiragino Sans GB",
        "Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei",
        "SimHei", "Arial Unicode MS",
    ]
    current = list(plt.rcParams.get("font.sans-serif", []))
    new_list = cjk_candidates + [f for f in current if f not in cjk_candidates]
    plt.rcParams["font.sans-serif"] = new_list
    plt.rcParams["axes.unicode_minus"] = False


def build_is_oos_scatter_fig() -> plt.Figure:
    """Build Figure: IS Calmar (x) vs OOS Calmar (y) scatter, 颜色=tag, 文本=name."""
    _ensure_cjk_font()

    fig, ax = plt.subplots(figsize=(10, 7))

    for d in SCATTER_DATA:
        color = TAG_COLORS.get(d["tag"], "#999999")
        ax.scatter(
            d["is_calmar"], d["oos_calmar"],
            c=color, s=150, alpha=0.85,
            edgecolors="black", linewidths=1,
        )
        ax.annotate(
            d["name"], (d["is_calmar"], d["oos_calmar"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=8, alpha=0.9,
        )

    # axis 范围
    is_vals = [d["is_calmar"] for d in SCATTER_DATA]
    oos_vals = [d["oos_calmar"] for d in SCATTER_DATA]
    lim_max = max(max(is_vals), max(oos_vals)) + 0.5
    lim_min = min(min(is_vals), min(oos_vals)) - 0.5

    # 45° 参考线 y = x
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        "k--", alpha=0.4, label="y=x (IS=OOS, 真 alpha 边界)",
    )

    # 阈值线
    ax.axhline(0, color="gray", alpha=0.3, linewidth=0.5)
    ax.axhline(
        OOS_ABORT_LINE, ls=":", color="red", alpha=0.5,
        label=f"OOS {OOS_ABORT_LINE:.2f} abort line",
    )
    ax.axhline(
        BASELINE_OOS, ls=":", color="blue", alpha=0.5,
        label=f"baseline OOS {BASELINE_OOS:.2f}",
    )

    ax.set_xlabel("IS Calmar (训练期 2017-2020 或 2014-2020)")
    ax.set_ylabel("OOS Calmar (2021-05 ~ 2026-04, 60 月)")
    ax.set_title(
        "Per-Factor IS → OOS Calmar Scatter (45° line = 真 alpha 边界)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    return fig


def _count_above_45deg() -> tuple[int, int]:
    """计算 45° 线上方 / 下方点数 (上方 = OOS > IS = 真 alpha)."""
    above = sum(1 for d in SCATTER_DATA if d["oos_calmar"] > d["is_calmar"])
    below = len(SCATTER_DATA) - above
    return above, below


def build_is_oos_scatter_section() -> str:
    """主入口: 返回 HTML fragment (chart + 解读)."""
    fig = build_is_oos_scatter_fig()
    img_uri = fig_to_base64(fig)
    plt.close(fig)

    above, below = _count_above_45deg()

    return (
        f'<img src="{img_uri}" alt="Per-Factor IS→OOS Calmar Scatter"/>'
        '<div class="card-explanation" style="margin-top:12px; font-size:13px; '
        'line-height:1.7;">'
        "<strong>解读</strong>:<br>"
        f"- <strong>45° 线上方</strong> = OOS &gt; IS (真 alpha): "
        f"<b>{above}</b> 个 (v19.6, v19.7) — 真正能外推的 sidecar.<br>"
        f"- <strong>45° 线下方</strong> = IS &gt; OOS (overfit): "
        f"<b>{below}</b> 个 — 大部分 Phase B 失败案例.<br>"
        "- <strong>右下角远点</strong> = catastrophic (IS &gt; 2, OOS &lt; 0.5): "
        "v20 shareholders (6.09→0.39), v19.9 unlock (2.70→0.09), "
        "v20 super_big_net (2.01→-0.07), v20 vol_z_5d (3.39→0.32), "
        "v20 industry_60d (2.95→0.65).<br>"
        "- <strong>QuantaAlpha 教训</strong>: n_months &lt; 60 + IS Calmar &gt; 1.5 → "
        "OOS 衰减 -90%+ (4 例 confirmed).<br>"
        "- <strong>颜色编码</strong>: "
        '<span style="color:#27ae60;">绿=production</span> / '
        '<span style="color:#16a085;">蓝绿=候选</span> / '
        '<span style="color:#9b59b6;">紫=shadow</span> / '
        '<span style="color:#e67e22;">橙=假 alpha</span> / '
        '<span style="color:#95a5a6;">灰=data gap</span> / '
        '<span style="color:#e74c3c;">红=abort</span> / '
        '<span style="color:#7f0000;">深红=catastrophic</span>.'
        "</div>"
    )
