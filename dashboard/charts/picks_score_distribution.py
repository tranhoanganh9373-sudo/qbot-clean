"""Panel — Picks Score Distribution.

显示 production 模型对全 universe (CSI300 ~296) 当日 z_pred 的 histogram,
并把 Top K picks 用红线高亮 + Top-(K+1) cutoff 绿色虚线.

数据源 (只读):
  - data_cache/picks_today.json — 需含 `full_distribution.z_pred_values` (296 floats).
    若没有则提示 paper_trade_today.py 升级后才支持.

用途:
  - 看 Top picks 是 outlier (远离分布尾) 还是平均水平 (信号弱).
  - 评估当日 production 模型 signal-to-noise.

只读, 不修改 production.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dashboard.utils import fig_to_base64
from dashboard.utils.stock_names import code_with_name

ROOT = Path(__file__).resolve().parent.parent.parent
PICKS_JSON = ROOT / "data_cache" / "picks_today.json"


def _placeholder(msg: str) -> str:
    return f'<div class="placeholder-content">{msg}</div>'


def build_picks_dist_section() -> str:
    """histogram of all universe z_pred + Top K markers."""
    if not PICKS_JSON.exists():
        return _placeholder(
            "picks_today.json 未找到 — paper_trade_today.py 尚未跑."
        )

    try:
        data = json.loads(PICKS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return _placeholder(
            f"picks_today.json 解析失败: {type(exc).__name__}: {exc}"
        )

    picks = data.get("picks", []) or []
    full = data.get("full_distribution", {}) or {}

    if not full or not full.get("z_pred_values"):
        return _placeholder(
            "picks_today.json 缺 <code>full_distribution</code> 字段 — "
            "paper_trade_today.py 升级 (~10 LOC dump 全 universe z_pred) 后才支持. "
            "下次 paper_trade 跑完即可看到."
        )

    all_z = np.array(full["z_pred_values"], dtype=float)
    if all_z.size == 0:
        return _placeholder("full_distribution.z_pred_values 为空数组.")

    top_z = np.array([p.get("z_pred", 0.0) for p in picks], dtype=float)
    top_syms = [p.get("sym", "") for p in picks]
    k = len(top_syms)

    fig, ax = plt.subplots(figsize=(11, 5))

    # histogram
    lo = float(all_z.min()) - 0.1
    hi = float(all_z.max()) + 0.1
    bins = np.linspace(lo, hi, 40)
    counts, _bin_edges, _patches = ax.hist(
        all_z,
        bins=bins,
        color="#9ca3af",
        alpha=0.6,
        edgecolor="black",
        linewidth=0.4,
        label=f"全 universe (n={all_z.size})",
    )

    # Top K markers (red, 标 sym at top)
    y_marker = (float(max(counts)) if len(counts) else 1.0) * 1.05
    for sym, z in zip(top_syms, top_z):
        ax.axvline(z, color="#e74c3c", linestyle="-", alpha=0.7, linewidth=1.2)
        ax.annotate(
            code_with_name(sym),
            xy=(z, y_marker),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8,
            rotation=45,
            ha="left",
            va="bottom",
            color="#e74c3c",
        )

    # Top-(K+1) cutoff: 第 (K+1) 名 z_pred, Top K 须 >= 此线
    sorted_z = np.sort(all_z)[::-1]
    cutoff_html = ""
    if k > 0 and len(sorted_z) > k:
        cutoff = float(sorted_z[k])
        ax.axvline(
            cutoff,
            color="#27ae60",
            linestyle="--",
            alpha=0.7,
            linewidth=1.4,
            label=f"Top-{k+1} cutoff = {cutoff:.2f}",
        )
        cutoff_html = f"<li>Top-{k+1} cutoff (绿色虚线) = <code>{cutoff:+.2f}</code></li>"

    ax.set_xlabel("z(pred) standardized score")
    ax.set_ylabel("count of stocks")
    title_date = data.get("as_of_date", "")
    n_total = int(full.get("n_total", all_z.size))
    ax.set_title(
        f"全 {n_total} universe z(pred) 分布 + Top {k} 高亮 ({title_date})"
    )
    # Top 标注会超出 axes 上沿, 抬高 ylim 给注释留空间
    if len(counts) > 0:
        ax.set_ylim(0, float(max(counts)) * 1.35)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    img_uri = fig_to_base64(fig)
    plt.close(fig)

    z_max = float(all_z.max())
    z_min = float(all_z.min())
    z_std = float(all_z.std())
    z_mean = float(all_z.mean())
    top_min = float(top_z.min()) if k else float("nan")
    top_max = float(top_z.max()) if k else float("nan")

    # outlier 评估: Top K 的最小 z_pred 是否 > μ + 1σ
    outlier_threshold = z_mean + z_std
    if k > 0 and top_min > outlier_threshold:
        verdict = (
            f"Top {k} 全部 > μ+1σ ({outlier_threshold:+.2f}) → 信号清晰, picks 是真 outlier."
        )
    elif k > 0 and top_min > z_mean:
        verdict = (
            f"Top {k} 最低 z={top_min:+.2f} 仍 > μ 但未到 μ+1σ → 信号中等, picks 偏弱."
        )
    else:
        verdict = (
            f"Top {k} 最低 z={top_min:+.2f} ≤ μ ({z_mean:+.2f}) → 信号弱, 当日 picks 高风险."
        )

    explanation = f"""
<div class='card-explanation' style='margin-top:8px; color: var(--muted); font-size: 12px;'>
分布解读 (z_pred = DEnsemble 标准化预测, 横截面 z-score):
<ul style='margin:6px 0; padding-left:20px;'>
<li>histogram 左/右 fat tail → 模型对 universe 区分度高, Top {k} 易选;
    histogram 集中 (μ±1σ 内 80%+) → 信号弱, Top {k} 跟 median 接近, 风险高.</li>
<li>Top {k} 红线全部 ≥ 绿色 cutoff → production 严格按 final_score 排 (z_pred + sidecar),
    若个别红线 &lt; cutoff 说明 sidecar 重排 (例如 v19.6 amp_imb_20d) 起作用.</li>
{cutoff_html}
<li>统计: μ=<code>{z_mean:+.2f}</code>, σ=<code>{z_std:.2f}</code>,
    min=<code>{z_min:+.2f}</code>, max=<code>{z_max:+.2f}</code>,
    Top {k} 范围 <code>[{top_min:+.2f}, {top_max:+.2f}]</code>.</li>
<li>当日评估: {verdict}</li>
</ul>
</div>
"""

    return (
        f'<img src="{img_uri}" alt="Picks score distribution histogram"/>'
        + explanation
    )


if __name__ == "__main__":
    print(build_picks_dist_section()[:500])
