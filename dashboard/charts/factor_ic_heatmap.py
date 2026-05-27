"""Dashboard Stage 3+ — Factor IC Heatmap (Phase A 月度).

可视化 Phase A 候选 factor 的月度 IC 时间序列, signed alignment 后用红绿 heatmap
呈现, 与 model_leaderboard 静态数字互补.

设计要点:
- 纵 axis = factor (11 个); 横 axis = month (2014-01 ~ csv 截止)
- color: 绿色 = 因子方向正确 (signed alignment 后 IC > 0), 红色 = 反向, 白 ≈ 0
- 黑色虚线 = regime shift 关键时点 (COVID / OOS 起 / 注册制扩 / 政策市)

数据源 (全只读 examples/*_is_monthly.csv):
- factor_ic_ibs_csi300_is_monthly.csv          (factor_signed,month_end,ic)
- factor_ic_shareholders_is_monthly.csv        (factor,asof_date,ic)
- factor_ic_industry_adj_ret_is_monthly.csv    (factor,asof_date,ic)
- v20_volume_zscore_is_monthly.csv             (variant,month_start,ic)
- super_big_net_is_monthly.csv                 (factor_signed,month_end,ic)
- factor_ic_technical_csi300_is_monthly.csv    (factor,month_start,ic)
- factor_ic_unlock_csi300_is_monthly.csv       (factor,asof_date,ic)
- factor_ic_fundamentals_csi300_is_monthly.csv (factor,month_start,ic)

注意 csv schema 不统一: factor 列名 / 月份列名 / sign 内嵌方式 都各不同, 因此
配置表 MONTHLY_CSVS 显式给出 (label, csv_path, factor_col, factor_name,
month_col, sign).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

from dashboard.utils import fig_to_base64


def _ensure_cjk_font() -> None:
    """把系统 CJK 字体 prepend 到 sans-serif 链, 防中文渲染成方框.

    幂等; 只在本模块 ax.set_title / set_xlabel / 文字 label 用到中文时生效.
    """
    candidates = (
        "PingFang SC", "PingFang HK", "Heiti TC", "STHeiti",
        "Hiragino Sans GB", "Songti SC", "Arial Unicode MS",
        "Noto Sans CJK SC", "SimHei",
    )
    available = {f.name for f in fm.fontManager.ttflist}
    chain = [c for c in candidates if c in available]
    if not chain:
        return
    current = plt.rcParams["font.sans-serif"]
    merged = chain + [f for f in current if f not in chain]
    plt.rcParams["font.sans-serif"] = merged
    plt.rcParams["axes.unicode_minus"] = False


_ensure_cjk_font()

ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
EXAMPLES: Final[Path] = ROOT / "examples"


# ---------------------------------------------------------------------------
# 配置: 11 个 Phase A 候选 factor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorSpec:
    label: str           # heatmap y 轴显示
    csv_path: Path       # 输入 csv
    factor_col: str      # csv 中 factor 名所在列 (factor / factor_signed / variant)
    factor_name: str     # 该列要 filter 的值
    month_col: str       # csv 中月份所在列 (month_end / month_start / asof_date)
    sign: int            # +1 / -1, 用于 signed alignment (期望方向)


MONTHLY_CSVS: Final[list[FactorSpec]] = [
    FactorSpec(
        label="IBS_60d (+)",
        csv_path=EXAMPLES / "factor_ic_ibs_csi300_is_monthly.csv",
        factor_col="factor_signed",
        factor_name="IBS_60d_mean",
        month_col="month_end",
        sign=+1,
    ),
    FactorSpec(
        label="count_change_12m (-)",
        csv_path=EXAMPLES / "factor_ic_shareholders_is_monthly.csv",
        factor_col="factor",
        factor_name="count_change_12m",
        month_col="asof_date",
        sign=-1,
    ),
    FactorSpec(
        label="industry_adj_ret_60d (+)",
        csv_path=EXAMPLES / "factor_ic_industry_adj_ret_is_monthly.csv",
        factor_col="factor",
        factor_name="industry_adj_ret_60d_sign+",
        month_col="asof_date",
        sign=+1,
    ),
    FactorSpec(
        label="vol_z_5d (+)",
        csv_path=EXAMPLES / "v20_volume_zscore_is_monthly.csv",
        factor_col="variant",
        factor_name="vol_z_5d_sign+1",
        month_col="month_start",
        sign=+1,
    ),
    FactorSpec(
        label="net_super_big_5d_chg (-)",
        csv_path=EXAMPLES / "super_big_net_is_monthly.csv",
        factor_col="factor_signed",
        factor_name="net_super_big_5d_chg__neg",
        month_col="month_end",
        sign=+1,  # 因子名带 __neg 后缀, csv 已 signed, 不再翻倍
    ),
    FactorSpec(
        label="margin_5d_chg (-)",
        csv_path=EXAMPLES / "factor_ic_technical_csi300_is_monthly.csv",
        factor_col="factor",
        factor_name="margin_5d_chg",
        month_col="month_start",
        sign=-1,
    ),
    FactorSpec(
        label="margin_20d_chg (-)",
        csv_path=EXAMPLES / "factor_ic_technical_csi300_is_monthly.csv",
        factor_col="factor",
        factor_name="margin_20d_chg",
        month_col="month_start",
        sign=-1,
    ),
    FactorSpec(
        label="net_buy_pct_evt (+)",
        csv_path=EXAMPLES / "factor_ic_technical_csi300_is_monthly.csv",
        factor_col="factor",
        factor_name="net_buy_pct_evt",
        month_col="month_start",
        sign=+1,
    ),
    FactorSpec(
        label="unlock_pct_next_60 (-)",
        csv_path=EXAMPLES / "factor_ic_unlock_csi300_is_monthly.csv",
        factor_col="factor",
        factor_name="unlock_pct_next_60",
        month_col="asof_date",
        sign=-1,
    ),
    FactorSpec(
        label="combo_neg_pct (+)",
        csv_path=EXAMPLES / "factor_ic_unlock_csi300_is_monthly.csv",
        factor_col="factor",
        factor_name="combo_neg_pct",
        month_col="asof_date",
        sign=+1,
    ),
    FactorSpec(
        label="revenue_yoy (+)",
        csv_path=EXAMPLES / "factor_ic_fundamentals_csi300_is_monthly.csv",
        factor_col="factor",
        factor_name="revenue_yoy",
        month_col="month_start",
        sign=+1,
    ),
]


# regime shift 关键时点 (label 显示在 heatmap 顶部)
REGIME_LINES: Final[list[tuple[pd.Timestamp, str]]] = [
    (pd.Timestamp("2020-03-01"), "COVID"),
    (pd.Timestamp("2021-05-01"), "OOS 起"),
    (pd.Timestamp("2023-01-01"), "注册制扩"),
    (pd.Timestamp("2024-09-01"), "政策市"),
]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def _load_factor_series(spec: FactorSpec) -> pd.Series:
    """读 long-format csv, filter 指定 factor, 返回 (month_start -> ic*sign)."""
    df = pd.read_csv(spec.csv_path)
    if spec.factor_col not in df.columns:
        return pd.Series(dtype=float, name=spec.label)
    sub = df[df[spec.factor_col] == spec.factor_name].copy()
    if sub.empty:
        return pd.Series(dtype=float, name=spec.label)
    sub[spec.month_col] = pd.to_datetime(sub[spec.month_col], errors="coerce")
    sub = sub.dropna(subset=[spec.month_col, "ic"])
    if sub.empty:
        return pd.Series(dtype=float, name=spec.label)
    # 月份统一对齐到月初 (1 号), 防止 month_end vs month_start 错位
    sub["month_aligned"] = sub[spec.month_col].dt.to_period("M").dt.to_timestamp()
    series = (
        sub.groupby("month_aligned")["ic"].mean() * spec.sign
    )
    series.name = spec.label
    return series


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def _build_panel() -> tuple[pd.DataFrame, list[FactorSpec]]:
    """返回 (DataFrame [factor × month], 实际命中的 spec 列表)."""
    data: dict[str, pd.Series] = {}
    hit_specs: list[FactorSpec] = []
    for spec in MONTHLY_CSVS:
        if not spec.csv_path.exists():
            continue
        try:
            series = _load_factor_series(spec)
        except Exception:  # noqa: BLE001 — 单 factor 失败不阻塞整图
            continue
        if not series.empty:
            data[spec.label] = series
            hit_specs.append(spec)

    if not data:
        return pd.DataFrame(), []

    df = pd.DataFrame(data).T  # rows = factor, cols = month
    df = df.sort_index(axis=1)
    return df, hit_specs


def build_factor_ic_heatmap_fig() -> plt.Figure | None:
    """11 factor × N month heatmap. NaN gap 显示为浅灰."""
    df, _hit_specs = _build_panel()
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, max(3.0, 0.42 * len(df))))

    # 缺失值的底色 (浅灰), 主体红绿 diverging
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#dddddd")

    masked = df.where(df.notna())
    vmin, vmax = -0.2, 0.2
    im = ax.imshow(
        masked.values, cmap=cmap,
        vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest",
    )

    # y 轴: factor name
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index.tolist(), fontsize=9)

    # x 轴: 每 12 月一个 year tick
    months = df.columns
    year_ticks = [i for i, m in enumerate(months) if m.month == 1]
    year_labels = [str(months[i].year) for i in year_ticks]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, rotation=0, fontsize=9)

    ax.set_title(
        f"Phase A 候选 factor 月度 IC heatmap (signed alignment, N={len(df)} factors)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Month")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("IC (signed; 绿 = 因子方向正确)", fontsize=9)

    # regime shift 黑色虚线 + 顶部 label
    months_idx = {m: i for i, m in enumerate(months)}
    for ts, label in REGIME_LINES:
        ts_month = ts.to_period("M").to_timestamp()
        if ts_month in months_idx:
            idx = months_idx[ts_month]
        else:
            # 找最接近的月份
            diffs = [(abs((m - ts_month).days), i) for i, m in enumerate(months)]
            diffs.sort()
            if not diffs or diffs[0][0] > 45:
                continue
            idx = diffs[0][1]
        ax.axvline(idx, color="black", linestyle=":", linewidth=1.0, alpha=0.65)
        ax.text(
            idx, -0.7, label, fontsize=8, ha="center", va="bottom",
            alpha=0.85, color="black",
        )

    fig.tight_layout()
    return fig


def _summary_table(df: pd.DataFrame) -> str:
    """每 factor 一行: 月数 / IC mean / IC pos% / 起止月. 按 IC mean 降序."""
    rows = []
    for label, series in df.iterrows():
        vals = series.dropna()
        if vals.empty:
            continue
        rows.append({
            "factor": label,
            "n_months": len(vals),
            "ic_mean": vals.mean(),
            "ic_pos_pct": (vals > 0).mean() * 100,
            "start": vals.index.min().strftime("%Y-%m"),
            "end": vals.index.max().strftime("%Y-%m"),
        })
    if not rows:
        return ""
    summary = (
        pd.DataFrame(rows)
        .sort_values("ic_mean", ascending=False)
        .reset_index(drop=True)
    )

    body = []
    for _, r in summary.iterrows():
        ic_mean = r["ic_mean"]
        color = "var(--green)" if ic_mean > 0 else "var(--red)"
        body.append(
            "<tr>"
            f"<td>{r['factor']}</td>"
            f"<td>{int(r['n_months'])}</td>"
            f'<td style="color:{color}; font-weight:500;">{ic_mean:+.4f}</td>'
            f"<td>{r['ic_pos_pct']:.1f}%</td>"
            f"<td>{r['start']} → {r['end']}</td>"
            "</tr>"
        )
    return (
        '<table class="data" style="margin-top:8px;">'
        "<thead><tr>"
        "<th>Factor (signed)</th><th>N months</th><th>IC mean</th>"
        "<th>IC > 0 %</th><th>Coverage</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


def _interpretation_html(df: pd.DataFrame, hit_specs: list[FactorSpec]) -> str:
    """heatmap 下方解读 (固定 + 数据驱动 hint)."""
    n_factors = len(df)
    n_months = df.shape[1]
    n_csvs = len({s.csv_path.name for s in hit_specs})
    month_start = df.columns.min().strftime("%Y-%m") if n_months else "—"
    month_end = df.columns.max().strftime("%Y-%m") if n_months else "—"
    return (
        '<ol style="margin-top:12px; font-size:13px; line-height:1.7;">'
        f"<li><b>数据规模</b>: {n_factors} factors × {n_months} months "
        f"({month_start} → {month_end}), 来自 {n_csvs} 个 monthly IC csv. "
        "Phase A 候选因子全 signed 对齐 — 绿色 = 期望方向, 红色 = 反向.</li>"
        "<li><b>跨期稳定 vs 单段强</b>: 长期偏绿的行 (e.g. margin / unlock) = 跨多 regime "
        "稳定 alpha 候选; 仅某段偏绿的行 = 时段性效应, 需警惕 regime change 后失效.</li>"
        "<li><b>regime shift 黑虚线</b>: COVID (2020-03) / OOS 起 (2021-05) / "
        "注册制扩 (2023-01) / 政策市 (2024-09). "
        "若线两侧颜色翻转, 该 factor 已被 regime 改写, 不可外推.</li>"
        "<li><b>本图为 IS 期</b> (Phase A 数据截至 2020-12), OOS 段需另外 Phase B "
        "数据补齐 (留 follow-up). 静态 IC mean 见 model_leaderboard, heatmap 互补展示<b>动态</b>.</li>"
        "</ol>"
    )


def build_factor_ic_heatmap_section() -> str:
    """主入口: 返回 HTML fragment (heatmap + 解读 + summary table)."""
    df, hit_specs = _build_panel()
    if df.empty:
        return (
            '<div class="placeholder-content">'
            "factor IC monthly csv 未找到或全为空. 期望路径: "
            "<code>examples/factor_ic_*_is_monthly.csv</code> + "
            "<code>examples/super_big_net_is_monthly.csv</code> + "
            "<code>examples/v20_volume_zscore_is_monthly.csv</code>."
            "</div>"
        )

    fig = build_factor_ic_heatmap_fig()
    if fig is None:
        return (
            '<div class="placeholder-content">'
            "factor IC heatmap 渲染失败 (panel 非空但 figure build 返回 None)."
            "</div>"
        )
    img_uri = fig_to_base64(fig)
    plt.close(fig)

    return (
        f'<img src="{img_uri}" alt="Phase A factor 月度 IC heatmap"/>'
        f"{_interpretation_html(df, hit_specs)}"
        '<div style="margin-top:16px;">'
        '<div style="font-size:13px; color:var(--muted); margin-bottom:4px;">'
        "Per-factor IS 概要 (按 IC mean 降序):"
        "</div>"
        f"{_summary_table(df)}"
        "</div>"
    )
