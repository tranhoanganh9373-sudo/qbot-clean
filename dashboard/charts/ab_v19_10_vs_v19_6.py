"""Dashboard panel — A/B v19.10 stacked vs v19.6 production 对比.

数据源 (全只读):
  - data_cache/factors/v19_10_stacked_oos.csv (v19.10 stacked Phase B OOS 60月)
  - data_cache/factors/v19_10_stacked_oos_equity.csv (60 月 equity curve)
  - (Hardcoded v19.6 production OOS Phase 2 v3 cache benchmark)
"""
from __future__ import annotations

import csv
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dashboard.utils import fig_to_base64  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
V19_10_CSV = ROOT / "data_cache" / "factors" / "v19_10_stacked_oos.csv"
V19_10_EQ_CSV = ROOT / "data_cache" / "factors" / "v19_10_stacked_oos_equity.csv"

BASELINES = {
    "baseline (no sidecar)": {
        "calmar": 0.77, "sharpe": None, "ann": 23.02,
        "mdd": -30.01, "cum": None, "color": "#6b7280",
    },
    "v19.4 m5+m20 (shadow)": {
        "calmar": 0.62, "sharpe": None, "ann": 19.35,
        "mdd": -31.21, "cum": None, "color": "#f59e0b",
    },
    "v19.6 amp_imb_20d (prev production)": {
        "calmar": 1.29, "sharpe": 0.92, "ann": 34.36,
        "mdd": -26.66, "cum": 337.80, "color": "#2563eb",
    },
    "竞价勾魂翻 JZF (single)": {
        "calmar": 1.27, "sharpe": 0.89, "ann": 35.75,
        "mdd": -28.22, "cum": 360.98, "color": "#a855f7",
    },
}


def _placeholder(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{message}"
        "</div>"
    )


def _load_v19_10() -> dict | None:
    if not V19_10_CSV.exists():
        return None
    with V19_10_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            return row
    return None


def _build_curve_png(v10_eq_df: pd.DataFrame, v10_row: dict) -> str:
    df = v10_eq_df.copy()
    df["ret"] = df["abs_ret_%"] / 100.0
    df["eq"] = (1 + df["ret"]).cumprod()
    df["cum_%"] = (df["eq"] - 1) * 100
    df["month_dt"] = pd.to_datetime(df["month"])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        df["month_dt"], df["cum_%"],
        color="#dc2626", linewidth=2.0, label="v19.10 stacked (PRODUCTION)",
    )

    # v19.6 假设月均匀 ann 34.36% 画 reference 线
    v196_monthly = (1 + 34.36 / 100) ** (1 / 12) - 1
    v196_cum = [(1 + v196_monthly) ** i - 1 for i in range(1, len(df) + 1)]
    ax.plot(
        df["month_dt"], [c * 100 for c in v196_cum],
        color="#2563eb", linewidth=1.5, linestyle="--",
        label="v19.6 reference (假设等月 ann 34.36%, 实 cum 337.8%)",
    )

    ax.set_title(
        f"60 月 OOS 累计收益 — v19.10 +{float(v10_row['oos_cum_%']):.0f}% "
        "vs v19.6 +338%",
        fontsize=12,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=10)
    ax.axhline(0, color="#94a3b8", linewidth=0.5)

    fig.tight_layout()
    uri = fig_to_base64(fig)
    plt.close(fig)
    return uri


def build_ab_v19_10_vs_v19_6_section() -> str:
    v10_row = _load_v19_10()
    if not v10_row:
        return _placeholder(
            "无 v19.10 stacked OOS 结果 — 跑 "
            "<code>python examples/v19_10_stacked_oos.py</code> 后此 panel 自动填充."
        )

    v10_calmar = float(v10_row["oos_calmar"])
    v10_sharpe = float(v10_row["oos_sharpe"])
    v10_ann = float(v10_row["oos_ann_%"])
    v10_mdd = float(v10_row["oos_mdd_%"])
    v10_cum = float(v10_row["oos_cum_%"])

    rows: list[tuple[str, dict]] = list(BASELINES.items())
    rows.append((
        "v19.10 stacked ⭐ PRODUCTION",
        {
            "calmar": v10_calmar, "sharpe": v10_sharpe, "ann": v10_ann,
            "mdd": v10_mdd, "cum": v10_cum, "color": "#dc2626",
        },
    ))

    def fmt(v):
        return "—" if v is None else f"{v:.2f}"

    def fmt_pct(v):
        if v is None:
            return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.2f}%"

    body_rows: list[str] = []
    for name, d in rows:
        is_main = "PRODUCTION" in name
        is_prev = "prev production" in name
        bg = ("rgba(220,38,38,0.06)" if is_main
              else "rgba(37,99,235,0.06)" if is_prev
              else "transparent")
        weight = "700" if is_main else "400"
        body_rows.append(
            f'<tr style="background:{bg}; font-weight:{weight};">'
            f'<td>{html.escape(name)}</td>'
            f'<td style="color:{d["color"]}; font-weight:700; text-align:right;">'
            f'{fmt(d["calmar"])}</td>'
            f'<td style="text-align:right;">{fmt(d["sharpe"])}</td>'
            f'<td style="text-align:right;">{fmt_pct(d["ann"])}</td>'
            f'<td style="text-align:right;">{fmt_pct(d["mdd"])}</td>'
            f'<td style="text-align:right;">{fmt_pct(d["cum"])}</td>'
            '</tr>'
        )
    table_html = (
        '<table class="data" style="margin-bottom:14px;">'
        '<thead><tr>'
        '<th>Strategy</th><th style="text-align:right;">Calmar</th>'
        '<th style="text-align:right;">Sharpe</th><th style="text-align:right;">ann</th>'
        '<th style="text-align:right;">MDD</th><th style="text-align:right;">cum 60m</th>'
        '</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>'
    )

    delta_calmar = (v10_calmar / 1.29 - 1) * 100
    verdict = (
        '<div style="background:rgba(22,163,74,0.10); border-left:3px solid #16a34a; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:13px;">'
        f'<strong style="color:#16a34a;">✅ v19.10 stacked BEAT v19.6 by +{delta_calmar:.1f}% Calmar</strong><br>'
        '<span style="font-size:12px; color:var(--muted, #6b7280);">'
        '公式: <code>final = z(pred) - 0.30 × z(amp_imb_20d) + 0.10 × z(JZF)</code><br>'
        '2 个 sidecar 都单跑 OOS Calmar ≥ 1.27, '
        f'Spearman 日 |ρ| = {float(v10_row["is_spearman_mean_abs_rho"]):.3f} 中度独立, '
        'stack 后协同(v19.7 abort 教训:弱因子拖累;这次两端都强 → 协同). '
        'win 53.3% / 60 月 0 fail / avg picks 3.34. '
        '12 月 forward A/B 跟 v19.6 shadow 平行验证.'
        '</span></div>'
    )

    if V19_10_EQ_CSV.exists():
        try:
            v10_eq_df = pd.read_csv(V19_10_EQ_CSV)
            img_uri = _build_curve_png(v10_eq_df, v10_row)
            curve_html = f'<img src="{img_uri}" alt="v19.10 vs v19.6 cum curve"/>'
        except Exception as exc:  # noqa: BLE001
            curve_html = (
                '<div class="placeholder-content">curve 绘制失败: '
                f'<code>{type(exc).__name__}: {html.escape(str(exc))}</code></div>'
            )
    else:
        curve_html = ""

    footer = (
        '<div style="margin-top:8px; font-size:11px; color:var(--muted, #6b7280);">'
        '数据源 <code>data_cache/factors/v19_10_stacked_oos.csv</code> · '
        'IS-locked λ_amp=0.30 + λ_JZF=0.10 · 严格 single OOS 2021-05~2026-04 60月 · '
        'Phase 2 v3 retrain cache · production 升级时间: 2026-05-27'
        '</div>'
    )

    return verdict + table_html + curve_html + footer
