"""Universe expansion progress panel — vertical bar chart of 3 universe sizes vs kline coverage.

数据源 (只读):
  - data_cache/qlib_baidu/instruments/csi300.txt   (300 codes, format: `SH600317\t2014-01-01\t2099-12-31`)
  - data_cache/qlib_baidu/instruments/csi500.txt   (494)
  - data_cache/qlib_baidu/instruments/top1500_no_st.txt  (1483)
  - data_cache/baidu_kline.parquet — 'code' column (no SH/SZ prefix, e.g. '600317')

Coverage 计算: 把 instrument 文件 code 去掉前 2 字符 (SH/SZ) → 跟 baidu_kline `code` 求交.

颜色编码:
  - ready (cov >= 95%) — green
  - partial (50% <= cov < 95%) — warn (amber)
  - pending (cov < 50%) — muted gray

输出: <img> 内嵌 base64 matplotlib bar chart (与本仓库其他 chart 一致).
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
INSTRUMENTS_DIR = ROOT / "data_cache" / "qlib_baidu" / "instruments"
KLINE_PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"

UNIVERSES: list[tuple[str, str, str]] = [
    # (label, filename, role)
    ("CSI300", "csi300.txt", "production"),
    ("CSI500", "csi500.txt", "expansion"),
    ("top1500_no_st", "top1500_no_st.txt", "historical"),
]

COLOR_READY = "#16a34a"
COLOR_PARTIAL = "#f59e0b"
COLOR_PENDING = "#6b7280"


def _read_universe_codes(path: Path) -> set[str]:
    """parse instrument file → set of 6-digit codes (strip SH/SZ/BJ prefix)."""
    codes: set[str] = set()
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            tok = line.split()[0]
            if len(tok) >= 2 and tok[:2] in ("SH", "SZ", "BJ"):
                codes.add(tok[2:])
            else:
                codes.add(tok)
    return codes


def _read_kline_codes() -> set[str]:
    df = pd.read_parquet(KLINE_PARQUET, columns=["code"])
    return set(df["code"].unique())


def _pick_color(cov_pct: float) -> str:
    if cov_pct >= 95.0:
        return COLOR_READY
    if cov_pct >= 50.0:
        return COLOR_PARTIAL
    return COLOR_PENDING


def _status_label(cov_pct: float) -> str:
    if cov_pct >= 95.0:
        return "ready"
    if cov_pct >= 50.0:
        return "partial"
    return "pending"


def build_universe_progress_section() -> str:
    """Build vertical bar chart + summary table."""
    kline_codes = _read_kline_codes()

    rows: list[dict] = []
    for label, fname, role in UNIVERSES:
        p = INSTRUMENTS_DIR / fname
        if not p.exists():
            rows.append({
                "label": label, "size": 0, "covered": 0, "cov_pct": 0.0,
                "role": role, "status": "missing",
            })
            continue
        u_codes = _read_universe_codes(p)
        cov = u_codes & kline_codes
        size = len(u_codes)
        covered = len(cov)
        cov_pct = (100.0 * covered / size) if size else 0.0
        rows.append({
            "label": label, "size": size, "covered": covered,
            "cov_pct": cov_pct, "role": role, "status": _status_label(cov_pct),
        })

    # paired bar chart per universe (universe size vs kline-covered).
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=110)
    labels = [r["label"] for r in rows]
    sizes = [r["size"] for r in rows]
    covered_vals = [r["covered"] for r in rows]

    x = np.arange(len(labels))
    width = 0.36
    max_y = max(sizes) if sizes else 1

    ax.bar(x - width / 2, sizes, width, label="universe size",
           color="#94a3b8", edgecolor="#475569", linewidth=0.5)
    bars_cov = ax.bar(x + width / 2, covered_vals, width, label="kline covered",
                      color=[_pick_color(r["cov_pct"]) for r in rows],
                      edgecolor="#1f2937", linewidth=0.5)

    for i, bar in enumerate(bars_cov):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max_y * 0.01,
                f"{rows[i]['cov_pct']:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=_pick_color(rows[i]["cov_pct"]))

    for i, s in enumerate(sizes):
        ax.text(x[i] - width / 2, s + max_y * 0.01, str(s),
                ha="center", va="bottom", fontsize=9, color="#475569")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("stock count")
    ax.set_title("Universe expansion vs baidu_kline coverage")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max_y * 1.12)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    img_html = f'<img src="data:image/png;base64,{img_b64}" alt="universe progress"/>'

    rows_html = ""
    for r in rows:
        color = _pick_color(r["cov_pct"])
        rows_html += (
            "<tr>"
            f"<td><strong>{r['label']}</strong></td>"
            f"<td>{r['role']}</td>"
            f"<td style='text-align:right;'>{r['size']}</td>"
            f"<td style='text-align:right;'>{r['covered']}</td>"
            f"<td style='text-align:right; color:{color}; font-weight:600;'>"
            f"{r['cov_pct']:.1f}%</td>"
            f"<td style='color:{color}; font-weight:600;'>{r['status']}</td>"
            "</tr>"
        )

    table_html = f"""
<table class='data' style='margin-top:14px;'>
  <thead><tr>
    <th>Universe</th><th>Role</th><th style='text-align:right;'>Size</th>
    <th style='text-align:right;'>Covered</th>
    <th style='text-align:right;'>Coverage%</th><th>Status</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<div style='font-size:11px; color:var(--muted, #6b7280); margin-top:8px;'>
  Coverage% = baidu_kline.parquet 中存在的 code 比例.
  Threshold: <span style='color:{COLOR_READY};font-weight:600;'>ready ≥ 95%</span> &middot;
  <span style='color:{COLOR_PARTIAL};font-weight:600;'>partial 50-95%</span> &middot;
  <span style='color:{COLOR_PENDING};font-weight:600;'>pending &lt; 50%</span>
</div>
"""
    return img_html + table_html
