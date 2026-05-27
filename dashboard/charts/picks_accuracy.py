"""Dashboard panel — picks 历史 T+1/T+5 准确率 vs CSI300 benchmark.

每条 BUY action: 看 buy_price → T+1/T+5 close 收益, 跟 CSI300 同期对比算 alpha,
量化 forward alpha 真实性 (production 真在赚钱吗 vs benchmark).

数据源 (全只读):
  - data_cache/paper_trade_log.csv (BUY actions: date, sym, name, price)
  - data_cache/baidu_kline.parquet (个股 T+1/T+5 close)
  - data_cache/index_kline.parquet (sh000300 同期 close 算 CSI300 benchmark)
"""
from __future__ import annotations

import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TRADE_LOG = ROOT / "data_cache" / "paper_trade_log.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
INDEX_PATH = ROOT / "data_cache" / "index_kline.parquet"


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _sym_to_code(sym: str) -> str:
    return sym[2:] if len(sym) >= 8 and sym[:2] in ("SH", "SZ") else sym


def _build_kline_lookup() -> dict:
    if not KLINE_PATH.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"])
        lookup: dict = {}
        for code, sub in df.groupby("code", observed=True):
            lookup[str(code)] = list(zip(sub["date"].tolist(), sub["close"].tolist()))
        return lookup
    except Exception:
        return {}


def _build_csi300_lookup() -> list[tuple]:
    if not INDEX_PATH.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_parquet(INDEX_PATH)
        df = df[df["code"] == "sh000300"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return list(zip(df["date"].tolist(), df["close"].tolist()))
    except Exception:
        return []


def _close_at_offset(timeline: list[tuple], buy_date, n_days: int) -> float | None:
    import pandas as pd
    if not timeline:
        return None
    bd = pd.to_datetime(buy_date)
    idx = None
    for i, (d, _) in enumerate(timeline):
        if d >= bd:
            idx = i
            break
    if idx is None:
        return None
    target_idx = idx + n_days
    if target_idx >= len(timeline):
        return None
    return float(timeline[target_idx][1])


def _compute_accuracy_rows() -> list[dict] | None:
    if not TRADE_LOG.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(TRADE_LOG)
        df = df[df["action"] == "BUY"].copy()
    except Exception:
        return None
    if len(df) == 0:
        return []
    kline = _build_kline_lookup()
    csi300 = _build_csi300_lookup()
    rows: list[dict] = []
    for _, r in df.iterrows():
        sym = r["symbol"]
        code = _sym_to_code(sym)
        buy_price = float(r["price"])
        buy_date = r["date"]
        timeline = kline.get(code, [])
        if not timeline:
            continue
        c_t1 = _close_at_offset(timeline, buy_date, 1)
        c_t5 = _close_at_offset(timeline, buy_date, 5)
        ret_t1 = ((c_t1 / buy_price) - 1) * 100 if c_t1 else None
        ret_t5 = ((c_t5 / buy_price) - 1) * 100 if c_t5 else None
        idx_buy = _close_at_offset(csi300, buy_date, 0)
        idx_t1 = _close_at_offset(csi300, buy_date, 1)
        idx_t5 = _close_at_offset(csi300, buy_date, 5)
        csi_t1 = ((idx_t1 / idx_buy) - 1) * 100 if (idx_t1 and idx_buy) else None
        csi_t5 = ((idx_t5 / idx_buy) - 1) * 100 if (idx_t5 and idx_buy) else None
        alpha_t1 = (ret_t1 - csi_t1) if (ret_t1 is not None and csi_t1 is not None) else None
        alpha_t5 = (ret_t5 - csi_t5) if (ret_t5 is not None and csi_t5 is not None) else None
        # A 股日内涨跌停 +/-10% (主板) / +/-20% (创业板/科创板);
        # |ret| > 25% = 除权除息/送转/拆股 artifact, 不是真实涨跌.
        # 标记 artifact 行, 统计时排除.
        EXTREME_THRESHOLD = 25.0
        is_artifact = (
            (ret_t1 is not None and abs(ret_t1) > EXTREME_THRESHOLD)
            or (ret_t5 is not None and abs(ret_t5) > EXTREME_THRESHOLD)
        )
        rows.append({
            "date": buy_date,
            "sym": sym,
            "name": r.get("name", ""),
            "buy_price": buy_price,
            "ret_t1": ret_t1, "ret_t5": ret_t5,
            "csi_t1": csi_t1, "csi_t5": csi_t5,
            "alpha_t1": alpha_t1, "alpha_t5": alpha_t5,
            "is_artifact": is_artifact,
        })
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}%"


def build_picks_accuracy_section() -> str:
    rows = _compute_accuracy_rows()
    if rows is None:
        return _placeholder(
            "无法读 <code>paper_trade_log.csv</code> 或 kline 数据."
        )
    if not rows:
        return _placeholder(
            "<code>paper_trade_log.csv</code> 中无 BUY actions — 尚无历史 picks 可评估."
        )

    # 统计排除 artifact rows (除权日跳价 distort 真实 alpha)
    valid_rows = [r for r in rows if not r.get("is_artifact")]
    n_artifact = sum(1 for r in rows if r.get("is_artifact"))
    n_t1 = sum(1 for r in valid_rows if r["alpha_t1"] is not None)
    n_t5 = sum(1 for r in valid_rows if r["alpha_t5"] is not None)
    avg_alpha_t1 = (
        sum(r["alpha_t1"] for r in valid_rows if r["alpha_t1"] is not None) / n_t1
    ) if n_t1 > 0 else 0
    avg_alpha_t5 = (
        sum(r["alpha_t5"] for r in valid_rows if r["alpha_t5"] is not None) / n_t5
    ) if n_t5 > 0 else 0
    hit_t1 = sum(1 for r in valid_rows if r["alpha_t1"] is not None and r["alpha_t1"] > 0)
    hit_t5 = sum(1 for r in valid_rows if r["alpha_t5"] is not None and r["alpha_t5"] > 0)
    hit_rate_t1 = (hit_t1 / n_t1 * 100) if n_t1 > 0 else 0
    hit_rate_t5 = (hit_t5 / n_t5 * 100) if n_t5 > 0 else 0

    alpha_t1_color = "#16a34a" if avg_alpha_t1 > 0 else "#dc2626"
    alpha_t5_color = "#16a34a" if avg_alpha_t5 > 0 else "#dc2626"

    artifact_note = (
        f' · <span style="color:#dc2626;">⚠ 已排除 {n_artifact} 笔除权 artifact '
        '(|ret| &gt; 25%, 高送转/分红日跳价)</span>'
        if n_artifact > 0 else ""
    )
    banner = (
        '<div style="background:rgba(107,114,128,0.08); border-left:3px solid #6b7280; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        f'<strong>📈 Forward Alpha</strong> · '
        f'BUY actions <strong>{len(rows)}</strong> 累计 '
        f'(T+1 评估 {n_t1} 笔 / T+5 评估 {n_t5} 笔){artifact_note}<br>'
        f'平均 α(T+1) <strong style="color:{alpha_t1_color};">'
        f'{avg_alpha_t1:+.2f}%</strong> · 击败 CSI300 <strong>{hit_rate_t1:.0f}%</strong> · '
        f'平均 α(T+5) <strong style="color:{alpha_t5_color};">'
        f'{avg_alpha_t5:+.2f}%</strong> · 击败 CSI300 <strong>{hit_rate_t5:.0f}%</strong><br>'
        '<span style="color:var(--muted, #6b7280); font-size:11px;">'
        "alpha = (个股 return) - (CSI300 同期 return). 正值 = production picks 跑赢大盘. "
        "样本少时 noise 大, 累积 3-6 月后看真实 forward alpha."
        "</span></div>"
    )

    cols = [
        ("#", 3), ("买入日", 9), ("sym", 7), ("名称", 14), ("买入价", 7),
        ("T+1 个股", 10), ("T+1 大盘", 8), ("T+1 α", 8),
        ("T+5 个股", 10), ("T+5 大盘", 8), ("T+5 α", 8), ("评估", 8),
    ]
    assert sum(w for _, w in cols) == 100, sum(w for _, w in cols)
    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{w}%;">' for _, w in cols
    ) + "</colgroup>"
    right_cols = {
        "买入价", "T+1 个股", "T+1 大盘", "T+1 α",
        "T+5 个股", "T+5 大盘", "T+5 α",
    }
    thead = "<tr>" + "".join(
        f'<th style="text-align:right;">{l}</th>' if l in right_cols
        else f"<th>{l}</th>"
        for l, _ in cols
    ) + "</tr>"

    body_rows: list[str] = []
    for i, r in enumerate(rows, 1):
        a1 = r["alpha_t1"]
        a5 = r["alpha_t5"]
        a1_c = (
            "#16a34a" if a1 is not None and a1 > 0
            else "#dc2626" if a1 is not None and a1 < 0
            else "#6b7280"
        )
        a5_c = (
            "#16a34a" if a5 is not None and a5 > 0
            else "#dc2626" if a5 is not None and a5 < 0
            else "#6b7280"
        )
        is_art = r.get("is_artifact", False)
        if is_art:
            verdict = "⚠ 除权日"
            verdict_c = "#f59e0b"
        elif a5 is not None:
            verdict = "✓ 跑赢" if a5 > 0 else "✗ 跑输"
            verdict_c = "#16a34a" if a5 > 0 else "#dc2626"
        elif a1 is not None:
            verdict = "✓ 跑赢" if a1 > 0 else "✗ 跑输"
            verdict_c = "#16a34a" if a1 > 0 else "#dc2626"
        else:
            verdict = "—"
            verdict_c = "#6b7280"
        row_style = ' style="opacity:0.45;"' if is_art else ""
        body_rows.append(
            f"<tr{row_style}>"
            f"<td>{i}</td>"
            f'<td style="font-size:11px;">{html.escape(str(r["date"]))}</td>'
            f"<td><code>{html.escape(r['sym'])}</code></td>"
            f"<td>{html.escape(str(r.get('name','')))}</td>"
            f'<td style="text-align:right;">{r["buy_price"]:.2f}</td>'
            f'<td style="text-align:right;">{_fmt_pct(r["ret_t1"])}</td>'
            f'<td style="text-align:right; font-size:11px; color:var(--muted, #6b7280);">'
            f'{_fmt_pct(r["csi_t1"])}</td>'
            f'<td style="text-align:right; color:{a1_c}; font-weight:600;">'
            f'{_fmt_pct(a1)}</td>'
            f'<td style="text-align:right;">{_fmt_pct(r["ret_t5"])}</td>'
            f'<td style="text-align:right; font-size:11px; color:var(--muted, #6b7280);">'
            f'{_fmt_pct(r["csi_t5"])}</td>'
            f'<td style="text-align:right; color:{a5_c}; font-weight:600;">'
            f'{_fmt_pct(a5)}</td>'
            f'<td style="color:{verdict_c}; font-weight:600; font-size:11px;">'
            f'{verdict}</td>'
            "</tr>"
        )

    table = (
        '<table class="data">'
        + colgroup
        + f"<thead>{thead}</thead>"
        + f"<tbody>{''.join(body_rows)}</tbody>"
        + "</table>"
    )

    footer = (
        '<div style="margin-top:10px; font-size:11px; color:var(--muted, #6b7280);">'
        "T+1 = 买入后第 1 个交易日 close return; T+5 = 第 5 个交易日.<br>"
        "alpha = 个股 return - CSI300 同期 return; 评估列取 T+5 (不可用则 T+1).<br>"
        "数据源 <code>paper_trade_log.csv</code> + <code>baidu_kline.parquet</code> + "
        "<code>index_kline.parquet (sh000300)</code>."
        "</div>"
    )

    return banner + table + footer
