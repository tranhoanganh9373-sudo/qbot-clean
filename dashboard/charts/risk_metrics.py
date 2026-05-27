"""Dashboard panel — Risk metrics (Beta / Vol / 集中度 / 行业 / VaR).

假设 portfolio 等权 (8 picks, 12.5% each, paper_trade_today K=8 默认),
基于 60d 日级收益序列计算各风险指标.

数据源 (全只读):
  - data_cache/portfolio_state.json (holdings)
  - data_cache/baidu_kline.parquet (60d close → daily returns)
  - data_cache/index_kline.parquet (sh000300 CSI300 daily returns)
  - data_cache/industry/industry_membership.parquet (sym → 行业 映射)
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
STATE_PATH = ROOT / "data_cache" / "portfolio_state.json"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
INDEX_PATH = ROOT / "data_cache" / "index_kline.parquet"
INDUSTRY_PATH = ROOT / "data_cache" / "industry" / "industry_membership.parquet"

LOOKBACK_DAYS = 60
CONCENTRATION_ALERT_PCT = 25.0


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def _sym_to_code(sym: str) -> str:
    return sym[2:] if len(sym) >= 8 and sym[:2] in ("SH", "SZ") else sym


def _load_holdings() -> list[str]:
    if not STATE_PATH.exists():
        return []
    try:
        d = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        h = d.get("holdings") or []
        if isinstance(h, list):
            return [s for s in h if isinstance(s, str)]
    except Exception:
        pass
    return []


def _load_returns(holdings: list[str]):
    if not holdings:
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        codes_wanted = {_sym_to_code(s) for s in holdings}
        sub = df[df["code"].isin(codes_wanted)].copy()
        sub = sub.sort_values(["code", "date"]).reset_index(drop=True)
        sub["ret"] = sub.groupby("code", observed=True)["close"].pct_change()
        sub = sub.dropna(subset=["ret"])
        sub = sub.groupby("code", observed=True).tail(LOOKBACK_DAYS)
        ret_df = sub.pivot_table(index="date", columns="code", values="ret")
        return ret_df
    except Exception:
        return None


def _load_csi300_returns():
    try:
        import pandas as pd
        df = pd.read_parquet(INDEX_PATH)
        df = df[df["code"] == "sh000300"].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["ret"] = df["close"].pct_change()
        df = df.dropna(subset=["ret"]).tail(LOOKBACK_DAYS)
        return df.set_index("date")["ret"]
    except Exception:
        return None


def _load_industry_map() -> dict[str, str]:
    if not INDUSTRY_PATH.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(INDUSTRY_PATH)
        return dict(zip(df["code"].astype(str), df["industry_name"]))
    except Exception:
        return {}


def build_risk_metrics_section() -> str:
    holdings = _load_holdings()
    if not holdings:
        return _placeholder("无 holdings — portfolio_state.json 空 / 解析失败.")
    n = len(holdings)
    equal_w = 100.0 / n
    max_single_pct = equal_w
    single_alert = max_single_pct > CONCENTRATION_ALERT_PCT

    ind_map = _load_industry_map()
    industry_counts: dict[str, int] = {}
    for sym in holdings:
        code = _sym_to_code(sym)
        ind = ind_map.get(code) or "未分类"
        industry_counts[ind] = industry_counts.get(ind, 0) + 1
    industry_pct = {ind: (cnt / n) * 100 for ind, cnt in industry_counts.items()}
    industry_sorted = sorted(industry_pct.items(), key=lambda x: -x[1])
    max_industry_pct = industry_sorted[0][1] if industry_sorted else 0
    max_industry_name = industry_sorted[0][0] if industry_sorted else ""
    industry_alert = max_industry_pct > CONCENTRATION_ALERT_PCT

    ret_df = _load_returns(holdings)
    csi_ret = _load_csi300_returns()
    portfolio_vol = None
    portfolio_beta = None
    var95 = None
    n_days = 0
    if ret_df is not None and not ret_df.empty:
        ret_df = ret_df.dropna(how="all")
        port_ret = ret_df.mean(axis=1).dropna()
        n_days = len(port_ret)
        if n_days >= 10:
            import numpy as np
            portfolio_vol = float(port_ret.std()) * (252 ** 0.5) * 100
            var95 = -float(np.percentile(port_ret, 5)) * 100
            if csi_ret is not None and len(csi_ret) >= 10:
                joined = port_ret.to_frame("port").join(
                    csi_ret.to_frame("csi"), how="inner",
                )
                if len(joined) >= 10:
                    cov = joined["port"].cov(joined["csi"])
                    var_csi = joined["csi"].var()
                    if var_csi > 0:
                        portfolio_beta = float(cov / var_csi)

    has_alert = single_alert or industry_alert
    banner_color = "#dc2626" if has_alert else "#16a34a"
    banner_lbl = "⚠ 集中度告警" if has_alert else "✓ 风险可控"
    banner = (
        '<div style="background:rgba(107,114,128,0.08); border-left:3px solid '
        f'{banner_color}; padding:10px 14px; margin-bottom:12px; '
        'border-radius:0 4px 4px 0; font-size:12px;">'
        f'<strong>⚖️ Risk Summary</strong> · <strong>{banner_lbl}</strong> · '
        f'{n} holdings 等权 (单股 {equal_w:.1f}% each)<br>'
        '<span style="color:var(--muted, #6b7280); font-size:11px;">'
        f'60d lookback ({n_days} 天数据可用), Beta vs CSI300, 年化波动率, '
        f'VaR 95% = 历史 5% percentile (单日最差 5% 情形 % 损失), '
        f'集中度告警阈值 {CONCENTRATION_ALERT_PCT}%.'
        "</span></div>"
    )

    def _card(label: str, value: str, color: str, note: str) -> str:
        return (
            '<div style="background:rgba(107,114,128,0.06); padding:10px 12px; '
            f'border-left:3px solid {color}; border-radius:0 4px 4px 0; '
            'flex:1; min-width:160px;">'
            f'<div style="font-size:11px; color:var(--muted, #6b7280);">{label}</div>'
            f'<div style="font-size:20px; font-weight:700; color:{color};">{value}</div>'
            f'<div style="font-size:10px; color:var(--muted, #6b7280); margin-top:2px;">{note}</div>'
            "</div>"
        )

    cards: list[str] = []
    if portfolio_beta is not None:
        beta_color = "#16a34a" if 0.7 <= portfolio_beta <= 1.3 else "#f59e0b"
        beta_note = (
            "贴近 CSI300" if 0.85 <= portfolio_beta <= 1.15 else
            "略 aggressive" if portfolio_beta > 1.3 else
            "略 defensive" if portfolio_beta < 0.7 else
            "中等偏离"
        )
        cards.append(_card("Beta vs CSI300", f"{portfolio_beta:.2f}", beta_color, beta_note))
    else:
        cards.append(_card("Beta vs CSI300", "—", "#6b7280", "数据不足"))

    if portfolio_vol is not None:
        vol_color = (
            "#16a34a" if portfolio_vol < 25 else
            "#f59e0b" if portfolio_vol < 35 else "#dc2626"
        )
        cards.append(_card("年化波动率", f"{portfolio_vol:.1f}%", vol_color,
                          "正常 <25%, 偏高 25-35%, 极高 >35%"))
    else:
        cards.append(_card("年化波动率", "—", "#6b7280", "数据不足"))

    if var95 is not None:
        var_color = (
            "#16a34a" if var95 < 2 else "#f59e0b" if var95 < 4 else "#dc2626"
        )
        cards.append(_card("VaR 95%", f"{var95:.2f}%", var_color,
                          "单日最差 5% 情形损失 %"))
    else:
        cards.append(_card("VaR 95%", "—", "#6b7280", "数据不足"))

    single_color = "#dc2626" if single_alert else "#16a34a"
    cards.append(_card("最大单股 %", f"{max_single_pct:.1f}%", single_color,
                      f"等权 {n} 只, < {CONCENTRATION_ALERT_PCT}% 安全"))

    industry_color = (
        "#dc2626" if industry_alert else
        "#f59e0b" if max_industry_pct > 20 else "#16a34a"
    )
    cards.append(_card("最大行业 %", f"{max_industry_pct:.1f}%", industry_color,
                      html.escape(max_industry_name) if max_industry_name else "未知"))

    cards_html = (
        '<div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px;">'
        + "".join(cards) + "</div>"
    )

    industry_rows: list[str] = []
    for ind, pct in industry_sorted:
        sample_syms = [
            s for s in holdings
            if ind_map.get(_sym_to_code(s)) == ind
            or (ind == "未分类" and ind_map.get(_sym_to_code(s)) is None)
        ]
        ind_alert = pct > CONCENTRATION_ALERT_PCT
        row_bg = "rgba(220,38,38,0.06)" if ind_alert else "transparent"
        alert_tag = (
            ' <span style="color:#dc2626; font-size:10px;">⚠ 超阈值</span>'
            if ind_alert else ""
        )
        industry_rows.append(
            f'<tr style="background:{row_bg};">'
            f'<td>{html.escape(ind)}{alert_tag}</td>'
            f'<td style="text-align:right;">{len(sample_syms)}</td>'
            f'<td style="text-align:right; font-weight:600;">{pct:.1f}%</td>'
            f'<td style="font-size:11px;">'
            f"{', '.join(html.escape(s) for s in sample_syms)}</td>"
            "</tr>"
        )
    industry_table = (
        '<h3 style="font-size:13px; margin:10px 0 6px 0; color:var(--muted, #6b7280);">'
        "🏭 行业暴露 (按 % 倒序)</h3>"
        '<table class="data">'
        '<colgroup><col style="width:22%;"><col style="width:10%;">'
        '<col style="width:10%;"><col style="width:58%;"></colgroup>'
        '<thead><tr><th>行业</th><th style="text-align:right;">持仓数</th>'
        '<th style="text-align:right;">% 占比</th><th>成分股</th></tr></thead>'
        f"<tbody>{''.join(industry_rows)}</tbody></table>"
    )

    footer = (
        '<div style="margin-top:10px; font-size:11px; color:var(--muted, #6b7280);">'
        "假设等权 portfolio. Beta = cov(port,csi) / var(csi); "
        "年化波动率 = daily σ × √252; VaR 95% = historical 5% percentile.<br>"
        "数据源 <code>portfolio_state.json</code> + <code>baidu_kline.parquet</code> + "
        "<code>index_kline.parquet</code> + <code>industry/industry_membership.parquet</code>."
        "</div>"
    )

    return banner + cards_html + industry_table + footer
