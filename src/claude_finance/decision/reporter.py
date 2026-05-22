"""Render Markdown reports from analyze_one() results.

Two renderers:
  - render_decision_report  for a small curated watchlist (per-symbol detail)
  - render_scan_report      for full-market scans (Top BUY / Top SELL tables)
"""
from __future__ import annotations

import pandas as pd

_SIG_EMOJI = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⚪ HOLD"}


def render_decision_report(results: list[dict]) -> str:
    today = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Qbot 中线决策报告  ({today})",
        "",
        "**数据源**: akshare（免费）  ",
        "**策略**: 多策略融合 (BIAS+MACD+KDJ+RSI+BOLL)，按 Qbot 默认权重投票  ",
        "**时间框架**: 日K，中线（2–8周）  ",
        "**风控**: ATR(14) 自适应止损 1.8×ATR，目标 2.0× / 3.5×ATR  ",
        "",
        "---",
        "## ⚠️ 重要声明",
        "本报告为基于公开行情数据的量化策略输出，**不构成投资建议**。所有交易决策与盈亏由您本人承担。",
        "**请务必结合大盘环境、个股基本面、资金管理纪律使用，并严格执行止损。**",
        "",
        "---",
        "## 决策一览",
        "",
        "| 标的 | 代码 | 现价 | 涨跌 | 信号 | 买分 | 卖分 | 60日趋势 | 建议仓位 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | {r['code']} | - | - | ❌ ERROR | - | - | - | - |")
            continue
        chg = f"{r['change_pct']:+.2f}%"
        pos = f"{r['suggested_position_pct']}%" if r["suggested_position_pct"] > 0 else "—"
        lines.append(
            f"| {r['name']} | {r['code']} | {r['price']} | {chg} | {_SIG_EMOJI[r['signal']]} | "
            f"{r['buy_score']} | {r['sell_score']} | {r['trend_ma60']} | {pos} |"
        )

    lines += ["", "---", "## 逐标的可执行清单", ""]
    for r in results:
        if "error" in r:
            lines += [f"### ❌ {r['name']} ({r['code']})", f"数据拉取失败：{r['error']}", ""]
            continue
        lines += [
            f"### {r['name']} ({r['code']}) — {r['signal']}",
            f"- **最新价**: {r['price']}（{r['date']}, {r['change_pct']:+.2f}%）",
            f"- **趋势**: {r['trend_ma60']}（现价相对 MA60 {r['above_ma60_pct']:+.2f}%）",
            f"- **加权得分**: 买 {r['buy_score']} / 卖 {r['sell_score']}",
            "- **各策略投票**: "
            + ", ".join(f"{s}={'+' if v > 0 else ''}{v}" for s, v in r["votes"].items()),
        ]
        if r["signal"] != "HOLD":
            lines += [
                f"- **建议入场区间**: {r['entry_range'][0]} ~ {r['entry_range'][1]}",
                f"- **止损价**: {r['stop_loss']}",
                f"- **目标 1 (减半仓)**: {r['target_1']}",
                f"- **目标 2 (全平)**: {r['target_2']}",
                f"- **建议仓位**: 总资金的 {r['suggested_position_pct']}%",
                f"- **执行**: 在券商 APP 中限价单 {r['entry_range'][0]}–{r['entry_range'][1]} 之间分两笔挂单；"
                f"开仓后立即挂止损 {r['stop_loss']}；到 {r['target_1']} 减半仓；到 {r['target_2']} 全平。",
            ]
        else:
            lines.append("- **执行**: 观望，无开仓动作；如已持仓则继续按原止损执行。")

        ind = r["indicators"]
        lines += [
            "",
            "<details><summary>关键指标快照</summary>",
            "",
            f"- MA5/10/20/60: {ind['MA5']} / {ind['MA10']} / {ind['MA20']} / {ind['MA60']}",
            f"- RSI14: {ind['RSI14']}",
            f"- KDJ: K={ind['K']}, D={ind['D']}, J={ind['J']}",
            f"- MACD: DIF={ind['DIF']}, DEA={ind['DEA']}, HIST={ind['MACD_hist']}",
            f"- BOLL: 上轨={ind['BOLL_upper']}, 中轨={ind['BOLL_mid']}, 下轨={ind['BOLL_lower']}",
            f"- ATR14: {ind['ATR14']}",
            "</details>",
            "",
        ]

    lines += [
        "---",
        "## 资金管理纪律（必读）",
        "1. **单笔风险**: 每笔交易最大亏损不超过总资金的 1.5%",
        "2. **总仓位上限**: 合计仓位不超过 60%，留 40% 现金应对极端行情",
        "3. **止损纪律**: 触及止损价**立刻平仓**，不抗单、不补仓",
        "4. **趋势优先**: 60日趋势为「下降」的 BUY 信号，仓位减半，止损更严（用 1.5×ATR）",
        "5. **复盘节奏**: 每周日重跑此脚本，根据新信号调整持仓",
    ]
    return "\n".join(lines)


def render_scan_report(results: list[dict], top_n: int = 50) -> str:
    today = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M")
    ok = [r for r in results if "error" not in r]
    fail = [r for r in results if "error" in r]

    n_buy = sum(1 for r in ok if r["signal"] == "BUY")
    n_sell = sum(1 for r in ok if r["signal"] == "SELL")
    n_hold = sum(1 for r in ok if r["signal"] == "HOLD")

    buys = [r for r in ok if r["signal"] == "BUY"]
    buys.sort(key=lambda r: (
        -r["buy_score"],
        0 if r["trend_ma60"] == "上升" else 1,
        r["indicators"]["RSI14"],
    ))
    top_buy = buys[:top_n]

    sells = [r for r in ok if r["signal"] == "SELL"]
    sells.sort(key=lambda r: (
        -r["sell_score"],
        0 if r["trend_ma60"] == "下降" else 1,
        -r["indicators"]["RSI14"],
    ))
    top_sell = sells[:top_n]

    lines = [
        f"# Qbot 全市场扫描报告 ({today})",
        "",
        f"**扫描数量**: {len(results)} 只（成功 {len(ok)}，失败 {len(fail)}）  ",
        f"**信号分布**: 🟢 BUY={n_buy}, 🔴 SELL={n_sell}, ⚪ HOLD={n_hold}  ",
        "**策略**: 多策略融合 (BIAS+MACD+KDJ+RSI+BOLL)  ",
        "**时间框架**: 日K，中线（2–8周）  ",
        "",
        "## ⚠️ 重要声明",
        "本报告不构成投资建议。技术信号需结合大盘环境、个股基本面和资金管理纪律使用。",
        "全市场扫描会产生大量信号，**绝不可全部建仓**。",
        "",
        "---",
        f"## 🟢 Top {min(top_n, len(buys))} 买入信号",
        "",
        "排序: 加权买分降序 → 60日上升趋势优先 → RSI 低（超卖）优先",
        "",
        "| 排名 | 代码 | 名称 | 现价 | 涨跌 | 买分 | 趋势 | RSI | KDJ-K | 建议入场 | 止损 | 目标1 | 目标2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(top_buy, 1):
        ind = r["indicators"]
        entry = f"{r['entry_range'][0]}~{r['entry_range'][1]}" if r["entry_range"] else "-"
        lines.append(
            f"| {i} | {r['code']} | {r['name']} | {r['price']} | {r['change_pct']:+.2f}% | "
            f"{r['buy_score']:.3f} | {r['trend_ma60']} | {ind['RSI14']} | {ind['K']} | "
            f"{entry} | {r['stop_loss']} | {r['target_1']} | {r['target_2']} |"
        )

    lines += [
        "",
        "---",
        f"## 🔴 Top {min(top_n, len(sells))} 卖出/做空信号",
        "",
        "排序: 加权卖分降序 → 60日下降趋势优先 → RSI 高（超买）优先",
        "",
        "| 排名 | 代码 | 名称 | 现价 | 涨跌 | 卖分 | 趋势 | RSI | KDJ-K | 建议出场 | 止损 | 目标1 | 目标2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(top_sell, 1):
        ind = r["indicators"]
        entry = f"{r['entry_range'][0]}~{r['entry_range'][1]}" if r["entry_range"] else "-"
        lines.append(
            f"| {i} | {r['code']} | {r['name']} | {r['price']} | {r['change_pct']:+.2f}% | "
            f"{r['sell_score']:.3f} | {r['trend_ma60']} | {ind['RSI14']} | {ind['K']} | "
            f"{entry} | {r['stop_loss']} | {r['target_1']} | {r['target_2']} |"
        )

    if fail:
        lines += [
            "",
            "---",
            f"## ❌ 失败列表（共 {len(fail)} 只，显示前 20）",
            "",
            "| 代码 | 名称 | 失败原因 |",
            "|---|---|---|",
        ]
        for r in fail[:20]:
            lines.append(f"| {r['code']} | {r['name']} | {r.get('error', '?')} |")
        lines += [
            "",
            "失败原因多为网络偶发或上市天数不足。重跑脚本会从缓存继续，只重试失败项。",
        ]

    lines += [
        "",
        "---",
        "## 如何使用这份清单",
        "1. **不要全买**：至多挑 3-5 个最契合自己持仓结构的标的",
        "2. **优先选 60 日趋势向上 + RSI 30-50 的**：避免接落下的刀子",
        "3. **结合基本面**：技术信号给方向，基本面给保障",
        "4. **严格止损**：每只票表里都给了止损价，触及立即平仓",
        "5. **单笔 ≤ 总资金 1.5%**",
    ]
    return "\n".join(lines)
