"""KPI Summary panel — 顶部 6 卡片纵览, 放在所有 section 之前.

汇总数据:
  1. 实际持仓 PnL 总和 (浮盈元 sum from portfolio.xlsx Positions sheet, 状态==持仓)
  2. 持仓数 N / 推荐数 M
  3. 今日 BUY 信号数 (paper_trade_log.csv 最新日期 BUY 行数)
  4. Production 版本 v19.6 + sidecar λ (从 paper_trade_today.py 解析)
  5. Forward OOS sample 月数 / 累积 cum%  (portfolio.xlsx Forward OOS Track Stats)
  6. Daily check 最后 exit + next launchd trigger + HTML render 时间戳

只读, 不修改任何 production. fallback to placeholder on any error.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from claude_finance.trades_log import aggregate_positions, load_trades  # noqa: E402

PORTFOLIO_XLSX = ROOT / "data_cache" / "portfolio.xlsx"
PAPER_TRADE_LOG = ROOT / "data_cache" / "paper_trade_log.csv"
PAPER_TRADE_PY = ROOT / "examples" / "paper_trade_today.py"
PORTFOLIO_STATE = ROOT / "data_cache" / "portfolio_state.json"
PLIST_PATH = ROOT / "examples" / "com.claude_finance.daily_check.plist"
KLINE_PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"

COLOR_OK = "#16a34a"
COLOR_WARN = "#f59e0b"
COLOR_FAIL = "#dc2626"
COLOR_MUTED = "#6b7280"
COLOR_ACCENT = "#2563eb"


def _kpi_card(label: str, value: str, sub: str = "", color: str = "",
              jump_to: str = "") -> str:
    """单卡片 HTML — 标签 + 大数值 + 副标题. jump_to: 可点击跳转的 panel keyword."""
    value_style = f"color:{color};" if color else ""
    sub_html = (
        f"<div style='font-size:11px; color:var(--muted, #6b7280); "
        f"margin-top:4px;'>{sub}</div>" if sub else ""
    )
    jump_attr = f' data-jump="{jump_to}" style="cursor:pointer;" title="点击跳转到对应面板"' if jump_to else ""
    return f"""
<div class='kpi-card'{jump_attr} style='background:rgba(0,0,0,0.02); border:1px solid var(--border, #e5e7eb);
            border-radius:8px; padding:12px 14px; min-width:160px; flex:1;'>
  <div style='font-size:11px; color:var(--muted, #6b7280); text-transform:uppercase;
              letter-spacing:0.5px; font-weight:500;'>{label}</div>
  <div style='font-size:18px; font-weight:700; margin-top:4px; {value_style}'>{value}</div>
  {sub_html}
</div>
"""


def _read_cost_and_value() -> tuple[float, float, int]:
    """从 trades.jsonl 聚合 + baidu_kline 最新 close, 返回 (总成本¥, 总市值¥, 持仓股数).

    总成本 = Σ WAC * net_shares (持仓股, net_shares > 0)
    总市值 = Σ close * net_shares (持仓股, close 来自 baidu_kline latest)
    """
    trades = load_trades()
    if not trades:
        return 0.0, 0.0, 0
    positions = aggregate_positions(trades)
    # 读 baidu_kline 最新 close per code
    close_map: dict[str, float] = {}
    if KLINE_PARQUET.exists():
        try:
            df = pd.read_parquet(KLINE_PARQUET, columns=["code", "date", "close"])
            df = df.sort_values(["code", "date"])
            latest = df.groupby("code", sort=False).tail(1)
            for _, r in latest.iterrows():
                c = str(r["code"]).zfill(6)
                prefix = "SH" if c[0] in ("6", "9") else "SZ"
                close_map[f"{prefix}{c}"] = float(r["close"])
        except Exception:
            pass
    total_cost = 0.0
    total_value = 0.0
    holding_n = 0
    for sym, p in positions.items():
        net = p.get("net_shares", 0)
        if net <= 0:
            continue
        wac = p.get("weighted_avg_cost") or 0
        close = close_map.get(sym)
        total_cost += wac * net
        if close is not None:
            total_value += close * net
        else:
            total_value += wac * net  # fallback: 当前价缺 → 用成本估市值
        holding_n += 1
    return round(total_cost, 2), round(total_value, 2), holding_n


def _read_positions_pnl() -> tuple[float, int, int]:
    """returns (pnl_sum_yuan, holding_count, recommended_count)."""
    df = pd.read_excel(PORTFOLIO_XLSX, sheet_name="Positions")
    pnl_col = "浮盈元" if "浮盈元" in df.columns else None
    status_col = "状态" if "状态" in df.columns else None
    holding_count = 0
    recommended_count = 0
    pnl_sum = 0.0
    if status_col is not None:
        holding_count = int((df[status_col] == "持仓").sum())
        recommended_count = int((df[status_col] == "推荐").sum())
    if pnl_col is not None and status_col is not None:
        sub = df.loc[df[status_col] == "持仓", pnl_col].dropna()
        pnl_sum = float(sub.sum()) if len(sub) else 0.0
    return pnl_sum, holding_count, recommended_count


def _read_today_buys() -> tuple[int, str]:
    """returns (latest_date_buy_count, latest_date_str)."""
    df = pd.read_csv(PAPER_TRADE_LOG)
    if df.empty or "date" not in df.columns:
        return 0, ""
    df["date"] = df["date"].astype(str)
    latest_date = df["date"].max()
    today_buy = int(((df["date"] == latest_date) & (df["action"] == "BUY")).sum())
    return today_buy, latest_date


def _read_production_version() -> tuple[str, str]:
    """从 paper_trade_today.py source 解析 production version + sidecar config."""
    text = PAPER_TRADE_PY.read_text(encoding="utf-8")
    v1910_m = re.search(r"^USE_V19_10_STACKED\s*=\s*(True|False)", text, re.MULTILINE)
    v196_m = re.search(r"^USE_V19_6_SIDECAR\s*=\s*(True|False)", text, re.MULTILINE)
    v194_m = re.search(r"^USE_V19_4_SIDECAR\s*=\s*(True|False)", text, re.MULTILINE)
    lam_amp_m = re.search(
        r"^SIDECAR_LAMBDA_AMP_20D\s*=\s*([\d.]+)", text, re.MULTILINE
    )
    lam_jzf_m = re.search(r"^SIDECAR_LAMBDA_JZF\s*=\s*([\d.]+)", text, re.MULTILINE)
    lam_m5_m = re.search(r"^SIDECAR_LAMBDA_M5\s*=\s*([\d.]+)", text, re.MULTILINE)
    lam_m20_m = re.search(r"^SIDECAR_LAMBDA_M20\s*=\s*([\d.]+)", text, re.MULTILINE)

    v1910_on = v1910_m is not None and v1910_m.group(1) == "True"
    v196_on = v196_m is not None and v196_m.group(1) == "True"
    v194_on = v194_m is not None and v194_m.group(1) == "True"

    if v1910_on:
        lam_a = lam_amp_m.group(1) if lam_amp_m else "?"
        lam_j = lam_jzf_m.group(1) if lam_jzf_m else "?"
        return "v19.10", f"amp_imb_20d λ={lam_a} + JZF λ={lam_j}"
    if v196_on:
        lam = lam_amp_m.group(1) if lam_amp_m else "?"
        return "v19.6", f"amp_imb_20d λ={lam}"
    elif v194_on:
        lam5 = lam_m5_m.group(1) if lam_m5_m else "?"
        lam20 = lam_m20_m.group(1) if lam_m20_m else "?"
        return "v19.4", f"m5+m20 λ=({lam5},{lam20})"
    else:
        return "v19.1", "baseline (no sidecar)"


def _read_forward_oos() -> tuple[int, float | None]:
    """returns (months, cum_pct). cum_pct is percent value (may be negative) or None."""
    df = pd.read_excel(PORTFOLIO_XLSX, sheet_name="Forward OOS Track Stats")
    months = 0
    cum_pct: float | None = None
    if "指标" in df.columns and "值" in df.columns:
        for _, row in df.iterrows():
            key = str(row["指标"]).strip()
            val = row["值"]
            if key == "累积月数":
                try:
                    months = int(val)
                except (ValueError, TypeError):
                    pass
            elif key == "forward cum return %":
                try:
                    cum_pct = float(val)
                except (ValueError, TypeError):
                    cum_pct = None
    return months, cum_pct


def _find_latest_daily_check_log() -> Path | None:
    """搜 /tmp/daily_check_YYYYMMDD.log 最新."""
    tmp = Path("/tmp")
    if not tmp.exists():
        return None
    candidates: list[Path] = []
    try:
        for p in tmp.glob("daily_check_*.log"):
            stem = p.stem.replace("daily_check_", "")
            if stem.isdigit() and len(stem) == 8:
                candidates.append(p)
    except OSError:
        return None
    candidates = [c for c in candidates if c.exists() and c.stat().st_size > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_daily_check_exit() -> int | None:
    """returns exit_code from latest /tmp/daily_check_YYYYMMDD.log, None if not found."""
    log_path = _find_latest_daily_check_log()
    if log_path is None:
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        m = re.search(r"Done\s*\(exit=(-?\d+)\)", line)
        if m:
            return int(m.group(1))
    return None


def _parse_plist_schedule() -> tuple[int, int]:
    """parse plist Hour/Minute, fallback (16, 30)."""
    if not PLIST_PATH.exists():
        return (16, 30)
    try:
        text = PLIST_PATH.read_text(encoding="utf-8")
    except OSError:
        return (16, 30)
    hour_m = re.search(r"<key>Hour</key>\s*<integer>\s*(\d+)\s*</integer>", text)
    min_m = re.search(r"<key>Minute</key>\s*<integer>\s*(\d+)\s*</integer>", text)
    hour = int(hour_m.group(1)) if hour_m else 16
    minute = int(min_m.group(1)) if min_m else 30
    return (hour, minute)


def _next_trigger(hour: int, minute: int) -> datetime:
    """next local trigger after now."""
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    return target


def build_kpi_summary_section() -> str:
    """Panel 0: 6-card KPI overview."""
    cards: list[str] = []

    # Card 1: 实际持仓 PnL
    try:
        pnl_sum, holding_n, rec_n = _read_positions_pnl()
        pnl_color = (
            COLOR_OK if pnl_sum > 0 else (COLOR_FAIL if pnl_sum < 0 else COLOR_MUTED)
        )
        pnl_sign = "+" if pnl_sum > 0 else ""
        cards.append(
            _kpi_card(
                "实际持仓 PnL",
                f"{pnl_sign}{pnl_sum:,.0f} 元",
                sub=f"{holding_n} 持仓 · {rec_n} 推荐",
                color=pnl_color,
                jump_to="positions-pnl",
            )
        )
    except Exception as exc:  # noqa: BLE001
        cards.append(_kpi_card("实际持仓 PnL", "N/A", sub=f"{type(exc).__name__}"))
        holding_n, rec_n = 0, 0  # fallback for card 2

    # Card 1.5: 总成本 / 总市值 (trades.jsonl 聚合)
    try:
        total_cost, total_value, hn = _read_cost_and_value()
        diff = total_value - total_cost
        diff_color = (
            COLOR_OK if diff > 0 else (COLOR_FAIL if diff < 0 else COLOR_MUTED)
        )
        cards.append(
            _kpi_card(
                "总成本 / 总市值",
                f"¥{total_cost:,.0f} / ¥{total_value:,.0f}",
                sub=f"差额 {diff:+,.0f} 元 · {hn} 只持仓",
                color=diff_color,
                jump_to="portfolio-curve",
            )
        )
    except Exception as exc:  # noqa: BLE001
        cards.append(_kpi_card("总成本 / 总市值", "N/A", sub=f"{type(exc).__name__}"))

    # Card 2: 持仓 / 推荐 计数
    try:
        _, holding_n, rec_n = _read_positions_pnl()
        cards.append(
            _kpi_card(
                "持仓 / 推荐",
                f"{holding_n} / {rec_n}",
                sub="state=持仓 vs state=推荐",
                color=COLOR_ACCENT,
                jump_to="recommended-picks",
            )
        )
    except Exception as exc:  # noqa: BLE001
        cards.append(_kpi_card("持仓 / 推荐", "N/A", sub=f"{type(exc).__name__}"))

    # Card 3: 最近 BUY 信号
    try:
        today_buys, latest_date = _read_today_buys()
        sub = f"latest log: {latest_date}" if latest_date else ""
        cards.append(
            _kpi_card(
                "最近 BUY 信号",
                f"{today_buys} 只",
                sub=sub,
                color=COLOR_ACCENT if today_buys > 0 else COLOR_MUTED,
            )
        )
    except Exception as exc:  # noqa: BLE001
        cards.append(_kpi_card("最近 BUY 信号", "N/A", sub=f"{type(exc).__name__}"))

    # Card 4: Production 版本 + sidecar λ
    try:
        version, lam_info = _read_production_version()
        cards.append(
            _kpi_card(
                "Production",
                version,
                sub=lam_info,
                color=COLOR_ACCENT,
                jump_to="sidecar-config",
            )
        )
    except Exception as exc:  # noqa: BLE001
        cards.append(_kpi_card("Production", "N/A", sub=f"{type(exc).__name__}"))

    # Card 5: Forward OOS 月数 / cum%
    try:
        months, cum_pct = _read_forward_oos()
        cum_str = f"{cum_pct:+.2f}%" if cum_pct is not None else "N/A"
        cum_color = COLOR_MUTED
        if cum_pct is not None:
            cum_color = COLOR_OK if cum_pct > 0 else COLOR_FAIL
        cards.append(
            _kpi_card(
                "Forward OOS",
                f"{months} 月 · {cum_str}",
                sub="12 月里程碑: Sharpe≥0.5",
                color=cum_color,
                jump_to="forward-oos",
            )
        )
    except Exception as exc:  # noqa: BLE001
        cards.append(_kpi_card("Forward OOS", "N/A", sub=f"{type(exc).__name__}"))

    # (Daily Check card 已删除 — 详情见 Today tab 末尾 Daily Check Status panel)

    grid_html = (
        "<div class='kpi-grid' style='display:flex; flex-wrap:wrap; gap:10px; "
        "align-items:stretch;'>" + "".join(cards) + "</div>"
    )
    return grid_html
