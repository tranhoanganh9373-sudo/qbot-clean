"""Daily Forward OOS Monitor — 4 级 alert 系统.

监控真实 forward OOS (始于 2026-05-25) 的累计 / 相对 / Sharpe / 连续负月度,
触发 green/yellow/orange/red/black 5 级状态. 红/黑灯需要立即行动.

读:
  data_cache/paper_trade_log.csv     (BUY/SELL 记录, 算 forward 累积价值)
  data_cache/portfolio_state.json    (当前持仓)
  data_cache/baidu_kline.parquet     (close prices)
  data_cache/index_kline.parquet     (CSI300 = sh000300, relative return)

写:
  data_cache/forward_oos_alerts.csv  (append, 每天一行, 完整时间序列)
  macOS notification                 (yellow+ 弹通知)
  STDOUT (ANSI 染色) 报告

退出码 (供 shell 检测):
  0 = green
  1 = yellow
  2 = orange
  3 = red
  4 = black

Run:  python examples/forward_oos_monitor.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data_cache" / "paper_trade_log.csv"
STATE = ROOT / "data_cache" / "portfolio_state.json"
KLINE = ROOT / "data_cache" / "baidu_kline.parquet"
INDEX_KLINE = ROOT / "data_cache" / "index_kline.parquet"
ALERTS_CSV = ROOT / "data_cache" / "forward_oos_alerts.csv"
PORTFOLIO_EXCEL_PY = ROOT / "examples" / "portfolio_excel.py"

TOTAL_CAPITAL = 50000.0
FORWARD_OOS_START = "2026-05-25"
CSI300_CODE = "sh000300"

# ---------- 阈值 (单一来源) ----------
# 黄灯
YELLOW_CUM_60D = -0.10
YELLOW_REL_60D = -0.05  # vs CSI300 underperform 5pp

# 橙灯
ORANGE_CUM_60D = -0.20
ORANGE_CONSEC_NEG_MONTHS = 3

# 红灯
RED_CUM_60D = -0.30
RED_SHARPE_3M_THRESHOLD = 0.3  # 用户指定: 3 月 rolling Sharpe < 0.3 + 连续 3 月负
RED_SHARPE_3M_CONSEC_NEG = 3

# 黑灯 (catastrophic, overrides 红)
BLACK_SHARPE_6M_THRESHOLD = 0.0

TRADING_DAYS_60 = 60  # ~3 月

ALERTS_HEADERS = [
    "date", "level", "triggers",
    "portfolio_value", "cum_60d", "rel_60d",
    "sharpe_3m", "sharpe_6m",
    "consec_neg_months",
    "actions_taken_notes",
]

LEVEL_EXIT_CODE = {
    "green": 0,
    "yellow": 1,
    "orange": 2,
    "red": 3,
    "black": 4,
}

LEVEL_NOTIFICATION_TITLE = {
    "yellow": "🟡 Forward OOS 黄灯",
    "orange": "🟠 Forward OOS 橙灯",
    "red": "🔴 Forward OOS 红灯",
    "black": "⚫ Forward OOS 黑灯",
}

ACTIONS_BY_LEVEL = {
    "green": [],
    "yellow": [
        "持续观察, 下周再看",
        "确认 paper_trade 仍在运行 + signal 正常出",
    ],
    "orange": [
        "减仓 50% (实盘)",
        "暂停加仓, 等下个月初再评估",
        "对照 backtest 24m OOS Sharpe 0.87 看是否仍在 ±1σ 内",
    ],
    "red": [
        "切回 ETF (沪深300 510300 / 中证500 510500)",
        "停 train24 至少 1 个月观察",
        "12 月后看 forward OOS 是否回升 ≥ 0.5",
    ],
    "black": [
        "立即全部清仓, 切 ETF",
        "停 train24 production, 回 v17 LGB legacy 或全 cash",
        "6 月 Sharpe < 0 = alpha 失效证据, 重训 / 换模型",
    ],
}

# ANSI 颜色
ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "orange": "\033[38;5;208m",
    "red": "\033[31m",
    "black": "\033[90m",  # 灰色 (terminal 没真黑底)
}

LEVEL_EMOJI = {
    "green": "🟢",
    "yellow": "🟡",
    "orange": "🟠",
    "red": "🔴",
    "black": "⚫",
}


# ---------- 加载 portfolio_excel.compute_forward_oos_track ----------
def _load_track_fn() -> Callable[..., pd.DataFrame]:
    """从 examples/portfolio_excel.py 借 compute_forward_oos_track.
    不修改 portfolio_excel.py 的接口, 只复用 pure function.
    """
    spec = importlib.util.spec_from_file_location(
        "portfolio_excel_mod", PORTFOLIO_EXCEL_PY
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_forward_oos_track


# ---------- 数据加载 ----------
@dataclass
class MonitorInputs:
    """Pure container."""

    today: str
    track_df: pd.DataFrame  # 月度 track (从 paper_trade_log 算)
    daily_curve: pd.DataFrame  # [date, portfolio_value] 日级累积曲线
    csi300: pd.DataFrame  # [date, close]
    holdings: list[str] = field(default_factory=list)


def load_inputs(
    today: str | None = None,
    log_path: Path = LOG,
    state_path: Path = STATE,
    kline_path: Path = KLINE,
    index_path: Path = INDEX_KLINE,
    total_capital: float = TOTAL_CAPITAL,
    start_date: str = FORWARD_OOS_START,
) -> MonitorInputs:
    """IO wrapper. 缺文件时返回 sane defaults."""
    if log_path.exists():
        log_df = pd.read_csv(log_path, dtype={"symbol": str})
    else:
        log_df = pd.DataFrame(
            columns=["date", "action", "symbol", "name", "score", "price"]
        )

    if state_path.exists():
        state = json.loads(state_path.read_text())
        holdings = list(state.get("holdings", []))
        if today is None:
            today = state.get("date") or _date.today().isoformat()
    else:
        holdings = []
        if today is None:
            today = _date.today().isoformat()

    if kline_path.exists():
        kline_df = pd.read_parquet(kline_path, columns=["code", "date", "close"])
        kline_df["code"] = kline_df["code"].astype(str).str.zfill(6)
    else:
        kline_df = pd.DataFrame(columns=["code", "date", "close"])

    if index_path.exists():
        idx = pd.read_parquet(index_path, columns=["code", "date", "close"])
        csi300 = (
            idx[idx["code"] == CSI300_CODE][["date", "close"]]
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        csi300 = pd.DataFrame(columns=["date", "close"])

    track_fn = _load_track_fn()
    track_df = track_fn(
        log_df, kline_df, holdings,
        total_capital=total_capital, start_date=start_date,
    )

    daily_curve = build_daily_curve(track_df, total_capital, start_date)

    return MonitorInputs(
        today=today,
        track_df=track_df,
        daily_curve=daily_curve,
        csi300=csi300,
        holdings=holdings,
    )


def build_daily_curve(
    track_df: pd.DataFrame,
    total_capital: float,
    start_date: str,
) -> pd.DataFrame:
    """从月度 track 推日级累积曲线.

    简化口径: 月内线性插值 start_value → end_value (够用于 60D cum return).
    无 track → 单点 (start_date, total_capital).
    """
    if track_df is None or track_df.empty:
        return pd.DataFrame(
            [{"date": pd.Timestamp(start_date), "portfolio_value": total_capital}]
        )

    rows: list[dict] = []
    for r in track_df.itertuples(index=False):
        ms = pd.Timestamp(r.month_start_date)
        me = pd.Timestamp(r.month_end_date)
        sv = float(r.start_value)
        ev_raw = r.end_value
        if ev_raw is None or (isinstance(ev_raw, float) and math.isnan(ev_raw)):
            ev = sv
        else:
            ev = float(ev_raw)
        # 月内每个工作日线性插值
        if ms == me:
            rows.append({"date": ms, "portfolio_value": ev})
            continue
        days = pd.date_range(ms, me, freq="B")  # business days
        if len(days) == 0:
            rows.append({"date": ms, "portfolio_value": sv})
            rows.append({"date": me, "portfolio_value": ev})
            continue
        denom = max(len(days) - 1, 1)
        for i, d in enumerate(days):
            frac = i / denom
            v = sv + (ev - sv) * frac
            rows.append({"date": d, "portfolio_value": v})

    df = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------- 指标计算 ----------
def lookup_value_n_days_ago(
    daily_curve: pd.DataFrame,
    today_ts: pd.Timestamp,
    n_trading_days: int,
) -> float | None:
    """daily_curve sorted by date asc. 回看 n_trading_days 前的 value.
    若 curve 长度 < n_trading_days+1 → 返回 None.
    """
    if daily_curve is None or daily_curve.empty:
        return None
    df = daily_curve[daily_curve["date"] <= today_ts]
    if len(df) < n_trading_days + 1:
        return None
    return float(df.iloc[-(n_trading_days + 1)]["portfolio_value"])


def lookup_csi300_n_days_ago(
    csi300: pd.DataFrame,
    today_ts: pd.Timestamp,
    n_trading_days: int,
) -> tuple[float | None, float | None]:
    """Return (today_close, n_days_ago_close)."""
    if csi300 is None or csi300.empty:
        return None, None
    df = csi300[csi300["date"] <= today_ts]
    if len(df) < n_trading_days + 1:
        return None, None
    today_close = float(df.iloc[-1]["close"])
    past_close = float(df.iloc[-(n_trading_days + 1)]["close"])
    return today_close, past_close


def count_trailing_neg_months(monthly_returns: Sequence[float]) -> int:
    """从末尾往前数连续负月度数. NaN 截断 (不算负)."""
    cnt = 0
    for r in reversed(monthly_returns):
        if isinstance(r, float) and math.isnan(r):
            break
        if r < 0:
            cnt += 1
        else:
            break
    return cnt


def compute_rolling_sharpe(
    monthly_returns: Sequence[float], n: int,
) -> float | None:
    """末尾 n 月的 Sharpe = mean/std * sqrt(12) (rf=0). 样本 < n → None."""
    arr = np.array([
        r for r in monthly_returns
        if not (isinstance(r, float) and math.isnan(r))
    ])
    if len(arr) < n:
        return None
    tail = arr[-n:]
    mu = float(tail.mean())
    sigma = float(tail.std(ddof=1))
    if sigma == 0:
        return None
    return (mu / sigma) * math.sqrt(12)


# ---------- 评估 alert level ----------
@dataclass
class AlertResult:
    level: str
    triggers: list[str]
    portfolio_value: float
    cum_60d: float | None
    rel_60d: float | None
    sharpe_3m: float | None
    sharpe_6m: float | None
    consec_neg_months: int
    csi300_60d: float | None


def evaluate_alerts(inputs: MonitorInputs) -> AlertResult:
    """核心决策逻辑. 严格按 black > red > orange > yellow > green 优先级."""
    today_ts = pd.Timestamp(inputs.today)

    # 当前 portfolio_value
    if not inputs.daily_curve.empty:
        cur_df = inputs.daily_curve[inputs.daily_curve["date"] <= today_ts]
        portfolio_value = (
            float(cur_df.iloc[-1]["portfolio_value"]) if not cur_df.empty
            else TOTAL_CAPITAL
        )
    else:
        portfolio_value = TOTAL_CAPITAL

    # 60D portfolio cum
    val_60d_ago = lookup_value_n_days_ago(inputs.daily_curve, today_ts, TRADING_DAYS_60)
    cum_60d: float | None = None
    if val_60d_ago is not None and val_60d_ago > 0:
        cum_60d = portfolio_value / val_60d_ago - 1.0

    # 60D CSI300 cum
    csi_today, csi_60d_ago = lookup_csi300_n_days_ago(
        inputs.csi300, today_ts, TRADING_DAYS_60
    )
    csi300_60d: float | None = None
    rel_60d: float | None = None
    if csi_today is not None and csi_60d_ago is not None and csi_60d_ago > 0:
        csi300_60d = csi_today / csi_60d_ago - 1.0
        if cum_60d is not None:
            rel_60d = cum_60d - csi300_60d

    # monthly returns (跳过 NaN)
    monthly_returns: list[float] = []
    if not inputs.track_df.empty:
        for r in inputs.track_df["monthly_return"].tolist():
            if r is None:
                continue
            if isinstance(r, float) and math.isnan(r):
                continue
            monthly_returns.append(float(r))
    consec_neg = count_trailing_neg_months(monthly_returns)
    sharpe_3m = compute_rolling_sharpe(monthly_returns, 3)
    sharpe_6m = compute_rolling_sharpe(monthly_returns, 6)

    # ---------- BLACK (catastrophic, 最高优先级) ----------
    if sharpe_6m is not None and sharpe_6m < BLACK_SHARPE_6M_THRESHOLD:
        return AlertResult(
            level="black",
            triggers=[f"6月 forward Sharpe = {sharpe_6m:.2f} < 0 (catastrophic)"],
            portfolio_value=portfolio_value,
            cum_60d=cum_60d,
            rel_60d=rel_60d,
            sharpe_3m=sharpe_3m,
            sharpe_6m=sharpe_6m,
            consec_neg_months=consec_neg,
            csi300_60d=csi300_60d,
        )

    # ---------- RED ----------
    red_triggers: list[str] = []
    if cum_60d is not None and cum_60d < RED_CUM_60D:
        red_triggers.append(f"60D cum_return = {cum_60d:.1%} < -30%")
    if (
        sharpe_3m is not None
        and sharpe_3m < RED_SHARPE_3M_THRESHOLD
        and consec_neg >= RED_SHARPE_3M_CONSEC_NEG
    ):
        red_triggers.append(
            f"3月 rolling Sharpe = {sharpe_3m:.2f} < 0.3 (用户阈值) "
            f"+ 连续 {consec_neg} 月负"
        )
    if red_triggers:
        return AlertResult(
            level="red",
            triggers=red_triggers,
            portfolio_value=portfolio_value,
            cum_60d=cum_60d,
            rel_60d=rel_60d,
            sharpe_3m=sharpe_3m,
            sharpe_6m=sharpe_6m,
            consec_neg_months=consec_neg,
            csi300_60d=csi300_60d,
        )

    # ---------- ORANGE ----------
    orange_triggers: list[str] = []
    if cum_60d is not None and cum_60d < ORANGE_CUM_60D:
        orange_triggers.append(f"60D cum_return = {cum_60d:.1%} < -20%")
    if consec_neg >= ORANGE_CONSEC_NEG_MONTHS:
        orange_triggers.append(f"连续 {consec_neg} 月度负收益")
    if orange_triggers:
        return AlertResult(
            level="orange",
            triggers=orange_triggers,
            portfolio_value=portfolio_value,
            cum_60d=cum_60d,
            rel_60d=rel_60d,
            sharpe_3m=sharpe_3m,
            sharpe_6m=sharpe_6m,
            consec_neg_months=consec_neg,
            csi300_60d=csi300_60d,
        )

    # ---------- YELLOW ----------
    yellow_triggers: list[str] = []
    if cum_60d is not None and cum_60d < YELLOW_CUM_60D:
        yellow_triggers.append(f"60D cum_return = {cum_60d:.1%} < -10%")
    if rel_60d is not None and rel_60d < YELLOW_REL_60D:
        yellow_triggers.append(
            f"60D vs CSI300 underperform = {rel_60d:.1%} < -5pp"
        )
    if yellow_triggers:
        return AlertResult(
            level="yellow",
            triggers=yellow_triggers,
            portfolio_value=portfolio_value,
            cum_60d=cum_60d,
            rel_60d=rel_60d,
            sharpe_3m=sharpe_3m,
            sharpe_6m=sharpe_6m,
            consec_neg_months=consec_neg,
            csi300_60d=csi300_60d,
        )

    # ---------- GREEN ----------
    return AlertResult(
        level="green",
        triggers=[],
        portfolio_value=portfolio_value,
        cum_60d=cum_60d,
        rel_60d=rel_60d,
        sharpe_3m=sharpe_3m,
        sharpe_6m=sharpe_6m,
        consec_neg_months=consec_neg,
        csi300_60d=csi300_60d,
    )


# ---------- 输出 ----------
def render_report(
    today: str,
    alert: AlertResult,
    n_months: int,
    n_holdings: int = 0,
    use_ansi: bool = True,
) -> str:
    """ASCII / ANSI 报告."""
    def c(name: str) -> str:
        return ANSI.get(name, "") if use_ansi else ""

    bar = "═" * 47
    level = alert.level
    emoji = LEVEL_EMOJI[level]
    col = c(level if level != "green" else "green")
    reset = c("reset")
    bold = c("bold")

    lines: list[str] = [bar]

    if level == "green":
        lines.append(f"Forward OOS Monitor — {today} (sample {n_months} 月)")
        lines.append(bar)
        lines.append(f"{col}{bold}{emoji} 状态: GREEN{reset}")
        lines.append("")
        if n_months < 3:
            lines.append("样本数据不足 (< 3 月), 仅记录, 不触发")
        else:
            lines.append("所有指标在阈值内, 正常运行")
        lines.append(f"当前组合值: ¥{alert.portfolio_value:,.0f}")
        lines.append(f"持仓: {n_holdings}")
        if alert.cum_60d is not None:
            lines.append(f"60D 累计: {alert.cum_60d:+.1%}")
        if alert.rel_60d is not None:
            lines.append(f"60D vs CSI300: {alert.rel_60d:+.1%}")
        lines.append(_milestone_line(n_months))
    else:
        title = (
            f"{emoji}{emoji}{emoji} {level.upper()} ALERT — {today} "
            f"(sample {n_months} 月) {emoji}{emoji}{emoji}"
        )
        lines.append(f"{col}{bold}{title}{reset}")
        lines.append(bar)
        lines.append(f"{col}触发器:{reset}")
        for t in alert.triggers:
            lines.append(f"  - {t}")
        lines.append("")
        lines.append(f"{bold}建议行动:{reset}")
        for i, a in enumerate(ACTIONS_BY_LEVEL[level], 1):
            lines.append(f"  {i}. {a}")
        lines.append("")
        lines.append(f"{bold}详细数据:{reset}")
        lines.append(
            f"  当前组合值: ¥{alert.portfolio_value:,.0f} (初始 ¥{TOTAL_CAPITAL:,.0f})"
        )
        if alert.cum_60d is not None:
            lines.append(f"  60D 累计回撤: {alert.cum_60d:+.1%}")
        if alert.csi300_60d is not None:
            lines.append(f"  CSI300 同期: {alert.csi300_60d:+.1%}")
        if alert.rel_60d is not None:
            lines.append(f"  相对回撤: {alert.rel_60d*100:+.1f}pp")
        if alert.sharpe_3m is not None:
            lines.append(f"  3月 rolling Sharpe: {alert.sharpe_3m:+.2f}")
        if alert.sharpe_6m is not None:
            lines.append(f"  6月 rolling Sharpe: {alert.sharpe_6m:+.2f}")
        if alert.consec_neg_months > 0:
            lines.append(f"  连续负月度: {alert.consec_neg_months} 月")

    lines.append(bar)
    return "\n".join(lines)


def _milestone_line(n_months: int) -> str:
    """下一个里程碑提示."""
    start = pd.Timestamp(FORWARD_OOS_START)
    if n_months < 3:
        next_n = 3
    elif n_months < 6:
        next_n = 6
    elif n_months < 12:
        next_n = 12
    else:
        next_n = n_months + 3
    target = start + pd.DateOffset(months=next_n)
    return f"下个里程碑: {next_n} 月节点 ({target.date().isoformat()})"


# ---------- 落盘 ----------
def append_alert_log(
    today: str,
    alert: AlertResult,
    alerts_path: Path = ALERTS_CSV,
    actions_notes: str = "",
) -> None:
    """每日一行. 同日重复 run → 用最新值覆盖该日."""
    row = {
        "date": today,
        "level": alert.level,
        "triggers": " | ".join(alert.triggers) if alert.triggers else "",
        "portfolio_value": round(alert.portfolio_value, 2),
        "cum_60d": (
            round(alert.cum_60d, 6) if alert.cum_60d is not None else ""
        ),
        "rel_60d": (
            round(alert.rel_60d, 6) if alert.rel_60d is not None else ""
        ),
        "sharpe_3m": (
            round(alert.sharpe_3m, 4) if alert.sharpe_3m is not None else ""
        ),
        "sharpe_6m": (
            round(alert.sharpe_6m, 4) if alert.sharpe_6m is not None else ""
        ),
        "consec_neg_months": alert.consec_neg_months,
        "actions_taken_notes": actions_notes,
    }

    if alerts_path.exists():
        existing = pd.read_csv(alerts_path, dtype={"date": str})
        existing = existing[existing["date"] != today]
        out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])

    out = out.reindex(columns=ALERTS_HEADERS)
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(alerts_path, index=False)


# ---------- macOS 通知 ----------
def notify(
    level: str,
    msg: str,
    runner: Callable[..., object] = subprocess.run,
) -> None:
    """macOS osascript display notification.

    green → 不打扰 (no-op).
    yellow/orange/red/black → 弹通知.
    runner kwarg 用于测试 monkeypatch.
    """
    if level == "green":
        return
    title = LEVEL_NOTIFICATION_TITLE.get(level)
    if not title:
        return
    safe_msg = msg.replace('"', "'")[:200]
    safe_title = title.replace('"', "'")
    script = f'display notification "{safe_msg}" with title "{safe_title}"'
    try:
        runner(["osascript", "-e", script], check=False, capture_output=True)
    except FileNotFoundError:
        # 非 macOS 环境 (CI/Linux) → 静默
        pass


# ---------- 主入口 ----------
def run(
    today: str | None = None,
    write_log: bool = True,
    notify_user: bool = True,
    use_ansi: bool | None = None,
) -> int:
    """End-to-end. 返回 exit code."""
    if use_ansi is None:
        use_ansi = sys.stdout.isatty()

    inputs = load_inputs(today=today)
    alert = evaluate_alerts(inputs)
    n_months = len(inputs.track_df) if inputs.track_df is not None else 0
    n_holdings = len(inputs.holdings)

    report = render_report(
        inputs.today, alert, n_months,
        n_holdings=n_holdings, use_ansi=use_ansi,
    )
    print(report)

    if write_log:
        append_alert_log(inputs.today, alert)

    if notify_user and alert.level != "green":
        short_msg = "; ".join(alert.triggers)[:180]
        notify(alert.level, short_msg)

    # WS broadcast (fail-tolerant): 让 dashboard 收到 oos_alert 事件后高亮告警中心
    try:
        from claude_finance.ws_notify import ws_notify
        ws_notify("oos_alert", {
            "level": alert.level,
            "triggers": list(alert.triggers)[:10],
            "exit_code": LEVEL_EXIT_CODE[alert.level],
            "today": str(inputs.today.date()) if hasattr(inputs.today, "date") else str(inputs.today),
        })
    except Exception:
        pass

    return LEVEL_EXIT_CODE[alert.level]


def main() -> None:
    code = run()
    sys.exit(code)


if __name__ == "__main__":
    main()
