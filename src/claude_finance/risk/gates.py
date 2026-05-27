"""PortfolioRiskGate — 投资组合层风控一票否决.

借鉴 jin-ce-zhi-suan 门下省 (MenxiaSheng) 设计, 但改为 portfolio-level
(claude_finance 是 TopK 等权 rebalance, 没有 signal-level stop_loss).

三道闸:
  1. drawdown    历史 NAV peak → 当前 NAV 回撤 > MAX_PORTFOLIO_DD (默认 15%)
  2. daily_loss  昨收 → 今收 NAV 跌幅 > MAX_DAILY_LOSS (默认 4%)
  3. position_weight  单票 weight > MAX_POS_PER_STOCK (默认 20%)

任一闸 trip → exit code != 0 阻塞 paper_trade.
每次 check 写一行 data_cache/risk_event_log.csv 审计 trail (借鉴刑部).

一行回滚: 把 RISK_ENABLED = False, gate 全 pass.

CLI self-check (daily_check.sh step 1.8):
    python -m claude_finance.risk.gates --self-check
        exit 0 = 全部 pass
        exit 2 = 任一 gate trip (drawdown / daily_loss)
        exit 1 = 数据不足或读取失败

历史教训 (claude_finance MEMORY.md):
  4 次 Phase B OOS 衰减 -88%~-116% (v19.7 / v19.9 / super_big_net / shareholders)
  都跑到 -33%~-37% MDD. 15% 熔断会全部截胡.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

# === 一行回滚: 设 False 让所有 gate 直接 pass ===
RISK_ENABLED = True

# 默认阈值 (单源真理, 不引入 config.json 双源)
DEFAULT_MAX_PORTFOLIO_DD = 0.15      # 15% 组合回撤熔断
DEFAULT_MAX_DAILY_LOSS = 0.04        # 4% 单日亏损熔断
DEFAULT_MAX_POS_PER_STOCK = 0.20     # 20% 单票权重上限
DEFAULT_MIN_NAV_HISTORY = 30          # 不到 30 天 history bypass dd check

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT_LOG = ROOT / "data_cache" / "risk_event_log.csv"
PAPER_TRADE_LOG = ROOT / "data_cache" / "paper_trade_log.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
PORTFOLIO_STATE = ROOT / "data_cache" / "portfolio_state.json"

AUDIT_FIELDS = (
    "dt", "gate_id", "symbol", "blocked",
    "nav", "peak", "dd_pct", "reason",
)


@dataclass
class GateResult:
    ok: bool
    reason: str
    gate_id: str
    metric: float = 0.0
    detail: dict = field(default_factory=dict)


class PortfolioRiskGate:
    """投资组合层风控. 纯函数式 — 阈值通过 ctor 注入, 无全局可变状态."""

    def __init__(
        self,
        max_drawdown: float = DEFAULT_MAX_PORTFOLIO_DD,
        max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
        max_pos_pct: float = DEFAULT_MAX_POS_PER_STOCK,
        min_history: int = DEFAULT_MIN_NAV_HISTORY,
        audit_log: Path | None = DEFAULT_AUDIT_LOG,
        enabled: bool = True,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_pos_pct = max_pos_pct
        self.min_history = min_history
        self.audit_log = audit_log
        # global RISK_ENABLED 与实例 enabled 的 AND
        self.enabled = enabled and RISK_ENABLED

    # ---------- gates ----------
    def check_drawdown(self, nav_series: Sequence[float]) -> GateResult:
        """nav_series: 时间正序 NAV. 返回 GateResult.
        history < min_history 时 bypass (insufficient_history)."""
        if not self.enabled:
            return GateResult(True, "disabled", "drawdown")
        if len(nav_series) < self.min_history:
            return GateResult(
                True, f"insufficient_history (n={len(nav_series)} < {self.min_history})",
                "drawdown",
            )
        nav_list = list(nav_series)
        peak = max(nav_list)
        cur = nav_list[-1]
        if peak <= 0:
            return GateResult(True, "peak_zero_or_negative", "drawdown")
        dd_pct = (cur - peak) / peak  # 负数
        if dd_pct <= -self.max_drawdown:
            reason = (f"drawdown {dd_pct*100:.2f}% <= -{self.max_drawdown*100:.0f}% "
                      f"(peak={peak:.2f} cur={cur:.2f})")
            return GateResult(
                False, reason, "drawdown",
                metric=dd_pct,
                detail={"peak": peak, "cur": cur, "n": len(nav_list)},
            )
        return GateResult(
            True, f"dd_ok {dd_pct*100:.2f}%", "drawdown",
            metric=dd_pct,
            detail={"peak": peak, "cur": cur, "n": len(nav_list)},
        )

    def check_daily_loss(
        self, nav_today: float, nav_yesterday: float,
    ) -> GateResult:
        if not self.enabled:
            return GateResult(True, "disabled", "daily_loss")
        if nav_yesterday <= 0:
            return GateResult(True, "prev_nav_zero_or_negative", "daily_loss")
        chg = (nav_today / nav_yesterday) - 1.0
        if chg <= -self.max_daily_loss:
            return GateResult(
                False,
                f"daily_loss {chg*100:.2f}% <= -{self.max_daily_loss*100:.0f}% "
                f"(yesterday={nav_yesterday:.2f} today={nav_today:.2f})",
                "daily_loss",
                metric=chg,
                detail={"prev": nav_yesterday, "today": nav_today},
            )
        return GateResult(
            True, f"daily_ok {chg*100:.2f}%", "daily_loss",
            metric=chg,
            detail={"prev": nav_yesterday, "today": nav_today},
        )

    def check_position_weight(self, weights: dict) -> GateResult:
        if not self.enabled:
            return GateResult(True, "disabled", "position_weight")
        if not weights:
            return GateResult(True, "no_positions", "position_weight")
        max_sym, max_w = max(weights.items(), key=lambda kv: kv[1])
        if max_w > self.max_pos_pct:
            return GateResult(
                False,
                f"max_weight {max_sym}={max_w*100:.1f}% > {self.max_pos_pct*100:.0f}%",
                "position_weight",
                metric=max_w,
                detail={"sym": max_sym, "weight": max_w, "n_pos": len(weights)},
            )
        return GateResult(
            True, f"weights_ok max={max_sym}={max_w*100:.1f}%",
            "position_weight",
            metric=max_w,
            detail={"sym": max_sym, "weight": max_w, "n_pos": len(weights)},
        )

    # ---------- audit log ----------
    def audit(self, result: GateResult) -> None:
        """Append 1 row to risk_event_log.csv. Failure-tolerant."""
        if self.audit_log is None:
            return
        path = Path(self.audit_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        try:
            with path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                if new_file:
                    writer.writerow(AUDIT_FIELDS)
                detail = result.detail or {}
                writer.writerow([
                    datetime.now().isoformat(),
                    result.gate_id,
                    detail.get("sym", ""),
                    0 if result.ok else 1,
                    f"{detail.get('cur', detail.get('today', 0.0)):.4f}",
                    f"{detail.get('peak', detail.get('prev', 0.0)):.4f}",
                    f"{result.metric*100:.4f}" if isinstance(result.metric, float) else "",
                    result.reason[:240],
                ])
        except OSError:
            pass  # 永不抛, 风控不该被 logging 失败拖垮


# === NAV 历史 helper (标准 cash + mark-to-market 会计) ===
def compute_nav_series_from_log(
    capital: float = 100000.0,
    trade_log: Path = PAPER_TRADE_LOG,
    kline: Path = KLINE_PATH,
    shares_per_lot: int = 100,
) -> list[tuple[str, float]]:
    """从 paper_trade_log.csv + baidu_kline.parquet 算 daily NAV 序列.

    返回 [(date_str, nav), ...] 时间正序. 仅覆盖 [first_trade_date, last_close_date].
    标准会计:
        cash 初始 = capital
        BUY: cash -= price × shares
        SELL: cash += price × shares
        NAV(d) = cash(d) + Σ (qty[sym](d) × close[sym](d))
    Failure-tolerant: 任何异常返 [].
    """
    if not Path(trade_log).exists() or not Path(kline).exists():
        return []
    try:
        import pandas as pd
        trades = pd.read_csv(trade_log)
        if len(trades) == 0:
            return []
        trades["date"] = pd.to_datetime(trades["date"])
        k = pd.read_parquet(kline, columns=["code", "date", "close"])
        k["date"] = pd.to_datetime(k["date"])
    except Exception:
        return []

    def sym_to_code(sym: str) -> str:
        return sym[2:] if len(sym) >= 8 and sym[:2] in ("SH", "SZ") else sym

    syms = sorted(set(trades["symbol"]))
    codes = {sym_to_code(s) for s in syms}
    trade_dates = sorted(trades["date"].unique())
    if not trade_dates:
        return []
    first_trade_date = trade_dates[0]

    # 用 cash + qty 状态机走 trade timeline, 然后展开到每个交易日
    cash = float(capital)
    qty: dict = {s: 0 for s in syms}
    qty_history: dict = {}   # 每个 trade date 后的快照
    cash_history: dict = {}
    for d in trade_dates:
        day_trades = trades[trades["date"] == d]
        for _, t in day_trades.iterrows():
            s = t["symbol"]
            price = float(t.get("price", 0) or 0)
            if price <= 0:
                # log 里 price=0 是 paper_trade 占位 (drop-from-top-8 自动 SELL),
                # 完全跳过避免 qty/cash 不一致
                continue
            if t["action"] == "BUY":
                cash -= price * shares_per_lot
                qty[s] = qty.get(s, 0) + shares_per_lot
            elif t["action"] == "SELL":
                cash += price * shares_per_lot
                qty[s] = max(0, qty.get(s, 0) - shares_per_lot)
        qty_history[d] = dict(qty)
        cash_history[d] = cash

    # close map 只取 first_trade_date 之后
    import pandas as pd
    sub = k[(k["code"].isin(codes)) & (k["date"] >= first_trade_date)].sort_values(["code", "date"])
    if sub.empty:
        return []
    close_map = sub.pivot_table(index="date", columns="code", values="close").ffill()

    sorted_trade_dates = sorted(qty_history.keys())

    def state_at(d):
        """返 (qty_dict, cash) 在 d 当天结束时的状态."""
        applicable = [td for td in sorted_trade_dates if td <= d]
        if not applicable:
            return {}, capital
        return qty_history[applicable[-1]], cash_history[applicable[-1]]

    nav_series: list[tuple[str, float]] = []
    for d in close_map.index:
        if d < first_trade_date:
            continue
        q, c = state_at(d)
        row = close_map.loc[d]
        mark = 0.0
        for s, n in q.items():
            if n <= 0:
                continue
            code = sym_to_code(s)
            close = row.get(code, None)
            if close is None or pd.isna(close):
                continue
            mark += float(close) * n
        nav_series.append((d.strftime("%Y-%m-%d"), float(c + mark)))

    return nav_series


def self_check(verbose: bool = True) -> int:
    """CLI self-check. Returns exit code (0=ok, 1=insufficient_data, 2=blocked)."""
    gate = PortfolioRiskGate()
    if not gate.enabled:
        if verbose:
            print("[risk_gates] disabled (RISK_ENABLED=False) — skip", flush=True)
        return 0

    nav = compute_nav_series_from_log()
    if verbose:
        print(f"[risk_gates] NAV history: {len(nav)} days", flush=True)
    if len(nav) < 2:
        if verbose:
            print(f"[risk_gates] insufficient history ({len(nav)} days), bypass",
                  flush=True)
        return 1

    nav_values = [n for _, n in nav]
    dd_res = gate.check_drawdown(nav_values)
    gate.audit(dd_res)
    if verbose:
        marker = "✓" if dd_res.ok else "✗ BLOCK"
        print(f"[risk_gates] {marker} drawdown: {dd_res.reason}", flush=True)

    dl_res = gate.check_daily_loss(nav_values[-1], nav_values[-2])
    gate.audit(dl_res)
    if verbose:
        marker = "✓" if dl_res.ok else "✗ BLOCK"
        print(f"[risk_gates] {marker} daily_loss: {dl_res.reason}", flush=True)

    # WS broadcast 任何 trip
    if not dd_res.ok or not dl_res.ok:
        try:
            from claude_finance.ws_notify import ws_notify
            ws_notify("risk_event", {
                "drawdown_ok": dd_res.ok,
                "drawdown_reason": dd_res.reason,
                "daily_loss_ok": dl_res.ok,
                "daily_loss_reason": dl_res.reason,
                "nav_n": len(nav_values),
            })
        except Exception:
            pass
        return 2
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio risk gates")
    parser.add_argument("--self-check", action="store_true",
                       help="跑全部 gate 检查 (daily_check.sh step 1.8 用)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        code = self_check(verbose=not args.quiet)
        sys.exit(code)
    parser.print_help()
    sys.exit(0)
