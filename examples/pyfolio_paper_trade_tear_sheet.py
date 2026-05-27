"""Generate a pyfolio tear-sheet from paper_trade_log.csv.

Reconstructs a daily portfolio-value series from BUY/SELL events in
``data_cache/paper_trade_log.csv`` against the kline parquet, then runs
``pyfolio.create_returns_tear_sheet``. Falls back to a structural stub if
pyfolio is not installed.

Install (only if missing): ``uv pip install pyfolio-reloaded``

Usage:
    .venv/bin/python examples/pyfolio_paper_trade_tear_sheet.py
    .venv/bin/python examples/pyfolio_paper_trade_tear_sheet.py \\
        --log data_cache/paper_trade_log.csv \\
        --kline data_cache/baidu_kline.parquet \\
        --out reports/pyfolio_paper_trade.png \\
        --initial-capital 100000
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f"[WARN] matplotlib unavailable: {exc}", file=sys.stderr)
    plt = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pyfolio as pf
    PYFOLIO_AVAILABLE = True
except Exception as exc:
    print(f"[WARN] pyfolio not importable ({exc}); will emit stub summary. "
          f"Install with: uv pip install pyfolio-reloaded",
          file=sys.stderr)
    PYFOLIO_AVAILABLE = False
    pf = None


# ---------------------------------------------------------------------------
# Symbol normalisation
# ---------------------------------------------------------------------------

def _strip_prefix(symbol: str) -> str:
    """SH600519 / 600519.SH / 600519 -> 600519 (kline code style)."""
    s = symbol.upper().strip()
    if s.startswith(("SH", "SZ", "BJ")):
        return s[2:]
    if s.endswith((".SH", ".SZ", ".BJ")):
        return s[:-3]
    return s


# ---------------------------------------------------------------------------
# Portfolio value reconstruction
# ---------------------------------------------------------------------------


def build_daily_returns_from_log(
    log_path: str | Path,
    kline_path: str | Path,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.20,
) -> pd.Series:
    """Reconstruct a daily portfolio-value series, then return daily simple returns.

    The paper trade log only records signal events (BUY/SELL with a quoted
    price), not actual fills/sizes. We assume:
      * each BUY allocates ``position_size_pct`` of current equity to the symbol;
      * each SELL liquidates the full current holding of that symbol;
      * positions are marked-to-market on kline close;
      * the trading-day calendar comes from kline rows in [log.min, log.max].

    Returns a pd.Series of simple daily returns indexed by Timestamp.
    """
    log_path = Path(log_path)
    kline_path = Path(kline_path)
    if not log_path.exists():
        raise FileNotFoundError(f"log not found: {log_path}")
    if not kline_path.exists():
        raise FileNotFoundError(f"kline not found: {kline_path}")

    log = pd.read_csv(log_path, parse_dates=["date"])
    log["code"] = log["symbol"].map(_strip_prefix)
    if log.empty:
        raise ValueError(f"empty log: {log_path}")

    start = log["date"].min()
    # Extend window to today's most-recent kline date so single-day logs still
    # produce a multi-day MTM curve.
    needed_codes = sorted(log["code"].unique())
    kline = pd.read_parquet(kline_path, columns=["code", "date", "close"])
    kline = kline[
        (kline["code"].isin(needed_codes))
        & (kline["date"] >= start)
    ]
    if kline.empty:
        raise ValueError(
            f"no kline rows for {needed_codes} after {start.date()}"
        )

    close_wide = (
        kline.pivot_table(index="date", columns="code", values="close",
                          aggfunc="last")
        .sort_index()
        .ffill()
    )
    if close_wide.empty:
        raise ValueError("close_wide is empty after pivot")

    cash = initial_capital
    shares = {c: 0 for c in close_wide.columns}
    equity_curve = []

    events_by_date = log.groupby("date")

    for trading_day in close_wide.index:
        if trading_day in events_by_date.groups:
            day_events = events_by_date.get_group(trading_day)
            for _, ev in day_events.iterrows():
                code = ev["code"]
                if code not in close_wide.columns:
                    continue
                px = close_wide.at[trading_day, code]
                if pd.isna(px) or px <= 0:
                    continue
                action = str(ev["action"]).upper()
                if action == "BUY":
                    notional = cash * position_size_pct
                    qty = int(notional / px // 100) * 100  # whole-lot
                    if qty > 0 and qty * px <= cash:
                        shares[code] = shares.get(code, 0) + qty
                        cash -= qty * px
                elif action == "SELL":
                    qty = shares.get(code, 0)
                    if qty > 0:
                        cash += qty * px
                        shares[code] = 0

        mtm = 0.0
        for code, qty in shares.items():
            if qty == 0:
                continue
            px = close_wide.at[trading_day, code]
            if pd.notna(px):
                mtm += qty * px
        equity_curve.append((trading_day, cash + mtm))

    equity = pd.Series(
        [e for _, e in equity_curve],
        index=pd.DatetimeIndex([d for d, _ in equity_curve], name="date"),
        name="equity",
    )
    returns = equity.pct_change().fillna(0.0)
    returns.name = "returns"
    return returns


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def load_benchmark_returns(
    kline_path: str | Path,
    benchmark_code: str = "000300",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series | None:
    """Load CSI300 returns from data_cache/index_kline.parquet if present.

    Returns None if no index data is available.
    """
    index_path = Path(kline_path).parent / "index_kline.parquet"
    if not index_path.exists():
        return None
    try:
        idx = pd.read_parquet(index_path)
    except Exception:
        return None
    if "code" not in idx.columns or "close" not in idx.columns:
        return None
    bench = idx[idx["code"].astype(str).str.contains(benchmark_code, na=False)]
    if bench.empty:
        return None
    bench = bench.sort_values("date").set_index("date")
    if start is not None:
        bench = bench[bench.index >= start]
    if end is not None:
        bench = bench[bench.index <= end]
    if bench.empty:
        return None
    rets = bench["close"].pct_change().fillna(0.0)
    rets.name = f"benchmark_{benchmark_code}"
    return rets


# ---------------------------------------------------------------------------
# Tear-sheet entry point
# ---------------------------------------------------------------------------


def run_tear_sheet(returns: pd.Series, benchmark: pd.Series | None,
                   out_path: Path) -> dict:
    """Run pyfolio (or stub) and save the figure. Returns a summary dict."""
    n = len(returns)
    if n < 2:
        return {"status": "no_data", "n_days": n}

    ann_factor = 252
    avg = returns.mean()
    std = returns.std()
    sharpe = (avg / std * np.sqrt(ann_factor)) if std > 0 else float("nan")
    cum = (1 + returns).prod() - 1
    cum_curve = (1 + returns).cumprod()
    drawdown = (cum_curve / cum_curve.cummax() - 1).min()

    summary = {
        "n_days": n,
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "cumulative_return": float(cum),
        "annualised_sharpe": float(sharpe),
        "max_drawdown": float(drawdown),
        "has_benchmark": benchmark is not None,
        "pyfolio_used": PYFOLIO_AVAILABLE,
    }

    if not PYFOLIO_AVAILABLE or plt is None:
        print("[STUB] pyfolio or matplotlib missing; skipping figure generation.")
        return {**summary, "status": "stub"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = pf.create_returns_tear_sheet(
                returns,
                benchmark_rets=benchmark,
                return_fig=True,
            )
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        summary["status"] = "ok"
        summary["figure"] = str(out_path)
    except Exception as exc:
        print(f"[WARN] pyfolio failed ({exc}); falling back to stub.",
              file=sys.stderr)
        summary["status"] = "pyfolio_error"
        summary["error"] = str(exc)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="data_cache/paper_trade_log.csv")
    parser.add_argument("--kline", default="data_cache/baidu_kline.parquet")
    parser.add_argument("--out", default="reports/pyfolio_paper_trade.png")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--position-pct", type=float, default=0.20)
    parser.add_argument("--benchmark", default="000300",
                        help="benchmark code to look up in index_kline.parquet")
    args = parser.parse_args(argv)

    returns = build_daily_returns_from_log(
        args.log, args.kline,
        initial_capital=args.initial_capital,
        position_size_pct=args.position_pct,
    )
    benchmark = load_benchmark_returns(
        args.kline, benchmark_code=args.benchmark,
        start=returns.index.min(), end=returns.index.max(),
    )
    summary = run_tear_sheet(returns, benchmark, Path(args.out))

    print("=" * 60)
    print("pyfolio paper-trade tear sheet")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:>22}: {v}")
    print("=" * 60)
    return 0 if summary.get("status") in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
