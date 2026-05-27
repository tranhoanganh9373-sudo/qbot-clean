"""Pre-run data sanity check — block downstream if K线数据 corruption 检出.

Usage:
    python examples/data_sanity_check.py [--strict] [--quiet]

Exit codes:
    0  = OK,数据 healthy(strict 模式全 pass / lenient 模式仅 critical pass)
    99 = FAIL,有 corruption,daily_check.sh 应中断后续 step

5 项 sanity check:
    1. check_no_neg_close       — 全表 close < 0 检测   (CRITICAL)
    2. check_no_extreme_jump    — 最近 N 天单日涨跌 > 11% (非 critical)
    3. check_data_freshness     — latest date 不超过 today - 3 天     (非 critical)
    4. check_coverage           — latest day 股票数 >= 80% universe   (非 critical)
    5. check_no_extreme_low     — 全表 close < 0.5 元                  (CRITICAL)

输出:
    STDOUT: 每个 check 状态 + 失败示例
    STDERR: 失败时红色 ANSI 醒目 banner
    LOG:    data_cache/sanity_check_log.csv (每天 append)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# --- 常量 (UPPER_SNAKE_CASE per coding-style.md) ---
DEFAULT_RECENT_DAYS = 30
EXTREME_JUMP_THRESHOLD = 0.11          # 单日涨跌幅 buffer(主板 10% 涨停 + 1% buffer)
EXTREME_LOW_PRICE = 0.5                # 元;低于此视为 corruption
MIN_COVERAGE_PCT = 0.80                # latest day 至少 80% universe
MAX_FRESHNESS_LAG_DAYS = 3             # latest date 距 today 最大滞后

KLINE_PATH = Path("data_cache/baidu_kline.parquet")
UNIVERSE_PATH = Path("data_cache/universe.csv")
LOG_PATH = Path("data_cache/sanity_check_log.csv")

EXIT_OK = 0
EXIT_FAIL = 99

ANSI_RED = "\033[31m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"


# --- 单个 check 函数 (返回 (passed, message)) ---

def check_no_neg_close(kl_df: pd.DataFrame, days: int | None = None) -> tuple[bool, str]:
    """全表(或最近 days)无 close < 0.

    CRITICAL — strict 与 lenient 都必须通过.
    days=None 表示扫全表(用于深度 corruption 检测).
    """
    if days is None:
        scope = kl_df
        scope_label = "all"
    else:
        latest = kl_df["date"].max()
        cutoff = latest - pd.Timedelta(days=days)
        scope = kl_df[kl_df["date"] >= cutoff]
        scope_label = f"last {days}d"

    neg = scope[scope["close"] < 0]
    if neg.empty:
        return True, f"check_no_neg_close [{scope_label}] OK (0 rows)"
    sample = neg.head(3)[["code", "date", "close"]].to_dict(orient="records")
    return False, f"check_no_neg_close [{scope_label}] FAIL: {len(neg)} rows; sample={sample}"


def check_no_extreme_jump(
    kl_df: pd.DataFrame,
    days: int = DEFAULT_RECENT_DAYS,
    threshold: float = EXTREME_JUMP_THRESHOLD,
) -> tuple[bool, str]:
    """最近 days 内无单日 abs(pct_change) > threshold.

    非 critical(创业板/科创板 20% 涨停会触发,但 default 30 天窗口 strict mode 才阻塞).
    """
    latest = kl_df["date"].max()
    cutoff = latest - pd.Timedelta(days=days)
    scope = kl_df[kl_df["date"] >= cutoff].sort_values(["code", "date"]).copy()
    scope["pct"] = scope.groupby("code")["close"].pct_change()
    jumps = scope[scope["pct"].abs() > threshold]
    if jumps.empty:
        return True, f"check_no_extreme_jump [last {days}d, >{threshold:.0%}] OK (0 rows)"
    # 取 abs(pct) 最大的 3 条作为示例
    jumps_sample = jumps.assign(abs_pct=jumps["pct"].abs()).nlargest(3, "abs_pct")
    sample = jumps_sample[["code", "date", "close", "pct"]].to_dict(orient="records")
    return False, f"check_no_extreme_jump [last {days}d, >{threshold:.0%}] FAIL: {len(jumps)} rows; sample={sample}"


def check_data_freshness(
    kl_df: pd.DataFrame,
    max_lag_days: int = MAX_FRESHNESS_LAG_DAYS,
    today: pd.Timestamp | None = None,
) -> tuple[bool, str]:
    """latest date 不超过 today - max_lag_days.

    非 critical(周末或节假日会有 1-3 天滞后).
    """
    today = today if today is not None else pd.Timestamp.today().normalize()
    latest = kl_df["date"].max()
    lag = (today - latest).days
    if lag <= max_lag_days:
        return True, f"check_data_freshness OK (latest={latest.date()}, lag={lag}d <= {max_lag_days}d)"
    return False, f"check_data_freshness FAIL (latest={latest.date()}, lag={lag}d > {max_lag_days}d)"


def check_coverage(
    kl_df: pd.DataFrame,
    universe_path: Path = UNIVERSE_PATH,
    min_pct: float = MIN_COVERAGE_PCT,
) -> tuple[bool, str]:
    """latest day 股票数应 >= min_pct * universe size.

    非 critical.
    """
    if not Path(universe_path).exists():
        return False, f"check_coverage FAIL (universe.csv missing at {universe_path})"
    universe = pd.read_csv(universe_path, dtype={"code": str})
    expected = len(universe)
    latest = kl_df["date"].max()
    actual = int((kl_df["date"] == latest).sum())
    pct = actual / expected if expected > 0 else 0.0
    if pct >= min_pct:
        return True, f"check_coverage OK ({actual}/{expected}={pct:.1%} >= {min_pct:.0%})"
    return False, f"check_coverage FAIL ({actual}/{expected}={pct:.1%} < {min_pct:.0%})"


def check_no_extreme_low(
    kl_df: pd.DataFrame,
    days: int | None = None,
    threshold: float = EXTREME_LOW_PRICE,
) -> tuple[bool, str]:
    """全表(或最近 days)无 close < threshold 元.

    CRITICAL — strict 与 lenient 都必须通过.
    days=None 表示扫全表.
    """
    if days is None:
        scope = kl_df
        scope_label = "all"
    else:
        latest = kl_df["date"].max()
        cutoff = latest - pd.Timedelta(days=days)
        scope = kl_df[kl_df["date"] >= cutoff]
        scope_label = f"last {days}d"

    low = scope[scope["close"] < threshold]
    if low.empty:
        return True, f"check_no_extreme_low [{scope_label}, <{threshold}元] OK (0 rows)"
    sample = low.head(3)[["code", "date", "close"]].to_dict(orient="records")
    return False, f"check_no_extreme_low [{scope_label}, <{threshold}元] FAIL: {len(low)} rows; sample={sample}"


# --- 主流程 ---

def _append_log(log_path: Path, row: dict) -> None:
    """落盘 sanity_check_log.csv (append).

    Schema: date,check_neg_close,check_extreme_jump,check_freshness,check_coverage,check_extreme_low,overall_pass,fail_details
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if log_path.exists():
        df_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False)


def _print_banner_fail(messages: list[str], *, stream=sys.stderr) -> None:
    """醒目红色 ANSI banner — 失败时 stderr 输出."""
    line = "=" * 70
    print(f"\n{ANSI_RED}{ANSI_BOLD}{line}", file=stream)
    print(f"🚨 DATA SANITY CHECK FAILED — production blocked", file=stream)
    print(f"{line}{ANSI_RESET}", file=stream)
    for msg in messages:
        print(f"{ANSI_RED}  ✗ {msg}{ANSI_RESET}", file=stream)
    print(f"{ANSI_RED}{ANSI_BOLD}{line}{ANSI_RESET}\n", file=stream)


def run_all_checks(
    kl_df: pd.DataFrame,
    universe_path: Path = UNIVERSE_PATH,
) -> dict:
    """运行 5 项 check 并返回 dict 结果(无副作用,便于测试)."""
    results: dict = {}

    # CRITICAL (扫全表 — corruption 不分时段都必须 0)
    results["check_neg_close"] = check_no_neg_close(kl_df, days=None)
    results["check_extreme_low"] = check_no_extreme_low(kl_df, days=None)

    # 非 critical (滚动窗口)
    results["check_extreme_jump"] = check_no_extreme_jump(kl_df)
    results["check_freshness"] = check_data_freshness(kl_df)
    results["check_coverage"] = check_coverage(kl_df, universe_path=universe_path)

    return results


def main(strict: bool = False, quiet: bool = False) -> int:
    """运行所有 check.

    strict=True  : 5 项全部 pass 才返回 0,否则 99
    strict=False : 仅 critical (neg_close + extreme_low) pass 即可返回 0,非 critical 仅 warn

    Returns: 0 (OK) or 99 (FAIL)
    """
    if not KLINE_PATH.exists():
        print(f"❌ K线数据缺失: {KLINE_PATH}", file=sys.stderr)
        return EXIT_FAIL

    kl_df = pd.read_parquet(KLINE_PATH)
    results = run_all_checks(kl_df, universe_path=UNIVERSE_PATH)

    critical_keys = ("check_neg_close", "check_extreme_low")
    non_critical_keys = ("check_extreme_jump", "check_freshness", "check_coverage")

    # 控制台报告
    if not quiet:
        print(f"--- Data sanity check ({'STRICT' if strict else 'LENIENT'}) ---")
        for key in (*critical_keys, *non_critical_keys):
            passed, msg = results[key]
            tag = "✓" if passed else "✗"
            label = "CRITICAL" if key in critical_keys else "warn   "
            print(f"  [{tag}] [{label}] {msg}")

    critical_fail = [k for k in critical_keys if not results[k][0]]
    non_critical_fail = [k for k in non_critical_keys if not results[k][0]]

    if strict:
        overall_pass = not (critical_fail or non_critical_fail)
    else:
        overall_pass = not critical_fail  # lenient 仅看 critical

    if not overall_pass:
        fail_msgs = [results[k][1] for k in (*critical_keys, *non_critical_keys) if not results[k][0]]
        _print_banner_fail(fail_msgs)
    elif non_critical_fail and not strict and not quiet:
        print(f"⚠️  lenient mode: {len(non_critical_fail)} non-critical fail(s) — warn only", file=sys.stderr)

    # 落盘 log
    fail_details = "; ".join(
        results[k][1] for k in (*critical_keys, *non_critical_keys) if not results[k][0]
    ) or ""
    log_row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "check_neg_close": results["check_neg_close"][0],
        "check_extreme_jump": results["check_extreme_jump"][0],
        "check_freshness": results["check_freshness"][0],
        "check_coverage": results["check_coverage"][0],
        "check_extreme_low": results["check_extreme_low"][0],
        "overall_pass": overall_pass,
        "fail_details": fail_details,
    }
    _append_log(LOG_PATH, log_row)

    return EXIT_OK if overall_pass else EXIT_FAIL


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strict", action="store_true", help="ALL 5 checks must pass (default: critical-only)")
    parser.add_argument("--quiet", action="store_true", help="Suppress STDOUT report (errors still on stderr)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(strict=args.strict, quiet=args.quiet))
