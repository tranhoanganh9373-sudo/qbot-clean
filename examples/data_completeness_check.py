"""每日 universe coverage 检查 — baidu_kline vs universe.csv + 必须大蓝筹.

Usage:
    python examples/data_completeness_check.py

Exit codes:
    0  = OK         (CSI300 cover >= 95% + 大蓝筹 10/10)
    1  = WARNING    (CSI300 < 95% OR universe < 90%) — 不阻塞
    99 = CRITICAL   (CSI300 < 80% OR 大蓝筹 < 8/10) — 阻塞 paper_trade

4 项 metric:
    1. overall      — baidu_kline 总 code 数
    2. universe_pct — baidu_kline ∩ universe / |universe|
    3. csi300_pct   — baidu_kline ∩ CSI300 / 300
    4. bluechip     — 10 只必持蓝筹存在数

Outputs:
    STDOUT: 4 section: overall / by-board / CSI300 / blue chips + verdict
    data_cache/data_completeness_log.csv: 一行追加
    macOS notification: warn / critical
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# --- 路径 ---
KLINE_PATH = Path("data_cache/baidu_kline.parquet")
UNIVERSE_PATH = Path("data_cache/universe.csv")
CSI300_PATH = Path("data_cache/csi300_constituents.csv")
LOG_PATH = Path("data_cache/data_completeness_log.csv")

# --- 阈值 ---
CSI300_OK_PCT = 0.95           # >= 95% → OK
CSI300_CRITICAL_PCT = 0.80     # < 80% → CRITICAL
UNIVERSE_WARN_PCT = 0.90       # < 90% → WARNING
BLUECHIP_CRITICAL_COUNT = 8    # < 8/10 → CRITICAL (即 8/10 = warn 边界)

# 必须存在的 10 只大蓝筹 (代码 → 名称)
MUST_HAVE_BLUE_CHIPS: dict[str, str] = {
    "600519": "贵州茅台",
    "601398": "工商银行",
    "601939": "建设银行",
    "600036": "招商银行",
    "601318": "中国平安",
    "600276": "恒瑞医药",
    "000858": "五粮液",
    "000333": "美的集团",
    "002594": "比亚迪",
    "300750": "宁德时代",
}

# --- Exit codes ---
EXIT_OK = 0
EXIT_WARNING = 1
EXIT_CRITICAL = 99


# --- 工具函数 ---

def _classify_board(code: str) -> str:
    """6 位代码 → 板块标签."""
    if not code or len(code) < 3:
        return "未知"
    prefix = code[:3]
    if prefix in {"000", "001", "002", "003"}:
        return "深主板"
    if prefix in {"300", "301"}:
        return "创业板"
    if prefix.startswith("60"):
        return "沪主板"
    if prefix == "688":
        return "科创板"
    if code[0] in {"8", "4"}:
        return "北交所"
    return "其他"


def _notify_macos(title: str, message: str) -> None:
    """macOS osascript notification. 失败静默(非 macOS / 无 osascript)."""
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
            check=False,
            timeout=5,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _append_log(log_path: Path, row: dict) -> None:
    """追加一行到 data_completeness_log.csv (header 自动)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if log_path.exists():
        df_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False)


# --- 核心计算 (无副作用,便于测试) ---

def compute_completeness(
    kline_codes: set[str],
    universe_codes: set[str],
    csi300_codes: set[str],
    must_have: dict[str, str] = MUST_HAVE_BLUE_CHIPS,
) -> dict:
    """计算 4 项 metric + verdict.

    Returns dict with: total, universe_*, csi300_*, bluechip_*, board_dist,
    verdict, exit_code, reasons.
    """
    # universe 覆盖
    universe_covered = len(universe_codes & kline_codes)
    universe_total = len(universe_codes)
    universe_pct = universe_covered / universe_total if universe_total > 0 else 0.0

    # CSI300 覆盖
    csi300_covered = len(csi300_codes & kline_codes)
    csi300_total = len(csi300_codes)
    csi300_pct = csi300_covered / csi300_total if csi300_total > 0 else 0.0

    # 大蓝筹
    bluechip_missing = [(c, n) for c, n in must_have.items() if c not in kline_codes]
    bluechip_covered = len(must_have) - len(bluechip_missing)
    bluechip_total = len(must_have)

    # 板块分布(以 universe 为分母,baidu_kline 命中为分子)
    board_dist: dict[str, dict] = {}
    for code in universe_codes:
        board = _classify_board(code)
        slot = board_dist.setdefault(board, {"covered": 0, "total": 0})
        slot["total"] += 1
        if code in kline_codes:
            slot["covered"] += 1
    for board in board_dist:
        total = board_dist[board]["total"]
        board_dist[board]["pct"] = (
            board_dist[board]["covered"] / total if total > 0 else 0.0
        )

    # 判定
    reasons: list[str] = []
    is_critical = False
    is_warning = False

    if csi300_pct < CSI300_CRITICAL_PCT:
        is_critical = True
        reasons.append(f"CSI300 cover {csi300_pct:.1%} < {CSI300_CRITICAL_PCT:.0%}")
    elif csi300_pct < CSI300_OK_PCT:
        is_warning = True
        reasons.append(f"CSI300 cover {csi300_pct:.1%} < {CSI300_OK_PCT:.0%}")

    if bluechip_covered < BLUECHIP_CRITICAL_COUNT:
        is_critical = True
        reasons.append(f"大蓝筹 {bluechip_covered}/{bluechip_total} < {BLUECHIP_CRITICAL_COUNT}/10")
    elif bluechip_covered < bluechip_total:
        is_warning = True
        reasons.append(f"大蓝筹 {bluechip_covered}/{bluechip_total}")

    if universe_pct < UNIVERSE_WARN_PCT:
        is_warning = True
        reasons.append(f"universe cover {universe_pct:.1%} < {UNIVERSE_WARN_PCT:.0%}")

    if is_critical:
        verdict, exit_code = "CRITICAL", EXIT_CRITICAL
    elif is_warning:
        verdict, exit_code = "WARNING", EXIT_WARNING
    else:
        verdict, exit_code = "OK", EXIT_OK

    return {
        "total": len(kline_codes),
        "universe_total": universe_total,
        "universe_covered": universe_covered,
        "universe_pct": universe_pct,
        "csi300_total": csi300_total,
        "csi300_covered": csi300_covered,
        "csi300_pct": csi300_pct,
        "bluechip_total": bluechip_total,
        "bluechip_covered": bluechip_covered,
        "bluechip_missing": bluechip_missing,
        "board_dist": board_dist,
        "verdict": verdict,
        "exit_code": exit_code,
        "reasons": reasons,
    }


def _print_report(result: dict) -> None:
    """4 section 人类可读报告."""
    verdict = result["verdict"]
    tag = {"OK": "✓", "WARNING": "⚠️ ", "CRITICAL": "🚨"}[verdict]

    print(f"--- Data completeness check [{tag} {verdict}] ---")

    # Section 1: overall
    print("\n[1] Overall")
    print(f"    baidu_kline: {result['total']} codes")

    # Section 2: by-board
    print("\n[2] By board (universe ∩ baidu_kline)")
    for board in sorted(result["board_dist"].keys()):
        d = result["board_dist"][board]
        print(f"    {board:<8s} {d['covered']:>5d}/{d['total']:<5d} = {d['pct']:.1%}")

    # Section 3: CSI300
    print("\n[3] CSI300 (核心覆盖)")
    print(
        f"    {result['csi300_covered']}/{result['csi300_total']} "
        f"= {result['csi300_pct']:.1%}  "
        f"(OK>={CSI300_OK_PCT:.0%} / CRITICAL<{CSI300_CRITICAL_PCT:.0%})"
    )

    # Section 4: 大蓝筹
    print(f"\n[4] 必持大蓝筹 ({result['bluechip_covered']}/{result['bluechip_total']})")
    if result["bluechip_missing"]:
        for code, name in result["bluechip_missing"]:
            print(f"    ✗ {code} {name} MISSING")
    else:
        print("    ✓ all 10 present")

    # Verdict summary
    print(f"\nVerdict: {tag} {verdict} (exit={result['exit_code']})")
    for r in result["reasons"]:
        print(f"  - {r}")


# --- 主流程 ---

def main() -> int:
    # 1. baidu_kline
    if not KLINE_PATH.exists():
        print(f"❌ baidu_kline 缺失: {KLINE_PATH}", file=sys.stderr)
        _notify_macos("🚨 数据残缺", f"baidu_kline 缺失: {KLINE_PATH}")
        return EXIT_CRITICAL

    kl_df = pd.read_parquet(KLINE_PATH, columns=["code"])
    kline_codes = set(kl_df["code"].astype(str).str.zfill(6).unique())

    # 2. universe
    if not UNIVERSE_PATH.exists():
        print(f"❌ universe.csv 缺失: {UNIVERSE_PATH}", file=sys.stderr)
        _notify_macos("🚨 数据残缺", f"universe.csv 缺失: {UNIVERSE_PATH}")
        return EXIT_CRITICAL

    uni_df = pd.read_csv(UNIVERSE_PATH, dtype={"code": str})
    universe_codes = set(uni_df["code"].astype(str).str.zfill(6).unique())
    if not universe_codes:
        print(f"❌ universe.csv 为空: {UNIVERSE_PATH}", file=sys.stderr)
        _notify_macos("🚨 数据残缺", "universe.csv 为空")
        return EXIT_CRITICAL

    # 3. CSI300 (可缺失 → 视为 0/0)
    if CSI300_PATH.exists():
        csi_df = pd.read_csv(CSI300_PATH, dtype={"code": str})
        csi300_codes = set(csi_df["code"].astype(str).str.zfill(6).unique())
    else:
        csi300_codes = set()
        print(f"⚠️  CSI300 文件缺失: {CSI300_PATH}", file=sys.stderr)

    # 4. 计算 + 打印
    result = compute_completeness(kline_codes, universe_codes, csi300_codes)
    _print_report(result)

    # 5. 落盘 log
    details = "; ".join(result["reasons"]) if result["reasons"] else ""
    log_row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total": result["total"],
        "universe_pct": round(result["universe_pct"], 4),
        "csi300_pct": round(result["csi300_pct"], 4),
        "bluechip_pct": round(result["bluechip_covered"] / result["bluechip_total"], 4),
        "verdict": result["verdict"],
        "details": details,
    }
    _append_log(LOG_PATH, log_row)

    # 6. macOS notification (仅 warn/critical)
    if result["verdict"] == "CRITICAL":
        _notify_macos(
            "🚨 数据残缺 CRITICAL",
            f"CSI300={result['csi300_pct']:.0%} 大蓝筹={result['bluechip_covered']}/10",
        )
    elif result["verdict"] == "WARNING":
        _notify_macos(
            "⚠️ 数据覆盖 WARNING",
            f"CSI300={result['csi300_pct']:.0%} universe={result['universe_pct']:.0%}",
        )

    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
