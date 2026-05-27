"""Static lookahead / leakage scanner.

借鉴 jin-ce-zhi-suan critic.py:74-128 的 AST + regex 静态扫描思想, 但仅检
lookahead bias (我们信任 import / open / eval 等, 不需要沙箱).

扫描模式 (按 severity):

  REJECT (CRITICAL — 真正的 lookahead bias):
    * df.shift(-N)            (N >= 1, peek 未来 N 期)
    * df.iloc[+N]             (N >= 1, 显式 forward indexing)
    * .loc[future_*:]         (loc 切到 future 命名变量)

  WARN (需要人工 review):
    * forward_returns / future_returns / next_close / tomorrow_close
    * groupby(...).transform("last") (可能跨 group leak)
    * rolling.agg("last")            (边界 leak)

  INFO (匹配但通常 ok):
    * "lookahead" / "leakage" 字面 (注释 / 安全说明)
    * 注释行内匹配自动降级

CLI:
    python -m tools.static_leak_check examples/strategy_v19_10.py
    python -m tools.static_leak_check --all
    python -m tools.static_leak_check --pattern examples/strategy_v19_*.py --report ...

Exit: 0 = pass, 2 = 任一 reject pattern 命中.
一行回滚: LEAK_CHECK_ENABLED = False.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LEAK_CHECK_ENABLED = True

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = ROOT / "data_cache" / "leak_scan_report.json"
DEFAULT_SCAN_PATTERNS = (
    "examples/strategy_v19_*.py",
    "examples/strategy_v20_*.py",
    "examples/factor_ic_*_is.py",
    "examples/paper_trade_today.py",
    "examples/paper_trade_v19_*.py",
    "src/claude_finance/strategies/*.py",
)

SEV_REJECT = "reject"
SEV_WARN = "warn"
SEV_INFO = "info"


@dataclass
class Finding:
    file: str
    line: int
    severity: str
    pattern_id: str
    matched: str
    description: str
    snippet: str = ""


@dataclass
class ScanReport:
    scanned_at: str = ""
    files_scanned: list = field(default_factory=list)
    findings: list = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.files_scanned)

    @property
    def n_findings(self) -> int:
        return len(self.findings)

    @property
    def n_rejects(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEV_REJECT)

    @property
    def n_warns(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEV_WARN)

    def to_dict(self) -> dict:
        return {
            "scanned_at": self.scanned_at,
            "n_files": self.n_files,
            "n_findings": self.n_findings,
            "n_rejects": self.n_rejects,
            "n_warns": self.n_warns,
            "files_scanned": self.files_scanned,
            "findings": [
                {
                    "file": f.file, "line": f.line, "severity": f.severity,
                    "pattern_id": f.pattern_id, "matched": f.matched,
                    "description": f.description, "snippet": f.snippet,
                }
                for f in self.findings
            ],
        }


REGEX_PATTERNS = [
    (
        re.compile(r"\.shift\s*\(\s*(-\s*\d+)"),
        SEV_REJECT, "shift_negative",
        "df.shift(-N) reads N periods AHEAD = lookahead bias",
    ),
    (
        re.compile(r"\.iloc\s*\[\s*\+\d+"),
        SEV_REJECT, "iloc_positive_offset",
        "df.iloc[+N] explicit forward indexing",
    ),
    (
        re.compile(r"\.loc\s*\[\s*future_\w+\s*:"),
        SEV_REJECT, "loc_future_slice",
        ".loc[future_*:] forward slice 直接 index 未来",
    ),
    (
        re.compile(
            r"\b(forward_returns?|future_returns?|next_close|tomorrow_close|future_price)\w*\b"
        ),
        SEV_WARN, "forward_keyword",
        "变量名含 forward/future/next/tomorrow — 必须 audit 看是否真 leak",
    ),
    (
        re.compile(r"\.transform\s*\(\s*[\"']last[\"']"),
        SEV_WARN, "transform_last",
        "groupby.transform('last') 可能跨 group leak T-1 → T",
    ),
    (
        re.compile(r"\.rolling\([^)]+\)\.\w+\([^)]*[\"']last[\"']"),
        SEV_WARN, "rolling_last",
        "rolling.agg('last') 在边界可能 leak",
    ),
    (
        re.compile(r"\b(lookahead|leakage|peek_future)\b", re.IGNORECASE),
        SEV_INFO, "lookahead_keyword_mention",
        "源码含 lookahead/leakage 字面 — 可能是注释或 safety 说明",
    ),
]


def scan_text(file_path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        is_comment = (
            stripped.startswith("#")
            or stripped.startswith('"""')
            or stripped.startswith("'''")
        )
        for pat, sev, pid, desc in REGEX_PATTERNS:
            for m in pat.finditer(line):
                actual_sev = SEV_INFO if (is_comment and sev != SEV_INFO) else sev
                snippet_lines = lines[max(0, lineno - 2): min(len(lines), lineno + 1)]
                snippet = "\n".join(snippet_lines).strip()
                findings.append(Finding(
                    file=str(file_path),
                    line=lineno,
                    severity=actual_sev,
                    pattern_id=pid,
                    matched=m.group(0),
                    description=desc,
                    snippet=snippet[:300],
                ))
    return findings


def scan_file(file_path: Path) -> list[Finding]:
    if not file_path.exists():
        return []
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_text(file_path, text)


def scan_paths(paths: list[Path]) -> ScanReport:
    if not LEAK_CHECK_ENABLED:
        return ScanReport(
            scanned_at=datetime.now().isoformat(),
            files_scanned=[],
            findings=[],
        )
    all_findings: list[Finding] = []
    files_scanned: list[str] = []
    for p in paths:
        if p.is_dir():
            for sub in p.rglob("*.py"):
                files_scanned.append(str(sub))
                all_findings.extend(scan_file(sub))
        elif p.is_file():
            files_scanned.append(str(p))
            all_findings.extend(scan_file(p))
    return ScanReport(
        scanned_at=datetime.now().isoformat(),
        files_scanned=files_scanned,
        findings=all_findings,
    )


def expand_glob(patterns, root: Path = ROOT) -> list[Path]:
    result: list[Path] = []
    for pat in patterns:
        for hit in root.glob(pat):
            if hit.is_file() and hit.suffix == ".py":
                result.append(hit)
    return sorted(set(result))


def write_report(report: ScanReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Static lookahead/leakage scanner")
    parser.add_argument("files", nargs="*", help="文件或目录路径")
    parser.add_argument("--all", action="store_true",
                       help="跑默认 strategy + factor_ic + src/strategies")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH),
                       help="JSON 报告输出路径")
    parser.add_argument("--no-report", action="store_true",
                       help="不写 JSON 报告 (仅 stdout)")
    parser.add_argument("--show-info", action="store_true",
                       help="也打印 info-level findings (默认仅 reject + warn)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.all:
        paths = expand_glob(list(DEFAULT_SCAN_PATTERNS))
    elif args.files:
        paths = []
        for f in args.files:
            p = Path(f)
            if not p.is_absolute():
                p = Path.cwd() / p
            paths.append(p)
    else:
        parser.print_help()
        return 1

    report = scan_paths(paths)
    if not args.no_report:
        write_report(report, Path(args.report))

    if not args.quiet:
        print(f"[leak_check] scanned {report.n_files} files, "
              f"{report.n_findings} findings "
              f"({report.n_rejects} reject, {report.n_warns} warn)", flush=True)
        for f in report.findings:
            if f.severity == SEV_INFO and not args.show_info:
                continue
            print(f"  [{f.severity.upper():6}] {f.file}:{f.line}  "
                  f"<{f.pattern_id}>  {f.matched!r}  — {f.description}",
                  flush=True)

    return 0 if report.n_rejects == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
