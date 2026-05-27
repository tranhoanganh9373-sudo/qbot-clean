"""清理 data_cache/ 下的旧 .bak* 文件 (idempotent).

策略 (group-aware retention):
- baidu_kline.*.bak / baidu_kline.parquet.*.bak     -> 保留最新 2 个
- portfolio.xlsx.bak_*                                -> 保留最新 2 个
  其中 bak_pre_positions_rebuild_* 强制保留(真实买入重建依赖)
- 其他单一文件 .bak / .bak_* (margin / state / predictions) -> 保留最新 1 个

特殊保护(永不删除):
- portfolio.xlsx.bak_pre_positions_rebuild_* (positions 重建)
- portfolio.xlsx.bak_pre_trade (首次 trade snapshot)

用法:
  python examples/cleanup_old_baks.py                # dry-run (默认)
  python examples/cleanup_old_baks.py --apply        # 实际删除
  python examples/cleanup_old_baks.py --apply -v     # 详细日志
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DC = ROOT / "data_cache"

# Files that must never be deleted regardless of retention rules
PROTECTED_PATTERNS = [
    re.compile(r"^portfolio\.xlsx\.bak_pre_positions_rebuild_"),
    re.compile(r"^portfolio\.xlsx\.bak_pre_trade$"),
]

# (group_name_pattern, retention_count, group_label)
GROUP_RULES = [
    # baidu_kline family: keep newest 2
    (re.compile(r"^baidu_kline(\.parquet)?\..*\.bak$"), 2, "baidu_kline"),
    # portfolio.xlsx family: keep newest 2
    (re.compile(r"^portfolio\.xlsx\.bak.*$"), 2, "portfolio.xlsx"),
    # csi300_margin family: keep newest 1
    (re.compile(r"^csi300_margin.*\.bak$"), 1, "csi300_margin"),
    # v17_dens predictions: keep newest 1 per file-stem
    (re.compile(r"^v17_dens_train24_predictions.*\.bak$"), 1, "v17_dens_train24_predictions"),
    (re.compile(r"^v17_dens_csi500_train24_predictions.*\.bak$"), 1, "v17_dens_csi500_train24_predictions"),
    # portfolio_state.json family: keep newest 1
    (re.compile(r"^portfolio_state\.json\.bak.*$"), 1, "portfolio_state.json"),
]


def is_protected(name: str) -> bool:
    return any(p.match(name) for p in PROTECTED_PATTERNS)


def group_for(name: str) -> tuple[str, int] | None:
    for pat, n, label in GROUP_RULES:
        if pat.match(name):
            return label, n
    return None


def find_baks() -> list[Path]:
    return sorted(
        [p for p in DC.iterdir() if p.is_file() and (".bak" in p.name)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def plan(verbose: bool = False) -> tuple[list[Path], list[Path], int]:
    """Return (delete_list, keep_list, ungrouped_count)."""
    all_baks = find_baks()
    by_group: dict[str, list[Path]] = defaultdict(list)
    ungrouped: list[Path] = []
    for p in all_baks:
        g = group_for(p.name)
        if g is None:
            ungrouped.append(p)
            continue
        by_group[g[0]].append(p)

    delete: list[Path] = []
    keep: list[Path] = []
    for label, files in by_group.items():
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        n = None
        for pat, count, lbl in GROUP_RULES:
            if lbl == label:
                n = count
                break
        assert n is not None
        for i, p in enumerate(files):
            if is_protected(p.name):
                keep.append(p)
                if verbose:
                    print(f"  [keep:protected] {p.name}")
                continue
            if i < n:
                keep.append(p)
                if verbose:
                    print(f"  [keep:newest-{i+1}] {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
            else:
                delete.append(p)
                if verbose:
                    print(f"  [DEL] {p.name}  ({p.stat().st_size/1e6:.1f} MB)")

    for p in ungrouped:
        keep.append(p)
        if verbose:
            print(f"  [keep:ungrouped] {p.name}")

    return delete, keep, len(ungrouped)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="explicit dry-run (default behavior)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    apply = args.apply and not args.dry_run
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[cleanup_old_baks] mode={mode}  dir={DC}")

    delete, keep, n_un = plan(verbose=args.verbose)

    total_free = sum(p.stat().st_size for p in delete)
    print(f"\n[plan] keep={len(keep)}  delete={len(delete)}  ungrouped={n_un}")
    print(f"[plan] would free: {total_free/1e6:.1f} MB")

    if not delete:
        print("[done] nothing to clean")
        return 0

    print("\n[delete list]")
    for p in delete:
        print(f"  {p.stat().st_size/1e6:>8.1f} MB  {p.name}")

    if not apply:
        print("\n[dry-run] re-run with --apply to actually delete")
        return 0

    print("\n[apply] deleting...")
    freed = 0
    for p in delete:
        sz = p.stat().st_size
        try:
            p.unlink()
            freed += sz
            print(f"  rm {p.name}")
        except Exception as e:
            print(f"  FAIL {p.name}: {e}", file=sys.stderr)
    print(f"\n[done] freed {freed/1e6:.1f} MB across {len(delete)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
