"""一次性脚本: 把 OOS subagent 抓的 data_cache/csi300_margin_14yr.parquet
拆分为 data_cache/margin_eastmoney/{code}.parquet 单股 cache.

用法 (OOS subagent 完成后, 手动跑):
    python examples/migrate_margin_cache.py

约束:
- 此脚本本身只读不抓 (不发起任何网络请求)
- 启动前检查 bulk parquet 存在 + 最后修改时间 > 5 分钟前 (避免半成品)
- 不修改 factor_mining.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow `python examples/migrate_margin_cache.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claude_finance import margin_cache  # noqa: E402

BULK_PATH = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
MIN_AGE_SECONDS = 5 * 60  # 5 分钟, 防半成品


def main() -> int:
    if not BULK_PATH.exists():
        print(f"[ERR] bulk parquet 不存在: {BULK_PATH}")
        print("      请先等 OOS subagent 完成抓取再跑此脚本.")
        return 1

    age = time.time() - BULK_PATH.stat().st_mtime
    if age < MIN_AGE_SECONDS:
        mins = age / 60
        print(f"[ERR] bulk parquet 最后修改距今仅 {mins:.1f} 分钟 (<5 min).")
        print("      可能 OOS subagent 还在写入. 请稍后再跑.")
        return 2

    print(f"[INFO] 读取 bulk parquet: {BULK_PATH}")
    print(f"[INFO] 输出目录: {margin_cache.CACHE_DIR}")
    n = margin_cache.bootstrap_from_bulk_parquet(BULK_PATH)
    print(f"[OK] 写入 {n} 只股的 per-stock cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
