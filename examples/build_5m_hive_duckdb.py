"""Phase 2: 把 per-stock 5m shards 合并为 Hive partition + 注册 DuckDB view.

输入: data_cache/kline_5m_shards/{QID}.parquet
输出:
  - data_cache/kline_5m_hive/symbol={QID}/data.parquet   (snappy)
  - data_cache/db.duckdb  (view: kline_5m)

注意:
  - 当前 daily kline_hive view 因 4690 files 慢 11-740× 已 dropped (见
    docs/duckdb_quickstart.md). 5m 在更小 universe (~3795 partitions) 上
    重做相同模式; 单股查询可经 path-direct read 加速。
  - 本脚本只新增 5m view, 不触现有 daily/margin/predictions views.

CLI:
  python examples/build_5m_hive_duckdb.py            # full build
  python examples/build_5m_hive_duckdb.py --force    # 强制重建 hive
  python examples/build_5m_hive_duckdb.py --no-smoke
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data_cache" / "kline_5m_shards"
HIVE_DIR = ROOT / "data_cache" / "kline_5m_hive"
DB_PATH = ROOT / "data_cache" / "db.duckdb"

VIEW_NAME = "kline_5m"


def build_hive(force: bool = False) -> tuple[int, int, float]:
    """Copy each shard to symbol={CODE}/data.parquet hive layout."""
    if not SHARDS_DIR.exists():
        raise FileNotFoundError(f"shards dir missing: {SHARDS_DIR}")

    shards = sorted(SHARDS_DIR.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"no shards in {SHARDS_DIR}")

    t0 = time.time()
    n_written = 0
    n_skipped = 0
    HIVE_DIR.mkdir(parents=True, exist_ok=True)

    for shard in shards:
        qid = shard.stem
        partition_dir = HIVE_DIR / f"symbol={qid}"
        out = partition_dir / "data.parquet"
        if out.exists() and not force:
            n_skipped += 1
            continue
        partition_dir.mkdir(parents=True, exist_ok=True)
        # parquet 已经 snappy compressed; 直接 copy 保留 schema/compression
        shutil.copyfile(shard, out)
        n_written += 1

    wall = time.time() - t0
    return n_written, n_skipped, wall


def register_view() -> tuple[int, int]:
    """Register kline_5m view in db.duckdb."""
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
        glob = str(HIVE_DIR / "**" / "*.parquet")
        con.execute(
            f"CREATE VIEW {VIEW_NAME} AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
        )
        row = con.execute(
            f"SELECT COUNT(*) AS n_rows, COUNT(DISTINCT symbol) AS n_codes FROM {VIEW_NAME}"
        ).fetchone()
        return int(row[0]), int(row[1])
    finally:
        con.close()


def smoke_queries() -> None:
    """Run 3 smoke queries to confirm view works end-to-end."""
    con = duckdb.connect(str(DB_PATH))
    try:
        print("\n--- smoke 1: 单股某日 5m bar 数 ---")
        df = con.execute(f"""
            SELECT symbol, COUNT(*) AS n_bars,
                   MIN(datetime) AS first_ts, MAX(datetime) AS last_ts
            FROM {VIEW_NAME}
            WHERE symbol = 'SH600519'
              AND datetime >= '2026-05-26' AND datetime < '2026-05-27'
            GROUP BY symbol
        """).fetchdf()
        print(df.to_string(index=False))

        print("\n--- smoke 2: 集合竞价 9:35 bar 全 universe 数 ---")
        df = con.execute(f"""
            SELECT datetime, COUNT(*) AS n_codes
            FROM {VIEW_NAME}
            WHERE datetime = '2026-05-26 09:35:00'
            GROUP BY datetime
        """).fetchdf()
        print(df.to_string(index=False))

        print("\n--- smoke 3: 集合竞价 9:35 bar - 2026-05-26 top 5 amount ---")
        df = con.execute(f"""
            SELECT symbol, open, close, volume, amount
            FROM {VIEW_NAME}
            WHERE datetime = '2026-05-26 09:35:00'
            ORDER BY amount DESC LIMIT 5
        """).fetchdf()
        print(df.to_string(index=False))
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="强制覆盖已存在 hive 文件")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()

    print(f"[main] shards: {SHARDS_DIR}")
    print(f"[main] hive:   {HIVE_DIR}")
    print(f"[main] db:     {DB_PATH}")

    n_written, n_skipped, wall = build_hive(force=args.force)
    print(f"\n[hive] written={n_written} skipped={n_skipped} wall={wall:.1f}s")

    hive_bytes = sum(p.stat().st_size for p in HIVE_DIR.rglob("*.parquet"))
    n_partitions = sum(1 for p in HIVE_DIR.iterdir() if p.is_dir())
    print(f"[hive] partitions={n_partitions} total={hive_bytes/1e9:.2f} GB")

    n_rows, n_codes = register_view()
    print(f"\n[view] kline_5m: {n_rows:,} rows × {n_codes} symbols")

    if not args.no_smoke:
        smoke_queries()

    print("\n[main] done.")


if __name__ == "__main__":
    main()
