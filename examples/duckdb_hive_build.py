"""DuckDB Stage 2 — 把 baidu_kline.parquet 拆成 Hive 分区 (per-stock).

布局:
    data_cache/baidu_kline_hive/code=XXXXXX/data.parquet

设计要点:
- 主 parquet 只读: data_cache/baidu_kline.parquet 不动
- atomic: 先 write 到 `_tmp` 同级目录,完成后 rename 替换最终目录
- 幂等: row-count 一致则跳过 (--force 重建)
- 用 DuckDB COPY ... TO ... (PARTITION_BY) 一次性写出,比 per-code python loop 快

用法:
    .venv/bin/python examples/duckdb_hive_build.py            # build (skip if up-to-date)
    .venv/bin/python examples/duckdb_hive_build.py --force    # rebuild full
    .venv/bin/python examples/duckdb_hive_build.py --check    # 仅 verify 分区数+行数
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DC = ROOT / "data_cache"
SRC_PARQUET = DC / "baidu_kline.parquet"
HIVE_DIR = DC / "baidu_kline_hive"
TMP_DIR = DC / "baidu_kline_hive._tmp"


def build(force: bool = False) -> None:
    if not SRC_PARQUET.exists():
        raise SystemExit(f"[fatal] missing source: {SRC_PARQUET}")

    if HIVE_DIR.exists() and not force:
        con = duckdb.connect()
        src_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{SRC_PARQUET}')").fetchone()[0]
        try:
            hive_n = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{HIVE_DIR}/**/*.parquet', hive_partitioning=true)"
            ).fetchone()[0]
        except Exception:
            hive_n = -1
        con.close()
        if src_n == hive_n:
            print(f"[skip] hive already up-to-date: {hive_n:,} rows == src ({src_n:,})")
            return
        print(f"[rebuild] hive {hive_n:,} != src {src_n:,}, rebuilding")

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    t0 = time.time()
    con = duckdb.connect()
    sql = f"""
        COPY (
            SELECT * FROM read_parquet('{SRC_PARQUET}')
        ) TO '{TMP_DIR}'
        (FORMAT PARQUET, PARTITION_BY (code), OVERWRITE_OR_IGNORE)
    """
    con.execute(sql)
    con.close()
    t1 = time.time()
    print(f"[write] tmp build wall = {t1 - t0:.1f}s")

    if HIVE_DIR.exists():
        backup = HIVE_DIR.with_suffix(".prev")
        if backup.exists():
            shutil.rmtree(backup)
        HIVE_DIR.rename(backup)
        print(f"[swap] moved old hive → {backup.name}")
    TMP_DIR.rename(HIVE_DIR)
    print(f"[swap] new hive at {HIVE_DIR}")

    verify()


def verify() -> None:
    if not HIVE_DIR.exists():
        print(f"[miss] {HIVE_DIR}")
        return
    parts = sorted(HIVE_DIR.glob("code=*"))
    files = list(HIVE_DIR.glob("code=*/*.parquet"))
    con = duckdb.connect()
    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{HIVE_DIR}/**/*.parquet', hive_partitioning=true)"
    ).fetchone()[0]
    src_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{SRC_PARQUET}')").fetchone()[0]
    con.close()
    total_size = sum(f.stat().st_size for f in files)
    src_size = SRC_PARQUET.stat().st_size
    print(f"[verify] partitions = {len(parts)}, parquet files = {len(files)}")
    print(f"[verify] hive rows  = {n_rows:,}")
    print(f"[verify] src rows   = {src_n:,}  match={'YES' if n_rows == src_n else 'NO'}")
    print(f"[verify] hive size  = {total_size / 1e6:.1f} MB  ({total_size / src_size:.2f}× src {src_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if up-to-date")
    ap.add_argument("--check", action="store_true", help="只 verify, 不 build")
    args = ap.parse_args()
    if args.check:
        verify()
        return
    build(force=args.force)


if __name__ == "__main__":
    main()
