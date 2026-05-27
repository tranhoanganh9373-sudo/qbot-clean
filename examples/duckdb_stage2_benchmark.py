"""DuckDB Stage 2 — benchmark Stage 1 (single parquet view `kline`) vs Stage 2 (Hive view `kline_hive`).

3 queries:
    A) single stock recent 60-day      (single code + date range)
    B) all-market latest day            (no code filter)
    C) full-table COUNT(*)              (no selectivity)

每 query 跑 3 次取 median wall + peak RSS delta (process-level).
"""
from __future__ import annotations

import resource
import statistics
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data_cache" / "db.duckdb"


def rss_mb() -> float:
    # macOS getrusage ru_maxrss as bytes; Linux as KB
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return val / 1e6 if val > 2**30 else val / 1024


QUERIES = [
    ("A_single_stock_recent60",
     "SELECT * FROM {v} WHERE code = '600519' AND date >= '2026-03-01'"),
    ("B_all_market_latest_day",
     "SELECT * FROM {v} WHERE date = (SELECT MAX(date) FROM {v})"),
    ("C_full_count",
     "SELECT COUNT(*) FROM {v}"),
]


def time_query(con: duckdb.DuckDBPyConnection, sql: str, runs: int = 3) -> tuple[float, float, int]:
    walls = []
    n_rows = 0
    rss_before = rss_mb()
    for _ in range(runs):
        t0 = time.perf_counter()
        rs = con.execute(sql).fetchall()
        t1 = time.perf_counter()
        walls.append(t1 - t0)
        n_rows = len(rs) if rs else 0
    rss_after = rss_mb()
    return statistics.median(walls), rss_after - rss_before, n_rows


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    print(f"[benchmark] db: {DB_PATH}\n")
    print(f"{'query':28s} {'view':12s} {'wall_ms':>10s} {'rss_d_MB':>10s} {'rows':>10s}")
    print("-" * 78)
    rows = []
    for qname, qtpl in QUERIES:
        for view in ("kline", "kline_hive"):
            sql = qtpl.format(v=view)
            wall, rss_d, n = time_query(con, sql, runs=3)
            print(f"{qname:28s} {view:12s} {wall * 1000:>10.2f} {rss_d:>10.2f} {n:>10d}")
            rows.append((qname, view, wall, rss_d, n))
        print()
    print("=" * 78)
    print(f"{'query':28s} {'stage1_ms':>10s} {'stage2_ms':>10s} {'speedup':>10s}")
    print("-" * 78)
    by_q: dict[str, dict[str, float]] = {}
    for qname, view, wall, _, _ in rows:
        by_q.setdefault(qname, {})[view] = wall
    for qname, d in by_q.items():
        s1 = d["kline"] * 1000
        s2 = d["kline_hive"] * 1000
        sp = s1 / s2 if s2 > 0 else float("inf")
        print(f"{qname:28s} {s1:>10.2f} {s2:>10.2f} {sp:>9.2f}×")


if __name__ == "__main__":
    main()
