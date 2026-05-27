"""Benchmark PostgreSQL + TimescaleDB vs DuckDB + parquet.

Runs 3 queries × 3 runs (median) on:
  - PG hypertable (kline) — TimescaleDB chunks + idx_kline_code_date
  - PG plain table (kline_plain) — same data, no hypertable, same indexes
  - DuckDB on baidu_kline.parquet (via duckdb.read_parquet)
  - Direct parquet read via pandas

Compared against published Stage 1 / Stage 2 numbers from
docs/duckdb_stage2_benchmark.md.

Queries:
  A: single stock, last 60 days
     SELECT * FROM kline WHERE code='600519' AND date >= '2026-03-01'
  B: full market, latest day
     SELECT * FROM kline WHERE date = (SELECT MAX(date) FROM kline)
  C: full-table COUNT
     SELECT COUNT(*) FROM kline

Reports median of 3 runs in ms (warm cache, matches Stage 2 protocol).

Usage:
    python examples/pg_benchmark.py
"""
from __future__ import annotations

import statistics
import time

import duckdb
import pandas as pd
import psycopg2

PG_DSN = "host=localhost port=5432 user=claudefinance password=claudefinance dbname=claude_finance"
KLINE_PATH = "data_cache/baidu_kline.parquet"


def time_pg(conn: psycopg2.extensions.connection, sql: str, fetch: str = "all") -> float:
    with conn.cursor() as cur:
        t0 = time.perf_counter()
        cur.execute(sql)
        if fetch == "all":
            _ = cur.fetchall()
        elif fetch == "one":
            _ = cur.fetchone()
        return (time.perf_counter() - t0) * 1000


def time_duckdb(con: duckdb.DuckDBPyConnection, sql: str, fetch: str = "all") -> float:
    t0 = time.perf_counter()
    res = con.execute(sql)
    if fetch == "all":
        _ = res.fetchall()
    elif fetch == "one":
        _ = res.fetchone()
    return (time.perf_counter() - t0) * 1000


def time_pandas(parquet_path: str, fn) -> float:
    t0 = time.perf_counter()
    _ = fn(parquet_path)
    return (time.perf_counter() - t0) * 1000


def median_of_n(runner, n: int = 3) -> tuple[float, list[float]]:
    samples = [runner() for _ in range(n)]
    return statistics.median(samples), samples


def create_plain_table_if_missing(conn: psycopg2.extensions.connection) -> bool:
    """Create kline_plain (vanilla PG table, no hypertable) for comparison.
    Returns True if newly created, False if already populated."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name='kline_plain'"
        )
        exists = cur.fetchone() is not None
        if exists:
            cur.execute("SELECT COUNT(*) FROM kline_plain")
            n = cur.fetchone()[0]
            if n > 0:
                return False
        cur.execute("DROP TABLE IF EXISTS kline_plain")
        cur.execute(
            """
            CREATE TABLE kline_plain (
                code            TEXT NOT NULL,
                date            TIMESTAMPTZ NOT NULL,
                open            DOUBLE PRECISION,
                close           DOUBLE PRECISION,
                high            DOUBLE PRECISION,
                low             DOUBLE PRECISION,
                vol             BIGINT,
                amount          DOUBLE PRECISION,
                ma5             DOUBLE PRECISION,
                ma10            DOUBLE PRECISION,
                ma20            DOUBLE PRECISION,
                turnoverratio   DOUBLE PRECISION
            )
            """
        )
        cur.execute("INSERT INTO kline_plain SELECT * FROM kline")
        cur.execute("CREATE INDEX idx_kline_plain_code_date ON kline_plain (code, date DESC)")
        cur.execute("CREATE INDEX idx_kline_plain_date ON kline_plain (date DESC)")
        cur.execute("ANALYZE kline_plain")
    conn.commit()
    return True


def main() -> None:
    print("=" * 78)
    print("PG + TimescaleDB benchmark vs DuckDB + parquet")
    print("=" * 78)

    conn = psycopg2.connect(PG_DSN)

    print("\n[setup] Ensuring kline_plain (vanilla PG, no hypertable) ...")
    t0 = time.perf_counter()
    created = create_plain_table_if_missing(conn)
    if created:
        print(f"    Created kline_plain in {time.perf_counter() - t0:.2f}s")
    else:
        print("    kline_plain already populated (reusing)")

    con_duck = duckdb.connect(":memory:")
    con_duck.execute(f"CREATE VIEW kline_pq AS SELECT * FROM read_parquet('{KLINE_PATH}')")

    sql_a = "SELECT * FROM kline WHERE code='600519' AND date >= '2026-03-01'"
    sql_b = "SELECT * FROM kline WHERE date = (SELECT MAX(date) FROM kline)"
    sql_c = "SELECT COUNT(*) FROM kline"
    print("\n[warmup] Each backend, all 3 queries ...")
    for s in (sql_a, sql_b, sql_c):
        time_pg(conn, s, fetch="all" if "COUNT" not in s else "one")
        time_pg(conn, s.replace("kline", "kline_plain"), fetch="all" if "COUNT" not in s else "one")
        time_duckdb(con_duck, s.replace("kline", "kline_pq"),
                    fetch="all" if "COUNT" not in s else "one")

    print("\n[bench] median of 3 runs (warm cache)\n")

    backends = [
        ("PG hypertable",   "kline",       conn),
        ("PG plain",        "kline_plain", conn),
        ("DuckDB parquet",  "kline_pq",    con_duck),
    ]

    queries = [
        ("A. single stock 60d",
         "SELECT * FROM {t} WHERE code='600519' AND date >= '2026-03-01'",
         "all"),
        ("B. full market latest day",
         "SELECT * FROM {t} WHERE date = (SELECT MAX(date) FROM {t})",
         "all"),
        ("C. full-table COUNT(*)",
         "SELECT COUNT(*) FROM {t}",
         "one"),
    ]

    results: dict[str, dict[str, tuple[float, list[float]]]] = {}
    for q_label, q_template, fetch in queries:
        print(f"--- {q_label}")
        results[q_label] = {}
        for backend_label, table, c in backends:
            sql = q_template.format(t=table)
            if isinstance(c, psycopg2.extensions.connection):
                med, samples = median_of_n(
                    lambda c=c, sql=sql, fetch=fetch: time_pg(c, sql, fetch=fetch)
                )
            else:
                med, samples = median_of_n(
                    lambda c=c, sql=sql, fetch=fetch: time_duckdb(c, sql, fetch=fetch)
                )
            results[q_label][backend_label] = (med, samples)
            samples_s = ", ".join(f"{s:.2f}" for s in samples)
            print(f"    {backend_label:<18s}  median={med:>9.2f} ms   runs=[{samples_s}]")
        print()

    print("=" * 78)
    print("SUMMARY (median ms, lower=better)")
    print("=" * 78)
    header = f"| {'query':<26s} |"
    for b, _, _ in backends:
        header += f" {b:>16s} |"
    print(header)
    print("|" + "-" * 28 + "|" + ("-" * 18 + "|") * len(backends))
    for q_label, _, _ in queries:
        row = f"| {q_label:<26s} |"
        for b, _, _ in backends:
            med = results[q_label][b][0]
            row += f" {med:>15.2f}m |"
        print(row)
    print()

    print("[extra] pandas read_parquet single-stock filter (vs DuckDB)")

    def pandas_a(_path):
        df = pd.read_parquet(_path)
        return df[(df["code"] == "600519") & (df["date"] >= pd.Timestamp("2026-03-01"))]

    samples = [time_pandas(KLINE_PATH, pandas_a) for _ in range(3)]
    print(f"    pandas read_parquet+filter  median={statistics.median(samples):>9.2f} ms   "
          f"runs={['%.0f' % s for s in samples]}")

    conn.close()
    con_duck.close()


if __name__ == "__main__":
    main()
