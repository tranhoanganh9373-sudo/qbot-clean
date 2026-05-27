"""Migrate baidu_kline.parquet + csi300_margin_14yr.parquet to PostgreSQL.

Used by docs/pg_timescale_benchmark.md task. Loads two parquet files into a
TimescaleDB hypertable + a vanilla PG table (margin), reports wall times,
row counts, and on-disk sizes. **Does not touch** any production data
(reads parquet only, writes to new PG db `claude_finance`).

Usage:
    python examples/pg_migrate_kline_margin.py

Connection: dockerized timescaledb-ha:pg16 on localhost:5432
            user=claudefinance pass=claudefinance db=claude_finance
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import psycopg2

PG_DSN = "host=localhost port=5432 user=claudefinance password=claudefinance dbname=claude_finance"
KLINE_PATH = Path("data_cache/baidu_kline.parquet")
MARGIN_PATH = Path("data_cache/csi300_margin_14yr.parquet")


def connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(PG_DSN)


def drop_and_create_kline(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS kline CASCADE")
        cur.execute(
            """
            CREATE TABLE kline (
                code            TEXT          NOT NULL,
                date            TIMESTAMPTZ   NOT NULL,
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
        cur.execute(
            "SELECT create_hypertable('kline', 'date', chunk_time_interval => INTERVAL '1 month')"
        )
    conn.commit()


def drop_and_create_margin(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS margin CASCADE")
        cur.execute(
            """
            CREATE TABLE margin (
                code            TEXT          NOT NULL,
                date            TIMESTAMPTZ   NOT NULL,
                rzye            DOUBLE PRECISION,
                rzmre           DOUBLE PRECISION,
                rzche           DOUBLE PRECISION,
                margin_5d_chg   DOUBLE PRECISION,
                margin_20d_chg  DOUBLE PRECISION
            )
            """
        )
    conn.commit()


def copy_dataframe(
    conn: psycopg2.extensions.connection,
    df: pd.DataFrame,
    table: str,
    columns: list[str],
    chunk_rows: int = 500_000,
) -> float:
    """Bulk-load a pandas DataFrame via psycopg2 COPY ... FROM STDIN, chunked.

    Single 7.9M-row StringIO buffer can stall the COPY (observed: 30+ min idle,
    0% CPU). Chunking to ~500k rows keeps the buffer in CPU L3 / page cache and
    commits incrementally so progress is observable. Total wall is ~equal to
    or better than monolithic, with no stall risk.
    """
    t0 = time.perf_counter()
    n = len(df)
    written = 0
    for start in range(0, n, chunk_rows):
        chunk = df.iloc[start:start + chunk_rows]
        buf = io.StringIO()
        chunk.to_csv(
            buf, index=False, header=False, na_rep="\\N", date_format="%Y-%m-%d %H:%M:%S"
        )
        buf.seek(0)
        with conn.cursor() as cur:
            cur.copy_expert(
                f"COPY {table} ({','.join(columns)}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')",
                buf,
            )
        conn.commit()
        written += len(chunk)
        elapsed = time.perf_counter() - t0
        print(f"        [{table}] {written:>9,}/{n:,} rows  ({100*written/n:5.1f}%)  "
              f"+{len(chunk):,} in chunk  elapsed={elapsed:.1f}s")
    return time.perf_counter() - t0


def create_indexes_kline(conn: psycopg2.extensions.connection) -> float:
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX idx_kline_code_date ON kline (code, date DESC)")
    conn.commit()
    return time.perf_counter() - t0


def create_indexes_margin(conn: psycopg2.extensions.connection) -> float:
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX idx_margin_code_date ON margin (code, date DESC)")
        cur.execute("CREATE INDEX idx_margin_date ON margin (date DESC)")
    conn.commit()
    return time.perf_counter() - t0


def hypertable_size_bytes(conn: psycopg2.extensions.connection, table: str) -> int:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT hypertable_size(%s)", (table,))
            return cur.fetchone()[0]
    except Exception:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT pg_total_relation_size(%s)", (table,))
            return cur.fetchone()[0]


def plain_table_size_bytes(conn: psycopg2.extensions.connection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_total_relation_size(%s)", (table,))
        return cur.fetchone()[0]


def fetch_row_count(conn: psycopg2.extensions.connection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]


def human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024.0:
            return f"{f:.1f} {unit}"
        f /= 1024.0
    return f"{f:.1f} TB"


def main() -> None:
    overall_t0 = time.perf_counter()

    print("=" * 70)
    print("PG migration: baidu_kline + csi300_margin_14yr → PostgreSQL 16 + TimescaleDB")
    print("=" * 70)

    t0 = time.perf_counter()
    print(f"\n[1] Reading {KLINE_PATH} ...")
    df_kline = pd.read_parquet(KLINE_PATH)
    print(f"    rows={len(df_kline):,}  read_wall={time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    print(f"\n[2] Reading {MARGIN_PATH} ...")
    df_margin = pd.read_parquet(MARGIN_PATH)
    print(f"    rows={len(df_margin):,}  read_wall={time.perf_counter() - t0:.2f}s")

    conn = connect()
    print("\n[3] Creating tables (kline hypertable, margin vanilla) ...")
    drop_and_create_kline(conn)
    drop_and_create_margin(conn)

    print("\n[4] Bulk loading kline via COPY ...")
    kline_cols = [
        "code", "date", "open", "close", "high", "low",
        "vol", "amount", "ma5", "ma10", "ma20", "turnoverratio",
    ]
    kline_copy_wall = copy_dataframe(conn, df_kline[kline_cols], "kline", kline_cols)
    print(f"    kline COPY wall={kline_copy_wall:.2f}s "
          f"({len(df_kline) / kline_copy_wall:,.0f} rows/sec)")

    print("\n[5] Bulk loading margin via COPY ...")
    margin_cols = [
        "code", "date", "rzye", "rzmre", "rzche", "margin_5d_chg", "margin_20d_chg",
    ]
    margin_copy_wall = copy_dataframe(conn, df_margin[margin_cols], "margin", margin_cols)
    print(f"    margin COPY wall={margin_copy_wall:.2f}s "
          f"({len(df_margin) / margin_copy_wall:,.0f} rows/sec)")

    print("\n[6] Building secondary indexes ...")
    idx_k = create_indexes_kline(conn)
    idx_m = create_indexes_margin(conn)
    print(f"    kline idx_code_date wall={idx_k:.2f}s")
    print(f"    margin idx_code_date + idx_date wall={idx_m:.2f}s")

    print("\n[7] ANALYZE ...")
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("ANALYZE kline")
        cur.execute("ANALYZE margin")
    conn.commit()
    print(f"    ANALYZE wall={time.perf_counter() - t0:.2f}s")

    print("\n[8] Stats:")
    k_rows = fetch_row_count(conn, "kline")
    m_rows = fetch_row_count(conn, "margin")
    k_size = hypertable_size_bytes(conn, "kline")
    m_size = plain_table_size_bytes(conn, "margin")
    print(f"    kline: {k_rows:,} rows, {human_size(k_size)} on disk")
    print(f"    margin: {m_rows:,} rows, {human_size(m_size)} on disk")
    print(f"    kline parquet: {human_size(KLINE_PATH.stat().st_size)}")
    print(f"    margin parquet: {human_size(MARGIN_PATH.stat().st_size)}")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM timescaledb_information.chunks WHERE hypertable_name='kline'"
        )
        n_chunks = cur.fetchone()[0]
        print(f"    kline hypertable chunks: {n_chunks}")

    conn.close()
    print(f"\nTotal wall: {time.perf_counter() - overall_t0:.2f}s")
    print("Done.")


if __name__ == "__main__":
    main()
