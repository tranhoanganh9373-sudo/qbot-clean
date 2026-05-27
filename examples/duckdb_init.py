"""DuckDB Stage 1 — 创建 views over 现有 parquet/csv,不动主存储.

幂等: 重跑安全,会 DROP + CREATE 全部 views.
零 server, embedded. db 路径: data_cache/db.duckdb (~1 KB metadata only, 数据走 parquet 引用).

用法:
  .venv/bin/python examples/duckdb_init.py        # 初始化/重建 views
  .venv/bin/python examples/duckdb_init.py --demo # 跑 demo 查询
  .venv/bin/python examples/duckdb_init.py --list # 列出当前 views + 行数
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DC = ROOT / "data_cache"
DB_PATH = DC / "db.duckdb"

# (view_name, source_kind, source_path, notes)
# source_kind: 'parquet' | 'csv'
VIEW_SPECS: list[tuple[str, str, str, str]] = [
    ("kline", "parquet", str(DC / "baidu_kline.parquet"),
     "主行情 daily OHLCV. 4625 codes, 7.9M rows, 2012-09 ~ 2026-05"),
    # Stage 2 kline_hive view removed 2026-05-26: 慢 11-740× vs 主表. Hive 物理文件仍在
    # data_cache/baidu_kline_hive/, 单股快路径用 dashboard/utils/kline_fast.py:get_stock_kline().
    ("margin", "parquet", str(DC / "csi300_margin_14yr.parquet"),
     "CSI300 融资融券 14yr + margin_5d_chg/20d_chg pre-computed. 300 codes."),
    ("predictions", "parquet", str(DC / "v17_dens_train24_predictions.parquet"),
     "v17 DEnsemble train24 walk-forward predictions. 112 月 OOS."),
    ("fund_flow", "parquet", str(DC / "fund_flow" / "fund_flow_csi300.parquet"),
     "Sina 资金流分层 (Phase A 203 股 IS 2014-2020)."),
    ("shareholders", "parquet", str(DC / "shareholders" / "shareholders_csi300.parquet"),
     "CSI300 股东户数季度 (289 股, PIT announce_date)."),
    ("industry_membership", "parquet", str(DC / "industry" / "industry_membership.parquet"),
     "SW level-1 行业归属 snapshot. 31 boards × 5203 stocks."),
    ("csi300", "csv", str(DC / "csi300_constituents.csv"),
     "CSI300 成分股名单 (current snapshot)."),
    ("portfolio_log", "csv", str(DC / "paper_trade_log.csv"),
     "production paper_trade daily 信号 log."),
]


def setup_views(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    for view_name, kind, src, _ in VIEW_SPECS:
        path = Path(src)
        if not path.exists():
            results.append((view_name, "miss", f"file not found: {src}"))
            continue
        try:
            con.execute(f"DROP VIEW IF EXISTS {view_name}")
            if kind == "parquet":
                con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{src}')")
            elif kind == "parquet_hive":
                con.execute(
                    f"CREATE VIEW {view_name} AS SELECT * FROM "
                    f"read_parquet('{src}/**/*.parquet', hive_partitioning=true)"
                )
            elif kind == "csv":
                con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_csv_auto('{src}')")
            results.append((view_name, "ok", path.name))
        except Exception as e:
            results.append((view_name, "fail", f"{type(e).__name__}: {e}"))
    return results


def list_views(con: duckdb.DuckDBPyConnection) -> None:
    views = con.execute(
        "SELECT view_name FROM duckdb_views WHERE schema_name='main' ORDER BY view_name"
    ).fetchall()
    print(f"\n[views in {DB_PATH.name}]")
    for (vn,) in views:
        try:
            desc = con.execute(f"DESCRIBE {vn}").fetchall()
            cols = [row[0] for row in desc]
            n = con.execute(f"SELECT COUNT(*) FROM {vn}").fetchone()[0]
            print(f"  {vn:22s}  {n:>10,} rows × {len(cols):2d} cols  {cols[:6]}{'...' if len(cols) > 6 else ''}")
        except Exception as e:
            print(f"  {vn:22s}  ! {type(e).__name__}: {e}")


def demo(con: duckdb.DuckDBPyConnection) -> None:
    print("\n[demo 1] CSI300 最近一日最高 5d 涨幅 top 10")
    print(con.execute("""
        WITH dated AS (
            SELECT
                LPAD(CAST(code AS VARCHAR), 6, '0') AS code_str,
                date, close
            FROM kline
        ),
        latest AS (SELECT MAX(date) AS d FROM dated),
        now AS (SELECT code_str, close AS now_close FROM dated WHERE date = (SELECT d FROM latest)),
        prev AS (
            SELECT code_str, close AS prev_close FROM dated
            WHERE date = (SELECT MAX(date) FROM dated WHERE date < (SELECT d FROM latest) - INTERVAL 4 DAY)
        )
        SELECT
            n.code_str AS code,
            ROUND((n.now_close - p.prev_close) / p.prev_close * 100, 2) AS chg_5d_pct,
            ROUND(n.now_close, 2) AS close_now
        FROM now n
        JOIN prev p USING (code_str)
        JOIN csi300 c ON n.code_str = LPAD(CAST(c.code AS VARCHAR), 6, '0')
        ORDER BY chg_5d_pct DESC
        LIMIT 10
    """).fetchdf().to_string(index=False))

    print("\n[demo 2] margin 最近 full-day, margin_5d_chg 前 10 (融资余额暴增)")
    print(con.execute("""
        WITH full_days AS (
            SELECT date FROM margin GROUP BY date HAVING COUNT(*) >= 200
        )
        SELECT
            LPAD(CAST(code AS VARCHAR), 6, '0') AS code,
            date,
            ROUND(margin_5d_chg * 100, 2) AS m5_chg_pct,
            ROUND(rzye / 1e8, 2) AS rzye_yi
        FROM margin
        WHERE date = (SELECT MAX(date) FROM full_days)
        ORDER BY margin_5d_chg DESC
        LIMIT 10
    """).fetchdf().to_string(index=False))

    print("\n[demo 3] 最新 predictions Top 8 + close + margin (跨 3 table)")
    print(con.execute("""
        WITH latest_pred AS (
            SELECT instrument, score FROM predictions
            WHERE month = (SELECT MAX(month) FROM predictions)
            ORDER BY score DESC LIMIT 8
        ),
        sym_xform AS (
            SELECT
                CASE
                    WHEN LPAD(CAST(code AS VARCHAR), 6, '0') LIKE '6%'
                      OR LPAD(CAST(code AS VARCHAR), 6, '0') LIKE '9%'
                    THEN 'SH' || LPAD(CAST(code AS VARCHAR), 6, '0')
                    ELSE 'SZ' || LPAD(CAST(code AS VARCHAR), 6, '0')
                END AS sym,
                close, date
            FROM kline
            WHERE date = (SELECT MAX(date) FROM kline)
        ),
        sym_margin AS (
            SELECT
                CASE
                    WHEN LPAD(CAST(code AS VARCHAR), 6, '0') LIKE '6%'
                      OR LPAD(CAST(code AS VARCHAR), 6, '0') LIKE '9%'
                    THEN 'SH' || LPAD(CAST(code AS VARCHAR), 6, '0')
                    ELSE 'SZ' || LPAD(CAST(code AS VARCHAR), 6, '0')
                END AS sym,
                margin_5d_chg
            FROM margin
            WHERE date = (
                SELECT MAX(date) FROM margin
                WHERE date IN (SELECT date FROM margin GROUP BY date HAVING COUNT(*) >= 200)
            )
        )
        SELECT p.instrument, ROUND(p.score, 4) AS score,
               ROUND(s.close, 2) AS close,
               ROUND(m.margin_5d_chg * 100, 2) AS m5_chg_pct
        FROM latest_pred p
        LEFT JOIN sym_xform s ON p.instrument = s.sym
        LEFT JOIN sym_margin m ON p.instrument = m.sym
        ORDER BY p.score DESC
    """).fetchdf().to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    con = duckdb.connect(str(DB_PATH))
    if args.list:
        list_views(con); return
    if args.demo:
        demo(con); return
    print(f"[duckdb-init] db: {DB_PATH}")
    results = setup_views(con)
    n_ok = sum(1 for _, s, _ in results if s == "ok")
    n_miss = sum(1 for _, s, _ in results if s == "miss")
    n_fail = sum(1 for _, s, _ in results if s == "fail")
    for v, st, msg in results:
        sym = {"ok": "✓", "miss": "?", "fail": "✗"}[st]
        print(f"  {sym} {v:22s}  {msg}")
    print(f"\n[done] ok={n_ok} miss={n_miss} fail={n_fail}")
    if n_ok > 0:
        list_views(con)


if __name__ == "__main__":
    main()
