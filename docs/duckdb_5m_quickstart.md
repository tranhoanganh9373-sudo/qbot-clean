# DuckDB 5m K 线 Quickstart

5m intraday OHLCV via DuckDB view on Hive-partitioned parquet。**主存储仍是 parquet**,
DuckDB 只是 query engine,跟现有 daily kline / margin / predictions views 共存。

## 数据来源 & 限制

- **抓取**: mootdx (TDX server pool) — 跟 `fetch_baidu_kline_v2_akshare.py` 同 stack。
- **历史深度上限**: TDX server hard-cap **~2.04 年** (实测 2024-05-10 → 今日 for 600519)。
  - 跨频率一致:1m=0.4 yr / 5m=2.04 yr / 15m=2.04 yr / 30m=2.04 yr / 1h=2.04 yr。
  - **不要期待 5 年深度**;5m 不是 daily,TDX 服务器只保留近期 intraday。
- **当前 ship 版**: 12 pages × 800 bar = **9600 bar/股 ≈ 0.79 年** (压缩 fetch wall 时间)。
  - 后续可重跑 `--max-pages 35` 拉满 2 年(预期 wall ≈ 2.5 hr)。

## 文件布局

```
data_cache/
├── kline_5m_shards/             # 中间产物 (per-stock raw shards from mootdx)
│   ├── SH600519.parquet
│   └── ...
├── kline_5m_hive/               # 主存储 (hive partition for DuckDB)
│   ├── symbol=SH600519/
│   │   └── data.parquet
│   └── ...
└── db.duckdb                    # view metadata (~kB, 数据不拷贝)
```

## Schema

`kline_5m` view (DuckDB):

| col       | type            | example                  | note |
|-----------|-----------------|--------------------------|------|
| code      | VARCHAR         | `SH600519`               | shard 内的 code 列 |
| datetime  | TIMESTAMP       | `2026-05-26 09:35:00`    | 5m bar 开始时刻 |
| open      | DOUBLE          | 1273.38                  | 不复权(TDX 原值) |
| high      | DOUBLE          | 1274.10                  |  |
| low       | DOUBLE          | 1272.50                  |  |
| close     | DOUBLE          | 1273.20                  |  |
| volume    | BIGINT          | 37900                    | 手数 |
| amount    | DOUBLE          | 4.83e7                   | 成交额 |
| symbol    | VARCHAR         | `SH600519`               | hive 分区键 (== code) |

> **重要**: TDX 5m 是**不复权**价。如果你想跟 daily hfq 对齐,需要 join hfq factor 表
> (sina hfq.js),或在策略里只用 intraday 相对量(振幅 / 资金流 / 量比 / 集合竞价乘数等)
> 而不直接跨日比 close。

## 命令

```bash
# Step 1: 抓 shards (~52 min @ 8 workers, 12 pages = ~9 months)
.venv/bin/python examples/fetch_mootdx_5m_5y_backfill.py --workers 8 --max-pages 12

# Step 2: build hive + 注册 view (~10 s)
.venv/bin/python examples/build_5m_hive_duckdb.py

# Step 3: 列出现有 views (5m 应出现)
.venv/bin/python examples/duckdb_init.py --list
```

## Python 用法

```python
import duckdb
con = duckdb.connect("data_cache/db.duckdb")

# 单股某日 5m
df = con.execute("""
    SELECT * FROM kline_5m
    WHERE symbol = 'SH600519' AND datetime >= '2026-05-26'
                              AND datetime <  '2026-05-27'
    ORDER BY datetime
""").fetchdf()
```

## 5 个示例 SQL

### 1. 单股某日 5m K 线

```sql
SELECT datetime, open, high, low, close, volume, amount
FROM kline_5m
WHERE symbol = 'SH600519'
  AND datetime >= '2026-05-26' AND datetime < '2026-05-27'
ORDER BY datetime;
```

### 2. 跨 universe 某 5m time slice (盘中横截面)

```sql
SELECT symbol, close, volume, amount
FROM kline_5m
WHERE datetime = '2026-05-26 14:55:00'
ORDER BY amount DESC
LIMIT 50;
```

### 3. 集合竞价 9:35 bar (盘前情绪)

```sql
-- 9:35 是 5m 周期内第一根 bar (覆盖 9:30~9:35), 含集合竞价撮合 + 盘前 5 分钟
SELECT symbol, open AS auction_open, close AS post_5m_close,
       (close - open) / open AS first_5m_ret,
       volume AS first_5m_vol
FROM kline_5m
WHERE datetime = '2026-05-26 09:35:00'
  AND volume > 0
ORDER BY first_5m_vol DESC
LIMIT 20;
```

### 4. 集合竞价乘数 (open-to-prev-close 跳空) — 需要 daily kline join

> ⚠️ **code 格式不一**: daily `kline` 用 6-digit (`600519`), 5m `kline_5m` 用 qlib-id (`SH600519`).
> 跨表 join 必须把 5m 的 `symbol` 剥掉前 2 字符。

```sql
WITH d AS (
    SELECT code AS raw_code, close AS prev_close
    FROM kline
    WHERE date = '2026-05-23'  -- 上一交易日
), o AS (
    SELECT symbol, substr(symbol, 3) AS raw_code,
           open AS auction_open
    FROM kline_5m
    WHERE datetime = '2026-05-26 09:35:00'
)
SELECT o.symbol, d.prev_close, o.auction_open,
       (o.auction_open - d.prev_close) / d.prev_close AS gap_pct
FROM o JOIN d USING (raw_code)
WHERE d.prev_close > 0
ORDER BY ABS(gap_pct) DESC
LIMIT 20;
```

### 5. 日内反转: 上午高开 + 下午跌 (T+1 不出货)

```sql
WITH morning_high AS (
    SELECT symbol, MAX(high) AS am_high
    FROM kline_5m
    WHERE datetime >= '2026-05-26 09:30:00'
      AND datetime <= '2026-05-26 11:30:00'
    GROUP BY symbol
), close_pm AS (
    SELECT symbol, close AS pm_close
    FROM kline_5m
    WHERE datetime = '2026-05-26 15:00:00'
)
SELECT m.symbol, m.am_high, c.pm_close,
       (c.pm_close - m.am_high) / m.am_high AS intraday_drawdown
FROM morning_high m JOIN close_pm c USING (symbol)
WHERE m.am_high > 0
ORDER BY intraday_drawdown ASC  -- 跌最狠
LIMIT 20;
```

## 数据规模 / 覆盖率 (2026-05-27 ship)

| 指标 | 值 |
|---|---|
| universe size | 3795 (all_no_st) |
| shards 成功 | **3762 (99.13%)** |
| 失败 (post-2025 IPO / 退市) | 33 (`empty` — TDX 无数据) |
| hive 总大小 | **0.98 GB** |
| 行数 | **35,906,513** |
| 平均 bar/股 | ~9543 (~199 trading day × 48 bar) |
| 日期范围 | **2025-06-10 ~ 2026-05-27** (~0.96 yr 总宽; 中位 first_date=2025-07-25 ≈ 0.84 yr) |
| Phase 1 wall | 71.6 min (8 workers × 12 pages) |
| Phase 2 wall | 3.1 s (file copy) |

## 跟现有 daily kline 对比

```sql
-- daily 主表 (baidu_kline.parquet): 7.9M rows × 4625 codes × 2012-09 ~ 2026-05
-- 5m view:                          ~30-40M rows × ~3700 codes × 2025-08 ~ 2026-05 (近 9 月)
SELECT 'daily' AS scale, COUNT(*) AS n_rows, COUNT(DISTINCT code) AS n_codes
FROM kline
UNION ALL
SELECT '5m', COUNT(*), COUNT(DISTINCT symbol) FROM kline_5m;
```

## 不复权警告 — strategy 注意事项

- 5m 是 **TDX 原价 (unadjusted)**,日间跨除权日 close 会跳。
- **盘中 (单日内) 策略**: 直接用 5m 安全。
- **跨日策略**: 必须 join hfq factor 或只用相对量。
- 后续若要 hfq 5m,改写 fetch 把 sina hfq factor merge_asof 进来即可
  (跟 fetch_baidu_kline_v2_akshare.py 同模式)。

## 严格 OOS 协议提醒

- 本数据集 IS/OOS 时间隔离仍按现有 v17 / paper_trade / forward_oos_monitor 的口径。
- 5m 仅 ~9 月数据 → **不足以独立做 60 月 OOS backtest**;只能做盘中策略 IS 研究 + paper trade。
- 待 mootdx 历史窗扩到 2 年 (重跑 `--max-pages 35`),才有 ~250 trading day 做横截面研究,
  但仍远低于 daily 数据的 60 月深度。

## Production 不受影响

`paper_trade_today.py` / `forward_oos_monitor.py` / `daily_check.sh` / `strategy_v*` /
qlib bin / `baidu_kline.parquet` 主表 — **全部不受影响**。5m 是新独立 stack,跟 daily 数据
完全隔离。
