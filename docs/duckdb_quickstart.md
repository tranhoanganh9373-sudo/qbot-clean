# DuckDB Stage 1 — Quickstart

DuckDB embedded SQL on parquet。**主存储仍是 parquet**,DuckDB 只是 query engine。

## 状态

- DuckDB 1.5.3 装在 venv (pyproject.toml 未改)
- `data_cache/db.duckdb` ~1 KB metadata,view 定义而非数据拷贝
- 8 个 view 现已 ok:

| view | rows | cols | source |
|---|---:|---:|---|
| kline | 7,954,537 | 12 | baidu_kline.parquet |
| margin | 752,994 | 7 | csi300_margin_14yr.parquet |
| predictions | 559,676 | 4 | v17_dens_train24_predictions.parquet |
| fund_flow | 393,920 | 17 | fund_flow_csi300.parquet |
| shareholders | 23,341 | 8 | shareholders_csi300.parquet |
| industry_membership | 5,203 | 6 | industry_membership.parquet |
| csi300 | 300 | 3 | csi300_constituents.csv |
| portfolio_log | 32 | 6 | paper_trade_log.csv |

**注: `kline_hive` view 已 drop**(慢 11-740×)。单股快路径用 `dashboard.utils.kline_fast.get_stock_kline()`(direct path-read 0.4-2 ms)。

## 命令

```bash
.venv/bin/python examples/duckdb_init.py        # 重建 views (幂等)
.venv/bin/python examples/duckdb_init.py --list # 列出 + 行数
.venv/bin/python examples/duckdb_init.py --demo # 3 个 demo cross-table queries
```

## Python 用法

```python
import duckdb
con = duckdb.connect("data_cache/db.duckdb")
df = con.execute("SELECT * FROM kline WHERE date = '2026-05-25' LIMIT 10").fetchdf()
```

## 现有 production 不受影响

`paper_trade_today.py` / `daily_check.sh` / `forward_oos_monitor.py` / v17 / qlib bin / strategy_v* 全部仍直接读 parquet。DuckDB 是**额外**的 query 工具,跟现有 production 互不干扰。

## 严格 OOS 协议

DuckDB views 没有 cache,每次 query 读最新 parquet。`predictions.month` 列是时间隔离 ground truth,严格 OOS 天然保留。

## 典型用法

```sql
-- 找出最近 30 天 margin_5d_chg > 30%
SELECT code, date, margin_5d_chg
FROM margin
WHERE margin_5d_chg > 0.30
  AND date >= (SELECT MAX(date) FROM margin) - INTERVAL 30 DAY
ORDER BY margin_5d_chg DESC LIMIT 20;

-- 户数减少股 ∩ 资金流入大股
WITH lo AS (
    SELECT code FROM shareholders
    WHERE announce_date >= '2025-01-01' AND count_change_pct < -0.10
)
SELECT m.code, m.margin_5d_chg
FROM margin m JOIN lo USING (code)
WHERE m.date = (SELECT MAX(date) FROM margin)
ORDER BY m.margin_5d_chg DESC LIMIT 20;
```

## Stage 2 — Hive 分区 (已做, 不推荐默认)

详细 benchmark + 决策见 [`duckdb_stage2_benchmark.md`](./duckdb_stage2_benchmark.md)。

**TL;DR**: 在当前 220 MB / 7.9M rows 量级, single-parquet `kline` view 已经 < 100 ms,
拆 Hive 分区后通过 `kline_hive` view 反而**慢 11~740×** (4690 files 的 enumerate + metadata 开销远超 row-group 谓词下推收益)。

```bash
# build (atomic, 4.6s wall, 幂等)
.venv/bin/python examples/duckdb_hive_build.py
# rebuild
.venv/bin/python examples/duckdb_hive_build.py --force
# verify
.venv/bin/python examples/duckdb_hive_build.py --check
# benchmark Stage 1 vs Stage 2 (A/B/C × 3 runs)
.venv/bin/python examples/duckdb_stage2_benchmark.py
```

### 何时用 `kline` view vs `kline_fast` helper

| 场景 | 推荐 |
|---|---|
| 默认 (production / sidecar / IC / paper_trade) | **`kline`** view (主表) |
| 单股深度分析, 已知 stock code | **`from dashboard.utils.kline_fast import get_stock_kline`** (Hive direct path-read, 0.4-2 ms warm) |
| 跨股 ad-hoc query | `kline` view |
| 全表 COUNT/聚合 | `kline` view |

`kline_hive` view 已 drop (2026-05-26, 慢 11-740×)。Stage 2 数据 `data_cache/baidu_kline_hive/` 跟主表并存, 仅供 `kline_fast.py` 单股快路径用, **swap 主表 不发生**, production 0 改动。

```python
from dashboard.utils.kline_fast import get_stock_kline, list_available_codes
df = get_stock_kline("SH600519")  # 茅台全 history, 2 ms warm
codes = list_available_codes()    # 全 Hive 分区 code list
```

## Stage 3 (未做)

- sidecar / factor IC / paper_trade 内部 pandas 替成 duckdb.sql, 评估 ROI。Stage 2 已证明 view 抽象在小数据集上 **不增加性能**, Stage 3 价值需用别的指标 (代码简洁性 / SQL 可读性 / cross-table JOIN) 来论证。

## Parquet row-group split by year (2026-05-26 ship)

主 `baidu_kline.parquet` 从 8 个 generic row groups (默认 ~1M rows/group) 改成 **15 个 year-aligned row groups** (2012..2026 每年一个 row group)。每 row group 在 footer 写 `date` 列 min/max statistics, DuckDB / pyarrow 谓词下推可跳过非相关 row groups。

Benchmark 改造前后 (3 query × 5 runs median, `kline` view):

| query | before (8 RG) | after (15 RG year-aligned) | 加速 |
|---|---:|---:|---:|
| A 单股最近 60d (`code='600519' AND date>='2026-03-01'`) | 4.65 ms | 5.82 ms | ~1× (noise) |
| B all-market latest day | 53.10 ms | **15.73 ms** | **3.4×** |
| C `COUNT(*)` (footer only) | 0.26 ms | 0.38 ms | ~1× |

B 加速来源: latest-day query 在 (year=2026) row group 上下推过滤, footer stat `date>=2026-01-05` 让其他 14 个 row group 全部跳过。A 单股 60d 已经依赖 row group 内 bloom filter, 无显著差异。C 只读 footer, 不受 row group 拆分影响。

文件大小: 220.8 → 217.0 MB (sort by year+code 改善 snappy dictionary 压缩)。

### 工具

```bash
# 重新做 split (atomic swap, backup -> baidu_kline.parquet.pre_rowgroup.bak)
.venv/bin/python examples/parquet_rowgroup_split.py            # dry-run
.venv/bin/python examples/parquet_rowgroup_split.py --apply    # 实际 swap

# 验证 row group 布局
.venv/bin/python -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('data_cache/baidu_kline.parquet')
print(f'row_groups: {pf.num_row_groups}')
for i in range(pf.num_row_groups):
    rg = pf.metadata.row_group(i)
    s = rg.column(1).statistics  # date col
    print(f'  RG {i}: {rg.num_rows:,} rows  {s.min}..{s.max}')
"
```

### Production 影响

零。`pd.read_parquet('data_cache/baidu_kline.parquet')` 读取语义不变(full materialize), 只是 row group metadata 更细。production paper_trade / forward_oos / portfolio_excel 全 untouched。

## Daily sidecar merge framework (2026-05-26 ship, framework-only)

`examples/merge_daily_into_main.py` — 占位 future infrastructure: 当多个 fetcher 并发往 `data_cache/baidu_kline_daily/*.parquet` 写单日 row 时, 周期性 merge 进主表 + dedupe + atomic swap。

**当前不 wire**: 主 daily fetch 仍直接更新主表 (220 MB rewrite ~1-2s 不算瓶颈)。等真出现 multi-fetcher / 失败容灾痛点再 enable。

```bash
# 当 baidu_kline_daily/ 有 sidecar 时
.venv/bin/python examples/merge_daily_into_main.py            # dry-run
.venv/bin/python examples/merge_daily_into_main.py --apply    # merge + swap
```

## .bak retention (2026-05-26 ship)

`examples/cleanup_old_baks.py` — idempotent retention sweep over `data_cache/*.bak*`:
- `baidu_kline.*.bak` family: 保留最新 2 (回退保险)
- `portfolio.xlsx.bak_*` family: 保留最新 2 (`bak_pre_positions_rebuild_*` 和 `bak_pre_trade` 永不删, 真实买入重建依赖)
- 其他单文件 .bak (margin / state / predictions): 保留最新 1

```bash
.venv/bin/python examples/cleanup_old_baks.py            # dry-run (默认)
.venv/bin/python examples/cleanup_old_baks.py -v         # 详细
.venv/bin/python examples/cleanup_old_baks.py --apply    # 实际删除
```

首次 sweep 释放 445.5 MB (5 个旧 baidu_kline / predictions / xlsx bak)。脚本 idempotent, 重跑 0 删。不 wire 进 daily_check.sh, 手动 / 临时清理用。

## 风险

- view 是 lazy reference,parquet 文件被 move/rename 后 view 失效 → 重跑 `duckdb_init.py`
- `data_cache/db.duckdb` 不要 commit (添加 .gitignore,view 可重建)
- duckdb 不读 .xlsx,`portfolio.xlsx` 仍 `pd.read_excel`
