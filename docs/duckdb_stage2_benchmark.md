# DuckDB Stage 2 — Hive 分区 benchmark + 决策

**日期**: 2026-05-26
**Hardware**: macOS / Apple Silicon / SSD (`/Volumes/SSD`)
**DuckDB**: 1.5.3, embedded
**数据**: `data_cache/baidu_kline.parquet` 220.8 MB, 7,954,537 行, 4676 codes, 2012-09-27 ~ 2026-05-25

## 1. 分区方案决策

试 **per-stock** (`baidu_kline_hive/code=XXXXXX/*.parquet`):

| 指标 | 值 |
|---|---:|
| 分区数 (`code=*` 目录) | 4676 |
| parquet 文件数 (DuckDB 自动按 row-group 切) | 4690 |
| 总大小 | 373.8 MB |
| 与主表 size ratio | 1.69× (220.8 → 373.8 MB) |
| 平均文件大小 | ~80 KB |
| Build wall time | **4.6s** (DuckDB native `COPY ... PARTITION_BY (code)`) |

Size 膨胀 1.69× 的原因: 每文件单独 parquet header + footer + zstd 字典需要重新构建, small-file overhead 显著。

per-year / per-stock+per-year 候选未实测, 理由见 §4。

## 2. macOS inode 检查

```
df -i /Volumes/SSD
Filesystem   ifree         %iused
/dev/disk7s1 39,815,048,120 0%
```

4690 inodes 占新增 ifree 比 < 0.0001%, **完全无 inode 压力**。

## 3. Benchmark 结果 (3 query × 2 view × 3 runs, median)

| query | Stage 1 (`kline`) | Stage 2 (`kline_hive`) | 加速比 | 备注 |
|---|---:|---:|---:|---|
| A — 单股最近 60 天 (`code='600519' AND date >= '2026-03-01'`) | **5.2 ms** | 156.7 ms | **0.03×** (慢 30×) | hive view 慢得多 |
| B — 全市场最新一天 (`date = MAX(date)`) | **52.9 ms** | 615.3 ms | **0.09×** (慢 11×) | hive view 慢得多 |
| C — 全表 `COUNT(*)` | **0.3 ms** | 241.1 ms | **0.00×** (慢 740×) | stage1 用 parquet metadata, hive 必须 scan 4690 footer |

第 2 次重测全部 ±5% reproducible。

### 3a. 决定性细节 — Hive 直接路径访问

```python
# 走 hive view (glob 4690 files)
"SELECT COUNT(*) FROM kline_hive WHERE code='600519' AND date>='2026-03-01'"   # 175 ms

# 直接读分区 (绕过 hive view, 不用 hive_partitioning)
"SELECT COUNT(*) FROM read_parquet('.../code=600519/*.parquet') WHERE date>='2026-03-01'"  # 0.4 ms
```

**Hive 分区 layout 的 query-engine view 比 single-parquet view 慢, 但直接 path-read 单分区是全场最快 (0.4 ms vs single-parquet 2.6 ms)**。Stage 1 单 parquet 已经利用 row-group 统计完成谓词下推, DuckDB 不需要扫描全部数据; Stage 2 hive view 的开销来自 enumerate 4690 files + 每文件读 footer。

## 4. 为何不试 per-year / per-stock+per-year

- **per-year (~15 files)**: query A (单股 60 天) 仍需读完整 1 year × 4676 codes 才过滤, 本质退化为单 parquet, 只在 query B (全市场单日) 有 marginal 改善。预期不会超过 single-parquet。
- **per-stock+per-year (~65,000 files)**: 文件数 14× per-stock, small-file overhead 更严重。本基准已显示 per-stock 4690 files 在 view 模式下慢 30~740×, 加 14× 文件只会更糟。

结论: 在 **8 GB 量级以下** 的 baidu_kline, Stage 1 single-parquet + row-group statistics 是最优解; Hive 分区在 **view-glob 模式下反而是劣解**, 只有当调用方愿意改成"先算 partition path 再 read_parquet"才有价值, 而那已经超出 DuckDB-as-query-engine 的 ergonomics 边界。

## 5. 推荐: 保留双表 (默认不替换)

**保留主表 `baidu_kline.parquet` + `kline` view 作 default**, Hive 目录 + `kline_hive` view **保留但不推**。

理由:
1. **典型 query 在 Stage 1 已经 < 100 ms**, Stage 2 view 反而拖慢 11~740×。
2. **production 全部走主表** (`paper_trade_today.py` / `forward_oos_monitor.py` / `strategy_v*.py` / `qlib_baidu/` 都直接 pd.read_parquet), 不动它就是 0 风险。
3. **Hive 仅在一个场景胜出**: 调用方明确知道 stock code, 愿意走 `read_parquet('.../code=XXXXXX/*.parquet')` 直接路径访问 → 0.4 ms 单股查询 (比主表 2.6 ms 快 6×)。这适合后续 streaming sidecar / 单股深度分析, 但与 view 抽象正交。
4. **不加 per-year sidecar** — 无证据收益, 且会引入第 3 个相同数据 view 增加 cognitive overhead。

### 何时 swap?

仅当满足以下任一才重新评估:
- 主表 > 2 GB (当前 220 MB, 10× 远期)
- 出现 query 在 single-parquet 上 > 1s 的真实瓶颈
- production 由 pd.read_parquet 迁移到 duckdb.sql 且证明 hive 加速 ≥ 10×

## 6. 风险

| 风险 | 等级 | 缓解 |
|---|---|---|
| Hive 目录占 373 MB (+ 主表 220 MB = 594 MB 同数据 2 份) | 低 | 用户已配 8 TB SSD; .gitignore 已忽略 data_cache |
| 主表与 hive 不同步 (主表后续合并新股, hive 不动) | 中 | `duckdb_hive_build.py --force` 重建; build wall 4.6s, 可加进 `daily_check.sh` (本任务不动) |
| `kline_hive` view 误用 → query 慢 11× | 中 | `docs/duckdb_quickstart.md` 注明默认用 `kline` |
| DuckDB upgrade 后 PARTITION_BY 行为改变 | 低 | rebuild 即可, 无业务依赖 |

## 7. 文件清单

- `examples/duckdb_hive_build.py` — atomic + idempotent build (4.6s wall)
- `examples/duckdb_stage2_benchmark.py` — A/B/C × kline/kline_hive × 3 runs
- `data_cache/baidu_kline_hive/code=XXXXXX/data_*.parquet` — 4676 partitions / 4690 files / 373.8 MB
- `examples/duckdb_init.py` — 新增 `kline_hive` view (parquet_hive kind)
- 主表 `data_cache/baidu_kline.parquet` 未触碰

production (paper_trade / forward_oos / portfolio_excel / strategy_v* / daily_check.sh / launchd) **0 改动**。
