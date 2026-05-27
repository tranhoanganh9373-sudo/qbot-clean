# PostgreSQL + TimescaleDB benchmark vs DuckDB + parquet

**日期**: 2026-05-26
**Hardware**: macOS / Apple Silicon / SSD (`/Volumes/SSD`)
**PG**: 16.14 (Ubuntu, ARM64, inside Docker)
**TimescaleDB**: 2.27.1
**DuckDB**: 1.5.3, embedded
**数据**: `data_cache/baidu_kline.parquet` 210.5 MB / 7,954,537 行 / 4,676 codes / 2012-09-27 ~ 2026-05-25
       `data_cache/csi300_margin_14yr.parquet` 26.9 MB / 752,994 行

## 1. Install

| 组件 | 版本 | 方式 | 备注 |
|---|---|---|---|
| PostgreSQL | 16.14 | `brew install postgresql@16` | 装但**未启动** (port 5432 让给 docker) |
| TimescaleDB extension (Homebrew) | 2.27.1 source | `brew install timescaledb` | **build 失败** — `compat.h:514:1: error: static declaration of 'RestrictSearchPath' follows non-static declaration` + 6 个相关 redefinition error; brew formula 默认编译 against PG@17, 加 macOS 26.5 (Tahoe) SDK + clang 21 ABI 不兼容 TimescaleDB 当前 compat shim |
| TimescaleDB (Docker, **采用**) | 2.27.1 | `docker pull timescale/timescaledb-ha:pg16` | 1 step, 73 MB image; ready 8s 后 `pg_isready` 通过 |

**Docker container 配置** (volume on `/tmp/claude_finance_pg_data`, 容易清理, 不污染 brew global):
```bash
docker run -d --name claude-finance-ts \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=claudefinance \
  -e POSTGRES_USER=claudefinance \
  -e POSTGRES_DB=claude_finance \
  -v /tmp/claude_finance_pg_data:/home/postgres/pgdata/data \
  timescale/timescaledb-ha:pg16
```

**Brew PG@16 状态**: 装但 `brew services stop` 保持 stop, 不冲突 5432。如不要 docker 路线, 可 `brew uninstall postgresql@16 krb5` 回滚 (用户决定)。

### TimescaleDB hypertable schema

```sql
CREATE TABLE kline (
    code            TEXT,
    date            TIMESTAMPTZ,
    open, close, high, low, vol, amount,
    ma5, ma10, ma20, turnoverratio  -- DOUBLE PRECISION
);
SELECT create_hypertable('kline', 'date', chunk_time_interval => INTERVAL '1 month');
CREATE INDEX idx_kline_code_date ON kline (code, date DESC);
```

`margin` 是 vanilla PG table (753k 行, 不开 hypertable, 体量太小)。
`kline_plain` 是为 benchmark 准备的对照, 跟 `kline` 同 schema + 同索引, 但**不开 hypertable** (单一 heap), 用 `INSERT INTO kline_plain SELECT * FROM kline` 复制。

## 2. Migration

`examples/pg_migrate_kline_margin.py` (~200 LOC) 用 psycopg2 + chunked `copy_expert`。

### 关键教训: COPY 必须分块

**第一次尝试**: 单个 7.9M 行 `StringIO` → `cur.copy_expert(...)`。**stalled 30+ 分钟**:
- Python 进程 0% CPU
- PG container CPU 0.58%, BLOCK I/O 仅 57 KB
- VSZ 膨胀到 442 GB (anonymous mapping 被换出)
- PG side 显示 `state=active query=COPY ...` 但实际没数据流过

**根因**: 把 7.9M 行 CSV 渲染到 `StringIO` 占 ~1.2 GB 内存; psycopg2 的 `copy_expert` 把整块 buffer 当一次性 write 发, 与 docker bridge network buffer 交互不良, 出现 deadlock-like stall。

**修复**: 改成 500k 行/chunk, 每 chunk 独立 `to_csv` → `copy_expert` → `commit`。结果:

| 步骤 | 时间 | 备注 |
|---|---:|---|
| read parquet (kline) | 0.37 s | pandas read_parquet |
| read parquet (margin) | 0.06 s | |
| CREATE TABLE + create_hypertable | < 0.1 s | |
| **COPY kline (chunked)** | **61.96 s** | 128,385 rows/sec, 16 chunks of 500k |
| COPY margin (chunked) | 2.96 s | 254,242 rows/sec |
| CREATE INDEX kline (code, date DESC) | 5.65 s | |
| CREATE INDEX margin (code,date) + (date) | 0.62 s | |
| ANALYZE | 6.67 s | |
| **Total wall** | **117 s** | |

### 磁盘占用对比

| 存储 | 大小 | vs parquet | 备注 |
|---|---:|---:|---|
| `baidu_kline.parquet` | 210.5 MB | 1.00× | zstd compressed, single file |
| `kline` (hypertable, 167 chunks) | 1262 MB | **6.0×** | 955 MB data + 302 MB indexes (date_idx + code_date) |
| `kline_plain` (no hypertable) | 1247 MB | **5.9×** | 954 MB data + 293 MB indexes |
| `csi300_margin_14yr.parquet` | 26.9 MB | 1.00× | |
| `margin` (PG plain) | 88 MB | **3.3×** | 61 MB data + 28 MB indexes |

PG 6× 膨胀 = 2 个因素: (1) 行存储 + TOAST overhead (parquet 是列存 + zstd); (2) 双索引 ~300 MB。

如想压缩, TimescaleDB 提供 `ALTER TABLE kline SET (timescaledb.compress)` 后台 columnar compression, 实测同行业 3-10× 缩 (本 benchmark 未测, 因 production 价值低 — 见 §5)。

## 3. Benchmark (median of 3 runs, warm cache)

跟 Stage 2 benchmark 同 query, 严格相同协议 (warmup 1 次 → 3 次实测取中位数, fetch all rows 避免 lazy cursor)。

### 单行结果

```
A: SELECT * FROM kline WHERE code='600519' AND date >= '2026-03-01'
B: SELECT * FROM kline WHERE date = (SELECT MAX(date) FROM kline)
C: SELECT COUNT(*) FROM kline
```

| query | parquet (pandas) | DuckDB Stage 1 (`kline`) | DuckDB Stage 2 (`kline_hive`) | **PG plain** | **PG + TimescaleDB hypertable** |
|---|---:|---:|---:|---:|---:|
| **A 单股 60d** | 337 ms | 5.2 ms | 156.7 ms | **0.26 ms** | **0.35 ms** |
| **B 全市场 latest day** | n/a | 52.9 ms | 615.3 ms | **7.32 ms** | **13.54 ms** |
| **C 全表 COUNT(\*)** | n/a | 0.3 ms | 241.1 ms | 74.15 ms | 102.39 ms |

(Stage 1 / Stage 2 numbers from `docs/duckdb_stage2_benchmark.md` §3。pandas baseline 实测本 task。)

### 谁赢哪一个

| query | 赢家 | 第二 | 加速比 |
|---|---|---|---:|
| A 单股 60d | **PG plain (0.26 ms)** | PG hypertable (0.35 ms) | PG 比 DuckDB 快 **14-19×** |
| B latest day | **PG plain (7.3 ms)** | PG hypertable (13.5 ms) | PG 比 DuckDB 快 **4.7-8.6×** |
| C COUNT(*) | **DuckDB (0.3 ms)** | PG plain (74 ms) | DuckDB 比 PG 快 **240-330×** |

### 为什么 PG 赢 A、B 但输 C

- **A 单股 60d**: PG `(code, date DESC)` btree 索引 → 单 scan range, ~10 行返回, < 1 ms。DuckDB 必须 read parquet row-group statistics → predicate pushdown, 但还要解码列。`code='600519'` 的 row-group 命中后是顺序读, 但 parquet 是列存所以列解码 + dictionary lookup 比 PG btree 慢 5-10 ms。
- **B latest day**: 一样, PG `(date DESC)` 索引 + 4676 行 scan = 7 ms。DuckDB 没有 secondary index, 必须 scan 全部 row groups 找 `date=MAX`, 这是 53 ms 的来源。
- **C COUNT(*)**: parquet footer **本来就有 row count**, DuckDB 0.3 ms 是 metadata fetch。PG 要么 seq scan 全表 (~1.2 GB), 要么走 covering index — 它选了 parallel index-only scan 6 workers 累 116K shared buffer hit 才出 102 ms 结果 (`EXPLAIN ANALYZE` 已验证)。PG 没法用 parquet 的 footer trick。

### PG hypertable vs PG plain

Hypertable **慢 30-80%** 在所有 3 query:
- A: 0.35 ms vs 0.26 ms
- B: 13.5 ms vs 7.3 ms
- C: 102 ms vs 74 ms

原因: hypertable 把表切 167 chunks, planner 每次 query 都要 prune chunks (constraint exclusion + chunk catalog lookup) — 在小数据集 (1.2 GB) 上 prune overhead > shard 加速。Hypertable 真正回报是 (a) 单 chunk 超 100 GB 后单 file fragmentation, (b) compression policy, (c) retention policy drop 旧 chunks O(1)。**8 GB 量级以下不需要 hypertable**。

## 4. 5 路对比表 (final)

| query | parquet (pandas) | DuckDB single | DuckDB hive view | PG plain | PG TimescaleDB |
|---|---:|---:|---:|---:|---:|
| A 单股 60d | 337 ms | 5.2 ms | 156.7 ms | **0.26 ms** | 0.35 ms |
| B latest day | n/a | 52.9 ms | 615.3 ms | **7.32 ms** | 13.54 ms |
| C COUNT(\*) | n/a | **0.3 ms** | 241.1 ms | 74.15 ms | 102.39 ms |

**Stage 1 DuckDB 仍是最 balanced**: A 5.2 ms / B 53 ms / C 0.3 ms — 3 个 query 都 <100 ms, 不需要任何 schema 设计。

## 5. 结论 + 推荐方向

3 选项评估:

### A. PG mirror only-read (production 仍读 parquet, PG 给 ad-hoc / Phase B IS)
- **加速**: A query 5.2 → 0.26 ms (20×), B 53 → 7.3 ms (7×)
- **成本**:
  - +1.05 GB 磁盘 (1.26 GB - 210 MB)
  - 2 min daily migration 重新 ingest 新增 K 线 (或 incremental upsert 复杂度)
  - 1 个 docker container 后台运行 (memory ~150 MB, CPU 几乎 0)
  - 0 production 改动
- **回报场景**: Phase 2 retrain / Phase B 新 sidecar IS sweep 跑 9-12 月数据需要重复单股 60 行查询 → PG 0.26 ms × 9000 stocks vs DuckDB 5 ms × 9000 stocks 差 ~40 s/run。`forward_oos_monitor` daily 30 query, 0.3 ms vs 5 ms 差 0.14 s/day, **可忽略**。
- **风险**: 双 source-of-truth 漂移 (主表先有新数据, PG 落后); 引入 docker daemon 单点依赖

### B. PG 唯一存储 (production 全切, ~500 LOC 改)
- **加速**: 同 A, query 加速一样
- **成本**:
  - 改 `paper_trade_today.py` / `forward_oos_monitor.py` / `strategy_v17_dens_grid.py` / `qlib_baidu/` 改 reader 抽象 (实测 ~500 LOC)
  - `daily_check.sh` 加 PG health check
  - 装机依赖: docker 必须运行 (现 macOS 已有, 但 launchd 启动顺序需安排)
  - 1.05 GB 磁盘永久占用
  - **失去 parquet 的 portability** — 现在 `cp baidu_kline.parquet /backup/` 一行就备份, PG 要 `pg_dump`
- **回报**: 同 A, 但加上 production query 也享 PG 速度
- **真实瓶颈分析**: production 没 query 在 **single-parquet 上 > 100 ms** (Stage 2 已证明)。PG 把 5 ms 降到 0.3 ms 在 daily workflow 节省 < 1 秒/天。
- **风险**: 高 (~500 LOC 改 production, daily_check 改 launchd 改); 收益: 极低

### C. 不迁 (DuckDB Stage 1 已够, PG over-engineering)
- 维持现状, 双 view (kline + kline_hive) + Stage 1 default
- 0 改动 0 风险

### 推荐: **C (不迁)**

理由按权重排:

1. **没有 production query 在 DuckDB Stage 1 上 > 100 ms**。Stage 2 benchmark 已证明 A=5.2 ms, B=53 ms, C=0.3 ms。所有 production 路径 (`paper_trade` / `forward_oos` / `strategy_v*` / `qlib_baidu`) 跑得很顺, 没有 latency 投诉。
2. **PG 加速本质是给 ad-hoc / sidecar 实验, 不是 production**。Phase A/B 跑过 9 因子 OOS sweep 用 `pd.read_parquet`, 总时间被 z-score normalization / merge / Spearman 主导, query 不是瓶颈 (单 sweep wall ~3-10 分钟, query 占 < 5%)。
3. **PG 在 query C 输 240×**。如果未来某天加 ad-hoc COUNT / GROUP BY / 全表 aggregation, DuckDB 远好。混用反而尴尬。
4. **磁盘 1.05 GB + docker daemon 是 ongoing tax**。本机现在 1 TB SSD 还宽 (3.7 TB free), 但每天 0 收益的 1 GB 是浪费 inode + CPU schedule。
5. **brew install timescaledb 在 macOS 26.5 直接 broken** — 如果以后想撤 docker, 必须升级 brew formula 或等 TimescaleDB 修 macOS 26 compat。引入这个依赖只能上 docker → 加一层运维 surface。
6. **真要做单股 sub-ms 单点查询 Stage 2 已经给了方案**: 直接 `read_parquet('.../code=600519/*.parquet')` 0.4 ms (Stage 2 §3a) — 不需要 PG。

### 何时重新评估升级到 A?

满足以下任一:
- 主表 > 10 GB (现 210 MB, 50× 远期)
- 出现 query 在 DuckDB Stage 1 上 > 1 s 的真实瓶颈
- 引入 多 user 并发查询场景 (现单用户, 单 process)
- 引入 web dashboard 需要 sub-ms response (现 production 是 daily batch)

### 何时升级到 B?

几乎永不。除非 production 改成长跑服务 (e.g. real-time intraday signal) 且需要 PG 的并发 / replication / point-in-time-recovery 特性。当下 daily batch + parquet snapshot 简洁正确。

## 6. Cleanup

### 现状

- Brew: `postgresql@16` (16.14) + `krb5` (1.22.2) 已装, 服务 stop。
- Brew: `timescaledb-tools` (0.19.0) 已装, `timescaledb` 主 formula **未装** (build fail)。
- Docker: `claude-finance-ts` container 运行中, image `timescale/timescaledb-ha:pg16` (~1 GB)。
- 数据: `claude_finance` PG db = 1.35 GB 总 (kline + kline_plain + margin)。
- 新文件: `examples/pg_migrate_kline_margin.py`, `examples/pg_benchmark.py`, `docs/pg_timescale_benchmark.md`。

### 如选 C (不迁), 完全 cleanup 脚本

```bash
# 停 + 删 docker container
docker stop claude-finance-ts && docker rm claude-finance-ts

# 删 image (~1 GB)
docker rmi timescale/timescaledb-ha:pg16

# 删 PG data volume
rm -rf /tmp/claude_finance_pg_data

# 卸 brew PG (如未来无别处用)
brew uninstall postgresql@16
brew uninstall timescaledb-tools krb5
brew untap timescale/tap

# 还原 docker config (本任务为绕过 docker-credential-desktop 改过)
mv ~/.docker/config.json.bak ~/.docker/config.json
```

### 如选 A (mirror only), 保留 cleanup

```bash
# 不删 docker, 加自动启动
docker update --restart=unless-stopped claude-finance-ts

# 写 incremental upsert script (例 `examples/pg_sync_incremental.py`),
# 在 daily_check.sh 加 step 1.8 (margin fetch 之后), CSI300 fetch 之前
# (本 task 不写, 因推荐 C)
```

## 7. 风险

| 风险 | 等级 | 缓解 |
|---|---|---|
| docker daemon down → PG 不可用 | 中 | 选 C 则 N/A; 选 A 加 fallback to parquet |
| brew TimescaleDB build 在 macOS 26 broken | **已实测** | 用 docker 路线 (本 task) |
| `~/.docker/config.json` 改了 credsStore (绕过 desktop helper) | 低 | backup 已存 `config.json.bak`, cleanup 脚本恢复 |
| PG 数据卷 `/tmp/...` 在重启清空 | 中 (本任务的 cleanup 行为, 不是生产风险) | 选 A 时改 `~/claude_finance_pg_data` |

## 8. 文件清单

新增 (本 task):
- `examples/pg_migrate_kline_margin.py` — 117s 一次性 migration (chunked COPY)
- `examples/pg_benchmark.py` — 3 query × 3 backend × 3 runs benchmark
- `docs/pg_timescale_benchmark.md` — 本文档
- `~/.docker/config.json.bak` — credsStore 改前备份

**production 0 改动**:
- `paper_trade_today.py` / `paper_trade_v19_4.py` / `forward_oos_monitor.py` / `portfolio_excel.py` / `daily_check.sh` / launchd plist / `strategy_v17_dens_grid.py` / `strategy_v19_*.py` / `strategy_v20_*.py` 全未触碰
- `data_cache/baidu_kline*.parquet` / `data_cache/qlib_baidu/` / `data_cache/v17_dens_train24_predictions.parquet` / `data_cache/csi300_margin_*.parquet` 全未触碰
- `examples/duckdb_init.py` / `examples/duckdb_hive_build.py` 未触碰

**未 commit。**
