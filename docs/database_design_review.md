# 行情数据库选型调研

> Read-only 浅读 `references/` 下 7 个量化项目 (`zipline-reloaded` / `zipline` / `pyfolio-reloaded`
> / `alphalens-reloaded` / `empyrical-reloaded` / `OpenBBTerminal` / `Lean`) 的数据存储方式,
> 对比当前项目 2.3 GB `data_cache/` 现状, 思考层面评估 6 个候选数据库, 给出选型建议 +
> 3 阶段迁移路径. **不引入新依赖, 不修改 references/, 不改 production, 不跑 backtest**.
>
> Date: 2026-05-26. Reviewer: claude-opus-4.7 (1M ctx).

---

## 1. 7 reference 项目数据存储方式对比

直接读 `pyproject.toml` / `setup.py` / `etc/requirements_locked.txt` + `data/` 目录关键源码 +
顶层 `README.md`. 注意区分 "库自己写存储 (zipline / Lean)" vs "库只是 pandas in / pandas out
(pyfolio / alphalens / empyrical)" vs "库做 API stream (OpenBB)".

| # | 项目 | 主存储格式 | 数据库? | Adapter 模式 | Daily 增量 |
|---|------|----------|--------|-------------|-----------|
| 1 | **zipline-reloaded** | **bcolz** (daily/minute OHLCV) + **HDF5** (新 backend) + **SQLite** (adjustments/assets) | SQLite | Bundle (`bcolz_daily_bars.py` / `hdf5_daily_bars.py` / `adjustments.py`) | `zipline ingest` 整库重抓; bundle 注册化, csvdir / quandl 等 |
| 2 | **zipline** (原版) | bcolz 1.2.1 + h5py 2.7.1 + SQLAlchemy 1.3 | SQLite | 同 zipline-reloaded, 但锁定老版本 (`requirements_locked.txt` 1.2.1 / 1.3.11) | 同上 |
| 3 | **pyfolio-reloaded** | **pandas in-memory** (DataFrame / Series) | 无 | 纯 analytics, 不管 storage. 上游传 returns Series 进来 | 无 (consumer 库) |
| 4 | **alphalens-reloaded** | **pandas in-memory** (MultiIndex factor / prices) | 无 | 同 pyfolio | 无 (consumer 库) |
| 5 | **empyrical-reloaded** | **pandas in-memory** + **peewee ORM** (lite SQLite) 给 yfinance benchmark cache | SQLite (轻量) | `utils.py` 通过 pandas_datareader / yfinance 拉 benchmark, 用 peewee 落缓存 | benchmark returns 按 symbol cache |
| 6 | **OpenBBTerminal** | **JSON over HTTP** (FastAPI / Pydantic models); provider 端可选 **parquet / csv** cache | 无 (核心); provider 自治 | Provider 模式 (`oecd/helpers.py` 用 `to_parquet`, `bls/helpers.py` 用 `to_csv`); 核心走 `OBBject.results` JSON | API 调用即拉, provider 内部缓存 |
| 7 | **Lean** (QuantConnect) | **zip + CSV/JSON 扁平文件** (`/data/securityType/marketName/resolution/ticker.zip`) | 无 | 自家 `Reader()` factory; 哲学 "human-readable, DB-agnostic" | 整 zip 下发, 按目录树 |

**核心观察**:

- **没有一个项目用 PostgreSQL / ClickHouse / DuckDB / TimescaleDB 做主存储**. Zipline 用
  SQLite 只装 adjustments + assets metadata (低频小表); 主行情走 bcolz/HDF5 列式二进制文件.
- **Lean 直接断言 "flat files on disk" 是哲学选择** (README: "open, human-readable data format -
  independent of any specific database"). 这个观点很重要 — 它意味着量化界并不把"上数据库"当默认.
- **三个 analytics 库 (pyfolio / alphalens / empyrical) 完全不管 storage**, 因为它们是 consumer
  layer — 调用方传 pandas 进来. 这条已经说明: **改 storage 不会影响 analytics tooling**.
- **OpenBB 的设计哲学是 API stream + 让 provider 决定缓存格式**. 它根本不假设有"统一数据库",
  因为不同 provider (OECD, BLS, yfinance, FRED, ...) 数据形状差太多, 强制统一 schema 反而是负担.
- 唯一用 ORM (peewee) 的是 empyrical 的 yfinance benchmark cache, 用例非常窄, 不能推广到主行情.

---

## 2. 当前项目数据现状

### 2.1 文件清单 (按容量)

| 路径 | 类型 | 容量 | 行数/股票数 | 说明 |
|------|------|------|-----------|------|
| `data_cache/baidu_kline.parquet` | parquet (单文件) | **215 MB** | 4625 codes × 7.9M rows × 12 cols | **主行情**, 2012-09 ~ 2026-05 hfq |
| `data_cache/qlib_baidu/` | qlib **二进制 bin** + txt 索引 | **211 MB** | 4610 features 目录 + 5 instruments (csi300/csi500/top1500_no_st/all_no_st/all) | **backtest 唯一格式**, Alpha158 走这条 |
| `data_cache/fund_flow/` | per-stock parquet × 219 | **75 MB** | 219 files | 资金流分钟级 |
| `data_cache/long_history.parquet` | parquet | 37 MB | — | 备份长历史 |
| `data_cache/csi300_margin_14yr.parquet` | parquet | 27 MB | 300 codes × 753k rows | 融资融券 14 年 |
| `data_cache/csi300_margin_full.parquet` | parquet | 27 MB | — | margin full snapshot |
| `data_cache/margin_180_backfill.parquet` | parquet | 17 MB | — | margin backfill |
| `data_cache/*_predictions.parquet` × 15+ | parquet | 5-11 MB 每个 | — | v17~v20 各 sidecar predictions |
| `data_cache/fundamentals/` | per-stock parquet × ~300 | 5.5 MB | 275 codes | EM 财务指标 IS |
| `data_cache/shareholders/per_stock/` | per-stock parquet × 290 | 4.2 MB | 290 files | 股东户数 |
| `data_cache/dragon_tiger/` | parquet | 2.5 MB | — | 龙虎榜 events |
| `data_cache/unlock/` | parquet | 1.2 MB | — | 解禁 events |
| `data_cache/industry/` | parquet | 84 KB | — | 行业归属 |
| `data_cache/portfolio.xlsx` | xlsx | 12 KB | — | 组合状态 |
| `data_cache/paper_trade_log*.csv` | csv | < 100 KB | daily append | 主 + v19_4 shadow + v3 backup |
| `data_cache/*.bak` × 5 | parquet/csv backup | ~250 MB 累计 | — | 每次大改 swap 留 .bak |
| 失败/审计 csv (`*_failed*.csv`, `corrupt_codes*.txt`, `forward_oos_alerts.csv`, ...) | csv/txt | < 1 MB | — | 审计 trail |

- **总容量**: `data_cache/` = 2.3 GB, 项目根 = 5.3 GB.
- **总文件数**: `data_cache/` 顶层 85 个直接文件 + 多个子目录, qlib_baidu 4610 个 feature dirs.

### 2.2 当前读写模式 (production 真实场景)

`grep -rE "to_parquet|read_parquet" src/ examples/` 得到:
- **src/** 38 处 (核心 fetcher / cache 模块, 11 个 lib file)
- **examples/** 220 处 (paper_trade / forward_oos / strategy backtest / margin fetch / ...)

`paper_trade_today.py` (production 主入口) 实际读:
```
pd.read_csv(UNIVERSE_PATH, dtype={"code": str})       # csv 一次
pd.read_parquet(INDEX_PARQUET)                         # parquet 一次
pd.read_csv(LOG_PATH) → out.to_csv(LOG_PATH)          # csv append
pd.read_parquet(MARGIN_DAILY_PARQUET, ...)            # parquet, 列裁剪
pd.read_parquet(MARGIN_PARQUET, ...)                   # parquet 14yr cache
pd.read_parquet(KLINE_PATH, columns=[...])             # parquet, 部分列
```

`strategy_v17_dens_grid.py` 用的是 **qlib `D.features()` API**, 走 `data_cache/qlib_baidu/`
二进制 bin, 跟 parquet 路径完全独立.

### 2.3 真实痛点排序 (decreasing severity)

1. **`baidu_kline.parquet` 单文件 215 MB 全量重写** — 任何 corrupt 修复 / Phase 1C 合并都得
   `.bak` 一份再覆盖. memory 里 `baidu_kline.pre_688_merge.bak` / `pre_phase1c_swap.bak` /
   `pre_v3_merge.bak` 三个 backup 累计 ~600 MB 占用. 一次写 215 MB IO ~3-5s, 不致命但不优雅.
2. **subset 读 OOM 风险** — 单文件 7.9M rows, 即使 `columns=[...]` 列裁剪后仍读全部行 (parquet
   row group 粒度 ~1M), 想取 "2020-01-01 ~ 2020-12-31 ∩ CSI300 296 stocks" 必须全量 load
   再 filter, 内存 ~600 MB peak.
3. **per-stock parquet 文件过多** — `fund_flow/` 219, `shareholders/per_stock/` 290,
   `fundamentals/` ~300, `qlib_baidu/features/` 4610 sub-dirs. macOS Spotlight / Time Machine
   扫描慢, `ls` 偶尔有延迟.
4. **没有 query 语言** — 想答 "2025 年涨幅超 50% 的股 + 同时 unlock_pct_next_60 > 10%" 必须写
   多步 pandas join. 用 SQL 一行的活, 现在要 30 行 python.
5. **并发写没保护** — `fetch_margin_today.py` 跟 `paper_trade_today.py` 同时跑时, 都可能写
   `csi300_margin_daily.parquet`. parquet 没行锁, 靠 launchd 时序错开 + alarm 5min wall-time.
6. **backup 是 file-level copy** — 一次 swap 一次 .bak, 没有 incremental snapshot, 没有 rollback
   到任意时间点. 想回到"昨天 22:00 的 kline 表"做不到.

**关键判断**: 当前痛点 **80% 是单大文件 swap 笨重 + subset 读慢**, **20% 是查询便利性**. 不是
performance 极限 (parquet+pandas 单机日级足够), 不是 concurrent writes (single-user), 不是
backup-restore (file copy 就够). 这意味着 **不需要服务型数据库 (Postgres/ClickHouse), embedded
列存 (DuckDB) 就能 cover 痛点**.

---

## 3. 候选数据库评估

为时间序列 + 行情数据评估 6 个常见方案. 不联网 / 不装新依赖, 只做思考层面对比.

| 候选 | 部署 | 查询性能 (kline) | pandas/qlib 兼容 | 备份/迁移 | 中文文档 | 单 user 适配 |
|------|------|----------------|-----------------|---------|---------|------------|
| **DuckDB** (embedded SQL on parquet) | **无服务, pip 一行** | **极强**: 直接 `SELECT ... FROM 'baidu_kline.parquet' WHERE ...` 列裁剪 + 谓词下推 | pandas: `con.execute(...).df()` 原生; qlib bin 不读 (要走 qlib API) | **极易**: parquet 文件本身就是存储, 复制即备份 | 良好 (社区中文 blog 多) | **极适配**, 即装即用 |
| **SQLite + parquet sidecar** | 内置 (zipline 同款) | 一般: SQLite 装 metadata, kline 仍读 parquet | pandas: `pd.read_sql_query(...)`; ORM 可选 (peewee/SQLAlchemy) | 文件级复制 | 极丰富 | 适配, 但 kline 大表存 SQLite 反而慢 |
| **PostgreSQL + TimescaleDB** | **server, 装 daemon, 配 user/pwd** | 强 (hyper-table 分区), 但需 schema design | pandas: `pd.read_sql_query` 走 psycopg2; 中等开销 | `pg_dump` + WAL; 需运维 | 丰富 (但 Timescale 偏英文) | **不适配**: single-user 上 server overkill |
| **ClickHouse** | server (轻量但仍要 daemon) | **极强** (column store, MergeTree, vectorized) | pandas: 通过 clickhouse-driver / arrow flight; 中等 | parts dir 复制, replication 复杂 | 中等 (cn.clickhouse.com 有, 不如 PG 普及) | **不适配**: 设计给亿级 row 多 user |
| **InfluxDB** | server (v2 用 Flux DSL) | 强 (time-first), 但 schema 是 tag/field 模型 | pandas 转换有摩擦, 不直接 read_parquet | snapshot API | 一般 (社区中文资料偏少) | **不适配**: 主要给 IoT / metrics, A 股 fundamentals 多 dim 不友好 |
| **HDF5** (h5py / pytables) | **无服务, lib only** | 中等: 列读 OK, 但 query 必须手写 indexer | pandas: `pd.read_hdf` 原生; zipline-reloaded 用这个 | 文件级复制, 但 HDF5 corrupt 修复麻烦 | 一般 (HDF5 整体在 quant 圈逐渐被 parquet 替代) | 适配, 但相比 DuckDB 没优势 (parquet 更新, 工具链更多) |

**额外加分项**:

- **DuckDB** 还支持: 读 csv 直接 `SELECT * FROM 'file.csv'` (省 pandas wrapper);
  view materialized; 多 parquet UNION (`read_parquet(['a.parquet', 'b.parquet'])`);
  Polars/Arrow zero-copy; window 函数 / `QUALIFY` / pivot 全是 SQL native;
  appender API 增量写不重写整文件 (虽然 A 股日级不真需要, 但 fund_flow 分钟级潜在有用).
- **SQLite** 适合 portfolio_state / paper_trade_log / 审计 trail (小表, 强 ACID, 单 writer
  锁定足够), **不适合** 7.9M row kline (列读慢, 索引膨胀).
- **TimescaleDB / ClickHouse / InfluxDB** 都是 multi-tenant / streaming 设计, 当前项目
  single-user + daily batch + 7.9M row, 用这些是 **杀鸡用牛刀, 且引入运维复杂度** (daemon,
  pwd, port, backup script, 容器化). 严格 OOS 协议看重的是"时间隔离审计性", server-side
  DB 反而增加状态泄漏风险 (cache, materialized view 谁也说不准是不是有 leak).

---

## 4. 推荐方案

### Top 1: **DuckDB (embedded) + 保留 parquet 作主存储**

**核心理念**: **parquet 不动, DuckDB 当 query engine + view layer**. 不做"迁移", 做"叠加".

**理由**:

1. **零迁移成本**. DuckDB 可以直接 `SELECT ... FROM 'data_cache/baidu_kline.parquet' WHERE
   trade_date >= '2025-01-01'` — 不需要先 import 到 DB, parquet 就是它的存储后端.
2. **解决痛点 #1, #2, #4 一次到位**:
   - 痛点 #1 (单文件笨重 swap): 可以用 `COPY ... TO 'kline_part_2026Q2.parquet'` 拆 Hive 分区,
     新增季度只写新 part, 不重写全表.
   - 痛点 #2 (subset 读慢): DuckDB 谓词下推 + row group 跳过, "2020 ∩ CSI300" 这种查询不再
     load 7.9M rows.
   - 痛点 #4 (没 SQL): 一行 `SELECT code, MAX(close)/MIN(close) - 1 AS gain FROM ... GROUP BY ...`.
3. **不破坏 production**. `paper_trade_today.py` 继续 `pd.read_parquet(KLINE_PATH)` 工作,
   DuckDB 是 **辅助分析层**, 不强制改主链路.
4. **不破坏 qlib**. `data_cache/qlib_baidu/` 仍是 bin, qlib 仍走 `D.features()`. DuckDB 跟
   qlib **不冲突, 各管各**.
5. **严格 OOS 协议天然保留**. parquet 文件本身的 trade_date 列就是时间隔离的 ground truth,
   DuckDB 加 `WHERE trade_date < '2021-05-01'` 就是 IS, `WHERE trade_date >= '2021-05-01'`
   就是 OOS, **不可能 leak**, 跟 server DB 的 cache/view 不一样.
6. **零运维**. pip install 一行 (现在没装就先不装, 评估通过再装), 文件级备份 (现在用的
   `.bak` 流程不变), 单 process 单进程, 没 daemon.

**典型用法示例** (思考层面, 不真跑):

```sql
-- 痛点 #4 一行查询
SELECT code,
       MAX(close)/MIN(close) - 1 AS gain_2025,
       (SELECT unlock_pct_next_60 FROM 'data_cache/unlock/unlock_detail_em.parquet' u
        WHERE u.code = k.code AND u.month = '2025-12') AS unlock_60d
FROM 'data_cache/baidu_kline.parquet' k
WHERE trade_date BETWEEN '2025-01-01' AND '2025-12-31'
  AND code IN (SELECT code FROM read_csv('data_cache/csi300_constituents.csv'))
GROUP BY code
QUALIFY gain_2025 > 0.5 AND unlock_60d > 0.1;
```

### Top 2 备选: **SQLite (小表) + parquet (大表) 双轨**

把 portfolio_state.xlsx / paper_trade_log.csv / dragon_tiger / unlock events 等 **行级强 ACID
需求的小表** 升到 SQLite (zipline 同款 pattern), kline / margin / fund_flow 大表继续 parquet.

**理由**: 比 DuckDB 更保守, 学习曲线更平. 跟 zipline-reloaded 的 `adjustments.py` 完全同款.
**缺点**: 没解决痛点 #4 (跨表 query), 还是要 pandas 拼.

### Top 3 不推荐但提一句: **HDF5**

zipline-reloaded 的 `hdf5_daily_bars.py` 是工业级实现, 但 **2026 年的工具链趋势是 parquet
+ Arrow 生态吸走了 HDF5 大部分用例**, 中文社区 HDF5 资料过气. 没理由在 2026 年新选 HDF5.

---

## 5. 迁移路径 (3 阶段)

**总原则**: **production 不动, 增量叠加, 任何阶段可回滚**.

### Stage 1: 兼容叠加 (1-2 天, 零风险)

目标: DuckDB 当 read-only query engine 用, **不写任何文件**.

- [ ] `pip install duckdb` (评估通过后才装; 当前 task 不装).
- [ ] 新建 `src/claude_finance/query.py`: 封装 `def query(sql: str) -> pd.DataFrame`,
      内部 `duckdb.sql(sql).df()`.
- [ ] 新建 `examples/duckdb_explore.ipynb` (新建, 不改老 notebook): 用 SQL 跑几个之前 30 行
      pandas 才能答的问题, 验证查询正确性.
- [ ] **production 完全不改**. `paper_trade_today.py` / `daily_check.sh` / launchd plist /
      forward_oos_monitor / strategy_v17_dens_grid 不动一行.
- [ ] **验证 gate**: dry-run picks 2026-05-22 = `[SH600039, SH688396, SH603993, SZ001965,
      SH601939, SZ300498, SH600018]` 100% 重合 (memory `project_v19_6_production_upgrade.md`).
      跟 DuckDB 引入前 baseline 必须 byte-identical.

### Stage 2: 主存储切换到 Hive 分区 parquet (1 周, 中风险)

目标: 把 `baidu_kline.parquet` 单文件 215 MB 拆成 Hive 分区, 解决痛点 #1, #2.

- [ ] 设计分区 schema: `data_cache/kline_v4/year=2014/data.parquet`, `year=2015/data.parquet`,
      ..., `year=2026/data.parquet`. 共 13 个 part, 每个 ~17 MB.
- [ ] 写 `examples/migrate_kline_to_hive.py`: `duckdb.sql("COPY (SELECT * FROM
      'data_cache/baidu_kline.parquet') TO 'data_cache/kline_v4/' (FORMAT PARQUET,
      PARTITION_BY (year))")`. 一次性脚本, 完了 rename `baidu_kline.parquet → .pre_hive.bak`.
- [ ] 修 `examples/paper_trade_today.py` 的 `KLINE_PATH`: 从单文件 path 变成 glob
      `data_cache/kline_v4/year=*/data.parquet`. **就一行改动**, 用 `pd.read_parquet(glob)`
      pandas 原生支持.
- [ ] 修 src/ 38 处 + examples/ 220 处的 `to_parquet/read_parquet`: 大多数指向 sub-domain
      文件 (margin, fund_flow, ...), 不动. 只动指向主 `baidu_kline.parquet` 的那 N 处.
- [ ] **验证 gate**: 同 Stage 1 dry-run picks 100% 重合. 再加: `examples/forward_oos_monitor.py`
      跑出来的 60D cum / 3M Sharpe 跟切换前 baseline 必须完全相等.

### Stage 3: 工具链同步 (3-5 天, 低风险)

目标: 让数据 fetcher / sanity check / completeness check 都用 DuckDB 的能力, 不再 .bak swap.

- [ ] 改 `examples/fetch_*.py` (margin_today / corrupt_fix / phase1c): 用 DuckDB 的
      `INSERT INTO 'data_cache/kline_v4/year=2026/data.parquet' SELECT ...` (append-only),
      不再覆盖整文件.
- [ ] `data_sanity_check.py` (`daily_check.sh` step 1.5) 升级: 用 SQL 跑 5 项 check
      (neg_close/extreme_jump/freshness/coverage/extreme_low). 现在是 pandas 实现, 切到
      SQL 后 single query 一次返回所有 metric, exit code 逻辑不变.
- [ ] `data_completeness_check.py` (step 1.6) 同样 SQL 化.
- [ ] 老 `.bak` 文件清理策略: 保留 3 个最近 swap 的 .bak (1.0 GB ~), 老的 archive 到外置
      盘. **不是 DB 决策, 是磁盘卫生**.
- [ ] **qlib bin 不动**. backtest / strategy_v17_dens_grid 继续走 `data_cache/qlib_baidu/`.
      qlib 跟 DuckDB 双轨永久并存.

### 回滚

- Stage 1 回滚: 删 `src/claude_finance/query.py` + `examples/duckdb_explore.ipynb`. 零影响.
- Stage 2 回滚: `cp data_cache/baidu_kline.parquet.pre_hive.bak data_cache/baidu_kline.parquet`,
  改回 `KLINE_PATH` 单文件 path. 一行 sed.
- Stage 3 回滚: 一阶段一阶段 revert 各 fetcher / check.

---

## 6. 严格保留 (不可妥协的红线)

1. **严格 OOS 协议保护** (memory `feedback_strict_oos_backtest.md`):
   - factor 选择期 2014-2020 / OOS 2021-05~2026-04 时间隔离 **不能因数据库变更而模糊**.
   - 任何 SQL `WHERE trade_date < '2021-05-01'` (IS) / `>= '2021-05-01'` (OOS) 要在 review 时
     人眼审一次, **不依赖 DB 自动 cache / materialized view** (那些可能跨期).
   - DuckDB 默认无持久 cache, 每次 query 重读 parquet, 这点天然符合.

2. **production `paper_trade_today.py` 不受影响**:
   - 现在: `pd.read_parquet(KLINE_PATH)` → Stage 2 后: `pd.read_parquet(glob_pattern)`.
   - **只动 KLINE_PATH 一行常量**, 不动 sidecar 数学 / 选股逻辑 / margin daily fetch / 限价过滤.
   - **Stage 2 后 dry-run picks 必须 byte-identical**, 否则全栈回滚.
   - 备份: 改动前 `cp paper_trade_today.py paper_trade_today.pre_duckdb.bak`.

3. **qlib bin 仍是 backtest 唯一格式**:
   - `data_cache/qlib_baidu/` 4610 features 目录 + 5 instruments txt **不被 DuckDB 取代**.
   - `strategy_v17_dens_grid.py` 继续 `D.features(instruments, fields, ...)` API.
   - Phase 2 clean retrain (memory `project_phase2_clean_retrain.md`) 的 113 min 重训路径
     不被打扰.
   - 双轨永久并存: parquet/DuckDB 给探索 + production paper_trade, qlib bin 给 Alpha158
     backtest.

4. **forward_oos_monitor.py / daily_check.sh / launchd plist 不动**:
   - launchd `com.claude_finance.daily_check` 已 16:30 trigger load,
     plist 路径写死. 改 plist 要重 unload/load.
   - forward_oos_monitor 已经 ship (memory `project_forward_oos_monitor.md`), 严格 OOS
     协议保护对象, 不在 refactor 范围.

5. **不引入 server / daemon**:
   - 不装 PostgreSQL / ClickHouse / InfluxDB / TimescaleDB.
   - DuckDB 是 embedded lib, `import duckdb` 等于 `import pandas`, 没 port 没 user 没 daemon.

---

## 7. 总结表

| 候选 | 推荐分 (0-10) | 部署成本 | 解决痛点 | 适用场景 |
|------|------------|--------|---------|---------|
| **DuckDB + parquet 叠加** | **9** | 极低 (pip 一行) | #1 / #2 / #4 全解决 | **首选** — single-user + daily batch + 看重 OOS 时间隔离 |
| **SQLite + parquet 双轨** | 6 | 低 (内置) | 部分解决 #1, 不解决 #2 / #4 | 备选 — 保守路径, 跟 zipline 同款 |
| **PostgreSQL + TimescaleDB** | 3 | 中 (装 daemon + 运维) | 解决 #4, 引入新痛 (运维) | 不推 — 多 user / streaming 场景才上 |
| **ClickHouse** | 2 | 中高 | 解决 #2 / #4, 引入新痛 | 不推 — 设计给亿级 row, 单机 7.9M 杀鸡用牛刀 |
| **InfluxDB** | 1 | 中 | 部分解决 #2 | 不推 — IoT/metrics 场景, A 股多维 fundamental 不友好 |
| **HDF5** | 4 | 低 (lib only) | 部分解决 #1 / #2 | 不推 — zipline 老路线, 2026 年生态已被 parquet 取代 |

---

## 8. 一句话总结

**当前项目主要痛点是 `baidu_kline.parquet` 单文件 215 MB swap 笨重 + 跨表 query 要 30 行
pandas, 不是 performance 极限, 不是 concurrent writes, 不是 backup. DuckDB embedded
直接读 parquet + Hive 分区拆主 kline 表能一周内解决, 零 server, 零 daemon, 严格 OOS 协议
天然保留 (因为 parquet trade_date 列就是时间隔离 ground truth, DuckDB 无持久 cache).
qlib bin / paper_trade / forward_oos_monitor 全部不动, 双轨永久并存.**

何时该真"上数据库" (升 PostgreSQL / ClickHouse)? 三个门槛同时满足:
1. **数据规模 > 100 GB 主行情** (现在 215 MB, 远不到);
2. **真实时流 / 多 fetcher 并发写竞争** (现在 launchd 时序错开就够);
3. **多用户协作 / 远程 query 需求** (现在 single-user, 0 远程).

三个都不满足的话, DuckDB 是天花板, 再往上是 over-engineering.
