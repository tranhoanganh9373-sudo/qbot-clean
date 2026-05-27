# QuantaAlpha Integration Plan

**Status**: research-only (read-only), NOT implementing evolutionary loop in this task.
**Date**: 2026-05-26
**Repo**: `references/QuantaAlpha` (cloned --depth 1, 45 MB)
**Paper**: arXiv 2602.07085, "QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining" (Han et al., 2026-02-06)
**License**: MIT

---

## 1. Paper Abstract (~100 字复刻)

QuantaAlpha 是一个 LLM 驱动的自演化 alpha 因子挖掘框架。核心创新有三：(1) **Diversified planning initialization** — 把单一研究方向扩展为 N 个并行 exploration directions；(2) **Trajectory-level evolution** — 每个 (hypothesis → factor expression → code → backtest → feedback) 链条视为一条 trajectory，通过 **mutation**（生成与父代正交的新假设）和 **crossover**（融合多个父代的互补优势）演化；(3) **Structured hypothesis-code constraint** — 通过 consistency checker 在 hypothesis ↔ description ↔ formulation ↔ expression ↔ code 五层间检查语义一致性，并通过 complexity / redundancy 约束防止过拟合与因子内卷。论文报告 CSI 300 上 IC=0.1501、ARR=27.75%、MDD=7.98%、Calmar=3.4774，并展示 zero-shot 迁移到 CSI 500 / S&P 500。

**Key empirical finding (我们关心的)**: 论文宣称在 2023 baseline collapse 时演化框架仍能发现 overnight info / volatility structure 等 structural factors 保持 IC 稳定 — 这是我们 2025-26 train24 失灵期最缺的能力。

---

## 2. Repo 结构 (主要模块)

```
QuantaAlpha/  (45 MB, MIT, Python 3.10+)
├── quantaalpha/
│   ├── pipeline/
│   │   ├── loop.py              (347 LOC) AlphaAgentLoop 主循环
│   │   ├── factor_mining.py     (656 LOC) parallel directions + evolution orchestration
│   │   ├── planning.py          (117 LOC) generate_parallel_directions
│   │   └── evolution/
│   │       ├── trajectory.py    StrategyTrajectory dataclass + RoundPhase enum
│   │       ├── mutation.py      MutationOperator (orthogonal exploration)
│   │       ├── crossover.py     CrossoverOperator (2+ parent fusion)
│   │       └── controller.py    Evolution orchestrator
│   ├── factors/
│   │   ├── proposal.py          (662 LOC) hypothesis → factor generation
│   │   ├── coder/               LLM 代码生成 (expr_parser, factor_ast)
│   │   ├── regulator/           consistency_checker.py + factor_regulator.py
│   │   ├── loader/              json / pdf 因子库 loader
│   │   ├── library.py           (343 LOC) FactorLibraryManager (JSON 持久化)
│   │   └── feedback.py          (410 LOC) 反馈生成
│   ├── llm/
│   │   ├── client.py            (987 LOC) APIBackend (OpenAI-compatible)
│   │   └── config.py            LLMSettings (Pydantic, env-driven)
│   ├── backtest/
│   │   ├── run_backtest.py      argparse 入口
│   │   ├── runner.py            (757 LOC) BacktestRunner
│   │   ├── factor_loader.py     (535 LOC) alpha158/360/custom/combined
│   │   └── custom_factor_calculator.py (619 LOC)
│   └── core/                    evolving_framework / evolving_agent / scenario
├── configs/
│   ├── experiment.yaml          全部 evolution / quality_gate / factor / llm 配置
│   └── backtest.yaml            qlib lgb + TopkDropoutStrategy
├── frontend-v2/                 React+Vite Web UI (FastAPI backend, optional)
├── docs/                        user_guide / experiment_guide / PROJECT_STRUCTURE
└── run.sh                       一键入口
```

**入口路径**: `run.sh → launcher.py → quantaalpha.pipeline.factor_mining` → AlphaAgentLoop（每个 direction 一个）→ Evolution Controller 调度 mutation / crossover rounds。

**数据契约** (摘自 configs/backtest.yaml + README §3):
- Qlib data: `~/.qlib/qlib_data/cn_data`（calendars/ features/ instruments/）
- Pre-computed pv: `git_ignore_folder/factor_implementation_source_data/daily_pv.h5`（398 MB，HuggingFace `QuantaAlpha/qlib_csi300`）
- 数据期: A 股 2016-01-01 ~ 2025-12-26，CSI 300
- 字段（推断自 alpha158 公式）: `[instrument, date, open, high, low, close, volume, vwap]`

**LLM**: OpenAI-compatible（默认 DeepSeek-V3，可换 gpt-4 / qwen-max）；客户端 `quantaalpha/llm/client.py` 987 LOC，自带 robust_json_parse、缓存、重试。

---

## 3. 我们项目 mapping 表

| QuantaAlpha concept | 我们的对应 | 状态 |
|---|---|---|
| **Trajectory** (hypothesis→factor→code→backtest→feedback) | Phase A (IS IC) + sweep + lock + OOS (Phase B) | **缺** mutation/crossover；目前是单一 trajectory，一锤定音 |
| **LLM hypothesis generator** | 人工脑暴 + reference 挖掘（10 candidate factors） | **缺**；可接 Claude API（已有 skill `claude-api`）或 DeepSeek |
| **Parallel planning** (N directions) | 串行 Phase A 因子（fundamentals / margin / DT / unlock 等） | **部分缺**；我们已有 8+ 因子探索，但无并行 LLM-driven 扩展 |
| **IC evaluation** | `factor_ic_*_is.py` 系列 + IS 2014-2020 strict | **已有，可复用**；适配格式即可 |
| **OOS test** | strict OOS protocol 60 月 (2021-05~2026-04) | **已有，可复用**；这是我们的硬资产 |
| **Semantic consistency** (hypothesis ↔ description ↔ expression ↔ code) | docs 分离，无 checker | **缺**；factor 名/公式/code 一致性靠人盯 |
| **Complexity constraint** | Spearman vs 现因子 + 文档记录复杂度 | **部分有**；symbol_length_threshold / base_features_threshold 没强约束 |
| **Redundancy / crowding control** | Phase 4 锁 λ 前手工跑 Spearman \|rho\|<0.10 | **部分有**；无 AST 子树匹配的自动检测 |
| **Multi-iteration evolutionary loop** | 单 trajectory pipeline（v19.4 → v19.6 是人工串行） | **缺**；7 次 sidecar 探索（v1/v5/v6/v7/v8/v9）全人工，无 mutation/crossover |
| **Factor library (JSON 持久化)** | `examples/v19_*_*_oos_stats.csv` 散落 | **缺统一存储**；可挪 |
| **Backtest** (Qlib TopkDropoutStrategy + lgb) | strategy_v17_dens_grid + paper_trade_today.py | **已有等价物**；Qlib lgb 是 close gap |
| **Dataset adapter** | baidu_kline parquet (v3 7.9M rows) + qlib_baidu bin (4604 features) | **需 adapter** to QuantaAlpha 期望 daily_pv.h5 格式（OHLCV+vwap by date×instrument） |

**5 行 summary**:
1. **Trajectory + Evolution 是 QuantaAlpha 的核心增量**；我们的强项是严格 OOS 协议和 clean hfq 数据，弱项是无 LLM-driven mutation/crossover、单 trajectory 探索。
2. **IC + OOS pipeline 100% 可复用**；只需把我们的 strict_oos_protocol 当成 QuantaAlpha backtest layer 的硬约束注入。
3. **Dataset adapter 是最小 blocker**（baidu_kline parquet → daily_pv.h5），其他都是上层抽象。
4. **LLM cost 是最大未知**；论文 N directions × M rounds × K factors 每 iteration ~30 min IC+OOS + LLM API token，单次实验 ~hundreds USD。
5. **Quality gate（consistency + complexity + redundancy）是我们目前最弱的一环**，但也是最容易先白嫖的（无需 evolution loop 也能用）。

---

## 4. 4 阶段实施路径

| Phase | 目标 | LOC 估 | wall 估 | LLM cost 估 |
|---|---|---|---|---|
| **Phase 1** Adapter + minimal trajectory | baidu_kline → daily_pv.h5 adapter；跑 1 个 manual hypothesis 走 QuantaAlpha pipeline；不开 evolution | ~200 LOC adapter + ~50 LOC config | 1-2 day | $0 (LLM off) |
| **Phase 2** LLM 调用 + 1 iteration | 接 Claude API（用 skill `claude-api` 的 prompt caching）+ 跑 1 round original phase（5 steps: propose/construct/calculate/backtest/feedback） | ~300 LOC + Claude wrapper | 3-5 day | ~$5-20 / iteration |
| **Phase 3** Mutation + crossover loop | 启用 evolution.enabled=true，跑 3 rounds (original→mutation→crossover)；接入我们的 strict OOS 协议作为 backtest acceptance gate | ~500 LOC (主要是 OOS guard + trajectory pool 落地) | 1-2 week | ~$50-200 / 完整 sweep |
| **Phase 4** Dashboard tab 7 evolution tree | dashboard 新 tab 展示 trajectory pool / mutation lineage / IC×OOS scatter / crowding heatmap | ~200 LOC (Streamlit / React) | 1 day | $0 |

**Total**: ~1250 LOC + 2-3 周 wall + ~$100-500 LLM cost for first end-to-end pass。

---

## 5. 风险 Top 5

1. **LLM cost 未知且可能失控**。论文用 DeepSeek-V3 + reasoning_model 双模型，单 trajectory ~50k-200k token；evolution mode 默认 N=2 directions × max_rounds=3 × crossover_n=2 ≈ 12 trajectories per run；按 Claude Sonnet 4.7 $3/M input $15/M output 估算单次完整 run **$50-300 区间**。需先在 Phase 2 实测 cost ceiling 再决定 Phase 3 上不上。**Mitigation**: 用 Claude prompt caching（skill `claude-api`）把 system prompt + factor library 缓存，可砍 70-90% input cost。

2. **Adapter 兼容性 + qlib data 双轨**。QuantaAlpha 期望 `~/.qlib/qlib_data/cn_data`（493 MB）+ `daily_pv.h5`（398 MB），我们有 `qlib_baidu bin`（4604 features）但格式差异未验证；HDF5 的 schema（fields, instrument×date hierarchy）需逐字段对齐。**Mitigation**: Phase 1 写一次性 dump 脚本 `dump_baidu_to_qlib_pv.py`，落到 `data_cache/quantaalpha_pv/`，与现有 production 数据物理隔离。

3. **严格 OOS 协议易被 evolution loop 破坏**。Mutation 利用 parent feedback 包含 OOS 信息时即构成 leak；crossover 选 parent 时若用 OOS 分数排序更是直接污染。论文 `parent_selection_strategy: best` 默认就是用历史表现选 parent — **必须把 best 限定为 IS 表现**，OOS 仅做 final acceptance gate。**Mitigation**: 在 controller.py / loop.py 加一层 wrapper，强制 feedback 阶段只暴露 IS metrics 给 LLM，OOS 仅 audit 不喂回。

4. **计算资源**：每次 iteration = IC 计算（~5 min for 300 stocks × 84 months）+ OOS backtest（lgb 训练 + portfolio sim ~10 min）+ LLM 调用（~2-5 min）≈ 20-30 min single trajectory；evolution 12 trajectories ≈ 4-6 hours wall。本地 CPU 跑 lgb 没问题，但 12 trajectory 全跑完一次实验 = 半天，迭代节奏受限。**Mitigation**: Phase 3 默认 max_rounds=1（只 original）validate 再扩 rounds。

5. **Crowding / overfitting risk increased**。论文报告 IC=0.15 Calmar=3.47 是 CSI300 in-distribution test 2022-2025 with Alpha158 baseline；我们 Phase 2 clean retrain 已观测到 train24 OOS Calmar 跌到 0.42（vs corrupt 1.21），sidecar 探索 v7-v9 全部 OOS 失败 → 当下市场可能本身 alpha 衰减，QuantaAlpha 的 IC 优势能否真实迁移到 CSI300 2025-26 是开放问题。**Mitigation**: Phase 1 直接复刻论文 CSI300 2022-2025 baseline 验证 IC=0.15 是否可重现；如重现失败说明 paper claim 在我们环境失效，不必再加 evolution layer。

---

## 6. Top 1 推荐 first step (~1 day)

**Goal**: 不装 venv、不调 LLM API，但跑通 baseline 数据兼容性。

**具体启动命令**:

```bash
# Step 1: 确认 clone 后 repo size + main entry
cd /Volumes/SSD/finance/claude_finance/references/QuantaAlpha
ls -la run.sh launcher.py configs/experiment.yaml configs/backtest.yaml

# Step 2: 读 PROJECT_STRUCTURE + experiment_guide
cat docs/PROJECT_STRUCTURE.md docs/experiment_guide.md

# Step 3: 写 adapter 草稿 (不实施,只规划)
# 输出: examples/quantaalpha_adapter_design.py  (~50 LOC, 仅 docstring + 数据流图)
#  - 读 data_cache/baidu_kline.parquet (4621 codes × 2014-2026)
#  - 转换到 daily_pv.h5 schema (HDF5, key="data", columns=[instrument,date,open,high,low,close,volume,vwap])
#  - 落 data_cache/quantaalpha_pv/daily_pv.h5
# 注意: 仅写设计文档,不真跑 to_hdf

# Step 4: 跑 dry-run sanity check (不调 LLM, 不跑 backtest)
# python -m quantaalpha.backtest.run_backtest \
#   -c configs/backtest.yaml --factor-source alpha158_20 --dry-run -v
# 这一步会加载 alpha158_20 内置 20 个因子,验证 qlib_data 兼容性

# Step 5: 落 follow-up checklist 到 docs/
# 不 commit,不改 production
```

**预期产出**: `examples/quantaalpha_adapter_design.py` 设计稿 + `docs/quantaalpha_first_step_results.md` 结果日志，总 ~100 LOC + 0 LLM cost + 0 production change，明确告诉我们 Phase 1 是否能开工。

---

## 7. 决策矩阵

| 决策 | 触发条件 | 行动 |
|---|---|---|
| **实施** Phase 1+2 | 复刻论文 CSI300 baseline IC≥0.10 + adapter 1 day 跑通 + Phase 2 Claude API cost <$30/iteration | 开 Phase 1 ticket，2 周内完成 Phase 2 first iteration |
| **跳过 QuantaAlpha** | 复刻论文 IC<0.05 或 adapter 工作量>3 day 或 LLM cost >$100/iteration | 停手，转而把 QuantaAlpha 的 consistency_checker / complexity gate 拆出来作为我们 sidecar workflow 的 sidekick，不上 evolution loop |
| **等更多 evidence** | 复刻 IC 介于 0.05-0.10 之间，或论文公布 v2 / community 验证更多 | 维持 read-only，每月 review 一次 arXiv / GitHub PR / community results；不上 Phase 1 |

**默认推荐**: **等更多 evidence**。理由：
1. 论文 2026-02 才发，距今 3 个月，还没有第三方独立复现。
2. 我们 Phase 4 sidecar 7 次失败已经证明 OOS 衰减是市场层面问题，不全是因子搜索深度问题。
3. 我们当下硬资产（baidu_kline v3 clean hfq + strict OOS + v19.6 in production Calmar 0.79）值得先消化稳定，不宜半年内再加大引擎风险。
4. 但 **consistency_checker 模式** 和 **complexity / redundancy gate** 可以单独拆出来用，无需 evolution loop — 这是我们能立即获益的子集。

---

## 8. Follow-up Open Questions

1. Paper 的 Alpha158 baseline IC=0.0843 在我们 baidu_kline v3 + qlib_baidu bin 上能不能复现？（Phase 1 直接验）
2. `daily_pv.h5` 的精确 schema 是什么？HuggingFace dataset 上有 README？（download 前要先确认 size 与字段）
3. Mutation / crossover 用 `parent_selection_strategy: best` 时，"best" 是 IS IC 还是 OOS IC？需要审 `controller.py` 选 parent 的具体 metric — 这关系 OOS leak 风险。
4. QuantaAlpha 自带 `factor_zoo` redundancy check 是否能直接挂上我们现有 11 个因子（pred / amp_imb_20d / margin_5d/20d / net_buy_pct_evt / unlock_*_60 等）？
5. Web UI（frontend-v2 + FastAPI backend）独立运行价值有多大？能不能复用到我们 dashboard tab 7？

---

## 9. 当前 task 严格遵守的边界

- Done: Clone `references/QuantaAlpha`（45 MB，shallow）
- Done: 写本 docs + memory 记录
- Skipped: 不实施 evolutionary loop
- Skipped: 不改 production
- Skipped: 不装 QuantaAlpha 到 venv（`pip install -e .` 没跑）
- Skipped: 不真调 LLM API（cost 未知）
- Skipped: 不下载 HuggingFace 数据集（493 MB + 398 MB）
- Skipped: 不 commit（user 决定何时 git add）
