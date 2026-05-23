# claude-finance

量化研究脚手架：vectorbt + backtrader 双回测引擎，akshare + tushare 数据源。

## 快速开始

```bash
# 1. 创建虚拟环境（推荐 uv，也可用 venv）
uv venv --python 3.11
source .venv/bin/activate

# 2. 安装
uv pip install -e ".[dev]"

# 3. 配置 tushare token（akshare 免 token）
cp .env.example .env
# 编辑 .env 填入 TUSHARE_TOKEN

# 4. 跑示例
python examples/backtest_vectorbt.py
python examples/backtest_backtrader.py

# 5. 测试
pytest
```

**注意（macOS arm64 + ML extra）**: 直接 `pytest` 跑包含 ML 策略的测试时，PyTorch + LightGBM + sklearn 会互相抢核死锁。`tests/conftest.py` 已经在 import 阶段固定 `OMP/OPENBLAS/MKL_NUM_THREADS=1` 解决；如果你**不通过 pytest** 直接运行（比如在 notebook / 命令行调 `lstm_signals()` 等），需要自己加：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## 目录结构

```
src/claude_finance/
├── data/          # 数据源 loader（akshare / tushare）统一返回 OHLCV DataFrame
├── strategies/    # 纯信号函数（无副作用，方便两个引擎复用）
├── backtest/      # vectorbt / backtrader 适配层
└── utils/         # 通用工具
examples/          # 可直接运行的端到端脚本
notebooks/         # Jupyter 探索性分析
tests/             # pytest 单测
data_cache/        # 本地行情缓存（gitignored）
```

## OHLCV 数据 schema

所有 `data/` loader 返回统一格式：

| 列 | 类型 | 说明 |
|----|------|------|
| index | `pd.DatetimeIndex` | 交易日，UTC naive，频率日级 |
| open / high / low / close | `float64` | 前复权价格 |
| volume | `int64` | 成交量（股） |

## 设计原则

- **策略与引擎解耦**：策略只写信号生成（`signals(df) -> (entries, exits)`），由 `backtest/` 适配到 vectorbt 或 backtrader
- **本地缓存**：raw 数据落 parquet 到 `data_cache/`，避免重复拉取

## 何时用哪个引擎

| 场景 | 引擎 |
|------|------|
| 大量参数扫描 / 多资产截面回测 | **vectorbt**（向量化，快） |
| 复杂订单管理 / 事件驱动逻辑 / 撮合细节 | **backtrader** |

---

## 🏆 生产部署目标：v13 K=8 / drop=2 / 5 万

**`examples/strategy_v13_k_sweep.py`** — qlib Alpha158 + LGB + TopkDropout（K=8 drop=2）

经 K ∈ {3,5,8,10,15,20} 全扫描，5 万账户在 **2017-2020 跨牛熊 OOS 44 月**最优：

| 配置 | cum % | ann % | Sharpe | MDD % | win % | 实际持仓 |
|------|------:|------:|------:|------:|------:|------:|
| **K=8 drop=2 ⭐** | **+207** | **+35.8** | 1.38 | -16.2 | 65.9 | 3.5 |
| K=20 drop=4（备选低波）| +160 | +30 | 1.51 | -13.1 | 70.5 | 7.7 |
| v10 K=30（弃用，对 5 万过大）| +142 | — | — | — | — | 5.5 |

K=8 比 v10 K=30 多出 **+65 pp**，因为 N_DROP 才是真正决定持仓数的参数，K=30 在 5 万账户上被 drop 节奏卡住。

### 验证为负向、已弃用的优化

| 版本 | 思路 | 实测结果 |
|------|------|---------|
| v12 | 行业 prior（白酒减仓 / AI+电力加仓）| -6.17 pp（破坏 alpha）|
| v14 | CSI300 MA200/vol/60d 风控开关 | MDD **-16% → -21%** / cum **-43 pp** |

教训：ML 模型本身已学到部分系统对冲；硬规则风控/行业 prior 多为后视镜信号，叠加只会干扰模型 timing。

参数说明：**目标持仓 8 只 / 每天最多换 2 只 / 跟着 LGB Top 8 走**，详见 [examples/v13_k_sweep_report.md](examples/v13_k_sweep_report.md)（本地）。

---

## 历史推荐（无 ML 简化版本）

`examples/strategy_recommended.py` — **v4-7d/Top3 + 保守增强**

### 配置

| 项 | 值 |
|----|----|
| 信号 | 价量动量（红盘 + 量能放大 + MA20 上 + 日内位置 0.5-0.95） |
| 频次 | 每周一选股，周二开盘买入，下下周三开盘卖出（持仓 7 个交易日） |
| 仓位 | 等权 Top 3，PE>30 转 Top 2 + 60% 仓位，PE>34 空仓 |
| 过滤 | 流通市值 ≥ 10 亿 / 换手率 ≥ 0.5% / WARMUP=252（去次新）/ 60 天 max\|chg\|>5.2%（去 ST）|
| 止损 | -5% 持仓期内任意 bar 触发即按 stop 价平仓 |
| 成本 | 0.25% 单次往返（印花 0.05 + 佣金 0.05 + 滑点 0.15）|
| **不加** | 板块轮换（v6 测试有害）、LGB ensemble（v7 测试有害）|

### 3 年长样本业绩（1,757 stocks, 2023-04 ~ 2026-05）

| 指标 | 值 |
|------|----|
| 总收益 | **+63.16%** |
| 年化 | +19.48% |
| Sharpe | **0.52** |
| 最大回撤 | -30.16% |
| 胜率 | 40.62% |
| 持仓周期数 | 96 / 99（轻仓 23，空仓 3）|

### ⚠️ 警告

- **Sharpe 0.52 是普通策略**，不是机构级 alpha
- 真实可执行 Sharpe 期望 **0.5 ± 0.3** 跨市场风格
- 强烈依赖 A 股结构（T+1、涨跌停、散户主导）
- 10 月样本曾出现 Sharpe 1.58 的"乐观结果"，3 年验证后回归 0.52（过拟合警告）
- **不构成投资建议；过往业绩不预示未来**

### 跑一遍

```bash
# 1. 拉 3 年长历史（首次跑约 15 分钟）
python examples/fetch_long_history.py

# 2. 跑推荐策略
python examples/strategy_recommended.py
```

---

## 探索过程的 8 个版本（教训记录）

| 版本 | 设计 | 教训 |
|------|------|------|
| v1 隔夜动量 | T close 选强势 → T+1 open 卖 | 10 月 net -83% 隔夜负 EV |
| v2 隔夜反转 | T close 选超跌 → T+1 open 卖 | 10 月 net -74% 仍隔夜负 EV |
| v3b 跨日 24h | T open 买 → T+1 open 卖 | net -30% 仍输 baseline |
| v3c 多日 + 止损 | 持 3 天 + -5% stop | gross +40% 但成本吃光 |
| v4 周 rebal Top 5 | 5d hold Top 5 | 10 月样本看似 Sharpe 1.58，3 年 -0.50（过拟合）|
| v5 LightGBM | 19 特征 ML 选股 | gross Sharpe 3.64 但 daily 换手吃光，3 年仅 +12% |
| v6 全增强 | + 板块 + ST + PE | 板块过滤砍 31pp，PE 帮助有限 |
| v7 ensemble | v4 + v5 软投票 | 全部跑输 v4 单跑 |
| **保守推荐** | **v4 + ST + warmup + PE，去板块去 ensemble** | 3 年 +63% / Sharpe 0.52 |
| **🚀 v8 qlib** | **Alpha158 (158 特征) + LGB + TopkDropout (k=30 drop 5)** | **20 月 walk-forward 累计 +152%，平均 IR 2.01，80% 月胜** |

详细见各 `examples/strategy_*.py`。

---

## 🚀 v8 qlib Alpha158 + TopkDropout（**新的最佳**）

`examples/strategy_v8_walkforward.py` — **借鉴 Microsoft qlib 三件套，做了 20 月 OOS walk-forward 验证**

### 三件套

| 件 | 内容 | 替代我们的 |
|----|-----|-----------|
| **Alpha158** | qlib 内置 158 个 cross-sectional 因子（量价/动量/反转/波动等）| 6 个 v4 手写规则 |
| **LightGBM** | 12 月滚动训练（每月 retrain）| 无（v4 是规则）|
| **TopkDropout** | 每天保 Top 30 + drop 5 + 进 5（平滑换手）| 周 rebal Top 3 一次性换 |

### 20 月 walk-forward 验证（2024-09 → 2026-04）

每月独立 retrain + test，零 look-ahead：

| 指标 | 值 |
|------|----|
| **正超额月** | **16 / 20 (80%)** |
| 平均月超额 | +4.80% |
| **累计超额（复利）** | **+134.58%** |
| **累计 absolute（复利）** | **+152.18%** |
| **平均月 IR** | **+2.01** |
| 折算年化 abs | **~78%** |
| 折算年化 excess | ~72% |
| 最差单月 | -10.26% (2026-03) |

### 跑一遍

```bash
# 0. 装 qlib (额外的 extra)
uv pip install -e ".[qlib]"

# 1. 拉 qlib 内置 cn_data (510MB, 跑 2017-2020 CSI300 demo 用)
python examples/strategy_v8_qlib_alpha158.py

# 2. 把我们 long_history.parquet 转 qlib bin 格式 (一次性, ~15s)
python examples/convert_parquet_to_qlib.py

# 3. v8 单窗口在我们 3 年数据上 (快, ~3 min)
python examples/strategy_v8_long_history.py

# 4. v8 walk-forward 20 月 OOS 验证 (~10-15 min)
python examples/strategy_v8_walkforward.py
```

### v8 vs recommended 对比

| 维度 | recommended (v4+) | **v8 qlib** |
|------|-----------------|-----------|
| Sharpe / IR | 0.52 (3 年全期) | **2.01 (20 月 walk-forward)** |
| 累计 net | +63% (3 年) | **+152% (20 月)** |
| 年化 | +19% | **~78%** |
| 持仓 | Top 3 周 rebal | Top 30 日 drop 5 |
| 特征 | 6 手写 | **158 cross-sectional** |
| 训练 | 无 | LGB 每月 retrain |
| 复杂度 | 低（纯规则）| 中（需 qlib）|

### v8 alpha 来源（不是 LGB 预测准！）

- **Rank IC 只有 0.012**（典型 alpha 0.03+）— LGB 预测其实不准
- 真正的 alpha 来自：
  1. **TopkDropout 调仓机制**：平滑换手 + 持续暴露在 LGB 预测的强势池
  2. **Alpha158 cross-sectional 因子**：捕捉了 v4 完全没有的市场结构
  3. **每月 retrain**：让模型跟上风格变化

### ⚠️ 警告

- **20 月样本仍然短**，需要 2017-2020 完整跨牛熊验证
- 历史 IR 2.01 不保证未来
- A 股结构（T+1、涨跌停）短期内变化 → 策略可能失效
- 真实交易 IR 期望 1.0-1.5（考虑容量、冲击成本）
