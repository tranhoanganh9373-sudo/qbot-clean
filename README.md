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

## ⭐ 推荐策略（生产可交易版）

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

## 探索过程的 7 个版本（教训记录）

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
| **推荐** | **v4 + ST + warmup + PE，去板块去 ensemble** | 3 年 +63% / Sharpe 0.52 |

详细见各 `examples/strategy_*.py`。
