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
