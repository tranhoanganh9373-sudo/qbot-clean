# CLAUDE.md — claude_finance 项目规则

## 强制规则

### 数据获取必须用 a-stock-data skill

**所有 A 股行情/估值/资金面/研报/公告/新闻 数据获取必须先调用 `a-stock-data` skill**。

不要 ad-hoc 写 HTTP 请求 / 调 akshare / mootdx / 手写抓取，先看 skill 里有没有对应端点。skill 里有 7 层 28 个端点：
- 行情层 (mootdx/腾讯/百度)
- 研报层 (东财/同花顺/iwencai)
- 信号层 (热点/北向/龙虎榜/解禁/行业)
- 资金面 (融资融券/大宗交易/股东户数/分红/资金流)
- 新闻 (东财/财联社)
- 基础数据 (mootdx finance/F10/东财/新浪)
- 公告 (巨潮/F10)

只有 skill 里没有的（如指数历史 K 线）才可以扩展，扩展也优先用 skill 同源端点。

### sandbox 网络已知能/不能

| 端点 | sandbox |
|------|----|
| 腾讯 qt.gtimg.cn (实时报价 + 指数 + ETF) | ✅ |
| 百度 finance.pae.baidu.com (股票日 K + MA) | ✅ |
| 同花顺 basic.10jqka.com.cn (EPS 一致预期) | ✅ |
| 东财 reportapi.eastmoney.com (研报列表+PDF) | ✅（间歇 SSL EOF 加 retry）|
| 东财 datacenter-web.eastmoney.com (龙虎榜/解禁/融资融券/分红/股东户数) | ✅（间歇 SSL EOF 加 retry）|
| 东财 push2delay.eastmoney.com (个股基本面/全A clist/板块) | ✅ |
| 东财 push2.eastmoney.com | ❌ 502 Bad Gateway，用 push2delay |
| 东财 push2his.eastmoney.com (资金流 120日/指数 K线) | ❌ proxy/timeout |
| sina money.finance.sina.com.cn/quotes_service (指数 K 线) | ✅ |
| mootdx TCP 7709 (盘口/分钟 K/财务) | ❌ TCP 被阻断 |
| akshare (mini_racer 依赖) | ❌ V8 dlsym 错误 |

如果要拉 skill 里没有的数据，**优先试 sina 直 HTTP**（指数/K线常用）或扩展 skill 的 fallback host 链。

### 数据 I/O 落盘必须 gitignore

`data_cache/` 整目录被 gitignore；output `examples/*.csv` / `examples/strategy_*.md` 也 gitignore。**回测/实盘数据保留本地**，不推 GitHub。

## 生产部署目标

**v17 CSI300 baseline**（详见 `examples/strategy_v17_csi300_2023_2026.py`）

```yaml
策略:    qlib Alpha158 + LightGBM + TopkDropout
K=8, N_DROP=2, TRAIN=12月 rolling
Universe: CSI300 (300 只, 当前成分股, 有 survivorship bias)
本金:    50,000 元
过滤层:  涨停 ≥9.5/19.5% + 跌停 ≤-9.5/-19.5% + 价格 >125元
后验:   +53.8% / 40月 / Sharpe 0.71 / MDD -27% (含 survivorship)
真实期望: +8~12% 年化 (扣 survivorship + 实摩擦)
```

## 已弃用版本（不要复活）

| 版本 | 弃用原因 |
|------|------|
| v12 行业 prior | -6.17pp 破坏 alpha |
| v13 全 A baseline | 14 月 -56% kill |
| v14 CSI300 + regime | MDD -16→-21%, cum -43pp |
| v15 全 A + regime | 24 月 -30% |
| v16 CSI300 + regime | 24 月 -22.1% (regime CSI300 上一致负向) |

**结论**：
- regime gate 在 CSI300 上一致负向（v14 -43pp、v16 -23pp 两次验证）
- regime gate 在 broken model 上正向（v15 救了 26pp 但仍 -30%）
- 全 A universe 选高 beta 小盘不可持续

## 每日工作流

```bash
cd /Volumes/SSD/finance/claude_finance
source .venv/bin/activate

# 每天 14:30 后:
python examples/fetch_baidu_kline.py        # 15 min (TODO 加增量)
python examples/convert_baidu_to_qlib.py    # 1 min
python examples/paper_trade_today.py        # 30 sec CSI300
python examples/portfolio_excel.py          # 5 sec
```

打开 `data_cache/portfolio.xlsx`：
- **Positions** sheet：填 I/J 列（实际买入价/数量）
- **Daily** sheet：自动更新当日总资产
- **Notes** sheet（**周一开盘前看一眼**）：决策树 + 跌幅档 + 心理提醒
- **Weekly/Training** sheet：手填周/月小结

## 工程纪律

1. **每次写新 strategy 前，先看 examples/ 已有哪些**，不要重复造（v13~v17 已覆盖 K/regime/universe 组合）
2. **survivorship bias 必须标注**（CSI300 用当前成分股反推会高估 5-15pp）
3. **回测累计 +X% 不等于实盘 +X%**，扣 survivorship + 摩擦 + 行为偏差后剩一半算运气好
4. **任何 backtest 结论 < 2 年 OOS 都不可信**（单 regime 主导）
5. **新模型/参数验证前**，先在 CSI300 2023-2026 上跑（跟 v17 baseline 对比）

## 项目结构

```
claude_finance/
├── examples/               # 所有可执行入口 (10+ 脚本)
│   ├── fetch_*.py          # 数据获取 (universe/Baidu K线/指数/CSI300成分股)
│   ├── convert_*.py        # parquet → qlib bin
│   ├── strategy_v*.py      # 回测策略 (生产版 v13 K=8 / v17 验证)
│   ├── paper_trade_today.py    # 实盘信号 (CSI300)
│   └── portfolio_excel.py  # Excel 跟踪表
├── src/claude_finance/     # 库代码 (indicators/strategies/backtest/scan/risk)
├── data_cache/             # 本地数据 (全部 gitignore)
│   ├── universe.csv
│   ├── csi300_constituents.csv
│   ├── baidu_kline.parquet
│   ├── index_kline.parquet
│   ├── qlib_baidu/         # qlib binary
│   ├── portfolio.xlsx      # 5 sheet 跟踪表
│   └── portfolio_state.json
└── .venv/                  # uv 管理 (无 pip)
```

## 关键依赖

```bash
uv pip install -e ".[ml,qlib]"
uv pip install mootdx requests pandas stockstats openpyxl
```

不要用 `pip install`（venv 用 uv 装的，无 pip 入口）；必要时用 `python -m pip` 或 `uv pip`。
