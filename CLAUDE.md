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
| mootdx TCP 7709 (盘口/分钟 K/财务 + qfq 历史 K) | ✅ 6 个 TDX server 实测可达(Phase 1B 用) |
| akshare 主 package | ✅ import 1.18.63 OK,大部分函数可用;**仅 stock_zh_a_hist 因底层 push2his 不可达失败**,需 fallback 到 mootdx + sina hfq.js |

如果要拉 skill 里没有的数据，**优先试 sina 直 HTTP**（指数/K线常用）或扩展 skill 的 fallback host 链。

### 数据 I/O 落盘必须 gitignore

`data_cache/` 整目录被 gitignore；output `examples/*.csv` / `examples/strategy_*.md` 也 gitignore。**回测/实盘数据保留本地**，不推 GitHub。

## 生产部署目标

**v19.1(production,2026-05-24 部署)**:`paper_trade_today.py`

```yaml
策略:        qlib Alpha158 + DoubleEnsemble + TopkDropout (no vol-target)
K=8, N_DROP=2, TRAIN_MONTHS=24, num_models=3
Universe:    CSI300 (300 只, 当前成分股, 有 survivorship bias)
实盘本金:     50,000 元
风控:        vol-target OFF (实测 vt=0.15 在 50k capital 上几乎无效)
过滤层:      涨停 ≥9.5/19.5% + 跌停 ≤-9.5/-19.5% + 价格 >125元
backtest 60月 (2021-05→2026-04): cum +695% / ann +51% / Sharpe 1.09 / MDD -26% / Calmar 1.96 ⭐⭐
月度胜率: 62% (全场最高)
真实期望 (扣 survivorship + 实摩擦): +27~33% 年化, MDD -30%~-35%
```

**v19 → v19.1 改动**:
- `VOL_TARGET_ANN = 0.15 → 0.0`(关闭 vol-target)
- 原因:60月 backtest 显示 vt=0.15 在 capital=50k 上 MDD 改善仅 0.7pp,cum 却损失 114pp
- 改后 Calmar 1.83 → **1.96**(全场冠军)

**之前 v17 / v18 / v19 production 历史**:
- v17 LGB(legacy):cum +244% / Sharpe 0.94 / MDD -27% — 保留作 safe-fallback
- v18 稳健 BD:DEns + train=12 + capital=25k backtest + vt=0.15 — cum +963% / Sharpe 1.18 / MDD -36%(2026-05-23 短期 production,已退役)
- v19:vt=0.15 未关 — Calmar 1.83(已升 v19.1)

**v18 激进（可选，未启用）**:DEns + capital=25000,**不加 vol-target**
- 后验 60月:cum +1988% / ann +84% / **Sharpe 1.32** / MDD -47%
- 实盘期望 ann +45~55%, **MDD -55%~-60%**(腰斩风险,99% 散户撑不住)
- 启动方式:`paper_trade_today.py` 把 `VOL_TARGET_ANN` 改 0.0

**v17 LGB（legacy / 安全回退）**:`strategy_v17_csi300_2023_2026.py`
- 后验 60月:cum +244% / ann +28% / Sharpe 0.94 / **MDD -27%**(最低)
- 实盘期望 ann +11~15%, MDD -30~-35%(最稳)
- 不再是生产默认,但保留作"心理承受不了 v18 -40% MDD"的回退选项

## 已弃用版本（不要复活）

### v12-v16(LGB 时代,全部弃用)

不启用。v12 行业 prior / v13-v15 全 A baseline / v14-v16 regime gate 全部显著负向(-22~-56%)。永久 archive。

### v17 衍生实验(12 月样本误导,60 月才真相)

| 版本 | 配置 | 12月 Sharpe | 60月 Sharpe | 弃用原因 |
|------|------|------|------|------|
| v17b | K=8 D=1(慢换手) | 1.37 | — | 慢于 D=2 sweet spot |
| v17c | K=8 D=2 xmonth(跨月持仓) | 0.48 | — | 2022Q1 暴击 MDD -40% |
| v17d | K=8 D=2 fastbuild(前 3 天 K=K) | -0.15 | — | 强填仓稀释 Q5 alpha |
| v17e | K=3 D=1 | 0.88 | — | N_DROP=1 慢建仓 |
| K=2 D=1 | 极致集中 | 0.67 | — | 单股风险过高 |
| K=3 D=2/3 | | 1.18~1.42 | — | 输 K=8 D=2 |
| K=4 D=2/4 | | 0.59~1.91 | — | 输 K=8 D=2 |
| K=5 D=2 | | 2.00 | — | 输 K=8 D=2 |
| K=8 D=4 | | 1.14 | — | 输 K=8 D=2 |

### v18 实验(DEnsemble 替换 LGB 后的降 MDD 方案)

| 方案 | 配置 | 60月 cum | 60月 Sharpe | 60月 MDD | 结论 |
|------|------|------|------|------|------|
| A Hybrid | LGB+DEns score 0.5/0.5 平均 | +897% | 1.04 | -41% | OK 但输纯 DEns |
| B Capital=25k(=v18稳健核心) | DEns + capital 减半 | +1988% | **1.32** | -47% | 单用 Sharpe 最高 |
| **C Stop-loss 10%** | DEns + 单股止损 | +1460% | 1.25 | -51% | ❌ MDD 反增,whip-saw 锁损 |
| D Vol-target 0.15 | DEns + 波动率减仓 | +794% | 1.15 | **-40%** | 单用 MDD 最低 |
| E 不动 | DEns 基线 | +957% | 1.14 | -50% | 对照 |
| **B+D 组合(=v18稳健)** | 全开 | +963% | 1.18 | **-36%** | **最优, 已 production** |
| F DEns + regime gate | bear/panic/drawdown | +167% | 0.65 | -33% | ❌ 跟 v14/v16 同样负向 |
| G DEns + Alpha360 (替 α158) | default capital=50k, no vol-tgt | +162% | 0.69 | -47% | ❌ single-fold ICIR=1.94 误导, 60月崩 |

### v19 / v20 实验(train_months sweep + universe sweep + ensemble 升级)

| 实验 | 配置 | 60月 cum | Sharpe | MDD | Calmar | 结论 |
|------|------|------|------|------|------|------|
| v19 CSI300 train=24 + vt=0.15 | 早期 v19 (vt 没关) | +581% | 1.09 | -26% | 1.83 | ✅ 好, 但 vt 浪费 cum |
| **v19.1 CSI300 train=24, no vt** | **production** | **+695%** | **1.09** | **-26%** | **1.96** ⭐⭐ | **生产部署** |
| CSI300 train=36 | 错误的 train 长度 | +130% | 0.63 | -46% | 0.39 ❌ | train=36 跨过 2022 bear regime 边界, 失败 |
| CSI300 train=48 | 略长 train | +407% | 1.08 | -26% | 1.48 | 比 train=24 略弱 |
| CSI500 train=24/36/48 | 中盘 universe | +12~+188% | 0.23~0.77 | -47~-65% | 0.04~0.46 ❌ | CSI500 universe 全部劣于 CSI300 |
| top1500_no_st train=24 | 全A大中盘剔 ST | +633% | 1.00 | **-62%** | 0.78 | win 65% 但 MDD 致命, 跟 v13/v15 全A 失败一致 |
| DEns num_models=5 | ensemble 加深 | +558% | 0.99 | -28% | 1.63 | ❌ 全面输 num=3, wall 2.8x |
| XGBoost (单模型) | 替 DEns | (失败) | (失败) | (失败) | (失败) | ❌ XGB 不容 NaN labels, 23/60 月 fail |
| CatBoost (单模型) | 替 DEns | (37/60 有效) | 0.70 | -31% | — | ❌ 同样 NaN label 问题 |

**永久结论**:
- regime gate 在 CSI300 上**三模型一致负向**(v14 LGB -43pp / v16 LGB -23pp / v18 DEns -790pp)→ 永久弃用
- stop-loss 在 walk-forward 下**只增 Sharpe 不降 MDD**(whip-saw 锁损)→ 不要用
- 跨月持仓延续在没显式止损时会 down-month 套牢(v17c MDD -40%)→ 月度清仓是 v17/v18 的"隐式止损"
- 强填 K=8 满仓稀释 Q5 头部 alpha(v17d 全面崩)→ 实际 picks=2-3 是 feature 不是 bug
- 全 A universe 选高 beta 小盘不可持续(v13 14月 -56%, v15 24月 -30%)
- **Alpha360 raw lagged price 在 walk-forward 下过拟合崩**(single-fold ICIR 1.94 在 60月 cum -795pp)→ 弃用; 用 Alpha158 (横截面 rolling stats) 才稳定; sequence model (LSTM/GRU) 需 GPU 跑, Mac CPU 死锁
- **TRAIN_MONTHS=24 是 sweet spot**(60月 walk-forward 实测: 12→Calmar 1.21, **24→1.96**, 36→0.39, 48→1.48); train=36 跨 regime 边界异常崩溃, 训练窗口 ≠ 单调
- **Universe 必须 CSI300**(CSI500/top1500 全部 train 配置都显著劣; 全 A high-β 小盘不可持续 v13/v15/top1500 三次验证)
- **DEnsemble num_models=3 已饱和**(num=5 反而拉低 Sharpe/win/cum; ensemble 加深稀释 alpha, 跟 multiwin/adaptive K 模式一致)
- **XGBoost / CatBoost 跟 qlib + Alpha158 不兼容**(NaN labels in CSZScoreNorm preprocessed data, 早期月份失败); 要用 XGB/CatBoost 需先 fillna 在 label 上, 但 qlib 默认 pipeline 不支持
- **vol-target 在 capital=50k(实盘 size)上几乎无效**(CSI300 实现波动率多数 < 15%, vt 不触发; 但 capital=25k backtest 集中度下 vt 有效是因为 cash 紧约束本身就在做减仓)→ v19.1 改为 vt=0

## 每日工作流

```bash
cd /Volumes/SSD/finance/claude_finance
source .venv/bin/activate

# 每天 14:30 后:
python examples/fetch_baidu_kline.py        # 15 min (TODO 加增量)
python examples/convert_baidu_to_qlib.py    # 1 min
python examples/paper_trade_today.py        # 40 sec CSI300 (v18 DEnsemble 比 LGB 慢)
python examples/portfolio_excel.py          # 5 sec
```

打开 `data_cache/portfolio.xlsx`：
- **Positions** sheet：填 I/J 列（实际买入价/数量）
- **Daily** sheet：自动更新当日总资产
- **Notes** sheet（**周一开盘前看一眼**）：决策树 + 跌幅档 + 心理提醒
- **Weekly/Training** sheet：手填周/月小结

## 工程纪律

1. **每次写新 strategy 前，先看 examples/ 已有哪些**，不要重复造（v12~v18 已覆盖 K/D/regime/capital/stop-loss/vol-target/universe/model 多维组合）
2. **survivorship bias 必须标注**（CSI300 用当前成分股反推会高估 5-15pp）
3. **回测累计 +X% 不等于实盘 +X%**，扣 survivorship + 摩擦 + 行为偏差后剩一半算运气好
4. **任何 backtest 结论 < 2 年 OOS 都不可信**(v18 实验血淋淋:12 月样本 Sharpe 2.76,60 月真相 Sharpe 0.94)
5. **严格 OOS 回测协议(CRITICAL,违反等于结果作废)**:
   - **Factor / 参数选择期 与 OOS 测试期 必须时间隔离**(默认 OOS = 2021-05 ~ 2026-04 60 月;选因子/调 λ 只能用 2014-2020 84 月)
   - **OOS 期上不允许 sweep 任何参数**(λ/K/D/train_months/factor combo 等都必须在选择期选定 → fix → OOS 跑一次)
   - **OOS 期上不允许对比多个变体取最佳**(只跑一个变体,失败就失败,不允许"试 5 个看哪个赢")
   - **看到 OOS 结果后不允许回头调任何参数**(看了就是用了,等同 leak)
   - **IC 分析期与 backtest OOS 期不能重叠**(过去 v22 系列 100 股 60 月 IC 跨 2021-2026 = 因子选择 leak,严格 OOS 要求在 2014-2020 重做 IC)
   - **predictions 必须严格 walk-forward**(train 用 T-N~T-2 月,validate T-1,predict T;参考 `strategy_v17_dens_grid.py:226-228`)
   - **margin / factor signal 必须 backward-looking**(`pct_change(N)` 在 sort by date asc 后是 OK 的,但 forward fill / 跨日 zscore 用 future 数据 = 作废)
6. **新模型/参数验证前**，**必须在 60 月 walk-forward 上跑**（跟 v17 LGB / v18 DEns 两个 baseline 对比）
7. **预测 cache 落盘**:任何 walk-forward 实验都应把 LGB/DEns predictions 写到 `data_cache/v17_*_predictions.parquet`,后续 grid sweep 可秒级复用（参考 `strategy_v17_grid.py` 的 `_persist_pred`）
8. **CSI300 实际持仓 picks=2-3 不是 bug**,是 cash 紧约束 + 涨停过滤的"集中筛选"效应（v18 B Capital=25k 把这个机制显式化）
9. **所有 fetch 数据必须落盘 `data_cache/`(CRITICAL)**:
   - **任何网络抓取(akshare/datacenter-web/push2/HKEX 等)的 raw 数据**必须存为 parquet/csv,不允许只在内存里处理后丢弃
   - **落盘命名约定**:`data_cache/{source}_{universe}_{kind}.parquet`(例:`baidu_kline.parquet` / `csi300_margin_14yr.parquet` / `index_kline.parquet`)
   - **per-stock cache 优先**:对慢接口(如 margin/财报/北向)用 `data_cache/{source}/{code}.parquet` 增量更新模式(参考 `src/claude_finance/margin_cache.py`)
   - **cache-first 强制**:scan / strategy / factor 脚本下游必须先读 cache,缺什么补什么,**禁止全量重抓**
   - **探查工具(scan_*/qbot_decision)抓的 sector/board 数据**也落 `data_cache/scan_cache/`(允许 stale,有 timestamp 标记即可)
   - **报表/聚合产物**(IC csv、stats csv、equity csv)仍写 `examples/` 或 `data_cache/reports/`,不与 raw data 混
   - 违反后果:每次 backtest 实验重抓 3 小时 + 上游接口变化无追溯 + 实验不可复现
10. **picks 推荐前必须全面检查数据完整性(CRITICAL,违反 = 推荐失效)**:
    - 来历:2026-05-25 发现 baidu_kline.parquet 缺 SH601939 建行 / SH601398 工行 / SH600519 茅台 等大蓝筹 → predictions cache 有这些股 → paper_trade 选它们入 picks 但 sidecar(amp_imb_20d / margin)算时数据缺失 fillna(0) 失效 → user 验证建行实际稳步上涨,本不应被 reverse sidecar 选中
    - **每只 picks 推荐前必须 verify 3 项:**
      - **a. kline 覆盖**:该股在 `data_cache/baidu_kline.parquet` 有近 N 日(N ≥ max sidecar lookback,典型 20-30 日)连续数据,无 missing date gap
      - **b. 复权一致性**:close > 0(无 neg)+ close ≥ 0.5 元(避免历史 corruption)+ 单日 |ret| ≤ 0.21(创业板/科创板 20% 涨跌停 buffer)
      - **c. sidecar 因子可算**:每只 picks 用到的 sidecar 因子(amp_imb_20d / margin_5d_chg / 龙虎榜 / etc.)必须**真实有计算值**,不是 fillna(0) 兜底
    - **paper_trade_today.py 应:**
      - 算 sidecar 时 record sidecar coverage(`covered: N/300`),低于阈值(< 90%)→ 警告 + 缺数据股**标 NaN 让其退出 picks**(不能 fillna 0 假装"中性")
      - 不让 ML pred 强(z 高)但 sidecar 缺数据的股**单纯靠 ML 信号**入榜(sidecar 是 production 设计的 alpha 核心,不能 silent ignore)
      - dry-run + 真 run 都 output 每只 picks 的"3 项 check 状态",user 一眼看出问题
    - **数据修复后必须重跑 sanity check 再上线**,确认 600/601 大蓝筹都在 baidu_kline 中,且 OHLC 复权正确
    - 违反后果:picks 决策基于残缺 sidecar 信号(失效但 picks 仍出列)→ user 实战买入逻辑错乱 → forward OOS 数据污染

## 项目结构

```
claude_finance/
├── examples/                       # 所有可执行入口
│   ├── fetch_*.py                  # 数据获取 (universe/Baidu K线/指数/CSI300成分股)
│   ├── convert_*.py                # parquet → qlib bin
│   ├── strategy_v*.py              # 回测策略 (v17 LGB legacy / v18 衍生实验)
│   ├── strategy_v17_grid.py        # 参数化 LGB grid (--k --drop --tag --first/last-test)
│   ├── strategy_v17_dens_grid.py   # 参数化 DEnsemble grid (+--capital --stop-loss --vol-target)
│   ├── strategy_v17_hybrid_grid.py # LGB+DEns score 平均
│   ├── paper_trade_today.py        # 实盘信号 (v18 稳健 DEns + capital=25k + vol-target=0.15)
│   ├── portfolio_excel.py          # Excel 跟踪表
│   ├── factor_analysis_v17.py      # alphalens 单因子 IC/IR
│   ├── perf_attribution_v17.py     # empyrical 业绩归因
│   └── qlib_benchmarks_v17.py      # LGB vs DEns single-fold IC 对比
├── src/claude_finance/             # 库代码 (indicators/strategies/backtest/scan/risk)
├── data_cache/                     # 本地数据 (全部 gitignore)
│   ├── universe.csv
│   ├── csi300_constituents.csv
│   ├── baidu_kline.parquet         # 全A股日K
│   ├── index_kline.parquet         # sh000300 等指数 K线
│   ├── qlib_baidu/                 # qlib binary
│   ├── v17_predictions.parquet     # LGB 60月 pred cache (~356k 行, grid 复用)
│   ├── v17_dens_predictions.parquet  # DEns 60月 pred cache
│   ├── v17_daily_returns.csv       # v17 LGB daily returns (alphalens/pyfolio 用)
│   ├── portfolio.xlsx              # 5 sheet 跟踪表
│   └── portfolio_state.json
└── .venv/                          # uv 管理 (无 pip)
```

## 关键依赖

```bash
uv pip install -e ".[ml,qlib]"
uv pip install mootdx requests pandas stockstats openpyxl
# v18 分析/归因栈 (alphalens-reloaded / pyfolio-reloaded / empyrical-reloaded
# 是 Stefan Jansen 的 2025 维护 fork, 替代已死的 quantopian/* 原版):
uv pip install alphalens-reloaded pyfolio-reloaded empyrical-reloaded
```

**禁用工具**:不要用 `pip install`(venv 用 uv 装的,无 pip 入口);必要时用 `python -m pip` 或 `uv pip`。

**已弃用的依赖**(2024 前停维护,2026 仍能装但用着风险大):
- `quantopian/alphalens`(用 `alphalens-reloaded` 替代)
- `quantopian/zipline`、`pyfolio`、`empyrical`(都换 `*-reloaded` fork)
- `mementum/backtrader`(2024-08 后无更新;考虑用 vectorbt)
- `waditu/tushare`(2024-03 后无更新;免费数据用 akshare/adata)

**仍活跃推荐**(2026 月度仍更新):
- `microsoft/qlib` (43k★, 本项目核心,持续维护)
- `akfamily/akshare` 
- `1nchaos/adata` (4.6k★, 免费 A 股数据备用)
