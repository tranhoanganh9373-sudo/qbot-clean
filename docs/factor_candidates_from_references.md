# Reference 项目因子挖掘 — Phase A 候选清单

**Scope**: Read-only 扫描 7 个 reference 项目, 提取**未测过**的新因子候选, 供下一轮 Phase A IS IC 分析使用。
**Production baseline**: v19.6 `final = z(pred) - 0.30 * z(amp_imb_20d)`, v3 OOS Calmar **1.29**。任何新候选必须能潜在超越此基线才有意义。
**Universe**: CSI300 296/300 cover, 数据已有 baidu_kline (v3, 2014-01~2026-05, hfq OHLCV) + qlib bin。
**Date**: 2026-05-26

---

## Step 1: 7 项目 factor 库清单

| 项目 | factor/indicator 库路径 | 数量 | 类别 |
|---|---|---|---|
| **zipline-reloaded** | `src/zipline/pipeline/factors/{basic,statistical,technical,events}.py` | ~25 | Returns, MaxDrawdown, RSI, BollingerBands, Aroon, FastStochastic, IchimokuKinkoHyo, MACDSignal, TrueRange, RollingPearson/Spearman/Beta, EWMA, AnnualizedVolatility, ROC, VWAP, AverageDollarVolume, PeerCount |
| **zipline** (原) | 与 zipline-reloaded 相同结构,差异在 reloaded 加了 `PercentChange`、`Clip` 等 utility | ~22 | 子集 |
| **Lean (QuantConnect)** | `Indicators/*.cs` | **169 个** indicator 类 | 全量技术指标库; 含 InternalBarStrength, AugenPriceSpike, ChoppinessIndex, RogersSatchellVolatility, HurstExponent, KaufmanEfficiencyRatio, Momersion, ConnorsRSI, RelativeDailyVolume, SqueezeMomentum, UltimateOscillator, MFI, ChandeMomentumOscillator, EaseOfMovement, WaveTrendOscillator, TomDemarkSequential, FractalAdaptiveMA, MesaAdaptiveMA, ParabolicSAR, SuperTrend, VortexIndicator, SchaffTrendCycle, NormalizedATR, OBV, McGinleyDynamic, FisherTransform, etc. |
| **OpenBBTerminal** | `extensions/technical/openbb_technical/{technical_router,helpers,relative_rotation}.py` | ~25 router + 7 vol estimators | 含 Clenow Volatility-Adjusted Momentum, Cones (vol quantiles), Parkinson/Garman-Klass/Hodges-Tompkins/Rogers-Satchell/Yang-Zhang, Relative Rotation (RRG), Fisher, Donchian, ADX, AROON, CCI, KC, Demark, Z-score normalize |
| **alphalens-reloaded** | `src/alphalens/{performance,utils,tears}.py` | factor analytics, 非因子库 | `factor_information_coefficient`, `factor_returns`, `quantile_turnover`, `factor_rank_autocorrelation`, `mean_return_by_quantile`, `compute_forward_returns`, `quantize_factor` — **方法论工具,非新因子** |
| **empyrical-reloaded** | `src/empyrical/stats.py` | ~50 perf metrics | `sortino_ratio`, `tail_ratio`, `down_capture`, `value_at_risk`, `cvar`, `gpd_risk_estimates`, `beta_fragility_heuristic`, `batting_average`, `stability_of_timeseries` — **绩效统计,但可改造成 cross-sectional 风险因子** |
| **pyfolio-reloaded** | `src/pyfolio/{performance,tears,plotting}.py` | tear-sheet generators | 同 empyrical, 单个组合分析,**没有 panel factor 库** |

**总评**:
- Lean 169 个 C# indicator 是最大金矿,大多数 daily OHLCV 可直接 port (公式都在注释里)
- OpenBB 7 个 volatility estimators (Yang-Zhang、Garman-Klass、Rogers-Satchell 等) 是 daily OHLC 已有数据可计算的**高质量波动率因子**, 跟 amp_imb_20d (已测) 机制相关但**信号源不同**
- alphalens / empyrical 是分析工具,不是新因子
- pyfolio 不提供 panel factor

---

## Step 2: 候选因子分类清单 (每项目 ≥ 5 个未测)

### A. zipline-reloaded (5 个未测)

| 因子 | 数据需求 | 机制 | vs 已测差异 |
|---|---|---|---|
| **RSI(14)** | close (已有) | gain/(gain+loss),momentum oscillator | 跟 v19.4 m5/m20 是融资融券, 跟 amp_imb 是高低价振幅 — **价格涨跌幅 momentum 维度未测** |
| **Aroon up/down(25)** | high, low (已有) | 距 N 期最高/最低点的天数 → trend 强度 | 跟 v20 industry_60d 不同 (那是相对行业, 这是个股自身高低点 timing) |
| **FastStochasticOscillator %K(14)** | close, high, low (已有) | (close - lowest_low) / (highest_high - lowest_low) | 跟 InternalBarStrength 单 bar 不同, 是 N-bar 窗口位置 — **窗口化位置因子,未测** |
| **IchimokuKinkoHyo (Tenkan/Kijun gap)** | high, low (已有) | 9d (9+26)/2 mid-point cross | 多段 mid-point spread, 跟现有简单 MA 不同 |
| **MACD Signal(12,26,9)** | close (已有) | fast/slow EWMA diff, signal line cross | Alpha158 可能有 MA 但 MACD 9d signal cross 离散 cross 信号未必有 |

### B. zipline (原, 与 reloaded 重叠)

跟 zipline-reloaded 一致,**不计入,避免双重 count**。

### C. Lean (≥ 8 个最有价值未测,高度差异化)

| 因子 | 数据需求 | 机制 | vs 已测差异 |
|---|---|---|---|
| **InternalBarStrength (IBS)** | OHLC (已有) | `IBS = (close - low) / (high - low)`, **单 bar** close 在 H-L 范围内的相对位置 | **跟 amp_imb_20d 完全不同**: amp_imb 是 20 日累计 amp_up vs amp_dn, IBS 是单日 bar 内 close 位置. Mean-reversion alpha (close 接近 low → bullish next day) |
| **AugenPriceSpike (APS)** | close (已有) | `(close - close[-1]) / (std(log_returns[-period]) * close[-1])`, 单日价格变动 **以滚动波动率为标尺**, 标准化后的 surprise | 跟现有 raw return / amplitude 不同, 是 **波动率归一的 jump 因子** |
| **Rogers-Satchell Volatility** | OHLC (已有) | `RSV = sqrt(sum(log(H/C)*log(H/O) + log(L/C)*log(L/O)) / n)`, drift-aware 波动率 | 跟 amp_imb_20d 是 amplitude 不对称, RSV 是**绝对波动率水平** — 信号源不同 (level 而非 imbalance) |
| **Hurst Exponent** | close (已有) | log-log 回归 sigma vs lag,衡量 **mean-revert (<0.5) / trend (>0.5) / random walk (=0.5)** | 完全独立机制 — 不是 momentum/reversion 强度,而是**判断 regime 类型**. CSI300 可作 cross-sectional regime detector |
| **Kaufman Efficiency Ratio (KER)** | close (已有) | `KER = abs(close[t] - close[t-N]) / sum(abs(daily diff))`, **noise-to-signal**: 直线度量 | 跟 industry_60d 不同 (那是相对超额收益), KER 是**轨迹直线度**, 直接打 trend quality |
| **Choppiness Index (CHOP)** | OHLC (已有) | `100 * log10(sum(TR)/(maxHigh-minLow)) / log10(N)`, 0-100 衡量市场 "震荡 vs 趋势" | 跟 Hurst 互补 (Hurst 是统计 long-memory, CHOP 是 ATR 累计 vs 价格跨度) |
| **Connors RSI (CRSI)** | close (已有) | RSI(3) + StreakRSI(2) + PercentRank(100) 的平均, 短期均值回归 oscillator | 跟传统 RSI 不同, **三段融合**,实测 short-term mean-rev 经典工具 |
| **Money Flow Index (MFI)** | OHLCV (已有) | 类似 RSI 但用 typical_price * volume 加权 — **量价 oscillator** | 跟 v20 super_big_net 不同 (那是资金流分层 catastrophic abort), MFI 是 OHLCV 自计算 — **数据完整,无 gap** |
| **Ease of Movement (EMV)** | OHLCV (已有) | `mid_diff / (vol/scale / (H-L))`, **少量 volume 推动大价格 → 大 EMV** | 跟 vol_z_5d 不同 (那是 volume 自身 z-score, 假 alpha abort), EMV 是 **volume-to-price-move efficiency**, 信号源截然不同 |
| **Chande Momentum Oscillator (CMO)** | close (已有) | `(gain-loss)/(gain+loss)`, **RSI 的对称形式** (-100 to 100) | RSI 变种,差异在 symmetric scale (RSI 0-100 vs CMO -100~100), 截面排序可能更 robust |
| **Ultimate Oscillator (UO)** | OHLC (已有) | 三 timeframe (7/14/28) buying-pressure / TR 加权平均 | **多周期融合**, 跟单一周期 RSI/CMO 不同 |
| **Squeeze Momentum (BB inside KC)** | OHLC (已有) | Bollinger Bands inside Keltner Channel → 1 (squeeze on) / -1 (off), 波动率收缩 regime | 跟 v19.6 amp_imb 不同 (那是已发生振幅, squeeze 是 **未来 breakout 预期**) |

### D. OpenBBTerminal (5 个未测)

| 因子 | 数据需求 | 机制 | vs 已测差异 |
|---|---|---|---|
| **Clenow Volatility-Adjusted Momentum** | close (已有) | log-price 线性回归,**slope * R^2** 联合打分, Andreas Clenow 经典做法 | 跟 raw N 日收益不同, R^2 过滤 "轨迹平滑度" 加权 — **slope * trajectory quality** 双因子合一 |
| **Yang-Zhang Volatility** | OHLC (已有) | `sigma_overnight^2 + k*sigma_opens^2 + (1-k)*sigma_RS^2`, 综合 overnight + open + 日内 | 跟 RS、Parkinson 不同, **最 unbiased 估计量** (Drost & Werker 2007 推荐), 跟 amp_imb 是 imbalance 不是 level — 互补 |
| **Garman-Klass Volatility** | OHLC (已有) | `0.5*log(H/L)^2 - (2log2-1)*log(C/O)^2`, **8 倍 close-to-close 估计效率** | 跟 RSV 是同族但 公式独立, 系数不同 |
| **Parkinson Volatility** | high, low (已有) | `(1/4log2)*log(H/L)^2`, **最简洁** 1-bar HL 范围 | 跟 amp_imb 直接相关 (都用 H-L), 但**未做 scaling**, 可能高度相关 → 候选可能 redundant 待 spearman 验证 |
| **Cones (vol quantiles)** | OHLC (已有) | 12 个 windows (3-360d) 的 vol 历史 quantile, 当前 vol 位置 | **新颖**: 跟单一 N 日 vol 不同, **vol 自身的 percentile rank** 是 fresh signal (低 vol 状态 → 可能预示突破) |

### E. alphalens-reloaded / empyrical-reloaded / pyfolio-reloaded

**结论**: 这三个项目主要是 **factor analytics 和 performance metrics**, 不是新因子库。
- alphalens: `factor_information_coefficient`、`quantile_turnover` 已被现有 IC 流程使用 (concept-level)
- empyrical: `sortino_ratio`、`tail_ratio` 是 portfolio-level metrics, 单股 panel 用处有限
- pyfolio: tear-sheet generator, 无 factor 库

**有限改造空间** (低优先级):
- **Sortino Ratio** 可改造为 cross-sectional **60d 个股 sortino z-score** (downside-only vol-adjusted return)
- **Tail Ratio** 可作 60d 个股 **upside/downside tail 比例**, 与 amp_imb 机制相似但**用 return percentile 不是 amplitude**

---

## Step 3: 评分排序

| 候选 | 数据可得性 | 独立性 (vs v19.6) | 样本充裕度 | 实施成本 | 总分 |
|---|---|---|---|---|---|
| **Internal Bar Strength (IBS)** | 5 (OHLC 已有) | 5 (单 bar 位置 vs 20 日累计 imb) | 5 (日频 ≥ 120 月) | 5 (单行公式) | **20/20** |
| **Kaufman Efficiency Ratio (KER)** | 5 | 4 (trajectory quality, 跟 industry_60d 有部分重叠) | 5 | 5 | **19/20** |
| **Aroon up/down (25)** | 5 | 4 (个股极值 timing) | 5 | 5 | **19/20** |
| **Clenow Volatility-Adjusted Momentum** | 5 | 4 (R^2 加权 momentum, slope 已在 Alpha158) | 5 | 4 (需 sklearn lr) | **18/20** |
| **Yang-Zhang Volatility** | 5 | 4 (vol level, amp_imb 是 imb) | 5 | 4 | **18/20** |
| Augen Price Spike (APS) | 5 | 4 (vol-norm jump) | 5 | 4 | 18/20 |
| Hurst Exponent | 5 | 5 | 4 (需 maxlag ≥ 20 → window ≥ 60d) | 3 (多次回归) | 17/20 |
| Rogers-Satchell Volatility | 5 | 4 | 5 | 4 | 18/20 |
| Choppiness Index | 5 | 4 | 5 | 4 | 18/20 |
| Connors RSI | 5 | 3 (跟 RSI 相关) | 5 | 4 | 17/20 |
| Money Flow Index | 5 | 4 (OHLCV 自算, 跟 super_big_net 资金流分层不同) | 5 | 4 | 18/20 |
| Ease of Movement | 5 | 4 (volume-to-price efficiency, 跟 vol_z_5d 不同) | 5 | 4 | 18/20 |
| Chande Momentum Oscillator | 5 | 3 (RSI 同族) | 5 | 5 | 18/20 |
| Ultimate Oscillator | 5 | 4 | 5 | 4 | 18/20 |
| Squeeze Momentum (BB ⊂ KC) | 5 | 4 (regime, 跟 Hurst 同 family 但更短期) | 5 | 3 (双 indicator) | 17/20 |
| Garman-Klass Volatility | 5 | 3 (vol level 跟 YZ 同族 rho>0.7 预期) | 5 | 4 | 17/20 |
| Cones (vol quantile) | 5 | 5 (vol 的 percentile rank, 信号源唯一) | 4 (需 360d max window) | 3 (12 个 sub-window) | 17/20 |
| RSI(14) | 5 | 3 (Alpha158 可能已有 momentum) | 5 | 5 | 18/20 |
| Fast Stochastic Oscillator | 5 | 4 (窗口化 close 位置, 跟 IBS 互补) | 5 | 5 | 19/20 |
| Parkinson Volatility | 5 | 2 (跟 amp 高度相关) | 5 | 5 | 17/20 |

---

## Step 4: Top 5 候选推荐

### 1. Internal Bar Strength (IBS) — **20/20 推荐 Top 1**

- 来源: Lean `Indicators/InternalBarStrength.cs`
- 数据: 已有 baidu_kline OHLC, 无需 fetch
- 机制: `IBS = (close - low) / (high - low)`, 单 bar close 在当日 H-L 范围内的归一化位置。学术界与 quant 文献多次证实其 **mean-reversion 短期 alpha** (close 接近 day-low → 次日反弹概率高), 见 Larry Connors 的多本短期交易书。
- 跟现 factor 差异: amp_imb_20d 是 **20 日累计** amp_up vs amp_dn 比例 (mid-term, imbalance), IBS 是 **单日** close 在 H-L 的相对位置 (intraday, level). 时间尺度不同, 信号源 (close 位置 vs 振幅极性) 不同, 预计 Spearman |rho| < 0.10
- 数据可得性: 5 (零 fetch)
- 独立性: 5 (跟 amp_imb 完全不同时间尺度+不同机制)
- 样本充裕度: 5 (日频, IS 2014-2020 84 月 × 296 股 ~ 17,640 月度观测)
- 实施成本: 4.5 (公式 1 行, 但需要 N 日 mean/median 平滑成因子)
- **总分**: **20/20**
- Phase A 草案 (详见 Step 5)

### 2. Kaufman Efficiency Ratio (KER) — **19/20**

- 来源: Lean `Indicators/KaufmanEfficiencyRatio.cs`
- 数据: 已有 close
- 机制: `KER = abs(close[t] - close[t-N]) / sum(abs(daily_diff))`, 衡量 N 日轨迹的 **直线度** (1.0 = 完全直线, 0 = 完全往返). 是 **noise-to-signal** 比, Perry Kaufman 1998 提出
- 跟现 factor 差异: amp_imb_20d 是振幅 imbalance (方向无关), KER 是 **轨迹平滑度** (含方向 sign(close[t]-close[t-N])); 与 industry_60d 不同 (相对超额 vs 个股 trajectory quality)
- 数据可得性: 5
- 独立性: 4 (跟 momentum 略相关但 R^2-like efficiency 维度独立)
- 样本充裕度: 5
- 实施成本: 5
- **总分**: 19/20
- Phase A 草案: `ker_20d = sign(ret_20d) * abs(ret_20d) / sum_abs_daily_ret_20d`, cross-sectional rank, IC vs next-month return

### 3. Aroon up/down (25) — **19/20**

- 来源: zipline-reloaded `pipeline/factors/technical.py:Aroon`
- 数据: 已有 high/low (无 fetch)
- 机制: `aroon_up = 100 * (N-1 - days_since_highest_high) / (N-1)`, `aroon_down` 同理用 low. 衡量距 N 期最高/最低点的天数 → 趋势强度
- 跟现 factor 差异: v20 industry_60d 是个股 vs 行业相对超额 (relative momentum), Aroon 是个股自身的 **极值 timing** (absolute extreme proximity). 维度完全不同
- 数据可得性: 5
- 独立性: 4
- 样本充裕度: 5
- 实施成本: 5 (`argmax/argmin` 一行)
- **总分**: 19/20
- Phase A 草案: 计算 `aroon_up_25 - aroon_down_25` 作为单一因子 (-100~100), 月末取值, IC 验证

### 4. Yang-Zhang Volatility — **18/20**

- 来源: OpenBB `helpers.py:yang_zhang` (基于 Yang & Zhang 2000)
- 数据: 已有 OHLC
- 机制: 综合 overnight + open + Rogers-Satchell drift-aware 波动率, **理论上最无偏的 OHLC vol 估计**
- 跟现 factor 差异: amp_imb_20d 是 amplitude **imbalance** (上振 vs 下振), YZ 是 absolute vol **level**. 信号源不同
- 数据可得性: 5
- 独立性: 4 (跟 amp_imb 有 H-L 共享输入,预期 Spearman 0.3~0.5 中等)
- 样本充裕度: 5
- 实施成本: 4 (公式 3 段)
- **总分**: 18/20
- Phase A 草案: 20 日 YZ vol, low-vol → next-month return higher (low-vol anomaly), 期望 sign 负

### 5. Clenow Volatility-Adjusted Momentum — **18/20**

- 来源: OpenBB `helpers.py:clenow_momentum` (Andreas Clenow "Stocks on the Move")
- 数据: 已有 close
- 机制: 90 日 log-price 线性回归, `factor = annualized_slope * R^2`. R^2 过滤掉"突然暴涨" 看似高 slope 实际不平滑的股票
- 跟现 factor 差异: Alpha158 大概率有简单 momentum, 但 **R^2 加权后的 slope** 是经典 Clenow 改造; 跟 KER 同族 (都是 trajectory quality) 但 KER 是 noise-to-signal, Clenow 是 slope * fit quality — **应做 Spearman 验证候选 2 与 5 是否高度相关**
- 数据可得性: 5
- 独立性: 4
- 样本充裕度: 5 (需 90d → 至少 7 个 lookback, IS 84 月 仍充裕)
- 实施成本: 4 (sklearn LinearRegression per stock × month)
- **总分**: 18/20
- Phase A 草案: 月末用 sklearn 算 90d clenow factor, cross-sectional rank, 期望 sign 正

---

## Step 5: Top 1 Phase A 实施草案 — IBS 单 bar 位置因子

### 公式

```
# 每日 (code, date)
ibs_t = (close_t - low_t) / (high_t - low_t + 1e-12)
# Rolling smoothing (per code)
ibs_5d_mean  = rolling.mean(ibs_t, window=5)
ibs_20d_mean = rolling.mean(ibs_t, window=20)
# 单期版 (无平滑)
ibs_raw = ibs_t
```

### Panel 构造 (IS 期 2014-01 ~ 2020-12)

1. Load `data_cache/baidu_kline.parquet` (v3, hfq, 已修) — schema 见 `examples/_factor_kline_panel.py:64` (columns: `code` 6-digit zero-padded str, `date` Timestamp, `open`, `high`, `low`, `close`, `volume`)
2. Filter `code` ∈ CSI300 universe (296/300 cover, 见 `data_cache/instruments/csi300.txt` 或 daily_check.sh universe 配置)
3. Compute `ibs_raw`, `ibs_5d_mean`, `ibs_20d_mean` per (code, date)
4. **月末取值**: `month_end = max date in month` (last trading day)
5. Drop NaN (window warm-up 前 20 日)
6. Panel: 84 月 × 296 股 ≈ 24,864 月度观测 (扣除 warm-up ≈ 296 行 ≈ < 1.5%)

### IC 计算方法

参照 `examples/factor_ic_volume_zscore_is.py` 既有 pattern:
- 严格 PIT (point-in-time): factor at month-end → 用 **next month return** (避免 forward look-ahead)
- Pearson IC (cross-sectional, monthly)
- ICIR = mean(IC) / std(IC)
- 三个 horizon 平行测试: `ibs_raw` (单日), `ibs_5d_mean`, `ibs_20d_mean`
- 输出: `examples/factor_ic_ibs_csi300_is.csv` + `_monthly.csv` + `_report.md`

### 期望 ICIR 范围

文献 prior (Connors 2008 + 多份 quant blog 实测):
- US S&P 500 daily IBS short-term mean-rev: **ICIR ~ -0.4 to -0.8** (sign 负, IBS 高 → next return 低)
- 中国 A 股 ST 退市筛选后预期: ICIR -0.3 to -0.6 (相对美股略弱因 T+1 摩擦但仍显著)
- **筛选门槛 (避免 Phase B 失败模式 "IS Calmar > 1.5 + λ > 0.20")**:
  - 若 |ICIR| < 0.5 → 暂不进 Phase B
  - 若 |ICIR| ∈ [0.5, 1.0] → 可进 Phase B 严格 OOS, **预设 λ ≤ 0.15** 避免 overfit
  - 若 |ICIR| > 1.5 (类比 unlock c8_combo IS 2.70 → OOS catastrophic) → **警惕 overfit**, λ 收紧到 ≤ 0.10

### 风险

1. **n_months 不是问题** (84 月 IS, 60 月 OOS, 远超 60 月最低阈)
2. **T+1 / 涨跌停过滤**: 涨停板当天 close == high → IBS = 1, 反向信号可能被涨停板放大. **建议 Phase A 加涨跌停 mask** (`(close - prev_close)/prev_close < 0.099` 过滤)
3. **跟 amp_imb_20d 的 Spearman**: 必须验证 |rho| < 0.10 (否则即使有 alpha 也是 sidecar redundant). 月度横截面 Spearman 直接 compute, 类比 `examples/v19_9_unlock_amp_spearman.csv` 既有套路
4. **Sample skew**: IBS 自然分布在 0~1 区间, 跨股可比性强 — 不需要 z-score (但跨月份 z-score 仍推荐 robust)
5. **mean-rev decay**: IBS 是短期反转因子, 月频可能过于稀疏. 强烈建议同时跑 **次周 (5 trading days) IC** 作为对照, 若 5d ICIR 显著 > 20d, 提示 **factor 更适合周频, 应做更高频版 sidecar**

### 估计 LOC

| 模块 | 文件 | 估算 LOC |
|---|---|---|
| factor builder | `examples/_factor_ibs_panel.py` (类比 `_factor_kline_panel.py`) | 80 |
| IS IC script | `examples/factor_ic_ibs_csi300_is.py` (类比 `factor_ic_volume_zscore_is.py`) | 220 |
| Spearman vs amp_imb_20d | inline in IS script | +30 |
| 涨跌停 mask | inline | +20 |
| 输出 CSV + report | inline | +40 |
| **总计** | **2 个新文件** | **~390 LOC** |

实现复杂度: **低**. 完全复用现有 IC pipeline 模板。

### Phase A → Phase B 决策树

```
ibs_raw / ibs_5d / ibs_20d 三 horizon IC ICIR
├── 全 |ICIR| < 0.3 → abort
├── max |ICIR| ∈ [0.3, 0.5] → 单独 sidecar IS sweep (λ ∈ {0.05, 0.10, 0.15})
├── max |ICIR| ∈ [0.5, 1.0] → 单独 sidecar + stack-with-amp_imb 测试 (Spearman 验证后)
└── max |ICIR| > 1.5 → 警惕, λ ≤ 0.10, IS→OOS 严格隔离
```

---

## 安全约束确认

- 本任务**未修改** references/ 任何文件 (全部 Read)
- 本任务**未修改** production code: paper_trade_today.py / forward_oos_monitor.py / portfolio_excel.py / daily_check.sh / strategy_v17_dens_grid.py / strategy_v19_*.py / strategy_v20_*.py / launchd plist 全 untouched
- 本任务仅写 **1 个 docs 文件**
- 不 commit
