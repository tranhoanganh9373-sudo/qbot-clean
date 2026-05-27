# Lean 架构 idea 提取与本项目对照

> Read-only 浅读 `references/Lean/Algorithm.Framework/` 五层抽象 + `Engine/` `Indicators/IndicatorBase.cs` 顶层结构,
> 对比 `examples/paper_trade_today.py` (667 行 production) 与 `examples/strategy_v17_dens_grid.py` (623 行 backtest),
> 输出可借鉴的 idea + 重构成本评估. **不引入任何 .NET 依赖, 不修改 Lean, 不动 production**.
>
> Date: 2026-05-25. Reviewer: claude-opus-4.7 (1M ctx).

---

## 1. Lean 五层抽象总览

Lean 把算法拆成五个 plug-and-play 接口, 主算法只负责 "组装" — 每一层都是独立的 SRP 单元:

```
                    ┌──────────────────────────────────────────┐
   ┌── data ────────►│  Universe Selection                       │
   │                 │  → 何时把哪些 securities 放进/拿出 universe │
   │                 └──────────────┬───────────────────────────┘
   │                                │  (securities_changed events)
   │                                ▼
   │                 ┌──────────────────────────────────────────┐
   │                 │  Alpha Model(s)                           │
   │                 │  update(algorithm, data) → list[Insight]  │
   │                 │  Insight = (symbol, direction, period,    │
   │                 │              magnitude?, confidence?,     │
   │                 │              weight?, source_model)       │
   │                 └──────────────┬───────────────────────────┘
   │                                │  insights
   │                                ▼
   │                 ┌──────────────────────────────────────────┐
   │                 │  Portfolio Construction Model             │
   │                 │  determine_target_percent(active_insights)│
   │                 │  → dict[Insight, target_pct]              │
   │                 │  (equal / insight-weighted / mean-var /   │
   │                 │   risk-parity / black-litterman)          │
   │                 └──────────────┬───────────────────────────┘
   │                                │  PortfolioTarget(symbol, qty)
   │                                ▼
   │                 ┌──────────────────────────────────────────┐
   │                 │  Risk Management Model                    │
   │                 │  manage_risk(algorithm, targets)          │
   │                 │  → risk_adjusted_targets                  │
   │                 │  (MaxDDPercentPortfolio / TrailingStop /   │
   │                 │   MaxSectorExposure)                      │
   │                 └──────────────┬───────────────────────────┘
   │                                │  adjusted targets
   │                                ▼
   │                 ┌──────────────────────────────────────────┐
   │                 │  Execution Model                          │
   │                 │  execute(algorithm, targets)              │
   │                 │  → market_order / limit_order             │
   │                 │  (Immediate / VWAP / StdDev favorable)    │
   │                 └──────────────────────────────────────────┘
   │
   └── Indicator Base (cross-cut): IndicatorBase.update(input)
                                   .is_ready / .current / .window
                                   .Updated event 给上层订阅
```

**关键观察**: 五层之间用 **数据类** (Insight / PortfolioTarget) 解耦, **不互相 import**.
更换 Alpha 不影响 PC, 更换 PC 不影响 Execution. `on_securities_changed(added, removed)` 是统一的
"universe 变化通知" 事件 — 子层可以决定要不要响应.

### 1.1 Insight 数据类 (Lean 跨层契约)

```python
Insight(
    symbol,                         # which security
    period,                         # how long this signal is valid
    type=InsightType.PRICE,         # PRICE / VOLATILITY
    direction=InsightDirection.UP,  # UP / DOWN / FLAT
    magnitude=None,                 # 预期 +/-% (optional)
    confidence=None,                # 0~1 信心 (optional)
    weight=None,                    # PC 直接使用的 portfolio weight (optional)
    source_model=None,              # 哪个 alpha 产生
)
```

Insight 是 **过期型 record** (period 字段决定何时 expire), Lean 内部有 `InsightManager` 维护
"active insights" 列表. PC 永远只看 active 的, 过期自动 cancel. 关键的 alpha-fusion idea:
**多个 AlphaModel 可以并行产生 Insight, PC 拿到的是 union, 用 `source_model` 区分**.

### 1.2 PortfolioTarget (PC → Risk → Execution 跨层契约)

```python
PortfolioTarget(symbol, target_quantity_or_percent, tag=None)
```

Risk model 可以把 target 改成 0 (liquidate), 或新增 target. Execution 只关心
"current quantity vs target quantity 的 delta", 不关心 alpha 来源.

### 1.3 IndicatorBase (横切关注点)

```csharp
abstract class IndicatorBase : IIndicator {
    Name, Samples, IsReady, Current, Previous, Window  // 状态
    event IndicatorUpdatedHandler Updated;             // 订阅
    abstract bool Update(IBaseData input);             // 推送
    abstract void Reset();
}
```

关键 idea: **Indicator 是 push-based stream processor**, 不是函数. 上层 `register_indicator(symbol, indicator, consolidator)`
把 indicator 挂到 symbol 数据流上, indicator 内部维护 `RollingWindow<DataPoint>`, `is_ready` 由 sample count
决定. 多个 indicator 可以 chain (一个 indicator 的 output 作为另一个的 input).

---

## 2. 本项目对照: paper_trade_today.py 的平铺职责

| Lean 层 | paper_trade_today.py 对应代码 | 当前实现方式 |
|---------|------------------------------|--------------|
| **Universe Selection** | `load_universe_names()` L145-153 + `instruments="csi300"` L459 | 静态 CSV 读 + qlib 硬编码 |
| **Alpha Model (main)** | `main()` L437-477 (qlib.init → Alpha158 → DEnsembleModel.fit/predict) | inline 一气呵成 |
| **Alpha Model (sidecar)** | `apply_v19_6_sidecar_overlay()` L392-433<br/>`apply_sidecar_overlay()` L264-309<br/>`load_amp_imb_20d_overlay()` L314-389<br/>`load_margin_overlay()` L210-261 | 4 个 free function, 三段 if/elif 选 sidecar |
| **Portfolio Construction** | L489-531 (candidate_pool → filter → top K)<br/>L610-614 (target/sell/buy/hold diff) | top-K 切片 + 集合差 |
| **Risk Management** | `compute_vol_scale()` L168-188 (vol target, 现 OFF)<br/>L502-528 (涨停/跌停/价格上限过滤) | inline if-skip; vol-target 已废 |
| **Execution** | L626-644 (`append_log` BUY/SELL row 写 CSV) | 直接落 CSV log, 无 broker |
| **Indicators** | qlib Alpha158 (黑盒)<br/>amp_imb_20d / margin_5d_chg (手算 rolling) | groupby/transform/rolling, 一次性算 |
| **State / Persistence** | `load_state/save_state` L156-165 JSON<br/>`append_log` L191-200 CSV | 平铺 read/write |
| **Time / Calendar** | L440-449 (cal[-1] → relativedelta 拆 train/valid/test) | inline 日期算术 |

**问题诊断**:

1. **Sidecar 选择是 if/elif 链** (L481-486). 加 v19.7 / v19.8 要再加一段, scale 性差.
2. **三个 sidecar overlay 函数职责重复**: 都做 "load → fillna 0 → cross-sectional z → 加权减号 → meta dict".
   抽 90% 共享代码到 `apply_zscore_sidecar(score, factor_loader, lambda, sign, name)` 即可.
3. **filter 跟 PC 混在一起** (L511-531): 涨停/跌停/价格上限是 risk-style filter, 但写在
   "candidate selection" 循环里. Lean 把这些分到 Risk Management 层.
4. **paper_trade 和 strategy_v17_dens_grid 主循环不共享代码**: 同样的 sidecar 逻辑要在 backtest
   里复刻一遍才能严格 OOS 验证. 现在 strategy 里没 sidecar (Phase 4 sweep 在独立脚本里跑).
5. **没有 Insight 抽象**: 没法做 "多 alpha 投票" — 比如 train24 + LightGBM 两个独立模型加权.
6. **Risk 跟 Execution 不可独立替换**: 想试 VWAP execution / 想加 max sector exposure 都得改 main().

---

## 3. 值得 borrow 的 idea

### 3.1 Alpha Model 抽象 (推荐)

**Lean 接口**:
```python
class AlphaModel:
    name: str
    def update(self, algorithm, data) -> list[Insight]: ...
    def on_securities_changed(self, algorithm, changes): ...
```

**本项目对应**: paper_trade_today.py L437-477 (qlib + DEnsemble) 是一个"隐式" AlphaModel,
L264-433 三个 sidecar overlay 是隐式的 second-stage alpha.

**重构提议** (Python, 不用 Lean 类型):
```python
@dataclass(frozen=True)
class Signal:
    instrument: str
    score: float           # 原始 alpha 分数 (任意 scale)
    z_score: float = None  # 横截面 z (PC 用)
    source: str = ""       # 'train24' / 'amp_imb_20d' / 'margin_5d'
    asof: pd.Timestamp = None

class AlphaModel(Protocol):
    name: str
    def generate(self, asof: pd.Timestamp, universe: list[str]) -> list[Signal]: ...

class QlibDEnsembleAlpha(AlphaModel):
    """L437-477 包装"""
    def __init__(self, train_months=24, dens_params=DENS_PARAMS): ...
    def generate(self, asof, universe) -> list[Signal]: ...

class AmpImbSidecarAlpha(AlphaModel):
    """L314-389 包装"""
    def __init__(self, window=20, sign=-1): ...
    def generate(self, asof, universe) -> list[Signal]: ...
```

**收益**: 加新 sidecar (v19.7 unlock / v19.8 industry) 只是 `alphas.append(NewSidecar())`,
不动 main loop. backtest 和 paper_trade 共享同一个 `AlphaModel` 实例 → 严格 OOS 协议天然不破.

**成本**: **medium** — 抽 4 个类 + 改 main, ~2-3 小时. 现有 4 个 overlay 函数共享代码量 70%+,
抽象天然契合.

**风险**: production 信号字节不变 (z-score 数学等价), 但函数签名变了, 必须用 paper_trade dry-run 跟
shadow 对比 picks 100% 重合再切换. 旧 `.bak_pre_v19_6` 保留.

### 3.2 Sidecar 加权融合统一 (强推, 几乎零风险)

**Lean idea**: `InsightWeightingPortfolioConstructionModel.determine_target_percent()` —
多个 insight 来源,用 `insight.weight` 字段直接加权汇总, weight_sum > 1 时按比例缩放.

**本项目对应**: 现在 L429 `out["final_score"] = out["pred_z"] - λ * out["amp_z"]` 硬编码减号.
v19.7 stacked sidecar 把 λ_m5/λ_a20 写成两行加加减减.

**重构提议**:
```python
@dataclass(frozen=True)
class SidecarSpec:
    name: str              # 'amp_imb_20d'
    sign: int              # +1 / -1
    lam: float             # 0.30
    loader: Callable[[pd.Timestamp], dict[str, float]]
    stale_days: int = 7

def combine_signals(
    base_pred: pd.Series,                  # z(pred), index=instrument
    sidecars: list[SidecarSpec],
    asof: pd.Timestamp,
) -> tuple[pd.Series, list[dict]]:
    """final = base_pred + Σ sign_i * lam_i * z_i. 缺数据填中性 0."""
    final = base_pred.copy()
    meta = []
    for spec in sidecars:
        fac_map = spec.loader(asof)
        status = check_freshness(fac_map, asof, spec.stale_days)
        if not status.ok:
            meta.append({"name": spec.name, "applied": False, "reason": status.reason})
            continue
        z = cross_sectional_z(base_pred.index.map(fac_map))  # fillna(0) 中性
        final = final + spec.sign * spec.lam * z
        meta.append({"name": spec.name, "applied": True, "n_covered": z.notna().sum()})
    return final, meta
```

**收益**: 切换 v19.4 / v19.6 / v19.7 stacked / v19.8 unlock 变成 **数据驱动 list comprehension**,
不再有 if/elif 三档. 严格 OOS 协议天然支持 — backtest 跟 paper_trade 用 **同一个 `combine_signals`**.

**成本**: **small** — 1.5 小时. 共享 90%+ 现有逻辑 (cross-sectional z / fillna(0) / freshness check).

**关键约束**: λ 必须 **锁定** (来自 IS sweep 结果, 不能跑时 sweep), `SidecarSpec` 不写默认值,
强制构造时给出. 现有 memory 锁定:
- v19.6 prod: `SidecarSpec("amp_imb_20d", sign=-1, lam=0.30, ...)`
- v19.4 shadow: `SidecarSpec("margin_5d_chg", -1, 0.10) + SidecarSpec("margin_20d_chg", -1, 0.10)`

### 3.3 PortfolioTarget 数据类 + filter 分层 (推荐)

**Lean idea**: PC 输出 `list[PortfolioTarget]`, **没有过滤逻辑**. Risk model 在中间层把不该买的
`PortfolioTarget(symbol, 0)` 改成 liquidate, 或直接从列表里去掉. Execution 只关心 delta.

**本项目对应**: L511-531 在 PC 内部 inline 做涨停/跌停/MAX_AFFORDABLE_PRICE 过滤 + 集合差.

**重构提议**:
```python
@dataclass(frozen=True)
class PortfolioTarget:
    symbol: str
    target_pct: float  # 1.0 = 满仓单股 / 0 = liquidate
    tag: str = ""      # 'buy' / 'hold' / 'sell' / 'skip-limit-up' / 'skip-price-cap'

class PortfolioConstructor(Protocol):
    def build(self, signals: pd.Series, asof) -> list[PortfolioTarget]: ...

class TopKEqualWeight(PortfolioConstructor):
    """现有 L489-531 干净版"""
    def __init__(self, k=8): ...
    def build(self, signals, asof) -> list[PortfolioTarget]:
        top = signals.nlargest(self.k * 4).index.tolist()  # pool
        return [PortfolioTarget(s, 1.0/self.k) for s in top]

class LimitUpFilter:
    """Risk-style filter"""
    def __init__(self, threshold=0.095, threshold_high=0.195): ...
    def filter(self, targets, asof, prices_2d) -> list[PortfolioTarget]: ...

class PriceCapFilter:
    """MAX_AFFORDABLE_PRICE filter"""
    def filter(self, targets, prices_today) -> list[PortfolioTarget]: ...
```

**收益**: 加 max_sector_exposure / max_drawdown 风控只是 append 一个 filter, 不动主循环.
backtest 跟 paper_trade 共享 filter, 严格一致.

**成本**: **medium** — 2 小时. 当前 filter 逻辑分散在 L502-528 + L611-614, 抽出来后 main 主循环
减约 50 行.

### 3.4 Risk Management — MaximumDrawdownPercentPortfolio 借鉴 (可选)

**Lean idea**: 算法跑过程中, portfolio.total_portfolio_value 触及 -X% 时, **所有 targets 改成 0** (全 liquidate).
trailing 模式从历史 max 起算, 非 trailing 从 starting value 起算.

**本项目对应**: 当前没有 portfolio-level drawdown protection. v19.6 OOS MDD -18.51% (可接受),
但 v19.4 MDD -21.23%, v19.1 MDD -29.86%. 实盘期间若触及 -20% 该如何反应?

**重构提议** (作为可选 module, 非默认开启):
```python
class MaxDrawdownGuard:
    """从 portfolio_state.json 历史读 high water mark, 触发后写 'liquidate' 信号."""
    def __init__(self, max_dd=0.20, mode="trailing"): ...
    def check(self, current_value: float, history: list[float]) -> list[PortfolioTarget]:
        if (current_value / max(history) - 1) <= -self.max_dd:
            return [PortfolioTarget(s, 0, tag="risk-liquidate") for s in self.holdings]
        return []
```

**成本**: **medium** — 2 小时, 需要 portfolio_state.json schema 加 `nav_history` 列. **建议 follow-up,
不进 v20**, 因为现在没有 NAV 真实数据 (paper_trade 没算 NAV, 只 log signals).

### 3.5 Indicator base — RollingWindow + is_ready (谨慎借鉴)

**Lean idea**: 每个 indicator 内部维护 `RollingWindow<IndicatorDataPoint>`, `is_ready` 由 sample count 决定,
`Updated` event 给上层 chain.

**本项目对应**: 现在用 `pandas.groupby().transform(lambda s: s.rolling(N).sum())` (L362-370).
一次性算所有历史, 不是 push-based.

**评估**: **不建议借鉴 push-based 模式**. pandas vectorized rolling 在 daily frequency + CSI300 universe
(300 股 × 30 天 = 9000 行) 上 1ms 级, push-based reactive 反而引入复杂度. **保持现有 pandas 风格**.

**唯一可借鉴**: `is_ready` 概念 — 当前 `KLINE_STALE_DAYS` / `MARGIN_STALE_DAYS` 已经是这个语义,
但散落在两个 loader 里. 可以抽个小工具:
```python
def is_factor_ready(factor_data, asof, max_stale_days, min_coverage_pct=0.3):
    """返回 (ok: bool, status: str). status 用于 meta log."""
```

**成本**: **small** — 30 分钟, 纯重命名 + 抽取.

---

## 4. 不建议 borrow

| Lean 组件 | 不借鉴原因 |
|----------|------------|
| **`.NET event system` (`event IndicatorUpdatedHandler`)** | C# 特有, Python 没必要. observable pattern (rxpy) 在 daily 频率纯增加复杂度. |
| **`IDataReader` / `Consolidator` / `Resolution` 框架** | Lean 处理 tick/minute/daily 多分辨率, 本项目纯 daily. qlib `D.features` 已足够. |
| **`AlgorithmManager.cs` / `Engine.cs` 主循环** | Lean 主循环跑 backtest + live trading 共用代码, 本项目用 cron + launchd plist 跑 daily script — 更简单. |
| **`Brokerages/` 真实下单接口** | 本项目是 paper_trade 信号, 不下真单. |
| **`Optimizer/` (genetic / particle swarm hyperparam sweep)** | qlib 已有 + 严格 OOS 协议禁止 sweep test 期参数. |
| **`Report/` 自带 PDF report** | 已有 pyfolio + alphalens (project_trading_costs_pyfolio_alphalens.md). |
| **`Algorithm.Framework/Selection/CoarseFundamentalUniverse`** | A 股没有 Lean 那种 coarse fundamental 数据源, 用 universe.csv 静态足够. |
| **`MeanVarianceOptimizationPortfolioConstructionModel`** | 需要协方差矩阵估计, K=8 等权简单稳定, 优化反而 overfitting. |
| **`BlackLittermanOptimizationPortfolioConstructionModel`** | 需要 market equilibrium prior, A 股 CSI300 难定义合适 prior. |

---

## 5. 优先级清单

| Idea | 价值 | 成本 | 风险 | 优先级 | 顺序 |
|------|------|------|------|--------|------|
| **3.2 Sidecar 加权融合 (SidecarSpec + combine_signals)** | HIGH — sidecar 实验加速 5×, backtest/paper 严格一致 | small (1.5h) | LOW — 数学等价, dry-run picks 可比对 | **P0** | 1 |
| **3.5 `is_factor_ready` 小工具抽取** | MEDIUM — 减少 freshness 散落 | small (0.5h) | NONE — 纯重命名 | **P1** | 2 |
| **3.1 AlphaModel 抽象 (Signal dataclass)** | HIGH — 多 alpha fusion 路径打开 | medium (2-3h) | MEDIUM — main loop 重排, picks 必须 100% reproducible | **P1** | 3 |
| **3.3 PortfolioTarget + filter 分层** | MEDIUM — 主循环减 50 行, sector/MDD 风控路径开 | medium (2h) | MEDIUM — filter 顺序敏感 (涨停 < 价格 < holdings diff) | **P2** | 4 |
| **3.4 MaxDrawdownGuard** | LOW (短期, 没 NAV) → HIGH (长期) | medium (2h, 需 schema 扩) | LOW — opt-in | **P3** | 5, follow-up |

**建议节奏**:
1. **本周**: 做 P0 (3.2) — 抽 `combine_signals(base_pred, sidecars)`, 用 v19.6 单 sidecar 跑 dry-run 对照
   `examples/paper_trade_today.py` 当前输出 picks **必须完全相同**.
2. **下周**: 做 P1 (3.5 + 3.1) — Signal dataclass + AlphaModel Protocol, 把 qlib + 4 个 sidecar 改成
   `list[AlphaModel]`. backtest (strategy_v17_dens_grid.py) 跟 paper_trade 共享 alpha list.
3. **再下周**: P2 (3.3) — PortfolioConstructor + Filter chain. 这一步打开 v20 风控空间 (sector / NAV / MDD).
4. **Follow-up**: P3 (3.4) — 等 NAV 真实数据有了之后做.

---

## 6. 严格 OOS 协议保护 (重构 backtest engine 时必守)

**重构过程中绝对不能破坏的约束** (来自 memory `strict_oos_backtest`):

1. **时间隔离不可破**:
   - factor 选择 / λ sweep / hyperparam tuning 必须在 IS 期 (2014-01 ~ 2020-12) 内做完;
   - OOS 期 (2021-05 ~ 2026-04) 只能跑 **一次** "锁定 λ 评估". 重构后跑 OOS 必须用 v19.6 锁定值 λ_amp20=0.30.

2. **AlphaModel / SidecarSpec 不能在 OOS 期内调整参数**:
   - 抽象后 `SidecarSpec` 的 `lam` 必须是 `final=True` (用 `@dataclass(frozen=True)`),
   - 任何调用方在 OOS 跑中尝试改 lam 应该报错 (而不是静默改).

3. **backtest 和 paper_trade 必须用同一个 alpha 实例**:
   - 重构后 `examples/strategy_v17_dens_grid.py` 应 import `paper_trade_today` 的 AlphaModel 类
     (反过来也行, 总之单一定义源).
   - 这是这次重构最大的收益: **memory 里多次提到 "backtest 跟 paper_trade 没共享主循环"** —
     重构后强制共享.

4. **walk-forward 月度结构不变**:
   - `strategy_v17_dens_grid.py` 的 `realistic_window()` 现在是 IS 训练 → 单月 OOS 测试 → 滚动. 抽象后
     主循环结构应保持: 每月调一次 `alpha.fit_or_load(train_window)` → 调一次 `alpha.predict(test_month)`.

5. **训练-验证-测试不能重叠**:
   - paper_trade L444-449: `train: T-25m ~ T-1m`, `valid: T-1m ~ T-1m`, `test: T-7d ~ T`.
   - 重构后 `AlphaModel.fit()` 接受 `(train_start, train_end, valid_start, valid_end)` 显式参数,
     调用方必须保证 `valid_end < test_start`. 用 assertion 强制.

6. **dry-run picks 对比关卡**:
   - 每次重构, 跑 `python examples/paper_trade_today.py --dry-run` 跟 重构前 baseline 对照 picks list.
   - 重构前 baseline 已记录: 2026-05-22 = `[SH600039, SH688396, SH603993, SZ001965, SH601939, SZ300498, SH600018]`
     (memory `project_v19_6_production_upgrade.md`).
   - 重构后必须 100% 重合. 不重合就回滚.

7. **不动文件 (production 锁)**:
   - `examples/paper_trade_today.py` 改造前必须先 `cp paper_trade_today.py paper_trade_today.v19_6.pre_refactor.bak`.
   - `examples/strategy_v17_dens_grid.py` 同样备份.
   - `examples/forward_oos_monitor.py` / `examples/data_sanity_check.py` / `examples/data_completeness_check.py`
     / `examples/daily_check.sh` / launchd plist **绝不动**.

8. **新增模块走 `src/claude_finance/lean_style/`**:
   - 不污染 `examples/` (那是 entry point).
   - 在 `src/claude_finance/lean_style/alpha.py` / `portfolio.py` / `risk.py` 写新抽象,
     `examples/paper_trade_today.py` 重构后 import 这些类, 但 entry point 依然在 `examples/`.

---

## 7. 一句话总结

Lean 的核心价值不是 "更好的 backtest", 而是 **Alpha / Universe / PC / Risk / Execution 五层用数据类
(Insight / PortfolioTarget) 解耦** — 这套 idea 用纯 Python (Protocol + dataclass) 就能落地, 不需要 .NET.

本项目最痛的两个点 — **sidecar 切换写成 if/elif** 和 **backtest 跟 paper_trade 主循环不共享** — 都是
缺少这层抽象的直接症状. P0 (sidecar 加权融合) 修第一个, P1 (AlphaModel 抽象) 修第二个, 8 小时内完成,
严格 OOS 协议天然保留 (因为重构是 **接口提取**, 数学等价).
