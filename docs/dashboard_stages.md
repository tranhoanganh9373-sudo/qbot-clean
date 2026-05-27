# Dashboard Stages — Lean template + matplotlib base64

> - **Stage 1** (ship): 3 panel + Stage 2/3 占位.
> - **Stage 2** (ship): 加 per-stock 因子贡献 stacked bar (chart 4).
> - **Stage 3 partial** (ship): v19.6 main vs v19.4 shadow A/B 对比 (section 5).
> - **Stage 3 cron monitor** (ship): Daily Check Status panel (section 12).
> - **Stage 4 dashboard expand** (本次 ship): 4 个新 section (A/B/C/D):
>   - Section A — Panel 0 KPI Summary (顶部 6 卡片汇总, `dashboard/charts/kpi_summary.py`).
>   - Section B — Panel 13 Universe Expansion Progress (CSI300/CSI500/top1500 bar + coverage, `dashboard/charts/universe_progress.py`).
>   - Section C — Section 1 Forward OOS 增强 (`< 3` 月 fallback + 1/3/6/12 月 milestone timeline; `≥ 3/6/12` 月自动加 monthly bar / rolling Sharpe / heatmap; 备份 `forward_oos_curve.py.bak`).
>   - Section D — Panel 14 Daily Picks Rotation (yesterday vs today Venn + BUY/HOLD/SELL diff, `dashboard/charts/picks_rotation.py`, fallback from `paper_trade_log.csv` when `portfolio_state.json.bak` 不存在).
> - **Stage 4 E** (本次 ship): Panel 15 Picks Score Distribution (全 universe z_pred histogram + Top K 红线 + Top-(K+1) cutoff 绿虚线, `dashboard/charts/picks_score_distribution.py`, 读 `data_cache/picks_today.json::full_distribution` — paper_trade_today.py +11 LOC dump).
> - **Stage 5 i18n + Glossary** (本次 ship): dashboard 中文化 + 金融术语速查表:
>   - 新增 `dashboard/charts/glossary.py` (~125 LOC) — 29 条术语 (PnL/Calmar/Sharpe/MDD/ann/cum/OOS/IS/Walk-forward/Alpha/Sidecar/Picks/Universe/IC/ICIR/Spearman/z-score/CSI300/CSI500/Margin/Limit-up·down/qfq·hfq/DEnsemble/Alpha158/TopK/Phase A/Phase B/v19.6·v19.4·baseline/amp_imb_20d), 2-col grid 显示;
>   - 占位 `{{glossary}}` 加在 Today tab **最顶** (KPI Summary 之前), `dashboard/render_report.py` import + try/except + replace 三处插桩;
>   - 17 个 section title 全加中文 (`English · 中文` 格式), 每 `<h2>` 同时含 `title="..."` hover tooltip 给出深一层解读.
> - **Stage 6 user-input editable form + auto-merge** (本次 ship): portfolio.xlsx 手填字段移到 dashboard inline form:
>   - **Positions panel** 改成 editable inline table — 每行 5 个 `<input data-user-input data-sym=... data-field=...>` 覆盖 5 个手填列:
>     - 实买价 (`buy_price`, `type=number step=0.01`)
>     - 实买数 (`buy_shares`, `type=number step=100`)
>     - 备注 (`note`, `type=text maxlength=40`)
>     - 卖出价 (`sell_price`, `type=number step=0.01`)
>     - 卖出日期 (`sell_date`, `type=date`).
>     CSS: input 无 border, hover/focus 高亮; 数字右对齐.
>   - **Notes panel** (Today tab 底部, collapsible default 折叠) — 自由 `<textarea id=notes-textarea>` 用户笔记.
>   - **localStorage 持久化**: key `cf_user_inputs_v1`, schema `{ "SH600547__buy_price": "29.945", ..., "__notes": "free text", "__exported_at": "ISO ts" }`. JS 监听 `blur`/`change` → 自动写; 页面 load 时回填 (覆盖 server-rendered initial value).
>   - **导出按钮** "💾 导出 user_input.json" — 把 localStorage + textarea 内容打包成 JSON 触发 `<a download>`, 用户重命名为 `user_input.json` 放到 `data_cache/`.
>   - **新建 `examples/merge_user_input.py`** (~250 LOC) — 读 `data_cache/user_input.json` → openpyxl merge 到 `portfolio.xlsx`:
>     - Positions: 只 update 现存 row 的 5 列, 空值跳过 (不覆盖 server 值), 未知 sym/field 跳过.
>     - Notes: 在固定 NOTES_TEXT 之后插入 marker `═══ User Notes (from dashboard) ═══` + 用户文本; 重 merge 覆盖 marker 之后, 不动 marker 之前的 psychology 参考文本.
>     - 类型强制: `buy_price/sell_price` → float, `buy_shares` → int, `sell_date` → `YYYY-MM-DD` 字符串.
>     - 失败模式: missing json → ok=True noop; corrupt json / missing xlsx → ok=False + warn print, 不抛.
>     - Idempotent (多次跑同 user_input.json 结果一致, marker 只出现一次).
>   - **portfolio_excel.py `main()` 末** try-call `merge_user_input.run_merge(verbose=True)` — daily_check.sh 不改, step 4 `portfolio_excel.py` 跑完后自动 merge. 失败 print warn, 不阻塞.
>   - **新建 `dashboard/charts/user_notes.py`** (~95 LOC) — 渲染 Notes textarea panel, 读 xlsx Notes sheet marker 之后的文本作为 textarea 初值 (server-side source of truth).
>   - **Tests** `tests/test_merge_user_input.py` (16 cases, 全 PASS) — 覆盖 happy path / unknown sym / unknown field / empty string skip / invalid number / sell_date str / int coercion / notes append / idempotent re-merge / dry-run / corrupt JSON / notes overwrite / all-5-fields / missing xlsx / non-dict JSON.
>   - HTML 增量: positions panel 旧 read-only `<table>` 改成 editable `<table>` + export bar + Notes collapsible panel; template.html 末加 ~95 行 JS (load/save/export).
>   - **不动**: daily_check.sh, launchd plist, paper_trade_today.py 选股逻辑, portfolio.xlsx 现有 schema (NOTES_TEXT 参考段 / Positions 21 列全保留), data_cache 主存储.
>   - **用户流程**: dashboard 编辑 → 自动 localStorage → 点 "💾 导出" 下载 JSON → 重命名 `user_input.json` 放 `data_cache/` → 下次 16:30 daily_check 跑 portfolio_excel.py 自动 merge 进 xlsx → dashboard 刷新看到 server 端 source of truth.
> - **Stage 3 剩余**: multi-agent debate transcript (待推理框架数据源).

Stage 1/2 实现一个 self-contained 单文件 HTML 日报, 模式借鉴 `references/Lean/Report/template.html`:
matplotlib fig → base64 PNG → inline `<img src="data:image/png;base64,...">`, 浏览器双击即可.

## 启动

```bash
# 默认今日
.venv/bin/python dashboard/render_report.py

# 指定日期
.venv/bin/python dashboard/render_report.py --date 2026-05-26

# 指定输出路径
.venv/bin/python dashboard/render_report.py --out reports/test.html
```

输出: `reports/daily_report_YYYYMMDD.html` (默认 9 KB ~ 数 MB, 视 chart 数据量).

## 文件结构

```
dashboard/
  __init__.py
  render_report.py              # 入口 (~130 LOC)
  template.html                 # Lean 改造后的 HTML 模板 (~360 LOC, Stage 4 4-tab layout)
  charts/
    __init__.py
    kpi_summary.py              # panel 0 (Stage 4 A): 顶部 6-card KPI 纵览
    forward_oos_curve.py        # panel 1: Forward OOS cum + drawdown (Stage 4 C 增强)
    forward_oos_curve.py.bak    # panel 1 Stage 3 备份 (Stage 4 C 修改前)
    positions_pnl.py            # panel 2: 持仓 PnL HTML 表
    factor_config.py            # panel 3: sidecar config panel
    factor_contrib.py           # panel 4 (Stage 2): per-stock 因子贡献 stacked bar
    ab_v19_6_vs_v19_4.py        # panel 5 (Stage 3): v19.6 vs v19.4 shadow A/B
    model_leaderboard.py        # panel 6
    factor_ic_heatmap.py        # panel 7: Factor IC Heatmap (Phase A 月度)
    sector_breakdown.py         # panel 8: Holdings sector breakdown (HHI + pie)
    is_oos_gap_scatter.py       # panel 9: Per-factor IS→OOS Calmar gap scatter
    factor_coverage_health.py   # panel 10: Factor coverage health (8 sources)
    trade_history_timeline.py   # panel 11: Trade history timeline (paper_trade_log)
    daily_check_status.py       # panel 12: Daily check cron monitor (log + launchd)
    universe_progress.py        # panel 13 (Stage 4 B): CSI300/500/top1500 bar + coverage
    picks_rotation.py           # panel 14 (Stage 4 D): yesterday vs today Venn + diff
    picks_score_distribution.py # panel 15 (Stage 4 E): 全 universe z_pred histogram + Top K 高亮
    glossary.py                 # panel 0' (Stage 5): 金融术语速查表 — 29 术语 2-col grid (Today tab 顶)
    user_notes.py               # panel 16 (Stage 6): 用户笔记 textarea (Today tab 底, collapsible)
  utils/
    __init__.py
    fig_to_base64.py            # Lean fig_to_base64 复刻 (无 .NET 依赖)
reports/                        # 输出目录 (gitignore 已含 reports/)
  daily_report_YYYYMMDD.html
```

## 3 个 Section 解读

### 1. Forward OOS Cumulative + Drawdown
- **数据源**: `data_cache/portfolio.xlsx` → sheet `Forward OOS Track` (header=1).
  Columns: `month`, `month_end_date`, `cum_return` (decimal), `alert_level`, etc.
- **行为**:
  - 0 月 → "尚无数据" 提示.
  - 1-2 月 → fallback "积累中 X 月 / 3 月" + 当前 cum_return + alert_level.
  - ≥ 3 月 → matplotlib 双子图: 上 cumulative return %, 下 drawdown %.
- **当前状态 (2026-05-26)**: 1 月数据, 显示 fallback.

### 2. Current Positions PnL
- **数据源**: `data_cache/portfolio.xlsx` → sheet `Positions`.
- **输出**: HTML table, 8 列 (Code / Name / Status / Buy Price / Shares / Last Price / PnL % / PnL ¥).
- **Conditional formatting**:
  - 浮盈% > 0 → `.pnl-pos` (绿色)
  - 浮盈% < 0 → `.pnl-neg` (红色)
  - NaN / 0 → `.pnl-flat` (灰色)
- 表底 summary 行: "持仓 N 行 / 已平 M 行 / 共 K 行".

### 3. Sidecar Factor Config
- **数据源**: `examples/paper_trade_today.py`, 用 `ast.parse` 抓 module-level 常量.
- **解析的常量**:
  - `USE_V19_6_SIDECAR` (bool)
  - `USE_V19_4_SIDECAR` (bool)
  - `SIDECAR_LAMBDA_AMP_20D` (float, default 0.30)
  - `SIDECAR_LAMBDA_M5`, `SIDECAR_LAMBDA_M20` (float, default 0.10)
- **输出**: 3 行表 (v19.6 / v19.4 / v19.1 baseline) + active 行高亮 + 当前 final_score 公式.

### 4. Per-Stock 因子贡献 Stacked Bar (Stage 2)
- **数据源** (read-only 重算, 不改 production):
  - `data_cache/v17_dens_train24_predictions.parquet` (取最新月 cross-section)
  - `data_cache/baidu_kline.parquet` (通过 `paper_trade_today.load_amp_imb_20d_overlay`)
  - `examples/paper_trade_today.py` (importlib 取 `SIDECAR_LAMBDA_AMP_20D` + 复用 overlay 函数)
- **公式** (与 production v19.6 完全一致):
  ```
  z_pred  = cross-sectional z-score of pred score
  z_amp   = cross-sectional z-score of amp_imb_20d (NaN 填 mean)
  sidecar = -SIDECAR_LAMBDA_AMP_20D × z_amp  (λ = 0.30)
  final   = z_pred + sidecar
  Top 8 = 按 final desc 排序前 8
  ```
- **输出**: matplotlib stacked bar (蓝色 z_pred 底段 + 红/绿 sidecar 顶段) + Top 8 sparkline grid + 详细表 + 解读文字.
  - 绿色顶段 = sidecar 推高 (amp_imb_20d 低于均值)
  - 红色顶段 = sidecar 压低 (amp_imb_20d 高于均值, 振幅过强 → 反转减分)
  - 红绿段越长 = 越是真正的 sidecar pick
- **Top 8 21 日 sparkline grid** (visual 补充):
  - 2×4 grid mini line charts, 每股最近 21 个交易日 close 走势.
  - title 显示 sym + 21d 涨跌幅 % (绿涨 / 红跌).
  - 数据源: `dashboard/utils/kline_fast.get_stock_kline` (Hive 分区 0.4 ms/股 path-read,
    `data_cache/baidu_kline_hive/code=XXXXXX/*.parquet`).
  - 单股 fallback: Hive 无分区或 <2 bars → 该 cell 显示 `no data` 占位, 不 crash.
  - 全 8 股 fallback: 全部 no data → sparkline 整块不渲染.
- **注意**: predictions cache 是月度落盘 (e.g. 2026-04-30), 而 production paper_trade_today.py
  每日实时重训 DEnsemble 模型. Dashboard 用 cache 是 **read-only 近似**, 与 production
  当日 picks 可能不完全重合, 但 sidecar 公式 / λ / amp_imb_20d 计算逻辑完全一致 →
  因子贡献分解结论可靠.

### 5. v19.6 main vs v19.4 shadow A/B (Stage 3 partial — ship)
**实现**: `dashboard/charts/ab_v19_6_vs_v19_4.py:build_ab_section` (~350 LOC).
- **数据源** (全只读): `data_cache/portfolio_state.json` (v19.6 main holdings) +
  `data_cache/portfolio_state_v19_4.json` (v19.4 shadow) +
  `data_cache/paper_trade_log.csv` + `data_cache/paper_trade_log_v19_4.csv`.
- **Venn diagram**: 手工 2-圆 matplotlib (无 matplotlib_venn 依赖), 显示
  v19.6 only / 重叠 / v19.4 only 三块数字, 上方 label 给两 strategy 的 holdings 总数.
- **Symbols 分类表**: 3 列 (v19.6 only / overlap / v19.4 only), 列出实际 ticker.
- **State meta**: 显示两 state 的 as_of 日期 + holdings 总数.
- **双 cum return 线图**: 等权重 close-to-close MTM (kline_fast Hive 0.4 ms/股),
  v19.6 蓝 / v19.4 橙. 累积 < 5 交易日时 fallback 显示 "积累中 N / 5 交易日".
- **当前状态 (2026-05-26)**: 2 / 5 trading days, 进入 fallback. Venn 数字
  v19.6=7 / overlap=0 / v19.4=2 (两 strategy 完全不重叠 — v19.6 picks
  SH600039/SH600426/SH600547/SH603993/SH688036/SZ002493/SZ300347,
  v19.4 picks SH688187/SH688396).

### 7. Factor IC Heatmap (Phase A 月度)
**实现**: `dashboard/charts/factor_ic_heatmap.py:build_factor_ic_heatmap_section` (~330 LOC).

- **数据源** (全只读 `examples/*_is_monthly.csv`, 8 个 csv):
  - `factor_ic_ibs_csi300_is_monthly.csv` (`factor_signed,month_end,ic`)
  - `factor_ic_shareholders_is_monthly.csv` (`factor,asof_date,ic`)
  - `factor_ic_industry_adj_ret_is_monthly.csv` (`factor,asof_date,ic`)
  - `v20_volume_zscore_is_monthly.csv` (`variant,month_start,ic`)
  - `super_big_net_is_monthly.csv` (`factor_signed,month_end,ic`)
  - `factor_ic_technical_csi300_is_monthly.csv` (`factor,month_start,ic`)
  - `factor_ic_unlock_csi300_is_monthly.csv` (`factor,asof_date,ic`)
  - `factor_ic_fundamentals_csi300_is_monthly.csv` (`factor,month_start,ic`)
- **配置表 `MONTHLY_CSVS`**: 11 个 Phase A 候选 factor × `(label, csv, factor_col,
  factor_name, month_col, sign)` 显式 spec, 兼容 csv schema 不统一 (factor 列名 /
  月份列名 / sign 内嵌方式 都不同).
- **可视化**: `matplotlib.imshow` heatmap, RdYlGn diverging cmap, vmin/vmax = ±0.2;
  绿 = signed alignment 后 IC > 0 (因子方向正确), 红 = 反向, NaN 浅灰.
- **regime shift 黑色虚线**: COVID (2020-03) / OOS 起 (2021-05) / 注册制扩 (2023-01) /
  政策市 (2024-09); 当前 csv 覆盖 2014-01 ~ 2020-12 仅 COVID 命中, OOS 段需 Phase B
  数据补齐.
- **CJK 字体**: 模块 import 时调 `_ensure_cjk_font()` 把 PingFang/Heiti 等 prepend 到
  `font.sans-serif`, 防中文方框 (其它 chart 共享同 rcParams).
- **三层 HTML**: chart `<img>` + 4 条解读 `<ol>` + 11 行 IC mean 降序 summary
  `<table.data>` (用 template 自带 `var(--green)` / `var(--red)`).
- **当前数据规模 (2026-05-26)**: 11 factors × 84 months (2014-01 → 2020-12),
  最强 signed IC mean = `count_change_12m (-)` +0.056.

### 8. Holdings Sector Breakdown
**实现**: `dashboard/charts/sector_breakdown.py:build_sector_breakdown_section` (~230 LOC).

- **数据源** (全只读):
  - `data_cache/portfolio_state.json` — `holdings` list (当前 7 只).
  - `data_cache/industry/industry_membership.parquet` — SW level-1 (5203 行, 31 个行业).
    `code` 6 位无前缀 → 内部 `_code_to_sym` 加 `SH/SZ` 前缀对齐 holdings 符号.
  - `data_cache/portfolio.xlsx` sheet `Positions` (col 10=实际买入价 / 11=实际买入数) —
    有实买数据的股按 (price × shares) 真实市值, 缺则等权 fallback (默认池 ¥50,000 / N).
- **HHI** (Herfindahl-Hirschman Index): `Σ(pct_i)² / 10000`, range [0, 1].
  - `< 0.18` 绿色 "低集中"
  - `0.18 – 0.25` 橙色 "中集中"
  - `> 0.25` 红色 "高集中风险"
- **单行业警示**: > 50% 红色, > 30% 橙色, 否则继承字色.
- **可视化**: matplotlib 双子图 — 左 pie (按行业 %, Set3 配色), 右 barh (个股权重, 颜色与 pie 行业匹配).
  CJK 字体由 `_ensure_cjk_font()` 注入 (与 section 7 同候选链).
- **HTML 三层**: `<img>` (base64 PNG) + HHI 摘要 div + 行业权重 `<table.data>` 降序.
- **当前状态 (2026-05-26)**: HHI = 0.178 (临界低集中), 6 个行业, 有色金属 26.2% 最重
  (SH603993 + SH600547), 医药生物 16.8% (SZ300347), 其余基础化工/建筑装饰/电子/石油石化
  各 14.3%. 2/7 持仓用 Positions 实际买入市值 (SH600547 ¥5,989 + SZ300347 ¥8,407), 余 5 等权.

### 9. Per-Factor IS → OOS Calmar Gap Scatter
**实现**: `dashboard/charts/is_oos_gap_scatter.py:build_is_oos_scatter_section` (~180 LOC).

- **数据源** (硬编码): 10 行 `SCATTER_DATA`, 与 `model_leaderboard.MODEL_DATA` 中
  `is_calmar` 非 None 的子集严格一致. baseline (train24 纯) 无 IS sidecar → 不入 scatter.
- **可视化**: matplotlib scatter, x=IS Calmar, y=OOS Calmar, 颜色=tag (与 section 6
  `TAG_COLORS` 同源), 文本 label=factor name. 设 `aspect=equal` 保 45° 视觉准确.
- **参考线 / 阈值线**:
  - 黑色虚线 y=x (45° 真 alpha 边界): 上方 = OOS > IS = 真 alpha, 下方 = overfit.
  - 红色点线 y=0.5 (OOS abort line).
  - 蓝色点线 y=0.77 (baseline OOS).
- **解读 (Phase B 教训视觉化)**:
  - 45° 线上方 (真 alpha): **2 个** — v19.6 (0.79→1.29), v19.7 (0.76→0.96).
  - 45° 线下方 (overfit): **8 个** — 大部分 Phase B 失败案例.
  - 右下角 catastrophic 区 (IS > 2 + OOS < 0.5): **4 例 confirmed** —
    shareholders (6.09→0.39), unlock (2.70→0.09), super_big_net (2.01→-0.07),
    vol_z_5d (3.39→0.32). industry_60d (2.95→0.65) 跨过 OOS 0.5 但仍假 alpha.
- **QuantaAlpha 教训内嵌**: `n_months < 60 + IS Calmar > 1.5 → OOS 衰减 -90%+`.

### 10. Factor Coverage Health
**实现**: `dashboard/charts/factor_coverage_health.py:build_factor_coverage_health_section` (~340 LOC).

- **数据源** (全部只读, 只 read_parquet 不写):
  1. `data_cache/baidu_kline.parquet` (全 A 股 universe, latest-day cover)
  2. `data_cache/baidu_kline.parquet` ∩ `data_cache/csi300_constituents.csv`
     (kline ∩ CSI300 latest-day, 这是 v19.6 sidecar 的实际 universe — silent
     bug 重灾区)
  3. `data_cache/csi300_margin_14yr.parquet` (CSI300 universe, latest-day cover)
  4. `data_cache/fund_flow/fund_flow_csi300.parquet` (CSI300, latest-day cover)
  5. `data_cache/shareholders/shareholders_csi300.parquet` (CSI300,
     event-frequency → any-data cover)
  6. `data_cache/industry/industry_membership.parquet` (snapshot 表 → any-data)
  7. `data_cache/unlock/unlock_detail_em.parquet` (event-frequency → any-data)
  8. `data_cache/dragon_tiger/*.parquet` (per-stock 目录, latest-day from
     抽样 30 个文件 freshness)

- **指标 / 阈值**:
  - 🟢 ok: cover ≥ 90% AND stale ≤ 3d
  - 🟡 partial: cover 50-90% 或 stale 4-30d
  - 🟠 sparse: cover < 50%
  - 🔴 stale: stale > 30d
  - ❌ missing / read_error / schema_unknown

- **coverage_mode 分类**:
  - `latest-day` (kline / margin / fund_flow / dragon_tiger): 算 latest
    交易日唯一 codes — 适合日频数据,catch sidecar NaN→0 fillna 类 silent bug.
  - `any-data` (shareholders / industry / unlock): 算全表唯一 codes —
    event-frequency 表用 latest-day cover 会假报 sparse.

- **产生背景**: 2026-05-25 发现 v19.4 sidecar margin covered=15/296
  (silent NaN→0 fillna 把 sidecar 静默降级成 z(pred)) 但无任何报警.
  本面板每日把 8 个数据源的 cover% + stale 摆出来,让 silent bug 立刻可见.

- **2026-05-26 ship 快照** (8 sources):
  - 🟢 baidu_kline 4659/5000 (93.2%, 1d)
  - 🟡 kline ∩ CSI300 225/300 (75.0%, 1d) — partial, 部分 CSI300 latest 缺
  - 🟡 margin 195/300 (65.0%, 4d)
  - 🔴 fund_flow stale 1972d (latest 2020-12-31)
  - 🟢 shareholders 289/300 (96.3% any-data, 1d)
  - 🟢 industry_membership any-data 100% (8d 可接受)
  - 🟢 unlock any-data 100% (0d, range 到 2026-06-30 含未来释放)
  - 🔴 dragon_tiger stale 1993d (latest 2020-12-10)

### 11. Trade History Timeline
**实现**: `dashboard/charts/trade_history_timeline.py:build_trade_timeline_section` (~245 LOC).

**数据源** (全只读):
- `data_cache/paper_trade_log.csv` (schema: date,action,symbol,name,score,price)
- `dashboard/utils/kline_fast.get_stock_kline(sym)` — Hive 分区单股 60 天 close

**视觉**:
- 上半 scatter: x=date, y=stock_idx, marker=^ (BUY) / v (SELL), color=green/red.
- 下半 line: top 6 picked stocks close 路径 (normalized 起点=1.0) + 实际 BUY/SELL 标记.
- summary table 列每个涉及 stock 的 BUY 次/SELL 次/最后 action/最后日期 (按 last-event
  日期降序, 最多 15 行).

**目的**: 看 model rotation 频率 + 哪些 stocks picked 过 + 真实下单时点 vs close 路径.

**2026-05-26 ship 快照**: 32 events / 17 unique stocks; Top 6 picked: SZ300661 (4×),
SH688396 (4×), SH600547 (3×), SZ300433/SZ300408/SZ300347 (each 2×). 累积期短
(<5 交易日), 真实 PnL 需 forward 3+ 月.

### 12. Daily Check Status (cron monitor)
**实现**: `dashboard/charts/daily_check_status.py:build_daily_check_status_section` (~280 LOC).

**数据源** (全只读, 不写):
- `/tmp/daily_check_YYYYMMDD.log` (daily_check.sh tee 主写, 按日期 stamped) — 优先
- `/tmp/daily_check_stdout.log` + `/tmp/daily_check_stderr.log` (launchd 兜底)
- `examples/com.claude_finance.daily_check.plist` (parse StartCalendarInterval Hour/Minute)
- `launchctl print gui/501/com.claude_finance.daily_check` (subprocess, 拿 last exit code)

**行为**:
- `_find_latest_log()`: 扫 `/tmp/daily_check_*.log` (8 位日期 stamped) + 静态 candidates,
  按 mtime 取最新.
- `_slice_last_run(lines)`: daily_check.sh 用 `tee -a`, 同一天可能含多次 run; 用
  `=== ... Daily check starting ===` 分隔取最后一次, 防 step 重复统计.
- 解析 step 行 `^\[(\d+(?:\.\d+)?)/\d+\]\s*(.+)$`, 但仅认 daily_check.sh 顶层 step
  (`OUTER_STEP_LABEL_PREFIXES` 8 个 prefix 白名单 — `Fetching today's kline /
  Data sanity check / Data completeness check / Margin incremental fetch / Running
  paper_trade signals / Forward OOS monitoring / Shadow v19.4 paper_trade / Syncing
  portfolio.xlsx`), 过滤子脚本 (fetch_baidu_kline.py / paper_trade_today.py) 嵌套的
  `[1/4]...[4/4]` 输出.
- step 状态分类 (`_classify_step_lines`): 该 step 范围内行扫
  - 🚨/FAIL/✗/❌/Error/ERROR/CRITICAL/Traceback → `fail`
  - ⚠/WARN/warning → `warn`
  - ✓/OK/saved/done/complete → `ok`
  - 否则 `unknown`
- 整体 exit code: `EXIT_PATTERN` 抓 `Done (exit=N)`, 配 overall 颜色 (0=绿 / 1-2=橙 / ≥3=红).
- 告警高亮: 扫 `🚨` / `SANITY CHECK FAILED` / `Alert level code` 三类关键行 (限 6 条防爆).
- launchd 下次 trigger: parse plist Hour/Minute (默认 16:30), `_next_trigger` 推下次本地时刻.
- log tail: 最后 12 行 (escape HTML, `<pre>` 暗色块).

**HTML 三层**:
1. 顶部 summary flexbox (Last run / Overall / Next launchd trigger 3 卡片) + log path metadata.
2. Steps `<table.data>` 3 列 (step id / label / status), status 列按颜色高亮.
3. Alerts `<ul>` 红色 + log tail `<pre>` 暗色块.

**Fallback**: log 找不到 → placeholder div 列候选位置 + 仍显示 next trigger.
任何 chart 函数失败 → render_report.py 顶层 try/except 兜底.

**2026-05-26 ship 快照**:
- log: `/tmp/daily_check_20260525.log` (40.5 KB)
- last run 2026-05-25 20:23:31 PDT, exit=0 success
- 最后一次 run 7 outer steps: kline=ok / sanity=fail (corruption 检出但 exit_code=0) /
  margin=ok / paper_trade=unknown / forward_oos=unknown / shadow_v19_4=unknown /
  portfolio_xlsx=ok
- 2 alerts: `🚨 DATA SANITY CHECK FAILED — production blocked` + `Alert level code: 0`
- next launchd trigger: 2026-05-26 16:30
- launchctl last exit = 126 (历史 plist 加载时 sandbox `Operation not permitted` 痕迹,
  与本次 run exit=0 不一致, 因 plist 已重新加载 + 路径授权后通畅)

**未触碰**: daily_check.sh / launchd plist / production 全 0 改动.

### 15. Picks Score Distribution (Stage 4 E — ship)

**数据源**: `data_cache/picks_today.json`
- `picks[]` — Top K (8) picks 含 `sym`, `z_pred`, `final_score`.
- `full_distribution` (新字段, paper_trade_today.py +11 LOC dump):
  `z_pred_values[]` (300 floats round 6 位) + `z_pred_mean/std/min/max/n_total`.

**渲染逻辑** (`dashboard/charts/picks_score_distribution.py`, ~180 LOC):
- `numpy.histogram` 40 bins, 灰色 bar.
- Top K 红色 axvline + 旋转 45° 标注 `code_with_name(sym)`.
- 第 (K+1) 名 z_pred 作 cutoff 绿色虚线 (Top K 须 ≥ 此值).
- 抬高 ylim 35% 给标注留空间.

**Verdict 逻辑** (评估当日 signal-to-noise):
- Top K 最低 z_pred &gt; μ+1σ → 信号清晰, picks 是真 outlier.
- μ &lt; Top K 最低 z_pred ≤ μ+1σ → 信号中等, picks 偏弱.
- Top K 最低 z_pred ≤ μ → 信号弱, 当日 picks 高风险.

**2026-05-26 ship 快照** (基于 dry-run 数据):
- as_of=2026-05-25, n_total=300 (CSI300 全 universe, 不是 296 — production overlay 实际为 300).
- z_pred μ=0.00 σ=1.00 (横截面 z 后必然 0/1), min=-7.21 (异常负 outlier 1 只) / max=+5.57.
- Top K=8 verdict: "Top 8 全部 > μ+1σ (+1.00) → 信号清晰, picks 是真 outlier".
- Top-9 cutoff ≈ 数据驱动 (运行时计算).
- panel img 大小 ~104 KB (b64 PNG dpi=150).
- 总 HTML 1177.4 KB → 1277.2 KB (Δ+99.8 KB).

**Fallback**:
- picks_today.json 缺失 → "paper_trade_today.py 尚未跑".
- 缺 `full_distribution` 字段 → 提示 "paper_trade_today.py 升级后才支持, 下次 paper_trade 跑完即可看到".
- JSON 解析失败 → 显示异常类型.

**Production 改动 (~11 LOC)**:
`examples/paper_trade_today.py` 在已有 `picks_today.json` dump 块内加 `full_distribution` 字段,
读 `overlay_df["pred_z"]` (apply_*_sidecar_overlay 输出, 含全 universe 横截面 z) 一并落盘.
**0 改 production 逻辑** — 只多 dump 一个 JSON 字段, 不影响 selection / sidecar / state.

### 16. Future Tab (Stage 3 剩余)
- multi-agent debate transcript (数据源待生, 等 multi-agent 推理框架落地)

## 数据源安全保证

Stage 1/2/3 **只读** 以下文件, 不写任何 production 数据:
- `data_cache/portfolio.xlsx` (Positions / Forward OOS Track)
- `data_cache/v17_dens_train24_predictions.parquet` (Stage 2)
- `data_cache/baidu_kline.parquet` (Stage 2, 间接, 通过 production 函数)
- `data_cache/baidu_kline_hive/code=*/*.parquet` (Stage 3, kline_fast Hive 分区)
- `data_cache/portfolio_state.json` + `data_cache/portfolio_state_v19_4.json` (Stage 3)
- `data_cache/paper_trade_log.csv` + `data_cache/paper_trade_log_v19_4.csv` (Stage 3)
- `data_cache/industry/industry_membership.parquet` (Section 8 — Sector Breakdown + Section 10)
- `data_cache/csi300_margin_14yr.parquet` (Section 10 — Factor Coverage Health)
- `data_cache/fund_flow/fund_flow_csi300.parquet` (Section 10)
- `data_cache/shareholders/shareholders_csi300.parquet` (Section 10)
- `data_cache/unlock/unlock_detail_em.parquet` (Section 10)
- `data_cache/dragon_tiger/*.parquet` (Section 10, 抽样 30 个文件)
- `data_cache/csi300_constituents.csv` (Section 10 — kline ∩ CSI300)
- `examples/paper_trade_today.py` (ast 解析 + importlib 复用 `load_amp_imb_20d_overlay`)
- `data_cache/picks_today.json` (Section 15 — Picks Score Distribution, 读 `full_distribution.z_pred_values`)
- `examples/factor_ic_*_is_monthly.csv` + `examples/super_big_net_is_monthly.csv` +
  `examples/v20_volume_zscore_is_monthly.csv` (Section 7 — Factor IC Heatmap)
- `/tmp/daily_check_YYYYMMDD.log` + `/tmp/daily_check_stdout.log` + `/tmp/daily_check_stderr.log`
  (Section 12 — Daily Check Status, log tail / step status / exit code)
- `examples/com.claude_finance.daily_check.plist` (Section 12 — parse Hour/Minute)
- `launchctl print gui/501/com.claude_finance.daily_check` (Section 12 — last exit, subprocess 只读)
- `dashboard/template.html` (模板)

**未触碰**: paper_trade_today.py / paper_trade_v19_4.py / forward_oos_monitor.py /
portfolio_excel.py / daily_check.sh / launchd plist / strategy_v*.py / data_cache/*.parquet.

## 错误恢复

每个 chart 函数内部都 try/except, 失败时 fallback 到 placeholder div 显示错误类型与
异常 message. 整页不会因为单个 panel 失败而崩.

## 可选: 集成到 daily_check.sh

不直接修 daily_check.sh — 用户可手动加 step 4.5 (失败不阻塞):

```bash
set +e
echo "[4.5/4] Generating dashboard report..."
.venv/bin/python dashboard/render_report.py 2>&1 | tee -a "$LOG"
set -e
```

## Stage 2 (本次 ship) — 因子贡献 stacked bar

Stage 2 范围:
- 新增 `dashboard/charts/factor_contrib.py` (~280 LOC)
- `render_report.py` 加 `build_factor_contrib_section` import + 调用 + 占位替换
- `template.html` 把原 section 4 占位 (4 个 future tab) 改为实际 chart + 新 section 5 留 2 个 Stage 3 future tab
- 重命名 `dashboard_stage1.md` → `dashboard_stages.md`

非范围 (留 Stage 3):
- Streamlit 交互 UI (静态报告够用)
- A/B v19.6 vs v19.4 shadow 对比 (paper_trade_log_v19_4.csv vs paper_trade_log.csv)
- multi-agent debate transcript (数据源待生)

## Stage 3 partial (本次 ship) — v19.6 vs v19.4 shadow A/B

Stage 3 partial 范围:
- 新增 `dashboard/charts/ab_v19_6_vs_v19_4.py` (~350 LOC)
- `render_report.py` 加 `build_ab_section` import + 调用 + `{{ab_v19_6_vs_v19_4}}` 占位替换
- `template.html` section 5 替换为实际 A/B chart, 新增 section 6 留 multi-agent
  debate transcript 占位
- footer 标识改为 "Stage 3 (partial)"

非范围 (留 Stage 3 剩余):
- multi-agent debate transcript (数据源待生, 等推理框架)
- 双 cum return 线图实际渲染 — 当前 forward 累积 < 5 交易日, fallback "积累中"
  自动提示, 累积 ≥ 5 个交易日后无需改代码自动切换为双线 MTM 走势.

## Stage 3 剩余 (multi-agent transcript) 触发条件

待 multi-agent 推理框架 (产生 `{agent, ts, msg, vote_for_stock}` JSON log)
落地后, 才能填实 transcript panel.

## Stage 4 layout 重排版 — 4 tab + 2-col grid + 大屏适配 (本次 ship)

**动机** (用户痛点):
1. 旧 layout `max-width: 1100px` 浪费大屏两侧空白.
2. 单 long page 17 sections → scroll fatigue, 难定位.
3. 信息密度低, panel 没有充分利用两侧.

**改动范围** (template.html only, render_report.py 0 改, charts/*.py 0 改):
- `--max-w` 从 `1100px` 升 `1600px`; `.page` padding 24/32 收紧到 16/24.
- 4 个 tab 分页 (sticky 顶部):
  - 📊 **Today** (4 panel) — KPI Summary (full-width) · Positions PnL · Sidecar Factor Config · Daily Check Status (full-width).
  - 🎯 **Picks & Trades** (4 panel) — Factor Contrib (full-width) · Picks Distribution · Picks Rotation · Trade History Timeline (full-width).
  - 🏆 **Models** (5 panel) — Forward OOS · A/B v19.6 vs v19.4 · Model Leaderboard (full-width) · IS→OOS Scatter (full-width) · Factor IC Heatmap (full-width).
  - 🌐 **Universe** (4 panel) — Sector Breakdown · Universe Progress · Factor Coverage Health (full-width) · Multi-Agent Debate (future, full-width).
- 2-column CSS grid (`.panel-grid` `grid-template-columns: 1fr 1fr`); wide chart 用 `.full-width` 跨双 col.
- Sticky tab nav (`position: sticky; top: 0; z-index: 100;`) + 切换时 `window.scrollTo({ top: 0 })`.
- Mobile breakpoint `<768px` → 1 col + tab-btn padding/font 减小 + header 改 column flex.
- `table.data td` 加 `font-variant-numeric: tabular-nums` (数字列对齐).
- Inter + Noto Sans SC font-family (Google Fonts 已加 preconnect, 复用旧 link).

**Placeholder 不变**: 17 个 `{{xxx}}` placeholder key 全保留 → render_report.py 0 改, charts/*.py 0 改.

**重排版后 panel 分布**:
| Tab | Panels | Full-width |
|---|---|---|
| Today | 4 | 2 (KPI Summary, Daily Check Status) |
| Picks & Trades | 4 | 2 (Factor Contrib, Trade Timeline) |
| Models | 5 | 3 (Leaderboard, Scatter, IC Heatmap) |
| Universe | 4 | 2 (Coverage Health, Multi-Agent) |
| **Total** | **17** | **9** |

**LOC delta**: template.html 276 → 360 (+84, 主要 sticky nav CSS + 4 tab pane + JS switcher).

**验证**:
- `.venv/bin/python dashboard/render_report.py` exit 0, 输出 ~1281 KB
  (与重排版前 1307 KB 基本一致 — chart payload 不变, 只省了少量重复 panel CSS).
- 浏览器 4 tab 切换正常, sticky 顶, 切换自动滚顶.
- 大屏 1440+ 2-col 渲染, 768- 单 col fallback.

## 参考

- 借鉴方案对比: [dashboard_borrow_plan.md](./dashboard_borrow_plan.md)
- Lean 模板来源: `references/Lean/Report/template.html` (Apache 2.0 license)
- Lean fig_to_base64 来源: `references/Lean/Report/ReportCharts.py:56-62`
