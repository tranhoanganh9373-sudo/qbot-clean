# Dashboard 技术 — 7 references 借鉴方案

只读调研 `/Volumes/SSD/finance/claude_finance/references/`,为本项目 dashboard 选一个能"直接照搬"的方案。

本项目 5 大需求:
1. 看模型当前用什么因子 (config view + factor list)
2. 看哪个因子起作用 (per-stock 因子贡献 stacked bar: pred_z + λ·z_factor)
3. 未来多 agent 讨论选股 (transcript + 投票)
4. Forward OOS perf chart (cum return / drawdown)
5. A/B comparison (v19.6 main vs v19.4 shadow 持仓差异)

---

## 1. 7 项目 viz tech stack 对比表

| # | 项目 | 主 viz 库 | 是否 web frontend | 静态 / 动态 | 关键文件路径 | LOC |
|---|---|---|---|---|---|---|
| 1 | **OpenBBTerminal** | Plotly (Python 后端) + Tauri/React (桌面壳) | 有 (FastAPI `platform_api` + Tauri/React `desktop/`) | 动态 (Plotly 交互) | `openbb_platform/obbject_extensions/charting/openbb_charting/` 含 `charting.py` 772, `core/openbb_figure.py` 1582, `core/backend.py` 482, `charts/*.py` 共 2124; 桌面壳 React/Tauri (TS, 不可直 port 到 Python 项目) | 总 ~11.4k Python LOC |
| 2 | **Lean (QuantConnect)** | matplotlib (Python, `ReportCharts.py`) + Jinja-风格 HTML template | 半-web (输出单文件 HTML 报告,无服务器) | 静态 (一次生成 self-contained HTML) | `Report/ReportCharts.py` 1148, `Report/template.html` 358, `Report/Report.cs` 206 (orchestration), `Report/ReportElements/*.cs` 28 个 element | ~1.7k Python + 358 HTML (C# 部分 ~2-3k 不需 port) |
| 3 | **pyfolio-reloaded** | matplotlib only | 无 | 静态 (PNG / Jupyter inline) | `src/pyfolio/tears.py` 1315 (8 个 `create_*_tear_sheet`), `src/pyfolio/plotting.py` 2081 (40+ `plot_*` 函数) | 7.2k 总 |
| 4 | **alphalens-reloaded** | matplotlib only | 无 | 静态 (PNG / Jupyter inline) | `src/alphalens/tears.py` 706 (7 个 `create_*_tear_sheet`), `src/alphalens/plotting.py` 1014 (20+ `plot_*` 函数) | 4k 总 |
| 5 | **zipline-reloaded** | examples 用 matplotlib (非 lib 自带) | 无 | N/A (engine,无 dashboard) | `src/zipline/examples/*.py` 仅画 perf curve | <500 viz 相关 |
| 6 | **zipline (legacy)** | 同上 | 无 | N/A | `zipline/examples/*.py` | 同上 |
| 7 | **empyrical-reloaded** | 无 (纯 stats lib) | 无 | N/A | `src/empyrical/stats.py` 等 | 0 viz |

**结论摘要**: 7 项目里只有 3 个值得借鉴 — pyfolio、alphalens、Lean (Report)。OpenBB Plotly+Tauri 体量大且耦合 Tauri 桌面壳,不适合直接 port。zipline / empyrical 无 dashboard。

---

## 2. 跟本项目需求 fit 评分 (1-5, 5=最好)

| 候选 | 因子贡献 bar | multi-agent 讨论 | forward OOS perf | A/B 比较 | 实施难度(易=5) | 合计 |
|---|---|---|---|---|---|---|
| pyfolio (静态 PNG/tear sheet) | 2 | 1 | **5** | 3 | **5** | 16 |
| alphalens (IC/quantile heatmap) | **5** | 1 | 2 | 2 | 4 | 14 |
| **Lean Report template.html 模式** | 4 | 3 | **5** | **5** | 4 | **21** ← 最高 |
| OpenBB Plotly+Tauri | 5 | 4 | 5 | 5 | 1 (要 port Plotly+Tauri 重资产) | 20 |
| zipline / empyrical | 0 | 0 | 1 | 0 | — | — |

**Top 1: Lean `Report/template.html` 模式** — HTML 模板 + matplotlib base64 PNG 占位符替换,单文件 self-contained 报告,无需 web server,既可邮件发送/归档,又方便增量加 section (因子贡献 / A/B / multi-agent transcript 都只是新增 `<table>` + 一段 `{{$PLOT-X}}`)。

**Top 2: pyfolio create_returns_tear_sheet** — 项目已用 (`examples/pyfolio_paper_trade_tear_sheet.py`),负责 forward OOS perf curve 这一项最熟最快,本来就是 Stage 1 必含组件。

---

## 3. 推荐方案: 借鉴 Lean Report 模板 + pyfolio/alphalens 现成图

### 3.1 直接 copy

| 源文件 | 目标文件 | 改动 |
|---|---|---|
| `references/Lean/Report/template.html` (358 行) | `dashboard/template.html` | **结构 copy**,把 Lean 的 KPI/Plot 占位符 (如 `{{$KPI-CAGR}}`、`{{$PLOT-CUMULATIVE-RETURNS}}`) 替换成本项目语义 (`{{$KPI-CALMAR-V196}}`, `{{$PLOT-FACTOR-CONTRIB-TOP7}}`, `{{$TRANSCRIPT-AGENT-DEBATE}}` etc.) |
| `references/Lean/Report/ReportCharts.py` 中的 `fig_to_base64()` 工具函数 (56-62 行) | `dashboard/chart_utils.py` | 直接 copy 此 12 行 helper,删除 `from clr import AddReference` 的 .NET 部分 |
| pyfolio `create_returns_tear_sheet` 已 wired 在 `examples/pyfolio_paper_trade_tear_sheet.py` | reuse | 把它从 PNG 改成返回 `fig` 对象,再用 `fig_to_base64` 嵌入 template |
| alphalens `plot_quantile_returns_bar` (plotting.py:372) + `plot_ic_ts` (plotting.py:214) | reuse via 调用 | 现成 API,可直接 import,不需复制源码 |

### 3.2 自己写 (改 adapter + 业务)

| 文件 | 用途 | LOC 估算 |
|---|---|---|
| `dashboard/render_report.py` | 主入口:读 `data_cache/paper_trade_log.csv` + `data_cache/forward_oos_*.parquet` + portfolio.xlsx → 喂数据给 chart 函数 → 模板替换 → 写出 `reports/daily_report_YYYYMMDD.html` | ~150 |
| `dashboard/charts/factor_contrib.py` | **新 chart 1**: per-stock 横向堆叠 bar — x 轴 final_score, 颜色分段标 `z(pred)`、`-0.30·z(amp_imb_20d)` 贡献; matplotlib 单图 | ~80 |
| `dashboard/charts/ab_diff.py` | **新 chart 2**: v19.6 vs v19.4 shadow 持仓 venn / overlap table + 7 日累计 perf 双线 | ~60 |
| `dashboard/charts/forward_oos_curve.py` | **包装 pyfolio** 的 cum return + underwater drawdown,适配本项目 forward OOS parquet | ~50 |
| `dashboard/charts/agent_transcript.py` (占位, future) | 多 agent 讨论 transcript → HTML `<div class="transcript">`; v0 留空 div | ~30 |
| `dashboard/template.html` (从 Lean copy 改) | 主模板 | ~200 (Lean 358 砍剔 Crisis/Parameters section) |

### 3.3 LOC 估算

| 类别 | LOC |
|---|---|
| copy (Lean template 改名 + fig_to_base64) | ~210 |
| 写新 adapter + 4 个 chart | ~370 |
| **Stage 1 total** | **~580** |

不含 multi-agent transcript 实际逻辑 (那是 future stage),只占位 div。

---

## 4. 实施 plan (分 stage)

### Stage 1 (~1 day): 周报式静态 HTML 报告

**第一个文件**: `dashboard/render_report.py`
**第一个 chart**: **forward OOS perf 累积曲线 + drawdown underwater** (复用 pyfolio, 改动最少, 先把 pipeline 跑通)
**Top 2 chart 紧接着**: per-stock 因子贡献 stacked bar (本项目核心信息)

**步骤**:
1. `cp references/Lean/Report/template.html dashboard/template.html` (read-only references **只是 reference**,真实 copy 命令在实施时执行)
2. 把 Lean template 里的 quantconnect.com 头/作者/crisis section 删掉,留下 KPI table + cumulative returns / monthly returns / underwater / annual returns 4 块
3. 加新 section: `<div class="factor-contrib">{{$PLOT-FACTOR-CONTRIB-TOP7}}</div>` + `<div class="ab-diff">{{$TABLE-AB-DIFF}}{{$PLOT-AB-CUM}}</div>` + `<div class="transcript-future">multi-agent transcript (TBD)</div>`
4. 实现 `render_report.py`: 读 paper_trade_log + forward_oos_monitor 输出 + portfolio.xlsx → 调 chart 函数 → `template.replace(...)` → 写 `reports/daily_report_<DATE>.html`
5. 启动命令: `.venv/bin/python -m dashboard.render_report --date 2026-05-25 --out reports/daily_report_20260525.html`

**Deliverable**: 单个 self-contained HTML (~1-2 MB,内嵌 base64 PNG),浏览器双击打开就能看全部 5 类信息。可塞进 `daily_check.sh` 的 step 4 一键日报。

### Stage 2 (~2 day): Streamlit 交互式 dashboard

**何时升级**: Stage 1 静态报告 + 1 周用户体验后,确认需要 (a) tab 切换、(b) ad-hoc date range、(c) 实时刷新 后才上 Streamlit。否则 Stage 1 已够用。

**借鉴模式**: OpenBB 的 `openbb_platform/extensions/platform_api/openbb_platform_api/utils/api.py` (FastAPI 启动 + widget JSON config 模式) **不直接 copy 代码**,只借鉴 "config-driven widget" 的概念 — 每个 chart 是独立可配置 widget,通过 `widgets.json` 声明。但 Streamlit 自带 `st.tabs`、`st.dataframe`、`st.plotly_chart`,实际不需要 OpenBB 那套复杂 widget framework。

**新增**:
- `dashboard/app.py` Streamlit 入口 (~200 LOC, 4 个 tab: Factor Config / Per-Stock Contrib / Forward OOS / A/B)
- 复用 Stage 1 的 `dashboard/charts/*.py` 函数 (返回 fig, Streamlit 用 `st.pyplot(fig)` 渲染)
- 启动: `.venv/bin/streamlit run dashboard/app.py`

### Stage 3 (future): Multi-agent debate panel

- 多 agent transcript 是 Stage 3 才落地,Stage 1/2 只保留 `<div class="transcript-future">` 占位。
- 落地形态: HTML `<div class="transcript">` 每条发言带 agent name + 投票图标; Streamlit `st.chat_message` API 也合适。
- 数据源: 多 agent 推理产物 (JSON log: `{agent, ts, msg, vote_for_stock}`), 目前还没有, 等 agent 框架就绪。

---

## 5. 为什么 **不** 选其他方案

- **OpenBB**: 体量过大 (11k+ Python LOC + Tauri 桌面壳 + FastAPI),且其 `openbb_charting` 已与 OBBject(ObbjectExtension) 框架深耦合, port 出来需重写大量胶水代码; 单是 `core/openbb_figure.py` 就 1582 LOC。**收益/成本不划算**, Stage 1 静态报告完全用不到 Plotly 交互。
- **pyfolio 单独用**: 它只画 returns/positions 标准 tear sheet, 无法承载因子贡献 / A/B / multi-agent 这 3 类业务定制视图; 但其图作为 Lean template 内嵌的 sub-figure 是最佳搭配。
- **alphalens 单独用**: 专 factor IC + quantile, 与 Stage 1 周报场景吻合度低, 留作 Stage 2 "因子健康度" 单独 tab 备用。
- **Lean 整套 C# Report 工程**: 不可 port (.NET clr/Deedle), 只取 `template.html` 与 `fig_to_base64` 设计模式。
- **zipline / empyrical**: 无 dashboard 基因, 无可借鉴。

---

## 6. 风险 / 已知 gap

1. **Lean template 是 Bootstrap 3 + CDN font**: 离线环境需把 CSS / 字体下到本地, 或接受 CDN 依赖。Stage 1 可保留 CDN (网络可用), Stage 2 再 vendoring。
2. **factor 贡献 bar 的 `z(factor)` 数值**: 现 `paper_trade_today.py` 算 final score 时 sidecar 项是局部 z-score, 需 export `z_amp_imb_20d` 这一列到日志才方便画图; 改动只在 paper_trade 日志列加 1 列, 不影响 production 选股逻辑。
3. **A/B compare**: v19.4 shadow log `data_cache/paper_trade_log_v19_4.csv` 已 ship (memory `project_v19_6_production_upgrade.md`), 数据齐; Venn 图 matplotlib 用 `matplotlib_venn` 第三方库, 或手画双圆 + 中文 overlap 数字。
4. **multi-agent**: 数据源未生, Stage 1 仅占位。

---

## 7. 最终一句话方案

**Stage 1 = Lean `template.html` 设计模式 + pyfolio 已有 returns figure + 2 个新 chart (factor contrib stacked bar / A/B diff) → 单个 self-contained HTML 日报, ~580 LOC**。一天可上线, 塞进 `daily_check.sh` step 4, 产物 `reports/daily_report_YYYYMMDD.html`。Stage 2/3 视使用反馈再升 Streamlit + multi-agent。
