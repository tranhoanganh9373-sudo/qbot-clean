"""金融术语词汇表 (中文释义) — 给非金融背景用户的快速参考.

放在 Today tab 最顶, 解释 dashboard 内所有 chart/table 用到的 25-30 个常见术语.
"""
from __future__ import annotations

import html

# 术语条目: (英文缩写, 中英全称, 中文释义)
GLOSSARY: list[tuple[str, str, str]] = [
    # ── 收益/风险指标 ──────────────────────────────────────────
    ("PnL", "Profit and Loss · 盈亏",
     "持仓收益 - 持仓成本。正数 = 赚钱, 负数 = 亏钱。"),
    ("Calmar", "Calmar Ratio · 卡玛比率",
     "ann_return / |MDD|。衡量「每单位最大亏损能赚多少」, 越高越好。"
     "基准: >1.0 优秀, 0.5-1.0 良好。"),
    ("Sharpe", "Sharpe Ratio · 夏普比率",
     "(收益 - 无风险利率) / 波动率。衡量风险调整后收益, 越高越好。基准: >1.0 优秀。"),
    ("MDD", "Max Drawdown · 最大回撤",
     "从顶点到谷底的最大跌幅。-30% 意味着最坏时刻账户从 100 跌到 70。"),
    ("ann", "Annualized Return · 年化收益率",
     "把多月/多年的累计收益按年折算。例: 60 月 cum 258% → ann ≈ 23%。"),
    ("cum", "Cumulative Return · 累计收益",
     "从起点至今的总涨跌幅。258% 表示翻 3.58 倍。"),

    # ── 训练/测试期 ─────────────────────────────────────────────
    ("OOS", "Out-of-Sample · 样本外",
     "测试期 (2021-05~2026-04, 60 月)。模型训练时没见过的数据, 真实考验 alpha。"),
    ("IS", "In-Sample · 样本内",
     "训练期 (2014~2020, 84 月)。模型在这里学规律。IS 表现强不等于 OOS 强。"),
    ("Walk-forward", "Walk-forward · 滚动训练",
     "每月用过去 24 月数据重训模型, 预测下月。避免一次性训练 + 永久测试的 leak。"),

    # ── 因子/选股 ───────────────────────────────────────────────
    ("Alpha", "Alpha · 超额收益",
     "跑赢市场基准的部分。例: CSI300 涨 10%, 我们 20% → alpha = 10%。"),
    ("Sidecar", "Sidecar · 旁路因子",
     "在 base 模型预测 z(pred) 上叠加额外因子做调整。"
     "如 v19.10: final = z(pred) - 0.30 × z(amp_imb_20d) + 0.10 × z(JZF)。"),
    ("Picks", "Picks · 选股",
     "模型每日推荐买入的 Top K 股票 (本项目 K=8)。"),
    ("Universe", "Universe · 股票池",
     "可被模型选择的全部股票集合。CSI300 = 沪深 300 大盘股, CSI500 = 中证 500 中盘股。"),

    # ── 因子有效性 ──────────────────────────────────────────────
    ("IC", "Information Coefficient · 信息系数",
     "因子值 vs 未来收益的横截面相关性。月度 IC = 该月所有股票因子值跟下月收益的 Spearman。"),
    ("ICIR", "IC Information Ratio · IC 信息比率",
     "IC 的均值 / 标准差 × √12。衡量因子稳定性。>0.4 良好, >0.6 优秀。"),
    ("Spearman", "Spearman Correlation · 斯皮尔曼秩相关",
     "用排名计算的相关性, 不受异常值影响。范围 [-1, +1]。"),
    ("z-score", "Standard Score · 标准分数",
     "(x - μ) / σ。把不同量纲转成可比的「几个标准差」。z=2 表示比平均高 2 个 σ。"),

    # ── A 股专业 ────────────────────────────────────────────────
    ("CSI300", "CSI300 · 沪深 300 指数",
     "300 只大盘股, 代表 A 股市场核心。本项目 production universe。"),
    ("CSI500", "CSI500 · 中证 500 指数",
     "中盘股 500 只。基础设施已扩, production 未启用。"),
    ("Margin", "Margin Trading · 融资融券",
     "投资者借钱买入。融资余额上升 = 资金涌入 = 短期热度高。"),
    ("Limit-up/down", "涨停 / 跌停",
     "A 股日内最大涨/跌幅: 主板 10%, 创业板/科创板 20%。触及涨停当日买不到。"),
    ("qfq / hfq", "前复权 / 后复权",
     "对除权除息 (分红/股票分割) 的两种调整方式。本项目 baidu_kline 是 hfq (后复权)。"),

    # ── 模型 ───────────────────────────────────────────────────
    ("DEnsemble", "Double Ensemble Model · 双层集成",
     "qlib 内置 GBM-based 模型, 多个子模型加权 + 样本/特征 reweight。"),
    ("Alpha158", "Alpha158 · 158 因子集",
     "qlib 内置 158 个技术指标因子 (K 线/量价/波动)。本项目 base feature set。"),
    ("TopK", "Top K Selection · 取前 K 名",
     "按 final_score 排序后买入排名前 K 只。本项目 K=8 (Dropout=2 表示每日换 2 只)。"),

    # ── 项目内术语 ─────────────────────────────────────────────
    ("Phase A", "Phase A · 因子 IS IC 分析",
     "新因子的样本内 IC 验证阶段, 看 ICIR 是否够强 (≥0.4) + 跟现有因子独立。"),
    ("Phase B", "Phase B · 因子 OOS sidecar 验证",
     "Phase A 通过后, 锁 λ 在 OOS 单跑一次, 看 Calmar 是否真提升 production。"
     "4 例 catastrophic 失败。"),
    ("v19.10 / v19.6 / v19.4 / baseline", "策略版本",
     "v19.10 = amp_imb_20d + JZF stacked sidecar (production, since 2026-05-27), "
     "v19.6 = amp_imb_20d 单因子 (shadow, paper_trade_v19_6.py 12月 A/B), "
     "v19.4 = margin sidecar (fallback, OFF), "
     "baseline = 无 sidecar 纯模型。"),
    ("amp_imb_20d", "20 日振幅不对称 (Amplitude Imbalance)",
     "(20 日上涨振幅 - 20 日下跌振幅) / 20 日总振幅。高 = 涨势猛 → 反转减分。"),
    ("JZF", "竞价勾魂翻 · 集合竞价跳空",
     "JZF = (open - prev_close) / prev_close × 100, 量化集合竞价跳空幅度。"
     "v19.10 stacked sidecar 第二因子, sign=+1 顺势, λ=0.10。"
     "60月 OOS Calmar 1.27 单跑, 跟 amp_imb_20d 协同 stacked 提升至 2.12 (+64%)。"),
    ("集合竞价", "Call Auction · 9:15-9:25 开盘前撮合",
     "A 股开盘前 10 分钟集中申报, 9:25 统一成交确定 open price。"
     "JZF 因子量化这个开盘跳空, 反映隔夜情绪。"),
    ("B 路线", "5m execution overlay (B route)",
     "选股不变 (v19.10), 仅用 5m bar 做入场点 audit (真跳空/假跳空 confirmation)。"
     "对应 A 路线 (intraday 调仓 — 被否决违反 OOS 协议) 和 C 路线 "
     "(5m features 加 model — 数据缺失受阻)。"),
    # ── Risk metrics ────────────────────────────────────────────
    ("Beta", "Beta · 系统性风险敏感度",
     "β = cov(portfolio_ret, csi300_ret) / var(csi300_ret)。"
     "β=1 跟大盘等强度, β>1.3 aggressive 放大波动, β<0.7 defensive。"),
    ("VaR", "Value at Risk · 风险价值",
     "VaR 95% = historical 5% percentile 的单日亏损 %。"
     "意思: 100 天里有 5 天 (最差的 5%) 会比这个数字亏更多。"),
    ("波动率 / Vol", "Volatility · 波动率",
     "年化波动率 = daily σ × √252。<25% 正常, 25-35% 偏高, >35% 极高。"
     "高 vol = 大幅振荡, MDD 风险大。"),
    ("集中度", "Concentration Risk · 集中度风险",
     "单股 / 行业占比过高时, 单一冲击放大组合损失。"
     "dashboard 告警阈值 25%: 等权 8 只 = 12.5% 安全。"),
    # ── 信号验证 / 策略评估 ──────────────────────────────────────
    ("Forward Alpha", "Forward Alpha (T+1/T+5)",
     "production picks 真实未来收益减 CSI300 同期 = forward α。"
     "T+1 = 买入次日收益, T+5 = 5 个交易日后。"
     "正值 = production 跑赢大盘, 累积 3-6 月样本看 forward 真实 alpha。"),
    ("PnL Attribution", "PnL 归因 by stock",
     "拆分每只持仓贡献 (realized + floating PnL)。"
     "按总 PnL 排序看 winners vs losers, 识别拖后腿股。"),
    ("Sparkline", "迷你 K 线图",
     "极简 line chart (~1-2h 5m close 序列), 嵌在表格单元格内, "
     "直观看 intra-day trend。绿涨红跌."),
    ("n_months gate", "Phase B 严格 60 月样本下限",
     "Phase A 候选必须 n_months ≥ 60 才允许进 Phase B。"
     "教训: 4 个 thin-sample (n<60) Phase B 全 OOS Calmar 衰减 -88% ~ -116%。"
     "baidu_kline 2014-2017 CSI300 sparse 是常见 ceiling 根因。"),
    # ── Value 因子 (Phase A 候选) ────────────────────────────────
    ("PE / PB / 股息率", "Value Factors · 价值因子",
     "PE = 价格 / 每股盈利, PB = 价格 / 每股净资产, 股息率 = 年股息 / 价格。"
     "经典 Fama-French value factor (低估值 → 涨)。Phase A 用 sign=-1。"),
    ("52w high distance", "52 周高距 momentum proxy",
     "(close - max(close last 252d)) / max(close last 252d), 范围 [-1, 0]。"
     "靠近 0 = 接近年高 (强势), 越负 = 跌得越深。"
     "George & Hwang 学术经典 momentum proxy。"),
    ("Earnings revision", "盈利预期修正 · 分析师 EPS 上下调",
     "分析师 30d EPS forecast 上下调比例 = revision_score。"
     "强势学术验证因子, 但实测 akshare/EM API 历史 forecast 全 NaN, "
     "退化为 coverage proxy (分析师覆盖度)。"),
    # ── 服务 / 工程 ─────────────────────────────────────────────
    ("launchd", "macOS service manager · daemon 管理",
     "macOS 系统级 daemon 管理. plist 在 ~/Library/LaunchAgents/。"
     "项目 3 个 service: dashboard_server (port 5557), poll_5m_picks (60s), "
     "daily_check (16:30 cron)。"),
    ("Daily Check", "Daily Check · 每日 16:30 cron",
     "launchd 触发 daily_check.sh, 跑 7 步: sanity / 完整性 / margin fetch / "
     "paper_trade / forward OOS monitor / shadow v19.4 / shadow v19.6。"
     "16:30 收盘后自动执行。"),
    ("mootdx", "mootdx · Python TDX client",
     "纯 Python 调用通达信 server (6 个 IP pool) 抓 OHLCV。"
     "支持 daily (freq=4) / 5m (freq=0) / 1m (freq=7)。"
     "项目用作 baidu_kline backfill + 5m fetch + 实时 polling 主数据源。"),
    # ── Trade log + 仓位会计 ────────────────────────────────────
    ("Trade Log", "Trade Log · 交易流水",
     "append-only 交易日志, 每笔 BUY/SELL 一行 (data_cache/trades.jsonl)。"
     "替代覆盖式 xlsx 输入, 支持同日多笔。"),
    ("WAC", "Weighted Average Cost · 加权平均成本",
     "perpetual moving average: BUY 时 (existing×avg + new×price) / (existing + new); "
     "SELL 时不变。Panel A '平均成本' 列。"),
    ("FIFO", "First In First Out · 先进先出",
     "另一种成本核算法 (SELL 时按最早买入价匹配)。dashboard 用 WAC 不是 FIFO。"),
    ("已实现盈亏", "Realized PnL",
     "SELL 时锁定的盈亏 = (sell_price - WAC) × sell_shares。卖出后才 lock,"
     "区别于浮动 PnL (持仓期间 mark-to-market)。"),
    ("浮动 PnL", "Floating / Unrealized PnL",
     "持仓期间 mark-to-market 盈亏 = (current_close - WAC) × net_shares。"
     "每天随收盘价变化, 未锁定。"),
    # ── 通达信公式 ──────────────────────────────────────────────
    ("TDX", "TongDaXin · 通达信公式",
     "中国券商通达信软件的指标公式语法, 形如 EMA(CLOSE,12), MA(CLOSE,5), CROSS(A,B)。"
     "mt180.com (指标公式评测室) 收录 ~24k 通达信公式。"),
    ("STICKLINE", "STICKLINE / DRAWICON · TDX 绘图函数",
     "通达信公式中的 chart 绘制函数, 不影响数值计算, 转 Python 因子时跳过。"),
    # ── 数据 / 工程 ─────────────────────────────────────────────
    ("PIT", "Point-In-Time · 时点对齐",
     "回测严格禁止 lookahead, 每个时点只用当时已知数据 (e.g. 基本面 lag 60 天)。"),
    ("hfq / qfq", "后复权 / 前复权",
     "hfq = 起点不变, 后期按比例放大 (回测用); qfq = 终点不变, 历史按比例缩小 (券商显示)。"
     "baidu_kline 用 qfq, corrupt-fix 27 股用腾讯 hfq。"),
    ("PII", "Personally Identifiable Info · 个人信息",
     "如手机号 (authorPhone)、ID。dashboard / 抓取 / log 全自动 strip,不持久化。"),
    ("Atomic Write", "Atomic Write · 原子写入",
     "写文件先到 .tmp 后 rename, 防止写入中崩溃损坏主表。"
     "fetch_baidu_kline / dashboard_submit_server 都用此模式。"),
]


def build_glossary_section() -> str:
    """compact 2-column grid 显示术语表."""
    items_html: list[str] = []
    for abbr, name, desc in GLOSSARY:
        items_html.append(
            '<div style="border-left:3px solid var(--accent); padding:6px 10px; '
            'margin-bottom:8px; background: rgba(37,99,235,0.04); '
            'border-radius: 0 4px 4px 0;">'
            f'<div style="font-weight:600; font-size:13px;">{html.escape(abbr)} '
            '<span style="color:var(--muted); font-weight:400; font-size:12px;">'
            f'· {html.escape(name)}</span></div>'
            '<div style="font-size:12px; color:var(--fg); margin-top:3px; '
            'line-height:1.5;">'
            f'{html.escape(desc)}</div></div>'
        )
    return (
        '<div style="display:grid; grid-template-columns: repeat(2, 1fr); '
        'gap:8px 16px;">'
        + "".join(items_html)
        + "</div>"
        '<div style="margin-top:12px; font-size:11px; color:var(--muted);">'
        '提示: 这些是项目最常用术语, 其他 chart 内具体公式见 chart 下方解读文字。'
        f'共 {len(GLOSSARY)} 条术语。'
        "</div>"
    )
