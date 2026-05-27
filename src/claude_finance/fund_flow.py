"""资金流(主力/超大单/大单/中单/小单) 因子探索 — sandbox 可达性记录.

# 结论 (2026-05-25 探查)

**Track B (资金流 daily history 2014-2020) 不可行**，原因记录：

1. **东财 push2his.eastmoney.com `/api/qt/stock/fflow/daykline/get`**:
   - sandbox ❌ 多次 retry 全部 ProxyError (CLAUDE.md 已标记 push2his 不通)
   - 即便偶尔通, 返回**最近 120 个交易日** (lmt 参数无法扩大历史深度)
   - 不能覆盖 IS 期 (2014-2020)

2. **东财 push2.eastmoney.com `/api/qt/stock/fflow/kline/get`**:
   - sandbox ❌ push2 502 (CLAUDE.md 记录)
   - 即便能通, klt=101 daily 仅返回 当日 1 条
   - 无历史

3. **东财 datacenter-web 资金流报表**:
   - 探测 RPT_DRC_STOCK_DC_FLOW / RPT_FCDC_FUND_FLOW / RPT_DAILYCASH /
     RPT_FLOW_BACK / RPT_FCDRC_STOCK_FLOW 均返回 "报表配置不存在"
   - datacenter 上没有暴露日级 fund flow

4. **akshare ak.stock_individual_fund_flow / ak.stock_main_fund_flow**:
   - 同样底层 push2his → ProxyError. 已实测.

5. **百度 PAE finance.pae.baidu.com `/finance/quotes/fundflow`**:
   - V3.1 标记已下线 (2026-05)

**结论:** sandbox 内无法获取 2014-2020 daily fund flow. 需付费数据.

# 备选 ✅ 用 fund flow signal 的等价物 — 龙虎榜 / 大宗交易 / 融资融券

这三个端点 sandbox-OK 且能反映"主力资金动向":
- **龙虎榜** (RPT_DAILYBILLBOARD_DETAILSNEW) → 大资金集中入场信号
- **大宗交易** (RPT_DATA_BLOCKTRADE) → 机构间过手
- **融资融券** (csi300_margin_14yr.parquet 已 cache) → 杠杆资金动向

这些放在 `dragon_tiger.py` / `block_trade.py` / `margin_cache.py` 模块.
"""
from __future__ import annotations


def is_ic_feasibility_note() -> str:
    """Returns the feasibility verdict for IS (2014-2020) IC analysis.

    See module docstring for full reasoning.
    """
    return (
        "Fund flow daily history NOT FEASIBLE for IS 2014-2020 in sandbox: "
        "(1) push2his proxy-blocked, even when through returns only last "
        "120 days; (2) push2 502 Bad Gateway; (3) datacenter-web has no "
        "daily fund flow report; (4) akshare routes to push2his; "
        "(5) baidu PAE fund flow endpoint dead since 2026-05. Substitutes: "
        "dragon-tiger (RPT_DAILYBILLBOARD_DETAILSNEW), block trade "
        "(RPT_DATA_BLOCKTRADE), margin trading (csi300_margin_14yr.parquet)."
    )
