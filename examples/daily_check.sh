#!/usr/bin/env bash
# Daily Forward OOS check pipeline
# Usage: bash examples/daily_check.sh
# 推荐 launchd 16:30 触发 (A 股收盘后).
#
# 退出码: 0=green, 1=yellow, 2=orange, 3=red, 4=black (来自 forward_oos_monitor.py)

set -e
set -o pipefail   # 关键: 让 `python | tee` 的 $? 反映 python 退出码而非 tee (永远 0)
cd "$(dirname "$0")/.."
LOG=/tmp/daily_check_$(date +%Y%m%d).log

echo "=== $(date) Daily check starting ===" | tee -a "$LOG"

# 1. 增量 K 线
echo "[1/4] Fetching today's kline..." | tee -a "$LOG"
.venv/bin/python examples/fetch_baidu_kline.py 2>&1 | tee -a "$LOG"

# 1.3 重 build qlib feature cache (baidu_kline.parquet → qlib_baidu binary)
# 关键: paper_trade 读 qlib cache, 不重 build 则 predictions 截止 cache 上次 build 日期
echo "[1.3/4] Rebuild qlib cache from baidu_kline..." | tee -a "$LOG"
set +e
.venv/bin/python examples/convert_baidu_to_qlib.py 2>&1 | tee -a "$LOG"
QLIB_REBUILD_CODE=$?
set -e
if [ $QLIB_REBUILD_CODE -ne 0 ]; then
    echo "⚠ qlib rebuild exit=$QLIB_REBUILD_CODE (不阻塞, paper_trade 可能用旧 cache)" | tee -a "$LOG"
fi

# 1.5 数据完整性 sanity check (corruption → 阻塞后续 step, exit 99)
echo "[1.5/4] Data sanity check..." | tee -a "$LOG"
set +e
.venv/bin/python examples/data_sanity_check.py 2>&1 | tee -a "$LOG"
SANITY_CODE=$?
set -e
if [ $SANITY_CODE -ne 0 ]; then
    echo "🚨 SANITY CHECK FAILED (exit=$SANITY_CODE) — blocking subsequent steps" | tee -a "$LOG"
    osascript -e 'display notification "Data sanity check FAILED — production blocked" with title "🚨 数据 corruption"' || true
    exit 99
fi

# 1.6 Universe completeness (baidu_kline coverage vs universe.csv + 大蓝筹 must-have)
echo "[1.6/4] Data completeness check..." | tee -a "$LOG"
set +e
.venv/bin/python examples/data_completeness_check.py 2>&1 | tee -a "$LOG"
COMPLETENESS_CODE=$?
set -e
if [ $COMPLETENESS_CODE -eq 99 ]; then
    echo "🚨 COMPLETENESS CRITICAL — blocking subsequent steps" | tee -a "$LOG"
    osascript -e 'display notification "Universe completeness CRITICAL — production blocked" with title "🚨 数据残缺"' || true
    exit 99
elif [ $COMPLETENESS_CODE -eq 1 ]; then
    echo "⚠️ COMPLETENESS WARNING — continuing" | tee -a "$LOG"
fi

# 1.7 Margin daily incremental (v19.4 sidecar overlay)
# 失败不阻塞: 若 fetch 失败, paper_trade 自动降级到 v19.1 (margin_status=stale 或 missing).
echo "[1.7/4] Margin incremental fetch (v19.4 sidecar)..." | tee -a "$LOG"
set +e
.venv/bin/python examples/fetch_margin_today.py 2>&1 | tee -a "$LOG"
set -e

# 1.8 风控前置 gate self-check (P0-B). exit 2 = MDD 或 daily_loss 触发, 阻塞 paper_trade.
# exit 1 = NAV 历史不足, 仅 warn 不阻塞.
echo "[1.8/4] Risk gates self-check (MDD/daily_loss)..." | tee -a "$LOG"
set +e
.venv/bin/python -m claude_finance.risk.gates --self-check 2>&1 | tee -a "$LOG"
RISK_GATE_CODE=$?
set -e
if [ $RISK_GATE_CODE -eq 2 ]; then
    echo "[1.8/4] 🚫 风控熔断 — paper_trade 阻塞. 见 data_cache/risk_event_log.csv" | tee -a "$LOG"
    exit 2
elif [ $RISK_GATE_CODE -eq 1 ]; then
    echo "[1.8/4] ⚠ 风控数据不足 — bypass, paper_trade 继续" | tee -a "$LOG"
fi

# 2. paper_trade 信号
echo "[2/4] Running paper_trade signals..." | tee -a "$LOG"
.venv/bin/python examples/paper_trade_today.py 2>&1 | tee -a "$LOG"

# 3. Forward OOS 监控 (set +e 临时关掉, 因为 monitor 用 exit code 报状态)
echo "[3/4] Forward OOS monitoring..." | tee -a "$LOG"
set +e
.venv/bin/python examples/forward_oos_monitor.py 2>&1 | tee -a "$LOG"
ALERT_CODE=$?
set -e
echo "Alert level code: $ALERT_CODE" | tee -a "$LOG"

# 3.5 Shadow v19.4 (跟踪 + 对比 OOS, 失败不阻塞主流)
echo "[3.5/4] Shadow v19.4 paper_trade (用于 forward OOS A/B 对比)..." | tee -a "$LOG"
set +e
.venv/bin/python examples/paper_trade_v19_4.py 2>&1 | tee -a "$LOG"
set -e

# 3.6 Multi-Agent Debate (rule-based, 失败不阻塞主流)
echo "[3.6/4] Multi-Agent Debate (Bull/Bear/Neutral 对当日 picks 投票)..." | tee -a "$LOG"
set +e
.venv/bin/python examples/multi_agent_debate.py 2>&1 | tee -a "$LOG"
set -e

# 3.7 Shadow v19.6 paper_trade (since 2026-05-27 v19.10 production upgrade)
# v19.6 fallback shadow, 跟主 production v19.10 平行跑, 12 月 forward A/B 真实对比
echo "[3.7/4] Shadow v19.6 paper_trade (v19.10 production fallback A/B)..." | tee -a "$LOG"
set +e
.venv/bin/python examples/paper_trade_v19_6.py 2>&1 | tee -a "$LOG"
set -e

# 4. 重新生成 dashboard HTML (xlsx 同步已弃用 — trades.jsonl 是 source of truth)
echo "[4/4] Rendering dashboard report..." | tee -a "$LOG"
.venv/bin/python dashboard/render_report.py 2>&1 | tee -a "$LOG"

# red(3) 或 black(4) → 打开 dashboard 提醒
if [ $ALERT_CODE -ge 3 ]; then
    echo "⚠️  HIGH ALERT — Open dashboard + 系统通知" | tee -a "$LOG"
    open reports/daily_report_$(date +%Y%m%d).html || true
fi

echo "=== $(date) Daily check done (exit=$ALERT_CODE) ===" | tee -a "$LOG"
exit $ALERT_CODE
