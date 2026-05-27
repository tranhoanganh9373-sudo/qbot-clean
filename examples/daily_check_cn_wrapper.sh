#!/bin/bash
# daily_check wrapper — 以北京时间为准 trigger daily_check.sh
#
# 为什么需要 wrapper:
#   - Mac 在美国 (PDT/PST), launchd StartCalendarInterval 用本机时间
#   - A 股以北京时间为准, 收盘 15:00 CST, daily_check 应 16:30 CST 跑
#   - 北京 16:30 = PDT 01:30 (夏令时) / PST 00:30 (冬令时)
#
# 设计:
#   - launchd plist 同时配 0:30 和 1:30 两个 trigger 覆盖 DST 全年
#   - 此 wrapper 用 TZ=Asia/Shanghai 判断当前是否在 16:25-16:59 窗口 + 工作日
#   - 在窗口内 → exec daily_check.sh; 否则 → no-op exit (避免 DST 误触)

ROOT=/Volumes/SSD/finance/claude_finance
TARGET_HOUR=16
WINDOW_START_MIN=25
WINDOW_END_MIN=59

CN_HOUR=$(TZ=Asia/Shanghai date +%-H)
CN_MIN=$(TZ=Asia/Shanghai date +%-M)
CN_WDAY=$(TZ=Asia/Shanghai date +%u)  # 1=Mon ... 7=Sun
CN_FULL=$(TZ=Asia/Shanghai date "+%Y-%m-%d %H:%M:%S %A")

echo "[wrapper] $(date '+%Y-%m-%d %H:%M:%S %Z') — Beijing time: $CN_FULL"

if [ "$CN_WDAY" -ge 6 ]; then
    echo "[wrapper] 北京周末 (wday=$CN_WDAY), 跳过 daily_check."
    exit 0
fi

if [ "$CN_HOUR" -ne "$TARGET_HOUR" ]; then
    echo "[wrapper] 北京小时 ${CN_HOUR}!=${TARGET_HOUR}, 跳过 (DST 边界外另一 trigger)."
    exit 0
fi
if [ "$CN_MIN" -lt "$WINDOW_START_MIN" ] || [ "$CN_MIN" -gt "$WINDOW_END_MIN" ]; then
    echo "[wrapper] 北京分钟 ${CN_MIN} 不在 [${WINDOW_START_MIN}, ${WINDOW_END_MIN}] 窗口, 跳过."
    exit 0
fi

echo "[wrapper] 北京 ${CN_HOUR}:${CN_MIN} 在窗口内, 启动 daily_check.sh"
exec /bin/bash "$ROOT/examples/daily_check.sh"
