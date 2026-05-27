"""User Notes — 自由 textarea 用户笔记 panel.

localStorage key `cf_user_inputs_v1.__notes` (与 positions_pnl 共用 storage).
Export 时 JS 把 textarea.value 写入 JSON `__notes` 字段 →
examples/merge_user_input.py 把它写回 portfolio.xlsx 'Notes' sheet
(在固定参考文本之后, 用一个 separator 隔开, 不破坏现有 NOTES_TEXT).
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

NOTES_SHEET = "Notes"
USER_NOTES_MARKER = "═══ User Notes (from dashboard) ═══"


def _load_existing_user_notes(portfolio_xlsx: Path) -> str:
    """从 portfolio.xlsx Notes sheet 读 server 侧已存 user notes (separator 之后).

    返回 plain text. 找不到/读取失败 → 空 str (前端 textarea 留空).
    """
    if not portfolio_xlsx.exists():
        return ""
    try:
        df = pd.read_excel(portfolio_xlsx, sheet_name=NOTES_SHEET, header=None)
    except Exception:
        return ""
    if df.empty:
        return ""
    # find row containing USER_NOTES_MARKER in col 0
    col0 = df.iloc[:, 0].astype(str).fillna("")
    mask = col0.str.contains(USER_NOTES_MARKER, regex=False, na=False)
    if not mask.any():
        return ""
    marker_idx = int(mask[mask].index[0])
    # collect rows after marker (col 0 only — user free-form text)
    lines: list[str] = []
    for i in range(marker_idx + 1, len(df)):
        val = df.iat[i, 0]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            lines.append("")
            continue
        lines.append(str(val))
    # strip trailing empty
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


_CSS = """
<style>
.user-notes-panel .notes-btn {
    padding: 4px 10px; font-size: 11px; cursor: pointer;
    background: rgba(37,99,235,0.08); color: var(--accent, #2563eb);
    border: 1px solid rgba(37,99,235,0.20); border-radius: 4px;
    margin-right: 8px;
}
.user-notes-panel .notes-btn:hover { filter: brightness(1.1); }
.user-notes-panel textarea#notes-textarea {
    width: 100%;
    min-height: 140px;
    box-sizing: border-box;
    padding: 10px 12px;
    font-family: "SF Mono", Menlo, Monaco, Consolas, "PingFang SC",
                 "Microsoft YaHei", monospace;
    font-size: 13px;
    color: var(--fg);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    line-height: 1.55;
    resize: vertical;
}
.user-notes-panel textarea#notes-textarea:focus {
    outline: none;
    border-color: var(--accent);
}
.user-notes-panel .notes-hint {
    margin-top: 6px;
    font-size: 12px;
    color: var(--muted);
}
.user-notes-panel .notes-saved {
    font-size: 12px;
    color: var(--muted);
}
.user-notes-panel .notes-saved.flash { color: var(--green); }
</style>
"""


def build_user_notes_panel(portfolio_xlsx: Path) -> str:
    """渲染 user notes textarea panel.

    优先用 xlsx 中已 merge 过的 user notes 作为 textarea 初值 (server-side
    source of truth). 前端 JS load 时若 localStorage `__notes` 有值则覆盖.
    """
    initial = _load_existing_user_notes(portfolio_xlsx)
    initial_escaped = html.escape(initial)
    return (
        _CSS
        + '<div class="user-notes-panel">'
        + '<textarea id="notes-textarea" '
        + 'placeholder="今日感想 / 持仓决策记录 / 待办...  失焦自动保存到 localStorage,'
        + ' 导出 user_input.json 时含 __notes 字段, merge 后落 portfolio.xlsx Notes sheet."'
        + '>'
        + initial_escaped
        + '</textarea>'
        + '<div class="notes-hint">'
        + '<button type="button" id="notes-insert-date" class="notes-btn" '
        + 'title="光标处插入 [YYYY-MM-DD HH:MM] 时间戳">📅 插入时间戳</button>'
        + '<button type="button" id="notes-clear" class="notes-btn" '
        + 'title="清空 (有确认)" style="background:rgba(220,38,38,0.10);">🗑 清空</button>'
        + '<span id="notes-saved-indicator" class="notes-saved">未编辑</span>'
        + ' · 失焦自动存 localStorage, 导出 JSON 含本内容.'
        + '</div>'
        + '</div>'
    )
