"""Sidecar factor config panel — 读 paper_trade_today.py 模块常量.

数据源: examples/paper_trade_today.py, 用 ast.parse 抓 module-level Assign 节点.
输出: HTML table, 显示 v19.6 / v19.4 哪个 active + 各 λ 数值.

关心的常量:
  USE_V19_6_SIDECAR        : bool   (v19.6 是否启用 → production)
  USE_V19_4_SIDECAR        : bool   (v19.4 fallback, paper_trade_today 默认 False)
  SIDECAR_LAMBDA_AMP_20D   : float  (v19.6 λ, default 0.30, sign -1)
  SIDECAR_LAMBDA_M5        : float  (v19.4 λ_m5, default 0.10, sign -1)
  SIDECAR_LAMBDA_M20       : float  (v19.4 λ_m20, default 0.10, sign -1)
"""
from __future__ import annotations

import ast
import html
from pathlib import Path
from typing import Any

CONSTANTS_OF_INTEREST = {
    "USE_V19_10_STACKED",
    "USE_V19_6_SIDECAR",
    "USE_V19_4_SIDECAR",
    "SIDECAR_LAMBDA_AMP_20D",
    "SIDECAR_LAMBDA_JZF",
    "SIDECAR_LAMBDA_M5",
    "SIDECAR_LAMBDA_M20",
}


def _parse_constants(source_path: Path) -> dict[str, Any]:
    """从 source_path 解析 module-level 常量赋值, 仅保留 CONSTANTS_OF_INTEREST."""
    if not source_path.exists():
        return {}
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return {}

    result: dict[str, Any] = {}
    for node in tree.body:
        # 仅看 module-level Assign / AnnAssign
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id not in CONSTANTS_OF_INTEREST:
                continue
            try:
                result[target.id] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
    return result


def _placeholder_html(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:32px 16px;">'
        f"{message}"
        "</div>"
    )


def _active_label(constants: dict[str, Any]) -> str:
    """根据 toggle 决定 active sidecar (v19.10 优先)."""
    if constants.get("USE_V19_10_STACKED", False):
        return "v19.10 stacked (amp_imb_20d + JZF)"
    if constants.get("USE_V19_6_SIDECAR", False):
        return "v19.6"
    if constants.get("USE_V19_4_SIDECAR", False):
        return "v19.4"
    return "v19.1 (no sidecar)"


def build_factor_config_panel(paper_trade_path: Path) -> str:
    """主入口: 返回可直接塞到 template `{{factor_config}}` 的 HTML 片段."""
    consts = _parse_constants(paper_trade_path)
    if not consts:
        return _placeholder_html(
            f"无法解析 <code>{html.escape(str(paper_trade_path))}</code> 的 sidecar 常量.<br>"
            "(文件不存在或 SyntaxError)"
        )

    active = _active_label(consts)
    use_v1910 = bool(consts.get("USE_V19_10_STACKED", False))
    use_v196 = bool(consts.get("USE_V19_6_SIDECAR", False))
    use_v194 = bool(consts.get("USE_V19_4_SIDECAR", False))

    lambda_amp = consts.get("SIDECAR_LAMBDA_AMP_20D")
    lambda_jzf = consts.get("SIDECAR_LAMBDA_JZF")
    lambda_m5 = consts.get("SIDECAR_LAMBDA_M5")
    lambda_m20 = consts.get("SIDECAR_LAMBDA_M20")

    def _badge(on: bool) -> str:
        return (
            '<span class="badge badge-on">ON</span>' if on
            else '<span class="badge badge-off">OFF</span>'
        )

    def _lambda(v) -> str:
        return f"{v:.2f}" if isinstance(v, (int, float)) else "-"

    # active 高亮逻辑: v19.10 优先, 然后 v19.6, 最后 v19.4
    cls_v1910 = ' class="active-row"' if use_v1910 else ""
    cls_v196 = ' class="active-row"' if (use_v196 and not use_v1910) else ""
    cls_v194 = ' class="active-row"' if (use_v194 and not use_v196 and not use_v1910) else ""

    if use_v1910:
        formula = (
            f"− {_lambda(lambda_amp)} · z(amp_imb_20d) "
            f"+ {_lambda(lambda_jzf)} · z(JZF)"
        )
    elif use_v196:
        formula = f"− {_lambda(lambda_amp)} · z(amp_imb_20d)"
    elif use_v194:
        formula = (
            f"− {_lambda(lambda_m5)} · z(m5) "
            f"− {_lambda(lambda_m20)} · z(m20)"
        )
    else:
        formula = ""

    table = f'''
<table class="data">
  <thead>
    <tr>
      <th>Sidecar</th>
      <th>Toggle</th>
      <th>Factor</th>
      <th>λ</th>
      <th>Sign</th>
      <th>OOS Calmar</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr{cls_v1910}>
      <td><b>v19.10</b> ⭐</td>
      <td>{_badge(use_v1910)}</td>
      <td>amp_imb_20d + JZF</td>
      <td>{_lambda(lambda_amp)} + {_lambda(lambda_jzf)}</td>
      <td>-1 / +1</td>
      <td>2.12</td>
      <td><b>production (stacked, +64% vs v19.6)</b></td>
    </tr>
    <tr{cls_v196}>
      <td><b>v19.6</b></td>
      <td>{_badge(use_v196 and not use_v1910)}</td>
      <td>amp_imb_20d</td>
      <td>{_lambda(lambda_amp)}</td>
      <td>-1</td>
      <td>1.29</td>
      <td>shadow (paper_trade_v19_6.py, 12月 A/B)</td>
    </tr>
    <tr{cls_v194}>
      <td><b>v19.4</b></td>
      <td>{_badge(use_v194)}</td>
      <td>margin_5d + margin_20d</td>
      <td>{_lambda(lambda_m5)} + {_lambda(lambda_m20)}</td>
      <td>-1 / -1</td>
      <td>0.62</td>
      <td>fallback (OFF)</td>
    </tr>
    <tr>
      <td><b>v19.1</b></td>
      <td>{_badge(not use_v1910 and not use_v196 and not use_v194)}</td>
      <td>(no sidecar)</td>
      <td>-</td>
      <td>-</td>
      <td>0.77</td>
      <td>train24 baseline</td>
    </tr>
  </tbody>
</table>
<div style="margin-top:10px; font-size:13px;">
  Active: <b>{active}</b>
  &middot; final_score = z(pred) {formula}
</div>
<div style="margin-top:4px; font-size:11px; color:var(--muted);">
  数据源: <code>{html.escape(str(paper_trade_path))}</code> (ast 解析 module-level 常量)
</div>
'''
    return table
