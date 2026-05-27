"""matplotlib Figure → base64 PNG data URI.

借鉴 references/Lean/Report/ReportCharts.py:56-62 的 fig_to_base64 helper,
去掉了 Lean 的 .NET clr 依赖以及写临时文件的步骤 (直接走 BytesIO).
"""
from __future__ import annotations

import io
from base64 import b64encode
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def fig_to_base64(fig: "Figure", dpi: int = 150) -> str:
    """Render fig to PNG, return `data:image/png;base64,...` URI string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        要编码的 figure 对象.
    dpi : int
        输出 PNG 分辨率, 默认 150 (Lean 用 200, 这里取 150 平衡文件大小).

    Returns
    -------
    str
        形如 `data:image/png;base64,iVBOR...` 可直接塞到 <img src="..."> 的 URI.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = b64encode(buf.read()).decode("utf-8").replace("\n", "")
    buf.close()
    return f"data:image/png;base64,{encoded}"
