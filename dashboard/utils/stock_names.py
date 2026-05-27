"""Stock code → name lookup for dashboard charts.

数据源: data_cache/csi300_constituents.csv (300 stocks). future: 可扩 csi500/top1500.

用法:
    from dashboard.utils.stock_names import code_with_name, name_of, load_name_map
    code_with_name("SH600519")  # → "SH600519 贵州茅台"
    name_of("SH600519")          # → "贵州茅台" or None if not found
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DC = ROOT / "data_cache"


@lru_cache(maxsize=1)
def load_name_map() -> dict[str, str]:
    """{SH600519: '贵州茅台', SZ300347: '泰格医药', ...}.

    cached, called once per session.
    """
    name_map: dict[str, str] = {}
    csi_path = DC / "csi300_constituents.csv"
    if csi_path.exists():
        try:
            df = pd.read_csv(csi_path)
            for _, r in df.iterrows():
                code = str(r["code"]).zfill(6)
                mkt = str(r.get("market", "")).lower().strip()
                prefix = "SH" if mkt == "sh" or code.startswith(("6", "9")) else "SZ"
                name = str(r["name"]).strip()
                name_map[f"{prefix}{code}"] = name
        except Exception:
            pass
    return name_map


def name_of(sym: str) -> str | None:
    """SH600519 → '贵州茅台' or None. 支持 SH600519 / 600519 / sh600519 多格式."""
    name_map = load_name_map()
    if sym in name_map:
        return name_map[sym]
    if sym.upper() in name_map:
        return name_map[sym.upper()]
    if sym.isdigit() and len(sym) == 6:
        prefix = "SH" if sym.startswith(("6", "9")) else "SZ"
        return name_map.get(f"{prefix}{sym}")
    return None


def code_with_name(sym: str, sep: str = " ", fallback_code_only: bool = True) -> str:
    """SH600519 → 'SH600519 贵州茅台'. 无 name 时 fallback 返 code."""
    nm = name_of(sym)
    if nm:
        return f"{sym}{sep}{nm}"
    return sym if fallback_code_only else f"{sym}{sep}?"


if __name__ == "__main__":
    nm = load_name_map()
    print(f"loaded {len(nm)} stock names")
    for sym in ["SH600519", "SZ300347", "SH600547", "SH688396", "SH999999"]:
        print(f"  {sym}: {code_with_name(sym)}")
