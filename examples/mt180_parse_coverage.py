"""mt180 Top-500 TDX 公式 parse 覆盖率统计.

读 `data_cache/mt180/top_n.jsonl` → 跑 compile_tdx → 输出 `data_cache/mt180/parse_coverage.csv`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from claude_finance.tdx_parser import compile_tdx

ROOT = Path("/Volumes/SSD/finance/claude_finance")
INPUT = ROOT / "data_cache" / "mt180" / "top_n.jsonl"
OUTPUT = ROOT / "data_cache" / "mt180" / "parse_coverage.csv"


def main() -> None:
    rows: list[dict] = []
    status_count: Counter[str] = Counter()
    unsup_overall: Counter[str] = Counter()

    with INPUT.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            iid = d.get("id", "")
            name = d.get("name", "")
            formula = d.get("formula", "") or ""
            cf = compile_tdx(formula)
            status_count[cf.status] += 1
            for u in cf.unsupported_funcs:
                unsup_overall[u] += 1
            rows.append({
                "id": iid,
                "name": name,
                "parse_status": cf.status,
                "n_output_cols": len(cf.output_cols),
                "output_cols": "|".join(cf.output_cols),
                "unsupported_funcs": "|".join(cf.unsupported_funcs),
                "error": cf.error or "",
                "formula_len": len(formula),
            })

    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    total = len(df)
    print(f"Top {total}: ok={status_count['ok']}, "
          f"skipped (no output / drawing) ={status_count['skipped']}, "
          f"unsupported_func={status_count['unsupported']}, "
          f"parse_error={status_count['error']}")
    print("\nTop 20 unsupported funcs:")
    for fn, cnt in unsup_overall.most_common(20):
        print(f"  {fn:24s} {cnt}")
    print(f"\nwrote: {OUTPUT}")


if __name__ == "__main__":
    main()
