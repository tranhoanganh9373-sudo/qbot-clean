"""拉沪深300成分股 + 生成 qlib_baidu/instruments/csi300.txt.

源: 东财 push2delay clist fs=b:BK0500

输出:
  data_cache/csi300_constituents.csv  - code, name
  data_cache/qlib_baidu/instruments/csi300.txt  - qlib 格式 (sym\\tstart\\tend)

Run:  python examples/fetch_csi300_constituents.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data_cache" / "csi300_constituents.csv"
QLIB_INSTR = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "csi300.txt"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0"
URL = "https://push2delay.eastmoney.com/api/qt/clist/get"


def fetch_csi300() -> list[dict]:
    rows = []
    page = 1
    while True:
        params = {
            "pn": str(page), "pz": "100", "po": "1", "np": "1",
            "fltt": "2", "invt": "2",
            "fs": "b:BK0500", "fields": "f12,f13,f14",
        }
        r = requests.get(URL, params=params,
                          headers={"User-Agent": UA}, timeout=20)
        d = r.json()
        items = (d.get("data") or {}).get("diff") or []
        total = (d.get("data") or {}).get("total") or 0
        if not items:
            break
        for it in items:
            mkt = "sh" if it.get("f13") == 1 else "sz"
            rows.append({
                "code": str(it.get("f12", "")).zfill(6),
                "name": it.get("f14", ""),
                "market": mkt,
            })
        if len(rows) >= total:
            break
        page += 1
        if page > 10:
            break
    return rows


def main():
    print("[1/2] 拉 CSI300 成分股 ...")
    rows = fetch_csi300()
    print(f"  共 {len(rows)} 只")

    df = pd.DataFrame(rows).drop_duplicates(subset=["code"]).sort_values("code")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  写 {OUT_CSV}")

    print("\n[2/2] 生成 qlib instruments file ...")
    QLIB_INSTR.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, r in df.iterrows():
        prefix = "SH" if r["market"] == "sh" else "SZ"
        sym = f"{prefix}{r['code']}"
        lines.append(f"{sym}\t2014-01-01\t2099-12-31")
    QLIB_INSTR.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  写 {QLIB_INSTR} ({len(lines)} 行)")
    print("\n  现可在 qlib 用 instruments='csi300'")


if __name__ == "__main__":
    main()
