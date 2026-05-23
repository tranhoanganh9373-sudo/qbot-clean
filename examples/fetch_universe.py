"""拉全 A 股 universe（剔除 ST / 退市 / 北交所暂不要）.

数据源: 东财 push2 clist (m:0+t:6/80 深A, m:1+t:2/23 沪A)

输出: data_cache/universe.csv (code, name, market, list_date)
剔除: 名称含 ST / *ST / 退 / 退市
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data_cache" / "universe.csv"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0"
PUSH2_HOSTS = [
    "https://push2delay.eastmoney.com/api/qt/clist/get",
    "https://push2.eastmoney.com/api/qt/clist/get",
]


def fetch_market(fs: str, label: str, retries: int = 3) -> list[dict]:
    """fs: 东财市场过滤串. 见 push2 API 文档."""
    rows = []
    page = 1
    while True:
        params = {
            "pn": str(page), "pz": "100", "po": "1", "np": "1",
            "fltt": "2", "invt": "2", "fs": fs,
            "fields": "f12,f14,f13,f189",
        }
        last_err = None
        d = None
        for url in PUSH2_HOSTS:
            for _ in range(retries):
                try:
                    r = requests.get(url, params=params,
                                      headers={"User-Agent": UA}, timeout=20)
                    if r.status_code != 200:
                        last_err = RuntimeError(f"HTTP {r.status_code}")
                        continue
                    d = r.json()
                    break
                except Exception as e:
                    last_err = e
            if d is not None:
                break
        if d is None:
            raise last_err
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
                "list_date": str(it.get("f189", "")),
            })
        if len(rows) >= total or len(items) < 100:
            break
        page += 1
        if page > 100:
            break
    print(f"  {label}: {len(rows)} 只")
    return rows


def is_st(name: str) -> bool:
    if not name:
        return True
    n = name.replace(" ", "").upper()
    return ("ST" in n) or ("退" in n) or ("PT" in n)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("[1/2] 拉沪深两市全 A 股 ...")
    all_rows = []
    all_rows += fetch_market("m:1+t:2,m:1+t:23", "沪市A (含科创板)")
    all_rows += fetch_market("m:0+t:6,m:0+t:80", "深市A (含创业板)")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["code"])
    print(f"\n[2/2] 全市场: {len(df)}, 剔除 ST/退 ...")
    df_clean = df[~df["name"].apply(is_st)].copy()
    print(f"  剔除后: {len(df_clean)} 只")
    print(f"  按市场: sh={len(df_clean[df_clean['market']=='sh'])} "
          f"sz={len(df_clean[df_clean['market']=='sz'])}")

    df_clean = df_clean.sort_values("code").reset_index(drop=True)
    df_clean.to_csv(OUT, index=False)
    print(f"\n输出: {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
    print(f"\n  前 5 只: {df_clean.head()[['code','name','market']].to_dict('records')}")


if __name__ == "__main__":
    main()
