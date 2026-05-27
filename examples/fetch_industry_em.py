"""Fetch 申万一级行业分类 + 成分股 → industry_membership.parquet.

数据源切换: 因东财 push2 API 在 sandbox 大量节流, 改用申万 (SW Hi-Index) 一级行业:
- akshare ak.sw_index_first_info()   返回 31 个一级行业 (代码+名称+成份个数+估值)
- akshare ak.index_component_sw(symbol='8010XX')  返回单行业成份股 (证券代码+权重+计入日期)

申万一级行业是 A 股标准行业分类, 2014-2026 期间分类相对稳定 (2014/2021 两次主调整),
适合用于 industry_adj_ret 因子 (sector-relative momentum).

Sandbox proxy: 系统默认 127.0.0.1:7897 代理路由不通,
本脚本 monkey-patch requests.Session(trust_env=False) + UA 直连源站.

Known limitation:
- akshare 给的是**当前 snapshot** (2021-12 或最新调整后的归属).
  2014 期的旧分类已不可逆向追溯; CAVEAT 写入 memory + 报告.
- 部分股票可能未被任何 SW 一级行业覆盖 (e.g. ST/退市/新股), 报告中给 coverage 数字.

Output:
  data_cache/industry/industry_boards.parquet          (raw SW 行业表)
  data_cache/industry/industry_membership.parquet
    columns: [code, industry_name, industry_code, weight, include_date, fetch_date]
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Clear proxy env (defensive)
for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

# Monkey-patch requests.Session BEFORE importing akshare
import requests  # noqa: E402

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/121.0.0.0 Safari/537.36")
_orig_session_init = requests.Session.__init__


def _patched_session_init(self):
    _orig_session_init(self)
    self.trust_env = False
    self.headers.update({"User-Agent": _UA})


requests.Session.__init__ = _patched_session_init  # type: ignore

import akshare as ak  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data_cache" / "industry"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "industry_membership.parquet"
BOARDS_PATH = OUT_DIR / "industry_boards.parquet"


def fetch_boards() -> pd.DataFrame:
    """31 SW level-1 industries."""
    print("[boards] fetching 申万一级行业 ak.sw_index_first_info() ...",
          flush=True)
    t0 = time.time()
    df = ak.sw_index_first_info()
    print(f"[boards] {len(df)} industries in {time.time() - t0:.1f}s",
          flush=True)
    df.to_parquet(BOARDS_PATH, index=False)
    print(df.to_string(index=False), flush=True)
    return df


def fetch_one_board(name: str, code_full: str,
                    retries: int = 3) -> pd.DataFrame:
    """code_full like '801010.SI' → strip suffix for index_component_sw."""
    code = code_full.split(".")[0]
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = ak.index_component_sw(symbol=code)
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    print(f"  [fail] {name} ({code}): {last_err}", flush=True)
    return pd.DataFrame()


def main() -> int:
    t_all = time.time()
    boards = fetch_boards()

    rows = []
    n = len(boards)
    fail = 0
    for i, row in boards.iterrows():
        name = str(row["行业名称"])
        code_full = str(row["行业代码"])
        cons = fetch_one_board(name, code_full)
        if cons.empty:
            fail += 1
        else:
            for _, r in cons.iterrows():
                rows.append({
                    "code": str(r["证券代码"]).zfill(6),
                    "industry_name": name,
                    "industry_code": code_full,
                    "weight": float(r.get("最新权重", 0) or 0),
                    "include_date": str(r.get("计入日期", "")),
                })
        print(f"  [{i + 1}/{n}] {name}: "
              f"{len(cons) if not cons.empty else 0} stocks, "
              f"elapsed: {time.time() - t_all:.0f}s", flush=True)
        time.sleep(0.3)

    if not rows:
        print("FATAL: no rows collected", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df["fetch_date"] = pd.Timestamp.today().normalize()
    # SW level-1: each stock belongs to ≤1 industry; dedup defensively
    df = df.drop_duplicates(subset=["code", "industry_code"])
    df = df.sort_values(["code", "industry_code"]).reset_index(drop=True)

    df.to_parquet(OUT_PATH, index=False)
    n_codes = df["code"].nunique()
    n_inds = df["industry_code"].nunique()
    print(f"\n[output] {OUT_PATH}", flush=True)
    print(f"  rows: {len(df):,}", flush=True)
    print(f"  unique codes: {n_codes:,}", flush=True)
    print(f"  unique industries: {n_inds}", flush=True)
    print(f"  fail boards: {fail}/{n}", flush=True)
    print(f"  wall time: {time.time() - t_all:.0f}s", flush=True)

    # Coverage vs kline universe + CSI300
    kline_path = ROOT / "data_cache" / "baidu_kline.parquet"
    csi_path = ROOT / "data_cache" / "csi300_constituents.csv"
    if kline_path.exists():
        k = pd.read_parquet(kline_path, columns=["code"])
        k["code"] = k["code"].astype(str).str.zfill(6)
        kline_codes = set(k["code"].unique())
        ind_codes = set(df["code"].unique())
        cover = len(kline_codes & ind_codes) / len(kline_codes) * 100
        print(f"  kline universe coverage: "
              f"{len(kline_codes & ind_codes):,}/{len(kline_codes):,} "
              f"({cover:.1f}%)", flush=True)
    if csi_path.exists():
        csi = pd.read_csv(csi_path, dtype={"code": str})
        csi["code"] = csi["code"].astype(str).str.zfill(6)
        ind_codes = set(df["code"].unique())
        cover = len(set(csi["code"]) & ind_codes) / 300 * 100
        print(f"  CSI300 coverage: "
              f"{len(set(csi['code']) & ind_codes)}/300 "
              f"({cover:.1f}%)", flush=True)

    # Per-industry stock count distribution
    ind_counts = df.groupby(["industry_code", "industry_name"]).size()
    ind_counts = ind_counts.reset_index(name="n_stocks").sort_values(
        "n_stocks", ascending=False,
    )
    print("\n[per-industry counts]")
    print(ind_counts.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
