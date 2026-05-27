"""Fetch CSI300 资金流(主力/超大单/大单/中单/小单) daily history 2014-2020 IS.

数据源 (sandbox-OK, 2026-05-26 新发现):
  http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/
  MoneyFlow.ssl_qsfx_lscjfb?page=N&num=500&sort=opendate&asc=0&daima=sh600519

字段 (Sina Schema):
  opendate     交易日
  trade        收盘价
  changeratio  涨跌幅
  turnover     换手率
  netamount    主力净流入额 (r0_net + r1_net)
  ratioamount  主力净流入占比
  r0           超大单总成交额
  r1           大单总成交额
  r2           中单总成交额
  r3           小单总成交额
  r0_net       超大单净流入
  r1_net       大单净流入
  r2_net       中单净流入
  r3_net       小单净流入

历史深度: Sina 单股可拉回 2010-03; 远超 push2his 的 120 日上限.

策略:
- per-stock paginate: page=1..N, num=500, 直到 last opendate < 2014-01-01
- 8-worker ThreadPool, retry 3x with backoff
- atomic write data_cache/fund_flow/{code}.parquet
- 结束合并 data_cache/fund_flow/fund_flow_csi300.parquet

【与 fund_flow.py docstring 偏离】fund_flow.py 写"sandbox 不可行";那是 push2his
路由; Sina money.finance 是独立端点, 实测可达. 本脚本 IS 期专用; OOS 期(2021+)
绝不抓 (严格 OOS 协议).

run:
  python examples/fetch_fund_flow_csi300.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
OUT_DIR = ROOT / "data_cache" / "fund_flow"
OUT_COMBINED = OUT_DIR / "fund_flow_csi300.parquet"

# 严格 IS: 2014-01-01 ~ 2020-12-31. 因子需 60d window 缓冲, 落 2021-03 cutoff.
# 但 IS 因子选择 只用 ≤ 2020-12-31 数据 → fetcher 抓 2014-01-01 ~ 2020-12-31.
IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")

NUM_PER_PAGE = 500
MAX_PAGES = 10  # 5000 行足够 ~14 年
# Sina money.finance 限频严格: 8-worker 突发 ≈30s 后 HTTP 456 (IP 封 ~8min).
# 实测 1-worker + 1s 间隔 长期稳定. 不要调高.
WORKERS = 1
PER_REQUEST_SLEEP = 1.0
REQUEST_TIMEOUT = 20
RETRY_TIMES = 5
RETRY_BACKOFF = 5.0  # 第 N 次重试 sleep N*5s, 给 ban 留恢复时间

SINA_URL = (
    "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "MoneyFlow.ssl_qsfx_lscjfb"
)


def _code_to_sina_symbol(code: str, market: str) -> str:
    return f"{market}{str(code).zfill(6)}"


def _fetch_page(symbol: str, page: int) -> list[dict]:
    """Single page fetch with retry. Throttled to respect Sina rate limits."""
    params = {
        "page": page,
        "num": NUM_PER_PAGE,
        "sort": "opendate",
        "asc": 0,
        "daima": symbol,
    }
    last_err: Exception | None = None
    for attempt in range(RETRY_TIMES):
        try:
            r = requests.get(SINA_URL, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                try:
                    data = json.loads(r.text)
                    if isinstance(data, list):
                        time.sleep(PER_REQUEST_SLEEP)
                        return data
                    return []
                except json.JSONDecodeError as e:
                    last_err = e
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
            # 456 = "拒绝访问" IP ban; longer cooldown
            cooldown = RETRY_BACKOFF * (attempt + 1) * (
                10 if r.status_code == 456 else 1
            )
            last_err = RuntimeError(f"HTTP {r.status_code}")
            time.sleep(cooldown)
        except requests.RequestException as e:
            last_err = e
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise RuntimeError(
        f"sina fetch failed for {symbol} page={page}: {last_err!r}"
    )


def _fetch_stock(code: str, market: str) -> tuple[str, pd.DataFrame, str]:
    """Fetch all IS-period rows for one stock. Returns (code, df, status)."""
    symbol = _code_to_sina_symbol(code, market)
    out_path = OUT_DIR / f"{code}.parquet"
    if out_path.exists():
        try:
            df = pd.read_parquet(out_path)
            if not df.empty:
                return code, df, "skip-cached"
        except Exception:
            out_path.unlink()  # corrupt, refetch

    rows: list[dict] = []
    try:
        for page in range(1, MAX_PAGES + 1):
            data = _fetch_page(symbol, page)
            if not data:
                break
            rows.extend(data)
            last_date = pd.Timestamp(data[-1]["opendate"])
            if last_date < IS_START:
                break
    except RuntimeError as e:
        return code, pd.DataFrame(), f"fail:{e}"

    if not rows:
        return code, pd.DataFrame(), "empty"

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["opendate"])
    # Strict IS filter (drop OOS-period rows even if returned by Sina)
    df = df[(df["date"] >= IS_START) & (df["date"] <= IS_END)]
    if df.empty:
        return code, df, "empty-is"

    # Coerce numeric (Sina returns strings)
    numeric_cols = [
        "trade", "changeratio", "turnover", "netamount", "ratioamount",
        "r0", "r1", "r2", "r3", "r0_net", "r1_net", "r2_net", "r3_net",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["code"] = str(code).zfill(6)

    # Derived columns aligning to user spec
    df["net_super_big"] = df["r0_net"]
    df["net_big"] = df["r1_net"]
    df["net_mid"] = df["r2_net"]
    df["net_small"] = df["r3_net"]
    df["net_main"] = df["r0_net"] + df["r1_net"]
    total_vol = df["r0"] + df["r1"] + df["r2"] + df["r3"]
    df["total_amount"] = total_vol
    # pct_super_big = r0_net / total amount (signed %)
    df["pct_super_big"] = df["r0_net"] / total_vol.where(total_vol > 0)
    df["pct_big"] = df["r1_net"] / total_vol.where(total_vol > 0)
    df["pct_main"] = df["net_main"] / total_vol.where(total_vol > 0)

    df = (
        df[[
            "code", "date", "trade",
            "net_main", "net_super_big", "net_big", "net_mid", "net_small",
            "pct_super_big", "pct_big", "pct_main",
            "r0", "r1", "r2", "r3", "total_amount", "turnover",
        ]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Atomic write
    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out_path)
    return code, df, "ok"


def main() -> int:
    if not CSI300_PATH.exists():
        print(f"FATAL: {CSI300_PATH} 缺", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    csi["market"] = csi["market"].str.lower()

    targets = list(zip(csi["code"].tolist(), csi["market"].tolist()))
    print(
        f"[init] CSI300 universe = {len(targets)} stocks; "
        f"IS = {IS_START.date()} ~ {IS_END.date()}; "
        f"workers={WORKERS}",
        flush=True,
    )

    t0 = time.time()
    n_ok = n_skip = n_empty = n_fail = 0
    fail_codes: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(_fetch_stock, c, m): (c, m) for c, m in targets
        }
        done = 0
        for fut in as_completed(futures):
            code, _df, status = fut.result()
            done += 1
            if status == "ok":
                n_ok += 1
            elif status == "skip-cached":
                n_skip += 1
            elif status in {"empty", "empty-is"}:
                n_empty += 1
            else:
                n_fail += 1
                fail_codes.append((code, status))
            if done % 25 == 0 or done == len(targets):
                elapsed = time.time() - t0
                print(
                    f"  [{done}/{len(targets)}] "
                    f"ok={n_ok} skip={n_skip} empty={n_empty} fail={n_fail} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    print(
        f"\n[fetch] DONE in {time.time()-t0:.1f}s | "
        f"ok={n_ok}, cached={n_skip}, empty={n_empty}, fail={n_fail}",
        flush=True,
    )
    if fail_codes:
        print(f"[fail] sample (first 10): {fail_codes[:10]}", flush=True)

    # Combine per-stock parquets
    parts = []
    for code in csi["code"]:
        p = OUT_DIR / f"{code}.parquet"
        if p.exists():
            try:
                parts.append(pd.read_parquet(p))
            except Exception as e:
                print(f"  [combine] skip corrupt {code}: {e}", flush=True)
    if not parts:
        print("FATAL: nothing to combine", file=sys.stderr)
        return 1

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
    tmp = OUT_COMBINED.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.rename(OUT_COMBINED)
    print(
        f"[combine] {OUT_COMBINED.name}: {len(combined):,} rows × "
        f"{combined['code'].nunique()} codes, "
        f"date {combined['date'].min().date()} ~ "
        f"{combined['date'].max().date()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
