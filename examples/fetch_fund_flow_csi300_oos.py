"""Phase B Stage 2 — Fetch CSI300 fund flow OOS period (2021-01-01 ~ 2026-04-30).

数据源同 Stage 1 (Sina money.finance MoneyFlow.ssl_qsfx_lscjfb).

为避免污染 Stage 1 的 IS-only per-stock parquets, 本脚本写到独立目录:
  data_cache/fund_flow_oos/{code}.parquet
  data_cache/fund_flow_oos/fund_flow_csi300_oos.parquet  (combined)

参数:
  OOS_START = 2021-01-01  (4 月 gap before OOS start 2021-05 给 60d 因子窗口)
  OOS_END   = 2026-04-30

策略:
- per-stock paginate Sina API, num=500/page, max_pages=10 (5000 行 ~14 年)
- 1-worker + 1s sleep (Phase A 实测必要, 高 worker -> IP ban)
- 失败自动 retry 5x with backoff; 456 重 sleep 50s
- atomic write per-stock parquet
- 结束合并

注意 (严格 OOS 协议):
  本 fetcher 抓的是 Phase A 锁定 sign 之后的 OOS 数据, 用于一次性 sidecar OOS
  跑. **不用于改 sign/horizon/factor 定义**.

Run:
  python examples/fetch_fund_flow_csi300_oos.py
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
OUT_DIR = ROOT / "data_cache" / "fund_flow_oos"
OUT_COMBINED = OUT_DIR / "fund_flow_csi300_oos.parquet"

OOS_START = pd.Timestamp("2021-01-01")
OOS_END = pd.Timestamp("2026-04-30")

NUM_PER_PAGE = 500
MAX_PAGES = 10
WORKERS = 1
PER_REQUEST_SLEEP = 1.0
REQUEST_TIMEOUT = 20
RETRY_TIMES = 5
RETRY_BACKOFF = 5.0

SINA_URL = (
    "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "MoneyFlow.ssl_qsfx_lscjfb"
)


def _code_to_sina_symbol(code: str, market: str) -> str:
    return f"{market}{str(code).zfill(6)}"


def _fetch_page(symbol: str, page: int) -> list[dict]:
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
    symbol = _code_to_sina_symbol(code, market)
    out_path = OUT_DIR / f"{code}.parquet"
    if out_path.exists():
        try:
            df = pd.read_parquet(out_path)
            if not df.empty:
                return code, df, "skip-cached"
        except Exception:
            out_path.unlink()

    rows: list[dict] = []
    try:
        for page in range(1, MAX_PAGES + 1):
            data = _fetch_page(symbol, page)
            if not data:
                break
            rows.extend(data)
            last_date = pd.Timestamp(data[-1]["opendate"])
            if last_date < OOS_START:
                break
    except RuntimeError as e:
        return code, pd.DataFrame(), f"fail:{e}"

    if not rows:
        return code, pd.DataFrame(), "empty"

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["opendate"])
    # Strict OOS filter
    df = df[(df["date"] >= OOS_START) & (df["date"] <= OOS_END)]
    if df.empty:
        return code, df, "empty-oos"

    numeric_cols = [
        "trade", "changeratio", "turnover", "netamount", "ratioamount",
        "r0", "r1", "r2", "r3", "r0_net", "r1_net", "r2_net", "r3_net",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["code"] = str(code).zfill(6)
    df["net_super_big"] = df["r0_net"]
    df["net_big"] = df["r1_net"]
    df["net_mid"] = df["r2_net"]
    df["net_small"] = df["r3_net"]
    df["net_main"] = df["r0_net"] + df["r1_net"]
    total_vol = df["r0"] + df["r1"] + df["r2"] + df["r3"]
    df["total_amount"] = total_vol
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
        f"[init] CSI300 OOS universe = {len(targets)} stocks; "
        f"OOS = {OOS_START.date()} ~ {OOS_END.date()}; "
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
            elif status in {"empty", "empty-oos"}:
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
