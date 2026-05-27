"""Backfill IS 2014-2017 daily kline for CSI300 via mootdx (frequency=4).

Goal: 修 baidu_kline.parquet CSI300 2014-2017 稀疏问题 (仅 10-22 stocks/月)
      让 Phase A 因子 IS IC 跑 ≥60 月 (而非 35-47 月).

约束:
  - 不动 baidu_kline.parquet 主表
  - 4 workers (轻量, 不与 PID 79571 的 5m fetch 抢资源)
  - 落 data_cache/kline_2014_2017_csi300_backfill.parquet (独立)
  - 0 neg close
  - paginate until enough history; 过滤到 2014-01-01 ~ 2017-12-31

schema (match baidu_kline.parquet 关键字段):
  code (6-digit str), date (datetime), open, close, high, low, vol (int), amount (float)

CLI:
  python examples/fetch_mootdx_daily_2014_2017_csi300.py
  python examples/fetch_mootdx_daily_2014_2017_csi300.py --limit 5
  python examples/fetch_mootdx_daily_2014_2017_csi300.py --workers 4
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from mootdx.quotes import Quotes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "csi300.txt"
OUTPUT_PARQUET = ROOT / "data_cache" / "kline_2014_2017_csi300_backfill.parquet"
FAILED_LOG = ROOT / "data_cache" / "kline_2014_2017_csi300_backfill_failed.csv"

TDX_SERVERS = [
    ("180.153.18.170", 7709),
    ("123.125.108.14", 7709),
    ("218.6.170.47", 7709),
    ("60.12.136.250", 7709),
    ("115.238.56.198", 7709),
    ("115.238.90.165", 7709),
]

FREQ_DAILY = 4
PAGE_SIZE = 800
# 2014-01-01 → today (2026) ≈ 12.5 yr × 244 = ~3050 trading days; 5 pages = 4000 bars
MAX_PAGES = 5
CLIENT_TIMEOUT = 15

START_DATE = pd.Timestamp("2014-01-01")
END_DATE = pd.Timestamp("2017-12-31")


def load_universe(limit: Optional[int] = None) -> list[Tuple[str, str]]:
    """Return list of (qid, raw_6digit) e.g. ('SH600519', '600519')."""
    items = []
    with open(UNIVERSE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            qid = parts[0]
            if len(qid) < 8:
                continue
            mkt = qid[:2].upper()
            raw = qid[2:]
            if mkt not in ("SH", "SZ"):
                continue
            items.append((qid, raw))
    if limit:
        items = items[:limit]
    return items


def make_client(server: Tuple[str, int]) -> Optional[Quotes]:
    try:
        return Quotes.factory(market="std", server=server, timeout=CLIENT_TIMEOUT)
    except Exception:
        return None


def try_init_client(primary_idx: int) -> Tuple[Optional[Quotes], Optional[Tuple]]:
    n = len(TDX_SERVERS)
    order = [TDX_SERVERS[(primary_idx + i) % n] for i in range(n)]
    for srv in order:
        c = make_client(srv)
        if c is None:
            time.sleep(0.3)
            continue
        try:
            df = c.bars(symbol="600519", frequency=FREQ_DAILY, start=0, offset=5)
            if df is not None and len(df) > 0:
                return c, srv
        except Exception:
            pass
    return None, None


def fetch_one_stock_daily(client: Quotes, raw_code: str) -> Optional[pd.DataFrame]:
    """Paginate daily bars; return clean DataFrame filtered to 2014-2017 or None."""
    pages = []
    start = 0
    for _ in range(MAX_PAGES):
        try:
            df = client.bars(
                symbol=raw_code, frequency=FREQ_DAILY,
                start=start, offset=PAGE_SIZE,
            )
        except Exception:
            return None
        if df is None or len(df) == 0:
            break
        pages.append(df)
        if len(df) < PAGE_SIZE:
            break
        start += PAGE_SIZE
        # Stop early if earliest date already past 2014-01-01
        try:
            earliest = pd.to_datetime(df["datetime"]).min()
            if earliest < START_DATE:
                break
        except Exception:
            pass
    if not pages:
        return None
    df = pd.concat(pages, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.normalize()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].reset_index(drop=True)
    if len(df) == 0:
        return None
    if (df["close"] <= 0).any():
        return None
    return df


def normalize_schema(df: pd.DataFrame, raw_code: str) -> pd.DataFrame:
    return pd.DataFrame({
        "code": raw_code,
        "date": df["date"],
        "open": df["open"].astype("float64"),
        "close": df["close"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "vol": df["vol"].astype("int64"),
        "amount": df["amount"].astype("float64"),
    })


def worker_batch(
    batch: list[Tuple[str, str]],
    server: Tuple[str, int],
    worker_id: int,
) -> list[Tuple[str, str, int, Optional[pd.DataFrame]]]:
    """Returns list of (qid, status, n_rows, df_or_none)."""
    results: list[Tuple[str, str, int, Optional[pd.DataFrame]]] = []
    primary_idx = TDX_SERVERS.index(server)
    client, _ = try_init_client(primary_idx)
    if client is None:
        for qid, _ in batch:
            results.append((qid, "client_init_fail", 0, None))
        return results

    consecutive_fails = 0
    for i, (qid, raw) in enumerate(batch):
        if consecutive_fails >= 5:
            try:
                client.client.disconnect()
            except Exception:
                pass
            client, _ = try_init_client(primary_idx)
            consecutive_fails = 0
            if client is None:
                results.append((qid, "no_client", 0, None))
                continue

        df = fetch_one_stock_daily(client, raw)
        if df is None or len(df) == 0:
            df = fetch_one_stock_daily(client, raw)

        if df is None or len(df) == 0:
            results.append((qid, "empty", 0, None))
            consecutive_fails += 1
            continue

        try:
            out = normalize_schema(df, raw)
            results.append((qid, "ok", len(out), out))
            consecutive_fails = 0
        except Exception as e:
            results.append((qid, f"normalize_fail:{type(e).__name__}", 0, None))
            consecutive_fails += 1

        if (i + 1) % 25 == 0:
            ok_n = sum(1 for _, s, _, _ in results if s == "ok")
            print(f"  [w{worker_id}] progress {i+1}/{len(batch)} ok={ok_n}", flush=True)

    try:
        client.client.disconnect()
    except Exception:
        pass
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    universe = load_universe(limit=args.limit)
    n_total = len(universe)
    n_workers = args.workers
    print(f"[main] CSI300 universe={n_total}, workers={n_workers}", flush=True)
    print(f"[main] output → {OUTPUT_PARQUET}", flush=True)

    chunks = []
    chunk_size = (n_total + n_workers - 1) // n_workers
    for i in range(n_workers):
        batch = universe[i * chunk_size : (i + 1) * chunk_size]
        if not batch:
            continue
        srv = TDX_SERVERS[i % len(TDX_SERVERS)]
        chunks.append((batch, srv, i))

    print(f"[main] dispatch {len(chunks)} workers, ~{chunk_size} stocks/worker", flush=True)

    t_start = time.time()
    all_results: list[Tuple[str, str, int, Optional[pd.DataFrame]]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker_batch, b, s, i) for b, s, i in chunks]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                all_results.extend(res)
            except Exception as e:
                print(f"[main] worker crashed: {e}", flush=True)

    wall = time.time() - t_start
    n_ok = sum(1 for _, s, _, _ in all_results if s == "ok")
    n_fail = len(all_results) - n_ok
    total_rows = sum(n for _, s, n, _ in all_results if s == "ok")
    print(f"\n[main] wall={wall/60:.1f} min  ok={n_ok}  fail={n_fail}  "
          f"total_rows={total_rows:,}", flush=True)

    dfs = [df for _, s, _, df in all_results if s == "ok" and df is not None]
    if not dfs:
        print("[main] ERROR: no data fetched", flush=True)
        return
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"[main] combined shape={combined.shape}, "
          f"date range={combined['date'].min()} → {combined['date'].max()}, "
          f"unique codes={combined['code'].nunique()}", flush=True)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PARQUET, compression="snappy", index=False)
    print(f"[main] wrote → {OUTPUT_PARQUET} "
          f"({OUTPUT_PARQUET.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)

    if n_fail > 0:
        failed = [(qid, status) for qid, status, _, _ in all_results
                  if status not in ("ok",)]
        fdf = pd.DataFrame(failed, columns=["code", "status"])
        fdf.to_csv(FAILED_LOG, index=False)
        print(f"[main] failure log → {FAILED_LOG}", flush=True)

    status_counts: dict[str, int] = {}
    for _, s, _, _ in all_results:
        key = s if s in ("ok", "empty", "no_client", "client_init_fail") else "other_fail"
        status_counts[key] = status_counts.get(key, 0) + 1
    print(f"[main] status: {status_counts}", flush=True)

    mc = combined.groupby(pd.Grouper(key="date", freq="MS"))["code"].nunique()
    print(f"[main] monthly CSI300 coverage 2014-2017 (head 6, tail 6):", flush=True)
    print(mc.head(6).to_string(), flush=True)
    print(mc.tail(6).to_string(), flush=True)


if __name__ == "__main__":
    main()
