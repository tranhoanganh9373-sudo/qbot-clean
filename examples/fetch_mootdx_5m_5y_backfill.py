"""Phase 1: 全 universe 5m K 线 backfill (实测可用窗 ~2 年, 不是 5 年).

数据源: mootdx via TDX server pool (与 fetch_baidu_kline_v2_akshare.py 同 stack)
窗口:   尽可能拉全 (实测 2024-05-10 ~ 今日, ~2.04 年)
输出:   data_cache/kline_5m_shards/{CODE}.parquet (per-stock shard)
        ^^^ CODE = market+code (与 universe id 一致: SH600519, SZ000001)

约束:
  - 不动主 parquet, 不动 production
  - 6-8 workers 并行
  - skip 已存在 shard (resumable)
  - 0 neg close 才允许落盘
  - paginate until empty (每股 ~30 page × 800 bar = 24000 bar)

schema (落盘):
  code           str   (SH600519 / SZ000001)
  datetime       datetime64[ns]
  open/high/low/close   float64
  volume         int64
  amount         float64

CLI:
  python examples/fetch_mootdx_5m_5y_backfill.py            # full universe
  python examples/fetch_mootdx_5m_5y_backfill.py --limit 50 # smoke test
  python examples/fetch_mootdx_5m_5y_backfill.py --workers 4
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

# ──────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "all_no_st.txt"
OUTPUT_DIR = ROOT / "data_cache" / "kline_5m_shards"
FAILED_LOG = ROOT / "data_cache" / "kline_5m_failed.csv"

TDX_SERVERS = [
    ("180.153.18.170", 7709),
    ("123.125.108.14", 7709),
    ("218.6.170.47", 7709),
    ("60.12.136.250", 7709),
    ("115.238.56.198", 7709),
    ("115.238.90.165", 7709),
]

FREQ_5M = 0
PAGE_SIZE = 800
DEFAULT_MAX_PAGES = 35   # 2-yr 上限 (server hard-cap)
CLIENT_TIMEOUT = 15

# 全局 max_pages, 由 CLI 设置
MAX_PAGES = DEFAULT_MAX_PAGES


def load_universe(limit: Optional[int] = None) -> list[Tuple[str, str]]:
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
            df = c.bars(symbol="600519", frequency=4, start=0, offset=5)
            if df is not None and len(df) > 0:
                return c, srv
        except Exception:
            pass
    return None, None


def fetch_one_stock_5m(client: Quotes, raw_code: str) -> Optional[pd.DataFrame]:
    """Paginate all 5m bars; return clean DataFrame or None."""
    pages = []
    start = 0
    for _ in range(MAX_PAGES):
        try:
            df = client.bars(
                symbol=raw_code, frequency=FREQ_5M,
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
    if not pages:
        return None
    df = pd.concat(pages, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    if (df["close"] <= 0).any():
        return None
    return df


def normalize_schema(df: pd.DataFrame, qid: str) -> pd.DataFrame:
    return pd.DataFrame({
        "code": qid,
        "datetime": df["datetime"],
        "open": df["open"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "close": df["close"].astype("float64"),
        "volume": df["vol"].astype("int64"),
        "amount": df["amount"].astype("float64"),
    })


def worker_batch(
    batch: list[Tuple[str, str]],
    server: Tuple[str, int],
    worker_id: int,
) -> list[Tuple[str, str, int]]:
    results: list[Tuple[str, str, int]] = []
    primary_idx = TDX_SERVERS.index(server)
    client, _ = try_init_client(primary_idx)
    if client is None:
        for qid, _ in batch:
            results.append((qid, "client_init_fail", 0))
        return results

    consecutive_fails = 0
    for i, (qid, raw) in enumerate(batch):
        shard_path = OUTPUT_DIR / f"{qid}.parquet"
        if shard_path.exists():
            results.append((qid, "skip", 0))
            continue

        if consecutive_fails >= 5:
            try:
                client.client.disconnect()
            except Exception:
                pass
            client, _ = try_init_client(primary_idx)
            consecutive_fails = 0
            if client is None:
                results.append((qid, "no_client", 0))
                continue

        df = fetch_one_stock_5m(client, raw)
        if df is None or len(df) == 0:
            df = fetch_one_stock_5m(client, raw)

        if df is None or len(df) == 0:
            results.append((qid, "empty", 0))
            consecutive_fails += 1
            continue

        try:
            out = normalize_schema(df, qid)
            out.to_parquet(shard_path, compression="snappy", index=False)
            results.append((qid, "ok", len(out)))
            consecutive_fails = 0
        except Exception as e:
            results.append((qid, f"write_fail:{type(e).__name__}", 0))
            consecutive_fails += 1

        if (i + 1) % 50 == 0:
            ok_n = sum(1 for _, s, _ in results if s == "ok")
            print(f"  [w{worker_id}] progress {i+1}/{len(batch)} ok={ok_n}", flush=True)

    try:
        client.client.disconnect()
    except Exception:
        pass
    return results


def main():
    global MAX_PAGES
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 只 (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                    help="paginate 上限 (12=1yr, 24=2yr, 35=server cap)")
    args = ap.parse_args()
    MAX_PAGES = args.max_pages
    print(f"[main] max_pages={MAX_PAGES} (≈{MAX_PAGES * 800 / 12096:.2f} yr)", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe(limit=args.limit)
    n_total = len(universe)
    # workers 可以 > server 数 (多 worker 共享 server pool via round-robin)
    n_workers = args.workers
    print(f"[main] universe={n_total}, workers={n_workers}, output={OUTPUT_DIR}",
          flush=True)

    chunks = []
    chunk_size = (n_total + n_workers - 1) // n_workers
    for i in range(n_workers):
        batch = universe[i * chunk_size : (i + 1) * chunk_size]
        if not batch:
            continue
        srv = TDX_SERVERS[i % len(TDX_SERVERS)]
        chunks.append((batch, srv, i))

    print(f"[main] dispatch {len(chunks)} workers, ~{chunk_size} stocks/worker",
          flush=True)

    t_start = time.time()
    all_results: list[Tuple[str, str, int]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker_batch, b, s, i) for b, s, i in chunks]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                all_results.extend(res)
            except Exception as e:
                print(f"[main] worker crashed: {e}", flush=True)

    wall = time.time() - t_start
    n_ok = sum(1 for _, s, _ in all_results if s == "ok")
    n_skip = sum(1 for _, s, _ in all_results if s == "skip")
    n_fail = len(all_results) - n_ok - n_skip
    total_rows = sum(n for _, s, n in all_results if s == "ok")
    print(f"\n[main] wall={wall/60:.1f} min  ok={n_ok}  skip={n_skip}  "
          f"fail={n_fail}  total_rows={total_rows:,}", flush=True)

    if n_fail > 0:
        failed = [(qid, status) for qid, status, _ in all_results
                  if status not in ("ok", "skip")]
        fdf = pd.DataFrame(failed, columns=["code", "status"])
        fdf.to_csv(FAILED_LOG, index=False)
        print(f"[main] failure log → {FAILED_LOG}", flush=True)

    status_counts: dict[str, int] = {}
    for _, s, _ in all_results:
        key = s if s in ("ok", "skip", "empty", "no_client") else "other_fail"
        status_counts[key] = status_counts.get(key, 0) + 1
    print(f"[main] status: {status_counts}", flush=True)


if __name__ == "__main__":
    main()
