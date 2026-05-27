"""Phase Backfill: 2014-2018 hfq K 线 (mootdx + sina hfq.js).

目的: baidu qfq 主 parquet 在 2014-2017 只有 23-44 codes/月, 历史 5 次 Phase B
abort 都低样本 (n_months ≤ 60). 用 mootdx qfq + sina hfq.js factor 重抓
全 universe 2014-01-01 ~ 2018-12-31 hfq base, **写独立 parquet**, IC 直接读
取 — 不混 hfq/qfq base.

写: data_cache/baidu_kline_extended_hfq.parquet      (独立, 不混主表)
失败清单: data_cache/baidu_kline_extended_hfq_failed.csv

约束:
  - 不动 baidu_kline.parquet (production qfq base)
  - resumable: 已抓 code 跳过 (per-code parquet shard, 最后 concat)
  - 6 workers (与 Phase 1B 一致, mootdx server pool 限制)
  - 温和 rate limit: sina hfq.js sleep 0.05s/req
  - 0 neg close 才允许写最终 parquet

复用 fetch_baidu_kline_v2_akshare.py 几乎全部 logic, 只改:
  - END_DATE: 2020-12-31 → 2018-12-31
  - OUTPUT_PATH: baidu_kline_v2.parquet → baidu_kline_extended_hfq.parquet
  - FAILED_PATH 同理
  - 加 per-code shard + resumable skip
  - progress 每 200 stocks 报 1 次 (代替 worker-batch)
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

# Bypass macOS system proxy
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from mootdx.quotes import Quotes  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = ROOT / "data_cache" / "universe.csv"
SHARD_DIR = ROOT / "data_cache" / "baidu_kline_extended_hfq_shards"
OUTPUT_PATH = ROOT / "data_cache" / "baidu_kline_extended_hfq.parquet"
FAILED_PATH = ROOT / "data_cache" / "baidu_kline_extended_hfq_failed.csv"

START_DATE = "2014-01-01"
END_DATE = "2018-12-31"

# TDX server pool — same as v2 fetch
TDX_SERVERS = [
    ("180.153.18.170", 7709),
    ("123.125.108.14", 7709),
    ("218.6.170.47", 7709),
    ("60.12.136.250", 7709),
    ("115.238.56.198", 7709),
    ("115.238.90.165", 7709),
]

SINA_HFQ_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"

N_WORKERS = 6

PROXIES_NULL = {"http": None, "https": None}
HTTP_TIMEOUT = 15
SINA_SLEEP = 0.05  # 温和 rate limit

# Progress
PROGRESS_INTERVAL = 200


# ──────────────────────────────────────────────────────────────────────────
# Universe
# ──────────────────────────────────────────────────────────────────────────

def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_PATH)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["market_int"] = df["market"].str.lower().map({"sh": 1, "sz": 0})
    df["sina_sym"] = df["market"].str.lower() + df["code"]
    df = df.dropna(subset=["market_int"]).copy()
    df["market_int"] = df["market_int"].astype(int)
    return df


def already_done_codes() -> set[str]:
    """Codes that already have shard parquet on disk (resumable)."""
    if not SHARD_DIR.exists():
        return set()
    return {p.stem for p in SHARD_DIR.glob("*.parquet")}


# ──────────────────────────────────────────────────────────────────────────
# Sina HFQ factor
# ──────────────────────────────────────────────────────────────────────────

def fetch_hfq_factor(session: requests.Session, sina_sym: str) -> Optional[pd.DataFrame]:
    try:
        r = session.get(
            SINA_HFQ_URL.format(symbol=sina_sym),
            timeout=HTTP_TIMEOUT,
            proxies=PROXIES_NULL,
        )
        if r.status_code != 200:
            return None
        text = r.text
        if "=" not in text or "data" not in text:
            return None
        parts = text.split("=", 1)
        if len(parts) < 2:
            return None
        body = parts[1].split("\n")[0].rstrip(";").rstrip()
        data = json.loads(body)
        rows = data.get("data") if isinstance(data, dict) else None
        if not rows:
            return pd.DataFrame({"date": [pd.Timestamp("1990-01-01")], "factor": [1.0]})
        df = pd.DataFrame(rows)
        df.columns = ["date", "factor"]
        df["date"] = pd.to_datetime(df["date"])
        df["factor"] = df["factor"].astype(float)
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# TDX qfq
# ──────────────────────────────────────────────────────────────────────────

def make_mootdx_client(server: Tuple[str, int]) -> Quotes:
    return Quotes.factory(market="std", server=server, timeout=15)


def fetch_qfq(client: Quotes, code: str) -> Optional[pd.DataFrame]:
    try:
        df = client.get_k_data(code, START_DATE, END_DATE)
        if df is None or len(df) == 0:
            return None
        df = df.reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# 单股 pipeline
# ──────────────────────────────────────────────────────────────────────────

def fetch_one_stock(
    client: Quotes,
    session: requests.Session,
    code: str,
    sina_sym: str,
) -> Tuple[str, Optional[pd.DataFrame], str]:
    df = fetch_qfq(client, code)
    if df is None or len(df) == 0:
        return code, None, "qfq_empty"

    factors = fetch_hfq_factor(session, sina_sym)
    if factors is None:
        return code, None, "hfq_factor_fail"
    time.sleep(SINA_SLEEP)

    df = df.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(df, factors, on="date", direction="backward")
    merged["factor"] = merged["factor"].fillna(1.0)

    for c in ["open", "close", "high", "low"]:
        merged[c] = merged[c] * merged["factor"]

    merged["ma5"] = merged["close"].rolling(5).mean()
    merged["ma10"] = merged["close"].rolling(10).mean()
    merged["ma20"] = merged["close"].rolling(20).mean()

    merged["code"] = code
    merged["turnoverratio"] = float("nan")
    merged["vol"] = merged["vol"].astype("int64")

    out = merged[
        ["code", "date", "open", "close", "high", "low",
         "vol", "amount", "ma5", "ma10", "ma20", "turnoverratio"]
    ].copy()

    return code, out, "ok"


# ──────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────

def _try_init_client(servers: list, primary_idx: int):
    n = len(servers)
    order = [servers[(primary_idx + i) % n] for i in range(n)]
    for srv in order:
        try:
            return make_mootdx_client(srv), srv
        except Exception:
            time.sleep(0.5)
            continue
    return None, None


# Shared progress counter
_progress = {"done": 0, "ok": 0, "fail": 0, "skipped": 0}
_t_start = [0.0]
_progress_lock = None  # set in main


def worker(rows: pd.DataFrame, server: Tuple[str, int], worker_id: int, total: int) -> list:
    results = []
    primary_idx = TDX_SERVERS.index(server) if server in TDX_SERVERS else 0
    client = None
    session = requests.Session()
    consecutive_fails = 0

    for _, row in rows.iterrows():
        code = row["code"]
        sina_sym = row["sina_sym"]
        shard_path = SHARD_DIR / f"{code}.parquet"

        # Resumable skip
        if shard_path.exists():
            results.append((code, "skipped"))
            with _progress_lock:
                _progress["done"] += 1
                _progress["skipped"] += 1
                _maybe_print(total)
            continue

        if client is None or consecutive_fails >= 5:
            if client is not None:
                try:
                    client.client.disconnect()
                except Exception:
                    pass
            client, _ = _try_init_client(TDX_SERVERS, primary_idx)
            consecutive_fails = 0
            if client is None:
                results.append((code, "client_init_fail_all_servers"))
                with _progress_lock:
                    _progress["done"] += 1
                    _progress["fail"] += 1
                    _maybe_print(total)
                continue

        c2, df, status = fetch_one_stock(client, session, code, sina_sym)
        if status != "ok":
            _, df2, status2 = fetch_one_stock(client, session, code, sina_sym)
            if status2 == "ok":
                df, status = df2, status2

        if status == "ok":
            consecutive_fails = 0
            df.to_parquet(shard_path)
            results.append((code, "ok"))
            with _progress_lock:
                _progress["done"] += 1
                _progress["ok"] += 1
                _maybe_print(total)
        else:
            consecutive_fails += 1
            results.append((code, status))
            with _progress_lock:
                _progress["done"] += 1
                _progress["fail"] += 1
                _maybe_print(total)

    return results


def _maybe_print(total: int) -> None:
    done = _progress["done"]
    if done % PROGRESS_INTERVAL == 0 or done == total:
        elapsed = time.time() - _t_start[0]
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else float("inf")
        print(
            f"[progress] {done}/{total} ok={_progress['ok']} fail={_progress['fail']} "
            f"skipped={_progress['skipped']} elapsed={elapsed:.0f}s "
            f"rate={rate:.2f}/s ETA={eta:.0f}s",
            flush=True,
        )


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    import threading
    global _progress_lock
    _progress_lock = threading.Lock()

    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    _t_start[0] = t_start

    universe = load_universe()
    n_total = len(universe)
    print(f"[main] universe size: {n_total}", flush=True)
    print(f"[main] period: {START_DATE} ~ {END_DATE}", flush=True)
    print(f"[main] shard dir: {SHARD_DIR}", flush=True)

    done = already_done_codes()
    if done:
        print(f"[main] resume: {len(done)} shards already exist", flush=True)

    batches = []
    chunk_size = (n_total + N_WORKERS - 1) // N_WORKERS
    for i in range(N_WORKERS):
        batch = universe.iloc[i * chunk_size : (i + 1) * chunk_size]
        if len(batch) > 0:
            server = TDX_SERVERS[i % len(TDX_SERVERS)]
            batches.append((batch, server, i))
    print(f"[main] dispatch {len(batches)} workers, ~{chunk_size}/worker", flush=True)

    all_results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(worker, batch, server, wid, n_total): wid
            for batch, server, wid in batches
        }
        for fut in as_completed(futures):
            wid = futures[fut]
            try:
                batch_results = fut.result()
            except Exception as e:
                print(f"[worker {wid}] FATAL: {type(e).__name__}: {e}", flush=True)
                continue
            all_results.extend(batch_results)

    t_fetch_end = time.time()
    print(f"\n[main] fetch phase done in {t_fetch_end - t_start:.1f}s", flush=True)

    failed = [(c, s) for c, s in all_results if s not in ("ok", "skipped")]
    n_ok = sum(1 for _, s in all_results if s == "ok")
    n_skipped = sum(1 for _, s in all_results if s == "skipped")
    n_failed = len(failed)
    print(
        f"[main] this run: ok={n_ok} skipped={n_skipped} failed={n_failed} "
        f"(this-run fail rate: {n_failed/max(1,n_ok+n_failed)*100:.1f}%)",
        flush=True,
    )

    if failed:
        pd.DataFrame(failed, columns=["code", "fail_reason"]).to_csv(FAILED_PATH, index=False)
        print(f"[main] failed list → {FAILED_PATH}", flush=True)
        rs = pd.Series([s for _, s in failed]).value_counts()
        print(f"[main] fail reasons:\n{rs.to_string()}", flush=True)

    # ── Aggregate shards → single parquet ──
    shard_files = sorted(SHARD_DIR.glob("*.parquet"))
    if not shard_files:
        print("[main] FATAL: no shard parquet, nothing to write", flush=True)
        return 1

    print(f"\n[agg] reading {len(shard_files)} shards ...", flush=True)
    t_agg = time.time()
    dfs = []
    for i, p in enumerate(shard_files):
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            print(f"  [bad shard] {p.name}: {e}", flush=True)
        if (i + 1) % 500 == 0:
            print(f"  [agg] read {i+1}/{len(shard_files)}", flush=True)
    full = pd.concat(dfs, ignore_index=True)
    print(f"[agg] concat shape: {full.shape}  (in {time.time()-t_agg:.1f}s)", flush=True)

    neg_count = int((full["close"] < 0).sum())
    extreme_low = int((full["close"] < 0.5).sum())
    nan_close = int(full["close"].isna().sum())
    full_sorted = full.sort_values(["code", "date"])
    pct_chg = full_sorted.groupby("code")["close"].pct_change().abs()
    extreme_jump = int((pct_chg > 0.11).sum())
    print(
        f"[verify] neg={neg_count} extreme_low(<0.5)={extreme_low} "
        f"extreme_jump(>11%)={extreme_jump} nan_close={nan_close}",
        flush=True,
    )

    if neg_count > 0:
        print("[main] FATAL: neg close present — abort write", flush=True)
        return 2

    full.to_parquet(OUTPUT_PATH)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"[main] WROTE {OUTPUT_PATH} ({size_mb:.1f} MB)", flush=True)

    n_codes_in_full = full["code"].nunique()
    n_rows_full = len(full)

    full["month"] = full["date"].dt.to_period("M")
    cov_by_month = full.groupby("month")["code"].nunique()
    print(f"\n[cov] codes/month sample:")
    for m in ["2014-01", "2014-06", "2015-01", "2016-01", "2017-01", "2018-01", "2018-12"]:
        try:
            v = cov_by_month.loc[m]
            print(f"  {m}: {v}")
        except KeyError:
            pass

    t_total = time.time() - t_start
    print(
        f"\n=== FINAL ===\n"
        f"universe: {n_total}\n"
        f"shards: {len(shard_files)}\n"
        f"rows: {n_rows_full:,}\n"
        f"codes: {n_codes_in_full}\n"
        f"this-run ok: {n_ok}, skipped: {n_skipped}, failed: {n_failed}\n"
        f"wall: {t_total:.0f}s ({t_total/60:.1f} min)\n"
        f"output: {OUTPUT_PATH}\n"
        f"failed list: {FAILED_PATH if failed else 'none'}\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
