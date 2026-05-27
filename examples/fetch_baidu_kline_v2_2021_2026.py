"""Phase 1C: 全 universe 2021-01-01 ~ 2026-05-25 后复权 K 线抓取.

与 Phase 1B (2014-2020 hfq) 互补, 合并后形成完整 2014-至今 hfq baidu_kline_v2.

数据源 (与 Phase 1B 一致):
  1) mootdx get_k_data via TDX server  → 前复权 daily OHLCV (qfq)
  2) sina hfq.js                       → hfq factor 复权事件
  3) merge_asof + close * factor       → 后复权 (hfq) OHLC

落盘: data_cache/baidu_kline_v2_2021_2026.parquet (不动 baidu_kline_v2.parquet)
失败清单: data_cache/akshare_fetch_failed_2021_2026.csv

约束:
  - 严格不修改主 parquet
  - 6 workers 并行 (mootdx + sina), retry 1 次
  - 0 neg close 才允许落盘
  - turnoverratio 在 mootdx 不可得, 填 NaN
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

# Bypass macOS system proxy (push2his blocked anyway, but tdx + sina need direct conn)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from mootdx.quotes import Quotes  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────

UNIVERSE_PATH = "data_cache/universe.csv"
OUTPUT_PATH = "data_cache/baidu_kline_v2_2021_2026.parquet"
FAILED_PATH = "data_cache/akshare_fetch_failed_2021_2026.csv"

START_DATE = "2021-01-01"
END_DATE = "2026-05-25"

# TDX server pool — 与 Phase 1B 一致, socket-probed 实测在当前网络可达的 IP
TDX_SERVERS = [
    ("180.153.18.170", 7709),    # 上海腾飞 (Shenwan)
    ("123.125.108.14", 7709),    # 北京华泰
    ("218.6.170.47", 7709),      # 南方双线
    ("60.12.136.250", 7709),     # 浙江电信
    ("115.238.56.198", 7709),    # 杭州
    ("115.238.90.165", 7709),    # 杭州 2
]

SINA_HFQ_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"

# 6 workers - 与可用 server 数一致
N_WORKERS = 6

# requests 通用 session 配置
PROXIES_NULL = {"http": None, "https": None}
HTTP_TIMEOUT = 15


# ──────────────────────────────────────────────────────────────────────────
# Universe 读取 + code 格式归一
# ──────────────────────────────────────────────────────────────────────────

def load_universe() -> pd.DataFrame:
    """Read universe.csv; produce columns:
        code        6-digit zero-padded string (baidu schema 主键)
        market_int  TDX market id (1=sh, 0=sz)
        sina_sym    sina hfq.js symbol (sh600519 / sz000001)
    """
    df = pd.read_csv(UNIVERSE_PATH)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["market_int"] = df["market"].str.lower().map({"sh": 1, "sz": 0})
    df["sina_sym"] = df["market"].str.lower() + df["code"]
    df = df.dropna(subset=["market_int"]).copy()
    df["market_int"] = df["market_int"].astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────
# Sina HFQ factor 拉取
# ──────────────────────────────────────────────────────────────────────────

def fetch_hfq_factor(session: requests.Session, sina_sym: str) -> Optional[pd.DataFrame]:
    """Return DataFrame[date, factor] sorted ascending, or None on failure."""
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
# TDX 拉前复权 daily K
# ──────────────────────────────────────────────────────────────────────────

def make_mootdx_client(server: Tuple[str, int]) -> Quotes:
    return Quotes.factory(market="std", server=server, timeout=15)


def fetch_qfq(client: Quotes, code: str) -> Optional[pd.DataFrame]:
    """TDX get_k_data; 返回 cols: open close high low vol amount date."""
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
# 单股完整 pipeline
# ──────────────────────────────────────────────────────────────────────────

def fetch_one_stock(
    client: Quotes,
    session: requests.Session,
    code: str,
    sina_sym: str,
) -> Tuple[str, Optional[pd.DataFrame], str]:
    """Returns (code, df or None, status)."""
    df = fetch_qfq(client, code)
    if df is None or len(df) == 0:
        return code, None, "qfq_empty"

    factors = fetch_hfq_factor(session, sina_sym)
    if factors is None:
        return code, None, "hfq_factor_fail"

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
    """Try to init client with primary server, fallback to others in pool."""
    n = len(servers)
    order = [servers[(primary_idx + i) % n] for i in range(n)]
    for srv in order:
        try:
            return make_mootdx_client(srv), srv
        except Exception:
            time.sleep(0.5)
            continue
    return None, None


def worker(rows: pd.DataFrame, server: Tuple[str, int], worker_id: int) -> list:
    """Process a batch of stocks with mootdx client + session."""
    results = []
    primary_idx = TDX_SERVERS.index(server) if server in TDX_SERVERS else 0
    client = None
    current_server = server
    session = requests.Session()
    consecutive_fails = 0

    for _, row in rows.iterrows():
        if client is None or consecutive_fails >= 5:
            if client is not None:
                try:
                    client.client.disconnect()
                except Exception:
                    pass
            client, current_server = _try_init_client(TDX_SERVERS, primary_idx)
            consecutive_fails = 0
            if client is None:
                results.append(
                    (row["code"], None, "client_init_fail_all_servers")
                )
                continue

        code, df, status = fetch_one_stock(
            client, session, row["code"], row["sina_sym"]
        )
        if status != "ok":
            _, df2, status2 = fetch_one_stock(
                client, session, row["code"], row["sina_sym"]
            )
            if status2 == "ok":
                df, status = df2, status2

        if status == "ok":
            consecutive_fails = 0
        else:
            consecutive_fails += 1

        results.append((code, df, status))
    return results


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    universe = load_universe()
    n_total = len(universe)
    print(f"[main] universe size: {n_total}", flush=True)
    print(f"[main] date range: {START_DATE} → {END_DATE}", flush=True)

    batches = []
    chunk_size = (n_total + N_WORKERS - 1) // N_WORKERS
    for i in range(N_WORKERS):
        batch = universe.iloc[i * chunk_size : (i + 1) * chunk_size]
        if len(batch) > 0:
            server = TDX_SERVERS[i % len(TDX_SERVERS)]
            batches.append((batch, server, i))
    print(f"[main] dispatch {len(batches)} workers, "
          f"~{chunk_size} stocks/worker", flush=True)

    all_results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(worker, batch, server, wid): wid
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
            completed += len(batch_results)
            ok_so_far = sum(1 for _, df, s in all_results if s == "ok")
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (n_total - completed) / rate if rate > 0 else float("inf")
            print(
                f"[worker {wid} done] cumulative {completed}/{n_total} "
                f"ok={ok_so_far} elapsed={elapsed:.0f}s "
                f"rate={rate:.2f}/s ETA={eta:.0f}s",
                flush=True,
            )

    t_fetch_end = time.time()
    print(f"\n[main] fetch phase done in {t_fetch_end - t_start:.1f}s", flush=True)

    ok_dfs = [df for code, df, status in all_results if status == "ok"]
    failed = [(code, status) for code, df, status in all_results if status != "ok"]
    n_ok = len(ok_dfs)
    n_failed = len(failed)
    print(f"[main] ok: {n_ok}, failed: {n_failed} "
          f"(fail rate: {n_failed/n_total*100:.1f}%)", flush=True)

    if failed:
        pd.DataFrame(failed, columns=["code", "fail_reason"]).to_csv(
            FAILED_PATH, index=False
        )
        print(f"[main] failed list → {FAILED_PATH}", flush=True)

        # Prefix breakdown
        fail_df = pd.DataFrame(failed, columns=["code", "fail_reason"])
        fail_df["prefix"] = fail_df["code"].str[:3]
        print("[main] failed by prefix (top 10):", flush=True)
        print(fail_df["prefix"].value_counts().head(10).to_string(), flush=True)

    if not ok_dfs:
        print("[main] FATAL: no ok data, aborting", flush=True)
        sys.exit(1)

    full = pd.concat(ok_dfs, ignore_index=True)
    print(f"[main] concat shape: {full.shape}", flush=True)

    # 验证质量
    neg_count = int((full["close"] < 0).sum())
    extreme_low = int((full["close"] < 0.5).sum())
    full_sorted = full.sort_values(["code", "date"])
    pct_chg = full_sorted.groupby("code")["close"].pct_change().abs()
    extreme_jump = int((pct_chg > 0.11).sum())
    nan_close = int(full["close"].isna().sum())
    print(
        f"[verify] neg_close={neg_count} extreme_low(<0.5)={extreme_low} "
        f"extreme_jump(>11%)={extreme_jump} nan_close={nan_close}",
        flush=True,
    )

    if neg_count > 0:
        print("[main] FATAL: still has neg close — abort write", flush=True)
        sys.exit(2)

    full.to_parquet(OUTPUT_PATH)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"[main] WROTE {OUTPUT_PATH} ({size_mb:.1f} MB)", flush=True)

    t_total = time.time() - t_start
    print(
        f"\n=== FINAL ===\n"
        f"universe: {n_total}\n"
        f"ok: {n_ok}\n"
        f"failed: {n_failed} ({n_failed/n_total*100:.1f}%)\n"
        f"wall time: {t_total:.0f}s ({t_total/60:.1f} min)\n"
        f"output: {OUTPUT_PATH}\n"
        f"failed list: {FAILED_PATH if failed else 'none'}\n"
        f"verification: neg_close=0 extreme_low<0.5={extreme_low} extreme_jump>11%={extreme_jump} nan_close={nan_close}",
        flush=True,
    )


if __name__ == "__main__":
    main()
