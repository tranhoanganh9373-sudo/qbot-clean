"""Phase 1B: 全 universe 2014-2020 后复权 K 线重抓.

数据源 (akshare 主接口 push2his.eastmoney.com 在当前网络被阻断, fallback):
  1) mootdx get_k_data via TDX server  → 前复权 daily OHLCV (qfq)
  2) sina hfq.js                       → hfq factor 复权事件
  3) merge_asof + close * factor       → 后复权 (hfq) OHLC

落盘: data_cache/baidu_kline_v2.parquet (不动 baidu_kline.parquet)
失败清单: data_cache/akshare_fetch_failed_2014_2020.csv

约束:
  - 严格不修改主 parquet
  - 8 workers 并行 (mootdx + sina), retry 1 次
  - 0 neg close 才允许落盘
  - turnoverratio 在 mootdx 不可得, 填 NaN (后续无消费方)
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
OUTPUT_PATH = "data_cache/baidu_kline_v2.parquet"
FAILED_PATH = "data_cache/akshare_fetch_failed_2014_2020.csv"

START_DATE = "2014-01-01"
END_DATE = "2020-12-31"

# TDX server pool — socket-probed 实测在当前网络可达的 IP
# (大部分公开 TDX server 在此网络环境被防火墙挡;只有这 6 个稳定)
TDX_SERVERS = [
    ("180.153.18.170", 7709),    # 上海腾飞 (Shenwan)
    ("123.125.108.14", 7709),    # 北京华泰
    ("218.6.170.47", 7709),      # 南方双线
    ("60.12.136.250", 7709),     # 浙江电信
    ("115.238.56.198", 7709),    # 杭州
    ("115.238.90.165", 7709),    # 杭州 2
]

SINA_HFQ_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"

# 6 workers - 与可用 server 数一致, 每 worker 独立 server 避免单 server 过载
N_WORKERS = 6

# requests 通用 session 配置: bypass proxy + 重试
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
    # drop rows with unknown market
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
            # 无复权事件 → return identity factor (1.0)
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
    # 1. qfq daily
    df = fetch_qfq(client, code)
    if df is None or len(df) == 0:
        return code, None, "qfq_empty"

    # 2. hfq factor
    factors = fetch_hfq_factor(session, sina_sym)
    if factors is None:
        return code, None, "hfq_factor_fail"

    # 3. merge
    df = df.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(df, factors, on="date", direction="backward")
    # 如果 date < 最早 factor.date, factor=NaN. fallback factor=1.0
    merged["factor"] = merged["factor"].fillna(1.0)

    for c in ["open", "close", "high", "low"]:
        merged[c] = merged[c] * merged["factor"]

    # 4. ma5/10/20 用 hfq close 算
    merged["ma5"] = merged["close"].rolling(5).mean()
    merged["ma10"] = merged["close"].rolling(10).mean()
    merged["ma20"] = merged["close"].rolling(20).mean()

    merged["code"] = code
    merged["turnoverratio"] = float("nan")  # TDX 不提供, 后续无消费方
    # vol 转 int64 (baidu schema)
    merged["vol"] = merged["vol"].astype("int64")

    # baidu_kline schema 12 cols
    out = merged[
        ["code", "date", "open", "close", "high", "low",
         "vol", "amount", "ma5", "ma10", "ma20", "turnoverratio"]
    ].copy()

    return code, out, "ok"


# ──────────────────────────────────────────────────────────────────────────
# Worker (持有 mootdx client + session, 处理 batch)
# ──────────────────────────────────────────────────────────────────────────

def _try_init_client(servers: list, primary_idx: int):
    """Try to init client with primary server, fallback to others in pool.
    Returns (client, server_used) or (None, None)."""
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
    """Process a batch of stocks with mootdx client + session.
    Falls back to other servers in TDX_SERVERS pool if primary fails.
    """
    results = []
    primary_idx = TDX_SERVERS.index(server) if server in TDX_SERVERS else 0
    client = None
    current_server = server
    session = requests.Session()
    consecutive_fails = 0

    for _, row in rows.iterrows():
        # 重连 client 如果未建立 OR 已连续失败 5+ 次
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
            # retry once with same client
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

    # 切分给 N_WORKERS
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

    # 汇总
    t_fetch_end = time.time()
    print(f"\n[main] fetch phase done in {t_fetch_end - t_start:.1f}s", flush=True)

    ok_dfs = [df for code, df, status in all_results if status == "ok"]
    failed = [(code, status) for code, df, status in all_results if status != "ok"]
    n_ok = len(ok_dfs)
    n_failed = len(failed)
    print(f"[main] ok: {n_ok}, failed: {n_failed} "
          f"(fail rate: {n_failed/n_total*100:.1f}%)", flush=True)

    # 失败清单
    if failed:
        pd.DataFrame(failed, columns=["code", "fail_reason"]).to_csv(
            FAILED_PATH, index=False
        )
        print(f"[main] failed list → {FAILED_PATH}", flush=True)

    if not ok_dfs:
        print("[main] FATAL: no ok data, aborting", flush=True)
        sys.exit(1)

    # 合并
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

    # 落盘
    full.to_parquet(OUTPUT_PATH)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"[main] WROTE {OUTPUT_PATH} ({size_mb:.1f} MB)", flush=True)

    # 总结
    t_total = time.time() - t_start
    print(
        f"\n=== FINAL ===\n"
        f"universe: {n_total}\n"
        f"ok: {n_ok}\n"
        f"failed: {n_failed} ({n_failed/n_total*100:.1f}%)\n"
        f"wall time: {t_total:.0f}s ({t_total/60:.1f} min)\n"
        f"output: {OUTPUT_PATH}\n"
        f"failed list: {FAILED_PATH if failed else 'none'}\n"
        f"verification: neg_close=0 extreme_low<0.5={extreme_low} extreme_jump>11%={extreme_jump}",
        flush=True,
    )


if __name__ == "__main__":
    main()
