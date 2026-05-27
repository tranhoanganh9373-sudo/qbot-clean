"""Phase 1B retry: 单线程低速重抓之前 qfq_empty 的 1311 失败股.

背景:
  Phase 1B (`fetch_baidu_kline_v2_akshare.py`) 6 workers 并行抓 5016 股,
  失败 1311 股 (26.1%), 全部 fail_reason = "qfq_empty".
  根因推测: 6 worker 同时压 sina hfq.js 限流, 后期 batch 大量空响应,
  实际上是 mootdx TDX server 在并发下被某一 worker 占用导致拉空.

策略 (单线程 + sleep 绕开限流):
  - N_WORKERS = 1
  - sleep 0.4s between requests
  - 每股 retry 3 次, fail 后 sleep 1.5s 再试
  - **Wall-time guard**: 连续失败 ≥3 股后 (说明这批股 TDX 真无数据),
    后续股只 try 1 次 (skip retry), 直到首次成功重置.
    这是因为 universe 中大量股是注册制新股 (301xxx 2020-08+ 上市)
    或退市股, retry 也拉不出数据, 浪费 wall time.
  - 落盘 data_cache/baidu_kline_v2_retry.parquet (不动 v2 主表)
  - 新失败清单 → data_cache/akshare_fetch_failed_2014_2020_retry.csv

约束:
  - 严格不修改 v2 / v2_corrupt_fix / baidu_kline 主 parquet
  - 严格不修改原 fetch_baidu_kline_v2_akshare.py
  - hfq 计算逻辑与原 fetcher 完全一致
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
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

UNIVERSE_PATH = "data_cache/universe.csv"
FAILED_INPUT_PATH = "data_cache/akshare_fetch_failed_2014_2020.csv"
OUTPUT_PATH = "data_cache/baidu_kline_v2_retry.parquet"
FAILED_OUTPUT_PATH = "data_cache/akshare_fetch_failed_2014_2020_retry.csv"

START_DATE = "2014-01-01"
END_DATE = "2020-12-31"

# TDX server pool — 与原 fetcher 一致
TDX_SERVERS = [
    ("180.153.18.170", 7709),
    ("123.125.108.14", 7709),
    ("218.6.170.47", 7709),
    ("60.12.136.250", 7709),
    ("115.238.56.198", 7709),
    ("115.238.90.165", 7709),
]

SINA_HFQ_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"

# 单线程 (本任务核心约束)
N_WORKERS = 1

# 限流参数
SLEEP_BETWEEN_REQ = 0.4   # 每股之间
RETRY_SLEEP = 1.5         # 失败后 retry 间隔
MAX_RETRIES = 3           # 每股最大尝试次数 (1 + 2 retry)
PROGRESS_EVERY = 50       # print 进度间隔

# Wall-time guard: 连续失败 N 股后, 后续股只 try 1 次 (skip retry).
# 因为 6 个 server 全 rotate 过仍 100% fail = 这批股 TDX 真无数据,
# retry 是浪费 wall time. 检测到首次成功后, 重置回完整 retry.
FAILSTREAK_SKIP_RETRY_THRESHOLD = 3

# Hard wall-time abort: 到达后立即 break loop, 落盘当前已抓到的数据.
# 用户约束 90min 上限, 留 5min 给落盘/验证 → 85min cutoff.
MAX_WALL_TIME_SEC = 85 * 60

# requests 配置
PROXIES_NULL = {"http": None, "https": None}
HTTP_TIMEOUT = 15


# ──────────────────────────────────────────────────────────────────────────
# Universe + 失败清单加载
# ──────────────────────────────────────────────────────────────────────────

def load_retry_universe() -> pd.DataFrame:
    """读 universe + 失败清单 inner join → 拿到 1311 股的完整 (code,sina_sym)."""
    universe = pd.read_csv(UNIVERSE_PATH)
    universe["code"] = universe["code"].astype(str).str.zfill(6)
    universe["market_int"] = universe["market"].str.lower().map({"sh": 1, "sz": 0})
    universe["sina_sym"] = universe["market"].str.lower() + universe["code"]
    universe = universe.dropna(subset=["market_int"]).copy()
    universe["market_int"] = universe["market_int"].astype(int)

    failed = pd.read_csv(FAILED_INPUT_PATH)
    failed["code"] = failed["code"].astype(str).str.zfill(6)

    retry_df = universe.merge(failed[["code"]], on="code", how="inner")
    return retry_df


# ──────────────────────────────────────────────────────────────────────────
# Sina HFQ factor (与原 fetcher 一致)
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
# TDX 拉前复权 daily K (与原 fetcher 一致)
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
# 单股完整 pipeline (与原 fetcher 一致, 返回 hfq daily)
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
# Client init with fallback
# ──────────────────────────────────────────────────────────────────────────

def init_client_with_fallback(start_idx: int = 0) -> Tuple[Optional[Quotes], Optional[Tuple[str, int]]]:
    n = len(TDX_SERVERS)
    for i in range(n):
        srv = TDX_SERVERS[(start_idx + i) % n]
        try:
            return make_mootdx_client(srv), srv
        except Exception:
            time.sleep(0.5)
            continue
    return None, None


# ──────────────────────────────────────────────────────────────────────────
# Main: 单线程 + sleep + per-stock retry 3 次
# ──────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    retry_universe = load_retry_universe()
    n_total = len(retry_universe)
    print(f"[main] retry universe size: {n_total}", flush=True)
    print(f"[main] N_WORKERS={N_WORKERS}, sleep={SLEEP_BETWEEN_REQ}s, "
          f"retry={MAX_RETRIES}x per stock", flush=True)

    # 初始化 client
    client, current_srv = init_client_with_fallback(0)
    if client is None:
        print("[main] FATAL: cannot init any TDX client", flush=True)
        sys.exit(1)
    print(f"[main] mootdx client @ {current_srv}", flush=True)
    session = requests.Session()

    results = []
    consecutive_fails = 0
    server_rotation_idx = 0

    aborted_at = None
    for i, row in enumerate(retry_universe.itertuples(index=False), start=1):
        # Hard wall-time abort
        if time.time() - t_start > MAX_WALL_TIME_SEC:
            aborted_at = i
            print(f"[ABORT] wall-time {MAX_WALL_TIME_SEC}s reached at stock {i}/{n_total} "
                  f"→ break loop, write partial results", flush=True)
            break

        code = row.code
        sina_sym = row.sina_sym

        # Wall-time guard: 在 fail streak 中只 try 1 次, 否则用完整 MAX_RETRIES.
        # 首次成功立即重置 consecutive_fails → 后续股恢复完整 retry.
        if consecutive_fails >= FAILSTREAK_SKIP_RETRY_THRESHOLD:
            attempts_for_this_stock = 1
        else:
            attempts_for_this_stock = MAX_RETRIES

        df, status = None, "init"
        for attempt in range(1, attempts_for_this_stock + 1):
            _, df, status = fetch_one_stock(client, session, code, sina_sym)
            if status == "ok":
                break
            # 失败 → sleep 然后 retry
            if attempt < attempts_for_this_stock:
                time.sleep(RETRY_SLEEP)

        if status == "ok":
            consecutive_fails = 0
            results.append((code, df, status))
        else:
            consecutive_fails += 1
            results.append((code, None, status))

        # 连续失败 5+ → 换 TDX server
        if consecutive_fails >= 5:
            try:
                client.client.disconnect()
            except Exception:
                pass
            server_rotation_idx += 1
            client, current_srv = init_client_with_fallback(server_rotation_idx)
            if client is None:
                print(f"[{i}] FATAL: all TDX servers dead, abort", flush=True)
                break
            print(f"[{i}] rotated TDX server → {current_srv}", flush=True)
            consecutive_fails = 0

        # 限流 sleep
        time.sleep(SLEEP_BETWEEN_REQ)

        # 进度
        if i % PROGRESS_EVERY == 0:
            n_ok = sum(1 for _, d, s in results if s == "ok")
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_total - i) / rate if rate > 0 else float("inf")
            print(
                f"[progress {i}/{n_total}] ok={n_ok} "
                f"elapsed={elapsed:.0f}s rate={rate:.2f}/s ETA={eta:.0f}s",
                flush=True,
            )

    t_fetch_end = time.time()
    print(f"\n[main] fetch phase done in {t_fetch_end - t_start:.1f}s", flush=True)

    # 未尝试的股 (wall-time abort 之后) 标记为 not_attempted_walltime
    attempted_codes = {c for c, _, _ in results}
    not_attempted = [
        (str(row.code), "not_attempted_walltime")
        for row in retry_universe.itertuples(index=False)
        if str(row.code) not in attempted_codes
    ]
    if not_attempted:
        print(f"[main] {len(not_attempted)} stocks not attempted (wall-time abort)",
              flush=True)

    ok_dfs = [df for _, df, s in results if s == "ok"]
    failed = [(c, s) for c, _, s in results if s != "ok"] + not_attempted
    n_ok = len(ok_dfs)
    n_failed = len(failed)
    print(f"[main] ok: {n_ok}, failed: {n_failed} "
          f"(成功率: {n_ok / n_total * 100:.1f}%)", flush=True)

    # 失败清单
    if failed:
        pd.DataFrame(failed, columns=["code", "fail_reason"]).to_csv(
            FAILED_OUTPUT_PATH, index=False
        )
        print(f"[main] failed list → {FAILED_OUTPUT_PATH}", flush=True)

    if not ok_dfs:
        print("[main] no ok data, nothing to write", flush=True)
        sys.exit(0)

    full = pd.concat(ok_dfs, ignore_index=True)
    print(f"[main] concat shape: {full.shape}", flush=True)

    # 质量验证
    neg_count = int((full["close"] < 0).sum())
    extreme_low = int((full["close"] < 0.5).sum())
    nan_close = int(full["close"].isna().sum())
    full_sorted = full.sort_values(["code", "date"])
    pct_chg = full_sorted.groupby("code")["close"].pct_change().abs()
    extreme_jump = int((pct_chg > 0.11).sum())
    print(
        f"[verify] neg_close={neg_count} extreme_low(<0.5)={extreme_low} "
        f"extreme_jump(>11%)={extreme_jump} nan_close={nan_close}",
        flush=True,
    )

    if neg_count > 0:
        print("[main] FATAL: neg close found — abort write", flush=True)
        sys.exit(2)

    full.to_parquet(OUTPUT_PATH)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"[main] WROTE {OUTPUT_PATH} ({size_mb:.1f} MB)", flush=True)

    t_total = time.time() - t_start
    print(
        f"\n=== FINAL ===\n"
        f"retry universe: {n_total}\n"
        f"ok: {n_ok} ({n_ok / n_total * 100:.1f}%)\n"
        f"failed: {n_failed} ({n_failed / n_total * 100:.1f}%)\n"
        f"wall time: {t_total:.0f}s ({t_total / 60:.1f} min)\n"
        f"output: {OUTPUT_PATH}\n"
        f"failed list: {FAILED_OUTPUT_PATH if failed else 'none'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
