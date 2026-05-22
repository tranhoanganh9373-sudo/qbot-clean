"""Concurrent multi-stock scanner with parquet-based resume cache."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from claude_finance.data.akshare_loader import load_akshare_daily
from claude_finance.decision import analyze_one

DEFAULT_MAX_WORKERS = 5      # 再高会被东财限速
DEFAULT_PER_REQ_SLEEP = 0.15
DEFAULT_MIN_TRADING_DAYS = 90  # 上市未满 N 个交易日跳过（指标不可靠）


def process_one(
    code: str,
    name: str,
    *,
    per_req_sleep: float = DEFAULT_PER_REQ_SLEEP,
    min_trading_days: int = DEFAULT_MIN_TRADING_DAYS,
) -> dict:
    """Fetch + analyze a single stock. Returns error dict on failure."""
    try:
        time.sleep(per_req_sleep)
        end = pd.Timestamp.today().strftime("%Y%m%d")
        start = (pd.Timestamp.today() - pd.Timedelta(days=300)).strftime("%Y%m%d")
        df = load_akshare_daily(code, start, end, adjust="qfq", use_cache=False)
        if len(df) < min_trading_days:
            return {"code": code, "name": name, "error": f"数据不足({len(df)}日)"}
        res = analyze_one(df.tail(180), name)
        res["code"] = code
        return res
    except Exception as e:
        return {"code": code, "name": name, "error": str(e)[:120]}


def run_scan(
    universe: pd.DataFrame,
    *,
    cache_file: Path | str | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    per_req_sleep: float = DEFAULT_PER_REQ_SLEEP,
    min_trading_days: int = DEFAULT_MIN_TRADING_DAYS,
    verbose: bool = True,
) -> list[dict]:
    """Run the full universe through analyze_one, with resume support.

    universe: DataFrame with columns 'code' and 'name'
    cache_file: parquet path to read/write; pass None to disable caching
    """
    cache_path = Path(cache_file) if cache_file else None
    t0 = time.time()
    total = len(universe)

    done: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            done = {row["code"]: row.to_dict() for _, row in cached.iterrows()}
            if verbose:
                print(f"  从缓存恢复 {len(done)} 条记录\n")
        except Exception as e:
            if verbose:
                print(f"  缓存读取失败 ({e})，从头开始\n")

    todo = [
        (c, n) for c, n in zip(universe["code"], universe["name"], strict=False) if c not in done
    ]
    results: list[dict] = list(done.values())
    fail_count = sum(1 for r in results if "error" in r)
    success_count = len(results) - fail_count

    if verbose:
        print(
            f"\n[2/3] 并发拉取 {len(todo)} 只 (workers={max_workers}, "
            f"预计 {total * per_req_sleep / max_workers / 60:.0f}-{total / 100:.0f} 分钟)..."
        )

    last_print = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                process_one, c, n, per_req_sleep=per_req_sleep, min_trading_days=min_trading_days
            ): (c, n)
            for c, n in todo
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if "error" in res:
                fail_count += 1
            else:
                success_count += 1

            n_done = success_count + fail_count
            if verbose and (n_done % 50 == 0 or time.time() - last_print > 10):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(
                    f"  [{n_done}/{total}] 成功={success_count} 失败={fail_count} "
                    f"速率={rate:.1f}/秒 已用={elapsed / 60:.1f}分 剩余≈{eta / 60:.1f}分"
                )
                last_print = time.time()

                if cache_path and n_done % 500 == 0:
                    pd.DataFrame(results).to_parquet(cache_path, index=False)

    if cache_path:
        pd.DataFrame(results).to_parquet(cache_path, index=False)

    if verbose:
        elapsed = time.time() - t0
        print(
            f"\n[3/3] 扫描完成。总耗时 {elapsed / 60:.1f} 分钟。"
            f"成功 {success_count} 失败 {fail_count}"
        )

    return results
