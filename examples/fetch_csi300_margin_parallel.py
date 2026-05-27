"""Parallel version of fetch_csi300_margin.py — ThreadPoolExecutor.

单线程版每只 ~30-90s (含 7 页 pagination + retry), 300 只 ~5-7 小时, 超 5 hr 预算.
此并行版用 8 workers, 预期 ~50 分钟.

写入相同的 data_cache/csi300_margin_14yr.parquet, 与下游 (v19_2 step1/step2) 兼容.

run:
  python examples/fetch_csi300_margin_parallel.py
"""
from __future__ import annotations

import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

warnings.filterwarnings("ignore")

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
OUT_PATH = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
WORKERS = 8
FLUSH_EVERY = 25


_lock = threading.Lock()


def _fetch_one(code: str):
    from factor_mining import margin_trading
    t0 = time.time()
    try:
        df = margin_trading(code)
    except Exception as e:
        return code, None, time.time() - t0, str(e)[:80]
    if df.empty:
        return code, None, time.time() - t0, "EMPTY"
    df = df.sort_values("date").reset_index(drop=True)
    df["margin_5d_chg"] = df["rzye"].pct_change(5)
    df["margin_20d_chg"] = df["rzye"].pct_change(20)
    return (code,
            df[["code", "date", "rzye", "rzmre", "rzche",
                "margin_5d_chg", "margin_20d_chg"]],
            time.time() - t0, None)


def main() -> int:
    if not CSI300_PATH.exists():
        print(f"FATAL: {CSI300_PATH} 不存在", file=sys.stderr)
        return 1

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    codes = csi["code"].tolist()
    n_total = len(codes)
    print(f"[init] CSI300: {n_total} 只", flush=True)

    cached_codes: set[str] = set()
    cached_rows: list[pd.DataFrame] = []
    if OUT_PATH.exists():
        old = pd.read_parquet(OUT_PATH)
        cached_codes = set(old["code"].astype(str).str.zfill(6).unique())
        cached_rows.append(old)
        print(f"[resume] 已 cache {len(cached_codes)} 只", flush=True)

    todo = [c for c in codes if c not in cached_codes]
    if not todo:
        print(f"[done] 全部 {n_total} 只都已 cache", flush=True)
        return 0

    print(f"[start] {len(todo)} 只 × {WORKERS} workers, 预计 ~"
          f"{len(todo)*30/WORKERS/60:.1f} 分钟", flush=True)

    new_rows: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []
    t_start = time.time()
    done_count = 0

    def _flush(label: str):
        with _lock:
            combined = cached_rows + new_rows
            if not combined:
                return
            big = pd.concat(combined, ignore_index=True)
            big.to_parquet(OUT_PATH, index=False)
            print(f"    [{label}] {len(big):,} rows → {OUT_PATH.name}",
                  flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_fetch_one, c): c for c in todo}
        for fut in as_completed(futures):
            code, df, elapsed, err = fut.result()
            done_count += 1
            if err:
                failed.append((code, err))
                if done_count % 20 == 0 or done_count <= 10:
                    print(f"  [{done_count:3d}/{len(todo)}] {code} FAIL: "
                          f"{err} ({elapsed:.1f}s)", flush=True)
            else:
                with _lock:
                    new_rows.append(df)
                if done_count % 10 == 0 or done_count <= 5:
                    elapsed_total = time.time() - t_start
                    done_frac = done_count / len(todo)
                    eta_min = (elapsed_total / done_frac *
                                (1 - done_frac) / 60) if done_frac > 0 else 0
                    print(f"  [{done_count:3d}/{len(todo)}] {code}: "
                          f"{len(df)} rows ({elapsed:.1f}s) | "
                          f"fail {len(failed)} | "
                          f"elapsed {elapsed_total/60:.1f}m | "
                          f"ETA {eta_min:.1f}m", flush=True)
            if done_count % FLUSH_EVERY == 0:
                _flush(f"flush@{done_count}")

    _flush("final-flush")

    n_success = len(todo) - len(failed)
    fail_pct = len(failed) / max(len(todo), 1) * 100
    print(f"\n=== summary ===", flush=True)
    print(f"  todo:    {len(todo)}", flush=True)
    print(f"  success: {n_success}", flush=True)
    print(f"  failed:  {len(failed)} ({fail_pct:.1f}%)", flush=True)
    print(f"  failed first 20: {[c for c, _ in failed[:20]]}", flush=True)
    print(f"  total elapsed: {(time.time()-t_start)/60:.1f}m", flush=True)
    if fail_pct > 10:
        print(f"FATAL: 失败率 > 10%, abort", file=sys.stderr, flush=True)
        return 2
    print(f"[ok] saved {OUT_PATH.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
