"""Stage 2 — backfill 180 missing CSI300 margin codes using EM datacenter.

CSI300 共 300 股, 现有 csi300_margin_14yr.parquet 只 120 股 (40% 覆盖). 此脚本
通过 EM `RPTA_WEB_RZRQ_GGMX` per-stock pagination API 把缺失 180 股的 2010+
全 history fetch 下来, 落临时 parquet `data_cache/margin_180_backfill.parquet`,
然后 merge 到新文件 `data_cache/csi300_margin_full.parquet` (296 股).

约束:
  - 绝不 overwrite 原 csi300_margin_14yr.parquet (只读)
  - 绝不 touch production paper_trade_today.py / 任何 v17/qlib 文件
  - 临时 parquet atomic write, 中断不丢
  - parallel workers 4-8, total wall time < 60 min

run:
  python examples/fetch_margin_backfill_180.py
  python examples/fetch_margin_backfill_180.py --workers 6  # custom worker count
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent

warnings.filterwarnings("ignore")

MISSING_CSV = ROOT / "data_cache" / "missing_margin_csi300.csv"
EXISTING_14YR = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
BACKFILL_PATH = ROOT / "data_cache" / "margin_180_backfill.parquet"
FULL_PATH = ROOT / "data_cache" / "csi300_margin_full.parquet"
FAIL_LOG = ROOT / "data_cache" / "margin_backfill_180_failed.csv"

DEFAULT_WORKERS = 6
FLUSH_EVERY = 20

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
EM_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
PAGE_SIZE = 500
MAX_PAGES = 20

_lock = threading.Lock()


def _fetch_page(code: str, pn: int, retries: int = 4) -> list[dict] | None:
    """Return list of row dicts or None on hard failure."""
    params = {
        "reportName": "RPTA_WEB_RZRQ_GGMX",
        "columns": "DATE,SCODE,RZYE,RZMRE,RZCHE",
        "filter": f'(SCODE="{code}")',
        "pageNumber": str(pn), "pageSize": str(PAGE_SIZE),
        "sortColumns": "DATE", "sortTypes": "-1",
        "source": "WEB", "client": "WEB",
    }
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(EM_URL, params=params,
                             headers={"User-Agent": UA}, timeout=20)
            d = r.json()
            res = d.get("result") or {}
            return res.get("data") or []
        except Exception as e:
            last_exc = e
            time.sleep(0.8 * (attempt + 1))
    return None


def _fetch_one(code: str):
    """Fetch full margin history for one stock via EM pagination."""
    t0 = time.time()
    all_data: list[dict] = []
    for pn in range(1, MAX_PAGES + 1):
        rows = _fetch_page(code, pn)
        if rows is None:
            # hard network failure even after retries — but maybe we have partial
            return code, None, time.time() - t0, f"NETFAIL_P{pn}"
        if not rows:
            break
        all_data.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.05)
    if not all_data:
        return code, None, time.time() - t0, "EMPTY"
    df = pd.DataFrame([{
        "code": code,
        "date": pd.to_datetime(str(r.get("DATE", ""))[:10]),
        "rzye": float(r.get("RZYE") or 0),
        "rzmre": float(r.get("RZMRE") or 0),
        "rzche": float(r.get("RZCHE") or 0),
    } for r in all_data]).drop_duplicates(subset=["code", "date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["margin_5d_chg"] = df["rzye"].pct_change(5)
    df["margin_20d_chg"] = df["rzye"].pct_change(20)
    return (code,
            df[["code", "date", "rzye", "rzmre", "rzche",
                "margin_5d_chg", "margin_20d_chg"]],
            time.time() - t0, None)


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume", action="store_true",
                        help="跳过已在 backfill 文件中的 code")
    args = parser.parse_args()

    if not MISSING_CSV.exists():
        print(f"FATAL: {MISSING_CSV} 不存在, 先生成 missing 列表",
              file=sys.stderr)
        return 1

    miss = pd.read_csv(MISSING_CSV, dtype={"code": str})
    miss["code"] = miss["code"].astype(str).str.zfill(6)
    todo_all = miss["code"].tolist()
    print(f"[init] missing CSI300: {len(todo_all)} 只", flush=True)

    cached_rows: list[pd.DataFrame] = []
    cached_codes: set[str] = set()
    if args.resume and BACKFILL_PATH.exists():
        old = pd.read_parquet(BACKFILL_PATH)
        cached_codes = set(old["code"].astype(str).str.zfill(6).unique())
        cached_rows.append(old)
        print(f"[resume] backfill 已 cache {len(cached_codes)} 只", flush=True)

    todo = [c for c in todo_all if c not in cached_codes]
    if not todo:
        print(f"[done] 180 只都已 cache", flush=True)
    else:
        print(f"[start] {len(todo)} 只 × {args.workers} workers, "
              f"预计 ~{len(todo)*10/args.workers/60:.1f} 分钟",
              flush=True)

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
            _atomic_write(big, BACKFILL_PATH)
            print(f"    [{label}] {len(big):,} rows → "
                  f"{BACKFILL_PATH.name}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_fetch_one, c): c for c in todo}
            for fut in as_completed(futures):
                code, df, elapsed, err = fut.result()
                done_count += 1
                if err:
                    failed.append((code, err))
                    if done_count % 10 == 0 or done_count <= 10:
                        print(f"  [{done_count:3d}/{len(todo)}] {code} "
                              f"FAIL: {err} ({elapsed:.1f}s)", flush=True)
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

    if failed:
        fail_df = pd.DataFrame(failed, columns=["code", "reason"])
        fail_df.to_csv(FAIL_LOG, index=False)
        print(f"[fail log] {len(failed)} → {FAIL_LOG.name}", flush=True)

    n_success = len(todo) - len(failed)
    fail_pct = len(failed) / max(len(todo), 1) * 100
    print(f"\n=== backfill summary ===", flush=True)
    print(f"  todo:    {len(todo)}", flush=True)
    print(f"  success: {n_success}", flush=True)
    print(f"  failed:  {len(failed)} ({fail_pct:.1f}%)", flush=True)
    if failed:
        print(f"  failed first 20: {[c for c, _ in failed[:20]]}",
              flush=True)
    print(f"  total elapsed: {(time.time()-t_start)/60:.1f}m", flush=True)
    print(f"[ok] saved {BACKFILL_PATH.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
