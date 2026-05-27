"""Fetch CSI500-only (CSI500 ∖ CSI300) margin history via EM datacenter.

复用 fetch_margin_backfill_180.py 的 EM `RPTA_WEB_RZRQ_GGMX` per-stock pagination 模式.

输入: data_cache/qlib_baidu/instruments/csi500.txt 取出 SH/SZ 前缀, strip 后跟
       csi300.txt 做 set 差, 得 csi500-only codes.

输出:
  data_cache/csi500_margin_full.parquet  (CSI500-only codes)
  data_cache/csi300_csi500_margin_full.parquet  (合 csi300_margin_14yr 后)

约束:
  - 不动 data_cache/csi300_margin_14yr.parquet
  - atomic write
  - 6 workers, ETA ~15-30 min
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

CSI500_INSTR = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "csi500.txt"
CSI300_INSTR = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "csi300.txt"
EXISTING_CSI300_14YR = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
OUT_CSI500_ONLY = ROOT / "data_cache" / "csi500_margin_full.parquet"
OUT_MERGED = ROOT / "data_cache" / "csi300_csi500_margin_full.parquet"
FAIL_LOG = ROOT / "data_cache" / "csi500_margin_failed.csv"

DEFAULT_WORKERS = 6
FLUSH_EVERY = 20

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
EM_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
PAGE_SIZE = 500
MAX_PAGES = 20

_lock = threading.Lock()


def _fetch_page(code: str, pn: int, retries: int = 4) -> list[dict] | None:
    params = {
        "reportName": "RPTA_WEB_RZRQ_GGMX",
        "columns": "DATE,SCODE,RZYE,RZMRE,RZCHE",
        "filter": f'(SCODE="{code}")',
        "pageNumber": str(pn), "pageSize": str(PAGE_SIZE),
        "sortColumns": "DATE", "sortTypes": "-1",
        "source": "WEB", "client": "WEB",
    }
    for attempt in range(retries):
        try:
            r = requests.get(EM_URL, params=params,
                             headers={"User-Agent": UA}, timeout=20)
            d = r.json()
            res = d.get("result") or {}
            return res.get("data") or []
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return None


def _fetch_one(code: str):
    t0 = time.time()
    all_data: list[dict] = []
    for pn in range(1, MAX_PAGES + 1):
        rows = _fetch_page(code, pn)
        if rows is None:
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


def _load_instrument_codes(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sym = line.split("\t")[0]
            out.add(sym[2:])  # strip SH/SZ
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume", action="store_true",
                        help="跳过已在 backfill 文件中的 code")
    args = parser.parse_args()

    csi500 = _load_instrument_codes(CSI500_INSTR)
    csi300 = _load_instrument_codes(CSI300_INSTR)
    csi500_only = sorted(csi500 - csi300)
    print(f"[init] CSI500={len(csi500)} CSI300={len(csi300)} "
          f"csi500-only={len(csi500_only)}", flush=True)

    cached_rows: list[pd.DataFrame] = []
    cached_codes: set[str] = set()
    if args.resume and OUT_CSI500_ONLY.exists():
        old = pd.read_parquet(OUT_CSI500_ONLY)
        cached_codes = set(old["code"].astype(str).str.zfill(6).unique())
        cached_rows.append(old)
        print(f"[resume] cache={len(cached_codes)}", flush=True)

    todo = [c for c in csi500_only if c not in cached_codes]
    print(f"[start] {len(todo)} 只 × {args.workers} workers, "
          f"预计 ~{len(todo)*8/args.workers/60:.1f} 分钟",
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
            _atomic_write(big, OUT_CSI500_ONLY)
            print(f"    [{label}] {len(big):,} rows → "
                  f"{OUT_CSI500_ONLY.name}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_fetch_one, c): c for c in todo}
            for fut in as_completed(futures):
                code, df, elapsed, err = fut.result()
                done_count += 1
                if err:
                    failed.append((code, err))
                    if done_count % 25 == 0 or done_count <= 5:
                        print(f"  [{done_count:3d}/{len(todo)}] {code} "
                              f"FAIL: {err}", flush=True)
                else:
                    with _lock:
                        new_rows.append(df)
                    if done_count % 25 == 0 or done_count <= 5:
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

    # Merge CSI500-only with existing CSI300 14yr
    if OUT_CSI500_ONLY.exists() and EXISTING_CSI300_14YR.exists():
        c500 = pd.read_parquet(OUT_CSI500_ONLY)
        c300 = pd.read_parquet(EXISTING_CSI300_14YR)
        c500["code"] = c500["code"].astype(str).str.zfill(6)
        c300["code"] = c300["code"].astype(str).str.zfill(6)

        print(f"[merge] csi500 cols: {list(c500.columns)}", flush=True)
        print(f"[merge] csi300_14yr cols: {list(c300.columns)}", flush=True)

        common_cols = [c for c in c500.columns if c in c300.columns]
        merged = pd.concat([c300[common_cols], c500[common_cols]], ignore_index=True)
        merged = merged.drop_duplicates(subset=["code", "date"]).sort_values(
            ["code", "date"]).reset_index(drop=True)
        _atomic_write(merged, OUT_MERGED)
        print(f"[merge] {OUT_MERGED.name}: {len(merged):,} rows × "
              f"{merged['code'].nunique()} codes "
              f"({merged['date'].min().date()} ~ {merged['date'].max().date()})",
              flush=True)

    n_success = len(todo) - len(failed)
    print(f"\n=== CSI500 margin summary ===", flush=True)
    print(f"  csi500-only todo: {len(csi500_only)}", flush=True)
    print(f"  success: {n_success}", flush=True)
    print(f"  failed:  {len(failed)} ({len(failed)/max(len(todo),1)*100:.1f}%)",
          flush=True)
    print(f"  total elapsed: {(time.time()-t_start)/60:.1f}m", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
