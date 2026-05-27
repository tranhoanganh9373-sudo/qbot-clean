"""Fetch shareholder count for CSI500-only (CSI500 ∖ CSI300) via akshare.

复用 fetch_shareholders_csi300.py 模板, 但写到独立合集 (不动 shareholders_csi300.parquet).

输入: data_cache/qlib_baidu/instruments/csi500.txt - csi300.txt → csi500-only codes
输出:
  data_cache/shareholders/per_stock/{code}.parquet  (复用现有 per_stock 目录, csi500-only 新增)
  data_cache/shareholders/shareholders_csi500.parquet  (csi500-only combined)
  data_cache/shareholders/shareholders_csi300_csi500.parquet  (合 csi300)

注意:
  - 复用 src/claude_finance/shareholders.py 的 fetch_stock + save_per_stock
  - **不调用** shareholders.merge_all() (它写 hardcoded MERGED_FILE=shareholders_csi300.parquet)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claude_finance import shareholders  # noqa: E402

CSI500_INSTR = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "csi500.txt"
CSI300_INSTR = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "csi300.txt"
EXISTING_CSI300_MERGED = ROOT / "data_cache" / "shareholders" / "shareholders_csi300.parquet"
OUT_CSI500_ONLY = ROOT / "data_cache" / "shareholders" / "shareholders_csi500.parquet"
OUT_MERGED = ROOT / "data_cache" / "shareholders" / "shareholders_csi300_csi500.parquet"

RATE_SLEEP = 0.4


def _load_instrument_codes(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sym = line.split("\t")[0]
            out.add(sym[2:])
    return out


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    csi500 = _load_instrument_codes(CSI500_INSTR)
    csi300 = _load_instrument_codes(CSI300_INSTR)
    csi500_only = sorted(csi500 - csi300)
    print(f"[init] csi500={len(csi500)} csi300={len(csi300)} "
          f"csi500-only={len(csi500_only)}", flush=True)

    t_start = time.time()
    n_ok = 0
    n_skip = 0
    failed: list[tuple[str, str]] = []

    shareholders.PER_STOCK_DIR.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(csi500_only, 1):
        per_stock_path = shareholders.PER_STOCK_DIR / f"{code}.parquet"
        if per_stock_path.exists():
            n_skip += 1
            if i % 50 == 0:
                print(f"  [{i:3d}/{len(csi500_only)}] skip-existing "
                      f"(ok={n_ok}, skip={n_skip}, fail={len(failed)}) "
                      f"@ {(time.time()-t_start)/60:.1f}m", flush=True)
            continue
        try:
            df = shareholders.fetch_stock(code, max_retries=3, retry_sleep=2.0)
            shareholders.save_per_stock(df, code)
            n_ok += 1
            if i % 25 == 0 or i == len(csi500_only):
                print(f"  [{i:3d}/{len(csi500_only)}] ok={n_ok} skip={n_skip} "
                      f"fail={len(failed)} latest={code} rows={len(df)} "
                      f"@ {(time.time()-t_start)/60:.1f}m", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append((code, str(exc)[:120]))
            print(f"  [{i:3d}/{len(csi500_only)}] FAIL {code}: "
                  f"{str(exc)[:80]}", flush=True)
        time.sleep(RATE_SLEEP)

    elapsed_min = (time.time() - t_start) / 60
    n_total = len(csi500_only)
    print(f"\n[fetch summary] ok={n_ok} skip={n_skip} fail={len(failed)} "
          f"/ {n_total} in {elapsed_min:.1f}m", flush=True)
    success_rate = (n_ok + n_skip) / n_total * 100 if n_total else 0
    if success_rate < 80:
        print(f"WARN: 成功率 {success_rate:.1f}% < 80%", file=sys.stderr, flush=True)

    # Build csi500-only combined parquet
    print("[merge] 收集 csi500-only per-stock → shareholders_csi500.parquet",
          flush=True)
    parts: list[pd.DataFrame] = []
    for code in csi500_only:
        p = shareholders.PER_STOCK_DIR / f"{code}.parquet"
        if p.exists():
            try:
                sub = pd.read_parquet(p)
                if not sub.empty:
                    sub["code"] = sub["code"].astype(str).str.zfill(6)
                    parts.append(sub)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: skip {p.name}: {exc}", flush=True)
    if parts:
        c500 = pd.concat(parts, ignore_index=True)
        c500["data_date"] = pd.to_datetime(c500["data_date"])
        c500["announce_date"] = pd.to_datetime(c500["announce_date"])
        c500 = (c500.sort_values(["code", "data_date", "announce_date"])
                .drop_duplicates(subset=["code", "data_date"], keep="last")
                .reset_index(drop=True))
        _atomic_write(c500, OUT_CSI500_ONLY)
        print(f"[merge] {OUT_CSI500_ONLY.name}: {len(c500):,} rows × "
              f"{c500['code'].nunique()} codes", flush=True)

        if EXISTING_CSI300_MERGED.exists():
            c300 = pd.read_parquet(EXISTING_CSI300_MERGED)
            c300["code"] = c300["code"].astype(str).str.zfill(6)
            common_cols = [c for c in c500.columns if c in c300.columns]
            merged = pd.concat([c300[common_cols], c500[common_cols]],
                               ignore_index=True)
            merged = (merged.sort_values(["code", "data_date", "announce_date"])
                      .drop_duplicates(subset=["code", "data_date"], keep="last")
                      .reset_index(drop=True))
            _atomic_write(merged, OUT_MERGED)
            print(f"[merge] {OUT_MERGED.name}: {len(merged):,} rows × "
                  f"{merged['code'].nunique()} codes", flush=True)

    covered = sum(
        1 for code in csi500_only
        if (shareholders.PER_STOCK_DIR / f"{code}.parquet").exists()
    )
    pct = covered / len(csi500_only) * 100
    print(f"[verify] csi500-only with per-stock parquet: "
          f"{covered}/{len(csi500_only)} ({pct:.1f}%)", flush=True)

    if failed:
        print(f"\n[failed] {len(failed)}:")
        for code, msg in failed[:20]:
            print(f"  {code}: {msg}")
        if len(failed) > 20:
            print(f"  ... +{len(failed)-20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
