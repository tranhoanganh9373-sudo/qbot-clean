"""Step 6: atomic swap baidu_kline.parquet + qlib_baidu bin with CSI500 51 kline fix.

流程:
  1. read baidu_kline_csi500_fix.parquet (51 missing CSI500 codes, 688*)
  2. backup baidu_kline.parquet → .pre_csi500.bak
  3. merge fix → write new baidu_kline.parquet via atomic tmp+replace
  4. backup current qlib_baidu/ → qlib_baidu.pre_csi500.bak (mv = atomic)
  5. preserve instrument files
  6. invoke examples/convert_baidu_to_qlib.py (rebuilds qlib_baidu)
  7. restore preserved instrument files to new qlib_baidu/instruments/
  8. verify CSI500 features 494/494 in new bin

约束:
  - 主表 baidu_kline.parquet 修改时有 backup
  - 旧 qlib_baidu → qlib_baidu.pre_csi500.bak (可回滚)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KLINE_MAIN = ROOT / "data_cache" / "baidu_kline.parquet"
KLINE_FIX = ROOT / "data_cache" / "baidu_kline_csi500_fix.parquet"
KLINE_BACKUP = ROOT / "data_cache" / "baidu_kline.parquet.pre_csi500.bak"
QLIB_BIN = ROOT / "data_cache" / "qlib_baidu"
QLIB_BACKUP = ROOT / "data_cache" / "qlib_baidu.pre_csi500.bak"
CONVERT_SCRIPT = ROOT / "examples" / "convert_baidu_to_qlib.py"

INSTRUMENT_FILES = [
    "csi300.txt", "csi500.txt", "all_no_st.txt", "top1500_no_st.txt",
]
INSTRUMENT_STAGING = ROOT / "data_cache" / "instruments_csi500_staging"


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    t_total = time.time()

    if not KLINE_FIX.exists():
        print(f"FATAL: {KLINE_FIX} 不存在, 先跑 fetch_csi500_missing_kline.py",
              file=sys.stderr)
        return 1

    fix = pd.read_parquet(KLINE_FIX)
    fix["code"] = fix["code"].astype(str).str.zfill(6)
    fix["date"] = pd.to_datetime(fix["date"])
    fix_codes = sorted(fix["code"].unique())
    print(f"[1/8] read fix: {len(fix):,} rows × {len(fix_codes)} codes "
          f"({fix['date'].min().date()}~{fix['date'].max().date()})",
          flush=True)

    if not KLINE_BACKUP.exists():
        print(f"[2/8] backup main → {KLINE_BACKUP.name} ...", flush=True)
        t = time.time()
        shutil.copy2(KLINE_MAIN, KLINE_BACKUP)
        print(f"  ok ({time.time()-t:.1f}s)", flush=True)
    else:
        print(f"[2/8] backup already exists, skip ({KLINE_BACKUP.name})",
              flush=True)

    print(f"[3/8] merge fix into main parquet ...", flush=True)
    t = time.time()
    main = pd.read_parquet(KLINE_MAIN)
    main["code"] = main["code"].astype(str).str.zfill(6)
    main["date"] = pd.to_datetime(main["date"])
    before_codes = main["code"].nunique()
    before_rows = len(main)
    print(f"  main: {before_rows:,} rows × {before_codes} codes", flush=True)

    common_cols = [c for c in main.columns if c in fix.columns]
    print(f"  schema intersect: {common_cols}", flush=True)

    combined = pd.concat([main[common_cols], fix[common_cols]],
                         ignore_index=True)
    combined = (combined.sort_values(["code", "date", "close"])
                .drop_duplicates(subset=["code", "date"], keep="last")
                .sort_values(["code", "date"]).reset_index(drop=True))
    new_codes = combined["code"].nunique()
    new_rows = len(combined)
    new_added = sorted(set(combined["code"].unique()) - set(main["code"].unique()))
    print(f"  merged: {new_rows:,} rows × {new_codes} codes "
          f"(added {new_codes - before_codes} new codes)", flush=True)
    print(f"  new codes added: {len(new_added)} (first 5: {new_added[:5]})",
          flush=True)

    _atomic_write(combined, KLINE_MAIN)
    print(f"  atomic write done ({time.time()-t:.1f}s) "
          f"{KLINE_MAIN.stat().st_size/1e6:.1f} MB", flush=True)

    if QLIB_BACKUP.exists():
        print(f"[4/8] qlib backup already exists; aborting swap "
              f"(remove {QLIB_BACKUP.name} first if intended)",
              file=sys.stderr)
        return 1

    print(f"[5/8] preserve instrument files to staging ...", flush=True)
    INSTRUMENT_STAGING.mkdir(parents=True, exist_ok=True)
    for fname in INSTRUMENT_FILES:
        src = QLIB_BIN / "instruments" / fname
        if src.exists():
            shutil.copy2(src, INSTRUMENT_STAGING / fname)
            print(f"  preserved {fname}", flush=True)

    print(f"[4/8] mv {QLIB_BIN.name} → {QLIB_BACKUP.name} (atomic) ...",
          flush=True)
    t = time.time()
    QLIB_BIN.rename(QLIB_BACKUP)
    print(f"  ok ({time.time()-t:.2f}s)", flush=True)

    print(f"[6/8] running convert_baidu_to_qlib.py to rebuild bin ...",
          flush=True)
    t = time.time()
    res = subprocess.run(
        [sys.executable, str(CONVERT_SCRIPT)],
        capture_output=True, text=True,
    )
    print(res.stdout[-2000:], flush=True)
    if res.returncode != 0:
        print(f"FATAL: convert script failed", file=sys.stderr)
        print(res.stderr[-2000:], file=sys.stderr)
        print(f"  rollback: mv {QLIB_BACKUP.name} {QLIB_BIN.name}",
              file=sys.stderr)
        print(f"  rollback: cp {KLINE_BACKUP.name} {KLINE_MAIN.name}",
              file=sys.stderr)
        return 2
    print(f"  rebuild done ({time.time()-t:.1f}s)", flush=True)

    print(f"[7/8] restore instrument files ...", flush=True)
    (QLIB_BIN / "instruments").mkdir(parents=True, exist_ok=True)
    for fname in INSTRUMENT_FILES:
        src = INSTRUMENT_STAGING / fname
        if src.exists():
            shutil.copy2(src, QLIB_BIN / "instruments" / fname)
            print(f"  restored {fname}", flush=True)

    print(f"[8/8] verify CSI500 features ...", flush=True)
    with (QLIB_BIN / "instruments" / "csi500.txt").open() as f:
        csi500 = [l.split("\t")[0].strip() for l in f if l.strip()]
    features_dir = QLIB_BIN / "features"
    have = sum(1 for sym in csi500 if (features_dir / sym.lower()).exists())
    print(f"  CSI500 features present: {have}/{len(csi500)}", flush=True)

    with (QLIB_BIN / "instruments" / "csi300.txt").open() as f:
        csi300 = [l.split("\t")[0].strip() for l in f if l.strip()]
    have300 = sum(1 for sym in csi300 if (features_dir / sym.lower()).exists())
    print(f"  CSI300 features present: {have300}/{len(csi300)}", flush=True)

    print(f"\n=== Step 6 done in {(time.time()-t_total)/60:.1f}m ===", flush=True)
    print(f"  rollback if needed:", flush=True)
    print(f"    mv {QLIB_BIN.name} qlib_baidu.failed && "
          f"mv {QLIB_BACKUP.name} {QLIB_BIN.name}", flush=True)
    print(f"    cp {KLINE_BACKUP.name} {KLINE_MAIN.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
