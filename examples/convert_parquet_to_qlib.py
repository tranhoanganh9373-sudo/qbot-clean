"""Convert data_cache/long_history.parquet → qlib bin format.

Steps:
  1. Read parquet (1,757 stocks × 3 年)
  2. For each code, write CSV with [date, symbol, open, high, low, close, volume, factor]
  3. Run qlib/scripts/dump_bin.py to convert to qlib bin
  4. Auto-creates instruments/all.txt + calendars/day.txt

Output: data_cache/qlib_bin/  ready for qlib.init(provider_uri=...)

Run:  python examples/convert_parquet_to_qlib.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data_cache" / "long_history.parquet"
CSV_DIR = ROOT / "data_cache" / "qlib_csv"
QLIB_BIN = ROOT / "data_cache" / "qlib_bin"
DUMP_BIN_SCRIPT = Path("/Volumes/SSD/finance/qlib/scripts/dump_bin.py")


def main() -> None:
    print(f"[1/3] loading {PARQUET.name} ...")
    t = time.time()
    df = pd.read_parquet(PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"  ok ({time.time() - t:.1f}s) — {df['code'].nunique()} stocks × {len(df):,} rows")

    print(f"\n[2/3] writing per-symbol CSV → {CSV_DIR} ...")
    t = time.time()
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for f in CSV_DIR.glob("*.csv"):
        f.unlink()

    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "factor"]
    for code, grp in df.groupby("code"):
        out = grp.copy()
        out["symbol"] = code
        out["factor"] = 1.0  # 已经前复权
        out = out[cols]
        out.to_csv(CSV_DIR / f"{code}.csv", index=False)
    print(f"  ok ({time.time() - t:.1f}s) — {df['code'].nunique()} CSVs written")

    print(f"\n[3/3] running qlib dump_bin (csv → qlib bin) ...")
    if QLIB_BIN.exists():
        for f in QLIB_BIN.rglob("*"):
            if f.is_file(): f.unlink()
    QLIB_BIN.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(DUMP_BIN_SCRIPT), "dump_all",
        "--data_path", str(CSV_DIR),
        "--qlib_dir", str(QLIB_BIN),
        "--freq", "day",
        "--date_field_name", "date",
        "--symbol_field_name", "symbol",
        "--include_fields", "open,high,low,close,volume,factor",
        "--max_workers", "4",
    ]
    print(f"  cmd: {' '.join(cmd[1:])}")
    t = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  exit code: {res.returncode}, time: {time.time() - t:.1f}s")
    if res.returncode != 0:
        print("STDERR:", res.stderr[-500:])
        sys.exit(1)

    n_features = sum(1 for _ in (QLIB_BIN / "features").iterdir())
    size_mb = sum(f.stat().st_size for f in QLIB_BIN.rglob('*') if f.is_file()) / 1e6
    print(f"\n[done] qlib bin written to {QLIB_BIN}")
    print(f"  features/ contains {n_features} stock dirs")
    print(f"  size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
