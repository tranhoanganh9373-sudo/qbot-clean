"""把 data_cache/baidu_kline.parquet 转 qlib bin 格式.

复用 convert_parquet_to_qlib.py 的 schema 模式, 但
适配 Baidu schema: code,date,open,close,high,low,vol,amount,ma5,ma10,ma20,turnoverratio

Output: data_cache/qlib_baidu/   (provider_uri 直接指向这里)

Run:  python examples/convert_baidu_to_qlib.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"
CSV_DIR = ROOT / "data_cache" / "qlib_baidu_csv"
QLIB_BIN = ROOT / "data_cache" / "qlib_baidu"
DUMP_BIN_SCRIPT = Path("/Volumes/SSD/finance/qlib/scripts/dump_bin.py")


def main() -> None:
    print(f"[1/3] loading {PARQUET.name} ...")
    t = time.time()
    df = pd.read_parquet(PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.rename(columns={"vol": "volume"})
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"  ok ({time.time() - t:.1f}s) — {df['code'].nunique()} stocks × "
          f"{len(df):,} rows  date {df['date'].min().date()} → "
          f"{df['date'].max().date()}")

    print(f"\n[2/3] writing per-symbol CSV → {CSV_DIR} ...")
    t = time.time()
    if CSV_DIR.exists():
        for f in CSV_DIR.glob("*.csv"):
            f.unlink()
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "factor"]
    n_written = 0
    for code, grp in df.groupby("code"):
        if grp["close"].isna().all():
            continue
        prefix = "sh" if code.startswith(("6", "9")) else (
            "bj" if code.startswith("8") else "sz")
        sym = f"{prefix}{code}"
        out = grp.copy()
        out["symbol"] = sym
        out["factor"] = 1.0
        out = out[cols].dropna(subset=["close"])
        if len(out) < 60:
            continue
        out.to_csv(CSV_DIR / f"{sym}.csv", index=False)
        n_written += 1
    print(f"  ok ({time.time() - t:.1f}s) — {n_written} CSVs written")

    print("\n[3/3] running qlib dump_bin ...")
    if QLIB_BIN.exists():
        for f in QLIB_BIN.rglob("*"):
            if f.is_file():
                f.unlink()
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
    print(f"  cmd: {' '.join(cmd[-10:])}")
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
