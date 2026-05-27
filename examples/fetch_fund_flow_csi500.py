"""Fetch fund_flow for CSI500-only (CSI500 ∖ CSI300) via Sina money.finance.

复用 fetch_fund_flow_csi300_oos.py 的 Sina API + 1-worker 1s sleep 模式.

输入: data_cache/qlib_baidu/instruments/csi500.txt - csi300.txt → csi500-only codes.
       从 csi500.txt 拿 SH/SZ prefix 推 market (sh/sz).

输出:
  data_cache/fund_flow_csi500/{code}.parquet  (per-stock)
  data_cache/fund_flow/fund_flow_csi500.parquet  (combined csi500-only)
  data_cache/fund_flow/fund_flow_csi300_csi500.parquet  (合 csi300)

约束:
  - 不动 data_cache/fund_flow/fund_flow_csi300.parquet 与 per-stock {code}.parquet
  - 1-worker 1s sleep (Sina IP ban 风险)
  - 抓全 history 2014-01 ~ 2026-05
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CSI500_INSTR = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "csi500.txt"
CSI300_INSTR = ROOT / "data_cache" / "qlib_baidu" / "instruments" / "csi300.txt"
PER_STOCK_DIR = ROOT / "data_cache" / "fund_flow_csi500"
OUT_CSI500_ONLY = ROOT / "data_cache" / "fund_flow" / "fund_flow_csi500.parquet"
EXISTING_CSI300 = ROOT / "data_cache" / "fund_flow" / "fund_flow_csi300.parquet"
OUT_MERGED = ROOT / "data_cache" / "fund_flow" / "fund_flow_csi300_csi500.parquet"

FETCH_START = pd.Timestamp("2014-01-01")
FETCH_END = pd.Timestamp("2026-05-31")

NUM_PER_PAGE = 500
MAX_PAGES = 12
PER_REQUEST_SLEEP = 1.0
REQUEST_TIMEOUT = 20
RETRY_TIMES = 5
RETRY_BACKOFF = 5.0

SINA_URL = (
    "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "MoneyFlow.ssl_qsfx_lscjfb"
)


def _load_instrument_pairs(path: Path) -> dict[str, str]:
    pairs: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sym = line.split("\t")[0]
            code = sym[2:]
            market = sym[:2].lower()
            pairs[code] = market
    return pairs


def _code_to_sina_symbol(code: str, market: str) -> str:
    return f"{market}{str(code).zfill(6)}"


def _fetch_page(symbol: str, page: int) -> list[dict]:
    params = {
        "page": page,
        "num": NUM_PER_PAGE,
        "sort": "opendate",
        "asc": 0,
        "daima": symbol,
    }
    last_err: Exception | None = None
    for attempt in range(RETRY_TIMES):
        try:
            r = requests.get(SINA_URL, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                try:
                    data = json.loads(r.text)
                    if isinstance(data, list):
                        time.sleep(PER_REQUEST_SLEEP)
                        return data
                    return []
                except json.JSONDecodeError as e:
                    last_err = e
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
            cooldown = RETRY_BACKOFF * (attempt + 1) * (
                10 if r.status_code == 456 else 1
            )
            last_err = RuntimeError(f"HTTP {r.status_code}")
            time.sleep(cooldown)
        except requests.RequestException as e:
            last_err = e
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise RuntimeError(
        f"sina fetch failed for {symbol} page={page}: {last_err!r}"
    )


def _fetch_stock(code: str, market: str) -> tuple[str, pd.DataFrame, str]:
    symbol = _code_to_sina_symbol(code, market)
    out_path = PER_STOCK_DIR / f"{code}.parquet"
    if out_path.exists():
        try:
            df = pd.read_parquet(out_path)
            if not df.empty:
                return code, df, "skip-cached"
        except Exception:
            out_path.unlink()

    rows: list[dict] = []
    try:
        for page in range(1, MAX_PAGES + 1):
            data = _fetch_page(symbol, page)
            if not data:
                break
            rows.extend(data)
            last_date = pd.Timestamp(data[-1]["opendate"])
            if last_date < FETCH_START:
                break
    except RuntimeError as e:
        return code, pd.DataFrame(), f"fail:{e}"

    if not rows:
        return code, pd.DataFrame(), "empty"

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["opendate"])
    df = df[(df["date"] >= FETCH_START) & (df["date"] <= FETCH_END)]
    if df.empty:
        return code, df, "empty-range"

    numeric_cols = [
        "trade", "changeratio", "turnover", "netamount", "ratioamount",
        "r0", "r1", "r2", "r3", "r0_net", "r1_net", "r2_net", "r3_net",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["code"] = str(code).zfill(6)
    df["net_super_big"] = df["r0_net"]
    df["net_big"] = df["r1_net"]
    df["net_mid"] = df["r2_net"]
    df["net_small"] = df["r3_net"]
    df["net_main"] = df["r0_net"] + df["r1_net"]
    total_vol = df["r0"] + df["r1"] + df["r2"] + df["r3"]
    df["total_amount"] = total_vol
    df["pct_super_big"] = df["r0_net"] / total_vol.where(total_vol > 0)
    df["pct_big"] = df["r1_net"] / total_vol.where(total_vol > 0)
    df["pct_main"] = df["net_main"] / total_vol.where(total_vol > 0)

    df = (
        df[[
            "code", "date", "trade",
            "net_main", "net_super_big", "net_big", "net_mid", "net_small",
            "pct_super_big", "pct_big", "pct_main",
            "r0", "r1", "r2", "r3", "total_amount", "turnover",
        ]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out_path)
    return code, df, "ok"


def main() -> int:
    PER_STOCK_DIR.mkdir(parents=True, exist_ok=True)
    csi500 = _load_instrument_pairs(CSI500_INSTR)
    csi300_codes = set(_load_instrument_pairs(CSI300_INSTR).keys())
    csi500_only = sorted(c for c in csi500 if c not in csi300_codes)
    print(
        f"[init] csi500={len(csi500)} csi300={len(csi300_codes)} "
        f"csi500-only={len(csi500_only)}; "
        f"range={FETCH_START.date()}~{FETCH_END.date()}",
        flush=True,
    )

    t0 = time.time()
    n_ok = n_skip = n_empty = n_fail = 0
    fail_codes: list[tuple[str, str]] = []

    for done, code in enumerate(csi500_only, 1):
        market = csi500[code]
        _, _df, status = _fetch_stock(code, market)
        if status == "ok":
            n_ok += 1
        elif status == "skip-cached":
            n_skip += 1
        elif status.startswith("empty"):
            n_empty += 1
        else:
            n_fail += 1
            fail_codes.append((code, status))
        if done % 25 == 0 or done == len(csi500_only):
            elapsed = time.time() - t0
            eta_min = elapsed / done * (len(csi500_only) - done) / 60
            print(
                f"  [{done}/{len(csi500_only)}] "
                f"ok={n_ok} skip={n_skip} empty={n_empty} fail={n_fail} "
                f"elapsed={elapsed/60:.1f}m ETA={eta_min:.1f}m",
                flush=True,
            )

    print(
        f"\n[fetch] DONE in {(time.time()-t0)/60:.1f}m | "
        f"ok={n_ok}, cached={n_skip}, empty={n_empty}, fail={n_fail}",
        flush=True,
    )
    if fail_codes:
        print(f"[fail] sample (first 10): {fail_codes[:10]}", flush=True)

    parts = []
    for code in csi500_only:
        p = PER_STOCK_DIR / f"{code}.parquet"
        if p.exists():
            try:
                parts.append(pd.read_parquet(p))
            except Exception as e:
                print(f"  [combine] skip corrupt {code}: {e}", flush=True)
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
        OUT_CSI500_ONLY.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_CSI500_ONLY.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, index=False)
        tmp.rename(OUT_CSI500_ONLY)
        print(
            f"[combine] {OUT_CSI500_ONLY.name}: {len(combined):,} rows × "
            f"{combined['code'].nunique()} codes",
            flush=True,
        )

        if EXISTING_CSI300.exists():
            csi300_df = pd.read_parquet(EXISTING_CSI300)
            common_cols = [c for c in combined.columns if c in csi300_df.columns]
            merged = pd.concat(
                [csi300_df[common_cols], combined[common_cols]], ignore_index=True
            )
            merged = merged.drop_duplicates(
                subset=["code", "date"]
            ).sort_values(["code", "date"]).reset_index(drop=True)
            tmp2 = OUT_MERGED.with_suffix(".parquet.tmp")
            merged.to_parquet(tmp2, index=False)
            tmp2.rename(OUT_MERGED)
            print(
                f"[merge] {OUT_MERGED.name}: {len(merged):,} rows × "
                f"{merged['code'].nunique()} codes "
                f"({merged['date'].min().date()}~{merged['date'].max().date()})",
                flush=True,
            )
    else:
        print("FATAL: nothing to combine", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
