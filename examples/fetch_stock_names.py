"""一次性 fetch A股 sym→中文名 cache via mootdx.

输出: data_cache/stock_names.json
schema: {"SH600519": "贵州茅台", "SZ000001": "平安银行", ...}

mootdx client.stocks(market=0) → SZ, market=1 → SH; 含基金/指数/B股, 需 filter A 股 code.

CLI:
  python examples/fetch_stock_names.py            # 默认 SH+SZ A股全量
  python examples/fetch_stock_names.py --print 5  # 打前 5 个 sample
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
warnings.filterwarnings("ignore")

from mootdx.quotes import Quotes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data_cache" / "stock_names.json"

TDX_SERVERS = [
    ("180.153.18.170", 7709),
    ("123.125.108.14", 7709),
    ("218.6.170.47", 7709),
    ("60.12.136.250", 7709),
    ("115.238.56.198", 7709),
    ("115.238.90.165", 7709),
]


def init_client():
    for srv in TDX_SERVERS:
        try:
            c = Quotes.factory(market="std", server=srv, timeout=10)
            probe = c.bars(symbol="600519", frequency=4, start=0, offset=2)
            if probe is not None and len(probe) > 0:
                return c, srv
        except Exception:
            continue
    return None, None


def is_a_share(market: str, code: str) -> bool:
    """A股 code 过滤. SH: 60*/68*; SZ: 00*/300*/301*/002*/003*."""
    if len(code) != 6 or not code.isdigit():
        return False
    if market == "SH":
        return code.startswith(("600", "601", "603", "605", "688", "689"))
    if market == "SZ":
        return code.startswith(("000", "001", "002", "003", "300", "301"))
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", type=int, default=0)
    args = parser.parse_args()

    print("[stock_names] connecting to mootdx...", flush=True)
    client, srv = init_client()
    if client is None:
        print("[stock_names] ERR: all mootdx servers failed.", file=sys.stderr)
        return 1
    print(f"[stock_names] connected {srv}", flush=True)

    name_map: dict[str, str] = {}
    for market_id, market_prefix in [(1, "SH"), (0, "SZ")]:
        print(f"[stock_names] fetching {market_prefix} (market={market_id})...", flush=True)
        df = client.stocks(market=market_id)
        if df is None or len(df) == 0:
            print(f"[stock_names] WARN: {market_prefix} empty result", file=sys.stderr)
            continue
        n_total = len(df)
        n_added = 0
        for _, row in df.iterrows():
            code = str(row["code"])
            name = str(row.get("name", "")).strip()
            if not is_a_share(market_prefix, code):
                continue
            if not name:
                continue
            sym = f"{market_prefix}{code}"
            name_map[sym] = name
            n_added += 1
        print(f"[stock_names] {market_prefix} total={n_total} A股 added={n_added}",
              flush=True)

    if not name_map:
        print("[stock_names] ERR: empty name_map, abort.", file=sys.stderr)
        return 1

    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(name_map, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    tmp.replace(OUT_PATH)
    print(f"[stock_names] wrote {OUT_PATH} ({len(name_map)} syms)", flush=True)

    if args.print > 0:
        for sym, name in sorted(name_map.items())[: args.print]:
            print(f"  {sym}\t{name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
