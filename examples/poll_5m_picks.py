"""5m execution overlay poller — B 路线 (不动 production 选股, 仅刷新行情供 dashboard 显示).

每 60s 一轮:
  1. 读 picks_today.json (Top 8) + portfolio_state.json (holdings)
  2. 合并 set, 标 category (pick / holding / both)
  3. mootdx 抓每只最近 2 个 5m bar (含未完成当前 bar)
  4. atomic write data_cache/live_5m_picks.json

为什么独立 daemon 而非接现有 fetch:
  - fetch_mootdx_5m_5y_backfill.py 是一次性 batch (8 worker 抢满 mootdx pool)
  - poller 是常驻 (~16 syms × 60s 抓一次, 跟 batch fetch 资源量级差 100x)
  - 失败容错: mootdx down 时 daemon 仍跑, dashboard 显示 stale 时间戳

CLI:
  python examples/poll_5m_picks.py                # 默认 60s interval
  python examples/poll_5m_picks.py --interval 120 # 自定义
  python examples/poll_5m_picks.py --once         # 跑一轮即退出 (smoke test)
"""
from __future__ import annotations

# 所有时间显示用北京时间 (A 股交易时区), 不参考 Mac 本机时区.
import os
os.environ["TZ"] = "Asia/Shanghai"
import time
time.tzset()

import argparse
import json
import signal
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from mootdx.quotes import Quotes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PICKS_PATH = ROOT / "data_cache" / "picks_today.json"
STATE_PATH = ROOT / "data_cache" / "portfolio_state.json"
OUT_PATH = ROOT / "data_cache" / "live_5m_picks.json"
# Persistent 5m bars storage — 复用 backfill 的 kline_5m_shards 目录, atomic append.
# 每 60s poller 拿到的 24 bars 跟现有 per-sym parquet dedup by datetime, 新 bars 追加.
# 长期累积让 DuckDB Hive view 自动 pick up (Hive rebuild 可手动 build_5m_hive_duckdb.py).
PERSIST_SHARDS_DIR = ROOT / "data_cache" / "kline_5m_shards"

# 沿用 fetch_mootdx_5m_5y_backfill.py 的 TDX server 池 (兼容性).
TDX_SERVERS = [
    ("180.153.18.170", 7709),
    ("123.125.108.14", 7709),
    ("218.6.170.47", 7709),
    ("60.12.136.250", 7709),
    ("115.238.56.198", 7709),
    ("115.238.90.165", 7709),
]
FREQ_5M = 0  # mootdx frequency code for 5-min bar (0=5m, 1=15m, 4=daily, 7=1m).
# 之前误用 freq=4 (日线), 表现是每只都返回 "YYYY-MM-DD 15:00" 收盘 bar.
# fetch_mootdx_5m_5y_backfill.py 也用 FREQ_5M=0, 保持一致.
BARS_FOR_SPARKLINE = 24  # 抓 2h intraday close 序列, dashboard 画 sparkline
CLIENT_TIMEOUT = 10
BARS_PER_POLL = 2  # 取最近 2 根, 包含未完成当前 bar (实时性)

_TMP_SUFFIX = ".poller.tmp"


def make_client(server: Tuple[str, int]) -> Optional[Quotes]:
    try:
        return Quotes.factory(market="std", server=server, timeout=CLIENT_TIMEOUT)
    except Exception:
        return None


def init_client_with_failover() -> Tuple[Optional[Quotes], Optional[Tuple]]:
    """Round-robin servers, return first one that successfully fetches a probe bar."""
    for srv in TDX_SERVERS:
        c = make_client(srv)
        if c is None:
            continue
        try:
            probe = c.bars(symbol="600519", frequency=FREQ_5M, start=0, offset=2)
            if probe is not None and len(probe) > 0:
                return c, srv
        except Exception:
            pass
    return None, None


def load_pick_syms() -> list[str]:
    if not PICKS_PATH.exists():
        return []
    try:
        d = json.loads(PICKS_PATH.read_text(encoding="utf-8"))
        return [p["sym"] for p in d.get("picks", []) if "sym" in p]
    except Exception:
        return []


def load_jzf_map() -> dict[str, float]:
    if not PICKS_PATH.exists():
        return {}
    try:
        d = json.loads(PICKS_PATH.read_text(encoding="utf-8"))
        m: dict[str, float] = {}
        for p in d.get("picks", []):
            sym = p.get("sym")
            jzf = p.get("jzf")
            if sym and jzf is not None:
                m[sym] = float(jzf)
        return m
    except Exception:
        return {}


def load_holding_syms() -> list[str]:
    """portfolio_state.json schema: {date: str, holdings: list[str]} — list of plain sym strings."""
    if not STATE_PATH.exists():
        return []
    try:
        d = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        holdings = d.get("holdings") or d.get("positions") or []
        if isinstance(holdings, dict):
            return list(holdings.keys())
        if isinstance(holdings, list):
            out = []
            for h in holdings:
                if isinstance(h, str):
                    out.append(h)
                elif isinstance(h, dict) and h.get("sym"):
                    out.append(h["sym"])
            return out
    except Exception:
        pass
    return []


def merged_universe() -> list[tuple[str, str]]:
    """合并 picks + holdings, 返 (sym, category) 列表."""
    picks = set(load_pick_syms())
    holdings = set(load_holding_syms())
    out: list[tuple[str, str]] = []
    for sym in sorted(picks | holdings):
        if sym in picks and sym in holdings:
            cat = "both"
        elif sym in picks:
            cat = "pick"
        else:
            cat = "holding"
        out.append((sym, cat))
    return out


def sym_to_raw(sym: str) -> Optional[str]:
    """SH600519 → 600519 (mootdx 要纯 6-digit code)."""
    if len(sym) < 8 or sym[:2] not in ("SH", "SZ"):
        return None
    return sym[2:]


def fetch_latest_bars(client: Quotes, raw: str) -> Optional[pd.DataFrame]:
    """抓最近 BARS_FOR_SPARKLINE 个 5m bars (含未完成当前 bar).
    返完整 DataFrame 供 latest_5m + sparkline 双用."""
    try:
        df = client.bars(symbol=raw, frequency=FREQ_5M, start=0,
                         offset=BARS_FOR_SPARKLINE)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    if (df["close"] <= 0).any():
        return None
    return df


def _persist_bars(sym: str, df: pd.DataFrame) -> int:
    """Append new 5m bars to persistent shard ({sym}.parquet) — dedup by datetime.
    返新增 row 数. Schema 跟 fetch_mootdx_5m_5y_backfill normalize_schema 一致."""
    shard_path = PERSIST_SHARDS_DIR / f"{sym}.parquet"
    # Schema normalize (跟 backfill 一致, code + 8 列).
    df_norm = pd.DataFrame({
        "code": sym,
        "datetime": pd.to_datetime(df["datetime"]),
        "open": df["open"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "close": df["close"].astype("float64"),
        "volume": df["vol"].astype("int64") if "vol" in df.columns
                  else df["volume"].astype("int64"),
        "amount": df["amount"].astype("float64") if "amount" in df.columns
                  else 0.0,
    })
    if shard_path.exists():
        try:
            existing = pd.read_parquet(shard_path)
            existing_dts = set(existing["datetime"])
            new_rows = df_norm[~df_norm["datetime"].isin(existing_dts)]
            if len(new_rows) == 0:
                return 0
            merged = pd.concat([existing, new_rows], ignore_index=True)
            merged = merged.sort_values("datetime").reset_index(drop=True)
            n_new = len(new_rows)
        except Exception:
            return 0
    else:
        merged = df_norm.sort_values("datetime").reset_index(drop=True)
        n_new = len(merged)
    # atomic write
    PERSIST_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = shard_path.with_suffix(".parquet.poller.tmp")
    try:
        merged.to_parquet(tmp, index=False)
        tmp.replace(shard_path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        return 0
    return n_new


def df_to_latest_dict(df: pd.DataFrame) -> dict:
    """取最后 1 根 bar + ret_5m_pct = (close/open - 1) × 100.

    Datetime shift: mootdx/TDX 用 end-of-period timestamp (13:25 bar = 13:20-13:25 区间).
    Dashboard 显示需 bar 起始时间 (避免 "未来 5min" 错觉) → datetime - 5min.
    Persist 到 parquet 仍用 TDX 原始 (跟 backfill 数据一致, DuckDB query 不破坏)."""
    last = df.iloc[-1]
    open_v = float(last["open"])
    close_v = float(last["close"])
    ret = ((close_v / open_v) - 1.0) * 100.0 if open_v > 0 else 0.0
    vol_col = "vol" if "vol" in df.columns else "volume"
    dt_start = pd.to_datetime(last["datetime"]) - pd.Timedelta(minutes=5)
    return {
        "datetime": str(dt_start)[:19],
        "open": round(open_v, 4),
        "high": round(float(last["high"]), 4),
        "low": round(float(last["low"]), 4),
        "close": round(close_v, 4),
        "volume": int(last[vol_col]) if vol_col in df.columns else 0,
        "ret_5m_pct": round(ret, 4),
    }


def poll_once(client: Quotes) -> dict:
    syms_cat = merged_universe()
    jzf_map = load_jzf_map()
    syms_out: list[dict] = []
    n_ok = 0
    n_fail = 0
    for sym, cat in syms_cat:
        raw = sym_to_raw(sym)
        if raw is None:
            n_fail += 1
            syms_out.append({"sym": sym, "category": cat, "error": "bad_sym_format"})
            continue
        df = fetch_latest_bars(client, raw)
        if df is None:
            n_fail += 1
            syms_out.append({"sym": sym, "category": cat, "error": "fetch_fail"})
            continue
        n_ok += 1
        # persist 新 bars 到 kline_5m_shards/{sym}.parquet (长期累积 5m 历史)
        try:
            _persist_bars(sym, df)
        except Exception:
            pass  # 持久化失败不影响 JSON dashboard 显示
        # sparkline data: 取最后 BARS_FOR_SPARKLINE 个 bar 的 close 列, ≤24 个 float
        spark_closes = [round(float(c), 4) for c in df["close"].tolist()]
        syms_out.append({
            "sym": sym, "category": cat,
            "latest_5m": df_to_latest_dict(df),
            "jzf": jzf_map.get(sym),
            "sparkline_closes": spark_closes,
        })
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "n_polled": len(syms_cat),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "syms": syms_out,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + _TMP_SUFFIX)
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def signal_handler_factory(stop_flag: dict):
    def _handler(signum, frame):
        stop_flag["stop"] = True
        print(f"[poll_5m] caught signal {signum}, will exit after current poll.", flush=True)
    return _handler


def main():
    parser = argparse.ArgumentParser(description="5m execution overlay poller (B 路线).")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    print(f"[poll_5m] start. interval={args.interval}s out={OUT_PATH}", flush=True)
    print(f"[poll_5m] mootdx server pool size={len(TDX_SERVERS)}", flush=True)

    client, srv = init_client_with_failover()
    if client is None:
        print("[poll_5m] ERR: mootdx all servers failed init, exit.", file=sys.stderr)
        return 1
    print(f"[poll_5m] connected to {srv}", flush=True)

    stop_flag = {"stop": False}
    signal.signal(signal.SIGTERM, signal_handler_factory(stop_flag))
    signal.signal(signal.SIGINT, signal_handler_factory(stop_flag))

    round_idx = 0
    while not stop_flag["stop"]:
        round_idx += 1
        t0 = time.time()
        try:
            payload = poll_once(client)
            atomic_write_json(OUT_PATH, payload)
            elapsed = time.time() - t0
            print(
                f"[poll_5m] round={round_idx} polled={payload['n_polled']} "
                f"ok={payload['n_ok']} fail={payload['n_fail']} elapsed={elapsed:.2f}s",
                flush=True,
            )
            # WS broadcast (fail-tolerant): 让 dashboard 收到事件后局部刷新 5m 行情
            try:
                from claude_finance.ws_notify import ws_notify
                ws_notify("bar_5m_update", {
                    "round": round_idx,
                    "n_polled": payload.get("n_polled", 0),
                    "n_ok": payload.get("n_ok", 0),
                    "n_fail": payload.get("n_fail", 0),
                    "elapsed_sec": round(elapsed, 2),
                    "asof": payload.get("asof", ""),
                })
            except Exception:
                pass
        except Exception as exc:
            print(f"[poll_5m] ERR round={round_idx}: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            client, srv = init_client_with_failover()
            if client is None:
                print("[poll_5m] re-init failed, exit to let launchd restart.",
                      file=sys.stderr, flush=True)
                return 1
            print(f"[poll_5m] reconnected to {srv}", flush=True)

        if args.once:
            break

        sleep_for = max(1, args.interval - int(time.time() - t0))
        for _ in range(sleep_for):
            if stop_flag["stop"]:
                break
            time.sleep(1)

    print("[poll_5m] stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
