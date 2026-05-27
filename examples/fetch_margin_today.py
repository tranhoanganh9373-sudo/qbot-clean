"""Daily incremental margin fetch for CSI300 (用于 v19.4 sidecar overlay).

策略: 按 DATE 维度从 east money datacenter 拉**最近若干交易日** (T-1) 全 A 股 margin
快照, 过滤 CSI300 成份, 计算 margin_5d_chg / margin_20d_chg (与 14yr cache 同公式).

落地到独立 sidecar parquet (data_cache/csi300_margin_daily.parquet) — **不修改**
production 文件 csi300_margin_14yr.parquet (per task 约束 "只读").

paper_trade_today.py 的 load_margin_overlay() 优先读 daily sidecar; 落空则 fallback
到 14yr 长 cache.

为了准确计算 5d/20d pct_change, 需要 ≥ 21 个交易日的历史 — 直接从 14yr cache
读最近 25 天种子, 再 append daily fetch 结果, 不动 14yr cache 本身.

输出 columns: 与 14yr cache 一致 [code, date, rzye, rzmre, rzche, margin_5d_chg,
margin_20d_chg].

Run:
  python examples/fetch_margin_today.py
  python examples/fetch_margin_today.py --start-date 2026-05-20 --end-date 2026-05-22
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent

# 总超时 (launchd 内可能限流, 防 hang 卡住整个 daily_check 流程).
SCRIPT_TIMEOUT_SEC = 300


class FetchTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise FetchTimeout(f"fetch_margin_today exceeded {SCRIPT_TIMEOUT_SEC}s wall time")

CACHE_LONG_PATH = ROOT / "data_cache" / "csi300_margin_14yr.parquet"   # read-only
CACHE_DAILY_PATH = ROOT / "data_cache" / "csi300_margin_daily.parquet"  # writable sidecar
CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"

DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
UA = "Mozilla/5.0"
PAGE_SIZE = 500
MAX_PAGES = 15        # 全 A ~4400 行/天 → 9 页足够, 留 buffer
PAGE_TIMEOUT = 8      # 单次 http 8 秒
PAGE_RETRIES = 2      # 失败 2 次重试即放弃该页 (page-level abort 该天)
SEED_LOOKBACK_DAYS = 35  # 14yr cache 取最近 35 天作 5d/20d pct_change 种子


def fetch_margin_one_day(date_str: str) -> pd.DataFrame:
    """拉 east money datacenter 全 A 股 (filter DATE='YYYY-MM-DD') 一日 margin 快照.
    单页失败 retry 上限后跳过 (返回 empty), 避免整批 hung."""
    all_data: list[dict] = []
    for pn in range(1, MAX_PAGES + 1):
        params = {
            "reportName": "RPTA_WEB_RZRQ_GGMX",
            "columns": "ALL",
            "filter": f"(DATE='{date_str}')",
            "pageNumber": str(pn),
            "pageSize": str(PAGE_SIZE),
            "sortColumns": "SCODE", "sortTypes": "1",
            "source": "WEB", "client": "WEB",
        }
        rows = None
        for attempt in range(PAGE_RETRIES + 1):
            try:
                r = requests.get(DATACENTER_URL, params=params,
                                  headers={"User-Agent": UA}, timeout=PAGE_TIMEOUT)
                rows = (r.json().get("result") or {}).get("data") or []
                break
            except Exception as e:
                if attempt == PAGE_RETRIES:
                    print(f"    [{date_str} page {pn}] fail after {PAGE_RETRIES+1}: {type(e).__name__}",
                          flush=True)
                    return pd.DataFrame()  # 整天放弃 (避免拼半截数据)
                time.sleep(0.3 * (attempt + 1))
        if not rows:
            break
        all_data.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.05)

    if not all_data:
        return pd.DataFrame()

    out_rows = []
    for r in all_data:
        scode = str(r.get("SCODE") or "").zfill(6)
        if not scode or scode == "000000":
            continue
        out_rows.append({
            "code": scode,
            "date": pd.to_datetime(str(r.get("DATE", ""))[:10]),
            "rzye": float(r.get("RZYE") or 0),
            "rzmre": float(r.get("RZMRE") or 0),
            "rzche": float(r.get("RZCHE") or 0),
        })
    return pd.DataFrame(out_rows).drop_duplicates(subset=["code", "date"])


def get_target_dates(args_start: str | None, args_end: str | None,
                     seed_max: pd.Timestamp | None) -> list[str]:
    """决定要抓哪几天 (跳周末)."""
    today = pd.Timestamp.today().normalize()
    if args_start and args_end:
        s = pd.to_datetime(args_start)
        e = pd.to_datetime(args_end)
    elif seed_max is not None:
        s = seed_max + pd.Timedelta(days=1)
        e = today
    else:
        s = today - pd.Timedelta(days=7)
        e = today
    if s > e:
        return []
    dates: list[str] = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5:
            dates.append(cur.strftime("%Y-%m-%d"))
        cur = cur + pd.Timedelta(days=1)
    return dates


def load_seed_from_long_cache(csi300_codes: set[str]) -> pd.DataFrame:
    """从 14yr cache 读最近 SEED_LOOKBACK_DAYS 天 CSI300 数据作种子 (read-only)."""
    if not CACHE_LONG_PATH.exists():
        return pd.DataFrame(columns=["code", "date", "rzye", "rzmre", "rzche"])
    df = pd.read_parquet(CACHE_LONG_PATH,
                         columns=["code", "date", "rzye", "rzmre", "rzche"])
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    cutoff = df["date"].max() - pd.Timedelta(days=SEED_LOOKBACK_DAYS)
    df = df[(df["date"] >= cutoff) & (df["code"].isin(csi300_codes))].copy()
    return df.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    # 总 wall-time 守护 (限流时不卡 launchd 任务).
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(SCRIPT_TIMEOUT_SEC)

    if not CSI300_PATH.exists():
        print(f"FATAL: {CSI300_PATH} 不存在", file=sys.stderr)
        return 1

    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi300_codes = set(csi["code"].astype(str).str.zfill(6).tolist())
    print(f"[init] CSI300: {len(csi300_codes)} 只", flush=True)

    seed = load_seed_from_long_cache(csi300_codes)
    seed_max = seed["date"].max() if not seed.empty else None
    print(f"[seed] 14yr cache 取近 {SEED_LOOKBACK_DAYS} 天种子: {len(seed):,} 行, "
          f"latest = {seed_max.date() if seed_max is not None else 'n/a'}", flush=True)

    # 合并已存在的 daily sidecar (上次跑的结果)
    if CACHE_DAILY_PATH.exists():
        existing_daily = pd.read_parquet(CACHE_DAILY_PATH,
                                         columns=["code", "date", "rzye", "rzmre", "rzche"])
        existing_daily["date"] = pd.to_datetime(existing_daily["date"])
        existing_daily["code"] = existing_daily["code"].astype(str).str.zfill(6)
        print(f"[daily] 已有 sidecar {len(existing_daily):,} 行, "
              f"latest = {existing_daily['date'].max().date()}", flush=True)
        seed = pd.concat([seed, existing_daily], ignore_index=True)
        seed = seed.drop_duplicates(subset=["code", "date"], keep="last")
        seed_max = seed["date"].max()

    dates_to_fetch = get_target_dates(args.start_date, args.end_date, seed_max)
    if not dates_to_fetch:
        print("[done] 无新交易日待抓, exit 0", flush=True)
        return 0
    print(f"[fetch] {len(dates_to_fetch)} 天: {dates_to_fetch[0]} → {dates_to_fetch[-1]}",
          flush=True)

    new_dfs: list[pd.DataFrame] = []
    try:
        for d in dates_to_fetch:
            t0 = time.time()
            df = fetch_margin_one_day(d)
            if df.empty:
                print(f"  {d}: 0 rows (休市/限流)", flush=True)
                continue
            df = df[df["code"].isin(csi300_codes)].reset_index(drop=True)
            print(f"  {d}: {len(df)} rows ({time.time()-t0:.1f}s)", flush=True)
            new_dfs.append(df)
            time.sleep(0.1)
    except FetchTimeout as e:
        print(f"[timeout] {e} — 落地已 fetch 的 {len(new_dfs)} 天部分结果", flush=True)
    finally:
        signal.alarm(0)

    if not new_dfs:
        print("[done] 所有 fetch 结果为空 (可能尚未到收盘时间或限流), exit 0", flush=True)
        return 0

    new_all = pd.concat(new_dfs, ignore_index=True)

    combined = pd.concat([seed, new_all], ignore_index=True)
    combined = combined.drop_duplicates(subset=["code", "date"], keep="last")
    combined = combined.sort_values(["code", "date"]).reset_index(drop=True)

    # 重算 margin_5d_chg / margin_20d_chg per code (与 14yr cache 同公式)
    combined["margin_5d_chg"] = combined.groupby("code")["rzye"].pct_change(5)
    combined["margin_20d_chg"] = combined.groupby("code")["rzye"].pct_change(20)

    # 落地策略: 写"本次 fetch 涉及的最大日期 + seed 段" 整个 combined.
    # paper_trade 优先读 daily sidecar (覆盖率更高的 daily by-date endpoint), 落空再 fallback long.
    # 总是覆写 sidecar (idempotent — 同样的 fetch 日期 → 同样的内容).
    output = combined.copy()
    new_max_date = new_all["date"].max()
    print(f"[merge] new_max_date={new_max_date.date()}, "
          f"output rows={len(output):,}, distinct codes={output['code'].nunique()}",
          flush=True)

    output.to_parquet(CACHE_DAILY_PATH, index=False)
    print(f"[ok] saved {len(output):,} rows → {CACHE_DAILY_PATH.name} "
          f"(date range: {output['date'].min().date()} → {output['date'].max().date()})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
