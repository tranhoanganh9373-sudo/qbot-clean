"""Phase 1A 速战: 修复已知 corrupt 股 2014-2020 K 线 (hfq 后复权).

corrupt 来源:
    1) baidu_kline 2018 close < 0 或 close < 0.5 (qfq 倒推时除权倒挂 → 负值/极低值)
    2) baidu_kline 2019-2020 close < 0
    3) 已知 5 只持续 corrupt: SH600039 / SH600188 / SH601225 / SH601919 / SZ000408

数据源:
    腾讯 https://web.ifzq.gtimg.cn/appstock/app/fqkline/get (akshare 数据源之一,
    eastmoney 临时 502 时 fallback). Schema 与 baidu_kline 完全同源
    (qfq 数值匹配到小数 0.001), 这里取 hfq (后复权) 避开 qfq 负值问题.

输出:
    data_cache/baidu_kline_corrupt_fix.parquet
        12 列 schema 对齐 baidu_kline.parquet:
        code (6位字符串), date, open, close, high, low, vol (股), amount,
        ma5, ma10, ma20, turnoverratio
    data_cache/corrupt_codes_2014_2020.txt
        corrupt 股代码清单 (每行 6 位).

注意 - merge 须知 (写给 main session):
    本表是 **hfq (后复权)**, baidu_kline.parquet 是 **qfq (前复权)** -
    数值不能直接替换. merge 时整只股的所有日期都要切到 hfq, 或者把 hfq 算 ratio
    转回 qfq 再 patch. 单股部分日期混合 adjust 会让模型时序错乱.
    具体 merge 代码见 memory/project_phase1a_corrupt_fix.md.

约束 (CLAUDE.md):
    - 不动 data_cache/baidu_kline.parquet
    - 不动 forward_oos_monitor 等其它任务在用的文件
    - 顺序拉, 22-23 股 wall time ~1-2 分钟
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
OUT_PARQUET = ROOT / "data_cache" / "baidu_kline_corrupt_fix.parquet"
OUT_LIST = ROOT / "data_cache" / "corrupt_codes_2014_2020.txt"

# 5 only-positive but persistently corrupt
KNOWN_PERSISTENT_CORRUPT = ["600039", "600188", "601225", "601919", "000408"]

TENCENT_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Referer": "https://gu.qq.com/",
}

START_DATE = "2014-01-01"
END_DATE = "2020-12-31"


def market_prefix(code: str) -> str:
    """6 位代码 → 'sh' / 'sz'."""
    if code.startswith("6") or code.startswith("9"):
        return "sh"
    return "sz"


def scan_corrupt_codes() -> list[str]:
    """扫描 baidu_kline 找出 corrupt 股 (close<0 或 2018 close<0.5)."""
    kl = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
    mask_2018 = (kl["date"].dt.year == 2018) & ((kl["close"] < 0) | (kl["close"] < 0.5))
    mask_19_20 = (kl["date"].dt.year.isin([2019, 2020])) & (kl["close"] < 0)
    found = set(kl.loc[mask_2018 | mask_19_20, "code"].unique())
    found |= set(KNOWN_PERSISTENT_CORRUPT)
    return sorted(found)


def _fetch_year_chunk(
    sym: str,
    year: int,
    retries: int = 3,
    retry_delay: float = 3.0,
) -> Optional[list[list[str]]]:
    """单股单年拉 hfq 日线. Tencent count 单次约 ~640 行上限, 按年分块更稳."""
    last_err: Optional[BaseException] = None
    param = f"{sym},day,{year}-01-01,{year}-12-31,300,hfq"
    for attempt in range(retries):
        try:
            r = requests.get(
                TENCENT_KLINE,
                params={"param": param},
                headers=HEADERS,
                timeout=20,
            )
            r.raise_for_status()
            payload = r.json()
            sec = (payload.get("data") or {}).get(sym) or {}
            rows = sec.get("hfqday") or sec.get("day") or []
            return rows  # empty list 也合法 (未上市年)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(retry_delay)
    print(f"  ! {sym} year {year} failed: {type(last_err).__name__}: {last_err}")
    return None


def fetch_one_hfq(code: str) -> Optional[pd.DataFrame]:
    """单股拉 2014-2020 日线 hfq (按年分块 + 合并).

    Tencent API 返回 [date, open, close, high, low, vol(手), ...].
    """
    sym = f"{market_prefix(code)}{code}"
    all_rows: list[list[str]] = []
    for year in range(2014, 2021):
        chunk = _fetch_year_chunk(sym, year)
        if chunk is None:
            return None  # network error after retries
        all_rows.extend(chunk)
        time.sleep(0.15)
    if not all_rows:
        return None
    # 腾讯在除权日会在行末追加一个 dividend metadata dict (7 列), 截前 6 列
    norm_rows = [r[:6] for r in all_rows]
    df = pd.DataFrame(
        norm_rows,
        columns=["date", "open", "close", "high", "low", "vol"],
    )
    df["code"] = code
    for c in ("open", "close", "high", "low"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # tencent vol 单位是手 → ×100 股, baidu_kline 是股
    df["vol"] = (pd.to_numeric(df["vol"], errors="coerce") * 100).astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    # dedup (年份边界可能重叠)
    df = df.drop_duplicates(subset=["code", "date"]).sort_values("date").reset_index(drop=True)
    # 腾讯不返回 amount/turnoverratio, 留 float NaN (与 baidu_kline schema 一致)
    df["amount"] = float("nan")
    df["turnoverratio"] = float("nan")
    # 重算 MA (基于 hfq close)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    return df[
        [
            "code", "date", "open", "close", "high", "low",
            "vol", "amount", "ma5", "ma10", "ma20", "turnoverratio",
        ]
    ]


def validate(df: Optional[pd.DataFrame], code: str) -> tuple[bool, str]:
    """质量 check. 返回 (pass, msg)."""
    if df is None or df.empty:
        return False, "empty"
    closes = df["close"].dropna()
    if closes.empty:
        return False, "all close NaN"
    if (closes < 0).any():
        return False, f"still has neg close: min={closes.min()}"
    if (closes < 0.5).any():
        bad_dates = df.loc[df["close"] < 0.5, "date"].dt.strftime("%Y-%m-%d").tolist()
        return False, f"close<0.5 still present at {bad_dates[:3]}"
    rets = closes.pct_change().abs().dropna()
    if (rets > 0.20).any():
        # 涨跌停 ±10% (创业板/科创板 ±20%), 留 buffer 20% 区分真异常
        max_ret = float(rets.max())
        idx = rets.idxmax()
        max_date = df.loc[idx, "date"] if idx in df.index else "?"
        return False, f"extreme jump {max_ret:.2%} at {max_date}"
    return True, "ok"


def main() -> None:
    t0 = time.time()
    print(f"[phase1a] scanning corrupt codes from {KLINE_PATH.name} ...")
    codes = scan_corrupt_codes()
    print(f"[phase1a] found {len(codes)} corrupt codes: {codes}")
    OUT_LIST.write_text("\n".join(codes) + "\n", encoding="utf-8")
    print(f"[phase1a] wrote list -> {OUT_LIST}")

    frames: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []
    for i, code in enumerate(codes, 1):
        elapsed = time.time() - t0
        print(f"[{i}/{len(codes)}] {code} (t+{elapsed:.1f}s) ...", end=" ", flush=True)
        df = fetch_one_hfq(code)
        ok, msg = validate(df, code)
        if not ok:
            failed.append((code, msg))
            print(f"FAIL ({msg})")
            continue
        assert df is not None
        frames.append(df)
        n = len(df)
        cmin = float(df["close"].min())
        cmax = float(df["close"].max())
        max_ret = float(df["close"].pct_change().abs().max())
        print(f"OK {n} rows close=[{cmin:.2f},{cmax:.2f}] max_ret={max_ret:.2%}")
        time.sleep(0.2)  # gentle

    if not frames:
        print("[phase1a] ABORT: no successful fetches")
        return

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["code", "date"]).reset_index(drop=True)
    # Schema sanity
    assert (all_df["close"] > 0).all(), "post-merge close has non-positive"
    assert (all_df["close"] >= 0.5).all(), "post-merge close < 0.5"

    all_df.to_parquet(OUT_PARQUET, index=False)
    size_mb = OUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"\n[phase1a] DONE in {time.time()-t0:.1f}s")
    print(f"[phase1a] wrote {OUT_PARQUET} ({size_mb:.2f} MB)")
    print(f"[phase1a] rows={len(all_df)}, codes ok={len(frames)}/{len(codes)}")
    if failed:
        print(f"[phase1a] FAILED codes ({len(failed)}):")
        for code, msg in failed:
            print(f"    {code}: {msg}")


if __name__ == "__main__":
    main()
