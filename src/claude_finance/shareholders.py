"""股东户数(shareholder count)数据 fetch + 因子工程.

数据源
------
akshare:
  - ak.stock_zh_a_gdhs_detail_em(symbol=code)
    单股全 history 季度/月度股东户数明细 (东财).
    返回字段 (中文):
        股东户数统计截止日, 区间涨跌幅, 股东户数-本次, 股东户数-上次,
        股东户数-增减, 股东户数-增减比例, 户均持股市值, 户均持股数量,
        总市值, 总股本, 股本变动, 股本变动原因, 股东户数公告日期, 代码, 名称

Cache schema (per-stock parquet + 合并全集 parquet)
-------------------------------------------------
    code               str   (6位补零)
    data_date          datetime64[ns]   股东户数统计截止日
    announce_date      datetime64[ns]   股东户数公告日期 (PIT key, 因子用)
    shareholders_count int    股东户数-本次
    count_change_pct   float  股东户数-增减比例 (%)
    avg_holding        float  户均持股数量 (股)
    total_mcap         float  总市值 (元)
    total_shares       int    总股本

关键 PIT 约束
-------------
T 月信号 = announce_date ≤ T 的 latest 季度变化.
绝对不能用 data_date 选样 — 那是 lookahead (季报披露 lag 30-45 天).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "shareholders"
PER_STOCK_DIR = CACHE_DIR / "per_stock"
MERGED_FILE = CACHE_DIR / "shareholders_csi300.parquet"

# 输出 schema 列顺序
COLS = [
    "code", "data_date", "announce_date",
    "shareholders_count", "count_change_pct", "avg_holding",
    "total_mcap", "total_shares",
]


def fetch_stock(
    code: str,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
) -> pd.DataFrame:
    """抓单股全 history 股东户数明细; 返回标准化 schema."""
    import akshare as ak  # 延迟导入

    code = str(code).zfill(6)
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = ak.stock_zh_a_gdhs_detail_em(symbol=code)
            return _normalize(raw, code)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(
        f"fetch_stock({code}) failed after {max_retries} tries: {last_err}"
    )


def _normalize(raw: pd.DataFrame | None, code: str) -> pd.DataFrame:
    """raw akshare → cache schema."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLS)

    df = raw.rename(columns={
        "股东户数统计截止日": "data_date",
        "股东户数公告日期": "announce_date",
        "股东户数-本次": "shareholders_count",
        "股东户数-增减比例": "count_change_pct",
        "户均持股数量": "avg_holding",
        "总市值": "total_mcap",
        "总股本": "total_shares",
    })
    keep_present = [
        "data_date", "announce_date", "shareholders_count",
        "count_change_pct", "avg_holding", "total_mcap", "total_shares",
    ]
    for c in keep_present:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[keep_present].copy()
    df["code"] = code
    df["data_date"] = pd.to_datetime(df["data_date"], errors="coerce")
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    for col in [
        "shareholders_count", "count_change_pct", "avg_holding",
        "total_mcap", "total_shares",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 必须有 announce_date (PIT key); data_date 也得有
    df = df.dropna(subset=["announce_date", "data_date"]).reset_index(drop=True)
    return df[COLS].reset_index(drop=True)


def save_per_stock(df: pd.DataFrame, code: str) -> None:
    """单股 parquet (atomic write)."""
    PER_STOCK_DIR.mkdir(parents=True, exist_ok=True)
    code = str(code).zfill(6)
    tgt = PER_STOCK_DIR / f"{code}.parquet"
    tmp = PER_STOCK_DIR / f"{code}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(tgt)


def load_per_stock(code: str) -> pd.DataFrame:
    code = str(code).zfill(6)
    p = PER_STOCK_DIR / f"{code}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_parquet(p)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["data_date"] = pd.to_datetime(df["data_date"])
    df["announce_date"] = pd.to_datetime(df["announce_date"])
    return df


def merge_all() -> pd.DataFrame:
    """合并 per-stock parquet 到全集 parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not PER_STOCK_DIR.exists():
        return pd.DataFrame(columns=COLS)
    parts: list[pd.DataFrame] = []
    for p in sorted(PER_STOCK_DIR.glob("*.parquet")):
        try:
            sub = pd.read_parquet(p)
            if not sub.empty:
                parts.append(sub)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN: skip {p.name}: {exc}")
    if not parts:
        return pd.DataFrame(columns=COLS)
    big = pd.concat(parts, ignore_index=True)
    big["code"] = big["code"].astype(str).str.zfill(6)
    big["data_date"] = pd.to_datetime(big["data_date"])
    big["announce_date"] = pd.to_datetime(big["announce_date"])
    # dedupe by (code, data_date) — 同一报告期可能有多次公告 (修正), 保 latest announce
    big = (
        big.sort_values(["code", "data_date", "announce_date"])
        .drop_duplicates(subset=["code", "data_date"], keep="last")
        .reset_index(drop=True)
    )
    big.to_parquet(MERGED_FILE, index=False)
    return big


def load_cache() -> pd.DataFrame:
    """读全集 parquet."""
    if not MERGED_FILE.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_parquet(MERGED_FILE)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["data_date"] = pd.to_datetime(df["data_date"])
    df["announce_date"] = pd.to_datetime(df["announce_date"])
    return df


# ---------- 因子工程 (PIT 严格) ----------


def build_factor_panel(
    df: pd.DataFrame,
    codes: list[str],
    asof_dates: list[pd.Timestamp],
    max_lookback_days: int = 365,
) -> pd.DataFrame:
    """构造 panel: (asof_date, code) → factor cols (PIT 严格).

    Factors per row (announce_date ≤ T):
      count_change_3m   : 季度环比变化 % (count_change_pct, akshare 直供)
      count_change_6m   : 近 ~180 日累积 (sum)
      count_change_12m  : 近 ~365 日累积 (sum)
      avg_holding_raw   : 户均持股数量 (中间量, 后续做横截面 concentration)
      lvl_concentration : 1 - avg_holding / cross-section median (per asof)
      announce_age_days : T - announce_date (新鲜度, 不做信号)
    """
    codes = [str(c).zfill(6) for c in codes]
    codes_set = set(codes)
    sub = df[df["code"].isin(codes_set)].copy()
    sub = sub.sort_values(["code", "announce_date"]).reset_index(drop=True)

    asof_dates = sorted({pd.Timestamp(d) for d in asof_dates})

    rows: list[dict] = []
    for asof in asof_dates:
        # 每个 asof_date 切一次 sub: announce_date ≤ asof
        pit = sub[sub["announce_date"] <= asof]
        if pit.empty:
            for code in codes:
                rows.append(_nan_row(asof, code))
            continue
        # 按 code 取 latest announce + 历史 (近 5 期, 6m/12m 用)
        per_code: dict[str, pd.DataFrame] = {}
        for c, sub_c in pit.groupby("code"):
            per_code[c] = sub_c.sort_values("announce_date").tail(6)
        for code in codes:
            sub_c = per_code.get(code)
            if sub_c is None or sub_c.empty:
                rows.append(_nan_row(asof, code))
                continue
            row_latest = sub_c.iloc[-1]
            age = (asof - row_latest["announce_date"]).days
            if age > max_lookback_days:
                rows.append(_nan_row(asof, code))
                continue
            change_3m = row_latest["count_change_pct"]
            change_6m = _cumulate_change(sub_c, asof, days=183)
            change_12m = _cumulate_change(sub_c, asof, days=366)
            rows.append({
                "asof_date": asof,
                "code": code,
                "count_change_3m": float(change_3m)
                    if pd.notna(change_3m) else float("nan"),
                "count_change_6m": change_6m,
                "count_change_12m": change_12m,
                "avg_holding_raw": float(row_latest["avg_holding"])
                    if pd.notna(row_latest["avg_holding"]) else float("nan"),
                "announce_age_days": int(age),
            })
    panel = pd.DataFrame(rows)
    # lvl_concentration: 1 - avg_holding / cross-section median (per asof_date)
    panel["lvl_concentration"] = panel.groupby("asof_date")[
        "avg_holding_raw"
    ].transform(_concentration)
    return panel


def _nan_row(asof: pd.Timestamp, code: str) -> dict:
    return {
        "asof_date": asof,
        "code": code,
        "count_change_3m": float("nan"),
        "count_change_6m": float("nan"),
        "count_change_12m": float("nan"),
        "avg_holding_raw": float("nan"),
        "announce_age_days": -1,
    }


def _cumulate_change(
    hist: pd.DataFrame,
    asof: pd.Timestamp,
    days: int,
) -> float:
    """近 N 日内累积 count_change_pct (sum, PIT).

    hist 必须已经 announce_date ≤ asof (上游过滤).
    """
    if hist.empty:
        return float("nan")
    cutoff = asof - pd.Timedelta(days=days)
    win = hist[hist["announce_date"] > cutoff]
    if win.empty:
        return float("nan")
    s = win["count_change_pct"].dropna()
    if s.empty:
        return float("nan")
    return float(s.sum())


def _concentration(s: pd.Series) -> pd.Series:
    """横截面 concentration: 1 - avg / median(avg).

    > 0 → 户均持股 < 截面 median → 户多筹码散 (低集中度)
    < 0 → 户均持股 > 截面 median → 户少筹码集中 (高集中度)
    """
    med = s.median()
    if pd.isna(med) or med == 0:
        return pd.Series(float("nan"), index=s.index)
    return 1.0 - s / med
