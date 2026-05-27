"""Phase A — CSI300 ANALYST EARNINGS REVISION 因子 IC 分析 (IS 2014-2020).

Source: akshare `stock_research_report_em(symbol=code)` 返回每股研报全历史.
schema per row: 股票代码, 日期, 东财评级, 机构, 报告名称, 2026/27/28-盈利预测-收益, ...

注意 (重要数据限制):
  - EPS forecast columns are FORWARD-LOOKING ONLY (current 3y rolling frame).
    2018 历史报告中 2018/19/20-EPS 列已经全为 NaN — API 只 serve 当前 3 年预测.
    → 无法严格 PIT 计算 EPS revision pct (the task's primary signal).
  - 评级 (rating: 买入/增持/中性/持有/减持/卖出) is timestamped at report date → PIT-safe.
  - `lastEmRatingName` in EM API equals `emRatingName` in practice (sample 437/437
    reports = reaffirm) — 不能用作 per-report upgrade/downgrade Δ.
    Workaround: 用 window-vs-window mean rating 差 (rating_chg_30d/90d).
  - 数据最早可回溯 2017-01 (top liquid CSI300); 2014-2016 全为空.
  - baidu_kline CSI300 codes pre-2018 仅 10-25 → effective IC 窗口 ~2018-01 → 2020-12
    (n_months ≈ 35, 跟 phase_a_volume_zscore / phase_a_industry_adj_ret 一致约束).

Workable signal proxies (rating-based 'earnings revision'):
  signed_rating_t = +2 买入 / +1 增持 / 0 中性|持有 / -1 减持 / -2 卖出
  - revision_30d_net:  Σ rating ∈ (T-30,T] / nReports — 平均评级 (recent month)
  - revision_30d_count: nReports (T-30,T] — analyst attention
  - revision_30d_up_pct: #买入+增持 / nReports — 看多比例
  - revision_30d_chg: avg_rating(T-30,T] - avg_rating(T-60,T-30] — 评级变化
  - revision_90d_net, revision_90d_chg — 季度 window

Strict no OOS peek — asof_date 严格 ≤ 2020-12-31.

Output: examples/v21_earnings_revision_phase_a_ic.csv
        data_cache/earnings_revision/research_report_em.parquet
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data_cache" / "earnings_revision"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PER_STOCK_DIR = CACHE_DIR / "per_stock"
PER_STOCK_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_PARQUET = CACHE_DIR / "research_report_em.parquet"
FETCH_LOG = CACHE_DIR / "fetch_log.csv"

CSI300_PATH = ROOT / "data_cache" / "csi300_constituents.csv"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"

OUT_GRID = ROOT / "examples" / "v21_earnings_revision_phase_a_ic.csv"
OUT_MONTHLY = ROOT / "examples" / "v21_earnings_revision_phase_a_monthly.csv"
OUT_SPEAR = ROOT / "examples" / "v21_earnings_revision_phase_a_spearman.csv"

IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
FORWARD_DAYS = 20  # forward 1-month return

RATING_MAP = {
    "买入": 2, "强烈推荐": 2, "强推": 2,
    "增持": 1, "推荐": 1, "谨慎推荐": 1, "审慎推荐": 1, "优于大市": 1,
    "持有": 0, "中性": 0, "同步大市": 0, "区间操作": 0,
    "减持": -1, "弱于大市": -1, "回避": -1,
    "卖出": -2,
}


def load_csi300_codes() -> list[str]:
    csi = pd.read_csv(CSI300_PATH, dtype={"code": str})
    csi["code"] = csi["code"].astype(str).str.zfill(6)
    return csi["code"].tolist()


# ------------------------------ FETCH ------------------------------


EM_API_URL = "https://reportapi.eastmoney.com/report/list"
EM_BEGIN = "2014-01-01"
EM_END = "2020-12-31"


def _fetch_one_em(code: str, max_retries: int = 3) -> tuple[pd.DataFrame, str]:
    """Direct EM API fetch (paginated, IS window only). Returns (df, status)."""
    import requests

    rows: list[dict] = []
    page = 1
    total_page = 1
    while page <= total_page:
        params = {
            "industryCode": "*", "pageSize": "100",
            "industry": "*", "rating": "*", "ratingChange": "*",
            "beginTime": EM_BEGIN, "endTime": EM_END,
            "pageNo": str(page), "fields": "", "qType": "0",
            "orgCode": "", "code": code, "rcode": "",
            "p": str(page), "pageNum": str(page), "pageNumber": str(page),
        }
        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.get(EM_API_URL, params=params, timeout=15)
                data = r.json()
                if not isinstance(data, dict) or "data" not in data:
                    last_err = f"bad_json_page{page}"
                    time.sleep(0.5 * (attempt + 1))
                    continue
                total_page = int(data.get("TotalPage", 1) or 1)
                rows.extend(data["data"])
                last_err = None
                break
            except Exception as e:
                last_err = str(e)[:120]
                time.sleep(0.5 * (attempt + 1))
        if last_err is not None:
            return pd.DataFrame(), f"fail:{last_err}"
        page += 1
        if page > 20:  # safety cap
            break
    if not rows:
        return pd.DataFrame(columns=[
            "code", "date", "rating", "institution", "title",
        ]), "empty"
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "stockCode": "code",
        "publishDate": "date",
        "emRatingName": "rating",
        "lastEmRatingName": "rating_prev",
        "ratingChange": "rating_change_code",
        "orgSName": "institution",
        "title": "title",
    })
    keep = ["code", "date", "rating", "rating_prev",
            "rating_change_code", "institution", "title"]
    for k in keep:
        if k not in df.columns:
            df[k] = ""
    df = df[keep].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    for c in ["rating", "rating_prev", "institution", "title"]:
        df[c] = df[c].astype(str).fillna("")
    # rating_change_code may be int (e.g. 2) or empty string; coerce to nullable Int
    df["rating_change_code"] = pd.to_numeric(
        df["rating_change_code"], errors="coerce",
    ).astype("Int64")
    return df, "ok"


def fetch_all(codes: list[str], n_workers: int = 6) -> None:
    """Threaded direct EM API fetch with incremental parquet cache."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    done = {p.stem for p in PER_STOCK_DIR.glob("*.parquet")}
    log_rows: list[dict] = []
    if FETCH_LOG.exists():
        log_rows = pd.read_csv(FETCH_LOG).to_dict("records")

    todo = [c for c in codes if c not in done]
    print(f"[fetch] total={len(codes)} cached={len(done)} todo={len(todo)} "
          f"workers={n_workers}", flush=True)
    if not todo:
        return
    t0 = time.time()
    lock = threading.Lock()
    counter = {"n": 0}

    def _worker(code: str) -> tuple[str, int, str]:
        df, status = _fetch_one_em(code)
        try:
            df.to_parquet(PER_STOCK_DIR / f"{code}.parquet")
        except Exception as e:
            return code, 0, f"write_fail:{str(e)[:80]}"
        with lock:
            counter["n"] += 1
            n = counter["n"]
            if n % 10 == 0 or n == len(todo):
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed > 0 else 0
                eta = (len(todo) - n) / rate if rate > 0 else 0
                print(f"[fetch] {n}/{len(todo)} "
                      f"rate={rate:.2f}/s eta={eta/60:.1f}m", flush=True)
        return code, len(df), status

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_worker, c): c for c in todo}
        for fut in as_completed(futures):
            code, n_rows, status = fut.result()
            log_rows.append({
                "code": code, "n_rows": n_rows,
                "status": status, "err": "",
            })

    pd.DataFrame(log_rows).to_csv(FETCH_LOG, index=False)
    print(f"[fetch] done in {(time.time()-t0)/60:.1f}m", flush=True)


def combine_per_stock() -> pd.DataFrame:
    """Combine per-stock parquets into a single cache."""
    parquets = list(PER_STOCK_DIR.glob("*.parquet"))
    if not parquets:
        return pd.DataFrame(columns=[
            "code", "date", "rating", "institution", "title",
        ])
    newest_per_stock = max(p.stat().st_mtime for p in parquets)
    if (COMBINED_PARQUET.exists()
            and COMBINED_PARQUET.stat().st_mtime > newest_per_stock):
        print(f"[combine] using existing {COMBINED_PARQUET}", flush=True)
        return pd.read_parquet(COMBINED_PARQUET)
    dfs = []
    for p in parquets:
        try:
            d = pd.read_parquet(p)
            if d.empty:
                continue
            # Drop rating_change_code (mixed dtypes across cache vintages) — unused.
            if "rating_change_code" in d.columns:
                d = d.drop(columns=["rating_change_code"])
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=[
            "code", "date", "rating", "institution", "title",
        ])
    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["rating"] = out["rating"].astype(str).fillna("")
    if "rating_prev" in out.columns:
        out["rating_prev"] = out["rating_prev"].astype(str).fillna("")
        out["rating_prev_score"] = out["rating_prev"].map(RATING_MAP)
    out["rating_score"] = out["rating"].map(RATING_MAP)
    out.to_parquet(COMBINED_PARQUET, index=False)
    print(f"[combine] {len(out):,} reports × {out['code'].nunique()} codes × "
          f"{out['date'].min().date()} → {out['date'].max().date()}",
          flush=True)
    return out


# ------------------------------ PANEL ------------------------------


def build_panel(reports: pd.DataFrame,
                codes: list[str]) -> pd.DataFrame:
    """Build PIT panel of factors + fwd returns at month-first trading day."""
    kline = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close"])
    kline["code"] = kline["code"].astype(str).str.zfill(6)
    kline = kline[kline["code"].isin(codes)]
    kline = kline[(kline["date"] >= IS_START) &
                  (kline["date"] <= IS_END + pd.Timedelta(days=45))]
    kline = kline.sort_values(["code", "date"]).reset_index(drop=True)

    kline_dates = pd.DatetimeIndex(sorted(kline["date"].unique()))
    is_dates = kline_dates[(kline_dates >= IS_START) & (kline_dates <= IS_END)]
    months = pd.Series(is_dates).dt.to_period("M")
    month_first = (
        pd.Series(is_dates).groupby(months).first().reset_index(drop=True)
    )
    print(f"[panel] {len(month_first)} 月度采样点 "
          f"({month_first.iloc[0].date()} → {month_first.iloc[-1].date()})",
          flush=True)

    # Restrict reports to IS window for safety (PIT)
    r = reports[reports["date"] <= IS_END].copy()
    r["rating_score"] = r["rating"].map(RATING_MAP)
    if "rating_prev" in r.columns:
        r["rating_prev_score"] = r["rating_prev"].map(RATING_MAP)
    else:
        r["rating_prev_score"] = np.nan
    # Per-report rating revision Δ (only when both ratings present)
    r["rating_delta"] = r["rating_score"] - r["rating_prev_score"]
    r["is_upgrade"] = (r["rating_delta"] > 0).astype(int)
    r["is_downgrade"] = (r["rating_delta"] < 0).astype(int)
    r["is_reaffirm"] = ((r["rating_delta"] == 0)
                        & r["rating_score"].notna()).astype(int)

    r = r.sort_values(["code", "date"]).reset_index(drop=True)
    r_by_code: dict[str, pd.DataFrame] = {
        c: g.reset_index(drop=True) for c, g in r.groupby("code", sort=False)
    }

    wide = kline.pivot_table(
        index="date", columns="code", values="close", aggfunc="first",
    ).sort_index()

    rows: list[dict] = []
    for T in month_first:
        idx = wide.index.get_indexer([T])[0]
        if idx < 0 or idx + FORWARD_DAYS >= len(wide.index):
            continue
        c_now = wide.iloc[idx]
        c_fut = wide.iloc[idx + FORWARD_DAYS]
        for code in codes:
            cn, cf = c_now.get(code), c_fut.get(code)
            if pd.isna(cn) or pd.isna(cf) or cn <= 0:
                continue
            fwd = cf / cn - 1.0
            rep = r_by_code.get(code)
            f = _compute_factors(rep, T)
            f["asof_date"] = T
            f["code"] = code
            f["fwd_ret"] = fwd
            rows.append(f)
    panel = pd.DataFrame(rows)
    print(f"[panel] {len(panel)} rows × {panel['code'].nunique()} codes × "
          f"{panel['asof_date'].nunique()} months", flush=True)
    return panel


def _compute_factors(rep: pd.DataFrame | None,
                     T: pd.Timestamp) -> dict:
    """Compute rating-based revision factors at asof T (strict T-exclusive)."""
    out = {
        "n_reports_30d": 0,
        "n_reports_90d": 0,
        "avg_rating_30d": np.nan,
        "avg_rating_90d": np.nan,
        "rating_chg_30d": np.nan,
        "rating_chg_90d": np.nan,
        "up_pct_30d": np.nan,
        "up_pct_90d": np.nan,
        "net_up_30d": 0,
        "net_up_90d": 0,
        # Per-report revisions (true Δ from emRating vs lastEmRating)
        "n_upgrade_30d": 0,
        "n_downgrade_30d": 0,
        "net_revision_30d": 0,   # upgrade - downgrade
        "n_upgrade_90d": 0,
        "n_downgrade_90d": 0,
        "net_revision_90d": 0,
        "mean_delta_30d": np.nan,
        "mean_delta_90d": np.nan,
    }
    if rep is None or rep.empty:
        return out
    mask_all = rep["date"] < T
    if not mask_all.any():
        return out
    sub = rep[mask_all]
    d30 = T - pd.Timedelta(days=30)
    d60 = T - pd.Timedelta(days=60)
    d90 = T - pd.Timedelta(days=90)
    d180 = T - pd.Timedelta(days=180)

    w30 = sub[sub["date"] >= d30]
    w30_60 = sub[(sub["date"] >= d60) & (sub["date"] < d30)]
    w90 = sub[sub["date"] >= d90]
    w90_180 = sub[(sub["date"] >= d180) & (sub["date"] < d90)]

    out["n_reports_30d"] = int(len(w30))
    out["n_reports_90d"] = int(len(w90))
    rated_30 = w30.dropna(subset=["rating_score"])
    rated_90 = w90.dropna(subset=["rating_score"])
    rated_30_60 = w30_60.dropna(subset=["rating_score"])
    rated_90_180 = w90_180.dropna(subset=["rating_score"])

    if len(rated_30) > 0:
        out["avg_rating_30d"] = float(rated_30["rating_score"].mean())
        out["up_pct_30d"] = float((rated_30["rating_score"] >= 1).mean())
        out["net_up_30d"] = int((rated_30["rating_score"] >= 1).sum()
                                - (rated_30["rating_score"] <= -1).sum())
    if len(rated_90) > 0:
        out["avg_rating_90d"] = float(rated_90["rating_score"].mean())
        out["up_pct_90d"] = float((rated_90["rating_score"] >= 1).mean())
        out["net_up_90d"] = int((rated_90["rating_score"] >= 1).sum()
                                - (rated_90["rating_score"] <= -1).sum())
    if len(rated_30) > 0 and len(rated_30_60) > 0:
        out["rating_chg_30d"] = (rated_30["rating_score"].mean()
                                 - rated_30_60["rating_score"].mean())
    if len(rated_90) > 0 and len(rated_90_180) > 0:
        out["rating_chg_90d"] = (rated_90["rating_score"].mean()
                                 - rated_90_180["rating_score"].mean())
    # Per-report revisions (true Δ)
    if "is_upgrade" in w30.columns:
        out["n_upgrade_30d"] = int(w30["is_upgrade"].sum())
        out["n_downgrade_30d"] = int(w30["is_downgrade"].sum())
        out["net_revision_30d"] = (out["n_upgrade_30d"]
                                   - out["n_downgrade_30d"])
        out["n_upgrade_90d"] = int(w90["is_upgrade"].sum())
        out["n_downgrade_90d"] = int(w90["is_downgrade"].sum())
        out["net_revision_90d"] = (out["n_upgrade_90d"]
                                   - out["n_downgrade_90d"])
        d30_rated = w30.dropna(subset=["rating_delta"])
        d90_rated = w90.dropna(subset=["rating_delta"])
        if len(d30_rated) > 0:
            out["mean_delta_30d"] = float(d30_rated["rating_delta"].mean())
        if len(d90_rated) > 0:
            out["mean_delta_90d"] = float(d90_rated["rating_delta"].mean())
    return out


# ------------------------------ IC ------------------------------


def monthly_ic(panel: pd.DataFrame, factor_col: str, sign: int = 1) -> dict:
    df = panel.dropna(subset=[factor_col, "fwd_ret"]).copy()
    df["signed"] = sign * df[factor_col]

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < 10 or g["signed"].nunique() < 2:
            return np.nan
        return g["signed"].corr(g["fwd_ret"], method="spearman")

    monthly = df.groupby("asof_date").apply(_corr).dropna()
    if monthly.empty:
        return {"factor": factor_col, "sign": sign, "n_months": 0,
                "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "ic_pos_pct": np.nan, "monthly": monthly}
    m = float(monthly.mean())
    s = float(monthly.std())
    icir = m / s * np.sqrt(12) if s > 0 else 0.0
    return {
        "factor": factor_col, "sign": sign,
        "n_months": int(len(monthly)),
        "ic_mean": m, "ic_std": s, "icir": icir,
        "ic_pos_pct": float((monthly > 0).mean() * 100),
        "monthly": monthly,
    }


def _zscore(g: pd.Series) -> pd.Series:
    if g.std(ddof=0) <= 0 or g.isna().all():
        return pd.Series(0.0, index=g.index)
    return (g - g.mean()) / g.std(ddof=0)


def compute_existing_factors_panel(panel: pd.DataFrame,
                                   codes: list[str]) -> pd.DataFrame:
    """Compute amp_imb_20d (v19.6 prod) + JZF proxy (vol-z-60d) for Spearman.

    JZF (净主力买入/'巨资'-flow) historic proxy: volume z-score 60d.
    True production v19.10 JZF uses 资金流 daily 数据 (sina money flow),
    but for Phase A correlation diagnostics, vol-z-60d is a reasonable proxy
    when sina money-flow cache may not cover full IS window.
    """
    kline = pd.read_parquet(
        KLINE_PATH,
        columns=["code", "date", "open", "high", "low", "close", "vol"],
    )
    kline = kline.rename(columns={"vol": "volume"})
    kline["code"] = kline["code"].astype(str).str.zfill(6)
    kline = kline[kline["code"].isin(codes)]
    kline = kline[(kline["date"] >= IS_START - pd.Timedelta(days=200)) &
                  (kline["date"] <= IS_END)]
    kline = kline.sort_values(["code", "date"]).reset_index(drop=True)

    grp = kline.groupby("code", sort=False)
    kline["prev_close"] = grp["close"].shift(1)
    kline["amp"] = (kline["high"] - kline["low"]) / kline["prev_close"]
    body_sign = np.sign(kline["close"] - kline["open"])
    kline["signed_amp"] = body_sign * kline["amp"]
    kline["amp_imb_20d"] = (
        kline.groupby("code", sort=False)["signed_amp"]
        .transform(lambda s: s.rolling(20, min_periods=10).mean())
    )
    kline["vol_mean60"] = (
        kline.groupby("code", sort=False)["volume"]
        .transform(lambda s: s.rolling(60, min_periods=20).mean())
    )
    kline["vol_std60"] = (
        kline.groupby("code", sort=False)["volume"]
        .transform(lambda s: s.rolling(60, min_periods=20).std())
    )
    kline["jzf_proxy"] = (
        (kline["volume"] - kline["vol_mean60"]) / kline["vol_std60"]
    )

    anchors = panel[["asof_date", "code"]].drop_duplicates()
    feat = (kline[["date", "code", "amp_imb_20d", "jzf_proxy"]]
            .rename(columns={"date": "asof_date"})
            .sort_values("asof_date"))
    anchors_s = anchors.sort_values("asof_date")
    out_feat = pd.merge_asof(
        anchors_s, feat,
        on="asof_date", by="code",
        direction="backward", allow_exact_matches=True,
    )
    return panel.merge(out_feat, on=["asof_date", "code"], how="left")


# ------------------------------ MAIN ------------------------------


def main(argv: list[str]) -> int:
    fetch_only = "--fetch-only" in argv
    skip_fetch = "--skip-fetch" in argv

    codes = load_csi300_codes()
    print(f"[main] CSI300 universe = {len(codes)}", flush=True)

    if not skip_fetch:
        fetch_all(codes, n_workers=6)
    if fetch_only:
        return 0

    reports = combine_per_stock()
    if reports.empty:
        print("FATAL: 0 reports cached — fetch failed entirely.",
              file=sys.stderr)
        return 1
    print(f"[reports] {len(reports):,} × {reports['code'].nunique()} codes; "
          f"earliest {reports['date'].min().date()}; "
          f"rating_score non-null = "
          f"{reports['rating_score'].notna().sum():,}", flush=True)

    panel = build_panel(reports, codes)
    if panel.empty:
        print("FATAL: panel empty", file=sys.stderr)
        return 1

    panel = compute_existing_factors_panel(panel, codes)

    z_cols = [
        "avg_rating_30d", "avg_rating_90d", "up_pct_30d", "up_pct_90d",
        "rating_chg_30d", "rating_chg_90d",
        "n_reports_30d", "n_reports_90d",
        "net_up_30d", "net_up_90d",
        "net_revision_30d", "net_revision_90d",
        "n_upgrade_30d", "n_downgrade_30d",
        "n_upgrade_90d", "n_downgrade_90d",
        "mean_delta_30d", "mean_delta_90d",
    ]
    for col in z_cols:
        if col in panel.columns:
            panel[f"z_{col}"] = (
                panel.groupby("asof_date")[col].transform(_zscore)
            )
    panel["z_combo_rating"] = (panel["z_avg_rating_30d"]
                               + panel["z_avg_rating_90d"]) / 2
    panel["z_combo_chg"] = (panel["z_rating_chg_30d"]
                            + panel["z_rating_chg_90d"]) / 2
    panel["z_combo_attention"] = (panel["z_n_reports_30d"]
                                  + panel["z_n_reports_90d"]) / 2
    panel["z_combo_revision"] = (panel["z_net_revision_30d"]
                                 + panel["z_net_revision_90d"]) / 2

    factors = [
        # Rating LEVEL (T-window avg)
        ("avg_rating_30d", +1), ("avg_rating_30d", -1),
        ("avg_rating_90d", +1), ("avg_rating_90d", -1),
        ("up_pct_30d", +1), ("up_pct_90d", +1),
        # Rating CHANGE (window-vs-window deltas)
        ("rating_chg_30d", +1), ("rating_chg_30d", -1),
        ("rating_chg_90d", +1), ("rating_chg_90d", -1),
        # Analyst ATTENTION
        ("n_reports_30d", +1), ("n_reports_30d", -1),
        ("n_reports_90d", +1), ("n_reports_90d", -1),
        # Net buy signal
        ("net_up_30d", +1), ("net_up_90d", +1),
        # Per-report TRUE REVISIONS (emRating vs lastEmRating)
        ("n_upgrade_30d", +1), ("n_upgrade_30d", -1),
        ("n_downgrade_30d", +1), ("n_downgrade_30d", -1),
        ("net_revision_30d", +1), ("net_revision_30d", -1),
        ("n_upgrade_90d", +1), ("n_downgrade_90d", +1),
        ("net_revision_90d", +1), ("net_revision_90d", -1),
        ("mean_delta_30d", +1), ("mean_delta_30d", -1),
        ("mean_delta_90d", +1), ("mean_delta_90d", -1),
        # Combos
        ("z_combo_rating", +1), ("z_combo_rating", -1),
        ("z_combo_chg", +1), ("z_combo_chg", -1),
        ("z_combo_attention", +1), ("z_combo_attention", -1),
        ("z_combo_revision", +1), ("z_combo_revision", -1),
    ]

    rows = []
    monthly_dict: dict[str, pd.Series] = {}
    for f, sgn in factors:
        if f not in panel.columns:
            continue
        res = monthly_ic(panel, f, sign=sgn)
        rows.append({k: v for k, v in res.items() if k != "monthly"})
        monthly_dict[f"{f}__sign{sgn:+d}"] = res["monthly"]

    grid = pd.DataFrame(rows).round(4)
    grid = grid.reindex(
        grid["icir"].abs().sort_values(
            ascending=False, na_position="last",
        ).index
    )
    grid.to_csv(OUT_GRID, index=False)
    print(f"\n[output] grid → {OUT_GRID}")
    print(grid.to_string(index=False))

    monthly_rows: list[dict] = []
    for key, ser in monthly_dict.items():
        for m, v in ser.items():
            monthly_rows.append({"factor_sign": key, "asof_date": m, "ic": v})
    pd.DataFrame(monthly_rows).to_csv(OUT_MONTHLY, index=False)
    print(f"[output] monthly → {OUT_MONTHLY}")

    best = grid.iloc[0]
    best_factor = best["factor"]
    best_sign = int(best["sign"])
    panel = panel.copy()
    panel["signed_best"] = best_sign * panel[best_factor]
    spear_rows = []
    for ref in ["amp_imb_20d", "jzf_proxy"]:
        df = panel.dropna(subset=["signed_best", ref]).copy()
        if df.empty:
            spear_rows.append({"ref": ref, "n_months": 0,
                               "mean_abs_rho": np.nan})
            continue
        mrho = df.groupby("asof_date").apply(
            lambda g: g["signed_best"].corr(g[ref], method="spearman")
            if g["signed_best"].nunique() > 1 and g[ref].nunique() > 1
            else np.nan
        ).dropna()
        spear_rows.append({
            "ref": ref,
            "n_months": int(len(mrho)),
            "mean_rho": float(mrho.mean()) if len(mrho) else np.nan,
            "mean_abs_rho": float(mrho.abs().mean()) if len(mrho) else np.nan,
        })
    spear = pd.DataFrame(spear_rows).round(4)
    spear.to_csv(OUT_SPEAR, index=False)
    print(f"\n[spearman vs production refs] → {OUT_SPEAR}")
    print(spear.to_string(index=False))

    best_icir = abs(best["icir"]) if pd.notna(best["icir"]) else 0.0
    best_n = int(best["n_months"]) if pd.notna(best["n_months"]) else 0
    # 项目规约: n_months < 72 → Phase B OOS 风险极高 (历史 4× REJECT 在 n<60).
    sample_thin = best_n < 72
    if best_icir > 0.5 and not sample_thin:
        verdict = ("STRONG — 推荐进 Phase B sidecar OOS "
                   f"(lock factor={best_factor} sign={best_sign:+d} "
                   "λ∈{0.10,0.20,0.30})")
    elif best_icir > 0.5 and sample_thin:
        verdict = (
            f"CAUTIOUS — IS ICIR={best_icir:.2f} 强, 但 n_months={best_n} "
            f"< 72 month gate. 历史 thin-sample Phase B 4× REJECT "
            f"(v19.7/v19.9/super_big_net/shareholders). 建议 ABORT 直到 "
            f"研报数据补足 ≥72 月 (跨 2 个完整 regime)."
        )
    elif best_icir > 0.3:
        verdict = "MARGINAL — 边际, 看 Spearman 是否独立"
    else:
        verdict = "WEAK — ABORT Phase A, 不进 Phase B"
    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"Best: {best_factor} sign={best_sign:+d} | "
          f"ICIR={best['icir']:.3f} | "
          f"IC mean={best['ic_mean']:.4f} | "
          f"n_months={best['n_months']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
