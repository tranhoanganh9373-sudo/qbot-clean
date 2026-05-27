"""train24 (v19.1 production) sub-period regime validation 2017-2020.

目的: 验证 train24 在跨 regime (2017-2020) 是否仍 robust.
策略: 严格不重训, 只读 cached predictions (v17_dens_train24_predictions.parquet).

复用 strategy_v17_dens_grid.py 的回测逻辑:
  - K=8, N_DROP=2
  - PORTFOLIO_VALUE=50000
  - max_affordable_price=125 (生产配置)
  - 涨停过滤
  - max_position_pct_of_vol=0.05

数据约束 (verified):
  cache 月份覆盖:
    2017-01~2017-12: 仅 2-17 个 instruments (CSI300 的 0.6%~5%) — 极度稀疏, 不可用
    2018-01:        51 instruments (16%) — 不完整, 跳过
    2018-02~2018-12: 236+ instruments (78%+) — 可用 (2018 熊市 11 月)
    2019-01~2020-12: 完全缺失 — 不可用 (需重训, 任务约束: 不重训)
    2021-05~2026-04: 278+ instruments (production OOS 范围)

  ∴ 此脚本实际可跑的 sub-period: 2018-02 ~ 2018-12 (11 个月, 单 regime: 全面熊市)
  非 cache 月份将标记 cache-missing.

Run:
  python examples/strategy_train24_subperiod_2017_2020.py
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from dateutil.relativedelta import relativedelta
from qlib.constant import REG_CN
from qlib.data import D

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = str(ROOT / "data_cache" / "qlib_baidu")
OUT_DIR = Path(__file__).resolve().parent
PRED_CACHE = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"

MARKET = "csi300"
PORTFOLIO_VALUE = 50000.0
K_NORMAL = 8
DROP_NORMAL = 2
MAX_PRICE = 125.0  # production constraint

IMPACT_COEF = 0.5
MAX_POSITION_PCT_OF_VOL = 0.05
PRICE_LIMIT_UP = 0.090
SIGNAL_DELAY_FACTOR = 0.5
MIN_LOT = 100

CANDIDATE_POOL_MULTIPLIER = 4
LIMIT_UP_THRESHOLD = 0.095
LIMIT_DOWN_THRESHOLD = -0.095
LIMIT_UP_THRESHOLD_HIGH = 0.195
LIMIT_DOWN_THRESHOLD_HIGH = -0.195

MIN_INSTRUMENTS_FOR_VALID_MONTH = 100  # 月 instruments < 100 视为 cache-missing


def is_limit_up(sym, chg):
    thresh = LIMIT_UP_THRESHOLD_HIGH if sym.startswith(("SH688", "SZ300")) else LIMIT_UP_THRESHOLD
    return chg >= thresh


def is_limit_down(sym, chg):
    thresh = LIMIT_DOWN_THRESHOLD_HIGH if sym.startswith(("SH688", "SZ300")) else LIMIT_DOWN_THRESHOLD
    return chg <= thresh


def month_start(d):
    return d.strftime("%Y-%m-01")


def month_end(d):
    nm = (d.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)
    return nm.strftime("%Y-%m-%d")


def get_price_data(start, end):
    df = D.features(
        D.instruments(market=MARKET),
        ["$open", "$close", "$volume"],
        start_time=start, end_time=end, freq="day",
    ).reset_index()
    df.columns = ["instrument", "date", "open", "close", "volume"]
    return df


_pred_disk_df: pd.DataFrame | None = None


def load_pred_for_month(month_key: str) -> pd.Series | None:
    """读 cached pred. 返回 None 如果缺失或样本数 < 阈值."""
    global _pred_disk_df
    if _pred_disk_df is None:
        _pred_disk_df = pd.read_parquet(PRED_CACHE)
    df = _pred_disk_df[_pred_disk_df["month"] == month_key]
    if len(df) == 0:
        return None
    n_inst = df["instrument"].nunique()
    if n_inst < MIN_INSTRUMENTS_FOR_VALID_MONTH:
        return None
    df = df.set_index(["datetime", "instrument"])
    return df["score"]


def realistic_window(test_month_start):
    test_start = month_start(test_month_start)
    test_end = month_end(test_month_start)
    key = test_month_start.strftime("%Y-%m")

    pred = load_pred_for_month(key)
    if pred is None:
        return {"abs_ret_%": np.nan, "avg_picks": 0, "n_days": 0,
                "regime_days": "cache-missing", "n_skipped_limit": 0,
                "n_instruments": 0, "status": "SKIP_NO_CACHE"}

    price_end = (pd.to_datetime(test_end) + timedelta(days=10)).strftime("%Y-%m-%d")
    price_df = get_price_data(test_start, price_end)
    if len(price_df) == 0:
        return {"abs_ret_%": np.nan, "avg_picks": 0, "n_days": 0,
                "regime_days": "no-price", "n_skipped_limit": 0,
                "n_instruments": 0, "status": "SKIP_NO_PRICE"}

    open_pv = price_df.pivot(index="date", columns="instrument", values="open")
    close_pv = price_df.pivot(index="date", columns="instrument", values="close")
    vol_pv = price_df.pivot(index="date", columns="instrument", values="volume")

    pred_unstacked = pred.unstack(level="instrument")
    test_dates = sorted(pred_unstacked.index)
    if len(test_dates) < 2:
        return {"abs_ret_%": np.nan, "avg_picks": 0, "n_days": 0,
                "regime_days": "few-days", "n_skipped_limit": 0,
                "n_instruments": pred_unstacked.shape[1], "status": "SKIP_FEW_DAYS"}

    current_holdings: dict = {}
    entry_price: dict = {}
    cash = PORTFOLIO_VALUE
    daily_ret = []
    n_picks_realized = []
    last_known_price: dict = {}
    n_skipped_limit = 0
    n_skipped_maxprice = 0

    def mark_to_market(td_):
        pv = cash
        for c, sh in current_holdings.items():
            p_use = None
            if c in close_pv.columns and td_ in close_pv.index:
                p_candidate = close_pv.loc[td_, c]
                if pd.notna(p_candidate) and p_candidate > 0:
                    if c in last_known_price and last_known_price[c] > 0:
                        chg = abs(p_candidate / last_known_price[c] - 1)
                        if chg <= 0.15:
                            p_use = p_candidate
                            last_known_price[c] = p_candidate
                        else:
                            p_use = last_known_price[c]
                    else:
                        p_use = p_candidate
                        last_known_price[c] = p_candidate
            if p_use is None:
                p_use = last_known_price.get(c, 0)
            pv += sh * p_use
        return pv

    for di, td in enumerate(test_dates):
        if di + 1 >= len(test_dates):
            break
        next_td = test_dates[di + 1]
        if next_td not in open_pv.index:
            continue

        topk, n_drop = K_NORMAL, DROP_NORMAL
        port_val = mark_to_market(td)

        scores_all = pred_unstacked.loc[td].dropna().sort_values(ascending=False)
        pool = scores_all.head(topk * CANDIDATE_POOL_MULTIPLIER)

        filtered = []
        for sym, score in pool.items():
            if sym not in close_pv.columns or di == 0:
                if td in close_pv.index and sym in close_pv.columns:
                    cur_p = close_pv.loc[td, sym]
                    if pd.notna(cur_p) and cur_p > MAX_PRICE:
                        n_skipped_maxprice += 1
                        continue
                filtered.append((sym, score))
                if len(filtered) >= topk:
                    break
                continue
            prev_td = test_dates[di - 1] if di > 0 else None
            curr_p = close_pv.loc[td, sym] if td in close_pv.index else None
            prev_p = close_pv.loc[prev_td, sym] if prev_td and prev_td in close_pv.index else None
            chg = (curr_p / prev_p - 1) if (pd.notna(curr_p) and pd.notna(prev_p)
                                            and prev_p > 0) else 0
            if is_limit_up(sym, chg) or is_limit_down(sym, chg):
                n_skipped_limit += 1
                continue
            if pd.notna(curr_p) and curr_p > MAX_PRICE:
                n_skipped_maxprice += 1
                continue
            filtered.append((sym, score))
            if len(filtered) >= topk:
                break

        target_topk = [s for s, _ in filtered]
        scores_for_drop = pred_unstacked.loc[td]
        to_drop_candidates = sorted(
            [c for c in current_holdings if c not in target_topk],
            key=lambda c: scores_for_drop.get(c, -np.inf),
        )
        to_drop = to_drop_candidates[:n_drop]
        excess = len(current_holdings) - topk
        if excess > 0:
            extra_drop = [c for c in current_holdings if c not in to_drop and c not in target_topk]
            to_drop.extend(extra_drop[:excess])
        buy_candidates = [c for c in target_topk if c not in current_holdings][:n_drop]

        for c in to_drop:
            shares = current_holdings[c]
            if c not in open_pv.columns or next_td not in open_pv.index:
                continue
            no_p = open_pv.loc[next_td, c]
            nc_p = close_pv.loc[next_td, c] if next_td in close_pv.index else no_p
            if pd.isna(no_p) or pd.isna(nc_p):
                continue
            exec_p = no_p * (1 - SIGNAL_DELAY_FACTOR) + nc_p * SIGNAL_DELAY_FACTOR
            if exec_p <= 0:
                continue
            order_amount = shares * exec_p
            daily_amount = (vol_pv.loc[next_td, c] * exec_p
                            if next_td in vol_pv.index and pd.notna(vol_pv.loc[next_td, c])
                            else order_amount * 100)
            impact = IMPACT_COEF * np.sqrt(min(1.0, order_amount / max(daily_amount, 1e3))) * 0.01
            cash += shares * exec_p * (1 - impact - 0.0005)
            del current_holdings[c]
            entry_price.pop(c, None)

        cash_per_pick = cash / max(len(buy_candidates), 1)
        for c in buy_candidates:
            if c not in open_pv.columns or next_td not in open_pv.index:
                continue
            prev_close = close_pv.loc[td, c] if td in close_pv.index else None
            next_open = open_pv.loc[next_td, c]
            if pd.notna(prev_close) and pd.notna(next_open):
                chg = next_open / prev_close - 1
                if chg >= PRICE_LIMIT_UP:
                    continue
            nc_p = close_pv.loc[next_td, c] if next_td in close_pv.index else next_open
            if pd.isna(next_open) or pd.isna(nc_p):
                continue
            exec_p = next_open * (1 - SIGNAL_DELAY_FACTOR) + nc_p * SIGNAL_DELAY_FACTOR
            if exec_p <= 0:
                continue
            daily_amount = (vol_pv.loc[next_td, c] * exec_p
                            if next_td in vol_pv.index and pd.notna(vol_pv.loc[next_td, c])
                            else 1e9)
            max_amount = daily_amount * MAX_POSITION_PCT_OF_VOL
            target_amount = min(cash_per_pick, max_amount)
            if target_amount < exec_p * MIN_LOT:
                continue
            shares = (target_amount // (exec_p * MIN_LOT)) * MIN_LOT
            order_amount = shares * exec_p
            impact = IMPACT_COEF * np.sqrt(min(1.0, order_amount / max(daily_amount, 1e3))) * 0.01
            cash -= shares * exec_p * (1 + impact + 0.0003)
            current_holdings[c] = shares
            entry_price[c] = exec_p
        n_picks_realized.append(len(current_holdings))

        new_port_val = mark_to_market(next_td)
        if port_val > 0:
            daily_ret.append(new_port_val / port_val - 1)

    if not daily_ret:
        return {"abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                "regime_days": "bear", "n_skipped_limit": n_skipped_limit,
                "n_instruments": pred_unstacked.shape[1], "status": "OK"}
    abs_ret = (1 + pd.Series(daily_ret)).prod() - 1
    return {
        "abs_ret_%": round(abs_ret * 100, 2),
        "avg_picks": round(np.mean(n_picks_realized), 1) if n_picks_realized else 0,
        "n_days": len(daily_ret),
        "regime_days": "bear",
        "n_skipped_limit": n_skipped_limit,
        "n_skipped_maxprice": n_skipped_maxprice,
        "n_instruments": pred_unstacked.shape[1],
        "status": "OK",
    }


def annualize_metrics(returns_pct):
    """returns_pct: pd.Series of monthly abs_ret_% (drop NaN before passing)."""
    returns = returns_pct.dropna()
    if len(returns) == 0:
        return {"cum_%": 0, "ann_%": 0, "sharpe": 0, "mdd_%": 0, "calmar": 0, "win_%": 0, "n": 0}
    n = len(returns)
    years = n / 12
    cum = (1 + returns / 100).prod() - 1
    ann_ret = (1 + cum) ** (1 / years) - 1 if years > 0 else 0
    mean = (returns / 100).mean()
    std = (returns / 100).std()
    sharpe = mean / std * np.sqrt(12) if std > 0 else 0
    cum_series = (1 + returns / 100).cumprod()
    peak = cum_series.cummax()
    mdd = ((cum_series - peak) / peak).min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else 0
    return {
        "cum_%": round(cum * 100, 2),
        "ann_%": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 2),
        "mdd_%": round(mdd * 100, 2),
        "calmar": round(calmar, 2),
        "win_%": round((returns > 0).sum() / len(returns) * 100, 2),
        "n": n,
    }


def main():
    qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
    print(f"[init] qlib OK; pred cache={PRED_CACHE.name}")
    print(f"[init] config: K={K_NORMAL} D={DROP_NORMAL} PV={PORTFOLIO_VALUE:.0f} "
          f"MAX_PRICE={MAX_PRICE} train24 (v19.1 production)")

    # 数据可用性预检
    pred = pd.read_parquet(PRED_CACHE)
    target_months = []
    cur = datetime(2017, 5, 1)
    end = datetime(2020, 12, 1)
    while cur <= end:
        target_months.append(cur)
        cur += relativedelta(months=1)
    print(f"\n[pre-check] target sub-period: 2017-05 ~ 2020-12 ({len(target_months)} months)")
    print(f"[pre-check] cache availability (min {MIN_INSTRUMENTS_FOR_VALID_MONTH} instruments to be valid):")

    valid_months = []
    missing_months = []
    for m in target_months:
        key = m.strftime("%Y-%m")
        sub = pred[pred["month"] == key]
        n_inst = sub["instrument"].nunique() if len(sub) > 0 else 0
        is_valid = n_inst >= MIN_INSTRUMENTS_FOR_VALID_MONTH
        if is_valid:
            valid_months.append(m)
        else:
            missing_months.append((key, n_inst))
    print(f"  valid (>= {MIN_INSTRUMENTS_FOR_VALID_MONTH} inst): {len(valid_months)} months")
    print(f"  missing/sparse: {len(missing_months)} months")
    if missing_months:
        sample = missing_months[:3] + (missing_months[-3:] if len(missing_months) > 6 else [])
        for k, n in sample:
            print(f"    {k}: n_inst={n}")
        if len(missing_months) > 6:
            print(f"    ... {len(missing_months) - 6} more missing")

    if not valid_months:
        print("\n[ABORT] no valid cache months in target sub-period — task cannot proceed.")
        return

    print(f"\n[run] backtesting {len(valid_months)} valid months from cache")
    all_rows = []
    for m in target_months:
        key = m.strftime("%Y-%m")
        if m in valid_months:
            try:
                res = realistic_window(m)
            except Exception as e:
                res = {"abs_ret_%": np.nan, "avg_picks": 0, "n_days": 0,
                       "regime_days": "error", "n_skipped_limit": 0,
                       "n_instruments": 0, "status": f"ERROR:{str(e)[:60]}"}
        else:
            res = {"abs_ret_%": np.nan, "avg_picks": 0, "n_days": 0,
                   "regime_days": "cache-missing", "n_skipped_limit": 0,
                   "n_instruments": next((n for k, n in missing_months if k == key), 0),
                   "status": "SKIP_NO_CACHE"}
        res["month"] = key
        res["config"] = f"K={K_NORMAL} D={DROP_NORMAL} train24"
        all_rows.append(res)
        if res["status"] == "OK":
            print(f"  {key}: abs_ret={res['abs_ret_%']:+6.2f}%  picks={res['avg_picks']:.1f}  "
                  f"days={res['n_days']}  n_inst={res['n_instruments']}", flush=True)
        else:
            print(f"  {key}: {res['status']:18s} n_inst={res['n_instruments']}", flush=True)

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "v17_dens_train24_subperiod_2017_2020_stats.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[output] {out_csv}")

    # 分期统计
    print("\n=== Sub-period regime breakdown ===")
    hdr = (f"{'Period':<26} {'Months':>7} {'Regime':<18} "
           f"{'Cum%':>8} {'Ann%':>7} {'Sharpe':>7} {'MDD%':>8} {'Calmar':>7} {'Win%':>7}")
    print(hdr)
    print("-" * len(hdr))

    def filter_period(df_, start_key, end_key, label, regime):
        period_df = df_[(df_["month"] >= start_key) & (df_["month"] <= end_key)]
        sub = period_df[period_df["status"] == "OK"]
        n_total = len(period_df)
        if len(sub) == 0:
            print(f"{label:<26} {f'0/{n_total}':>7} {regime:<18} (no valid cache)")
            return None
        n_valid = len(sub)
        mm = annualize_metrics(sub["abs_ret_%"])
        print(f"{label:<26} {f'{n_valid}/{n_total}':>7} {regime:<18} "
              f"{mm['cum_%']:>8.2f} {mm['ann_%']:>7.2f} {mm['sharpe']:>7.2f} "
              f"{mm['mdd_%']:>8.2f} {mm['calmar']:>7.2f} {mm['win_%']:>7.2f}")
        return mm

    p17 = filter_period(df, "2017-05", "2017-12", "2017 H2 (value-bluechip)", "价值蓝筹")
    p18 = filter_period(df, "2018-01", "2018-12", "2018 (bear)", "全面熊市")
    p19 = filter_period(df, "2019-01", "2019-12", "2019 (tech rebound)", "科技反弹")
    p20 = filter_period(df, "2020-01", "2020-12", "2020 (COVID+V)", "COVID+V反转")
    p_all = filter_period(df, "2017-05", "2020-12", "sub-period total", "mixed")

    print()
    print("=== Reference: OOS production (from CLAUDE.md / prior backtest) ===")
    print(f"{'2021-05~2026-04 (60m)':<26} {'60/60':>7} {'known':<18} "
          f"{694.90:>8.2f} {51.40:>7.2f} {1.09:>7.2f} {-26.30:>8.2f} {1.96:>7.2f} {61.70:>7.2f}")

    # 判定
    print("\n=== Verdict ===")
    valid_yearly = [p for p in (p17, p18, p19, p20) if p is not None]
    n_yearly_valid = len(valid_yearly)
    n_yearly_target = 4
    all_periods_positive = all(p["ann_%"] > 0 for p in valid_yearly) if valid_yearly else False
    has_big_drop = any(p["cum_%"] < -40 for p in valid_yearly) if valid_yearly else False

    if n_yearly_valid < n_yearly_target:
        print(f"  [INCONCLUSIVE — INSUFFICIENT CACHE]")
        print(f"  Only {n_yearly_valid}/{n_yearly_target} yearly regimes have cache coverage.")
        print(f"  Cannot make robust/not-robust determination on cross-regime basis.")
        print(f"  Available regime tests above are real, but only represent partial regime spectrum.")
    elif all_periods_positive and p_all and p_all["calmar"] > 1.0 and not has_big_drop:
        print("  [A. ROBUST] every yearly regime positive + sub-period Calmar > 1.0 + no crash")
    elif p_all and p_all["ann_%"] > 0 and p_all["calmar"] > 0.5 and not has_big_drop:
        print("  [B. MARGINAL] sub-period positive + Calmar > 0.5 but some regime-sensitive")
    elif has_big_drop or (p_all and p_all["ann_%"] < 0):
        print("  [C. FAIL] multi-year loss or crash > 40% — train24 is 2021-2026 regime-specific")
    else:
        print("  [B. MARGINAL] unclear")


if __name__ == "__main__":
    main()
