"""参数化 v17 grid sweep — CLI 跑任意 (K, N_DROP) 组合, 复用已有 LGB 预测.

设计:
  优先读 data_cache/v17_predictions.parquet (12 月 LGB pred 已存),
  缺月时才训新月. 每个 (K, D) config 跑时间 ~5-20s, 比从头训快 10-20x.

CLI:
  --k        K_NORMAL (持仓上限)
  --drop     DROP_NORMAL (每日换手数)
  --tag      artifact / output 文件名后缀
  --months   walk-forward 月数 (默认 12, last-N)

输出:
  examples/v17_<tag>_stats.csv
  控制台 summary

Run examples:
  python examples/strategy_v17_grid.py --k 3 --drop 2 --tag k3d2 --months 12
  python examples/strategy_v17_grid.py --k 8 --drop 4 --tag k8d4 --months 12
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from dateutil.relativedelta import relativedelta
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.data import D
from qlib.data.dataset import DatasetH

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = str(ROOT / "data_cache" / "qlib_baidu")
INDEX_PARQUET = ROOT / "data_cache" / "index_kline.parquet"
INDEX_CODE = "sh000300"
OUT_DIR = Path(__file__).resolve().parent
PRED_CACHE = ROOT / "data_cache" / "v17_predictions.parquet"

MARKET = "csi300"
TRAIN_MONTHS = 12
PORTFOLIO_VALUE = 5e4

# CLI 覆盖这两个 (默认 v17 值)
K_NORMAL = 8
DROP_NORMAL = 2
K_HALF = 4
DROP_HALF = 1
BEAR_MA_RATIO = 0.95
PANIC_VOL_ANN = 0.35
DRAWDOWN_60D = -0.15

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


def is_limit_up(sym, chg):
    thresh = LIMIT_UP_THRESHOLD_HIGH if sym.startswith(("SH688", "SZ300")) else LIMIT_UP_THRESHOLD
    return chg >= thresh


def is_limit_down(sym, chg):
    thresh = LIMIT_DOWN_THRESHOLD_HIGH if sym.startswith(("SH688", "SZ300")) else LIMIT_DOWN_THRESHOLD
    return chg <= thresh


LGB_PARAMS = dict(
    loss="mse", colsample_bytree=0.8879, learning_rate=0.0421,
    subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768,
    max_depth=8, num_leaves=210, num_threads=1,
)


def month_start(d):
    return d.strftime("%Y-%m-01")


def month_end(d):
    nm = (d.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)
    return nm.strftime("%Y-%m-%d")


def build_market_proxy() -> pd.Series:
    df = pd.read_parquet(INDEX_PARQUET)
    df = df[df["code"] == INDEX_CODE].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


def regime_for_day(td, proxy):
    if td not in proxy.index:
        return K_NORMAL, DROP_NORMAL, "normal"
    pos = proxy.index.get_loc(td)
    if pos < 200:
        return K_NORMAL, DROP_NORMAL, "normal"
    ma200 = proxy.iloc[pos - 199: pos + 1].mean()
    px = proxy.iloc[pos]
    if px < ma200 * BEAR_MA_RATIO:
        return 0, 0, "bear"
    rets20 = proxy.iloc[pos - 19: pos + 1].pct_change().dropna()
    vol_ann = rets20.std() * np.sqrt(252) if len(rets20) > 0 else 0
    if vol_ann > PANIC_VOL_ANN:
        return K_HALF, DROP_HALF, "panic"
    if pos >= 60:
        ret_60d = proxy.iloc[pos] / proxy.iloc[pos - 60] - 1
        if ret_60d < DRAWDOWN_60D:
            return K_HALF, DROP_HALF, "drawdown"
    return K_NORMAL, DROP_NORMAL, "normal"


def get_price_data(start, end):
    df = D.features(
        D.instruments(market=MARKET),
        ["$open", "$close", "$volume"],
        start_time=start, end_time=end, freq="day",
    ).reset_index()
    df.columns = ["instrument", "date", "open", "close", "volume"]
    return df


_pred_cache: dict[str, pd.Series] = {}
_pred_disk_df: pd.DataFrame | None = None


def _load_pred_from_disk(month_key: str) -> pd.Series | None:
    global _pred_disk_df
    if not PRED_CACHE.exists():
        return None
    if _pred_disk_df is None:
        _pred_disk_df = pd.read_parquet(PRED_CACHE)
    df = _pred_disk_df[_pred_disk_df["month"] == month_key]
    if len(df) == 0:
        return None
    df = df.set_index(["datetime", "instrument"])
    return df["score"]


def get_pred_for_month(test_month_start):
    key = test_month_start.strftime("%Y-%m")
    if key in _pred_cache:
        return _pred_cache[key]
    cached = _load_pred_from_disk(key)
    if cached is not None:
        _pred_cache[key] = cached
        return cached
    test_start = month_start(test_month_start)
    test_end = month_end(test_month_start)
    train_start = month_start(test_month_start - relativedelta(months=TRAIN_MONTHS))
    valid_start = month_start(test_month_start - relativedelta(months=1))
    train_end = month_end(test_month_start - relativedelta(months=2))
    valid_end = month_end(test_month_start - relativedelta(months=1))
    handler = Alpha158(
        start_time=train_start, end_time=test_end,
        fit_start_time=train_start, fit_end_time=train_end,
        instruments=MARKET,
    )
    dataset = DatasetH(handler=handler, segments={
        "train": (train_start, train_end),
        "valid": (valid_start, valid_end),
        "test": (test_start, test_end),
    })
    model = LGBModel(**LGB_PARAMS)
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    _pred_cache[key] = pred
    _persist_pred(key, pred)
    return pred


def _persist_pred(month_key: str, pred: pd.Series) -> None:
    """每月训完追加 pred 到磁盘 cache (idempotent by month)."""
    new_df = pred.to_frame("score").reset_index()
    new_df["month"] = month_key
    if PRED_CACHE.exists():
        old = pd.read_parquet(PRED_CACHE)
        old = old[old["month"] != month_key]
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    PRED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRED_CACHE, index=False)
    global _pred_disk_df
    _pred_disk_df = out


def realistic_window(test_month_start, proxy, with_regime: bool = True):
    test_start = month_start(test_month_start)
    test_end = month_end(test_month_start)

    pred = get_pred_for_month(test_month_start)

    price_end = (pd.to_datetime(test_end) + timedelta(days=10)).strftime("%Y-%m-%d")
    price_df = get_price_data(test_start, price_end)
    open_pv = price_df.pivot(index="date", columns="instrument", values="open")
    close_pv = price_df.pivot(index="date", columns="instrument", values="close")
    vol_pv = price_df.pivot(index="date", columns="instrument", values="volume")

    pred_unstacked = pred.unstack(level="instrument")
    test_dates = sorted(pred_unstacked.index)
    if len(test_dates) < 2:
        return {"abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                "regime_days": "", "n_skipped_limit": 0}

    current_holdings: dict = {}
    cash = PORTFOLIO_VALUE
    daily_ret = []
    n_picks_realized = []
    last_known_price: dict = {}
    n_skipped_limit = 0
    regime_counts = {"normal": 0, "bear": 0, "panic": 0, "drawdown": 0}

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

    def liquidate_all(next_td):
        nonlocal cash
        for c in list(current_holdings.keys()):
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

    for di, td in enumerate(test_dates):
        if di + 1 >= len(test_dates):
            break
        next_td = test_dates[di + 1]
        if next_td not in open_pv.index:
            continue

        if with_regime:
            topk, n_drop, label = regime_for_day(td, proxy)
        else:
            topk, n_drop, label = K_NORMAL, DROP_NORMAL, "normal"
        regime_counts[label] += 1

        port_val = mark_to_market(td)

        if topk == 0:
            liquidate_all(next_td)
            n_picks_realized.append(0)
            new_port_val = mark_to_market(next_td)
            if port_val > 0:
                daily_ret.append(new_port_val / port_val - 1)
            continue

        scores_all = pred_unstacked.loc[td].dropna().sort_values(ascending=False)
        pool = scores_all.head(topk * CANDIDATE_POOL_MULTIPLIER)

        filtered = []
        for sym, score in pool.items():
            if sym not in close_pv.columns or di == 0:
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
        n_picks_realized.append(len(current_holdings))

        new_port_val = mark_to_market(next_td)
        if port_val > 0:
            daily_ret.append(new_port_val / port_val - 1)

    if not daily_ret:
        return {"abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                "regime_days": "", "n_skipped_limit": n_skipped_limit}
    abs_ret = (1 + pd.Series(daily_ret)).prod() - 1
    rg = "/".join(f"{k}={v}" for k, v in regime_counts.items() if v > 0)
    return {
        "abs_ret_%": round(abs_ret * 100, 2),
        "avg_picks": round(np.mean(n_picks_realized), 1) if n_picks_realized else 0,
        "n_days": len(daily_ret),
        "regime_days": rg,
        "n_skipped_limit": n_skipped_limit,
    }


def annualize_metrics(returns, n_periods_per_year=12):
    cum = (1 + returns / 100).prod() - 1
    n = len(returns)
    years = n / n_periods_per_year
    ann_ret = (1 + cum) ** (1 / years) - 1 if years > 0 else 0
    mean = (returns / 100).mean()
    std = (returns / 100).std()
    sharpe = mean / std * np.sqrt(n_periods_per_year) if std > 0 else 0
    cum_series = (1 + returns / 100).cumprod()
    peak = cum_series.cummax()
    mdd = ((cum_series - peak) / peak).min()
    return {
        "cum_%": round(cum * 100, 2),
        "ann_%": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 2),
        "mdd_%": round(mdd * 100, 2),
        "win_%": round((returns > 0).sum() / len(returns) * 100, 2),
        "n": n,
    }


def main():
    global K_NORMAL, DROP_NORMAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--drop", type=int, required=True)
    parser.add_argument("--tag", type=str, required=True,
                         help="artifact 文件名后缀, e.g. k3d2")
    parser.add_argument("--months", type=int, default=0,
                         help="0=use full [first-test, last-test] range; >0=truncate to last N")
    parser.add_argument("--first-test", type=str, default="2023-01",
                         help="first test month YYYY-MM (default 2023-01)")
    parser.add_argument("--last-test", type=str, default="2026-04",
                         help="last test month YYYY-MM (default 2026-04)")
    args = parser.parse_args()

    K_NORMAL = args.k
    DROP_NORMAL = args.drop

    qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
    print(f"[init] qlib OK; config K={args.k} D={args.drop} tag={args.tag}")
    if PRED_CACHE.exists():
        sz = PRED_CACHE.stat().st_size / 1e6
        print(f"[init] pred cache available: {PRED_CACHE.name} ({sz:.1f} MB)")
    else:
        print("[init] WARN: no pred cache, will train LGB per month (slow)")

    proxy = build_market_proxy()

    first_test = datetime.strptime(args.first_test + "-01", "%Y-%m-%d")
    last_test = datetime.strptime(args.last_test + "-01", "%Y-%m-%d")
    months = []
    cur = first_test
    while cur <= last_test:
        months.append(cur)
        cur += relativedelta(months=1)
    if args.months and args.months < len(months):
        months = months[-args.months:]
    print(f"[run] {len(months)} 月 walk-forward (baseline only)")

    all_rows = []
    for i, m in enumerate(months, 1):
        try:
            res = realistic_window(m, proxy, with_regime=False)
            res["month"] = m.strftime("%Y-%m")
            res["config"] = f"K={args.k} D={args.drop}"
            all_rows.append(res)
            print(f"  {i:2d}/{len(months)} {res['month']}: "
                  f"abs_ret={res['abs_ret_%']:+6.2f}%  picks={res['avg_picks']:.1f}",
                  flush=True)
        except Exception as e:
            print(f"  {i:2d}/{len(months)} {m.strftime('%Y-%m')} FAIL: {str(e)[:80]}")
            all_rows.append({"month": m.strftime("%Y-%m"), "config": f"K={args.k} D={args.drop}",
                              "abs_ret_%": 0, "avg_picks": 0, "n_days": 0,
                              "regime_days": "", "n_skipped_limit": 0})

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / f"v17_{args.tag}_stats.csv"
    df.to_csv(out_csv, index=False)

    mm = annualize_metrics(df["abs_ret_%"])
    mm["avg_picks"] = round(df["avg_picks"].mean(), 1)
    mm["k"] = args.k
    mm["drop"] = args.drop
    mm["tag"] = args.tag
    print(f"\n=== SUMMARY tag={args.tag} K={args.k} D={args.drop} ===")
    print(pd.Series(mm).to_string())
    print(f"\n输出: v17_{args.tag}_stats.csv")


if __name__ == "__main__":
    main()
