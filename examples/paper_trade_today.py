"""今日选股信号 — v13 K=8 drop=2 在 Baidu 全 A 股数据上.

流程:
  1. qlib.init(provider_uri=data_cache/qlib_baidu)
  2. Alpha158 handler 训练 LGB on 最近 12 个月 (ending yesterday)
  3. 预测今日截面排名
  4. Top 8 候选 + 跟昨日持仓 diff → buy/sell/hold 操作建议
  5. 保存今日持仓到 data_cache/portfolio_state.json (下次跑读取做 diff)

输出:
  - stdout: 操作建议表
  - data_cache/paper_trade_log.csv: 历史信号 log
  - data_cache/portfolio_state.json: 最新持仓 (供下次 diff)

Run:  python examples/paper_trade_today.py
"""
from __future__ import annotations

import json
import warnings
from datetime import timedelta
from pathlib import Path

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
QLIB_DIR = ROOT / "data_cache" / "qlib_baidu"
UNIVERSE_PATH = ROOT / "data_cache" / "universe.csv"
STATE_PATH = ROOT / "data_cache" / "portfolio_state.json"
LOG_PATH = ROOT / "data_cache" / "paper_trade_log.csv"

K = 8
N_DROP = 2
TRAIN_MONTHS = 12
# 候选池扩 4 倍, 过滤涨停+跌停后取前 K. backtest 同样有这层过滤.
CANDIDATE_POOL_MULTIPLIER = 4
LIMIT_UP_THRESHOLD = 0.095  # 当日涨幅 ≥9.5% 视为涨停/接近涨停, 买不到
LIMIT_DOWN_THRESHOLD = -0.095  # ≤-9.5% 视为跌停, 别去接落刀
# 科创板/创业板 (688/300) 涨跌停 ±20%, 阈值放宽
LIMIT_UP_THRESHOLD_HIGH = 0.195
LIMIT_DOWN_THRESHOLD_HIGH = -0.195

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


def load_universe_names() -> dict[str, str]:
    """key 用大写, 跟 qlib 输出的 SH/SZ 对齐."""
    df = pd.read_csv(UNIVERSE_PATH, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    out = {}
    for _, r in df.iterrows():
        prefix = "SH" if r["code"].startswith(("6", "9")) else "SZ"
        out[f"{prefix}{r['code']}"] = r["name"]
    return out


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"date": None, "holdings": []}


def save_state(date_str: str, holdings: list[str]) -> None:
    STATE_PATH.write_text(json.dumps(
        {"date": date_str, "holdings": holdings}, ensure_ascii=False, indent=2,
    ))


def append_log(date_str: str, action: str, symbol: str, name: str,
               score: float, price: float) -> None:
    new_row = {"date": date_str, "action": action, "symbol": symbol,
               "name": name, "score": round(score, 5), "price": price}
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        out = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        out = pd.DataFrame([new_row])
    out.to_csv(LOG_PATH, index=False)


def main():
    qlib.init(provider_uri=str(QLIB_DIR), region=REG_CN)
    print(f"[1/4] qlib initialized — {QLIB_DIR}")

    cal = D.calendar(freq="day")
    latest = pd.Timestamp(cal[-1])
    print(f"     最新交易日: {latest.date()}")

    train_end = month_end(latest - relativedelta(months=1))
    train_start = month_start(latest - relativedelta(months=TRAIN_MONTHS + 1))
    valid_start = month_start(latest - relativedelta(months=1))
    valid_end = train_end
    test_start = (latest - timedelta(days=7)).strftime("%Y-%m-%d")
    test_end = latest.strftime("%Y-%m-%d")

    print("[2/4] Alpha158 handler ...")
    print(f"     train: {train_start} → {train_end}")
    print(f"     valid: {valid_start} → {valid_end}")
    print(f"     test : {test_start} → {test_end}")

    handler = Alpha158(
        start_time=train_start, end_time=test_end,
        fit_start_time=train_start, fit_end_time=train_end,
        instruments="csi300",
    )
    dataset = DatasetH(handler=handler, segments={
        "train": (train_start, train_end),
        "valid": (valid_start, valid_end),
        "test":  (test_start, test_end),
    })

    print("\n[3/4] LGB train + predict ...")
    model = LGBModel(**LGB_PARAMS)
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")

    pred_df = pred.reset_index()
    pred_df.columns = ["datetime", "instrument", "score"]
    pred_df = pred_df.sort_values("datetime")
    today = pred_df["datetime"].max()
    # 候选池: 扩大 K * MULTIPLIER 然后过滤涨停后取前 K
    candidate_pool = pred_df[pred_df["datetime"] == today].sort_values(
        "score", ascending=False).head(K * CANDIDATE_POOL_MULTIPLIER).reset_index(drop=True)

    # 拉候选池近 2 日 close 算当日涨幅
    pool_syms = candidate_pool["instrument"].tolist()
    pool_prices = D.features(pool_syms, ["$close"],
                              start_time=test_start, end_time=test_end, freq="day")
    closes_today_all = pool_prices.xs(today, level="datetime")["$close"].to_dict()
    pool_dates = sorted(pool_prices.index.get_level_values("datetime").unique())
    prev_td = pool_dates[-2] if len(pool_dates) >= 2 else None
    closes_prev = (pool_prices.xs(prev_td, level="datetime")["$close"].to_dict()
                    if prev_td is not None else {})

    def is_limit_up(sym, chg):
        thresh = LIMIT_UP_THRESHOLD_HIGH if sym.startswith(("SH688", "SZ300")) else LIMIT_UP_THRESHOLD
        return chg >= thresh

    def is_limit_down(sym, chg):
        thresh = LIMIT_DOWN_THRESHOLD_HIGH if sym.startswith(("SH688", "SZ300")) else LIMIT_DOWN_THRESHOLD
        return chg <= thresh

    filtered_rows = []
    skipped = []
    for _, row in candidate_pool.iterrows():
        sym = row["instrument"]
        prev = closes_prev.get(sym, 0)
        curr = closes_today_all.get(sym, 0)
        chg = (curr / prev - 1) if prev > 0 else 0
        if is_limit_up(sym, chg):
            skipped.append((sym, chg, "涨停"))
            continue
        if is_limit_down(sym, chg):
            skipped.append((sym, chg, "跌停"))
            continue
        filtered_rows.append(row)
        if len(filtered_rows) >= K:
            break

    today_pred = pd.DataFrame(filtered_rows).reset_index(drop=True)
    closes_today = {sym: closes_today_all.get(sym, 0)
                     for sym in today_pred["instrument"].tolist()}
    if skipped:
        n_up = sum(1 for _, _, t in skipped if t == "涨停")
        n_dn = sum(1 for _, _, t in skipped if t == "跌停")
        print(f"\n[过滤涨跌停] 跳过 {len(skipped)} 只 (涨停 {n_up} / 跌停 {n_dn}):")
        for s, c, t in skipped[:10]:
            sign = "+" if c >= 0 else ""
            print(f"  - {s} {sign}{c*100:.2f}%  [{t}]")

    names = load_universe_names()

    def name_of(sym):
        return names.get(sym) or names.get(sym.upper()) or names.get(sym.lower()) or "?"

    print(f"\n[4/4] === 今日选股 ({today.date()}) ===\n")
    print(f"Model:    qlib Alpha158 + LGB, K={K} drop={N_DROP}")
    print(f"Train:    {train_start} → {train_end} ({TRAIN_MONTHS}个月)")
    print("Universe: CSI300 (300 只, v17 验证 40月 +53.8% / Sharpe 0.71)\n")

    print(f"[今日 Top {K} 候选]")
    top_symbols = today_pred["instrument"].tolist()
    for i, row in today_pred.iterrows():
        sym = row["instrument"]
        name = name_of(sym)
        price = closes_today.get(sym, 0)
        print(f"  {i+1}. {sym}  {name:8s}  score={row['score']:+.4f}  最新价={price:.2f}")

    state = load_state()
    last_date = state.get("date") or "(无前次记录)"
    holdings = state.get("holdings") or []

    target = top_symbols
    sell_candidates = [h for h in holdings if h not in target][:N_DROP]
    buy_candidates = [t for t in target if t not in holdings][:N_DROP]
    hold = [h for h in holdings if h in target]

    today_str = today.strftime("%Y-%m-%d")
    print(f"\n[今日操作建议] (上次状态: {last_date}, 持仓 {len(holdings)} 只)")
    if not holdings:
        print("  首次跑 — 建仓: 选 Top N_DROP=2 买入, 后续每天最多换 2 只")
        new_holdings = []
        for s in target[:N_DROP]:
            name = name_of(s)
            print(f"  BUY  {s}  {name}")
            new_holdings.append(s)
            append_log(today_str, "BUY", s, name,
                        today_pred[today_pred["instrument"] == s]["score"].iloc[0],
                        closes_today.get(s, 0))
    else:
        if sell_candidates:
            for s in sell_candidates:
                name = name_of(s)
                print(f"  SELL {s}  {name}  (跌出 Top {K})")
                append_log(today_str, "SELL", s, name, 0,
                            closes_today.get(s, 0))
        if buy_candidates:
            for s in buy_candidates:
                name = name_of(s)
                price = closes_today.get(s, 0)
                score = today_pred[today_pred["instrument"] == s]["score"].iloc[0]
                print(f"  BUY  {s}  {name}  最新价={price:.2f}  score={score:+.4f}")
                append_log(today_str, "BUY", s, name, score, price)
        if not sell_candidates and not buy_candidates:
            print(f"  HOLD — 持仓维持 (Top {K} 与昨日无差异)")
        new_holdings = hold + [b for b in buy_candidates]

    save_state(today_str, new_holdings)
    print(f"\n[state] 已保存今日持仓 ({len(new_holdings)} 只) → {STATE_PATH.name}")
    if LOG_PATH.exists():
        print(f"[log]   信号 log → {LOG_PATH.name}")


if __name__ == "__main__":
    main()
