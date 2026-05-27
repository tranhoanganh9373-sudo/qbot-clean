"""今日选股信号 — v19.4 SHADOW (forward OOS A/B 跟踪 against v19.6 主 production).

This is the shadow copy of v19.4 (margin sidecar) kept alongside production v19.6
(amplitude sidecar). Output is written to SEPARATE log/state files so the main
production log/state stays untouched.

  主 production (v19.6): examples/paper_trade_today.py → paper_trade_log.csv
  Shadow (v19.4):        examples/paper_trade_v19_4.py → paper_trade_log_v19_4.csv

forward_oos_monitor.py 仍消费主 paper_trade_log (v19.6). shadow 数据由用户
手动 / 定期对比两者真实 Sharpe / Calmar, 不集成 alert.

v19.4 (v19.1 base + margin sidecar overlay, Phase 4 严格 OOS 通过):
  final_score = z(train24_pred) - 0.10 × z(margin_5d_chg) - 0.10 × z(margin_20d_chg)
  margin 反向使用 (融资余额放大 = 拥挤 → 减分).
  OOS 验证: Calmar 0.61 vs v19.1 0.42 (+45%), Sharpe 0.76 vs 0.68 (+12%),
            MDD -21.2% vs -29.9% (+8.6pp).

  Toggle: 把 USE_V19_4_SIDECAR=False 即可回 v19.1 行为.

v19.1 (取消 vol-target, 因实测无效) = B' 配置, 60月 walk-forward 全场冠军:
  cum +695% / ann +51% / Sharpe 1.09 / MDD -26% / Calmar 1.96 ⭐⭐
  月度胜率 62% (全场最高)
  实盘期望: ann +25~30%, MDD -30%~-35% (扣 survivorship + 摩擦)

vs v19.0 (vt=0.15):
  v19.1 cum 提升 +114pp (681→695), MDD 几乎无变化 (-25.55→-26.27).
  vol-target 在 CSI300 实现 vol 多数 < 15% 时几乎不触发,反而切掉 alpha.

流程:
  1. qlib.init(provider_uri=data_cache/qlib_baidu)
  2. Alpha158 handler 训练 DEnsemble on 最近 24 个月
  3. 预测今日截面排名
  4. (v19.4) 横截面 z(pred) - 0.10*z(m5) - 0.10*z(m20) — 缺 margin 数据者保留 pred
  5. Top K=8 候选 (过滤涨停+跌停+贵价) → 跟昨日持仓 diff → BUY/SELL
  6. 算今日 CSI300 20日实现波动率 (仅显示, 不再用于减仓)
  7. 保存今日持仓到 data_cache/portfolio_state.json

输出:
  - stdout: 操作建议 + 风控提示
  - data_cache/paper_trade_log.csv  (历史信号 log, schema 不变)
  - data_cache/portfolio_state.json (最新持仓, schema 不变)

Run:  python examples/paper_trade_today.py
      python examples/paper_trade_today.py --dry-run     (仅打印 picks, 不写 log/state)
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from dateutil.relativedelta import relativedelta
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.double_ensemble import DEnsembleModel
from qlib.data import D
from qlib.data.dataset import DatasetH

warnings.filterwarnings("ignore")

# v19.4-shadow: 给所有 print 加 prefix 以便从 daily_check.sh log 区分主 production
_SHADOW_PREFIX = "[v19.4-shadow] "
_builtin_print = print


def print(*args, **kwargs):  # noqa: A001 — intentional shadow of builtin
    if args and isinstance(args[0], str):
        args = (_SHADOW_PREFIX + args[0],) + args[1:]
    else:
        args = (_SHADOW_PREFIX,) + args
    return _builtin_print(*args, **kwargs)


ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = ROOT / "data_cache" / "qlib_baidu"
UNIVERSE_PATH = ROOT / "data_cache" / "universe.csv"
# v19.4 shadow: 写入独立 path 避免覆盖 production v19.6 主 log/state
STATE_PATH = ROOT / "data_cache" / "portfolio_state_v19_4.json"
LOG_PATH = ROOT / "data_cache" / "paper_trade_log_v19_4.csv"
INDEX_PARQUET = ROOT / "data_cache" / "index_kline.parquet"
MARGIN_PARQUET = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
MARGIN_DAILY_PARQUET = ROOT / "data_cache" / "csi300_margin_daily.parquet"  # v19.4 daily sidecar
INDEX_CODE = "sh000300"

K = 8
N_DROP = 2
TRAIN_MONTHS = 24  # v19: 12 → 24 (60月 walk-forward 验证: Calmar 1.96 vs train=12 的 1.21)
PORTFOLIO_VALUE = 5e4  # 实盘本金 5 万 (backtest 内部 capital=25k 只是 alpha-集中参数, 不是本金)
VOL_TARGET_ANN = 0.0     # v19.1: 0=off (60月 backtest 显示 vt=0.15 实测无效, vol 多数 < 15%, 改 0 后 Calmar 1.83→1.96)

# v19.4 sidecar overlay (Phase 4 严格 OOS 通过, Calmar 0.42 → 0.61):
#   final = z(pred) - SIDECAR_LAMBDA_M5 * z(m5) - SIDECAR_LAMBDA_M20 * z(m20)
# margin 反向: 融资余额放大 = 拥挤 = 减分.
USE_V19_4_SIDECAR = True   # False = 回 v19.1 行为
SIDECAR_LAMBDA_M5 = 0.10
SIDECAR_LAMBDA_M20 = 0.10
# margin cache 最新数据距今 ≤ MARGIN_STALE_DAYS 才认可, 否则降级为 v19.1
MARGIN_STALE_DAYS = 7
# 候选池扩 4 倍, 过滤涨停+跌停后取前 K. backtest 同样有这层过滤.
CANDIDATE_POOL_MULTIPLIER = 4
LIMIT_UP_THRESHOLD = 0.095  # 当日涨幅 ≥9.5% 视为涨停/接近涨停, 买不到
LIMIT_DOWN_THRESHOLD = -0.095  # ≤-9.5% 视为跌停, 别去接落刀
# 科创板/创业板 (688/300) 涨跌停 ±20%, 阈值放宽
LIMIT_UP_THRESHOLD_HIGH = 0.195
LIMIT_DOWN_THRESHOLD_HIGH = -0.195
# 价格上限: PORTFOLIO_VALUE / (K / N_DROP) / 100 = 125 元 (v17 legacy formula, 沿用)
# 5万 / 4 / 100 = 125 元 — 即使本金 5 万也只挑 ≤125 元股 (保留 v17 sweet spot)
MAX_AFFORDABLE_PRICE = PORTFOLIO_VALUE / (K / N_DROP) / 100

DENS_PARAMS = dict(
    base_model="gbm",
    loss="mse",
    num_models=3,
    enable_sr=True,
    enable_fs=True,
    alpha1=1.0,
    alpha2=1.0,
    bins_sr=10,
    bins_fs=5,
    decay=0.5,
    sample_ratios=[0.8, 0.7, 0.6, 0.5, 0.4],
    sub_weights=[1, 0.2, 0.2],
    epochs=20,
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


def compute_vol_scale() -> tuple[float, float]:
    """读 CSI300 (sh000300) 算近 20日实现波动率 (年化), 返回 (vol_scale, realized_vol_ann).
    VOL_TARGET_ANN <= 0 表示 off, 永远满仓 (vol_scale=1.0).
    否则 vol_scale = min(VOL_TARGET_ANN / max(realized_vol, 0.05), 1.0).
    """
    if not INDEX_PARQUET.exists():
        return 1.0, 0.0
    idx = pd.read_parquet(INDEX_PARQUET)
    idx = idx[idx["code"] == INDEX_CODE].copy()
    if len(idx) < 21:
        return 1.0, 0.0
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").tail(25)
    rets = idx["close"].pct_change().dropna().tail(20)
    if len(rets) < 5:
        return 1.0, 0.0
    realized_vol = float(rets.std() * np.sqrt(252))
    if VOL_TARGET_ANN <= 0:
        return 1.0, realized_vol  # off
    scale = min(VOL_TARGET_ANN / max(realized_vol, 0.05), 1.0)
    return scale, realized_vol


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


def _qlib_sym_to_code(sym: str) -> str:
    """qlib 'SH600000' → margin 'code' '600000' (6 位数字)."""
    if len(sym) >= 8 and sym[:2] in ("SH", "SZ"):
        return sym[2:]
    return sym


def load_margin_overlay(today: pd.Timestamp) -> tuple[dict[str, float], dict[str, float], str]:
    """读 csi300_margin_daily.parquet (优先, 来自 fetch_margin_today.py 增量), 落空
    fallback 到 csi300_margin_14yr.parquet (只读, 长 cache).

    返回 today (T-1 fallback) 的 margin_5d_chg / margin_20d_chg, 按 qlib symbol 索引
    ('SH600000' → margin code '600000' → 'SH600000').

    返回 (m5_map, m20_map, status_str). status_str = 'ok' / 'stale-N-days' /
    'missing' / 'ok-daily' / 'ok-long'.
    """
    df = pd.DataFrame()
    source = ""
    # 优先 daily sidecar (T-1 最新)
    if MARGIN_DAILY_PARQUET.exists():
        try:
            df = pd.read_parquet(MARGIN_DAILY_PARQUET,
                                 columns=["code", "date", "margin_5d_chg", "margin_20d_chg"])
            source = "daily"
        except Exception:
            df = pd.DataFrame()
    if df.empty and MARGIN_PARQUET.exists():
        df = pd.read_parquet(MARGIN_PARQUET,
                             columns=["code", "date", "margin_5d_chg", "margin_20d_chg"])
        source = "long"
    if df.empty:
        return {}, {}, "missing"
    df["date"] = pd.to_datetime(df["date"])
    # 取 ≤ today 的最新交易日 margin (margin 数据 T-1 节奏)
    df = df[df["date"] <= today]
    if df.empty:
        return {}, {}, "missing"
    latest_margin_date = df["date"].max()
    stale_days = (today.normalize() - latest_margin_date.normalize()).days
    snap = df[df["date"] == latest_margin_date]
    m5: dict[str, float] = {}
    m20: dict[str, float] = {}
    for _, r in snap.iterrows():
        code = str(r["code"]).zfill(6)
        # qlib 用 'SH' for 6/9 开头, 'SZ' for 0/3 开头
        prefix = "SH" if code.startswith(("6", "9")) else "SZ"
        sym = f"{prefix}{code}"
        m5_v = r["margin_5d_chg"]
        m20_v = r["margin_20d_chg"]
        if pd.notna(m5_v):
            m5[sym] = float(m5_v)
        if pd.notna(m20_v):
            m20[sym] = float(m20_v)
    if stale_days > MARGIN_STALE_DAYS:
        status = f"stale-{stale_days}-days-from-{source}"
    else:
        status = f"ok-{source}({stale_days}d)"
    return m5, m20, status


def apply_sidecar_overlay(pred_today: pd.DataFrame, today: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """v19.4: 在 today 截面上加 margin overlay.
    pred_today: 至少含 [instrument, score] 列.
    返回: (含 final_score 的 DataFrame [按 final_score desc 排], meta_dict).

    缺 margin 数据的股票: m5_z / m20_z 视为 0 (即 final = z(pred) 单独).
    若 margin cache missing 或 stale, 静默退回 v19.1 (final = pred).
    """
    meta: dict = {"sidecar_applied": False, "n_with_margin": 0, "n_total": len(pred_today),
                  "margin_status": "n/a"}
    out = pred_today.copy()
    out["pred_z"] = (out["score"] - out["score"].mean()) / (out["score"].std(ddof=0) or 1.0)
    out["final_score"] = out["pred_z"]  # default fallback

    if not USE_V19_4_SIDECAR:
        meta["margin_status"] = "toggle-off"
        return out.sort_values("final_score", ascending=False).reset_index(drop=True), meta

    m5_map, m20_map, status = load_margin_overlay(today)
    meta["margin_status"] = status
    if not status.startswith("ok"):
        # 降级 v19.1: missing 或 stale, 仅警告打印, sidecar 不应用
        return out.sort_values("final_score", ascending=False).reset_index(drop=True), meta

    out["margin_5d_chg"] = out["instrument"].map(m5_map)
    out["margin_20d_chg"] = out["instrument"].map(m20_map)

    # 横截面 z (skipna=True). 缺数据用 mean 填 → z=0 (中性).
    m5_series = out["margin_5d_chg"]
    m20_series = out["margin_20d_chg"]
    m5_mean = m5_series.mean()
    m20_mean = m20_series.mean()
    m5_std = m5_series.std(ddof=0)
    m20_std = m20_series.std(ddof=0)
    m5_filled = m5_series.fillna(m5_mean if pd.notna(m5_mean) else 0.0)
    m20_filled = m20_series.fillna(m20_mean if pd.notna(m20_mean) else 0.0)
    out["m5_z"] = (m5_filled - (m5_mean if pd.notna(m5_mean) else 0.0)) / (m5_std or 1.0)
    out["m20_z"] = (m20_filled - (m20_mean if pd.notna(m20_mean) else 0.0)) / (m20_std or 1.0)

    out["final_score"] = (out["pred_z"]
                          - SIDECAR_LAMBDA_M5 * out["m5_z"]
                          - SIDECAR_LAMBDA_M20 * out["m20_z"])

    meta["sidecar_applied"] = True
    meta["n_with_margin"] = int(out["margin_5d_chg"].notna().sum())
    return out.sort_values("final_score", ascending=False).reset_index(drop=True), meta


def main(dry_run: bool = False):
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

    print("\n[3/4] DEnsemble train + predict (3 sub-models, ~30-60s) ...")
    model = DEnsembleModel(**DENS_PARAMS)
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]

    pred_df = pred.reset_index()
    pred_df.columns = ["datetime", "instrument", "score"]
    pred_df = pred_df.sort_values("datetime")
    today = pred_df["datetime"].max()

    # v19.4: 在 today 全截面应用 sidecar overlay (margin reverse), 再按 final_score 取候选池.
    today_full = pred_df[pred_df["datetime"] == today].reset_index(drop=True)
    overlay_df, overlay_meta = apply_sidecar_overlay(today_full, today)

    # final_score 已替代 score 作为排序依据 (sidecar off 时 final_score = z(pred), 排序等价).
    candidate_pool = overlay_df.head(K * CANDIDATE_POOL_MULTIPLIER).reset_index(drop=True)
    # 候选池里仍把 "score" 字段保留作为原 pred 输出, 但用 "final_score" 做排序.
    # 下游 (BUY/SELL log) 写入的 score 字段保持原 pred score (避免 schema 改动).

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
        if curr > MAX_AFFORDABLE_PRICE:
            skipped.append((sym, chg, f"贵({curr:.0f})"))
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
        n_exp = sum(1 for _, _, t in skipped if t.startswith("贵"))
        print(f"\n[过滤] 跳过 {len(skipped)} 只 (涨停 {n_up} / 跌停 {n_dn} / "
              f"贵>{MAX_AFFORDABLE_PRICE:.0f}元 {n_exp}):")
        for s, c, t in skipped[:10]:
            sign = "+" if c >= 0 else ""
            print(f"  - {s} {sign}{c*100:.2f}%  [{t}]")

    names = load_universe_names()

    def name_of(sym):
        return names.get(sym) or names.get(sym.upper()) or names.get(sym.lower()) or "?"

    print(f"\n[4/4] === 今日选股 ({today.date()}) ===\n")
    vt_label = f"{VOL_TARGET_ANN*100:.0f}%" if VOL_TARGET_ANN > 0 else "OFF"
    sidecar_label = ("v19.4 ON" if overlay_meta["sidecar_applied"]
                     else f"OFF ({overlay_meta['margin_status']})")
    print(f"Model:    qlib Alpha158 + DoubleEnsemble, K={K} drop={N_DROP}")
    print(f"Capital:  {PORTFOLIO_VALUE:.0f} 元  Vol-target: {vt_label}  Margin-sidecar: {sidecar_label}")
    print(f"Train:    {train_start} → {train_end} ({TRAIN_MONTHS}个月)")
    if overlay_meta["sidecar_applied"]:
        print(f"Sidecar:  λ_m5={SIDECAR_LAMBDA_M5}  λ_m20={SIDECAR_LAMBDA_M20}  "
              f"covered={overlay_meta['n_with_margin']}/{overlay_meta['n_total']} "
              f"(OOS Calmar 0.61 vs v19.1 0.42)")
    print("Universe: CSI300 (300 只, v19.1 60月: ann +51% / Sharpe 1.09 / MDD -26% / Calmar 1.96)\n")

    vol_scale, realized_vol = compute_vol_scale()
    if realized_vol > 0:
        if VOL_TARGET_ANN <= 0:
            print(f"[风控] CSI300 20日年化波动率: {realized_vol*100:.1f}% "
                  f"(vol-target OFF, 永远满仓)")
        else:
            action = "满仓" if vol_scale >= 0.99 else f"减仓至 {vol_scale*100:.0f}%"
            print(f"[风控] CSI300 20日年化波动率: {realized_vol*100:.1f}%  "
                  f"→ vol-target={VOL_TARGET_ANN*100:.0f}% 建议 {action}")
            if vol_scale < 1.0:
                print(f"       原 BUY 单股 cash_per_pick={PORTFOLIO_VALUE/N_DROP:.0f} 元 "
                      f"→ 按 {vol_scale*100:.0f}% 缩 = "
                      f"{PORTFOLIO_VALUE/N_DROP*vol_scale:.0f} 元")
        print()

    print(f"[今日 Top {K} 候选]")
    top_symbols = today_pred["instrument"].tolist()
    has_final = "final_score" in today_pred.columns
    for i, row in today_pred.iterrows():
        sym = row["instrument"]
        name = name_of(sym)
        price = closes_today.get(sym, 0)
        if has_final and overlay_meta["sidecar_applied"]:
            print(f"  {i+1}. {sym}  {name:8s}  pred={row['score']:+.4f}  "
                  f"final={row['final_score']:+.4f}  最新价={price:.2f}")
        else:
            print(f"  {i+1}. {sym}  {name:8s}  score={row['score']:+.4f}  最新价={price:.2f}")

    state = load_state()
    last_date = state.get("date") or "(无前次记录)"
    holdings = state.get("holdings") or []

    target = top_symbols
    sell_candidates = [h for h in holdings if h not in target][:N_DROP]
    buy_candidates = [t for t in target if t not in holdings][:N_DROP]
    hold = [h for h in holdings if h in target]

    today_str = today.strftime("%Y-%m-%d")
    dry_tag = " [DRY-RUN]" if dry_run else ""
    print(f"\n[今日操作建议]{dry_tag} (上次状态: {last_date}, 持仓 {len(holdings)} 只)")
    if not holdings:
        print("  首次跑 — 建仓: 选 Top N_DROP=2 买入, 后续每天最多换 2 只")
        new_holdings = []
        for s in target[:N_DROP]:
            name = name_of(s)
            print(f"  BUY  {s}  {name}")
            new_holdings.append(s)
            if not dry_run:
                append_log(today_str, "BUY", s, name,
                            today_pred[today_pred["instrument"] == s]["score"].iloc[0],
                            closes_today.get(s, 0))
    else:
        if sell_candidates:
            for s in sell_candidates:
                name = name_of(s)
                print(f"  SELL {s}  {name}  (跌出 Top {K})")
                if not dry_run:
                    append_log(today_str, "SELL", s, name, 0,
                                closes_today.get(s, 0))
        if buy_candidates:
            for s in buy_candidates:
                name = name_of(s)
                price = closes_today.get(s, 0)
                score = today_pred[today_pred["instrument"] == s]["score"].iloc[0]
                print(f"  BUY  {s}  {name}  最新价={price:.2f}  score={score:+.4f}")
                if not dry_run:
                    append_log(today_str, "BUY", s, name, score, price)
        if not sell_candidates and not buy_candidates:
            print(f"  HOLD — 持仓维持 (Top {K} 与昨日无差异)")
        if vol_scale < 1.0 and buy_candidates:
            print(f"\n  [风控提示] 上述 BUY 单股金额应按 {vol_scale*100:.0f}% 缩 "
                  f"(vol-target 当前生效)")
        # 修 bug: 保留所有未被 SELL 的旧 holdings + 新 BUY (之前只保留 hold ∩ target, 漏掉未被换的)
        new_holdings = [h for h in holdings if h not in sell_candidates] + buy_candidates

    if dry_run:
        print(f"\n[dry-run] 不写 state / log. picks = {top_symbols}")
    else:
        save_state(today_str, new_holdings)
        print(f"\n[state] 已保存今日持仓 ({len(new_holdings)} 只) → {STATE_PATH.name}")
        if LOG_PATH.exists():
            print(f"[log]   信号 log → {LOG_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v19.4 paper trade signals")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印 picks, 不写 paper_trade_log.csv / portfolio_state.json")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
