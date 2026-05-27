"""今日选股信号 — v19.6: train24 + amplitude imbalance 20d sidecar (production).

v19.6 (v19.1 base + amp_imb_20d sidecar overlay, 60月 OOS sweep 单因子冠军):
  final_score = z(train24_pred) - 0.30 × z(amp_imb_20d)
  振幅 imbalance 反向使用 (涨势振幅过强 → 后续反转 → 减分).
  OOS 验证 (单因子 sweep): Calmar 0.79 vs v19.4 0.61 (+30%); stacked combo (v19.7)
  反而 Calmar 0.65 → 单因子 v19.6 胜出, stacked overfit abort.

  振幅因子定义 (参考 hugo2046/QuantsPlaybook):
      amp        = (high - low) / prev_close
      amp_up     = max(0, close - prev_close) / prev_close * amp
      amp_dn     = max(0, prev_close - close) / prev_close * amp
      amp_imb_Nd = (sum(amp_up,Nd) - sum(amp_dn,Nd)) / sum(amp,Nd)

  Toggle:
    - USE_V19_6_SIDECAR=False → v19.4 行为 (margin sidecar, 若 USE_V19_4_SIDECAR=True)
    - USE_V19_6_SIDECAR=False 且 USE_V19_4_SIDECAR=False → v19.1 baseline (无 sidecar)
    回退一行: 把 USE_V19_6_SIDECAR=False (产线立刻退回 v19.4 行为).

v19.4 (v19.1 base + margin sidecar overlay, Phase 4 严格 OOS 通过) — 保留作 shadow:
  final_score = z(train24_pred) - 0.10 × z(margin_5d_chg) - 0.10 × z(margin_20d_chg)
  margin 反向使用 (融资余额放大 = 拥挤 → 减分).
  OOS 验证: Calmar 0.61 vs v19.1 0.42 (+45%), Sharpe 0.76 vs 0.68 (+12%),
            MDD -21.2% vs -29.9% (+8.6pp).
  Shadow 跑在 examples/paper_trade_v19_4.py (独立 log/state, 不污染主流).

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

ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = ROOT / "data_cache" / "qlib_baidu"
UNIVERSE_PATH = ROOT / "data_cache" / "universe.csv"
STATE_PATH = ROOT / "data_cache" / "portfolio_state.json"
LOG_PATH = ROOT / "data_cache" / "paper_trade_log.csv"
INDEX_PARQUET = ROOT / "data_cache" / "index_kline.parquet"
MARGIN_PARQUET = ROOT / "data_cache" / "csi300_margin_14yr.parquet"
MARGIN_DAILY_PARQUET = ROOT / "data_cache" / "csi300_margin_daily.parquet"  # v19.4 daily sidecar
KLINE_PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"  # v19.6 amplitude sidecar source
INDEX_CODE = "sh000300"

K = 8
N_DROP = 2
TRAIN_MONTHS = 24  # v19: 12 → 24 (60月 walk-forward 验证: Calmar 1.96 vs train=12 的 1.21)
PORTFOLIO_VALUE = 5e4  # 实盘本金 5 万 (backtest 内部 capital=25k 只是 alpha-集中参数, 不是本金)
VOL_TARGET_ANN = 0.0     # v19.1: 0=off (60月 backtest 显示 vt=0.15 实测无效, vol 多数 < 15%, 改 0 后 Calmar 1.83→1.96)

# v19.6 sidecar overlay (production, 60月 OOS 单因子冠军, Calmar 0.79):
#   final = z(pred) - SIDECAR_LAMBDA_AMP_20D * z(amp_imb_20d)
# amp_imb_20d 反向: 涨势振幅过强 → 反转 = 减分.
USE_V19_6_SIDECAR = True   # 2026-05-26 reverted: v3 clean+full margin OOS Calmar 1.29 (vs v19.4 0.62 vs baseline 0.77).
SIDECAR_LAMBDA_AMP_20D = 0.30
# kline 数据 (baidu_kline.parquet) 最新数据距 today ≤ KLINE_STALE_DAYS 才认可, 否则降级.
KLINE_STALE_DAYS = 7

# v19.10 stacked sidecar (2026-05-27, Phase B OOS 60 月 Calmar 2.12 vs v19.6 1.29 = +64%):
#   final = z(pred) - 0.30 × z(amp_imb_20d) + 0.10 × z(JZF)
# 在 v19.6 final_score 上加 JZF (overnight gap) 二阶 sidecar.
# JZF = (open - prev_close) / prev_close × 100 = 集合竞价跳空幅度
# Spearman |ρ|(amp_imb_20d × JZF) IS 期 日截面 mean = 0.108 (中度独立)
# 历史 v19.7 abort 教训: 弱因子拖累; 这次 JZF 单 OOS 1.27 ≈ amp_imb 1.29, 两端都强 → stack 协同
# 回退一行: USE_V19_10_STACKED=False → 立刻退回 v19.6 单 sidecar 行为.
USE_V19_10_STACKED = True   # 2026-05-27 升级 v19.10 (stacked amp + JZF)
SIDECAR_LAMBDA_JZF = 0.10

# v19.4 sidecar overlay (Phase 4 严格 OOS 通过, Calmar 0.42 → 0.61) — 保留作 fallback / shadow:
#   final = z(pred) - SIDECAR_LAMBDA_M5 * z(m5) - SIDECAR_LAMBDA_M20 * z(m20)
# margin 反向: 融资余额放大 = 拥挤 = 减分.
# 注: production paper_trade_today.py 默认 USE_V19_4_SIDECAR=False (避免与 v19.6 双 sidecar 叠加).
#     v19.4 shadow 跑在 examples/paper_trade_v19_4.py 独立文件.
USE_V19_4_SIDECAR = False  # 2026-05-26 demoted again: v3 clean OOS Calmar 0.62 (v2 1.28 是 margin 15% 覆盖伪 best).
SIDECAR_LAMBDA_M5 = 0.10
SIDECAR_LAMBDA_M20 = 0.10

# === Multi-agent debate veto (P2-opt, 默认 OFF, 30 日 shadow A/B 后才上 production) ===
# 借鉴 jin-ce-zhi-suan 门下省一票否决思想, 让 multi_agent_debate 真参与 picks.
# 逻辑: 在 sidecar 出 Top K 后, 跑 DebateVeto.filter_picks();
#   neutral agent 投 SELL → 移出 picks (K 缩小为 K - veto_count, 不 backfill).
# 默认 False (本 toggle), 仅 shadow 阶段 dashboard preview;
# 升级 production 需 30 日真实 A/B 对比 (Calmar / Sharpe 显著优于) 才考虑.
# 一行回滚: USE_DEBATE_VETO=False (本行) 或全局 DEBATE_VETO_ENABLED=False (debate_veto.py).
USE_DEBATE_VETO = False
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
    # 2026-05-26: 跳过 partial-day (fetch_margin_today.py 部分失败留下的稀疏行).
    # 在最近 10 天里取 coverage ≥ 80% of max 的最新天 (相对阈值, 适应 stage2 swap 后 300 codes 数据规模).
    daily_cov = df.groupby("date")["code"].nunique().sort_index(ascending=False).head(10)
    if daily_cov.empty:
        latest_margin_date = df["date"].max()
    else:
        max_cov = int(daily_cov.max())
        # 阈值 = max * 0.80 (e.g. 300 → 240; 120 → 96). 严格防 partial-day silent demote.
        threshold = int(max_cov * 0.80)
        good = daily_cov[daily_cov >= threshold]
        latest_margin_date = good.index.max() if len(good) else df["date"].max()
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


# ===================== v19.6 amplitude sidecar (production) =====================

def load_amp_imb_20d_overlay(today: pd.Timestamp) -> tuple[dict[str, float], str]:
    """读 baidu_kline.parquet 实时算 amp_imb_20d, 返回 today (或最近 ≤today 交易日)
    每只股票的 amp_imb_20d, 按 qlib symbol 索引.

    amp        = (high - low) / prev_close
    amp_up     = max(0, close - prev_close) / prev_close * amp
    amp_dn     = max(0, prev_close - close) / prev_close * amp
    amp_imb_20d = (sum(amp_up,20) - sum(amp_dn,20)) / sum(amp,20)

    返回 (amp_map, status_str). status_str 形如 'ok(0d)' / 'stale-N-days' / 'missing'.
    """
    if not KLINE_PARQUET.exists():
        return {}, "missing"

    try:
        k = pd.read_parquet(
            KLINE_PARQUET,
            columns=["code", "date", "open", "high", "low", "close"],
        )
    except Exception as e:
        return {}, f"missing-read-error-{type(e).__name__}"

    k["code"] = k["code"].astype(str).str.zfill(6)
    k["date"] = pd.to_datetime(k["date"])
    # 只需要 today 前最近 ~30 个交易日 (20d rolling 需要 prev_close + 20 个点)
    min_date = today - pd.Timedelta(days=90)
    k = k[(k["date"] <= today) & (k["date"] >= min_date)].copy()
    if k.empty:
        return {}, "missing"

    latest_date = k["date"].max()
    stale_days = (today.normalize() - latest_date.normalize()).days
    source_status = f"ok({stale_days}d)" if stale_days <= KLINE_STALE_DAYS else f"stale-{stale_days}-days"

    k = k.sort_values(["code", "date"]).reset_index(drop=True)
    k["prev_close"] = k.groupby("code", sort=False)["close"].shift(1)
    valid = k["prev_close"].notna() & (k["prev_close"] > 0)

    amp = (k["high"] - k["low"]) / k["prev_close"]
    delta = k["close"] - k["prev_close"]
    amp_up = np.where(delta > 0, delta, 0.0) / k["prev_close"] * amp
    amp_dn = np.where(delta < 0, -delta, 0.0) / k["prev_close"] * amp
    k["amp"] = amp
    k["amp_up"] = amp_up
    k["amp_dn"] = amp_dn
    k.loc[~valid, ["amp", "amp_up", "amp_dn"]] = np.nan

    grp = k.groupby("code", sort=False)
    k["amp_sum_20d"] = grp["amp"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["amp_up_sum_20d"] = grp["amp_up"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    k["amp_dn_sum_20d"] = grp["amp_dn"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    den = k["amp_sum_20d"]
    num = k["amp_up_sum_20d"] - k["amp_dn_sum_20d"]
    k["amp_imb_20d"] = np.where(den > 0, num / den, np.nan)

    # 取 latest_date 截面 (PIT: latest ≤ today)
    snap = k[k["date"] == latest_date]
    amp_map: dict[str, float] = {}
    for _, r in snap.iterrows():
        code = str(r["code"]).zfill(6)
        v = r["amp_imb_20d"]
        if pd.isna(v):
            continue
        prefix = "SH" if code.startswith(("6", "9")) else "SZ"
        sym = f"{prefix}{code}"
        amp_map[sym] = float(v)

    if not amp_map:
        return {}, "missing-no-coverage"
    return amp_map, source_status


def load_jzf_overlay(today: pd.Timestamp) -> tuple[dict[str, float], str]:
    """v19.10: 读 baidu_kline.parquet 算今日 JZF (overnight gap) per symbol.

    JZF = (open - prev_close) / prev_close × 100  (集合竞价跳空幅度, %)

    返回 ({qlib_sym: jzf_value}, status):
      - status='ok-{date}': 用 today 的 open vs yesterday close
      - status='ok-stale-{date}': 用最近 ≤today 的交易日 (fallback)
      - status='missing-...': 数据缺失/stale, 上层调用应退回 v19.6 不 stack
    """
    if not KLINE_PARQUET.exists():
        return {}, "missing-no-cache"
    try:
        k = pd.read_parquet(
            KLINE_PARQUET, columns=["code", "date", "open", "close"]
        )
    except Exception as exc:  # noqa: BLE001
        return {}, f"missing-read-error-{type(exc).__name__}"
    k["date"] = pd.to_datetime(k["date"])
    latest = k["date"].max()
    if (today - latest).days > KLINE_STALE_DAYS:
        return {}, f"missing-stale-{latest.date()}"

    k = k.sort_values(["code", "date"])
    k["prev_close"] = k.groupby("code")["close"].shift(1)
    # 取每股 latest 行 (最新交易日)
    latest_per = k.groupby("code", sort=False).tail(1)
    latest_per = latest_per.dropna(subset=["prev_close"])
    latest_per = latest_per[latest_per["prev_close"] > 0]
    if latest_per.empty:
        return {}, "missing-no-rows"

    jzf_map: dict[str, float] = {}
    for _, r in latest_per.iterrows():
        c = str(r["code"]).zfill(6)
        prefix = "SH" if c[0] in ("6", "9") else "SZ"
        sym = f"{prefix}{c}"
        jzf = (r["open"] - r["prev_close"]) / r["prev_close"] * 100
        if pd.notna(jzf):
            jzf_map[sym] = float(jzf)

    if not jzf_map:
        return {}, "missing-no-coverage"
    source_status = (
        f"ok-{latest.date()}" if (today - latest).days == 0
        else f"ok-stale-{latest.date()}"
    )
    return jzf_map, source_status


def apply_v19_6_sidecar_overlay(
    pred_today: pd.DataFrame, today: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    """v19.6: 在 today 截面上加 amp_imb_20d overlay.
    pred_today: 至少含 [instrument, score] 列.
    返回: (含 final_score 的 DataFrame [按 final_score desc 排], meta_dict).

    缺 kline 数据: amp_z 填 0 (中性).
    若 kline cache missing 或 stale: 静默退回 baseline (final = z(pred)).
    """
    meta: dict = {
        "sidecar_applied": False, "version": "v19.6",
        "n_with_amp": 0, "n_total": len(pred_today), "kline_status": "n/a",
    }
    out = pred_today.copy()
    out["pred_z"] = (out["score"] - out["score"].mean()) / (out["score"].std(ddof=0) or 1.0)
    out["final_score"] = out["pred_z"]  # default fallback

    if not USE_V19_6_SIDECAR:
        meta["kline_status"] = "toggle-off"
        return out.sort_values("final_score", ascending=False).reset_index(drop=True), meta

    amp_map, status = load_amp_imb_20d_overlay(today)
    meta["kline_status"] = status
    if not status.startswith("ok"):
        return out.sort_values("final_score", ascending=False).reset_index(drop=True), meta

    out["amp_imb_20d"] = out["instrument"].map(amp_map)
    amp_series = out["amp_imb_20d"]
    amp_mean = amp_series.mean()
    amp_std = amp_series.std(ddof=0)
    amp_filled = amp_series.fillna(amp_mean if pd.notna(amp_mean) else 0.0)
    out["amp_z"] = (
        (amp_filled - (amp_mean if pd.notna(amp_mean) else 0.0))
        / (amp_std or 1.0)
    )

    out["final_score"] = out["pred_z"] - SIDECAR_LAMBDA_AMP_20D * out["amp_z"]

    meta["sidecar_applied"] = True
    meta["n_with_amp"] = int(out["amp_imb_20d"].notna().sum())

    # v19.10 stacked: 在 v19.6 final_score 之上 stack JZF (overnight gap, sign=+1)
    # `final = (v19.6 final) + 0.10 × z(JZF)` = `z(pred) - 0.30×z(amp) + 0.10×z(JZF)`
    meta["stacked_v19_10"] = False
    meta["lambda_jzf"] = SIDECAR_LAMBDA_JZF
    meta["jzf_status"] = "n/a"
    meta["n_with_jzf"] = 0
    if USE_V19_10_STACKED:
        jzf_map, jzf_status = load_jzf_overlay(today)
        meta["jzf_status"] = jzf_status
        if jzf_status.startswith("ok"):
            out["jzf"] = out["instrument"].map(jzf_map)
            jzf_series = out["jzf"]
            jzf_mean = jzf_series.mean()
            jzf_std = jzf_series.std(ddof=0)
            jzf_filled = jzf_series.fillna(jzf_mean if pd.notna(jzf_mean) else 0.0)
            out["jzf_z"] = (
                (jzf_filled - (jzf_mean if pd.notna(jzf_mean) else 0.0))
                / (jzf_std or 1.0)
            )
            out["final_score"] = out["final_score"] + SIDECAR_LAMBDA_JZF * out["jzf_z"]
            meta["stacked_v19_10"] = True
            meta["n_with_jzf"] = int(out["jzf"].notna().sum())

    return out.sort_values("final_score", ascending=False).reset_index(drop=True), meta


def main(dry_run: bool = False):
    # === P0-B 风控前置 gate (跑 qlib 之前 fail-fast) ===
    # drawdown < -15% 或 daily_loss < -4% → exit(2) 阻塞当日 trade
    # 一行回滚: src/claude_finance/risk/gates.py 顶部 RISK_ENABLED = False
    try:
        from claude_finance.risk.gates import (
            PortfolioRiskGate, compute_nav_series_from_log,
        )
        risk_gate = PortfolioRiskGate()
        if risk_gate.enabled:
            nav = compute_nav_series_from_log()
            print(f"[0/4] risk_gate NAV history: {len(nav)} days", flush=True)
            if len(nav) >= 2:
                nav_values = [n for _, n in nav]
                dd_res = risk_gate.check_drawdown(nav_values)
                risk_gate.audit(dd_res)
                marker = "✓" if dd_res.ok else "✗ BLOCK"
                print(f"[0/4] {marker} drawdown: {dd_res.reason}", flush=True)
                dl_res = risk_gate.check_daily_loss(nav_values[-1], nav_values[-2])
                risk_gate.audit(dl_res)
                marker = "✓" if dl_res.ok else "✗ BLOCK"
                print(f"[0/4] {marker} daily_loss: {dl_res.reason}", flush=True)
                if not dd_res.ok or not dl_res.ok:
                    print("[0/4] risk_gate BLOCKED — paper_trade aborted",
                          flush=True)
                    try:
                        from claude_finance.ws_notify import ws_notify
                        ws_notify("risk_event", {
                            "stage": "pre_paper_trade",
                            "drawdown_ok": dd_res.ok,
                            "drawdown_reason": dd_res.reason,
                            "daily_loss_ok": dl_res.ok,
                            "daily_loss_reason": dl_res.reason,
                            "dry_run": bool(dry_run),
                        })
                    except Exception:
                        pass
                    sys.exit(2)
            else:
                print(f"[0/4] risk_gate bypass (n={len(nav)} < 2)", flush=True)
        else:
            print("[0/4] risk_gate disabled (RISK_ENABLED=False)", flush=True)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        # gate 自身故障不应阻塞 production
        print(f"[0/4] risk_gate ERROR (ignored): {type(e).__name__}: {e}",
              flush=True)

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

    # v19.6 (production) / v19.4 (fallback) / v19.1 (baseline) 三档 sidecar 选择.
    today_full = pred_df[pred_df["datetime"] == today].reset_index(drop=True)
    if USE_V19_6_SIDECAR:
        overlay_df, overlay_meta = apply_v19_6_sidecar_overlay(today_full, today)
    elif USE_V19_4_SIDECAR:
        overlay_df, overlay_meta = apply_sidecar_overlay(today_full, today)
    else:
        overlay_df, overlay_meta = apply_sidecar_overlay(today_full, today)  # USE_V19_4_SIDECAR=False → 内部仅 z(pred), 等价 v19.1

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

    # === P2-opt debate veto (默认 OFF, 30 日 shadow A/B 才考虑 ON) ===
    # neutral agent 投 SELL → 移出 picks. 不 backfill, K 缩小.
    if USE_DEBATE_VETO:
        try:
            from claude_finance.debate_veto import DebateVeto
            veto = DebateVeto()
            input_picks = today_pred["instrument"].tolist()
            veto_result = veto.filter_picks(input_picks)
            if not veto_result.skipped and veto_result.n_vetoed > 0:
                kept_set = set(veto_result.kept)
                vetoed_syms = [v["sym"] for v in veto_result.vetoed]
                print(f"[debate_veto] removed {veto_result.n_vetoed} picks: "
                      f"{vetoed_syms} (source={veto_result.source_date})", flush=True)
                today_pred = today_pred[
                    today_pred["instrument"].isin(kept_set)
                ].reset_index(drop=True)
                try:
                    from claude_finance.ws_notify import ws_notify
                    ws_notify("risk_event", {
                        "stage": "debate_veto",
                        "source_date": veto_result.source_date,
                        "n_input": veto_result.total_input,
                        "n_kept": veto_result.n_kept,
                        "n_vetoed": veto_result.n_vetoed,
                        "vetoed_syms": vetoed_syms,
                    })
                except Exception:
                    pass
            elif veto_result.skipped:
                print(f"[debate_veto] skipped (no debate log for date)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[debate_veto] ERROR (ignored): {type(e).__name__}: {e}",
                  flush=True)

    closes_today = {sym: closes_today_all.get(sym, 0)
                     for sym in today_pred["instrument"].tolist()}

    # === dump picks_today.json for dashboard (2026-05-26 Option A) ===
    # Schema 含 z_pred + sidecar contrib + final, dashboard 读此 JSON 替代 cache 重算,
    # 让 dashboard Top 8 跟 production 实时 picks 100% 一致.
    try:
        if USE_V19_10_STACKED and overlay_meta.get("stacked_v19_10"):
            _pv, _factor, _lam, _sign = (
                "v19.10",
                "amp_imb_20d + JZF",
                [SIDECAR_LAMBDA_AMP_20D, SIDECAR_LAMBDA_JZF],
                [-1, +1],
            )
        elif USE_V19_6_SIDECAR and overlay_meta.get("sidecar_applied"):
            _pv, _factor, _lam, _sign = "v19.6", "amp_imb_20d", SIDECAR_LAMBDA_AMP_20D, -1
        elif USE_V19_4_SIDECAR and overlay_meta.get("sidecar_applied"):
            _pv, _factor, _lam, _sign = "v19.4", "margin_5d+20d_chg", [SIDECAR_LAMBDA_M5, SIDECAR_LAMBDA_M20], -1
        else:
            _pv, _factor, _lam, _sign = "baseline", None, None, None
        picks_json = {
            "as_of_date": str(today.date()),
            "production_version": _pv,
            "sidecar": {
                "factor": _factor, "lambda": _lam, "sign": _sign,
                "applied": bool(overlay_meta.get("sidecar_applied")),
                "stacked_v19_10": bool(overlay_meta.get("stacked_v19_10")),
                "lambda_jzf": overlay_meta.get("lambda_jzf"),
                "jzf_status": overlay_meta.get("jzf_status"),
                "n_with_jzf": int(overlay_meta.get("n_with_jzf", 0)),
            },
            "picks": [
                {
                    "sym": row["instrument"],
                    "score": float(row.get("score", 0)),
                    "z_pred": float(row.get("pred_z", 0)),
                    "z_amp": float(row["amp_z"]) if "amp_z" in row.index else None,
                    "amp_imb_20d": float(row["amp_imb_20d"]) if "amp_imb_20d" in row.index else None,
                    "z_jzf": float(row["jzf_z"]) if "jzf_z" in row.index else None,
                    "jzf": float(row["jzf"]) if "jzf" in row.index else None,
                    "m5_z": float(row["m5_z"]) if "m5_z" in row.index else None,
                    "m20_z": float(row["m20_z"]) if "m20_z" in row.index else None,
                    "final_score": float(row.get("final_score", row.get("score", 0))),
                    "close_today": float(closes_today.get(row["instrument"], 0)),
                }
                for _, row in today_pred.iterrows()
            ],
            "kline_status": overlay_meta.get("kline_status", "n/a"),
            "margin_status": overlay_meta.get("margin_status", "n/a"),
            "n_with_factor": int(overlay_meta.get("n_with_amp", overlay_meta.get("n_with_margin", 0))),
            "n_total": int(overlay_meta.get("n_total", 0)),
            "generated_at": pd.Timestamp.now().isoformat(),
        }
        # 全 universe z_pred 分布 (dashboard picks_score_distribution 用) — 仅 dump pred_z 列.
        if "pred_z" in overlay_df.columns:
            _zp = overlay_df["pred_z"].dropna().astype(float)
            picks_json["full_distribution"] = {
                "z_pred_values": [round(float(v), 6) for v in _zp.tolist()],
                "z_pred_mean": float(_zp.mean()) if len(_zp) else 0.0,
                "z_pred_std": float(_zp.std(ddof=0)) if len(_zp) else 0.0,
                "z_pred_min": float(_zp.min()) if len(_zp) else 0.0,
                "z_pred_max": float(_zp.max()) if len(_zp) else 0.0,
                "n_total": int(len(_zp)),
            }
        (ROOT / "data_cache" / "picks_today.json").write_text(
            json.dumps(picks_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"  [picks_today.json dump warn] {type(e).__name__}: {e}", flush=True)

    # === dump sandbox_factors.json — 全 universe z-scores for dashboard 因子沙盒 ===
    # 纯 additive logging, 不改任何策略逻辑.
    # 提供 z_pred / z_amp / z_jzf 三因子 + 当日 close + 名称,
    # 让 dashboard JS 滑块在浏览器内实时重算 final_score 并重排 Top 8.
    try:
        sb_names = load_universe_names()
        if USE_V19_10_STACKED and overlay_meta.get("stacked_v19_10"):
            sb_pv = "v19.10"
            sb_weights = {
                "z_pred": 1.0,
                "amp_imb_20d": -float(SIDECAR_LAMBDA_AMP_20D),
                "JZF": float(SIDECAR_LAMBDA_JZF),
            }
        elif USE_V19_6_SIDECAR and overlay_meta.get("sidecar_applied"):
            sb_pv = "v19.6"
            sb_weights = {"z_pred": 1.0, "amp_imb_20d": -float(SIDECAR_LAMBDA_AMP_20D), "JZF": 0.0}
        elif USE_V19_4_SIDECAR and overlay_meta.get("sidecar_applied"):
            sb_pv = "v19.4"
            sb_weights = {"z_pred": 1.0, "amp_imb_20d": 0.0, "JZF": 0.0}
        else:
            sb_pv = "baseline"
            sb_weights = {"z_pred": 1.0, "amp_imb_20d": 0.0, "JZF": 0.0}
        # full universe close lookup — D.features 一次 ~1-2s, 让沙盒 Top 8 也能显示 close
        try:
            sb_all_syms = overlay_df["instrument"].tolist()
            sb_prices = D.features(
                sb_all_syms, ["$close"],
                start_time=test_end, end_time=test_end, freq="day",
            )
            sb_close_map = (
                sb_prices.xs(today, level="datetime")["$close"].to_dict()
                if not sb_prices.empty else {}
            )
        except Exception:
            sb_close_map = dict(closes_today_all)
        sb_rows = []
        for _, row in overlay_df.iterrows():
            sym = row["instrument"]
            rec = {
                "sym": sym,
                "name": (sb_names.get(sym) or sb_names.get(sym.upper())
                          or sb_names.get(sym.lower()) or ""),
                "score": float(row.get("score", 0)),
                "z_pred": (float(row["pred_z"])
                            if "pred_z" in row.index and pd.notna(row.get("pred_z"))
                            else 0.0),
                "z_amp": (float(row["amp_z"])
                           if "amp_z" in row.index and pd.notna(row.get("amp_z"))
                           else 0.0),
                "amp_imb_20d": (float(row["amp_imb_20d"])
                                 if "amp_imb_20d" in row.index and pd.notna(row.get("amp_imb_20d"))
                                 else None),
                "z_jzf": (float(row["jzf_z"])
                           if "jzf_z" in row.index and pd.notna(row.get("jzf_z"))
                           else 0.0),
                "jzf": (float(row["jzf"])
                         if "jzf" in row.index and pd.notna(row.get("jzf"))
                         else None),
                "close": (float(sb_close_map[sym])
                           if sym in sb_close_map and pd.notna(sb_close_map[sym])
                           else None),
            }
            sb_rows.append(rec)
        sandbox_json = {
            "as_of_date": str(today.date()),
            "production_version": sb_pv,
            "production_weights": sb_weights,
            "factor_meta": {
                "z_pred": {
                    "sign": "+", "default_lambda": 1.0,
                    "desc": "Alpha158 + DoubleEnsemble 综合分 z-score",
                },
                "amp_imb_20d": {
                    "sign": "-", "default_lambda": 0.30,
                    "desc": "20日振幅不平衡反转 (sign=-1, 强者反转)",
                },
                "JZF": {
                    "sign": "+", "default_lambda": 0.10,
                    "desc": "集合竞价跳空率 (sign=+1, 高开多方意愿)",
                },
            },
            "k": K,
            "candidate_pool_size": int(K * CANDIDATE_POOL_MULTIPLIER),
            "max_price": MAX_AFFORDABLE_PRICE,
            "n_universe": len(sb_rows),
            "universe": sb_rows,
            "production_picks": [r["instrument"] for _, r in today_pred.iterrows()],
            "generated_at": pd.Timestamp.now().isoformat(),
        }
        (ROOT / "data_cache" / "sandbox_factors.json").write_text(
            json.dumps(sandbox_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"  [sandbox_factors.json dump warn] {type(e).__name__}: {e}", flush=True)

    # WS broadcast (fail-tolerant): 让 dashboard 收到 picks_changed 事件后局部刷新推荐栏
    try:
        from claude_finance.ws_notify import ws_notify
        ws_notify("picks_changed", {
            "as_of_date": str(today.date()),
            "production_version": (
                "v19.10" if USE_V19_10_STACKED else
                "v19.6" if USE_V19_6_SIDECAR else
                "v19.4" if USE_V19_4_SIDECAR else "baseline"
            ),
            "k": K,
            "n_picks": int(len(today_pred)),
            "sidecar_factor": (
                "amp_imb_20d+JZF" if USE_V19_10_STACKED else
                "amp_imb_20d" if USE_V19_6_SIDECAR else
                "margin_5d+20d_chg" if USE_V19_4_SIDECAR else "none"
            ),
            "dry_run": bool(dry_run),
        })
    except Exception:
        pass

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
    # 三档 sidecar 标签
    version = overlay_meta.get("version", "v19.4")  # apply_sidecar_overlay 没 version key, 默认 v19.4
    if overlay_meta["sidecar_applied"] and version == "v19.6":
        sidecar_label = (
            f"v19.6 ON (amp_imb_20d λ={SIDECAR_LAMBDA_AMP_20D})"
        )
    elif overlay_meta["sidecar_applied"]:
        sidecar_label = "v19.4 ON (margin sidecar)"
    elif USE_V19_6_SIDECAR:
        sidecar_label = f"v19.6 OFF ({overlay_meta.get('kline_status', 'n/a')}) → baseline"
    elif USE_V19_4_SIDECAR:
        sidecar_label = f"v19.4 OFF ({overlay_meta.get('margin_status', 'n/a')}) → baseline"
    else:
        sidecar_label = "OFF (v19.1 baseline)"
    print(f"Model:    qlib Alpha158 + DoubleEnsemble, K={K} drop={N_DROP}")
    print(f"Capital:  {PORTFOLIO_VALUE:.0f} 元  Vol-target: {vt_label}  Margin-sidecar: {sidecar_label}")
    print(f"Train:    {train_start} → {train_end} ({TRAIN_MONTHS}个月)")
    if overlay_meta["sidecar_applied"] and version == "v19.6":
        print(f"Sidecar:  λ_amp20={SIDECAR_LAMBDA_AMP_20D}  "
              f"covered={overlay_meta['n_with_amp']}/{overlay_meta['n_total']} "
              f"(OOS Calmar 0.79 vs v19.4 0.61, stacked v19.7 0.65 abort)")
    elif overlay_meta["sidecar_applied"]:
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
    parser = argparse.ArgumentParser(description="v19.6 paper trade signals (amp_imb_20d sidecar)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印 picks, 不写 paper_trade_log.csv / portfolio_state.json")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
