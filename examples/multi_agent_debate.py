"""Multi-Agent Debate 推理框架 (MVP, rule-based + LLM upgrade hook).

3 个 agents 对今日 paper_trade picks 出意见 + 投票:
  - 🐂 Bull (看多): 找入场理由
  - 🐻 Bear (看空): 找风险信号
  - ⚖️ Neutral (中性): 综合 vote (BUY / HOLD / SELL)

数据源 (全只读):
  - data_cache/picks_today.json (paper_trade picks 8 只)
  - data_cache/baidu_kline.parquet (ma5/ma20 趋势)
  - data_cache/portfolio.xlsx Positions (推荐价 / 止损价)

输出: data_cache/multi_agent_log.jsonl
  每行: {agent, ts, sym, name, msg, vote, score}
  - agent: "bull" | "bear" | "neutral"
  - ts: ISO8601 UTC
  - sym: SH600547
  - msg: 中文讨论文本
  - vote: "BUY" | "SELL" | "HOLD"
  - score: 0.0~1.0 信心度

未来 LLM 升级 hook:
  替换 _bull_message / _bear_message / _neutral_message 为 LLM call
  (e.g. anthropic Haiku, prompt 含 metrics + 出意见 message).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PICKS_PATH = ROOT / "data_cache" / "picks_today.json"
KLINE_PATH = ROOT / "data_cache" / "baidu_kline.parquet"
XLSX_PATH = ROOT / "data_cache" / "portfolio.xlsx"
OUT_PATH = ROOT / "data_cache" / "multi_agent_log.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_picks() -> list[dict]:
    if not PICKS_PATH.exists():
        return []
    with PICKS_PATH.open(encoding="utf-8") as f:
        d = json.load(f)
    return d.get("picks", [])


def _load_kline_metrics(syms: list[str]) -> dict[str, dict]:
    if not KLINE_PATH.exists() or not syms:
        return {}
    codes = {s[2:] for s in syms if len(s) == 8}
    df = pd.read_parquet(KLINE_PATH, columns=["code", "date", "close", "vol"])
    df = df[df["code"].isin(codes)].sort_values(["code", "date"])
    out: dict[str, dict] = {}
    for code, sub in df.groupby("code", sort=False):
        if len(sub) < 30:
            continue
        c = sub["close"].tail(30).reset_index(drop=True)
        v = sub["vol"].tail(30).reset_index(drop=True)
        ma5 = float(c.tail(5).mean())
        ma20 = float(c.tail(20).mean())
        last_close = float(c.iloc[-1])
        prev_close = float(c.iloc[-2]) if len(c) >= 2 else last_close
        ret_1d = (last_close / prev_close - 1) if prev_close > 0 else 0
        vol_recent_avg = float(v.tail(5).mean())
        vol_prev_avg = float(v.head(20).mean())
        vol_ratio = vol_recent_avg / vol_prev_avg if vol_prev_avg > 0 else 1
        prefix = "SH" if str(code)[0] in ("6", "9") else "SZ"
        out[f"{prefix}{code}"] = {
            "ma5": ma5, "ma20": ma20, "last_close": last_close,
            "ret_1d": ret_1d, "vol_ratio": vol_ratio,
        }
    return out


def _load_picks_meta() -> dict[str, dict]:
    if not XLSX_PATH.exists():
        return {}
    try:
        df = pd.read_excel(XLSX_PATH, sheet_name="Positions")
    except Exception:
        return {}
    out: dict[str, dict] = {}
    if "代码" not in df.columns:
        return {}
    for _, r in df.iterrows():
        sym = str(r.get("代码") or "")
        if not sym:
            continue
        out[sym] = {
            "name": str(r.get("名称") or ""),
            "rec_price": r.get("推荐价"),
            "stop_loss": r.get("止损价(-8%)"),
        }
    return out


# ──────────────────── 3 个 Agent 决策逻辑 ────────────────────


def _bull_message(pick: dict, metrics: dict, meta: dict) -> dict:
    """🐂 Bull 看多 agent — 找入场理由."""
    sym = pick["sym"]
    score = float(pick.get("final_score") or 0)
    z_pred = float(pick.get("z_pred") or 0)
    ma5 = metrics.get("ma5", 0)
    ma20 = metrics.get("ma20", 0)
    last = metrics.get("last_close", 0)
    vol_ratio = metrics.get("vol_ratio", 1)
    rec_price = meta.get("rec_price")

    points: list[str] = []
    confidence = 0.4
    if z_pred > 2:
        points.append(f"模型 z_pred={z_pred:.2f} 强信号 (>2σ)")
        confidence += 0.20
    if ma5 > ma20:
        points.append(f"短均线在上 (MA5={ma5:.2f} > MA20={ma20:.2f}) — 趋势向上")
        confidence += 0.15
    if vol_ratio > 1.3:
        points.append(f"成交量放大 ({vol_ratio:.2f}× 近 20 日均量) — 资金关注")
        confidence += 0.10
    if rec_price and isinstance(rec_price, (int, float)) and last < rec_price * 1.02:
        points.append(f"当前 {last:.2f} 接近推荐价 {rec_price:.2f} — 入场点合适")
        confidence += 0.10
    if not points:
        points.append(f"final_score={score:.2f}, 模型整体看多")
    msg = " · ".join(points)
    return {
        "agent": "bull", "ts": _now_iso(), "sym": sym,
        "name": meta.get("name", ""), "msg": "🐂 " + msg,
        "vote": "BUY", "score": round(min(confidence, 1.0), 2),
    }


def _bear_message(pick: dict, metrics: dict, meta: dict) -> dict:
    """🐻 Bear 看空 agent — 找风险信号."""
    sym = pick["sym"]
    z_amp = float(pick.get("z_amp") or 0)
    amp_imb = float(pick.get("amp_imb_20d") or 0)
    ma5 = metrics.get("ma5", 0)
    ma20 = metrics.get("ma20", 0)
    last = metrics.get("last_close", 0)
    ret_1d = metrics.get("ret_1d", 0)
    vol_ratio = metrics.get("vol_ratio", 1)
    stop = meta.get("stop_loss")

    points: list[str] = []
    confidence = 0.3
    if z_amp < -1.5:
        points.append(f"振幅不对称 z_amp={z_amp:.2f} 偏低 — 涨势已弱")
        confidence += 0.15
    if amp_imb < -0.015:
        points.append(f"20 日振幅不对称 {amp_imb*100:.1f}% — 下跌振幅占主导")
        confidence += 0.15
    if ma5 < ma20:
        points.append(f"短均线下穿 (MA5={ma5:.2f} < MA20={ma20:.2f}) — 短期偏弱")
        confidence += 0.15
    if ret_1d < -0.03:
        points.append(f"昨日跌幅 {ret_1d*100:+.2f}% — 短期动量负")
        confidence += 0.10
    if stop and isinstance(stop, (int, float)) and last < stop * 1.05:
        points.append(f"当前 {last:.2f} 距止损 {stop:.2f} 仅 {(last/stop-1)*100:+.1f}% — 风险大")
        confidence += 0.20
    if vol_ratio < 0.7:
        points.append(f"成交量萎缩 ({vol_ratio:.2f}× 近 20 日均量) — 关注度下降")
        confidence += 0.10
    if not points:
        points.append("无明显风险信号, 但短期波动需警惕")
        confidence = 0.20
    msg = " · ".join(points)
    vote = "SELL" if confidence > 0.55 else "HOLD"
    return {
        "agent": "bear", "ts": _now_iso(), "sym": sym,
        "name": meta.get("name", ""), "msg": "🐻 " + msg,
        "vote": vote, "score": round(min(confidence, 1.0), 2),
    }


def _neutral_message(pick: dict, bull: dict, bear: dict, metrics: dict, meta: dict) -> dict:
    """⚖️ Neutral 综合判断."""
    sym = pick["sym"]
    b_conf = bull["score"]
    s_conf = bear["score"]
    if b_conf > s_conf + 0.15:
        vote = "BUY"
        verdict = f"看多胜出 (bull {b_conf:.2f} > bear {s_conf:.2f})"
    elif s_conf > b_conf + 0.15:
        vote = "SELL"
        verdict = f"看空胜出 (bear {s_conf:.2f} > bull {b_conf:.2f})"
    else:
        vote = "HOLD"
        verdict = f"分歧 (bull {b_conf:.2f} vs bear {s_conf:.2f}) — 观望"
    final_score = float(pick.get("final_score") or 0)
    msg = f"综合: {verdict}. 模型 final_score={final_score:.2f}, 建议 {vote}."
    return {
        "agent": "neutral", "ts": _now_iso(), "sym": sym,
        "name": meta.get("name", ""), "msg": "⚖️ " + msg,
        "vote": vote, "score": round(abs(b_conf - s_conf), 2),
    }


def run_debate() -> int:
    picks = _load_picks()
    if not picks:
        print("[debate] picks_today.json 为空 — abort")
        return 0
    syms = [p["sym"] for p in picks]
    metrics = _load_kline_metrics(syms)
    meta_all = _load_picks_meta()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        n_lines = 0
        for pick in picks:
            sym = pick["sym"]
            m = metrics.get(sym, {})
            meta = meta_all.get(sym, {})
            bull = _bull_message(pick, m, meta)
            bear = _bear_message(pick, m, meta)
            neutral = _neutral_message(pick, bull, bear, m, meta)
            for line in (bull, bear, neutral):
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                n_lines += 1
    print(f"[debate] wrote {n_lines} lines → {OUT_PATH}")
    print(f"  agents=3, picks={len(picks)}")
    print("\n--- sample (first pick, 3 agents) ---")
    with OUT_PATH.open() as fh:
        for line in [fh.readline() for _ in range(3)]:
            d = json.loads(line)
            print(f"  [{d['agent']:7s}] {d['sym']} {d['name']:8s} → "
                  f"{d['vote']:4s} ({d['score']:.2f})")
            print(f"             {d['msg']}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    return run_debate()


if __name__ == "__main__":
    raise SystemExit(main() or 0)
