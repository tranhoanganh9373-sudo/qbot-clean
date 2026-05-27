"""Multi-agent debate veto — picks 二次过滤.

借鉴 jin-ce-zhi-suan 多 agent 决策思想, 把现有 multi_agent_debate.py 从
"展示性 panel" 升级为 "第二风控闸": neutral agent 投 SELL → veto.

设计原则:
  * 默认 OFF (USE_DEBATE_VETO=False in paper_trade_today), 30 日 shadow A/B 后才考虑 ON
  * 跟 risk_gates / phase_b_gate 一致: 全局 DEBATE_VETO_ENABLED + 实例 enabled 双开关
  * 一行回滚: DEBATE_VETO_ENABLED = False
  * 严格 OOS 协议保留: 仅做"剔除"过滤, 不动 sidecar / lambda / strategy 层

调用方式:
    from claude_finance.debate_veto import DebateVeto
    veto = DebateVeto()
    result = veto.filter_picks(["SH600346", "SZ000338", ...])
    # result.kept, result.vetoed = [{sym, votes, reason}]
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# 一行回滚 (全局)
DEBATE_VETO_ENABLED = True

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = ROOT / "data_cache" / "multi_agent_log.jsonl"
DEFAULT_AUDIT_LOG = ROOT / "data_cache" / "debate_veto_log.csv"

# 哪个 vote 触发 veto. 默认 SELL — 仅 neutral 投 SELL 才剔.
DEFAULT_VETO_VOTE = "SELL"
# 哪个 agent 的 vote 主导. neutral 是综合裁决, 不取多数票.
PRIMARY_AGENT = "neutral"

AUDIT_FIELDS = (
    "dt", "date_of_debate", "sym",
    "bull_vote", "bear_vote", "neutral_vote",
    "kept_or_vetoed", "reason",
)


@dataclass
class VetoResult:
    kept: list = field(default_factory=list)
    vetoed: list = field(default_factory=list)
    total_input: int = 0
    skipped: bool = False
    source_date: str = ""

    @property
    def n_vetoed(self) -> int:
        return len(self.vetoed)

    @property
    def n_kept(self) -> int:
        return len(self.kept)


def load_debate_votes(
    log_path: Path = DEFAULT_LOG_PATH,
    date_str: str | None = None,
):
    """读 multi_agent_log.jsonl, 返 (votes_map, source_date).

    votes_map: {sym: {bull|bear|neutral: vote}}
    date_str: 指定日期 (YYYY-MM-DD), None 则取 log 里最新日期
    """
    if not Path(log_path).exists():
        return {}, ""
    try:
        records = []
        with Path(log_path).open(encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                ts = d.get("ts", "")
                d["_date"] = ts[:10] if len(ts) >= 10 else ""
                records.append(d)
    except OSError:
        return {}, ""

    if not records:
        return {}, ""

    if date_str:
        rec_today = [r for r in records if r["_date"] == date_str]
        if not rec_today:
            return {}, ""
        source = date_str
    else:
        all_dates = sorted({r["_date"] for r in records if r["_date"]})
        if not all_dates:
            return {}, ""
        source = all_dates[-1]
        rec_today = [r for r in records if r["_date"] == source]

    votes: dict = {}
    for r in rec_today:
        sym = r.get("sym")
        agent = r.get("agent")
        vote = r.get("vote")
        if not sym or not agent:
            continue
        votes.setdefault(sym, {})[agent] = vote
    return votes, source


class DebateVeto:
    def __init__(
        self,
        log_path: Path = DEFAULT_LOG_PATH,
        audit_log: Path | None = DEFAULT_AUDIT_LOG,
        veto_vote: str = DEFAULT_VETO_VOTE,
        primary_agent: str = PRIMARY_AGENT,
        enabled: bool = True,
    ) -> None:
        self.log_path = log_path
        self.audit_log = audit_log
        self.veto_vote = veto_vote
        self.primary_agent = primary_agent
        self.enabled = enabled and DEBATE_VETO_ENABLED

    def filter_picks(
        self,
        picks: list,
        date_str: str | None = None,
    ) -> VetoResult:
        result = VetoResult(total_input=len(picks))
        if not self.enabled:
            result.kept = list(picks)
            result.skipped = True
            result.source_date = "disabled"
            return result

        votes_map, source = load_debate_votes(self.log_path, date_str)
        result.source_date = source

        if not votes_map:
            result.kept = list(picks)
            result.skipped = True
            return result

        for sym in picks:
            sym_votes = votes_map.get(sym, {})
            primary = sym_votes.get(self.primary_agent)
            if primary == self.veto_vote:
                result.vetoed.append({
                    "sym": sym,
                    "votes": sym_votes,
                    "reason": f"{self.primary_agent}={self.veto_vote}",
                })
                self._audit(source, sym, sym_votes, kept=False,
                           reason=f"{self.primary_agent}={self.veto_vote}")
            else:
                result.kept.append(sym)
                self._audit(source, sym, sym_votes, kept=True,
                           reason=f"{self.primary_agent}={primary or 'unknown'}")
        return result

    def preview(
        self,
        picks: list,
        date_str: str | None = None,
    ) -> VetoResult:
        """跟 filter_picks 一致但不写 audit — 用于 dashboard 预览."""
        saved_audit = self.audit_log
        self.audit_log = None
        try:
            return self.filter_picks(picks, date_str)
        finally:
            self.audit_log = saved_audit

    def _audit(self, source_date: str, sym: str, votes: dict,
               kept: bool, reason: str) -> None:
        if self.audit_log is None:
            return
        path = Path(self.audit_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        try:
            with path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                if new_file:
                    writer.writerow(AUDIT_FIELDS)
                writer.writerow([
                    datetime.now().isoformat(),
                    source_date,
                    sym,
                    votes.get("bull", ""),
                    votes.get("bear", ""),
                    votes.get("neutral", ""),
                    "kept" if kept else "VETOED",
                    reason,
                ])
        except OSError:
            pass


def filter_picks(picks: list, **kwargs) -> VetoResult:
    """Convenience one-shot."""
    return DebateVeto(**kwargs).filter_picks(picks)
