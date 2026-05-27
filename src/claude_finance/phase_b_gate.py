"""Phase B 候选自动 gate.

借鉴 jin-ce-zhi-suan AnalysisAgent → prompt_context_patch 反馈回路, 但**只做
reject gate, 不喂 LLM 自动生成** (claude_finance 严格 OOS 协议保留).

输入: Candidate(name, n_months, is_calmar, lambda_locked, lambda_max,
              spearman_max_abs, icir).
输出: CheckResult(pass_overall, fired_modes[], severities{}, reasons{}).

规则源: data_cache/phase_b_failure_modes.json (P1.1).

一行回滚: 设 PHASE_B_GATE_ENABLED = False, 任何 candidate 直接 pass.

CLI: python -m claude_finance.phase_b_gate --help
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# 一行回滚
PHASE_B_GATE_ENABLED = True

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODES_PATH = ROOT / "data_cache" / "phase_b_failure_modes.json"
DEFAULT_CHECK_LOG = ROOT / "data_cache" / "phase_b_check_log.csv"

CHECK_LOG_FIELDS = (
    "dt", "candidate", "n_months", "is_calmar",
    "lambda_locked", "lambda_max", "icir", "spearman_max_abs",
    "pass_overall", "fired_modes", "rejects", "warns",
)


@dataclass
class Candidate:
    """Phase B 候选规格. 派生属性 (icir_abs / lambda_at_max) 用 @property 自动算."""
    name: str
    n_months: int
    is_calmar: float
    lambda_locked: object = 0.0   # float | list[float]
    lambda_max: object = None      # float | list[float] | None
    spearman_max_abs: float = 0.0
    icir: float = 0.0              # signed
    notes: str = ""

    @property
    def icir_abs(self) -> float:
        return abs(self.icir)

    @property
    def lambda_at_max(self) -> bool:
        if self.lambda_max is None:
            return False
        a, b = self.lambda_locked, self.lambda_max
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return list(a) == list(b)
        return a == b


@dataclass
class CheckResult:
    pass_overall: bool
    fired_modes: list = field(default_factory=list)
    severities: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)

    @property
    def rejects(self) -> list:
        return [m for m, s in self.severities.items() if s == "reject"]

    @property
    def warns(self) -> list:
        return [m for m, s in self.severities.items() if s == "warn"]

    def summary(self) -> str:
        if not self.fired_modes:
            return "PASS (no fired modes)"
        parts = []
        if self.rejects:
            parts.append(f"REJECT ({len(self.rejects)}): {', '.join(self.rejects)}")
        if self.warns:
            parts.append(f"WARN ({len(self.warns)}): {', '.join(self.warns)}")
        return " | ".join(parts)


def _get_value(candidate: Candidate, key: str):
    if hasattr(candidate, key):
        return getattr(candidate, key)
    return None


def _apply_op(val, op: str, threshold) -> bool:
    if val is None:
        return False
    try:
        if op == "<":  return val < threshold
        if op == "<=": return val <= threshold
        if op == ">":  return val > threshold
        if op == ">=": return val >= threshold
        if op == "==": return val == threshold
        if op == "!=": return val != threshold
    except TypeError:
        return False
    return False


class PhaseBGate:
    def __init__(
        self,
        modes_path: Path | None = None,
        audit_log: Path | None = DEFAULT_CHECK_LOG,
        enabled: bool = True,
    ) -> None:
        self.modes_path = Path(modes_path) if modes_path else DEFAULT_MODES_PATH
        self.audit_log = audit_log
        self.enabled = enabled and PHASE_B_GATE_ENABLED
        self._modes: list | None = None
        self._thresholds: dict = {}

    def _load(self) -> None:
        if self._modes is not None:
            return
        with self.modes_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        self._modes = data.get("failure_modes", [])
        self._thresholds = data.get("thresholds", {})

    def check(self, candidate: Candidate) -> CheckResult:
        if not self.enabled:
            return CheckResult(True, [], {}, {})
        self._load()
        fired = []
        severities = {}
        reasons = {}
        for mode in self._modes or []:
            mode_id = mode["id"]
            rules = mode.get("rules", [])
            logic = mode.get("logic", "all")
            severity = mode.get("severity", "reject")
            results = []
            for rule in rules:
                key = rule.get("key")
                op = rule.get("op", "==")
                threshold = rule.get("threshold")
                val = _get_value(candidate, key)
                results.append(_apply_op(val, op, threshold))
            triggered = all(results) if logic == "all" else any(results)
            if triggered:
                fired.append(mode_id)
                severities[mode_id] = severity
                reasons[mode_id] = mode.get("description", "")
        pass_overall = not any(s == "reject" for s in severities.values())
        result = CheckResult(pass_overall, fired, severities, reasons)
        self._audit(candidate, result)
        return result

    def _audit(self, candidate: Candidate, result: CheckResult) -> None:
        if self.audit_log is None:
            return
        path = Path(self.audit_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        try:
            with path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                if new_file:
                    writer.writerow(CHECK_LOG_FIELDS)
                writer.writerow([
                    datetime.now().isoformat(),
                    candidate.name,
                    candidate.n_months,
                    f"{candidate.is_calmar:.4f}",
                    json.dumps(candidate.lambda_locked, ensure_ascii=False),
                    json.dumps(candidate.lambda_max, ensure_ascii=False),
                    f"{candidate.icir:.4f}",
                    f"{candidate.spearman_max_abs:.4f}",
                    int(result.pass_overall),
                    ",".join(result.fired_modes),
                    ",".join(result.rejects),
                    ",".join(result.warns),
                ])
        except OSError:
            pass


def check_candidate(candidate: Candidate, **kwargs) -> CheckResult:
    """Convenience: 单候选 one-shot."""
    return PhaseBGate(**kwargs).check(candidate)


# ============ CLI ============

def _parse_lambda(s: str):
    if "," in s:
        return [float(x) for x in s.split(",")]
    return float(s)


def _cli_load_candidate(args) -> Candidate:
    return Candidate(
        name=args.name,
        n_months=args.n_months,
        is_calmar=args.is_calmar,
        lambda_locked=_parse_lambda(args.lambda_locked),
        lambda_max=_parse_lambda(args.lambda_max) if args.lambda_max else None,
        spearman_max_abs=args.spearman or 0.0,
        icir=args.icir or 0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B candidate gate")
    parser.add_argument("--name", required=True, help="候选名 e.g. v19.10")
    parser.add_argument("--n-months", type=int, required=True, dest="n_months",
                       help="Phase A 实际 IC 月数")
    parser.add_argument("--is-calmar", type=float, required=True, dest="is_calmar",
                       help="IS sweep 锁定后 IS Calmar")
    parser.add_argument("--lambda-locked", required=True, dest="lambda_locked",
                       help="锁定 lambda, 单值 '0.30' 或多值 '0.30,0.10'")
    parser.add_argument("--lambda-max", dest="lambda_max", default=None,
                       help="sweep 最大候选 (检 lambda_at_max)")
    parser.add_argument("--icir", type=float, default=0.0,
                       help="Phase A IC IR (signed)")
    parser.add_argument("--spearman", type=float, default=0.0,
                       help="跟 production 因子 Spearman |rho| max")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cand = _cli_load_candidate(args)
    gate = PhaseBGate()
    result = gate.check(cand)
    if not args.quiet:
        print(f"[phase_b_gate] candidate: {cand.name}", flush=True)
        print(f"  n_months={cand.n_months} is_calmar={cand.is_calmar} "
              f"icir={cand.icir} lambda_locked={cand.lambda_locked} "
              f"lambda_max={cand.lambda_max} spearman_max_abs={cand.spearman_max_abs}",
              flush=True)
        print(f"  result: {result.summary()}", flush=True)
        for m_id in result.fired_modes:
            print(f"  - {m_id} [{result.severities[m_id]}]: "
                  f"{result.reasons.get(m_id, '')}", flush=True)
    return 0 if result.pass_overall else 2


if __name__ == "__main__":
    sys.exit(main())
