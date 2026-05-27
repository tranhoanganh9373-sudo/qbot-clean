"""QuantaAlpha-inspired factor quality gate (rule-based, 0 LLM cost).

Source
------
Adapted from QuantaAlpha (arXiv 2602.07085) `factors/regulator/consistency_checker.py`
subset that is *rule-based*. The original `FactorConsistencyChecker` is LLM-driven
(calls `APIBackend().build_messages_and_create_chat_completion(json_mode=True)`).
Here we extract only the structural/regex/AST validators so this module is
zero-LLM-cost and zero new external deps (stdlib `ast` + `re`, pandas/numpy
already required by the project).

Three validators
----------------
1. ``consistency_check(hypothesis, factor_expr, code)`` — checks that all
   symbols used in `factor_expr` actually appear in `code`, that the operators
   in `factor_expr` are reflected in `code`, and that the directional language
   in `hypothesis` (reversal / momentum / increase / decrease) matches the sign
   pattern of `factor_expr`. Returns ``(ok, issues)``.
2. ``complexity_score(factor_expr)`` — numeric complexity in [0, 10] derived
   from token count, operator count, nesting depth and unique-symbol count.
   Lower is simpler.
3. ``redundancy_check(new_factor, existing_panel, threshold)`` — cross-sectional
   Spearman of `new_factor` vs each column of `existing_panel` (monthly group,
   mean |rho|). Returns ``(independent, {existing_col: mean_abs_rho})``.

The thresholds default to QuantaAlpha values where they translate; Spearman
redundancy uses our project convention |rho| < 0.30 (Phase A practice).

This module is standalone — it does NOT import production code and is not wired
into paper_trade or forward_oos_monitor. Phase A factor pipelines may import it
as an optional quality gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Common OHLCV / fundamentals symbols a factor expression may reference.
_KNOWN_SYMBOLS: frozenset[str] = frozenset(
    {
        "open", "high", "low", "close", "vwap", "volume", "vol", "amount",
        "turn", "turnover",
        "pe", "pb", "ps", "roe", "roa", "net_margin", "revenue", "revenue_yoy",
        "margin", "margin_5d", "margin_20d", "amp_imb_20d",
        "ibs", "vol_z_5d", "net_buy_pct_evt",
        "unlock_pct", "unlock_pct_20", "unlock_pct_60",
    }
)

# Operator tokens recognised in factor formulae (regex-friendly).
_OPERATOR_TOKENS: tuple[str, ...] = (
    "+", "-", "*", "/", "**", "<", ">", "<=", ">=", "==", "!=",
)

# Function-like operations frequently seen in alpha expressions.
_FUNCTION_TOKENS: frozenset[str] = frozenset(
    {
        "log", "sqrt", "abs", "sign", "max", "min",
        "sum", "mean", "std", "var", "median", "rank",
        "delta", "shift", "lag", "ts_rank", "ts_mean", "ts_std",
        "corr", "cov", "z", "zscore", "winsorize", "neutralize",
    }
)

# Directional language buckets. Both `factor_expr` and `hypothesis` are scanned
# for sign cues; mismatches are reported as consistency issues.
_DIRECTION_POSITIVE: tuple[str, ...] = (
    "momentum", "动量", "上升", "上涨", "增长", "bullish", "增加", "增多",
    "扩张", "走强",
)
_DIRECTION_NEGATIVE: tuple[str, ...] = (
    "reversal", "反转", "下跌", "下降", "回落", "bearish", "减少", "走弱",
    "收缩", "回调",
)

# Default Spearman redundancy threshold (Phase A practice: < 0.30 = independent).
DEFAULT_SPEARMAN_THRESHOLD: float = 0.30

# Minimum cross-sectional obs per group before computing Spearman.
_MIN_GROUP_OBS: int = 30


# --------------------------------------------------------------------------- #
# 1. consistency_check                                                        #
# --------------------------------------------------------------------------- #

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _extract_identifiers(text: str) -> set[str]:
    """All identifier-like tokens in ``text`` (lower-cased)."""
    return {m.group(0).lower() for m in _IDENT_RE.finditer(text)}


def _extract_operators(expr: str) -> set[str]:
    """All operator tokens present in ``expr`` (longest-match first)."""
    found: set[str] = set()
    for op in sorted(_OPERATOR_TOKENS, key=len, reverse=True):
        if op in expr:
            found.add(op)
    return found


def _sign_of_expression(expr: str) -> str:
    """Coarse sign heuristic for an alpha expression.

    Heuristic: an expression with an explicit leading unary minus
    (``-expr`` or ``-(...)``) signals a *negated* / reversal direction. An
    expression that starts with a positive value or symbol signals a
    *positive* direction. We are NOT parsing full semantics — this is only
    meant to catch the obvious case where the hypothesis says "reversal" but
    the formula has no leading negation.

    Returns one of ``"positive"``, ``"negative"``, ``"neutral"``.
    """
    e = expr.strip()
    if not e:
        return "neutral"
    if e.startswith("-"):
        return "negative"
    # Any other leading token (digit, identifier, opening paren) signals a
    # positive-direction formula. A `(` is treated as positive because we
    # then look inside for a leading `-`.
    if e.startswith("("):
        inner = e[1:].lstrip()
        if inner.startswith("-"):
            return "negative"
    return "positive"


def _direction_in_text(text: str) -> str:
    """Coarse direction inferred from natural-language hypothesis."""
    t = text.lower()
    pos = sum(1 for w in _DIRECTION_POSITIVE if w in t)
    neg = sum(1 for w in _DIRECTION_NEGATIVE if w in t)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def consistency_check(
    hypothesis: str, factor_expr: str, code: str
) -> tuple[bool, list[str]]:
    """Check that hypothesis, factor expression and code agree.

    Three rules:
      R1. Every identifier in `factor_expr` that looks like a known data symbol
          (close/high/...) must also appear in `code`.
      R2. Every operator token in `factor_expr` must appear in `code`
          (or its function equivalent, e.g. ``mean`` ↔ ``.mean()``).
      R3. If `hypothesis` carries a clear directional cue (reversal / momentum
          etc.), the factor expression sign should not contradict it.

    Returns
    -------
    (ok, issues)
        ``ok=True`` iff no issues. ``issues`` is a list of human-readable
        strings (empty when ok).
    """
    issues: list[str] = []

    expr_idents = _extract_identifiers(factor_expr)
    code_idents = _extract_identifiers(code)

    # R1: data symbols referenced in expression must appear in code.
    missing_symbols = sorted(
        sym for sym in expr_idents
        if sym in _KNOWN_SYMBOLS and sym not in code_idents
    )
    if missing_symbols:
        issues.append(
            "R1 symbol mismatch: expression uses "
            f"{missing_symbols} but they do not appear in code"
        )

    # R2: operators in expression should appear (or be implied) in code.
    expr_ops = _extract_operators(factor_expr)
    code_ops = _extract_operators(code)
    missing_ops = sorted(expr_ops - code_ops)
    if missing_ops:
        issues.append(
            f"R2 operator mismatch: expression uses {missing_ops} "
            "but code does not"
        )

    # R2b: function-like tokens in expression (e.g. ``mean``, ``log``) should
    # appear in code either as call or attribute.
    expr_fn = expr_idents & _FUNCTION_TOKENS
    missing_fn = sorted(fn for fn in expr_fn if fn not in code_idents)
    if missing_fn:
        issues.append(
            f"R2 function mismatch: expression uses {sorted(expr_fn)} "
            f"but code does not reference {missing_fn}"
        )

    # R3: directional consistency.
    direction = _direction_in_text(hypothesis)
    expr_sign = _sign_of_expression(factor_expr)
    if direction != "neutral" and expr_sign != "neutral" and direction != expr_sign:
        issues.append(
            f"R3 direction mismatch: hypothesis implies {direction!r} "
            f"but expression sign is {expr_sign!r}"
        )

    return len(issues) == 0, issues


# --------------------------------------------------------------------------- #
# 2. complexity_score                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ComplexityBreakdown:
    """Per-component complexity contributions for transparency."""

    token_count: int
    operator_count: int
    nesting_depth: int
    unique_symbols: int
    score: float


def _nesting_depth(expr: str) -> int:
    """Max parenthesis-nesting depth in ``expr``."""
    depth = max_depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return max_depth


def _operator_count(expr: str) -> int:
    """Count of arithmetic/comparison operators in ``expr``.

    Longest-match first to avoid double-counting ``**`` as two ``*``.
    """
    count = 0
    work = expr
    for op in sorted(_OPERATOR_TOKENS, key=len, reverse=True):
        count += work.count(op)
        work = work.replace(op, " ")
    # Function-like operators each count once per occurrence.
    for fn in _FUNCTION_TOKENS:
        count += len(re.findall(rf"\b{re.escape(fn)}\b", expr))
    return count


def complexity_breakdown(factor_expr: str) -> ComplexityBreakdown:
    """Return the four components plus the aggregated score."""
    tokens = _IDENT_RE.findall(factor_expr)
    token_count = len(tokens)
    op_count = _operator_count(factor_expr)
    depth = _nesting_depth(factor_expr)
    uniq = len({t.lower() for t in tokens})

    # Score in [0, 10] — bounded so callers can threshold safely. Each
    # component contributes up to ~2.5 points; we cap to keep noise bounded.
    score = (
        min(token_count, 20) * 0.125     # 20 tokens → 2.5
        + min(op_count, 10) * 0.25       # 10 ops → 2.5
        + min(depth, 5) * 0.5            # depth 5 → 2.5
        + min(uniq, 10) * 0.25           # 10 unique → 2.5
    )
    return ComplexityBreakdown(
        token_count=token_count,
        operator_count=op_count,
        nesting_depth=depth,
        unique_symbols=uniq,
        score=round(float(score), 3),
    )


def complexity_score(factor_expr: str) -> float:
    """Aggregated complexity score in [0, 10]. Lower is simpler.

    Threshold guidance (QuantaAlpha-aligned but with our scale):
      * 0-3   simple    (e.g. ``close / low``)
      * 4-6   medium    (e.g. ``(close - low) / (high - low)``)
      * 7-10  complex   (high overfit risk)
    """
    return complexity_breakdown(factor_expr).score


# --------------------------------------------------------------------------- #
# 3. redundancy_check                                                         #
# --------------------------------------------------------------------------- #


def _monthly_groupby_key(index: pd.MultiIndex) -> pd.Series:
    """Build a 'month_end' grouping key from the (code, date) MultiIndex."""
    # Index level 1 is expected to be the date.
    dates = pd.to_datetime(index.get_level_values(-1))
    # Month-end period → Timestamp for stable groupby.
    return dates.to_period("M").to_timestamp("M")


def _mean_abs_spearman(
    new_factor: pd.Series, existing: pd.Series
) -> float:
    """Mean |Spearman rho| across monthly cross-sections.

    Both inputs must share an index aligned at (code, date); rows where either
    side is NaN are dropped. Returns ``nan`` if no month meets ``_MIN_GROUP_OBS``.
    """
    df = pd.concat(
        [new_factor.rename("a"), existing.rename("b")], axis=1
    ).dropna()
    if df.empty:
        return float("nan")

    month = _monthly_groupby_key(df.index)
    df = df.assign(_m=month)

    def _r(g: pd.DataFrame) -> float:
        if len(g) < _MIN_GROUP_OBS:
            return np.nan
        if g["a"].std() <= 0 or g["b"].std() <= 0:
            return np.nan
        return g["a"].corr(g["b"], method="spearman")

    rho = df.groupby("_m")[["a", "b"]].apply(_r).dropna()
    if rho.empty:
        return float("nan")
    return float(rho.abs().mean())


def redundancy_check(
    new_factor: pd.Series,
    existing_panel: pd.DataFrame,
    threshold: float = DEFAULT_SPEARMAN_THRESHOLD,
) -> tuple[bool, dict[str, float]]:
    """Test whether ``new_factor`` is independent of every column of ``existing_panel``.

    Parameters
    ----------
    new_factor
        ``pd.Series`` indexed by ``(code, date)`` MultiIndex, float values.
    existing_panel
        ``pd.DataFrame`` with the same MultiIndex; each column is an existing
        factor.
    threshold
        Maximum permissible mean |Spearman rho| (cross-sectional monthly).
        Defaults to 0.30 (Phase A convention).

    Returns
    -------
    (independent, rho_map)
        ``independent=True`` iff every column has |rho| < threshold.
        ``rho_map`` reports the mean |rho| per existing column.
    """
    if not isinstance(new_factor, pd.Series):
        raise TypeError("new_factor must be a pandas Series")
    if not isinstance(existing_panel, pd.DataFrame):
        raise TypeError("existing_panel must be a pandas DataFrame")

    rho_map: dict[str, float] = {}
    for col in existing_panel.columns:
        rho_map[str(col)] = _mean_abs_spearman(new_factor, existing_panel[col])

    finite = [v for v in rho_map.values() if np.isfinite(v)]
    if not finite:
        # No comparable months → cannot conclude independence; treat as not
        # independent to fail safe, but caller can decide based on NaN map.
        return False, rho_map
    independent = max(finite) < threshold
    return independent, rho_map


# --------------------------------------------------------------------------- #
# Self-test (python -m claude_finance.factor_quality_gate)                    #
# --------------------------------------------------------------------------- #


def _self_test() -> None:
    """Smoke demonstration with three realistic factors from the project."""
    print("=== consistency_check ===")

    # Case 1: IBS — high IBS → mean reversion (close near high should fall).
    ibs_hyp = "高 IBS 反转: 收盘价接近日内高点后倾向下跌"
    ibs_expr = "-(close - low) / (high - low)"  # 负号体现反转 (negative sign)
    ibs_code = (
        "def ibs(df):\n"
        "    return -(df['close'] - df['low']) / (df['high'] - df['low'])"
    )
    ok, issues = consistency_check(ibs_hyp, ibs_expr, ibs_code)
    print(f"  IBS reversal: ok={ok}, issues={issues}")

    # Case 2: shareholders — fewer holders → concentration up → bullish.
    sh_hyp = "户数减少 → 集中度上升 → bullish 增长"
    sh_expr = "shareholders_t_minus_1 - shareholders_t"
    sh_code = (
        "def delta_holders(df):\n"
        "    return df['shareholders_t_minus_1'] - df['shareholders_t']"
    )
    ok, issues = consistency_check(sh_hyp, sh_expr, sh_code)
    print(f"  shareholders: ok={ok}, issues={issues}")

    # Case 3: deliberate mismatch — hypothesis says reversal but expr is momentum.
    bad_hyp = "高 IBS 反转: 应当下跌"
    bad_expr = "(close - low) / (high - low)"  # 无负号 = 动量方向
    bad_code = "def x(df): return (df['close'] - df['low']) / (df['high'] - df['low'])"
    ok, issues = consistency_check(bad_hyp, bad_expr, bad_code)
    print(f"  mismatch (expect FAIL): ok={ok}, issues={issues}")

    print()
    print("=== complexity_score ===")
    for label, expr in [
        ("simple   close/low", "close / low"),
        ("medium   IBS",       "(close - low) / (high - low)"),
        ("complex  amp_imb",   "((high - low) - (close - open)) / (high - low + 1e-9)"),
    ]:
        br = complexity_breakdown(expr)
        print(
            f"  {label:30s} score={br.score:.2f} "
            f"tokens={br.token_count} ops={br.operator_count} "
            f"depth={br.nesting_depth} uniq={br.unique_symbols}"
        )

    print()
    print("=== redundancy_check (synthetic) ===")
    rng = np.random.default_rng(42)
    codes = [f"SH60000{i}" for i in range(50)]
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    idx = pd.MultiIndex.from_product([codes, dates], names=["code", "date"])
    n = len(idx)
    base = pd.Series(rng.standard_normal(n), index=idx, name="amp_imb_20d")
    # Independent factor: pure noise.
    indep = pd.Series(rng.standard_normal(n), index=idx, name="noise")
    # Correlated factor: 0.9 * base + small noise.
    corr = pd.Series(
        0.9 * base.values + 0.1 * rng.standard_normal(n),
        index=idx, name="copy_of_amp",
    )
    panel = pd.DataFrame({"amp_imb_20d": base})
    ok1, rho1 = redundancy_check(indep, panel)
    ok2, rho2 = redundancy_check(corr, panel)
    print(f"  noise vs amp_imb_20d:       independent={ok1}, rho={rho1}")
    print(f"  copy_of_amp vs amp_imb_20d: independent={ok2}, rho={rho2}")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
