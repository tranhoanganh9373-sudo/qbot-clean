# Factor Quality Gate (QuantaAlpha rule-based subset)

**Status**: helper module, standalone, NOT wired into production.
**Module**: `src/claude_finance/factor_quality_gate.py`
**Tests**: `tests/test_factor_quality_gate.py` (13/13 PASS)
**Date adapted**: 2026-05-26
**LOC**: ~410 (incl. self-test) — three public validators + one breakdown dataclass.

---

## 1. Source attribution

Adapted from the QuantaAlpha repo:
`references/QuantaAlpha/quantaalpha/factors/regulator/consistency_checker.py`

The upstream file contains four classes:
- `FactorConsistencyChecker` — **LLM-driven** (calls `APIBackend().build_messages_and_create_chat_completion(json_mode=True)`).
- `ComplexityChecker` — **rule-based** but depends on QuantaAlpha's pyparsing-based `factor_ast` helpers (`calculate_symbol_length`, `count_base_features`, `count_free_args`, `count_all_nodes`).
- `RedundancyChecker` — **AST-subtree-match** based, delegates to `FactorRegulator` and the project's "factor zoo" JSON cache.
- `FactorQualityGate` — orchestrator that runs the three above.

We extract **only the rule-based ideas**, drop the LLM dependency, drop the pyparsing AST (heavy + repo-specific), and substitute Spearman redundancy (the convention already used in our Phase A pipelines, e.g. `examples/factor_ic_ibs_csi300_is.py:361 spearman_orth`). The result has zero LLM cost and zero new external dependencies.

---

## 2. Public API

### `consistency_check(hypothesis, factor_expr, code) -> (ok, issues)`

Three rules:

| Rule | What it checks |
|------|----------------|
| R1   | Every known data symbol (`close`, `high`, `volume`, `pe`, …) used in `factor_expr` must also appear in `code`. |
| R2   | Operators (`+`, `-`, `*`, `/`, `**`, comparisons) and named functions (`mean`, `std`, `log`, …) used in `factor_expr` must appear in `code`. |
| R3   | If `hypothesis` carries a directional cue (`反转`/`reversal` → negative, `动量`/`bullish`/`增长` → positive), the factor expression's leading sign must not contradict it. |

Returns `(True, [])` when all three rules pass. Returns `(False, [issue, …])` listing every rule that failed, e.g. `"R3 direction mismatch: hypothesis implies 'negative' but expression sign is 'positive'"`.

### `complexity_score(factor_expr) -> float`

Aggregated complexity in **[0, 10]**, lower is simpler. The four components (each capped, then weighted to ~2.5 max):

| Component        | Cap | Weight | Source                              |
|------------------|-----|--------|-------------------------------------|
| token_count      | 20  | 0.125  | identifier-like substrings          |
| operator_count   | 10  | 0.25   | arithmetic/comparison + function names |
| nesting_depth    | 5   | 0.5    | max parenthesis depth               |
| unique_symbols   | 10  | 0.25   | distinct identifiers                |

`complexity_breakdown(...)` returns the underlying four components plus the aggregated score, for inspection.

### `redundancy_check(new_factor, existing_panel, threshold=0.30) -> (independent, rho_map)`

- `new_factor`: `pd.Series` indexed by `(code, date)` MultiIndex.
- `existing_panel`: `pd.DataFrame` with the same MultiIndex; each column is an existing factor.
- For each existing column we compute the **mean |Spearman ρ| across monthly cross-sections** (min 30 obs per month), the same convention as `spearman_orth` in `examples/factor_ic_ibs_csi300_is.py`.
- `independent=True` iff every column has `|ρ| < threshold`.
- If all comparisons return NaN (no overlapping months meeting the obs threshold), we fail-safe to `independent=False` and the caller can inspect `rho_map` to decide.

---

## 3. Phase A integration sketch

```python
from claude_finance.factor_quality_gate import (
    consistency_check,
    complexity_breakdown,
    redundancy_check,
)

hypothesis = "高 IBS 反转: 收盘价接近日内高点后倾向下跌"
expr       = "-(close - low) / (high - low)"
code       = "def ibs(df): return -(df['close'] - df['low']) / (df['high'] - df['low'])"

ok, issues = consistency_check(hypothesis, expr, code)
if not ok:
    raise RuntimeError(f"Hypothesis ↔ formula ↔ code mismatch: {issues}")

br = complexity_breakdown(expr)
if br.score > 6:
    print(f"WARNING: complexity {br.score:.2f} > 6 — possible overfit risk")

# After computing the new factor series:
ok, rho = redundancy_check(new_factor, existing_panel, threshold=0.30)
if not ok:
    print(f"Redundant with existing factors: {rho}")
```

The intended call site is **Phase A** scripts (`examples/factor_ic_*_is.py`) — after the factor panel is built and the IC is computed, but before deciding whether to promote the factor to sidecar sweep. No production code (`paper_trade_today.py`, `forward_oos_monitor.py`, `portfolio_excel_forward_oos.py`) imports this module.

---

## 4. Threshold guidance

| Validator         | Threshold     | Action                                                 |
|-------------------|---------------|--------------------------------------------------------|
| `consistency_check` | 0 issues    | block factor on any issue; rewrite hypothesis or expression |
| `complexity_score`  | ≤ 3 (simple), ≤ 6 (medium), > 7 risky | log a warning above 6, manual review above 7 |
| `redundancy_check`  | mean \|ρ\| < 0.30 | block when above; consider stacking only when \|ρ\| < 0.10 (see Phase 4 v19.7 stacked sidecar learning) |

The redundancy threshold mirrors the Phase A convention in our project. The 0.10 stacking guard is a **necessary-but-not-sufficient** condition (see `project_phase4_sidecar_v7_stacked.md` memory).

---

## 5. Difference vs the existing Phase A Spearman check

The Phase A scripts each carry a local `spearman_orth(panel, factor_col, ref_col)` (e.g. `examples/factor_ic_ibs_csi300_is.py:361`). The new module:

1. Operates on `pd.Series` / `pd.DataFrame` directly (no need to pre-pivot into a `month_end` column).
2. Iterates all columns of the existing panel automatically, returning `dict[name, ρ]`.
3. Returns a boolean independence verdict against a configurable threshold.
4. Lives in `src/` so future scripts can `from claude_finance.factor_quality_gate import redundancy_check` instead of copy-pasting.

The two implementations agree numerically — they share the same monthly-Spearman / min-30-obs definition.

---

## 6. Self-test output (2026-05-26)

```
=== consistency_check ===
  IBS reversal:        ok=True,  issues=[]
  shareholders:        ok=True,  issues=[]
  mismatch (expect F): ok=False, issues=["R3 direction mismatch: hypothesis implies 'negative' but expression sign is 'positive'"]

=== complexity_score ===
  simple   close/low             score=1.00  tokens=2 ops=1 depth=0 uniq=2
  medium   IBS                   score=2.50  tokens=4 ops=3 depth=1 uniq=3
  complex  amp_imb               score=4.88  tokens=7 ops=7 depth=2 uniq=5

=== redundancy_check (synthetic) ===
  noise vs amp_imb_20d:       independent=True,  rho={'amp_imb_20d': 0.026}
  copy_of_amp vs amp_imb_20d: independent=False, rho={'amp_imb_20d': 0.993}
```

Run via:

```bash
.venv/bin/python -m claude_finance.factor_quality_gate
```

---

## 7. Known limitations

1. `consistency_check` is intentionally shallow. It catches gross mismatches (missing symbols, missing operators, contradictory direction). It does NOT verify mathematical equivalence between formula and code. Use the upstream LLM-driven checker for that (out of scope here).
2. The directional word lists (`_DIRECTION_POSITIVE`/`_DIRECTION_NEGATIVE`) are hand-curated CN+EN bilingual. Adding domain-specific cues (e.g. "解禁", "净流出") may be needed when new factor families appear.
3. `complexity_score` upper bound is reached easily by very long expressions — interpret the breakdown rather than the aggregate when factors approach 8+.
4. `redundancy_check` requires the `new_factor` and the columns of `existing_panel` to share the same `(code, date)` MultiIndex. Misaligned indexes degrade gracefully (NaN per month, see fail-safe behaviour).
