"""Meta-leak validation: split 60m OOS into 36m selection + 24m verification.

Question: Was train24's #1 ranking driven by look-ahead leak across 24 variants?
Method: Pick the winner using only the first 36 months (2021-05 ~ 2024-04);
verify in the held-out last 24 months (2024-05 ~ 2026-04).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sstats

EX_DIR = Path("/Volumes/SSD/finance/claude_finance/examples")

# (variant_tag, stats_csv_filename)
VARIANTS: list[tuple[str, str]] = [
    ("train24",              "v17_dens_train24_60m_stats.csv"),
    ("label5d",              "v17_dens_label5d_v19_label5d_60m_stats.csv"),
    ("v19_csi300",           "v17_dens_v19_csi300_60m_stats.csv"),
    ("dens_capital25k",      "v17_dens_dens_capital25k_60m_stats.csv"),
    ("capital25k_voltgt15",  "v17_dens_dens_capital25k_voltgt15_60m_stats.csv"),
    ("v19_dens_n5",          "v17_dens_v19_dens_n5_60m_stats.csv"),
    ("adaptive_k8d2",        "v17_dens_adaptive_adaptive_k8d2_60m_stats.csv"),
    ("train48",              "v17_dens_train48_60m_stats.csv"),
    ("dens_stoploss10",      "v17_dens_dens_stoploss10_60m_stats.csv"),
    ("dens_voltgt15",        "v17_dens_dens_voltgt15_60m_stats.csv"),
    ("v18_prod",             "v17_dens_v18_prod_60m_stats.csv"),
    ("dens_k8d2",            "v17_dens_dens_k8d2_60m_stats.csv"),
    ("v17_k8d2",             "v17_k8d2_60m_stats.csv"),
    ("multiwin",             "v17_dens_multiwin_multiwin_k8d2_60m_stats.csv"),
    ("train24_BD",           "v17_dens_train24_BD_60m_stats.csv"),
    ("v19_top1500",          "v17_dens_v19_top1500_60m_stats.csv"),
    ("dens_regime",          "v17_dens_dens_regime_60m_stats.csv"),
    ("v19_catboost",         "v17_dens_v19_catboost_60m_stats.csv"),
    ("alpha360",             "v17_dens_a360_dens_a360_k8d2_60m_stats.csv"),
    ("v19_csi500_train48",   "v17_dens_v19_csi500_train48_60m_stats.csv"),
    ("train36",              "v17_dens_train36_60m_stats.csv"),
    ("v19_csi500",           "v17_dens_v19_csi500_60m_stats.csv"),
    ("v19_csi500_train36",   "v17_dens_v19_csi500_train36_60m_stats.csv"),
    ("v19_xgb",              "v17_dens_v19_xgb_60m_stats.csv"),
    ("v17_hybrid",           "v17_hybrid_hybrid_60m_stats.csv"),
]

SEL_END = 36   # first 36 months = selection window
TOTAL = 60     # full OOS window


def load_returns(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "abs_ret_%" not in df.columns or "month" not in df.columns:
        return None
    df = df[["month", "abs_ret_%"]].copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)
    return df


def metrics(returns_pct: pd.Series) -> dict:
    """Compute Calmar, Sharpe, cum, MDD, Win% on monthly pct returns (5.0 means +5%)."""
    r = returns_pct.to_numpy(dtype=float) / 100.0
    n = len(r)
    if n == 0:
        return {"cum_%": np.nan, "ann_%": np.nan, "mdd_%": np.nan,
                "calmar": np.nan, "sharpe": np.nan, "win_%": np.nan}
    eq = np.cumprod(1.0 + r)
    cum = eq[-1] - 1.0
    ann = (1.0 + cum) ** (12.0 / n) - 1.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    mdd = float(dd.min())  # negative
    mu = r.mean()
    sd = r.std(ddof=1) if n > 1 else 0.0
    sharpe = (mu / sd * np.sqrt(12.0)) if sd > 0 else np.nan
    calmar = (ann / abs(mdd)) if mdd < 0 else np.nan
    win = float((r > 0).mean())
    return {
        "cum_%": round(cum * 100, 2),
        "ann_%": round(ann * 100, 2),
        "mdd_%": round(mdd * 100, 2),
        "calmar": round(calmar, 3) if not np.isnan(calmar) else np.nan,
        "sharpe": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "win_%": round(win * 100, 2),
    }


def main() -> int:
    rows = []
    missing = []
    for tag, fname in VARIANTS:
        path = EX_DIR / fname
        df = load_returns(path)
        if df is None or len(df) < TOTAL:
            missing.append((tag, fname, "missing" if df is None else f"only {len(df)} months"))
            continue
        sel = df.iloc[:SEL_END]
        oos = df.iloc[SEL_END:TOTAL]
        sel_start = sel["month"].iloc[0].strftime("%Y-%m")
        sel_end = sel["month"].iloc[-1].strftime("%Y-%m")
        oos_start = oos["month"].iloc[0].strftime("%Y-%m")
        oos_end = oos["month"].iloc[-1].strftime("%Y-%m")
        m_sel = metrics(sel["abs_ret_%"])
        m_oos = metrics(oos["abs_ret_%"])
        m_full = metrics(df.iloc[:TOTAL]["abs_ret_%"])
        rows.append({
            "variant": tag,
            "sel_span": f"{sel_start}~{sel_end}",
            "oos_span": f"{oos_start}~{oos_end}",
            "sel_cum_%": m_sel["cum_%"], "sel_ann_%": m_sel["ann_%"],
            "sel_mdd_%": m_sel["mdd_%"], "sel_calmar": m_sel["calmar"],
            "sel_sharpe": m_sel["sharpe"], "sel_win_%": m_sel["win_%"],
            "oos_cum_%": m_oos["cum_%"], "oos_ann_%": m_oos["ann_%"],
            "oos_mdd_%": m_oos["mdd_%"], "oos_calmar": m_oos["calmar"],
            "oos_sharpe": m_oos["sharpe"], "oos_win_%": m_oos["win_%"],
            "full_calmar": m_full["calmar"], "full_sharpe": m_full["sharpe"],
            "full_cum_%": m_full["cum_%"],
        })

    if missing:
        print("=== MISSING / SHORT FILES ===")
        for tag, fname, why in missing:
            print(f"  - {tag}: {fname} ({why})")
        print()

    res = pd.DataFrame(rows)
    if res.empty:
        print("No usable stats found.", file=sys.stderr)
        return 1

    # Rank (descending) — best Calmar = rank 1. NaN ranks left as NaN.
    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(ascending=False, method="min", na_option="bottom").astype("Int64")

    res["sel_rank_calmar"] = _rank(res["sel_calmar"])
    res["oos_rank_calmar"] = _rank(res["oos_calmar"])
    res["sel_rank_sharpe"] = _rank(res["sel_sharpe"])
    res["oos_rank_sharpe"] = _rank(res["oos_sharpe"])
    res["sel_rank_cum"]    = _rank(res["sel_cum_%"])
    res["oos_rank_cum"]    = _rank(res["oos_cum_%"])

    res = res.sort_values("sel_rank_calmar").reset_index(drop=True)

    # Spearman rank correlation
    rho_calmar, p_calmar = sstats.spearmanr(res["sel_rank_calmar"], res["oos_rank_calmar"])
    rho_sharpe, p_sharpe = sstats.spearmanr(res["sel_rank_sharpe"], res["oos_rank_sharpe"])
    rho_cum, p_cum       = sstats.spearmanr(res["sel_rank_cum"],    res["oos_rank_cum"])

    out_csv = EX_DIR / "meta_leak_validation_36m_24m_stats.csv"
    res.to_csv(out_csv, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 60)

    print(f"=== Variants loaded: {len(res)}/{len(VARIANTS)} ===\n")

    print("=== Selection 36m (2021-05 ~ 2024-04) — sorted by sel_calmar ===")
    cols_sel = ["sel_rank_calmar", "variant", "sel_cum_%", "sel_ann_%",
                "sel_mdd_%", "sel_calmar", "sel_sharpe", "sel_win_%"]
    print(res[cols_sel].to_string(index=False))

    print("\n=== OOS 24m (2024-05 ~ 2026-04) — sorted by oos_calmar ===")
    by_oos = res.sort_values("oos_rank_calmar")[
        ["oos_rank_calmar", "variant", "oos_cum_%", "oos_ann_%",
         "oos_mdd_%", "oos_calmar", "oos_sharpe", "oos_win_%",
         "sel_rank_calmar"]
    ]
    print(by_oos.to_string(index=False))

    print("\n=== Spearman rank correlation (selection vs OOS) ===")
    print(f"  Calmar : rho = {rho_calmar:+.3f}  (p = {p_calmar:.4f})")
    print(f"  Sharpe : rho = {rho_sharpe:+.3f}  (p = {p_sharpe:.4f})")
    print(f"  CumRet : rho = {rho_cum:+.3f}  (p = {p_cum:.4f})")

    print("\n=== Key questions ===")
    sel_top = res.loc[res["sel_rank_calmar"] == 1, "variant"].iloc[0]
    oos_top = res.loc[res["oos_rank_calmar"] == 1, "variant"].iloc[0]
    print(f"  Selection-period winner (by Calmar): {sel_top}")
    print(f"  OOS-period winner (by Calmar)      : {oos_top}")

    train24_row = res[res["variant"] == "train24"]
    if not train24_row.empty:
        t = train24_row.iloc[0]
        print(f"\n  train24 selection-period rank: #{int(t['sel_rank_calmar'])} "
              f"(Calmar {t['sel_calmar']}, Sharpe {t['sel_sharpe']}, Cum {t['sel_cum_%']}%)")
        print(f"  train24 OOS-period rank      : #{int(t['oos_rank_calmar'])} "
              f"(Calmar {t['oos_calmar']}, Sharpe {t['oos_sharpe']}, Cum {t['oos_cum_%']}%)")

    sel_winner_oos_rank = int(res.loc[res["variant"] == sel_top, "oos_rank_calmar"].iloc[0])
    print(f"\n  Selection #1 ({sel_top}) lands at OOS rank #{sel_winner_oos_rank}")

    print(f"\n[saved] {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
