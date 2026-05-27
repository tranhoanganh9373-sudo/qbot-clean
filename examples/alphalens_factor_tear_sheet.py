"""Run an alphalens IC tear-sheet for a single candidate factor.

Pulls a factor series from one of the factor parquets in ``data_cache/`` and
the close prices from ``data_cache/baidu_kline.parquet``, formats them into
the MultiIndex DataFrame alphalens expects, then runs
``al.tears.create_full_tear_sheet``. Falls back to a structural stub if
alphalens is not installed.

Install (only if missing): ``uv pip install alphalens-reloaded``

Usage:
    .venv/bin/python examples/alphalens_factor_tear_sheet.py \\
        --factor margin_5d_chg \\
        --start 2014-01-01 --end 2020-12-31 \\
        --universe csi300

Built-in factor sources:
    margin_5d_chg, margin_20d_chg  -> data_cache/csi300_margin_14yr.parquet
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f"[WARN] matplotlib unavailable: {exc}", file=sys.stderr)
    plt = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import alphalens as al
    ALPHALENS_AVAILABLE = True
except Exception as exc:
    print(f"[WARN] alphalens not importable ({exc}); will emit stub summary. "
          f"Install with: uv pip install alphalens-reloaded",
          file=sys.stderr)
    ALPHALENS_AVAILABLE = False
    al = None


FACTOR_SOURCES = {
    "margin_5d_chg": "data_cache/csi300_margin_14yr.parquet",
    "margin_20d_chg": "data_cache/csi300_margin_14yr.parquet",
}


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def _load_universe(universe: str, root: Path) -> set[str] | None:
    """Return a set of bare-code strings, or None for the full kline universe."""
    if not universe or universe.lower() in {"all", "none"}:
        return None
    if universe.lower() == "csi300":
        path = root / "data_cache" / "csi300_constituents.csv"
        if not path.exists():
            print(f"[WARN] universe file missing: {path}", file=sys.stderr)
            return None
        df = pd.read_csv(path, dtype={"code": str})
        return set(df["code"].astype(str).str.zfill(6))
    raise ValueError(f"unknown universe: {universe}")


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def prepare_alphalens_data(
    factor_df: pd.DataFrame,
    kline_df: pd.DataFrame,
    factor_name: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return (factor MultiIndex Series, prices wide DataFrame).

    - ``factor_df`` must have columns ``code, date, <factor_name>``.
    - ``kline_df`` must have columns ``code, date, close``.
    """
    if factor_name not in factor_df.columns:
        raise KeyError(
            f"factor '{factor_name}' not in factor_df columns "
            f"{factor_df.columns.tolist()}"
        )
    f = factor_df[["date", "code", factor_name]].copy()
    f = f.dropna(subset=[factor_name])
    f["date"] = pd.to_datetime(f["date"])
    factor = (
        f.set_index(["date", "code"])[factor_name]
        .sort_index()
    )
    factor.index.set_names(["date", "asset"], inplace=True)

    p = kline_df[["date", "code", "close"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    prices = (
        p.pivot_table(index="date", columns="code", values="close",
                      aggfunc="last")
        .sort_index()
    )
    return factor, prices


# ---------------------------------------------------------------------------
# Tear-sheet entry point
# ---------------------------------------------------------------------------


def run_tear_sheet(factor: pd.Series, prices: pd.DataFrame,
                   periods: tuple[int, ...], quantiles: int,
                   out_dir: Path, factor_name: str) -> dict:
    summary = {
        "n_factor_obs": int(len(factor)),
        "n_price_dates": int(len(prices.index)),
        "n_price_assets": int(prices.shape[1]),
        "periods": list(periods),
        "quantiles": quantiles,
        "alphalens_used": ALPHALENS_AVAILABLE,
    }

    if not ALPHALENS_AVAILABLE or plt is None:
        print("[STUB] alphalens/matplotlib missing; emitting summary only.")
        try:
            fwd1 = (prices.shift(-1) / prices - 1).stack()
            fwd1.index.set_names(["date", "asset"], inplace=True)
            joined = pd.concat([factor.rename("factor"),
                                fwd1.rename("fwd")], axis=1).dropna()
            ic = (joined.groupby(level="date")
                  .apply(lambda g: g["factor"].corr(g["fwd"], method="spearman"))
                  .mean())
            summary["rank_ic_1d_stub"] = float(ic)
        except Exception as exc:
            summary["rank_ic_1d_stub_error"] = str(exc)
        summary["status"] = "stub"
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            factor_data = al.utils.get_clean_factor_and_forward_returns(
                factor=factor,
                prices=prices,
                quantiles=quantiles,
                periods=periods,
                max_loss=0.5,
            )
            ic = al.performance.factor_information_coefficient(factor_data)
            summary["mean_ic"] = {str(k): float(v) for k, v in ic.mean().items()}
            summary["ic_ir"] = {
                str(k): (float(ic[k].mean() / ic[k].std()) if ic[k].std() > 0
                         else float("nan"))
                for k in ic.columns
            }

            al.tears.create_returns_tear_sheet(factor_data)
            fig_path_returns = out_dir / f"alphalens_{factor_name}_returns.png"
            plt.savefig(fig_path_returns, dpi=100, bbox_inches="tight")
            plt.close("all")

            al.tears.create_information_tear_sheet(factor_data)
            fig_path_ic = out_dir / f"alphalens_{factor_name}_ic.png"
            plt.savefig(fig_path_ic, dpi=100, bbox_inches="tight")
            plt.close("all")

            summary["figures"] = [str(fig_path_returns), str(fig_path_ic)]
            summary["status"] = "ok"
    except Exception as exc:
        print(f"[WARN] alphalens failed ({exc}); falling back to stub.",
              file=sys.stderr)
        summary["status"] = "alphalens_error"
        summary["error"] = str(exc)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", default="margin_5d_chg")
    parser.add_argument("--factor-path", default=None,
                        help="override factor parquet path; auto for built-ins")
    parser.add_argument("--kline", default="data_cache/baidu_kline.parquet")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--universe", default="csi300", help="csi300 | all")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--periods", default="1,5,10,21",
                        help="comma-separated forward-return horizons")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args(argv)

    root = Path(".").resolve()
    factor_path = args.factor_path or FACTOR_SOURCES.get(args.factor)
    if factor_path is None:
        print(f"[ERROR] no built-in source for factor '{args.factor}'; "
              f"pass --factor-path", file=sys.stderr)
        return 2

    factor_df = pd.read_parquet(factor_path)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    factor_df = factor_df[(factor_df["date"] >= start)
                          & (factor_df["date"] <= end)]

    universe = _load_universe(args.universe, root)
    if universe is not None:
        factor_df = factor_df[
            factor_df["code"].astype(str).str.zfill(6).isin(universe)
        ]
    if factor_df.empty:
        print("[ERROR] no factor rows after universe/date filter",
              file=sys.stderr)
        return 2

    kline = pd.read_parquet(args.kline, columns=["code", "date", "close"])
    kline = kline[(kline["date"] >= start)
                  & (kline["date"] <= end + pd.Timedelta(days=60))]
    if universe is not None:
        kline = kline[kline["code"].astype(str).str.zfill(6).isin(universe)]
    factor_codes = set(factor_df["code"].astype(str))
    kline = kline[kline["code"].astype(str).isin(factor_codes)]

    factor, prices = prepare_alphalens_data(factor_df, kline, args.factor)
    periods = tuple(int(x.strip()) for x in args.periods.split(",")
                    if x.strip())

    summary = run_tear_sheet(
        factor=factor, prices=prices, periods=periods,
        quantiles=args.quantiles, out_dir=Path(args.out_dir),
        factor_name=args.factor,
    )

    print("=" * 60)
    print(f"alphalens factor tear sheet: {args.factor}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:>22}: {v}")
    print("=" * 60)
    return 0 if summary.get("status") in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
