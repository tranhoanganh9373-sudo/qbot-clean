"""Monte Carlo GBM path simulation + risk metrics.

Refactor of qbot's pytrader/strategies/monte_carlo.py — original was just a
matplotlib script computing CAGR and annual volatility from AAPL prices, with
no actual signal logic. This module exposes the same primitives as proper
risk utilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cagr(close: pd.Series) -> float:
    """Compound annual growth rate from a price series."""
    days = (close.index[-1] - close.index[0]).days
    if days <= 0:
        raise ValueError("close series must span > 0 days")
    return float((close.iloc[-1] / close.iloc[0]) ** (365.0 / days) - 1)


def annualized_volatility(close: pd.Series) -> float:
    """Annualised stdev of daily returns."""
    returns = close.pct_change().dropna()
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def simulate_paths(
    init_price: float,
    mu: float,
    sigma: float,
    horizon_days: int,
    n_paths: int = 1000,
    seed: int | None = 42,
) -> np.ndarray:
    """Simulate geometric-Brownian-motion price paths.

    mu / sigma are ANNUALISED. Returns array of shape (n_paths, horizon_days+1),
    where column 0 is init_price.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS_PER_YEAR
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    shocks = rng.normal(0, 1, size=(n_paths, horizon_days))
    log_steps = drift + diffusion * shocks
    log_paths = np.cumsum(log_steps, axis=1)
    return np.concatenate(
        [np.full((n_paths, 1), init_price), init_price * np.exp(log_paths)], axis=1
    )


def value_at_risk(
    close: pd.Series,
    horizon_days: int = 20,
    confidence: float = 0.95,
    n_paths: int = 1000,
    seed: int | None = 42,
) -> float:
    """Monte-Carlo VaR: positive number = expected worst-case loss as a fraction.

    A return of 0.08 at 95% confidence means: there is a 5% probability
    you lose more than 8% over the horizon.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    mu = cagr(close)
    sigma = annualized_volatility(close)
    paths = simulate_paths(
        init_price=float(close.iloc[-1]),
        mu=mu,
        sigma=sigma,
        horizon_days=horizon_days,
        n_paths=n_paths,
        seed=seed,
    )
    final_returns = paths[:, -1] / paths[:, 0] - 1
    return float(-np.quantile(final_returns, 1 - confidence))
