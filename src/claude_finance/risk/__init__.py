"""Risk utilities — Monte Carlo path simulation, VaR estimation.

Note: this is intentionally NOT a strategies/ subpackage. Monte Carlo path
simulation isn't a signal-generating strategy (the qbot source was just a
CAGR + vol calculation), so it's exposed here as a risk-analysis tool.
"""

from claude_finance.risk.monte_carlo import (
    annualized_volatility,
    cagr,
    simulate_paths,
    value_at_risk,
)

__all__ = ["annualized_volatility", "cagr", "simulate_paths", "value_at_risk"]
