from __future__ import annotations

import numpy as np
import pandas as pd

from claude_finance.data.schema import validate_ohlcv


def load_synthetic_daily(
    n: int = 500,
    start: str = "2023-01-01",
    seed: int = 42,
    init_price: float = 100.0,
    mu: float = 0.0005,
    sigma: float = 0.015,
) -> pd.DataFrame:
    """Generate deterministic OHLCV bars via geometric Brownian motion.

    Pure in-memory, no network, no disk. Use for smoke tests when sandboxed
    or when you just want a quick run without hitting akshare/tushare.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    close = init_price * np.exp(np.cumsum(rets))

    intra = rng.uniform(0.002, 0.012, n)
    high = close * (1 + intra)
    low = close * (1 - intra)
    open_ = np.concatenate([[init_price], close[:-1]])
    volume = rng.integers(500_000, 5_000_000, n).astype("int64")

    idx = pd.date_range(start, periods=n, freq="B")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "date"
    return validate_ohlcv(df)
