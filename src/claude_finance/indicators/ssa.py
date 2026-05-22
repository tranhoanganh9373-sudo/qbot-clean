"""Singular Spectrum Analysis (SSA) — noise-reduced trend reconstruction.

Rolling-window SVD: for each window of length ``window``, embed into a
trajectory matrix, keep the largest singular component, and read off the
reconstructed last value.

The result is a trend Series aligned to the input; the first ``window`` bars
are NaN (insufficient history for embedding).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ssa_last_value(arr: np.ndarray) -> float:
    """Reconstruct the last bar of `arr` via rank-1 SVD on its trajectory matrix."""
    n = len(arr)
    m = n // 2 + 1
    k = n - m + 1

    traj = np.lib.stride_tricks.sliding_window_view(arr, window_shape=m).T  # (m, k)
    u, s, vt = np.linalg.svd(traj, full_matrices=False)
    rank1 = s[0] * np.outer(u[:, 0], vt[0])

    # Diagonal averaging at the anti-diagonal index n-1
    total = 0.0
    count = 0
    for i in range(m):
        j = (n - 1) - i
        if 0 <= j < k:
            total += rank1[i, j]
            count += 1
    return total / count


def ssa(close: pd.Series, window: int = 30) -> pd.Series:
    """Return SSA-smoothed series aligned to ``close``.

    Window length controls the trade-off: longer = smoother but laggier.
    """
    arr = close.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        out[i] = _ssa_last_value(arr[i - window : i])
    return pd.Series(out, index=close.index, name="ssa")
