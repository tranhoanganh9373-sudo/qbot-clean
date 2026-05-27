"""TDX 函数库 (pandas 实现).

每个函数接受 scalar / pd.Series 并返回 pd.Series (或 scalar).
TDX 语义优先 (非标准库的 SMA(x,n,m) 递推等需匹配).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_series(x: Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series([x])


def _to_int(n: Any, default: int = 1) -> int:
    """TDX 周期参数可能是 float / param series, 取首个有效 int."""
    if isinstance(n, pd.Series):
        try:
            v = n.dropna().iloc[0]
        except IndexError:
            v = default
    else:
        v = n
    try:
        i = int(round(float(v)))
        return max(1, i)
    except (TypeError, ValueError):
        return default


def MA(x: Any, n: Any) -> pd.Series:
    """简单移动平均."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n)
    return s.rolling(n_int, min_periods=1).mean()


def EMA(x: Any, n: Any) -> pd.Series:
    """指数移动平均, alpha = 2/(n+1)."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n)
    return s.ewm(span=n_int, adjust=False, min_periods=1).mean()


def SMA(x: Any, n: Any, m: Any = 1) -> pd.Series:
    """SMA(x,n,m): Y = (m*X + (n-m)*ref(Y,1)) / n, 通达信定义 (alpha = m/n)."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n)
    m_int = _to_int(m, default=1)
    if n_int <= 0:
        return pd.Series(np.nan, index=s.index)
    alpha = m_int / n_int
    return s.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def WMA(x: Any, n: Any) -> pd.Series:
    """加权移动平均: 权重 1..n (后者权重大)."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n)

    def _w(arr: np.ndarray) -> float:
        w = np.arange(1, len(arr) + 1, dtype=float)
        return float(np.dot(arr, w) / w.sum())

    return s.rolling(n_int, min_periods=1).apply(_w, raw=True)


def DMA(x: Any, a: Any) -> pd.Series:
    """DMA(x,a): Y = a*X + (1-a)*ref(Y,1). a 可以是 series 或 scalar."""
    s = _as_series(x).astype(float)
    if isinstance(a, pd.Series):
        a_s = a.reindex(s.index).fillna(0.5).clip(0, 1).to_numpy()
    else:
        a_val = float(a) if a is not None else 0.5
        a_val = max(0.0, min(1.0, a_val))
        a_s = np.full(len(s), a_val)
    out = np.empty(len(s))
    last = np.nan
    arr = s.to_numpy()
    for i in range(len(s)):
        xv = arr[i]
        av = a_s[i]
        if np.isnan(last):
            last = xv
        else:
            if not np.isnan(xv):
                last = av * xv + (1 - av) * last
        out[i] = last
    return pd.Series(out, index=s.index)


def REF(x: Any, n: Any) -> pd.Series:
    """REF(x,n): lag n 期."""
    s = _as_series(x)
    n_int = _to_int(n)
    return s.shift(n_int)


def HHV(x: Any, n: Any) -> pd.Series:
    """N 期最高 (N=0 -> 累计 cummax)."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n, default=0)
    if n_int <= 0:
        return s.cummax()
    return s.rolling(n_int, min_periods=1).max()


def LLV(x: Any, n: Any) -> pd.Series:
    """N 期最低 (N=0 -> 累计 cummin)."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n, default=0)
    if n_int <= 0:
        return s.cummin()
    return s.rolling(n_int, min_periods=1).min()


def SUM(x: Any, n: Any) -> pd.Series:
    """N 期求和; N=0 -> 累计和."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n, default=0)
    if n_int <= 0:
        return s.cumsum()
    return s.rolling(n_int, min_periods=1).sum()


def MAX(a: Any, b: Any) -> Any:
    """element-wise max."""
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        idx = (a.index if isinstance(a, pd.Series) else b.index)
        sa = a.reindex(idx) if isinstance(a, pd.Series) else pd.Series(float(a), index=idx)
        sb = b.reindex(idx) if isinstance(b, pd.Series) else pd.Series(float(b), index=idx)
        return pd.concat([sa, sb], axis=1).max(axis=1)
    return max(a, b)


def MIN(a: Any, b: Any) -> Any:
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        idx = (a.index if isinstance(a, pd.Series) else b.index)
        sa = a.reindex(idx) if isinstance(a, pd.Series) else pd.Series(float(a), index=idx)
        sb = b.reindex(idx) if isinstance(b, pd.Series) else pd.Series(float(b), index=idx)
        return pd.concat([sa, sb], axis=1).min(axis=1)
    return min(a, b)


def ABS(x: Any) -> Any:
    if isinstance(x, pd.Series):
        return x.abs()
    return abs(x)


def CROSS(a: Any, b: Any) -> pd.Series:
    """A 上穿 B (golden cross): A>B 且 上一期 A<=B."""
    sa = _as_series(a).astype(float)
    if isinstance(b, pd.Series):
        sb = b.reindex(sa.index).astype(float)
    else:
        sb = pd.Series(float(b), index=sa.index)
    prev_a = sa.shift(1)
    prev_b = sb.shift(1)
    out = ((sa > sb) & (prev_a <= prev_b)).astype(float)
    return out


def EVERY(x: Any, n: Any) -> pd.Series:
    """N 期内每一期都为真."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n)
    bool_s = (s > 0).astype(int)
    return (bool_s.rolling(n_int, min_periods=n_int).sum() >= n_int).astype(float)


def EXIST(x: Any, n: Any) -> pd.Series:
    """N 期内至少出现一次真."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n)
    bool_s = (s > 0).astype(int)
    return (bool_s.rolling(n_int, min_periods=1).sum() > 0).astype(float)


def COUNT(x: Any, n: Any) -> pd.Series:
    """N 期内真的次数."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n)
    return (s > 0).astype(int).rolling(n_int, min_periods=1).sum().astype(float)


def BARSLAST(x: Any) -> pd.Series:
    """距上次真发生的 bar 数; 没有则 NaN."""
    s = _as_series(x).astype(float)
    bool_arr = (s > 0).to_numpy()
    out = np.empty(len(s))
    last = -1
    for i, b in enumerate(bool_arr):
        if b:
            last = i
        out[i] = (i - last) if last >= 0 else np.nan
    return pd.Series(out, index=s.index)


def IF(cond: Any, a: Any, b: Any) -> Any:
    """IF(cond, a, b): cond>0 取 a, 否则 b."""
    if isinstance(cond, pd.Series):
        c = (cond > 0).astype(float)
        if isinstance(a, pd.Series):
            a_s = a.reindex(c.index).astype(float)
        else:
            a_s = pd.Series(float(a) if a is not None else np.nan, index=c.index)
        if isinstance(b, pd.Series):
            b_s = b.reindex(c.index).astype(float)
        else:
            b_s = pd.Series(float(b) if b is not None else np.nan, index=c.index)
        return c * a_s.fillna(0) + (1 - c) * b_s.fillna(0)
    return a if (cond is not None and cond > 0) else b


def STD(x: Any, n: Any) -> pd.Series:
    """N 期标准差 (sample, ddof=1)."""
    s = _as_series(x).astype(float)
    n_int = _to_int(n)
    return s.rolling(n_int, min_periods=2).std(ddof=1)


def LOG(x: Any) -> Any:
    if isinstance(x, pd.Series):
        return np.log(x.where(x > 0))
    return float(np.log(x)) if x > 0 else float("nan")


def SQRT(x: Any) -> Any:
    if isinstance(x, pd.Series):
        return np.sqrt(x.where(x >= 0))
    return float(np.sqrt(x)) if x >= 0 else float("nan")
