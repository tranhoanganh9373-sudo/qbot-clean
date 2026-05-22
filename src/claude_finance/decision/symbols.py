from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str  # "index" | "stock"
    code: str           # plain code, e.g. "600519" or "000300"
    ak_symbol: str      # akshare symbol, e.g. "sh000001" for indices


SYMBOLS: list[Symbol] = [
    Symbol("上证指数", "index", "000001", "sh000001"),
    Symbol("沪深300", "index", "000300", "sh000300"),
    Symbol("创业板指", "index", "399006", "sz399006"),
    Symbol("贵州茅台", "stock", "600519", "600519"),
    Symbol("宁德时代", "stock", "300750", "300750"),
    Symbol("招商银行", "stock", "600036", "600036"),
    Symbol("中国平安", "stock", "601318", "601318"),
]
