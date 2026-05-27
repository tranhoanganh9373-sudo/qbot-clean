"""A-share trading cost models: slippage + commission + stamp duty + transfer fee.

Reference: zipline-reloaded slippage.py / commission.py — but rewritten for
A-share local rules (not a direct port).

A-share cost stack (retail broker, post 2023-08-28):
  - Broker commission: 0.025% (双边), 5 RMB minimum per trade
  - Stamp duty:        0.05%  (sell only, 2023-08-28 cut from 0.1%)
  - Transfer fee:      0.001% (双边, Shanghai-listed 6/9 only; Shenzhen exempt)
  - Slippage:          ~5 bps (1 minute mid-fill assumption, retail)

Convention: prices are quoted RMB per share, shares are integer 100-share lots
in real A-share trading but we accept any positive int here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from typing import Literal

Side = Literal["BUY", "SELL"]

# Policy switch: stamp duty was cut from 0.1% to 0.05% on 2023-08-28
STAMP_DUTY_CUT_DATE = _date(2023, 8, 28)
STAMP_DUTY_RATE_PRE = 0.001   # 0.1%
STAMP_DUTY_RATE_POST = 0.0005  # 0.05%


# ---------------------------------------------------------------------------
# Slippage models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedSlippageModel:
    """Fixed-bps slippage applied symmetrically (worse fill for both sides).

    Default 5 bps (0.05%) reflects a retail mid-fill assumption — a buy fills
    slightly above quoted price, a sell fills slightly below.
    """

    bps: float = 5.0

    def adjust(self, price: float, side: Side) -> float:
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        side_u = side.upper()
        delta = self.bps / 10000.0
        if side_u == "BUY":
            return price * (1.0 + delta)
        if side_u == "SELL":
            return price * (1.0 - delta)
        raise ValueError(f"side must be BUY or SELL, got {side!r}")


@dataclass(frozen=True)
class VolumeImpactModel:
    """Volume-share linear impact slippage.

    Fill price moves against you in proportion to the share of bar volume
    consumed. For most retail-size (5万 RMB) orders against CSI300-tier
    liquidity this is negligible — prefer ``FixedSlippageModel`` unless
    you actually have intraday volume data.

    impact = self.impact * (order_volume / bar_volume)
    """

    impact: float = 0.1

    def adjust(self, price: float, side: Side, *, order_volume: float = 0.0,
               bar_volume: float = 0.0) -> float:
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if bar_volume <= 0 or order_volume <= 0:
            # Degenerate: no impact computable, return clean price.
            return float(price)
        share = order_volume / bar_volume
        delta = self.impact * share
        side_u = side.upper()
        if side_u == "BUY":
            return price * (1.0 + delta)
        if side_u == "SELL":
            return price * (1.0 - delta)
        raise ValueError(f"side must be BUY or SELL, got {side!r}")


def apply_slippage(price: float, side: Side, model) -> float:
    """Apply a slippage model's adjust() and return the filled price."""
    return model.adjust(price, side)


# ---------------------------------------------------------------------------
# Commission, stamp duty, transfer fee
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommissionAShare:
    """Broker commission: rate × notional, with a per-trade minimum (RMB).

    Default 0.025% (双边) and 5 RMB minimum — typical retail rate after the
    2014-2024 broker-fee compression wave. Some brokers charge as low as
    0.012%; institutions ~0.005%. Set ``rate`` accordingly.
    """

    rate: float = 0.00025
    min_fee: float = 5.0

    def calculate(self, price: float, shares: int) -> float:
        if price < 0 or shares < 0:
            raise ValueError("price and shares must be non-negative")
        notional = price * shares
        fee = notional * self.rate
        return max(fee, self.min_fee) if notional > 0 else 0.0


@dataclass(frozen=True)
class StampDutyAShare:
    """Stamp duty: SELL-only tax (印花税).

    Use ``rate`` to override; pass ``trade_date`` to ``calculate`` to use the
    historical schedule (0.1% before 2023-08-28, 0.05% on/after).
    """

    rate: float = STAMP_DUTY_RATE_POST  # 0.05%

    def calculate(self, price: float, shares: int, side: Side,
                  *, trade_date: _date | None = None) -> float:
        if price < 0 or shares < 0:
            raise ValueError("price and shares must be non-negative")
        if side.upper() != "SELL":
            return 0.0
        if trade_date is not None:
            # Historical-aware override
            rate = (STAMP_DUTY_RATE_POST
                    if trade_date >= STAMP_DUTY_CUT_DATE
                    else STAMP_DUTY_RATE_PRE)
        else:
            rate = self.rate
        return price * shares * rate


@dataclass(frozen=True)
class TransferFee:
    """Transfer fee: Shanghai (6xx / 9xx) only, 双边.

    Default 0.001% — the rate cut took effect 2022-04-29 (from 0.002%); we use
    the current rate. Shenzhen (00xx / 30xx) and Beijing (8xx / 4xx) are
    exempt.
    """

    rate: float = 0.00001  # 0.001%

    @staticmethod
    def applies_to(symbol: str) -> bool:
        """Return True if symbol is a Shanghai-listed equity."""
        if not symbol:
            return False
        s = symbol.upper().strip()
        # Accept "SH600519" / "600519.SH" / "600519" / "688xxx" / "9xxxxx"
        if s.startswith("SH"):
            code = s[2:]
        elif s.endswith(".SH"):
            code = s[:-3]
        else:
            code = s
        if not code or not code[0].isdigit():
            return False
        # Reject explicit non-SH prefixes that survived stripping
        if s.startswith(("SZ", "BJ")) or s.endswith((".SZ", ".BJ")):
            return False
        first = code[0]
        # SH equities: 600/601/603/605 (main) + 688 (STAR) + 900 (B-share)
        return first in {"6", "9"}

    def calculate(self, price: float, shares: int, symbol: str) -> float:
        if price < 0 or shares < 0:
            raise ValueError("price and shares must be non-negative")
        if not self.applies_to(symbol):
            return 0.0
        return price * shares * self.rate


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def total_transaction_cost(
    price: float,
    shares: int,
    side: Side,
    symbol: str,
    commission_model: CommissionAShare,
    stamp_duty_model: StampDutyAShare,
    transfer_fee_model: TransferFee,
    slippage_model,
    *,
    trade_date: _date | None = None,
) -> dict:
    """Aggregate slippage + 3 fees into a single transaction breakdown.

    Returns a dict with:
      - gross_price:    quoted price (input)
      - filled_price:   price after slippage
      - commission:     broker commission (RMB)
      - stamp_duty:     stamp duty (SELL only, RMB)
      - transfer_fee:   过户费 (SH only, RMB)
      - slippage_cost:  (filled - gross) * shares for BUY, (gross - filled) * shares for SELL
      - net_amount:     cash delta — negative for BUY (you pay), positive for SELL (you receive)
      - total_fees:     commission + stamp_duty + transfer_fee
    """
    if shares < 0:
        raise ValueError("shares must be non-negative")
    side_u = side.upper()
    if side_u not in {"BUY", "SELL"}:
        raise ValueError(f"side must be BUY or SELL, got {side!r}")

    filled_price = apply_slippage(price, side_u, slippage_model)
    commission = commission_model.calculate(filled_price, shares)
    stamp_duty = stamp_duty_model.calculate(filled_price, shares, side_u,
                                            trade_date=trade_date)
    transfer_fee = transfer_fee_model.calculate(filled_price, shares, symbol)
    total_fees = commission + stamp_duty + transfer_fee

    if side_u == "BUY":
        slippage_cost = (filled_price - price) * shares
        net_amount = -(filled_price * shares + total_fees)
    else:  # SELL
        slippage_cost = (price - filled_price) * shares
        net_amount = filled_price * shares - total_fees

    return {
        "gross_price": float(price),
        "filled_price": float(filled_price),
        "commission": float(commission),
        "stamp_duty": float(stamp_duty),
        "transfer_fee": float(transfer_fee),
        "slippage_cost": float(slippage_cost),
        "total_fees": float(total_fees),
        "net_amount": float(net_amount),
    }


__all__ = [
    "FixedSlippageModel",
    "VolumeImpactModel",
    "apply_slippage",
    "CommissionAShare",
    "StampDutyAShare",
    "TransferFee",
    "total_transaction_cost",
    "STAMP_DUTY_CUT_DATE",
    "STAMP_DUTY_RATE_PRE",
    "STAMP_DUTY_RATE_POST",
]
