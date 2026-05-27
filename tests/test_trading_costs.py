"""Unit tests for A-share trading cost models."""
from __future__ import annotations

from datetime import date

import pytest

from claude_finance.trading_costs import (
    CommissionAShare,
    FixedSlippageModel,
    STAMP_DUTY_CUT_DATE,
    StampDutyAShare,
    TransferFee,
    VolumeImpactModel,
    apply_slippage,
    total_transaction_cost,
)


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------


class TestFixedSlippage:
    def test_buy_increases_price(self):
        model = FixedSlippageModel(bps=5)
        assert apply_slippage(100.0, "BUY", model) == pytest.approx(100.05)

    def test_sell_decreases_price(self):
        model = FixedSlippageModel(bps=5)
        assert apply_slippage(100.0, "SELL", model) == pytest.approx(99.95)

    def test_zero_bps_is_no_op(self):
        model = FixedSlippageModel(bps=0)
        assert apply_slippage(50.0, "BUY", model) == 50.0
        assert apply_slippage(50.0, "SELL", model) == 50.0

    def test_invalid_side_raises(self):
        model = FixedSlippageModel()
        with pytest.raises(ValueError, match="side must be BUY or SELL"):
            apply_slippage(100.0, "HOLD", model)

    def test_negative_price_raises(self):
        model = FixedSlippageModel()
        with pytest.raises(ValueError, match="price must be positive"):
            apply_slippage(-1.0, "BUY", model)


class TestVolumeImpact:
    def test_zero_volume_is_no_op(self):
        model = VolumeImpactModel(impact=0.1)
        assert model.adjust(100.0, "BUY", order_volume=1000, bar_volume=0) == 100.0

    def test_buy_with_volume_impact(self):
        # 1% of bar volume * 0.1 impact = 0.1% slippage
        model = VolumeImpactModel(impact=0.1)
        filled = model.adjust(100.0, "BUY", order_volume=100, bar_volume=10_000)
        assert filled == pytest.approx(100.0 * (1 + 0.1 * 0.01))  # 100.01

    def test_sell_with_volume_impact(self):
        model = VolumeImpactModel(impact=0.1)
        filled = model.adjust(100.0, "SELL", order_volume=100, bar_volume=10_000)
        assert filled == pytest.approx(100.0 * (1 - 0.1 * 0.01))  # 99.99


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------


class TestCommission:
    def test_normal_trade_uses_rate(self):
        # 100 RMB * 1000 shares * 0.025% = 25 RMB > 5 min
        model = CommissionAShare(rate=0.00025, min_fee=5.0)
        assert model.calculate(100.0, 1000) == pytest.approx(25.0)

    def test_small_trade_hits_minimum(self):
        # 10 RMB * 100 shares * 0.025% = 0.25 RMB -> bumped to 5 RMB min
        model = CommissionAShare(rate=0.00025, min_fee=5.0)
        assert model.calculate(10.0, 100) == 5.0

    def test_zero_notional_no_fee(self):
        model = CommissionAShare()
        assert model.calculate(100.0, 0) == 0.0
        assert model.calculate(0.0, 1000) == 0.0

    def test_custom_minimum(self):
        # Cheap broker: 1 RMB minimum
        model = CommissionAShare(rate=0.00015, min_fee=1.0)
        # 10 * 100 * 0.00015 = 0.15 -> bumped to 1
        assert model.calculate(10.0, 100) == 1.0


# ---------------------------------------------------------------------------
# Stamp duty
# ---------------------------------------------------------------------------


class TestStampDuty:
    def test_sell_triggers(self):
        # 100 * 1000 * 0.05% = 50 RMB
        model = StampDutyAShare(rate=0.0005)
        assert model.calculate(100.0, 1000, "SELL") == pytest.approx(50.0)

    def test_buy_no_duty(self):
        model = StampDutyAShare(rate=0.0005)
        assert model.calculate(100.0, 1000, "BUY") == 0.0

    def test_historical_pre_2023_uses_higher_rate(self):
        # Before 2023-08-28 -> 0.1% rate
        model = StampDutyAShare()
        pre = date(2023, 8, 27)
        # 100 * 1000 * 0.1% = 100 RMB
        assert model.calculate(100.0, 1000, "SELL", trade_date=pre) == pytest.approx(100.0)

    def test_historical_post_2023_uses_lower_rate(self):
        # On/after 2023-08-28 -> 0.05% rate
        model = StampDutyAShare()
        assert (model.calculate(100.0, 1000, "SELL", trade_date=STAMP_DUTY_CUT_DATE)
                == pytest.approx(50.0))


# ---------------------------------------------------------------------------
# Transfer fee
# ---------------------------------------------------------------------------


class TestTransferFee:
    def test_sh_main_board_triggers(self):
        # SH600519 -> main board, fee applies
        model = TransferFee(rate=0.00001)
        # 100 * 1000 * 0.001% = 1 RMB
        assert model.calculate(100.0, 1000, "SH600519") == pytest.approx(1.0)

    def test_sz_exempt(self):
        model = TransferFee(rate=0.00001)
        assert model.calculate(100.0, 1000, "SZ300347") == 0.0
        assert model.calculate(100.0, 1000, "SZ000001") == 0.0

    def test_sh_star_board_triggers(self):
        # 688xxx STAR market is Shanghai
        model = TransferFee(rate=0.00001)
        assert model.calculate(100.0, 1000, "SH688981") == pytest.approx(1.0)

    def test_dot_suffix_format(self):
        model = TransferFee(rate=0.00001)
        assert model.calculate(100.0, 1000, "600519.SH") == pytest.approx(1.0)
        assert model.calculate(100.0, 1000, "300347.SZ") == 0.0

    def test_applies_to_helper(self):
        assert TransferFee.applies_to("SH600519")
        assert TransferFee.applies_to("SH688981")
        assert TransferFee.applies_to("600519")  # bare 6xx -> SH
        assert not TransferFee.applies_to("SZ300347")
        assert not TransferFee.applies_to("000001")  # SZ main board
        assert not TransferFee.applies_to("")
        assert not TransferFee.applies_to("ABC")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class TestTotalTransactionCost:
    def _models(self):
        return (
            CommissionAShare(rate=0.00025, min_fee=5.0),
            StampDutyAShare(rate=0.0005),
            TransferFee(rate=0.00001),
            FixedSlippageModel(bps=5),
        )

    def test_buy_sh_full_stack(self):
        comm, stamp, transfer, slip = self._models()
        out = total_transaction_cost(
            price=100.0, shares=1000, side="BUY", symbol="SH600519",
            commission_model=comm, stamp_duty_model=stamp,
            transfer_fee_model=transfer, slippage_model=slip,
        )
        # filled = 100 * 1.0005 = 100.05
        assert out["filled_price"] == pytest.approx(100.05)
        # commission = 100.05 * 1000 * 0.00025 = 25.0125
        assert out["commission"] == pytest.approx(25.0125)
        # BUY -> no stamp duty
        assert out["stamp_duty"] == 0.0
        # SH -> transfer fee = 100.05 * 1000 * 0.00001 = 1.0005
        assert out["transfer_fee"] == pytest.approx(1.0005)
        # slippage_cost = (100.05 - 100) * 1000 = 50
        assert out["slippage_cost"] == pytest.approx(50.0)
        # net_amount = -(100.05*1000 + 25.0125 + 1.0005)
        assert out["net_amount"] == pytest.approx(-(100050 + 25.0125 + 1.0005))

    def test_sell_sz_no_transfer_fee(self):
        comm, stamp, transfer, slip = self._models()
        out = total_transaction_cost(
            price=100.0, shares=1000, side="SELL", symbol="SZ300347",
            commission_model=comm, stamp_duty_model=stamp,
            transfer_fee_model=transfer, slippage_model=slip,
        )
        # filled = 100 * 0.9995 = 99.95
        assert out["filled_price"] == pytest.approx(99.95)
        # stamp duty applies on SELL: 99.95 * 1000 * 0.0005 = 49.975
        assert out["stamp_duty"] == pytest.approx(49.975)
        # SZ -> no transfer fee
        assert out["transfer_fee"] == 0.0
        expected_comm = 99.95 * 1000 * 0.00025
        expected_net = 99950 - (expected_comm + 49.975)
        assert out["net_amount"] == pytest.approx(expected_net)

    def test_invalid_side_raises(self):
        comm, stamp, transfer, slip = self._models()
        with pytest.raises(ValueError, match="side must be BUY or SELL"):
            total_transaction_cost(
                price=100.0, shares=100, side="HOLD", symbol="SH600519",
                commission_model=comm, stamp_duty_model=stamp,
                transfer_fee_model=transfer, slippage_model=slip,
            )

    def test_round_trip_cost_approx_0_2pct(self):
        """A typical SH round-trip burns ~0.20% in friction:
          - slip BUY 5bps + SELL 5bps = 10 bps on notional
          - commission 2 * 0.025% = 5 bps
          - stamp 0.05% on SELL only = 5 bps
          - transfer 2 * 0.001% = 0.2 bps
        Total ~20 bps = 0.20% = ~200 RMB on 100k notional.
        """
        comm, stamp, transfer, slip = self._models()
        buy = total_transaction_cost(
            price=100.0, shares=1000, side="BUY", symbol="SH600519",
            commission_model=comm, stamp_duty_model=stamp,
            transfer_fee_model=transfer, slippage_model=slip,
        )
        sell = total_transaction_cost(
            price=100.0, shares=1000, side="SELL", symbol="SH600519",
            commission_model=comm, stamp_duty_model=stamp,
            transfer_fee_model=transfer, slippage_model=slip,
        )
        paid = -buy["net_amount"]
        received = sell["net_amount"]
        friction = paid - received
        # On 100k notional, ~0.18-0.22% = ~180-220 RMB.
        assert 180 < friction < 230, f"friction={friction:.2f}"
