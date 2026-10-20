from dataclasses import replace
from datetime import time

import pytest

from trader.broker.slippage import SlippageModel
from trader.core.types import Side
from trader.market import INDIA_NSE, NoPriceBands, ShareRules, TickSchedule, get_profile
from trader.market.base import CostBreakdown, round_half_up, round_trip_cost
from trader.market.india import (
    NSE_TICKS,
    NSEPriceBands,
    ZerodhaDeliveryCosts,
    ZerodhaIntradayCosts,
)

COSTS = ZerodhaDeliveryCosts()


@pytest.mark.parametrize("label, breakdown, expected", [
    # Hand-computed in the main plan (section 8.8), to the paisa.
    ("buy 3 @ 412.90", COSTS.buy(3 * 412.90, 3), {"stt": 1.24, "exchange": 0.04, "sebi": 0.0, "stamp": 0.19, "gst": 0.01, "total": 1.48}),
    ("sell 3 @ 450", COSTS.sell(3 * 450.0, 3), {"stt": 1.35, "exchange": 0.04, "gst": 0.01, "dp": 15.34, "total": 16.74}),
    ("buy Rs 1 lakh", COSTS.buy(100_000, 100), {"stt": 100.0, "exchange": 3.07, "sebi": 0.10, "stamp": 15.0, "gst": 0.57, "total": 118.74}),
    ("sell Rs 1 lakh", COSTS.sell(100_000, 100), {"stt": 100.0, "exchange": 3.07, "sebi": 0.10, "gst": 0.57, "dp": 15.34, "total": 119.08}),
])
def test_zerodha_delivery_matches_hand_computed_values(label, breakdown, expected):
    for name, value in expected.items():
        actual = breakdown.total if name == "total" else breakdown.get(name)
        assert actual == pytest.approx(value, abs=1e-9), f"{label}: {name}"


def test_delivery_cost_rules():
    assert COSTS.sell(1350.0, 3, first_sell_of_day=False).get("dp") == 0.0  # DP once per scrip per day
    assert COSTS.buy(0.0, 0).total == 0.0 and COSTS.sell(0.0, 0).total == 0.0
    big = COSTS.buy(1_000_000, 1)
    assert big.get("gst") == pytest.approx(round_half_up(0.18 * (30.70 + 1.0)))  # GST excludes STT and stamp duty
    assert round_trip_cost(COSTS, 1238.70, 1350.0, 3) == pytest.approx(18.22)


def test_intraday_costs_cap_brokerage_and_charge_stt_on_sells_only():
    model = ZerodhaIntradayCosts()
    assert model.buy(10_000_000, 1).get("brokerage") == 20.0
    assert model.buy(10_000, 1).get("brokerage") == pytest.approx(3.0)
    assert model.buy(10_000, 1).get("stt") == 0.0 and model.sell(10_000, 1).get("stt") == pytest.approx(2.5)
    assert model.sell(10_000, 1).get("dp", ) == 0.0


def test_round_half_up_is_not_bankers_rounding():
    assert round_half_up(0.125) == 0.13 and round_half_up(0.124999) == 0.12 and round_half_up(2.675) == 2.68


def test_share_rules():
    whole = INDIA_NSE.shares
    assert whole.round_down(3.99) == 3.0 and whole.round_down(0.7) == 0.0 and whole.round_down(-1) == 0.0
    assert whole.is_tradable(1, 500) and not whole.is_tradable(0, 500)
    fractional = ShareRules(fractional=True, step=0.0001, min_quantity=0.0001, min_order_value=1.0)
    assert fractional.round_down(0.123456) == pytest.approx(0.1234)
    assert not fractional.is_tradable(0.001, 500) and fractional.is_tradable(0.01, 500)


@pytest.mark.parametrize("price, tick, up, down", [
    (249.991, 0.01, 250.0, 249.99), (412.9125, 0.05, 412.95, 412.90), (1186.43, 0.10, 1186.50, 1186.40),
    (7004.2, 0.50, 7004.5, 7004.0), (25001.0, 5.0, 25005.0, 25000.0),
])
def test_nse_tick_schedule(price, tick, up, down):
    assert NSE_TICKS.tick(price) == tick
    assert NSE_TICKS.round_up(price) == pytest.approx(up) and NSE_TICKS.round_down(price) == pytest.approx(down)
    assert NSE_TICKS.round_up(up) == pytest.approx(up)  # already on a tick: unchanged


def test_nse_price_band_lock_detection():
    bands = NSEPriceBands()
    assert bands.locked_lower(prev_close=100.0, high=95.0, low=95.0, close=95.0, tick=0.05)  # stuck at -5% all day
    assert bands.locked_lower(100.0, 95.05, 95.0, 95.0, 0.05)  # within one tick of the band all day
    assert not bands.locked_lower(100.0, 99.0, 95.0, 95.0, 0.05)  # traded from 99 down: stops above 95 filled
    assert not bands.locked_lower(100.0, 93.0, 93.0, 93.0, 0.05)  # -7% is not a band
    assert bands.locked_upper(100.0, 110.0, 110.0, 110.0, 0.05)
    assert not bands.locked_upper(100.0, 110.0, 101.0, 110.0, 0.05)  # opened 101 and rallied: an open buy filled
    # LT on 2026-08-03 (a stock with derivatives, so no band): +2.19% closing at its high is not a lock.
    assert not bands.locked_upper(3938.75, 4025.0, 3962.5, 4025.0, 0.10)
    assert not NoPriceBands().locked_lower(100.0, 95.0, 95.0, 95.0, 0.05)


def test_slippage_is_adverse_and_tick_rounded():
    slip = SlippageModel(bps=10)
    assert slip.fill_price(412.50, Side.BUY, NSE_TICKS) == pytest.approx(412.95)
    assert slip.fill_price(450.00, Side.SELL, NSE_TICKS) == pytest.approx(449.55)
    wide = SlippageModel(bps=10, atr_k=0.1)
    assert wide.fill_price(100.0, Side.BUY, NSE_TICKS, atr_pct=0.05) == pytest.approx(100.60)  # 10 bps + 0.1 x 5% = 0.6%
    assert wide.fill_price(100.0, Side.BUY, NSE_TICKS, atr_pct=float("nan")) == pytest.approx(100.10)  # warm-up: base only


def test_slippage_accepts_sides_read_back_from_storage_and_rejects_bad_input():
    slip = SlippageModel(bps=10)
    assert slip.fill_price(100.0, "BUY", NSE_TICKS) == slip.fill_price(100.0, Side.BUY, NSE_TICKS) == pytest.approx(100.10)
    assert slip.fill_price(100.0, "SELL", NSE_TICKS) == pytest.approx(99.90)
    for bad_side in ("buy", "HOLD", None):
        with pytest.raises(ValueError):
            slip.fill_price(100.0, bad_side, NSE_TICKS)
    for bad_price in (float("nan"), 0.0, -5.0):
        with pytest.raises(ValueError):
            slip.fill_price(bad_price, Side.BUY, NSE_TICKS)


@pytest.mark.parametrize("price, expected", [
    # half ticks round up on every tier (the auditor found 1002.55 -> 1002.5 and 4999.95 -> 4999.9 before)
    (1002.55, 1002.60), (4999.95, 5000.0), (412.925, 412.95), (412.924, 412.90), (249.995, 250.0), (100.004, 100.0),
])
def test_round_nearest_keeps_prices_on_the_grid(price, expected):
    assert NSE_TICKS.round_nearest(price) == pytest.approx(expected, abs=1e-9)


def test_share_rounding_never_exceeds_what_cash_allows():
    whole = INDIA_NSE.shares
    assert whole.round_down(10_000 / 3333.333334) == 2.0  # 2.9999999994 shares affordable: buy 2
    assert whole.round_down(0.3 / 0.1) == 3.0  # float noise below a whole number still counts
    fractional = ShareRules(fractional=True, step=0.1, min_quantity=0.1)
    assert fractional.round_down(0.3) == pytest.approx(0.3)


def test_profile_registry():
    assert get_profile("IN_NSE") is INDIA_NSE
    assert INDIA_NSE.session_close == time(15, 30) and INDIA_NSE.currency == "INR"
    with pytest.raises(KeyError):
        get_profile("XX")


def test_engine_parts_work_with_another_market_profile():
    """A different profile (fractional shares, no bands, flat costs) drops into the same components."""

    class FlatCosts:
        name = "flat_test"

        def buy(self, turnover, quantity):
            return CostBreakdown.of(fee=0.0)

        def sell(self, turnover, quantity, *, first_sell_of_day=True):
            return CostBreakdown.of(fee=round(turnover * 20.6e-6, 6))

    other = replace(INDIA_NSE, code="TEST", currency="USD", shares=ShareRules(True, 0.0001, 0.0001, 1.0),
                    ticks=TickSchedule(((float("inf"), 0.01),)), price_bands=NoPriceBands(), cost_model=FlatCosts())
    assert other.shares.round_down(20 / 187.37) == pytest.approx(0.1067)
    assert SlippageModel(bps=3).fill_price(187.37, Side.BUY, other.ticks) == pytest.approx(187.43)
    assert round_trip_cost(other.cost_model, 20.0, 20.0, 0.1067) == pytest.approx(0.000412)
    assert not other.price_bands.locked_lower(100.0, 99.0, 95.0, 95.0, 0.01)
