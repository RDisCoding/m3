"""NSE India, cash segment, delivery trades: the market the prototype trades."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time

from trader.market.base import (
    CostBreakdown,
    MarketProfile,
    ShareRules,
    TickSchedule,
    round_half_up,
)


@dataclass(frozen=True, slots=True)
class ZerodhaDeliveryCosts:
    """Zerodha equity delivery charges as published on 2026-10-02.

    Brokerage 0; STT 0.1% on buy and sell; NSE transaction charge 0.00307%; SEBI fee Rs 10 per crore;
    stamp duty 0.015% on buys; GST 18% on brokerage + exchange + SEBI charges; DP charge Rs 15.34 per
    scrip on each day it is sold (GST included). Each component is computed unrounded, rounded to the
    paisa, and the total is the sum of the rounded components.
    """

    name: str = "zerodha_delivery_2026_10"
    brokerage_pct: float = 0.0
    stt_pct: float = 0.001
    exchange_pct: float = 0.0000307
    sebi_pct: float = 0.000001
    stamp_buy_pct: float = 0.00015
    gst_pct: float = 0.18
    dp_charge: float = 15.34

    def _common(self, turnover: float) -> tuple[float, float, float, float, float]:
        brokerage = turnover * self.brokerage_pct
        stt = turnover * self.stt_pct
        exchange = turnover * self.exchange_pct
        sebi = turnover * self.sebi_pct
        gst = (brokerage + exchange + sebi) * self.gst_pct
        return brokerage, stt, exchange, sebi, gst

    def buy(self, turnover: float, quantity: float) -> CostBreakdown:
        if turnover <= 0:
            return CostBreakdown.of()
        brokerage, stt, exchange, sebi, gst = self._common(turnover)
        return CostBreakdown.of(
            brokerage=round_half_up(brokerage), stt=round_half_up(stt), exchange=round_half_up(exchange),
            sebi=round_half_up(sebi), stamp=round_half_up(turnover * self.stamp_buy_pct), gst=round_half_up(gst),
        )

    def sell(self, turnover: float, quantity: float, *, first_sell_of_day: bool = True) -> CostBreakdown:
        if turnover <= 0:
            return CostBreakdown.of()
        brokerage, stt, exchange, sebi, gst = self._common(turnover)
        return CostBreakdown.of(
            brokerage=round_half_up(brokerage), stt=round_half_up(stt), exchange=round_half_up(exchange),
            sebi=round_half_up(sebi), gst=round_half_up(gst), dp=self.dp_charge if first_sell_of_day else 0.0,
        )


@dataclass(frozen=True, slots=True)
class ZerodhaIntradayCosts:
    """Zerodha equity intraday charges (2026-10-02). Kept for completeness; the prototype trades delivery."""

    name: str = "zerodha_intraday_2026_10"
    brokerage_pct: float = 0.0003
    brokerage_cap: float = 20.0
    stt_sell_pct: float = 0.00025
    exchange_pct: float = 0.0000307
    sebi_pct: float = 0.000001
    stamp_buy_pct: float = 0.00003
    gst_pct: float = 0.18

    def _charges(self, turnover: float, *, is_buy: bool) -> CostBreakdown:
        if turnover <= 0:
            return CostBreakdown.of()
        brokerage = min(turnover * self.brokerage_pct, self.brokerage_cap)
        exchange = turnover * self.exchange_pct
        sebi = turnover * self.sebi_pct
        return CostBreakdown.of(
            brokerage=round_half_up(brokerage),
            stt=0.0 if is_buy else round_half_up(turnover * self.stt_sell_pct),
            exchange=round_half_up(exchange), sebi=round_half_up(sebi),
            stamp=round_half_up(turnover * self.stamp_buy_pct) if is_buy else 0.0,
            gst=round_half_up((brokerage + exchange + sebi) * self.gst_pct),
        )

    def buy(self, turnover: float, quantity: float) -> CostBreakdown:
        return self._charges(turnover, is_buy=True)

    def sell(self, turnover: float, quantity: float, *, first_sell_of_day: bool = True) -> CostBreakdown:
        return self._charges(turnover, is_buy=False)


@dataclass(frozen=True, slots=True)
class NSEPriceBands:
    """Detects stocks locked at an NSE daily price band all day, without per-stock band data.

    NSE bands are 2%, 5%, 10% or 20% either way (stocks with derivatives have none). A day counts as
    locked at the lower band only when the stock never traded away from it: the whole day's range is
    within one tick and the move from the previous close matches a band size; symmetrically for the
    upper band. Then no sell (lower) or buy (upper) could have filled. A stock that traded freely and
    only closed at its band is not locked: a stop above the band filled on the way down.
    Prices must be the traded prices of that day, not adjusted ones.
    """

    bands: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    tolerance: float = 0.003

    def _at_band(self, move: float) -> bool:
        return any(abs(abs(move) - band) <= self.tolerance for band in self.bands)

    def locked_lower(self, prev_close: float, high: float, low: float, close: float, tick: float) -> bool:
        if prev_close <= 0:
            return False
        move = close / prev_close - 1
        return move < 0 and high - low <= tick * 1.5 and self._at_band(move)

    def locked_upper(self, prev_close: float, high: float, low: float, close: float, tick: float) -> bool:
        if prev_close <= 0:
            return False
        move = close / prev_close - 1
        return move > 0 and high - low <= tick * 1.5 and self._at_band(move)


# Price-based tick sizes NSE applies to equities (reviewed monthly from the previous month's price).
# Before mid-2025 most stocks used Rs 0.05; the difference is far smaller than modelled slippage.
NSE_TICKS = TickSchedule(((250.0, 0.01), (1_000.0, 0.05), (5_000.0, 0.10), (10_000.0, 0.50), (20_000.0, 1.00), (math.inf, 5.00)))

INDIA_NSE = MarketProfile(
    code="IN_NSE",
    name="NSE India equities, cash segment, delivery",
    currency="INR",
    timezone="Asia/Kolkata",
    session_open=time(9, 15),
    session_close=time(15, 30),
    calendar_index="Nifty 50",
    regime_index="Nifty 50",
    benchmarks=("Nifty 50", "Nifty 500"),
    risk_free_rate=0.065,
    trading_days_per_year=248,  # observed average 2015-2025
    settlement_days=1,
    shares=ShareRules(fractional=False, step=1.0, min_quantity=1.0),
    ticks=NSE_TICKS,
    price_bands=NSEPriceBands(),
    cost_model=ZerodhaDeliveryCosts(),
    shorting_allowed=False,
)
