"""Market profiles: everything that differs between exchanges, kept out of the trading engine.

A `MarketProfile` bundles the market's currency, time zone, session hours, calendar and regime
index, benchmarks, risk-free rate, share-quantity rules, tick sizes, price-band rule and cost model.
The engine, sizing, broker and features read it once when they are built. Nothing in a per-bar
loop looks values up through it, so the indirection has no run-time cost and cannot change results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from typing import Protocol


def round_half_up(value: float, decimals: int = 2) -> float:
    """Round like a contract note (half away from zero for positive amounts), not banker's rounding."""
    scale = 10**decimals
    return math.floor(value * scale + 0.5 + 1e-9) / scale


# Costs ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Itemized charges for one fill, in the market's currency. `total` is the sum of components."""

    total: float
    components: tuple[tuple[str, float], ...]

    def get(self, name: str) -> float:
        return next((v for k, v in self.components if k == name), 0.0)

    def as_dict(self) -> dict[str, float]:
        return dict(self.components)

    @staticmethod
    def of(**components: float) -> CostBreakdown:
        items = tuple((k, float(v)) for k, v in components.items())
        return CostBreakdown(round(sum(v for _, v in items), 10), items)


class CostModel(Protocol):
    name: str

    def buy(self, turnover: float, quantity: float) -> CostBreakdown: ...

    def sell(self, turnover: float, quantity: float, *, first_sell_of_day: bool = True) -> CostBreakdown: ...


def round_trip_cost(model: CostModel, buy_turnover: float, sell_turnover: float, quantity: float) -> float:
    return model.buy(buy_turnover, quantity).total + model.sell(sell_turnover, quantity).total


# Quantities and prices ------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ShareRules:
    """How many shares may be traded: whole shares (step 1) or fractions (e.g. step 0.0001)."""

    fractional: bool
    step: float
    min_quantity: float
    min_order_value: float = 0.0

    def round_down(self, quantity: float) -> float:
        if quantity <= 0 or not math.isfinite(quantity):
            return 0.0
        # Round the ratio first so 0.3 / 0.1 = 2.9999999999999996 counts as 3 steps, but never
        # round up a real shortfall (10000 / 3333.333334 = 2.9999999994 is 2 shares, not 3).
        steps = math.floor(round(quantity / self.step, 9))
        return round(steps * self.step, 10)

    def is_tradable(self, quantity: float, price: float) -> bool:
        return quantity >= self.min_quantity and quantity * price >= self.min_order_value


@dataclass(frozen=True, slots=True)
class TickSchedule:
    """Tick size by price tier: ((upper bound, tick), ...), bounds exclusive and ascending."""

    tiers: tuple[tuple[float, float], ...]

    def tick(self, price: float) -> float:
        for upper, tick in self.tiers:
            if price < upper:
                return tick
        return self.tiers[-1][1]

    def round_up(self, price: float) -> float:
        tick = self.tick(price)
        return round(math.ceil(price / tick - 1e-9) * tick, 6)

    def round_down(self, price: float) -> float:
        tick = self.tick(price)
        return round(math.floor(price / tick + 1e-9) * tick, 6)

    def round_nearest(self, price: float) -> float:
        """Nearest tick, halves rounded up (1002.55 on a 0.05 tick stays 1002.55)."""
        tick = self.tick(price)
        return round(math.floor(price / tick + 0.5 + 1e-9) * tick, 6)


class PriceBandRule(Protocol):
    """Detects days on which a stock was locked at its daily price limit (no trading possible)."""

    def locked_lower(self, prev_close: float, high: float, low: float, close: float, tick: float) -> bool: ...

    def locked_upper(self, prev_close: float, high: float, low: float, close: float, tick: float) -> bool: ...


class NoPriceBands:
    """Markets without hard daily limits (e.g. US stocks, which only pause briefly)."""

    def locked_lower(self, prev_close: float, high: float, low: float, close: float, tick: float) -> bool:
        return False

    def locked_upper(self, prev_close: float, high: float, low: float, close: float, tick: float) -> bool:
        return False


# Profile --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MarketProfile:
    code: str
    name: str
    currency: str
    timezone: str
    session_open: time
    session_close: time
    calendar_index: str  # index whose bars define the trading calendar
    regime_index: str  # index used for the market-trend (regime) filter
    benchmarks: tuple[str, ...]
    risk_free_rate: float  # annual, for Sharpe ratios and the cash benchmark
    trading_days_per_year: int  # for annualizing volatility and returns
    settlement_days: int
    shares: ShareRules
    ticks: TickSchedule
    price_bands: PriceBandRule
    cost_model: CostModel
    shorting_allowed: bool = False
