"""Slippage: the price moves against us between the reference price and the fill."""

from __future__ import annotations

import math
from dataclasses import dataclass

from trader.core.types import Side
from trader.market.base import TickSchedule


@dataclass(frozen=True, slots=True)
class SlippageModel:
    """Adverse slippage of `bps` basis points, optionally widened by volatility, then tick rounding.

    Buys round up and sells round down to the market's tick, so rounding never helps us.
    `atr_k` adds k x (ATR / price) of extra slippage for volatile stocks (0 disables it).
    """

    bps: float = 10.0
    atr_k: float = 0.0

    def fill_price(self, raw: float, side: Side | str, ticks: TickSchedule, atr_pct: float | None = None) -> float:
        if not (math.isfinite(raw) and raw > 0):
            raise ValueError(f"cannot fill at price {raw}")
        volatile = self.atr_k and atr_pct is not None and math.isfinite(atr_pct)  # no ATR during warm-up
        rate = self.bps / 10_000 + (self.atr_k * atr_pct if volatile else 0.0)
        # == not `is`: a side read back from storage is the plain string "BUY"
        if side == Side.BUY:
            return ticks.round_up(raw * (1 + rate))
        if side == Side.SELL:
            return ticks.round_down(raw * (1 - rate))
        raise ValueError(f"unknown side {side!r}")
