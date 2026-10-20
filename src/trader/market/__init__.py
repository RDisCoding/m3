"""Market profiles. The active market comes from the TRADER_MARKET setting (default IN_NSE)."""

from trader.market.base import (
    CostBreakdown,
    CostModel,
    MarketProfile,
    NoPriceBands,
    ShareRules,
    TickSchedule,
)
from trader.market.india import INDIA_NSE

PROFILES: dict[str, MarketProfile] = {INDIA_NSE.code: INDIA_NSE}


def get_profile(code: str | None = None) -> MarketProfile:
    if code is None:
        from trader.settings import get_settings

        code = get_settings().market
    try:
        return PROFILES[code]
    except KeyError:
        raise KeyError(f"unknown market {code!r}; available: {sorted(PROFILES)}") from None


__all__ = ["PROFILES", "CostBreakdown", "CostModel", "MarketProfile", "NoPriceBands", "ShareRules", "TickSchedule",
           "get_profile", "INDIA_NSE"]
