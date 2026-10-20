"""Provider-neutral types for daily market data.

Canonical in-memory bar frame (what every provider returns):
    date    datetime64[ns], tz-naive midnight of the IST session date, ascending, unique
    open, high, low, close   float64
    volume  int64
Prices are adjusted for splits and bonuses (not dividends), matching Upstox history.
Storage converts `date` to Parquet date32 (see trader.storage.parquet).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

import pandas as pd

BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str  # NSE trading symbol ("RELIANCE") or index name ("Nifty 50")
    instrument_key: str  # Upstox key: "NSE_EQ|INE002A01018" or "NSE_INDEX|Nifty 50"
    kind: Literal["EQ", "INDEX"]
    name: str
    isin: str | None = None
    instrument_type: str | None = None  # Upstox type: EQ, BE, SM, INDEX, ...
    tick_size: float | None = None  # rupees
    lot_size: int = 1
    exchange_token: str | None = None

    @property
    def yf_ticker(self) -> str | None:
        return f"{self.symbol}.NS" if self.kind == "EQ" else None


class ProviderError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, error_codes: Sequence[str] = ()):
        super().__init__(message)
        self.status_code = status_code
        self.error_codes = tuple(error_codes)


class AuthError(ProviderError):
    """Missing, invalid or expired credentials. Never retried."""


class RateLimitedError(ProviderError):
    """HTTP 429 after all retries."""


class TransientError(ProviderError):
    """Network failure or HTTP 5xx after all retries."""


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    ok: bool
    provider: str
    detail: str
    latency_ms: float | None = None


class MarketDataProvider(Protocol):
    name: str

    def fetch_daily(self, inst: Instrument, start: date, end: date) -> pd.DataFrame: ...

    def healthcheck(self) -> ProviderHealth: ...


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            **{c: pd.Series(dtype="float64") for c in PRICE_COLUMNS},
            "volume": pd.Series(dtype="int64"),
        }
    )


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the canonical frame: dtypes, ascending dates, last value wins on duplicates."""
    if df.empty:
        return empty_bars()
    out = df[BAR_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize().astype("datetime64[ns]")
    for column in PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column]).astype("float64")
    out["volume"] = pd.to_numeric(out["volume"]).round().astype("int64")
    out = out.drop_duplicates(subset="date", keep="last").sort_values("date", kind="stable")
    return out.reset_index(drop=True)
