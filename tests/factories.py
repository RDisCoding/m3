"""Builders for synthetic test data."""

import pandas as pd

from trader.data.providers.base import Instrument, normalize_bars


def bars(dates, closes, opens=None, volume=1000) -> pd.DataFrame:
    """Canonical bars; open defaults to close, high/low bracket the body by 1%."""
    opens = list(closes) if opens is None else list(opens)
    closes = list(closes)
    return normalize_bars(pd.DataFrame({
        "date": pd.to_datetime(list(dates)),
        "open": opens,
        "high": [max(o, c) * 1.01 for o, c in zip(opens, closes)],
        "low": [min(o, c) * 0.99 for o, c in zip(opens, closes)],
        "close": closes,
        "volume": [volume] * len(closes),
    }))


def action_rows(*rows) -> pd.DataFrame:
    """Corporate-action rows as stored: (symbol, ex_date, subject[, face_value])."""
    records = []
    for row in rows:
        symbol, ex_date, subject, *rest = row
        records.append({"symbol": symbol, "isin": None, "series": "EQ", "ex_date": ex_date, "record_date": None,
                        "subject": subject, "face_value": rest[0] if rest else None})
    return pd.DataFrame(records)


def equity(symbol: str, isin: str = "INE000A01011") -> Instrument:
    return Instrument(symbol, f"NSE_EQ|{isin}", "EQ", f"{symbol} LTD", isin=isin, instrument_type="EQ")


NIFTY = Instrument("Nifty 50", "NSE_INDEX|Nifty 50", "INDEX", "Nifty 50", instrument_type="INDEX")
