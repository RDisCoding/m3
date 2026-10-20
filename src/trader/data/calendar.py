"""NSE trading calendar.

Sessions up to the last stored Nifty 50 bar are *observed*: a date is a session exactly when the
index has a daily bar. That is ground truth and includes special sessions such as Diwali Muhurat
trading and Budget-day Saturdays. Dates after the last observed session follow the weekday rule,
minus the holidays Upstox publishes for the current year, plus the special sessions in that list.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from bisect import bisect_left, bisect_right

MARKET_CLOSE = time(15, 30)


@dataclass(frozen=True, slots=True)
class Holiday:
    date: date
    description: str
    holiday_type: str
    nse_closed: bool
    nse_special: bool  # NSE holds a special session that day


def parse_holidays(rows: Iterable[dict]) -> list[Holiday]:
    """Parse Upstox /v2/market/holidays rows, keeping only entries that affect NSE equities."""
    out: dict[date, Holiday] = {}
    for row in rows:
        if (row.get("holiday_type") or "").upper() == "SETTLEMENT_HOLIDAY":
            continue  # trading happens; only settlement is closed
        closed = "NSE" in (row.get("closed_exchanges") or [])
        special = any((o or {}).get("exchange") == "NSE" for o in (row.get("open_exchanges") or []))
        if not closed and not special:
            continue
        day = date.fromisoformat(row["date"])
        existing = out.get(day)
        holiday = Holiday(day, row.get("description") or "", row.get("holiday_type") or "", closed and not special, special)
        # If a date has both a closure and a special session (e.g. Diwali + Muhurat), the session wins.
        if existing is None or holiday.nse_special:
            out[day] = holiday
    return sorted(out.values(), key=lambda h: h.date)


def store_holidays(conn: sqlite3.Connection, holidays: Iterable[Holiday], fetched_at: str) -> int:
    rows = [(h.date.isoformat(), h.description, h.holiday_type, int(h.nse_closed), int(h.nse_special), fetched_at) for h in holidays]
    conn.executemany(
        "INSERT INTO market_holidays (date, description, holiday_type, nse_closed, nse_special, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(date) DO UPDATE SET description=excluded.description, "
        "holiday_type=excluded.holiday_type, nse_closed=excluded.nse_closed, nse_special=excluded.nse_special, "
        "fetched_at=excluded.fetched_at",
        rows,
    )
    return len(rows)


def load_holidays(conn: sqlite3.Connection) -> list[Holiday]:
    rows = conn.execute("SELECT date, description, holiday_type, nse_closed, nse_special FROM market_holidays ORDER BY date").fetchall()
    return [Holiday(date.fromisoformat(r[0]), r[1], r[2], bool(r[3]), bool(r[4])) for r in rows]


class TradingCalendar:
    def __init__(self, observed: Iterable[date], holidays: Iterable[Holiday] = ()):
        self._observed = sorted(set(observed))
        self._observed_set = set(self._observed)
        self.last_observed: date | None = self._observed[-1] if self._observed else None
        self.first_observed: date | None = self._observed[0] if self._observed else None
        holidays = list(holidays)
        self._closed = {h.date for h in holidays if h.nse_closed}
        self._special = {h.date for h in holidays if h.nse_special}
        self.holidays = {h.date: h for h in holidays}

    def is_session(self, day: date) -> bool:
        if self.last_observed is not None and self.first_observed <= day <= self.last_observed:
            return day in self._observed_set
        return (day.weekday() < 5 and day not in self._closed) or day in self._special

    def sessions(self, start: date, end: date) -> list[date]:
        if end < start:
            return []
        out: list[date] = []
        if self._observed:
            lo, hi = bisect_left(self._observed, start), bisect_right(self._observed, end)
            out.extend(self._observed[lo:hi])
        cursor = start
        while cursor <= end:
            observed_span = self.last_observed is not None and self.first_observed <= cursor <= self.last_observed
            if not observed_span and self.is_session(cursor):
                out.append(cursor)
            cursor += timedelta(days=1)
        return sorted(out)

    def next_session(self, day: date, n: int = 1) -> date:
        cursor = day
        for _ in range(n):
            cursor += timedelta(days=1)
            while not self.is_session(cursor):
                cursor += timedelta(days=1)
        return cursor

    def prev_session(self, day: date, n: int = 1) -> date:
        cursor = day
        for _ in range(n):
            cursor -= timedelta(days=1)
            while not self.is_session(cursor):
                cursor -= timedelta(days=1)
        return cursor

    def latest_completed_session(self, now_local: datetime, close: time = MARKET_CLOSE) -> date:
        """Today if it is a session and the market has closed, else the previous session.

        `now_local` is the market's local time; `close` comes from the market profile (NSE: 15:30).
        """
        today = now_local.date()
        if self.is_session(today) and now_local.time() >= close:
            return today
        return self.prev_session(today)
