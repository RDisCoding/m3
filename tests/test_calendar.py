import json
from datetime import date, datetime, timedelta, timezone

from trader.data.calendar import (
    Holiday,
    TradingCalendar,
    load_holidays,
    parse_holidays,
    store_holidays,
)
from trader.storage.db import connect

IST = timezone(timedelta(hours=5, minutes=30))
OBSERVED = [date(2026, 10, 15), date(2026, 10, 16), date(2026, 10, 19)]  # Thu, Fri, Mon
HOLIDAYS = [
    Holiday(date(2026, 10, 20), "Dussehra", "TRADING_HOLIDAY", True, False),
    Holiday(date(2026, 11, 8), "Muhurat Trading", "SPECIAL_TIMING", False, True),  # a Sunday
]


def test_parse_real_upstox_holiday_list(fixtures):
    holidays = {h.date: h for h in parse_holidays(json.loads((fixtures / "upstox_holidays.json").read_text())["data"])}
    assert holidays[date(2026, 10, 20)].nse_closed and holidays[date(2026, 10, 20)].description == "Dussehra"
    budget_sunday = holidays[date(2026, 2, 1)]
    assert budget_sunday.nse_special and not budget_sunday.nse_closed
    assert all(h.nse_closed or h.nse_special for h in holidays.values())  # MCX-only entries are dropped


def test_observed_history_is_ground_truth_and_rules_apply_afterwards():
    cal = TradingCalendar(OBSERVED, HOLIDAYS)
    assert cal.is_session(date(2026, 10, 19)) and not cal.is_session(date(2026, 10, 17))
    assert not cal.is_session(date(2026, 10, 20))  # holiday after the observed range
    assert cal.is_session(date(2026, 10, 21))
    assert cal.is_session(date(2026, 11, 8))  # special Sunday session
    assert cal.next_session(date(2026, 10, 19)) == date(2026, 10, 21)
    assert cal.prev_session(date(2026, 10, 21)) == date(2026, 10, 19)
    assert cal.sessions(date(2026, 10, 16), date(2026, 10, 22)) == [
        date(2026, 10, 16), date(2026, 10, 19), date(2026, 10, 21), date(2026, 10, 22)]


def test_observed_gap_inside_history_is_not_a_session():
    cal = TradingCalendar([date(2026, 10, 1), date(2026, 10, 5)], [])
    assert not cal.is_session(date(2026, 10, 2))  # a weekday holiday learned from the data alone


def test_latest_completed_session_respects_the_close():
    cal = TradingCalendar(OBSERVED, HOLIDAYS)
    assert cal.latest_completed_session(datetime(2026, 10, 21, 15, 29, tzinfo=IST)) == date(2026, 10, 19)
    assert cal.latest_completed_session(datetime(2026, 10, 21, 15, 30, tzinfo=IST)) == date(2026, 10, 21)
    assert cal.latest_completed_session(datetime(2026, 10, 20, 18, 0, tzinfo=IST)) == date(2026, 10, 19)


def test_holidays_round_trip_through_the_database(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    store_holidays(conn, HOLIDAYS, "2026-10-20T00:00:00+00:00")
    store_holidays(conn, HOLIDAYS, "2026-10-21T00:00:00+00:00")  # upsert, no duplicates
    assert load_holidays(conn) == HOLIDAYS


def test_settlement_holidays_are_neither_closures_nor_special_sessions():
    rows = [{"date": "2026-02-19", "description": "Chhatrapati Shivaji Maharaj Jayanti", "holiday_type": "SETTLEMENT_HOLIDAY",
             "closed_exchanges": [], "open_exchanges": [{"exchange": "NSE"}]}]
    assert parse_holidays(rows) == []
