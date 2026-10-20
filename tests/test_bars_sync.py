from datetime import date

import pandas as pd
import pytest
from factories import NIFTY, bars, equity

from trader.data.bars import SOURCE_HISTORY, SOURCE_LIVE, BarSync
from trader.data.providers.base import AuthError, ProviderError
from trader.storage.db import connect
from trader.storage.parquet import BarStore

AAA, BBB = equity("AAA", "INE000A01011"), equity("BBB", "INE000B01011")
DATES = list(pd.bdate_range("2026-09-01", "2026-10-16"))


class FakeProvider:
    def __init__(self, history: dict, live: pd.DataFrame | None = None, fail: set | None = None):
        self.history, self.live, self.fail = history, live, fail or set()
        self.calls: list[tuple] = []

    def fetch_daily(self, inst, start, end):
        self.calls.append((inst.symbol, start, end))
        if inst.symbol in self.fail:
            raise ProviderError("boom")
        frame = self.history[inst.symbol]
        return frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))].reset_index(drop=True)

    def fetch_latest_daily(self, insts):
        keys = {i.instrument_key for i in insts}
        return self.live[self.live["instrument_key"].isin(keys)].reset_index(drop=True)


def series(n_days, start_price=100.0):
    return bars(DATES[:n_days], [start_price + i for i in range(n_days)])


@pytest.fixture
def env(tmp_path):
    return BarStore(tmp_path / "daily"), BarStore(tmp_path / "index"), connect(tmp_path / "t.sqlite")


def test_backfill_is_resumable(env):
    daily, index, conn = env
    provider = FakeProvider({"AAA": series(30), "Nifty 50": series(30, 22000.0)})
    sync = BarSync(provider, daily, index, conn)
    results = sync.backfill([NIFTY, AAA], date(2015, 1, 1), date(2026, 10, 16))
    assert [r.status for r in results] == ["ok", "ok"] and daily.exists("AAA") and index.exists("Nifty 50")
    meta = sync.meta(AAA)
    assert (meta["rows"], meta["first_date"], meta["backfill_from"]) == (30, "2026-09-01", "2015-01-01")
    again = sync.backfill([NIFTY, AAA], date(2015, 1, 1), date(2026, 10, 16))
    assert [r.status for r in again] == ["skipped", "skipped"] and len(provider.calls) == 2
    sync.backfill([AAA], date(2015, 1, 1), date(2026, 10, 16), force=True)
    assert len(provider.calls) == 3


def test_errors_are_isolated_but_auth_errors_stop_the_run(env):
    daily, index, conn = env
    provider = FakeProvider({"AAA": series(10), "BBB": series(10)}, fail={"BBB"})
    results = BarSync(provider, daily, index, conn).backfill([AAA, BBB], date(2015, 1, 1), date(2026, 10, 16))
    assert [r.status for r in results] == ["ok", "error"]

    class Rejecting(FakeProvider):
        def fetch_daily(self, inst, start, end):
            raise AuthError("token rejected")

    with pytest.raises(AuthError):
        BarSync(Rejecting({}), daily, index, conn).backfill([BBB], date(2015, 1, 1), date(2026, 10, 16), force=True)


def test_refresh_fetches_only_the_overlap_window(env):
    daily, index, conn = env
    provider = FakeProvider({"AAA": series(25)})
    sync = BarSync(provider, daily, index, conn)
    sync.backfill([AAA], date(2015, 1, 1), DATES[19].date())  # 20 days stored
    results = sync.refresh([AAA], DATES[24].date(), today=date(2026, 10, 20), new_start=date(2015, 1, 1), overlap=10)
    assert results[0].added == 5 and not results[0].repulled
    assert provider.calls[-1] == ("AAA", DATES[10].date(), DATES[24].date())  # 10th-last stored session onward
    assert len(daily.read("AAA")) == 25


def test_vendor_readjustment_triggers_a_full_repull(env):
    daily, index, conn = env
    history = series(20, 1000.0)
    provider = FakeProvider({"AAA": history})
    sync = BarSync(provider, daily, index, conn)
    sync.backfill([AAA], date(2015, 1, 1), DATES[19].date())
    adjusted = history.copy()
    adjusted[["open", "high", "low", "close"]] /= 2  # a 1:1 bonus re-adjusts all of history
    provider.history["AAA"] = adjusted
    result = sync.refresh([AAA], DATES[19].date(), today=date(2026, 10, 20), new_start=date(2015, 1, 1))[0]
    assert result.repulled
    assert daily.read_bars("AAA")["close"].tolist() == adjusted["close"].tolist()
    assert conn.execute("SELECT COUNT(*) FROM repull_events WHERE name = 'AAA'").fetchone()[0] == 1


def test_todays_live_bar_is_provisional_until_history_confirms_it(env):
    daily, index, conn = env
    history = series(25)
    today = DATES[24]
    live = pd.DataFrame([{"instrument_key": AAA.instrument_key, "date": today, "open": 124.0, "high": 125.24,
                          "low": 122.76, "close": 124.0, "volume": 1000}])
    provider = FakeProvider({"AAA": history[history["date"] < today].reset_index(drop=True)}, live=live)
    sync = BarSync(provider, daily, index, conn)
    sync.backfill([AAA], date(2015, 1, 1), DATES[23].date())
    sync.refresh([AAA], today.date(), today=today.date(), new_start=date(2015, 1, 1))
    stored = daily.read("AAA")
    assert stored["date"].iloc[-1] == today and stored["source"].iloc[-1] == SOURCE_LIVE
    assert sync.meta(AAA)["provisional"] == 1

    provider.history["AAA"] = history  # next day: history now includes yesterday's session
    sync.refresh([AAA], today.date(), today=date(2026, 10, 20), new_start=date(2015, 1, 1))
    stored = daily.read("AAA")
    assert stored["source"].iloc[-1] == SOURCE_HISTORY and sync.meta(AAA)["provisional"] == 0


def test_refresh_backfills_instruments_with_no_file(env):
    daily, index, conn = env
    provider = FakeProvider({"AAA": series(15)})
    result = BarSync(provider, daily, index, conn).refresh([AAA], DATES[14].date(), today=date(2026, 10, 20),
                                                          new_start=date(2015, 1, 1))[0]
    assert result.status == "ok" and result.added == 15


# Regressions from the audit (2026-10-20) ------------------------------------------------------
def test_failed_full_repull_leaves_the_file_untouched_and_retries_next_time(env):
    daily, index, conn = env
    history = series(20, 1000.0)
    provider = FakeProvider({"AAA": history})
    sync = BarSync(provider, daily, index, conn)
    sync.backfill([AAA], date(2015, 1, 1), DATES[19].date())
    before = daily.path("AAA").read_bytes()
    adjusted = history.copy()
    adjusted[["open", "high", "low", "close"]] /= 2

    class FailsOnFullHistory(FakeProvider):
        def fetch_daily(self, inst, start, end):
            if start <= DATES[0].date():
                raise ProviderError("HTTP 503 on the full-history request")
            return super().fetch_daily(inst, start, end)

    sync.provider = FailsOnFullHistory({"AAA": adjusted})
    assert sync.refresh([AAA], DATES[19].date(), today=date(2026, 10, 20), new_start=date(2015, 1, 1))[0].status == "error"
    assert daily.path("AAA").read_bytes() == before  # nothing half-written
    sync.provider = FakeProvider({"AAA": adjusted})
    assert sync.refresh([AAA], DATES[19].date(), today=date(2026, 10, 20), new_start=date(2015, 1, 1))[0].repulled


def test_provisional_bar_revision_does_not_trigger_a_full_repull(env):
    daily, index, conn = env
    history = series(20)
    today = DATES[19]
    provisional = history[history["date"] == today].assign(close=lambda f: f["close"] * 1.01)  # live close 1% off
    provider = FakeProvider({"AAA": history[history["date"] < today].reset_index(drop=True)})
    sync = BarSync(provider, daily, index, conn)
    sync.backfill([AAA], date(2015, 1, 1), DATES[18].date())
    daily.upsert("AAA", provisional, source=SOURCE_LIVE, meta={"name": "AAA"})
    provider.history["AAA"] = history
    result = sync.refresh([AAA], today.date(), today=date(2026, 10, 21), new_start=date(2015, 1, 1))[0]
    assert not result.repulled and daily.read("AAA")["source"].iloc[-1] == SOURCE_HISTORY
    assert conn.execute("SELECT COUNT(*) FROM repull_events").fetchone()[0] == 0


def test_rename_moves_the_file_instead_of_starting_a_new_one(env):
    daily, index, conn = env
    provider = FakeProvider({"AAA": series(10), "NEWAAA": series(10)})
    sync = BarSync(provider, daily, index, conn)
    sync.backfill([AAA], date(2015, 1, 1), DATES[9].date())
    renamed = equity("NEWAAA", AAA.isin)  # same ISIN, so the same instrument key
    assert sync.reconcile_renames([renamed]) == [("AAA", "NEWAAA")]
    assert daily.stems() == ["NEWAAA"] and sync.meta(renamed)["rows"] == 10


def test_refresh_as_of_before_the_overlap_window_does_not_crash(env):
    daily, index, conn = env
    provider = FakeProvider({"AAA": series(25)})
    sync = BarSync(provider, daily, index, conn)
    sync.backfill([AAA], date(2015, 1, 1), DATES[24].date())
    result = sync.refresh([AAA], DATES[5].date(), today=date(2026, 10, 20), new_start=date(2015, 1, 1), overlap=10)[0]
    assert result.status == "ok"
