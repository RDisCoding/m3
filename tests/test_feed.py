from datetime import date

import pandas as pd
import pytest
from factories import action_rows, bars

from trader.data.corporate_actions import compute_factors, store_factors
from trader.data.feed import LocalDataFeed, panel_contract_check
from trader.data.universe import update_membership
from trader.settings import Settings
from trader.storage.db import connect
from trader.storage.parquet import BarStore

DATES = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]


@pytest.fixture
def feed(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    daily, index = BarStore(settings.bars_dir / "daily"), BarStore(settings.bars_dir / "index")
    itc = bars(DATES, [492.0, 489.05, 481.6, 442.65, 441.1], opens=[490.0, 490.0, 488.0, 455.0, 443.0])
    late = bars(DATES[2:], [100.0, 101.0, 102.0])  # listed later
    meta = {"name": "x"}
    daily.upsert("ITC", itc, source="upstox", meta=meta)
    daily.upsert("NEWCO", late, source="upstox", meta=meta)
    index.upsert("Nifty 50", bars(DATES, [23000.0, 23100.0, 23200.0, 23050.0, 23300.0]), source="upstox", meta=meta)
    conn = connect(settings.db_path)
    store_factors(conn, "ITC", compute_factors("ITC", itc, action_rows(("ITC", "2025-01-06", "Demerger"))), "now")
    update_membership(conn, "nifty500", pd.DataFrame([
        {"symbol": "ITC", "isin": "INE154A01025", "instrument_key": "NSE_EQ|INE154A01025", "company_name": "ITC",
         "industry": "Fast Moving Consumer Goods", "status": "ok"},
        {"symbol": "NEWCO", "isin": "INE999A01011", "instrument_key": "NSE_EQ|INE999A01011", "company_name": "New Co",
         "industry": "Capital Goods", "status": "ok"}]), date(2026, 10, 20))
    conn.commit()
    conn.close()
    return LocalDataFeed(settings)


def test_panel_is_aligned_adjusted_and_passes_the_contract(feed):
    panel = feed.load_panel()
    close = panel["close"]
    assert list(close.columns) == ["ITC", "NEWCO"] and len(close) == 5
    assert close["NEWCO"].isna().sum() == 2  # not listed yet
    assert close.loc["2025-01-03", "ITC"] == pytest.approx(455.0)  # demerger-adjusted
    assert panel["unadj_close"].loc["2025-01-03", "ITC"] == pytest.approx(481.6)  # as traded
    assert feed.load_panel(adjustment="none")["close"].loc["2025-01-03", "ITC"] == pytest.approx(481.6)
    assert panel_contract_check(panel) == []


def test_contract_check_catches_broken_bars(feed):
    panel = feed.load_panel()
    panel["high"].loc["2025-01-06", "ITC"] = 1.0
    assert any("high/low" in p for p in panel_contract_check(panel))


def test_slicing_calendar_universe_and_sectors(feed):
    assert len(feed.load_bars("ITC", start=date(2025, 1, 3))) == 3
    assert feed.trading_days(date(2025, 1, 1), date(2025, 1, 7)) == [date(2025, 1, d) for d in (1, 2, 3, 6, 7)]
    assert feed.universe_at(date(2025, 1, 1)) == ["ITC", "NEWCO"]
    assert feed.sector_map()["NEWCO"] == "Capital Goods"


def test_fingerprint_changes_when_data_changes(feed):
    before = feed.fingerprint()
    feed.daily.upsert("NEWCO", bars(["2025-01-08"], [103.0]), source="upstox", meta={"name": "y"})
    assert LocalDataFeed(feed.settings).fingerprint() != before
