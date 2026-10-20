"""Live checks against Upstox. Opt in with: uv run pytest -m network"""

from datetime import date

import pytest

from trader.data.instruments import download_instruments, parse_instruments
from trader.data.providers.base import Instrument
from trader.data.providers.upstox import UpstoxProvider
from trader.data.quality import run_checks
from trader.settings import get_settings

pytestmark = pytest.mark.network

RELIANCE = Instrument("RELIANCE", "NSE_EQ|INE002A01018", "EQ", "RELIANCE INDUSTRIES LTD", isin="INE002A01018")


@pytest.fixture(scope="module")
def provider():
    token = get_settings().token_value()
    if not token:
        pytest.skip("UPSTOX_ACCESS_TOKEN is not configured")
    with UpstoxProvider(token) as p:
        yield p


def test_token_is_accepted(provider):
    health = provider.healthcheck()
    assert health.ok, health.detail


def test_one_month_of_daily_bars(provider):
    bars = provider.fetch_daily(RELIANCE, date(2026, 8, 1), date(2026, 8, 31))
    assert 18 <= len(bars) <= 23
    assert not [f for f in run_checks("RELIANCE", bars) if f.severity == "error"]


def test_instrument_master_is_downloadable(tmp_path):
    ref = parse_instruments(download_instruments(tmp_path))
    assert (ref["segment"] == "NSE_EQ").sum() > 5000
    assert "NSE_INDEX|Nifty 50" in set(ref["instrument_key"])


def test_batch_latest_candles_and_holidays(provider):
    nifty = Instrument("Nifty 50", "NSE_INDEX|Nifty 50", "INDEX", "Nifty 50")
    latest = provider.fetch_latest_daily([RELIANCE, nifty])
    assert set(latest["instrument_key"]) == {RELIANCE.instrument_key, nifty.instrument_key}
    assert (latest["close"] > 0).all()
    assert any("NSE" in (h.get("closed_exchanges") or []) for h in provider.market_holidays())


def test_nse_corporate_actions_feed():
    from trader.data.corporate_actions import NSEClient, fetch_corporate_actions

    with NSEClient() as nse:
        rows = fetch_corporate_actions(nse, date(2025, 1, 1), date(2025, 1, 31))
    itc = [r for r in rows if r.get("symbol") == "ITC" and r.get("subject") == "Demerger"]
    assert len(rows) > 20 and itc and itc[0]["exDate"] == "06-Jan-2025"
