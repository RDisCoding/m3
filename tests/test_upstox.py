import base64
import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest

from trader.data.providers.base import (
    AuthError,
    Instrument,
    ProviderError,
    TransientError,
)
from trader.data.providers.ratelimit import SlidingWindowLimiter
from trader.data.providers.upstox import (
    UpstoxProvider,
    date_chunks,
    jwt_claims,
    parse_candles,
    token_expiry,
)

FAKE_TOKEN = "test-token-not-real"
RELIANCE = Instrument("RELIANCE", "NSE_EQ|INE002A01018", "EQ", "RELIANCE INDUSTRIES LTD", isin="INE002A01018")
NIFTY = Instrument("Nifty 50", "NSE_INDEX|Nifty 50", "INDEX", "Nifty 50")


def candles(*rows):
    return {"status": "success", "data": {"candles": [list(r) for r in rows]}}


def make_provider(handler, **kwargs):
    client = httpx.Client(base_url="https://api.upstox.com", transport=httpx.MockTransport(handler))
    sleeps: list[float] = []
    provider = UpstoxProvider(
        FAKE_TOKEN, client=client, limiter=SlidingWindowLimiter([(10_000, 1.0)]), sleep=sleeps.append, **kwargs
    )
    return provider, sleeps


def test_parse_real_response_is_ascending_with_session_dates(fixtures):
    bars = parse_candles(json.loads((fixtures / "upstox_candles_reliance.json").read_text()))
    assert list(bars.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert bars["date"].is_monotonic_increasing and bars["date"].is_unique
    assert str(bars["date"].dtype) == "datetime64[ns]" and bars["volume"].dtype == "int64"
    assert bars["date"].dt.dayofweek.max() <= 4
    last = bars.iloc[-1]
    assert last["date"] == pd.Timestamp("2026-09-30")
    assert (last["open"], last["high"], last["low"], last["close"], last["volume"]) == (1182.0, 1196.5, 1181.7, 1187.0, 16376789)


def test_parse_converts_utc_timestamps_to_the_ist_session_date():
    bars = parse_candles(candles(["2026-09-29T18:30:00Z", 1, 2, 0.5, 1.5, 10, 0]))
    assert bars["date"].iloc[0] == pd.Timestamp("2026-09-30")


def test_parse_empty_response():
    bars = parse_candles(candles())
    assert bars.empty and list(bars.columns) == ["date", "open", "high", "low", "close", "volume"]


def test_date_chunks_cover_the_range_without_gaps_or_overlap():
    chunks = list(date_chunks(date(2015, 1, 1), date(2026, 10, 5), 3650))
    assert chunks[0][0] == date(2015, 1, 1) and chunks[-1][1] == date(2026, 10, 5)
    assert all(b[0] == a[1] + timedelta(days=1) for a, b in zip(chunks, chunks[1:]))
    assert all((end - start).days + 1 <= 3650 for start, end in chunks)
    assert len(chunks) == 2


def test_fetch_daily_chunks_requests_and_url_encodes_keys():
    paths = []

    def handler(request):
        paths.append(request.url.raw_path.decode())
        assert request.headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
        if request.url.path.endswith("/2015-01-01"):
            return httpx.Response(200, json=candles(["2016-01-04T00:00:00+05:30", 1, 2, 0.5, 1.5, 100, 0]))
        return httpx.Response(200, json=candles(["2026-09-30T00:00:00+05:30", 3, 4, 2.5, 3.5, 200, 0]))

    provider, _ = make_provider(handler)
    bars = provider.fetch_daily(RELIANCE, date(2015, 1, 1), date(2026, 10, 5))
    assert paths == [
        "/v3/historical-candle/NSE_EQ%7CINE002A01018/days/1/2024-12-28/2015-01-01",
        "/v3/historical-candle/NSE_EQ%7CINE002A01018/days/1/2026-10-05/2024-12-29",
    ]
    assert list(bars["date"]) == [pd.Timestamp("2016-01-04"), pd.Timestamp("2026-09-30")]


def test_fetch_daily_encodes_index_names_with_spaces():
    paths = []

    def handler(request):
        paths.append(request.url.raw_path.decode())
        return httpx.Response(200, json=candles())

    provider, _ = make_provider(handler)
    provider.fetch_daily(NIFTY, date(2026, 9, 1), date(2026, 9, 30))
    assert paths == ["/v3/historical-candle/NSE_INDEX%7CNifty%2050/days/1/2026-09-30/2026-09-01"]


def test_fetch_daily_trims_rows_outside_the_requested_range():
    def handler(request):
        return httpx.Response(200, json=candles(
            ["2026-10-01T00:00:00+05:30", 3, 4, 2, 3, 1, 0],
            ["2026-09-30T00:00:00+05:30", 3, 4, 2, 3, 1, 0],
            ["2026-08-29T00:00:00+05:30", 3, 4, 2, 3, 1, 0],
        ))

    provider, _ = make_provider(handler)
    bars = provider.fetch_daily(RELIANCE, date(2026, 9, 1), date(2026, 9, 30))
    assert list(bars["date"]) == [pd.Timestamp("2026-09-30")]


def test_fetch_today_uses_the_intraday_endpoint():
    def handler(request):
        assert request.url.raw_path.decode() == "/v3/historical-candle/intraday/NSE_EQ%7CINE002A01018/days/1"
        return httpx.Response(200, json=candles(["2026-10-05T00:00:00+05:30", 1171.2, 1193.0, 1171.2, 1186.4, 14254907, 0]))

    provider, _ = make_provider(handler)
    today = provider.fetch_today(RELIANCE)
    assert len(today) == 1 and today["date"].iloc[0] == pd.Timestamp("2026-10-05") and today["close"].iloc[0] == 1186.4


def test_rate_limited_requests_are_retried_honouring_retry_after():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "2"},
                                  json={"status": "error", "errors": [{"errorCode": "UDAPI10005", "message": "Too Many Request Sent"}]})
        return httpx.Response(200, json=candles())

    provider, sleeps = make_provider(handler)
    provider.fetch_daily(RELIANCE, date(2026, 9, 1), date(2026, 9, 30))
    assert calls["n"] == 3 and len(sleeps) == 2 and min(sleeps) >= 2.0


def test_server_errors_give_up_after_max_retries():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    provider, sleeps = make_provider(handler, max_retries=2)
    with pytest.raises(TransientError):
        provider.fetch_daily(RELIANCE, date(2026, 9, 1), date(2026, 9, 30))
    assert calls["n"] == 3 and len(sleeps) == 2


def test_network_errors_are_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("connect timed out", request=request)
        return httpx.Response(200, json=candles())

    provider, sleeps = make_provider(handler)
    provider.fetch_daily(RELIANCE, date(2026, 9, 1), date(2026, 9, 30))
    assert calls["n"] == 2 and len(sleeps) == 1


def test_auth_errors_are_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"status": "error", "errors": [{"errorCode": "UDAPI100050", "message": "Invalid token used to access API"}]})

    provider, _ = make_provider(handler)
    with pytest.raises(AuthError) as excinfo:
        provider.fetch_daily(RELIANCE, date(2026, 9, 1), date(2026, 9, 30))
    assert calls["n"] == 1 and excinfo.value.error_codes == ("UDAPI100050",)
    assert FAKE_TOKEN not in str(excinfo.value)


def test_invalid_instrument_key_is_a_provider_error_with_upstox_code(fixtures):
    body = (fixtures / "upstox_error_invalid_key.json").read_text()

    def handler(request):
        return httpx.Response(400, text=body, headers={"Content-Type": "application/json"})

    provider, sleeps = make_provider(handler)
    with pytest.raises(ProviderError) as excinfo:
        provider.fetch_daily(RELIANCE, date(2026, 9, 1), date(2026, 9, 30))
    assert not isinstance(excinfo.value, (AuthError, TransientError))
    assert excinfo.value.error_codes == ("UDAPI100011",) and "Invalid Instrument key" in str(excinfo.value)
    assert sleeps == []


def test_ltp_is_keyed_by_instrument_key(fixtures):
    body = (fixtures / "upstox_ltp.json").read_text()

    def handler(request):
        assert request.url.params["instrument_key"] == "NSE_EQ|INE002A01018,NSE_INDEX|Nifty 50"
        return httpx.Response(200, text=body, headers={"Content-Type": "application/json"})

    provider, _ = make_provider(handler)
    assert provider.ltp(["NSE_EQ|INE002A01018", "NSE_INDEX|Nifty 50"]) == {
        "NSE_EQ|INE002A01018": 1186.4,
        "NSE_INDEX|Nifty 50": 22555.75,
    }


def test_healthcheck_reports_rejected_token():
    def handler(request):
        return httpx.Response(401, json={"status": "error", "errors": [{"errorCode": "UDAPI100050", "message": "Invalid token used to access API"}]})

    provider, _ = make_provider(handler)
    health = provider.healthcheck()
    assert not health.ok and "rejected the token" in health.detail


def test_missing_token_is_rejected_before_any_request():
    with pytest.raises(AuthError):
        UpstoxProvider("")


def _b64(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def test_token_expiry_is_read_from_the_jwt_payload():
    token = f"{_b64({'alg': 'none'})}.{_b64({'exp': 1806000000, 'sub': 'someone'})}.signature"
    assert token_expiry(token) == datetime.fromtimestamp(1806000000, tz=timezone.utc)
    assert token_expiry("not-a-jwt") is None
    assert jwt_claims("a.!!!.c") is None


def test_latest_daily_batches_500_keys_and_dates_each_candle(fixtures):
    body = json.loads((fixtures / "upstox_ohlc.json").read_text())
    calls = []

    def handler(request):
        calls.append(request.url.params["instrument_key"].split(","))
        return httpx.Response(200, json=body)

    provider, _ = make_provider(handler)
    insts = [RELIANCE, NIFTY] + [Instrument(f"S{i}", f"NSE_EQ|INE{i:09d}", "EQ", "x") for i in range(499)]
    latest = provider.fetch_latest_daily(insts)
    assert [len(c) for c in calls] == [500, 1]
    reliance = latest[latest["instrument_key"] == "NSE_EQ|INE002A01018"].iloc[0]
    expected = body["data"]["NSE_EQ:RELIANCE"]["live_ohlc"]
    assert reliance["close"] == expected["close"] and reliance["volume"] == expected["volume"]
    session = pd.Timestamp(expected["ts"], unit="ms", tz="UTC").tz_convert("Asia/Kolkata").tz_localize(None).normalize()
    assert reliance["date"] == session and session.hour == 0


def test_market_holidays_returns_the_list(fixtures):
    body = (fixtures / "upstox_holidays.json").read_text()

    def handler(request):
        assert request.url.path == "/v2/market/holidays"
        return httpx.Response(200, text=body, headers={"Content-Type": "application/json"})

    provider, _ = make_provider(handler)
    assert any(h["description"] == "Dussehra" for h in provider.market_holidays())
