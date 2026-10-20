"""Upstox market-data client for the read-only analytics token.

Endpoints used:
  GET /v3/historical-candle/{instrument_key}/days/1/{to_date}/{from_date}   daily OHLCV
  GET /v3/market-quote/ltp?instrument_key=k1,k2                              last traded price
The analytics token covers market-data GET APIs without a static IP and cannot place orders.
Daily candles exist from January 2000 and are adjusted for splits and bonuses, not dividends.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import httpx
import pandas as pd

from trader.data.providers.base import (
    AuthError,
    Instrument,
    ProviderError,
    ProviderHealth,
    RateLimitedError,
    TransientError,
    empty_bars,
    normalize_bars,
)
from trader.data.providers.ratelimit import SlidingWindowLimiter

log = logging.getLogger(__name__)

API_BASE = "https://api.upstox.com"
EARLIEST_DAILY = date(2000, 1, 1)
# Documented retrieval limit for daily candles is one decade per request; 3650 days stays inside it.
MAX_DAYS_PER_REQUEST = 3650
# Upstox limits: 50/s, 500/min, 2000 per 30 min. Stay well below each.
DEFAULT_LIMITS: tuple[tuple[int, float], ...] = ((8, 1.0), (400, 60.0), (1800, 1800.0))
HEALTHCHECK_KEY = "NSE_EQ|INE002A01018"  # RELIANCE INDUSTRIES LTD
OHLC_BATCH_SIZE = 500  # API maximum per request (UDAPI100043 above this)
# api.upstox.com resolves to two Cloudflare addresses; from some networks one of them never answers.
# A short connect timeout makes the socket layer fail over to the next address in seconds
# instead of waiting out the OS TCP timeout (21 s on Windows).
CONNECT_TIMEOUT_S = 3.0


def date_chunks(start: date, end: date, max_days: int) -> Iterator[tuple[date, date]]:
    """Split [start, end] (both inclusive) into consecutive windows of at most max_days days."""
    if max_days < 1:
        raise ValueError("max_days must be >= 1")
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=max_days - 1))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def parse_candles(payload: dict) -> pd.DataFrame:
    """Parse a historical-candle response into the canonical bar frame.

    Rows arrive as [timestamp, open, high, low, close, volume, open_interest], timestamp in
    ISO-8601 with the +05:30 offset; open interest is always 0 for equities and is dropped.
    """
    candles = ((payload or {}).get("data") or {}).get("candles") or []
    if not candles:
        return empty_bars()
    frame = pd.DataFrame([row[:6] for row in candles], columns=["ts", "open", "high", "low", "close", "volume"])
    session = pd.to_datetime(frame["ts"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    frame["date"] = session.dt.normalize()
    return normalize_bars(frame)


def jwt_claims(token: str) -> dict | None:
    """Decode a JWT payload locally (no signature check) to read expiry. Returns None if not a JWT."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, UnicodeError):
        return None
    return claims if isinstance(claims, dict) else None


def token_expiry(token: str) -> datetime | None:
    claims = jwt_claims(token)
    exp = claims.get("exp") if claims else None
    if isinstance(exp, (int, float)) and not isinstance(exp, bool):
        return datetime.fromtimestamp(exp, tz=timezone.utc)
    return None


def _error_details(resp: httpx.Response) -> tuple[list[str], str]:
    try:
        payload = resp.json()
    except ValueError:
        return [], (resp.text or "")[:300]
    codes: list[str] = []
    messages: list[str] = []
    errors = payload.get("errors") if isinstance(payload, dict) else None
    for err in errors or []:
        if isinstance(err, dict):
            code = err.get("errorCode") or err.get("error_code")
            if code:
                codes.append(str(code))
            if err.get("message"):
                messages.append(str(err["message"]))
    return codes, "; ".join(messages) or json.dumps(payload)[:300]


def _classify_failure(resp: httpx.Response) -> ProviderError:
    codes, message = _error_details(resp)
    status = resp.status_code
    detail = f"(HTTP {status}{', ' + ', '.join(codes) if codes else ''}): {message}"
    if status in (401, 403):
        return AuthError(f"Upstox rejected the token {detail}", status_code=status, error_codes=codes)
    if status == 429:
        return RateLimitedError(f"Upstox rate limit {detail}", status_code=status, error_codes=codes)
    if status >= 500:
        return TransientError(f"Upstox server error {detail}", status_code=status, error_codes=codes)
    return ProviderError(f"Upstox request failed {detail}", status_code=status, error_codes=codes)


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class UpstoxProvider:
    name = "upstox"

    def __init__(
        self,
        token: str | None,
        *,
        base_url: str = API_BASE,
        timeout_s: float = 30.0,
        connect_timeout_s: float = CONNECT_TIMEOUT_S,
        client: httpx.Client | None = None,
        limiter: SlidingWindowLimiter | None = None,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        max_days_per_request: int = MAX_DAYS_PER_REQUEST,
    ):
        if not token:
            raise AuthError("Upstox access token is missing; set UPSTOX_ACCESS_TOKEN in a .env file")
        headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        if client is None:
            timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)
            client = httpx.Client(base_url=base_url, timeout=timeout, headers=headers)
        else:
            client.headers.update(headers)
        self._client = client
        self._limiter = limiter or SlidingWindowLimiter(DEFAULT_LIMITS)
        self._max_retries = max_retries
        self._sleep = sleep
        self._max_days = max_days_per_request
        self.request_count = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> UpstoxProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # HTTP --------------------------------------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            self.request_count += 1
            retry_after: float | None = None
            try:
                resp = self._client.get(path, params=params)
            except httpx.TransportError as exc:  # connection errors and timeouts
                failure: ProviderError = TransientError(f"network error calling Upstox: {type(exc).__name__}")
            else:
                if resp.status_code == 200:
                    return self._success_payload(resp)
                failure = _classify_failure(resp)
                if not isinstance(failure, (RateLimitedError, TransientError)):
                    raise failure
                retry_after = _retry_after_seconds(resp)
            if attempt == self._max_retries:
                raise failure
            delay = max(retry_after or 0.0, min(30.0, 2.0**attempt) + random.uniform(0.0, 0.5))
            log.warning("%s; retrying in %.1fs (retry %d of %d)", failure, delay, attempt + 1, self._max_retries)
            self._sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _success_payload(resp: httpx.Response) -> dict:
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ProviderError("Upstox returned a non-JSON response", status_code=resp.status_code) from exc
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise ProviderError(f"Upstox returned an unexpected payload: {str(payload)[:300]}", status_code=resp.status_code)
        return payload

    # Market data -------------------------------------------------------------------
    def fetch_daily(self, inst: Instrument, start: date, end: date) -> pd.DataFrame:
        """Daily bars for [start, end] inclusive, ascending, in the canonical frame."""
        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        start = max(start, EARLIEST_DAILY)
        key = quote(inst.instrument_key, safe="")
        frames = []
        for chunk_start, chunk_end in date_chunks(start, end, self._max_days):
            path = f"/v3/historical-candle/{key}/days/1/{chunk_end:%Y-%m-%d}/{chunk_start:%Y-%m-%d}"
            frames.append(parse_candles(self._get(path)))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return empty_bars()
        bars = normalize_bars(pd.concat(frames, ignore_index=True))
        in_range = (bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))
        return bars.loc[in_range].reset_index(drop=True)

    def fetch_today(self, inst: Instrument) -> pd.DataFrame:
        """The current session's daily candle from the intraday endpoint (empty frame if none).

        The historical endpoint covers completed sessions up to the previous day only. Treat this
        candle as provisional until the next historical fetch confirms it.
        """
        key = quote(inst.instrument_key, safe="")
        return parse_candles(self._get(f"/v3/historical-candle/intraday/{key}/days/1"))

    def fetch_latest_daily(self, insts: Sequence[Instrument]) -> pd.DataFrame:
        """Latest session's daily candle for many instruments via the OHLC quote API.

        One request per 500 instruments (the API maximum). Returns columns
        instrument_key, date, open, high, low, close, volume, where `date` is the session the
        candle belongs to (from the quote's timestamp), so callers can check it is the day they want.
        """
        rows: list[dict] = []
        for start in range(0, len(insts), OHLC_BATCH_SIZE):
            batch = insts[start : start + OHLC_BATCH_SIZE]
            payload = self._get("/v3/market-quote/ohlc",
                                params={"instrument_key": ",".join(i.instrument_key for i in batch), "interval": "1d"})
            for entry in (payload.get("data") or {}).values():
                live = (entry or {}).get("live_ohlc")
                key = (entry or {}).get("instrument_token")
                if not live or not key or live.get("ts") is None:
                    continue
                session = pd.Timestamp(live["ts"], unit="ms", tz="UTC").tz_convert("Asia/Kolkata").tz_localize(None).normalize()
                rows.append({"instrument_key": key, "date": session, **{c: live.get(c) for c in ("open", "high", "low", "close", "volume")}})
        frame = pd.DataFrame(rows, columns=["instrument_key", "date", "open", "high", "low", "close", "volume"])
        if not frame.empty:
            frame["date"] = frame["date"].astype("datetime64[ns]")
            frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].astype("float64")
            frame["volume"] = pd.to_numeric(frame["volume"]).fillna(0).round().astype("int64")
        return frame

    def market_holidays(self) -> list[dict]:
        """Exchange holidays and special sessions for the current year (Upstox market information API)."""
        return list(self._get("/v2/market/holidays").get("data") or [])

    def ltp(self, instrument_keys: Sequence[str]) -> dict[str, float]:
        """Last traded prices keyed by instrument key."""
        if not instrument_keys:
            return {}
        payload = self._get("/v3/market-quote/ltp", params={"instrument_key": ",".join(instrument_keys)})
        prices: dict[str, float] = {}
        for response_key, entry in (payload.get("data") or {}).items():
            if not isinstance(entry, dict) or entry.get("last_price") is None:
                continue
            prices[str(entry.get("instrument_token") or response_key)] = float(entry["last_price"])
        return prices

    def healthcheck(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            prices = self.ltp([HEALTHCHECK_KEY])
        except ProviderError as exc:
            return ProviderHealth(False, self.name, str(exc))
        latency_ms = (time.perf_counter() - started) * 1000
        if not prices:
            return ProviderHealth(False, self.name, "LTP request succeeded but returned no price", latency_ms)
        key, price = next(iter(prices.items()))
        return ProviderHealth(True, self.name, f"LTP {key} = {price:.2f}", latency_ms)
