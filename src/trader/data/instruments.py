"""NSE instrument master (Upstox) and symbol <-> ISIN <-> instrument_key mapping.

The master is public (no token) and refreshed by Upstox daily around 06:00 IST:
    https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz
Facts verified on the live file (2026-10-05):
  * instrument_key is "NSE_EQ|<ISIN>" for equities and "NSE_INDEX|<index name>" for indices.
  * ISIN and instrument_key are unique within NSE_EQ; trading symbols are unique only within the
    share series below (warrants and bonds reuse a few symbols), so symbols resolve within them.
  * tick_size is reported in paise (1, 5, 10, 50, 100, 500) and is converted to rupees here.
  * instrument_type EQ also covers ETFs (ISIN prefix INF).
"""

from __future__ import annotations

import difflib
import gzip
import json
import os
import re
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trader.data.providers.base import Instrument

INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
# Share series on NSE: EQ rolling settlement, BE/BZ trade-for-trade (BZ = Z category),
# SM SME platform, ST SME trade-for-trade.
EQUITY_TYPES = ("EQ", "BE", "BZ", "SM", "ST")
T2T_TYPES = ("BE", "BZ", "ST")
SME_TYPES = ("SM", "ST")
REF_COLUMNS = [
    "instrument_key",
    "segment",
    "instrument_type",
    "trading_symbol",
    "name",
    "short_name",
    "isin",
    "exchange_token",
    "lot_size",
    "tick_size",
    "freeze_quantity",
    "security_type",
]
REF_FILENAME = "instruments_NSE.parquet"
_ISIN_RE = re.compile(r"^IN[A-Z0-9]{10}$")
_GZIP_MAGIC = b"\x1f\x8b"


def download_instruments(
    dest_dir: Path, *, client: httpx.Client | None = None, today: date | None = None
) -> Path:
    """Save today's master as dest_dir/NSE_YYYYMMDD.json.gz (atomic write) and return the path."""
    today = today or date.today()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"NSE_{today:%Y%m%d}.json.gz"
    owns_client = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0), follow_redirects=True)
    try:
        resp = client.get(INSTRUMENTS_URL)
        resp.raise_for_status()
        content = resp.content
    finally:
        if owns_client:
            client.close()
    if not content.startswith(_GZIP_MAGIC):  # transport-level decompression already happened
        content = gzip.compress(content)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, dest)
    return dest


def parse_instruments(path: Path) -> pd.DataFrame:
    """Parse the master, keeping NSE_EQ and NSE_INDEX rows with a fixed column set."""
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode("utf-8") if raw.startswith(_GZIP_MAGIC) else raw.decode("utf-8")
    df = pd.DataFrame(json.loads(text))
    df = df[df["segment"].isin(["NSE_EQ", "NSE_INDEX"])].copy()
    for column in REF_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[REF_COLUMNS]
    is_equity_segment = df["segment"] == "NSE_EQ"
    df["tick_size"] = (pd.to_numeric(df["tick_size"], errors="coerce") / 100.0).where(is_equity_segment)
    df["lot_size"] = pd.to_numeric(df["lot_size"], errors="coerce").astype("Int64")
    df["freeze_quantity"] = pd.to_numeric(df["freeze_quantity"], errors="coerce")
    for column in ("instrument_key", "segment", "instrument_type", "trading_symbol", "name", "short_name",
                   "isin", "exchange_token", "security_type"):
        df[column] = df[column].astype("string")
    return df.sort_values(["segment", "trading_symbol", "instrument_key"]).reset_index(drop=True)


def save_reference(df: pd.DataFrame, ref_dir: Path, as_of: date) -> Path:
    ref_dir.mkdir(parents=True, exist_ok=True)
    path = ref_dir / REF_FILENAME
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"trader"] = json.dumps({"as_of": as_of.isoformat(), "source": INSTRUMENTS_URL}).encode()
    table = table.replace_schema_metadata(metadata)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path


def reference_as_of(ref_dir: Path) -> date | None:
    path = ref_dir / REF_FILENAME
    if not path.is_file():
        return None
    metadata = pq.read_schema(path).metadata or {}
    info = json.loads(metadata.get(b"trader", b"{}"))
    return date.fromisoformat(info["as_of"]) if "as_of" in info else None


def ensure_reference(
    raw_dir: Path, ref_dir: Path, *, refresh: bool = False, today: date | None = None
) -> tuple[pd.DataFrame, date, bool]:
    """Return (reference table, as_of, downloaded). Downloads at most once per day unless refresh=True."""
    today = today or date.today()
    as_of = reference_as_of(ref_dir)
    if not refresh and as_of == today:
        return pd.read_parquet(ref_dir / REF_FILENAME), as_of, False
    path = download_instruments(raw_dir / "instruments", today=today)
    df = parse_instruments(path)
    save_reference(df, ref_dir, today)
    return df, today, True


def _clean(value: object) -> str | None:
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


class SymbolMap:
    """Look up instruments by NSE symbol, ISIN, Upstox instrument key or index name."""

    def __init__(self, ref: pd.DataFrame):
        self._by_key: dict[str, Instrument] = {}
        self._by_symbol: dict[str, Instrument] = {}
        self._by_isin: dict[str, Instrument] = {}
        self._by_index_name: dict[str, Instrument] = {}
        for row in ref.to_dict("records"):
            inst = self._to_instrument(row)
            self._by_key[inst.instrument_key] = inst
            if inst.kind == "INDEX":
                self._by_index_name[inst.symbol.casefold()] = inst
                alias = _clean(row.get("trading_symbol"))
                if alias:
                    self._by_index_name.setdefault(alias.casefold(), inst)
                continue
            if inst.isin:
                self._by_isin[inst.isin] = inst
            if inst.instrument_type in EQUITY_TYPES:
                self._by_symbol[inst.symbol] = inst

    @staticmethod
    def _to_instrument(row: dict) -> Instrument:
        if row["segment"] == "NSE_INDEX":
            return Instrument(
                symbol=str(row["name"]),
                instrument_key=str(row["instrument_key"]),
                kind="INDEX",
                name=str(row["name"]),
                instrument_type="INDEX",
                exchange_token=_clean(row.get("exchange_token")),
            )
        lot = row.get("lot_size")
        tick = row.get("tick_size")
        return Instrument(
            symbol=str(row["trading_symbol"]),
            instrument_key=str(row["instrument_key"]),
            kind="EQ",
            name=str(row["name"]),
            isin=_clean(row.get("isin")),
            instrument_type=_clean(row.get("instrument_type")),
            tick_size=None if tick is None or pd.isna(tick) else float(tick),
            lot_size=1 if lot is None or pd.isna(lot) else int(lot),
            exchange_token=_clean(row.get("exchange_token")),
        )

    @property
    def equity_symbols(self) -> list[str]:
        return sorted(self._by_symbol)

    def equity(self, symbol: str) -> Instrument | None:
        return self._by_symbol.get(symbol.strip().upper())

    def by_isin(self, isin: str) -> Instrument | None:
        return self._by_isin.get(isin.strip().upper())

    def by_key(self, instrument_key: str) -> Instrument | None:
        return self._by_key.get(instrument_key.strip())

    def index(self, name: str) -> Instrument | None:
        return self._by_index_name.get(name.strip().casefold())

    def resolve(self, token: str) -> Instrument:
        """Resolve an instrument key, ISIN, equity symbol or index name (in that order)."""
        text = token.strip()
        if "|" in text:
            found = self.by_key(text)
        elif _ISIN_RE.match(text.upper()):
            found = self.by_isin(text)
        else:
            found = self.equity(text) or self.index(text)
        if found is None:
            close = difflib.get_close_matches(text.upper(), self._by_symbol.keys(), n=3, cutoff=0.75)
            hint = f"; close matches: {', '.join(close)}" if close else ""
            raise KeyError(f"no NSE instrument found for {token!r}{hint}")
        return found
